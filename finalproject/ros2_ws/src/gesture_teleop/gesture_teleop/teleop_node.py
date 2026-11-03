"""/gesture(std_msgs/String, JSON) 구독 -> 메시지마다 판정 -> /cmd_vel publish.

판정은 ESP32 프레임(/gesture 메시지)이 들어올 때마다 한다. 타이머는 끊김 감시 전용.
안전 규칙
- timeout초 동안 /gesture 무응답 -> 즉시 정지 (ESP32 끊김, 10Hz 타이머가 감시)
- five(손바닥) -> 즉시 정지
- 동작 제스처는 같은 제스처가 debounce번 나와야 반영 (사이에 낀 no_gesture/no_hand는 무시)
  모델이 실제로 one, no_gesture, one, no_gesture...처럼 번갈아 내놓기 때문
- 동작 제스처 없이 중립 프레임(no_hand, no_gesture, s < min_score, 모르는 제스처)이
  hold프레임 넘게 이어지면 정지. 6프레임 ≈ 1초 (ESP32 약 6fps)
충돌 안전 (Safety)
- /scan 전방(후방) ±30° 안에 stop_dist보다 가까우면 전진(후진) 차단. 회전·반대 방향은 허용
- 충돌 감지 -> ERROR 로그 + 정지 + 잠금. ✋ 또는 손을 내려(중립 hold프레임 초과) 풀어야 다시 움직임
  (잠금이 없으면 👍를 들고 있는 동안 2프레임 만에 재출발해 계속 부딪힘)
  · /imu 수평 가속도 > bump_accel. 실측: 평상시 < 0.2, 충돌 4~9, 출발·정지 순간에도 ~7이 튀므로
    속도 명령이 바뀐 뒤 0.5초는 무시
  · 직진 명령 중인데 /odom 실제 속도가 명령의 25% 미만으로 stuck_time초 (벽에 걸림)
- 명령이 0인데 계속 움직이면(밀림·튕김) WARN 로그
"""
import math
import signal
import json
import time

import rclpy
from rclpy.signals import SignalHandlerOptions
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import String

# 제스처 -> (전진 배수, 회전 배수). 실제 속도 = 배수 * linear_speed / angular_speed
ACTIONS = {
    'like': (1, 0),      # 👍 전진
    'dislike': (-1, 0),  # 👎 후진
    'one': (0, 1),       # ☝️ 좌회전
    'two': (0, -1),      # ✌️ 우회전
    'five': (0, 0),      # ✋ 정지
}
STOP = (0, 0)
MAX_LIN, MAX_ANG = 0.22, 2.84  # TB3 burger 최대 속도


class GestureFilter:
    """ROS와 무관한 판정 로직 (테스트 대상). update()는 프레임 하나당 한 번."""

    def __init__(self, min_score=0.7, debounce=2, hold=6, timeout=0.5):
        self.min_score, self.debounce, self.hold, self.timeout = min_score, debounce, hold, timeout
        self.reset()
        self.last_t = None

    def reset(self):
        self.latched = False
        self.action = STOP
        self.candidate, self.count = STOP, 0
        self.miss = 0  # 마지막 동작 제스처 이후 중립 프레임 수

    def update(self, msg, now):
        """프레임 하나 판정 후 현재 동작 반환."""
        self.last_t = now
        g, s = msg.get('g'), msg.get('s', 0.0)
        if not (g in ACTIONS and isinstance(s, (int, float)) and s >= self.min_score):
            self.miss += 1
            if self.miss > self.hold:
                self.reset()  # 잠금도 풀림
            return self.action
        if g == 'five':
            self.reset()
            return self.action
        if self.latched:  # 충돌 후: 동작 제스처 무시, 손을 내릴 때까지 대기
            self.miss = 0
            return STOP
        want = ACTIONS[g]
        if want == self.candidate:
            self.count += 1
        else:
            self.candidate, self.count = want, 1
        self.miss = 0
        if self.count >= self.debounce:
            self.action = want
        return self.action

    def latch(self):
        """충돌 시 호출. 정지 후 ✋/손 내림 전까지 동작 제스처 무시."""
        self.reset()
        self.latched = True

    def timed_out(self, now):
        """메시지가 timeout초 넘게 없으면 정지 상태로 리셋하고 True."""
        if self.last_t is None or now - self.last_t > self.timeout:
            self.reset()
            return True
        return False


def sector_min(scan, center, half=math.radians(30)):
    """center(rad) ±half 구간의 유효 거리 최솟값. 유효값이 없으면 inf."""
    best = math.inf
    for i, r in enumerate(scan.ranges):
        a = scan.angle_min + i * scan.angle_increment
        if abs(math.atan2(math.sin(a - center), math.cos(a - center))) <= half \
                and scan.range_min <= r <= scan.range_max:
            best = min(best, r)
    return best


class Safety:
    """ROS와 무관한 충돌 안전 로직 (테스트 대상)."""

    def __init__(self, stop_dist=0.2, bump_accel=3.0, stuck_time=0.7, drift_time=0.5):
        self.stop_dist, self.bump_accel = stop_dist, bump_accel
        self.stuck_time, self.drift_time = stuck_time, drift_time
        self.stuck_t = self.drift_t = self.idle_t = None
        self.drift_reported = False

    def clip(self, a, front, rear):
        """장애물 쪽 직진 성분을 0으로. (clip된 action, 막힌 방향 or None)"""
        if a[0] > 0 and front < self.stop_dist:
            return (0, a[1]), f'전방 {front:.2f}m'
        if a[0] < 0 and rear < self.stop_dist:
            return (0, a[1]), f'후방 {rear:.2f}m'
        return a, None

    def bump(self, accel):
        return accel > self.bump_accel

    def check(self, cmd_v, cmd_w, v, w, now):
        """'stuck'(충돌) / 'drift'(명령 없이 움직임, 이벤트당 1번) / None."""
        if abs(cmd_v) >= 0.05 and abs(v) < 0.25 * abs(cmd_v):
            if self.stuck_t is None:
                self.stuck_t = now
            if now - self.stuck_t >= self.stuck_time:
                self.stuck_t = None
                return 'stuck'
        else:
            self.stuck_t = None

        if cmd_v == 0 and cmd_w == 0:
            if self.idle_t is None:
                self.idle_t = now
            # 정지 명령 직후 1초는 감속 중일 수 있으니 제외
            moving = abs(v) > 0.03 or abs(w) > 0.3
            if moving and now - self.idle_t >= 1.0:
                if self.drift_t is None:
                    self.drift_t = now
                if now - self.drift_t >= self.drift_time and not self.drift_reported:
                    self.drift_reported = True
                    return 'drift'
            elif not moving:
                self.drift_t, self.drift_reported = None, False
        else:
            self.idle_t = self.drift_t = None
            self.drift_reported = False
        return None


class TeleopNode(Node):
    def __init__(self):
        super().__init__('gesture_teleop')
        p = lambda name, default: self.declare_parameter(name, default).value  # noqa: E731
        # 튜닝용 파라미터: --ros-args -p linear_speed:=0.1
        self.lin = min(float(p('linear_speed', 0.15)), MAX_LIN)
        self.ang = min(float(p('angular_speed', 1.0)), MAX_ANG)
        self.filter = GestureFilter(float(p('min_score', 0.7)), int(p('debounce', 2)),
                                    int(p('hold', 6)), float(p('timeout', 0.5)))
        self.safety = Safety(float(p('stop_dist', 0.2)), float(p('bump_accel', 3.0)),
                             float(p('stuck_time', 0.7)))
        self.prev = None
        self.blocked = None
        self.cmd = (0.0, 0.0)
        self.front = self.rear = math.inf
        self.v = self.w = 0.0
        self.bump_t = 0.0
        self.cmd_t = 0.0  # 속도 명령이 마지막으로 바뀐 시각
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(String, '/gesture', self.on_gesture, 10)
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        self.create_subscription(Imu, '/imu', self.on_imu, qos_profile_sensor_data)
        self.create_timer(0.1, self.tick)
        self.get_logger().info(f'gesture_teleop: lin={self.lin} ang={self.ang}')

    def on_gesture(self, msg):
        try:
            d = json.loads(msg.data)
        except ValueError:
            d = None
        if not isinstance(d, dict):
            self.get_logger().warn(f'잘못된 /gesture 무시: {msg.data!r}')
            return
        self.apply(self.filter.update(d, time.monotonic()), d.get('g'))

    def on_scan(self, msg):
        self.front, self.rear = sector_min(msg, 0.0), sector_min(msg, math.pi)
        if self.prev and self.prev != STOP:
            self.apply(self.filter.action, 'scan')  # 움직이는 중이면 새 거리로 다시 차단 판정

    def on_odom(self, msg):
        self.v, self.w = msg.twist.twist.linear.x, msg.twist.twist.angular.z

    def on_imu(self, msg):
        acc = math.hypot(msg.linear_acceleration.x, msg.linear_acceleration.y)
        now = time.monotonic()
        if now - self.cmd_t < 0.5:  # 출발·정지 반동
            return
        if self.safety.bump(acc) and now - self.bump_t > 1.0:  # 튕김으로 연달아 뜨는 것 방지
            self.bump_t = now
            self.collide(f'IMU 충격 {acc:.1f} m/s²')

    def collide(self, reason):
        self.get_logger().error(f'충돌 감지: {reason} (front={self.front:.2f}m rear={self.rear:.2f}m) -> 정지. ✋ 또는 손을 내렸다가 다시')
        self.filter.latch()
        self.apply(STOP, 'collision')

    def tick(self):
        now = time.monotonic()
        if self.filter.timed_out(now):
            self.apply(STOP, 'timeout')
        r = self.safety.check(*self.cmd, self.v, self.w, now)
        if r == 'stuck':
            self.collide(f'명령 v={self.cmd[0]:.2f}인데 실제 v={self.v:.2f} (막힘)')
        elif r == 'drift':
            self.get_logger().warn(f'명령 없이 움직임: v={self.v:.2f} w={self.w:.2f} (밀림/튕김)')
            self.publish(STOP)

    def apply(self, a, why):
        a, blocked = self.safety.clip(a, self.front, self.rear)
        side = blocked and blocked.split()[0]  # '전방'/'후방'. 거리 변화마다 로그가 반복되지 않게 방향만 비교
        if side != self.blocked:
            if side:
                self.get_logger().warn(f'장애물 근접 {blocked}: 그 방향 이동 차단')
            self.blocked = side
        if a != self.prev:
            self.get_logger().info(f'action {self.prev} -> {a} ({why})')
            self.prev = a
        self.publish(a)

    def publish(self, a):
        t = Twist()
        t.linear.x = a[0] * self.lin
        t.angular.z = a[1] * self.ang
        if (t.linear.x, t.angular.z) != self.cmd:
            self.cmd, self.cmd_t = (t.linear.x, t.angular.z), time.monotonic()
        self.pub.publish(t)


def _raise_interrupt(*_):
    raise KeyboardInterrupt


def main():
    # rclpy 기본 시그널 핸들러는 Ctrl+C에서 context를 먼저 내려 아래 정지 명령 publish가 실패한다
    # (로봇이 마지막 속도로 계속 감). 직접 처리해서 STOP을 보낸 뒤 종료.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGTERM, _raise_interrupt)
    node = TeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish(STOP)  # 종료 시 로봇 정지
        node.destroy_node()
        rclpy.try_shutdown()
