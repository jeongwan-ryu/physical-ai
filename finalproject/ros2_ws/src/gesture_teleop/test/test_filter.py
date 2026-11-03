"""GestureFilter 판정 로직 체크. 실행: python3 test/test_filter.py (ROS 환경 source 후)"""
from gesture_teleop.teleop_node import STOP, GestureFilter

LEFT = (0, 1)
N = ('no_gesture', 1.0)


def run(f, seq, dt=0.15):
    """seq: [(g, s), ...]를 프레임마다 넣고 각 프레임의 판정 결과 반환."""
    return [f.update({'g': g, 's': s}, i * dt) for i, (g, s) in enumerate(seq)]


def test_filter():
    # 실제 관측 패턴: one과 no_gesture가 번갈아 나와도 좌회전 (2번째 one에서 시작)
    out = run(GestureFilter(), [('one', 1.0), N] * 4)
    assert out[:2] == [STOP, STOP] and out[2:] == [LEFT] * 6, out

    # 손을 내리면 중립 hold(6)프레임까지 유지, 7번째에 정지
    out = run(GestureFilter(), [('one', 1.0)] * 2 + [('no_hand', 1.0)] * 7)
    assert out[2:8] == [LEFT] * 6 and out[8] == STOP, out

    # 다른 동작 제스처가 끼면 카운트 리셋
    assert run(GestureFilter(), [('one', 1.0), ('like', 1.0), ('one', 1.0)]) == [STOP] * 3

    # 신뢰도 미달은 카운트 안 됨
    assert run(GestureFilter(), [('like', 0.5)] * 5)[-1] == STOP

    # five는 즉시 정지, 이후 다시 디바운스 필요
    out = run(GestureFilter(), [('two', 0.9)] * 2 + [('five', 0.9), ('two', 0.9)])
    assert out == [STOP, (0, -1), STOP, STOP], out

    # 타임아웃: 0.5초 넘게 무응답이면 정지
    f = GestureFilter()
    run(f, [('like', 0.9)] * 2, dt=0.1)  # 마지막 t=0.1
    assert not f.timed_out(0.5) and f.action == (1, 0)
    assert f.timed_out(0.7) and f.action == STOP

    # 메시지 없음 / 이상한 s 타입
    assert GestureFilter().timed_out(0.0)
    assert run(GestureFilter(), [('like', 'high')] * 3)[-1] == STOP


def test_latch():
    # 충돌 후 잠금: 👍를 계속 들고 있어도 정지 유지
    f = GestureFilter()
    run(f, [('like', 1.0)] * 2)
    f.latch()
    assert run(f, [('like', 1.0), N] * 5)[-1] == STOP
    # ✋로 해제 -> 다시 디바운스 후 전진
    f.update({'g': 'five', 's': 1.0}, 0)
    assert run(f, [('like', 1.0)] * 2)[-1] == (1, 0)
    # 손을 내려서(중립 7프레임) 해제
    f.latch()
    run(f, [N] * 7)
    assert run(f, [('like', 1.0)] * 2)[-1] == (1, 0)


def test_safety():
    import math
    from types import SimpleNamespace
    from gesture_teleop.teleop_node import Safety, sector_min

    # 360칸 라이다(angle_min=0): 0°=전방. 350°에 0.15m, 180°에 0.5m, 0.12 미만은 무효
    r = [3.0] * 360
    r[350], r[180], r[90] = 0.15, 0.5, 0.05
    scan = SimpleNamespace(ranges=r, angle_min=0.0, angle_increment=math.radians(1), range_min=0.12, range_max=3.5)
    assert sector_min(scan, 0.0) == 0.15
    assert sector_min(scan, math.pi) == 0.5

    s = Safety()
    assert s.clip((1, 0), 0.15, 0.5) == ((0, 0), '전방 0.15m')   # 전진 차단
    assert s.clip((-1, 0), 0.15, 0.5) == ((-1, 0), None)        # 후진은 허용
    assert s.clip((0, 1), 0.15, 0.15) == ((0, 1), None)         # 회전은 허용
    assert s.clip((-1, 0), 1.0, 0.1)[0] == (0, 0)               # 후방 차단
    assert s.bump(5.0) and not s.bump(0.5)

    # 막힘: 전진 명령인데 실제 속도 0 -> 0.7초 후 stuck
    s = Safety()
    assert s.check(0.15, 0, 0.0, 0, 0.0) is None
    assert s.check(0.15, 0, 0.0, 0, 0.5) is None
    assert s.check(0.15, 0, 0.0, 0, 0.8) == 'stuck'
    assert s.check(0.15, 0, 0.14, 0, 1.0) is None              # 잘 가고 있으면 None

    # 명령 없이 움직임: 정지 후 1초 감속 유예, 그 뒤 0.5초 이상 움직이면 1번만 drift
    s = Safety()
    out = [s.check(0, 0, 0.1, 0, t / 10) for t in range(0, 25)]
    assert out.count('drift') == 1 and out.index('drift') == 15, out
    assert s.check(0, 0, 0.0, 0, 2.6) is None                  # 멈추면 리셋


if __name__ == '__main__':
    test_filter()
    test_latch()
    test_safety()
    print('ok')
