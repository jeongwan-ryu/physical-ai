"""웹 대시보드: Gazebo 상단 카메라 + ESP32 카메라 영상 + /gesture, /cmd_vel 상태 + /rosout 로그.

http://localhost:8090  (포트 변경: --ros-args -p port:=XXXX)
"""
import json
import threading
import time
from collections import deque

import cv2
import rclpy
from cv_bridge import CvBridge
from flask import Flask, Response, jsonify
from geometry_msgs.msg import Twist
from rcl_interfaces.msg import Log
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String

LEVELS = {10: 'DEBUG', 20: 'INFO', 30: 'WARN', 40: 'ERROR', 50: 'FATAL'}


class Dashboard(Node):
    def __init__(self):
        super().__init__('dashboard')
        self.lock = threading.Lock()
        self.jpeg = None
        self.esp_jpeg = None
        self.logs = deque(maxlen=300)
        self.gesture, self.gesture_t = None, 0.0
        self.cmd = (0.0, 0.0)
        self.bridge = CvBridge()
        self.create_subscription(Image, '/overhead_cam/image_raw', self.on_image, qos_profile_sensor_data)
        self.create_subscription(CompressedImage, '/esp32_cam/image/compressed', self.on_esp_image, 1)
        self.create_subscription(String, '/gesture', self.on_gesture, 10)
        self.create_subscription(Twist, '/cmd_vel', self.on_cmd, 10)
        self.create_subscription(Log, '/rosout', self.on_rosout, 100)

    def log(self, src, msg, level='INFO'):
        with self.lock:
            self.logs.append({'t': time.strftime('%H:%M:%S'), 'src': src, 'level': level, 'msg': msg})

    def on_image(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.jpeg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()

    def on_esp_image(self, msg):
        self.esp_jpeg = bytes(msg.data)  # 이미 JPEG라 재인코딩 없이 그대로 전달

    def on_gesture(self, msg):
        try:
            g = json.loads(msg.data)
        except ValueError:
            self.log('/gesture', f'JSON 파싱 실패: {msg.data!r}', 'WARN')
            return
        # 5~8Hz로 들어오므로 제스처가 바뀔 때만 로그
        if not self.gesture or g.get('g') != self.gesture.get('g'):
            self.log('/gesture', msg.data)
        self.gesture, self.gesture_t = g, time.time()

    def on_cmd(self, msg):
        cmd = (round(msg.linear.x, 2), round(msg.angular.z, 2))
        if cmd != self.cmd:
            self.log('/cmd_vel', f'linear.x={cmd[0]} angular.z={cmd[1]}')
        self.cmd = cmd

    def on_rosout(self, msg):
        if msg.name != self.get_name():
            self.log(msg.name, msg.msg, LEVELS.get(msg.level, str(msg.level)))

    def snapshot(self):
        with self.lock:
            logs = list(self.logs)
        age = time.time() - self.gesture_t if self.gesture_t else None
        return {'gesture': self.gesture, 'gesture_age': age, 'cmd': self.cmd,
                'has_video': self.jpeg is not None, 'logs': logs}


def main():
    rclpy.init()
    node = Dashboard()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    app = Flask(__name__)
    @app.get('/')
    def index():
        return PAGE

    @app.get('/state')
    def state():
        return jsonify(node.snapshot())

    def mjpeg(attr):
        def frames():
            while True:
                jpg = getattr(node, attr)
                if jpg:
                    yield b'--f\r\nContent-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n'
                time.sleep(0.1)
        return Response(frames(), mimetype='multipart/x-mixed-replace; boundary=f')

    @app.get('/stream')
    def stream():
        return mjpeg('jpeg')

    @app.get('/stream_esp')
    def stream_esp():
        return mjpeg('esp_jpeg')

    port = node.declare_parameter('port', 8090).value  # 8080/8000은 다른 프로젝트가 사용 중
    node.get_logger().info(f'dashboard: http://localhost:{port}')
    # 다른 기기에서 보려면 host='0.0.0.0'
    app.run(host='127.0.0.1', port=port, threaded=True)


PAGE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gesture Bot Dashboard</title>
<style>
  :root { --bg:#0f1115; --panel:#181b22; --line:#262a33; --text:#e6e8eb; --dim:#8a919c;
          --ok:#3ecf8e; --warn:#f5a524; --err:#f04f5e; --accent:#5b9cff; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font:14px/1.45 system-ui,sans-serif; }
  header { padding:12px 16px; border-bottom:1px solid var(--line); display:flex; gap:12px; align-items:center; }
  header h1 { font-size:16px; margin:0; font-weight:600; }
  .dot { width:9px; height:9px; border-radius:50%; background:var(--err); }
  .dot.on { background:var(--ok); }
  main { display:grid; grid-template-columns:minmax(0,1.4fr) minmax(0,1fr); gap:12px; padding:12px 16px; }
  @media (max-width:900px) { main { grid-template-columns:1fr; } }
  .panel { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px; min-width:0; }
  .panel h2 { font-size:12px; color:var(--dim); text-transform:uppercase; letter-spacing:.06em; margin:0 0 8px; }
  #esp { width:100%; max-width:480px; aspect-ratio:4/3; background:#000; border-radius:6px; object-fit:contain; display:block; }
  #video { width:100%; aspect-ratio:4/3; background:#000; border-radius:6px; object-fit:contain; display:block; }
  .stats { display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin-top:12px; }
  .stat .k { color:var(--dim); font-size:12px; }
  .stat .v { font-size:22px; font-weight:600; font-variant-numeric:tabular-nums; }
  #logs { height:calc(100vh - 140px); min-height:300px; overflow-y:auto; font:12px/1.5 ui-monospace,monospace; }
  .row { display:grid; grid-template-columns:62px 52px 130px 1fr; gap:6px; padding:2px 0; border-bottom:1px solid var(--line); }
  .row span { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .row .m { white-space:normal; word-break:break-word; }
  .WARN { color:var(--warn); } .ERROR, .FATAL { color:var(--err); } .INFO { color:var(--ok); } .DEBUG { color:var(--dim); }
  .src { color:var(--accent); }
</style></head><body>
<header><span id="dot" class="dot"></span><h1>Gesture Bot</h1><span id="link" style="color:var(--dim)">ESP32 대기 중</span></header>
<main>
  <section class="panel">
    <h2>Gazebo · overhead camera</h2>
    <img id="video" src="/stream" alt="Gazebo 상단 카메라 영상">
    <h2 style="margin-top:12px">ESP32 · camera</h2>
    <img id="esp" src="/stream_esp" alt="ESP32 카메라 영상">
    <div class="stats">
      <div class="stat"><div class="k">제스처</div><div class="v" id="g">-</div></div>
      <div class="stat"><div class="k">신뢰도</div><div class="v" id="s">-</div></div>
      <div class="stat"><div class="k">cmd_vel (v, ω)</div><div class="v" id="cmd">0, 0</div></div>
    </div>
  </section>
  <section class="panel">
    <h2>Logs</h2>
    <div id="logs"></div>
  </section>
</main>
<script>
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
let shown = 0;
async function tick() {
  try {
    const st = await (await fetch('/state')).json();
    const live = st.gesture_age !== null && st.gesture_age < 1.0;
    $('dot').className = 'dot' + (live ? ' on' : '');
    $('link').textContent = live ? 'ESP32 연결됨' : (st.gesture_age === null ? 'ESP32 대기 중' : 'ESP32 끊김 (' + st.gesture_age.toFixed(1) + 's)');
    $('g').textContent = st.gesture ? st.gesture.g : '-';
    $('s').textContent = st.gesture && st.gesture.s != null ? st.gesture.s.toFixed(2) : '-';
    $('cmd').textContent = st.cmd[0] + ', ' + st.cmd[1];
    const box = $('logs'), atBottom = box.scrollTop + box.clientHeight >= box.scrollHeight - 4;
    if (st.logs.length < shown) { box.innerHTML = ''; shown = 0; }
    // 서버 deque(maxlen=300)가 꽉 차면 앞에서 밀려나므로 전체 다시 그림
    if (st.logs.length === 300) { box.innerHTML = ''; shown = 0; }
    box.insertAdjacentHTML('beforeend', st.logs.slice(shown).map(l =>
      `<div class="row"><span>${l.t}</span><span class="${l.level}">${l.level}</span><span class="src">${esc(l.src)}</span><span class="m">${esc(l.msg)}</span></div>`).join(''));
    shown = st.logs.length;
    if (atBottom) box.scrollTop = box.scrollHeight;
  } catch (e) { $('dot').className = 'dot'; $('link').textContent = '대시보드 서버 연결 끊김'; }
}
setInterval(tick, 500); tick();
</script></body></html>
"""
