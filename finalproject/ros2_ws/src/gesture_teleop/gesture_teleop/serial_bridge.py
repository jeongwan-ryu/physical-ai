"""ESP32 Serial의 JSON -> /gesture(std_msgs/String) publish.

ESP_LOG 출력과 같은 포트를 쓰므로 한 줄 안에서 {...} 구간만 뽑아 쓴다.
"IMG:<base64 JPEG>" 줄은 /esp32_cam/image/compressed 로 publish.
실행: ros2 run gesture_teleop serial_bridge --ros-args -p port:=/dev/ttyACM0
"""
import base64
import binascii
import json
import re
import threading

import rclpy
import serial
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

JSON_RE = re.compile(r'\{[^{}]*\}')


def extract(line):
    """한 줄에서 유효한 JSON 객체 문자열들을 꺼낸다."""
    out = []
    for m in JSON_RE.findall(line):
        try:
            json.loads(m)
            out.append(m)
        except ValueError:
            pass
    return out


def extract_jpeg(line):
    """'IMG:<base64>' 줄이면 JPEG bytes, 아니면(또는 로그와 섞여 깨졌으면) None."""
    i = line.find('IMG:')
    if i < 0:
        return None
    try:
        jpg = base64.b64decode(line[i + 4:].strip(), validate=True)
    except (binascii.Error, ValueError):
        return None
    return jpg if jpg.startswith(b'\xff\xd8') else None


class SerialBridge(Node):
    def __init__(self):
        super().__init__('serial_bridge')
        port = self.declare_parameter('port', '/dev/ttyACM0').value
        baud = self.declare_parameter('baud', 115200).value
        self.pub = self.create_publisher(String, '/gesture', 10)
        self.img_pub = self.create_publisher(CompressedImage, '/esp32_cam/image/compressed', 1)
        self.ser = serial.Serial(port, baud, timeout=1)
        self.get_logger().info(f'serial_bridge: {port} -> /gesture')
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self):
        while rclpy.ok():
            try:
                line = self.ser.readline().decode(errors='replace')
            except serial.SerialException as e:
                # 끊기면 teleop_node 타임아웃으로 로봇이 정지한다
                self.get_logger().error(f'serial 끊김: {e}')
                return
            jpg = extract_jpeg(line)
            if jpg is not None:
                msg = CompressedImage(format='jpeg', data=jpg)
                msg.header.stamp = self.get_clock().now().to_msg()
                self.img_pub.publish(msg)
                continue
            for m in extract(line):
                self.pub.publish(String(data=m))


def main():
    rclpy.init()
    node = SerialBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
