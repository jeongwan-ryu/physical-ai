"""serial_bridge.extract 체크. 실행: python3 test/test_extract.py (ROS 환경 source 후)"""
from gesture_teleop.serial_bridge import extract


def test_extract():
    # 실제로 관측된 로그+JSON 혼합 줄
    assert extract('I (246) esp_image: segment 1: vaddr=3fc9ac{"g":"no_hand","s":1.00,"x":-1,"seq":60}\n') == \
        ['{"g":"no_hand","s":1.00,"x":-1,"seq":60}']
    assert extract('{"g":"like","s":0.97,"x":0.42,"seq":1}') == ['{"g":"like","s":0.97,"x":0.42,"seq":1}']
    assert extract('\x1b[0;32mI (13861) gesture_cam: 7.4 fps\x1b[0m') == []
    assert extract('{"g":"like","s":0.9') == []  # 잘린 줄
    assert extract('{broken}') == []


def test_extract_jpeg():
    import base64
    from gesture_teleop.serial_bridge import extract_jpeg
    jpg = b'\xff\xd8\xff\xe0fakejpeg\xff\xd9'
    b64 = base64.b64encode(jpg).decode()
    assert extract_jpeg('IMG:' + b64 + '\n') == jpg
    assert extract_jpeg('I (123) gesture_cam: IMG:' + b64) == jpg  # 앞에 로그가 붙어도 OK
    assert extract_jpeg('IMG:' + b64[:-3] + 'I (1') is None          # 로그가 끼어들어 깨진 줄
    assert extract_jpeg('{"g":"no_hand"}') is None


if __name__ == '__main__':
    test_extract()
    test_extract_jpeg()
    print('ok')
