"""
models/pt/yolo26n.pt -> NCNN / OpenVINO / TFLite 변환 (ONNX는 yolotoomnx.py 가 parity check 와 함께 변환)

실행 (PC 에서, 무거운 변환 의존성은 일회용 컨테이너에 설치):
  docker run --rm -v "$PWD:/app" -w /app python:3.11-slim-bookworm sh -c \
    "apt-get update -qq && apt-get install -y -qq libgl1 libglib2.0-0 >/dev/null && \
     pip install -q --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision ultralytics && \
     python export_models.py && chown -R $(id -u):$(id -g) models"

출력 (models/<포맷>/):
  ncnn/       model.ncnn.param, model.ncnn.bin
  openvino/   yolo26n.xml, yolo26n.bin
  tflite/     yolo26n.tflite   (ultralytics 8.4.83+ 는 LiteRT 로 export)

모두 FP32, 640x640, nms=False (YOLO26 end2end 헤드 -> 출력 (1, 300, 6), yolotoomnx.py 와 동일).
NCNN 은 end2end 를 지원하지 않아 ultralytics 가 자동으로 일반 헤드 (84, 8400) 로 export 한다.
"""
import shutil
from pathlib import Path

from ultralytics import YOLO

MODELS = Path(__file__).resolve().parent / "models"
PT_PATH = MODELS / "pt" / "yolo26n.pt"

for fmt in ("ncnn", "openvino", "tflite"):
    print(f"\n===== {fmt} =====")
    out = Path(YOLO(str(PT_PATH)).export(format=fmt, imgsz=640, nms=False))  # .pt 옆에 생성됨
    dst = MODELS / fmt
    shutil.rmtree(dst, ignore_errors=True)
    if out.is_dir():
        shutil.move(out, dst)
    else:
        dst.mkdir()
        shutil.move(out, dst / out.name)
    print(f"-> {dst}")
