# 라즈베리파이 배포 환경 흉내용 임시 리눅스 컨테이너
#   - Raspberry Pi OS (Bookworm) 와 같은 Debian 12 + Python 3.11
#   - 런타임 의존성만 설치 (numpy, opencv-headless, onnxruntime, ncnn, openvino, ai-edge-litert) -> torch/ultralytics 없이 동작하는지 확인
#   - --platform linux/arm64 로 빌드하면 라즈베리파이와 같은 ARM64 이미지 (QEMU 에뮬레이션)
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements-pi.txt .
# ncnn 이 GUI 버전 opencv-python 을 끌고 와서 headless 를 덮어씀 (slim 이미지엔 libxcb 없음) -> headless 로 되돌림
RUN pip install -r requirements-pi.txt \
 && pip uninstall -y opencv-python \
 && pip install --force-reinstall --no-deps opencv-python-headless

# 모델/이미지/코드 (.dockerignore 로 .pt, runs/ 등은 제외)
COPY . .

CMD ["python", "benchmark.py", "--backend", "onnx", "--threads", "4"]
