# Build on either linux/amd64 or linux/arm64; no GPU or PyTorch required.
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    ORT_THREADS=2 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app
# Headless OpenCV needs no X11, Mesa, GTK or libGL packages.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --no-create-home app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
RUN python -c "import cv2, onnxruntime; print('CPU runtime imports OK')"
COPY main.py download_model.py ./
COPY templates ./templates
COPY static ./static
COPY models/NOTICE.md models/MODEL_LICENSE.txt ./models/
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["python", "main.py"]
