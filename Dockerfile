FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FFMPEG_PATH=/usr/bin/ffmpeg

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        build-essential \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

COPY requirements.txt .
# CPU-сборка onnxruntime из requirements заменяется на GPU
RUN pip install -r requirements.txt \
    && pip uninstall -y onnxruntime \
    && pip install --no-cache-dir onnxruntime-gpu==1.19.2

COPY bot.py config.py faceswap.py packs.py subs.py db.py ./
COPY smoke_test.py ./
RUN mkdir -p stickers_src cache

CMD ["python", "bot.py"]
