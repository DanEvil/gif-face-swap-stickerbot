FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FFMPEG_PATH=/usr/bin/ffmpeg

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY bot.py config.py faceswap.py packs.py subs.py db.py ./
COPY smoke_test.py ./
RUN mkdir -p stickers_src cache

CMD ["python", "bot.py"]
