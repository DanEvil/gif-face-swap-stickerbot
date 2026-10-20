#!/usr/bin/env bash
# Обновление бота из GitHub и перезапуск контейнера.
set -e
cd "$(dirname "$0")"

if [ ! -f .env ]; then
  echo "Ошибка: нет .env. Скопируйте .env.example и заполните BOT_TOKEN."
  exit 1
fi

if [ ! -d .git ]; then
  echo "Ошибка: это не git-репозиторий. Клонируйте репозиторий и настройте .env."
  exit 1
fi

echo "[1/3] git pull..."
git pull --ff-only

echo "[2/3] Сборка образа..."
docker compose build

echo "[3/3] Перезапуск контейнеров..."
docker compose up -d --build

echo "Готово. Логи: docker compose logs -f bot"
