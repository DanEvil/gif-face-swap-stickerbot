#!/usr/bin/env bash
# Первый деплой на VPS, запуск внутри склонированного репозитория.
set -e
cd "$(dirname "$0")"

if ! grep -qE "^BOT_TOKEN=.+" .env 2>/dev/null; then
  echo "[1/2] Настройка бота..."
  python3 setup.py
fi

echo "[2/2] Запуск контейнеров..."
docker compose up -d --build
echo "Деплой готов. Логи: docker compose logs -f bot"
echo "Повторная настройка: python3 setup.py && docker compose up -d --build"
