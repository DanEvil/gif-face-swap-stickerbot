"""Интерактивная настройка: заполняет .env. Запускается на хосте VPS: python3 setup.py"""
import sys

ENV_PATH = ".env"
MASK = "******"


def read_env() -> dict:
    data = {}
    try:
        with open(ENV_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                data[k.strip()] = v.strip()
    except FileNotFoundError:
        pass
    return data


def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        raw = input(f"{prompt}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nОтменено")
        sys.exit(1)
    return raw or default


def ask_bool(prompt: str, default: bool = True) -> bool:
    d = "Y/n" if default else "y/N"
    while True:
        raw = ask(f"{prompt} ({d})").lower()
        if not raw:
            return default
        if raw in ("1", "y", "yes", "д", "да"):
            return True
        if raw in ("0", "n", "no", "н", "нет"):
            return False
        print("Отвечайте y или n")


def normalize_channel(raw: str) -> str | None:
    raw = raw.strip()
    for prefix in ("https://t.me/", "http://t.me/", "t.me/"):
        if raw.lower().startswith(prefix):
            raw = raw[len(prefix):]
    raw = raw.lstrip("@")
    if not raw or "/" in raw or "?" in raw:
        return None
    return "@" + raw


def normalize_token(raw: str) -> str:
    return raw.strip().replace(" ", "")


def validate_token(token: str) -> str | None:
    import json
    import urllib.request

    try:
        with urllib.request.urlopen(
            f"https://api.telegram.org/bot{token}/getMe", timeout=10
        ) as resp:
            data = json.load(resp)
        if data.get("ok"):
            return data["result"]["username"]
    except Exception as e:
        print(f"  Токен не принят Telegram API: {e}")
    return None


def ask_ids(default: list[str]) -> list[int]:
    raw = ask("ID администраторов через запятую", ",".join(default))
    out = []
    for part in raw.replace(" ", "").split(","):
        part = part.strip().lstrip("@")
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            print(f"  Пропущено не-число: {part}")
    return out


def main():
    env = read_env()

    print("\n=== Настройка Faceswap GIF Sticker Bot ===\n")

    existing = [x for x in env.get("ADMIN_IDS", "").split(",") if x]
    if existing:
        print(f"  Текущие админы: {existing}")
    admin_ids = ask_ids(existing)

    existing_channels = [c for c in env.get("REQUIRED_CHANNELS", "").split(",") if c]
    require_channel = ask_bool("Требовать подписку на канал?", bool(existing_channels))
    channels: list[str] = []
    if require_channel:
        default_n = str(max(1, len(existing_channels))) if existing_channels else "1"
        n = int(ask("На сколько каналов проверять подписку", default_n) or "1")
        allowed = bool(existing_channels) and n == len(existing_channels)
        for i in range(1, max(1, n) + 1):
            existing = existing_channels[i - 1] if i <= len(existing_channels) else ""
            while True:
                raw = ask(f"Ссылка/имя канала #{i}", existing)
                ch = normalize_channel(raw or "")
                if ch:
                    break
                print("  Не похоже на ссылку канала. Пример: @mychannel или https://t.me/mychannel")
            channels.append(ch)

    gateway = env.get("GATEWAY_BOT_USERNAME", "KO_Gateway_bot")
    if gateway:
        print(f"  Текущий гейт-бот: @{gateway}")
    gateway_name = ask("Имя гейт-бота (@username)", gateway).lstrip("@")
    require_gateway = ask_bool("Требовать старт гейт-бота?", env.get("GATEWAY_REQUIRED", "1") != "0")

    print("\n-- Токен бота")
    if env.get("BOT_TOKEN"):
        print("  Токен уже сохранён — Enter оставит прежний.")
    while True:
        raw = ask("BOT_TOKEN", MASK if env.get("BOT_TOKEN") else "")
        token = env.get("BOT_TOKEN", "") if raw in ("", MASK) else normalize_token(raw)
        if not token:
            print("  Нужен токен от @BotFather. Получите его и введите.")
            continue
        username = validate_token(token)
        if username:
            print(f"  OK: бот @{username}")
            break

    lines = [
        f"BOT_TOKEN={token}",
        f"ADMIN_IDS={','.join(map(str, admin_ids))}",
        f"REQUIRED_CHANNELS={','.join(channels)}",
        f"GATEWAY_REQUIRED={'1' if require_gateway else '0'}",
        f"GATEWAY_BOT_USERNAME={gateway_name}",
        f"STICKER_SRC_DIR={env.get('STICKER_SRC_DIR', 'stickers_src')}",
        f"CACHE_DIR={env.get('CACHE_DIR', 'cache')}",
        f"MAX_STICKERS={env.get('MAX_STICKERS', '10')}",
        f"PACK_TITLE_TEMPLATE={env.get('PACK_TITLE_TEMPLATE', 'Faceswap {{name}} by {{bot}}')}",
        f"STICKER_EMOJI={env.get('STICKER_EMOJI', '\N{GRINNING FACE}')}",
    ]
    with open(ENV_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nСохранено в {ENV_PATH}:")
    for ln in lines:
        print(" ", "BOT_TOKEN=******" if ln.startswith("BOT_TOKEN") else ln)
    print("\nЗапуск: docker compose up -d --build")


if __name__ == "__main__":
    main()
