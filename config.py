import os

import dotenv

dotenv.load_dotenv()


def _split(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _bool(value: str, default: bool) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "y", "on")


BOT_TOKEN = os.getenv("BOT_TOKEN", "")
TELEGRAM_PROXY = os.getenv("TELEGRAM_PROXY", "").strip()
TELEGRAM_API_BASE_URL = os.getenv("TELEGRAM_API_BASE_URL", "").strip()

_required_channels_env = os.getenv("REQUIRED_CHANNELS")
if _required_channels_env is None:
    _required_channels_env = os.getenv("REQUIRED_CHANNEL", "")
REQUIRED_CHANNELS = _split(_required_channels_env)

GATEWAY_BOT_USERNAME = os.getenv("GATEWAY_BOT_USERNAME", "KO_Gateway_bot")
GATEWAY_REQUIRED = _bool(os.getenv("GATEWAY_REQUIRED"), True)

ADMIN_IDS = [int(x) for x in _split(os.getenv("ADMIN_IDS", ""))]

STICKER_SRC_DIR = os.getenv("STICKER_SRC_DIR", "stickers_src")
CACHE_DIR = os.getenv("CACHE_DIR", "cache")
BOT_DB_PATH = os.getenv("BOT_DB_PATH", "data/bot.db")
MAX_STICKERS = int(os.getenv("MAX_STICKERS", "10"))
PACK_TITLE_TEMPLATE = os.getenv("PACK_TITLE_TEMPLATE", "Faceswap {name} by {bot}")
STICKER_EMOJI = os.getenv("STICKER_EMOJI", "\U0001f600")
