from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError

import config
import db

_GATEWAY_API_URL = None  # заполняется, если у KO Gateway есть API проверки старта


def is_admin(user_id: int) -> bool:
    return user_id in config.ADMIN_IDS


async def channel_member(bot: Bot, user_id: int, channel: str) -> bool:
    try:
        member = await bot.get_chat_member(chat_id=channel, user_id=user_id)
    except (TelegramForbiddenError, TelegramBadRequest):
        return False
    status = getattr(member, "status", None)
    if status in ("creator", "administrator", "member"):
        return True
    if status == "restricted":
        return bool(getattr(member, "is_member", False))
    return False


async def is_gateway_started(bot: Bot, user_id: int) -> bool:
    if _GATEWAY_API_URL:
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    _GATEWAY_API_URL, params={"user_id": user_id}, timeout=10
                ) as resp:
                    if resp.status == 200:
                        return bool((await resp.json()).get("started", False))
        except Exception:
            return False
    return True


async def checks(bot: Bot, user_id: int) -> dict:
    """Полная проверка условий доступа.

    Для админов всё считается пройденным.
    Гейт-бот: старт проверяется только если ранее не подтверждён.
    """
    if is_admin(user_id):
        return {
            "channels": {ch: True for ch in config.REQUIRED_CHANNELS},
            "gateway": {"required": config.GATEWAY_REQUIRED, "ok": True},
        }
    channels = {}
    for ch in config.REQUIRED_CHANNELS:
        channels[ch] = await channel_member(bot, user_id, ch)
    gateway = {"required": config.GATEWAY_REQUIRED, "ok": True}
    if config.GATEWAY_REQUIRED:
        user = db.get_user(user_id)
        if user and user.get("gateway_passed"):
            gateway["ok"] = True
        else:
            gateway["ok"] = await is_gateway_started(bot, user_id)
    return {"channels": channels, "gateway": gateway}


def flattened(checks_dict: dict) -> bool:
    return all(checks_dict["channels"].values()) and checks_dict["gateway"]["ok"]
