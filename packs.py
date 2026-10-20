import math
import random

from aiogram import Bot
from aiogram.types import FSInputFile, InputSticker

import config


def build_pack_name(user_id: int, bot_username: str) -> str:
    bot_name = bot_username.lstrip("@")
    name = f"fs{user_id}_{random.randrange(100, 999)}_by_{bot_name}"
    name = "".join(ch for ch in name if ch.isalnum() or ch == "_")
    if len(name) > 64:
        name = name[:64]
    suffix = f"_by_{bot_name}"
    if not name.endswith(suffix):
        name = f"{user_id}_by_{bot_name}"[:64]
        if len(name) > 64:
            name = f"fs{user_id % 10**7}_by_{bot_name}"
    return name


def build_pack_title(first_name: str, bot_username: str) -> str:
    title = config.PACK_TITLE_TEMPLATE.format(name=first_name, bot="@" + bot_username.lstrip("@"))
    return title[:64]


async def create_sticker_pack(bot: Bot, user_id: int, first_name: str, webm_paths: list[str]) -> str:
    me = await bot.get_me()
    bot_username = me.username or ""
    pack_name = build_pack_name(user_id, bot_username)
    title = build_pack_title(first_name, bot_username)
    stickers = [
        InputSticker(sticker=FSInputFile(path), emoji_list=[config.STICKER_EMOJI], format="video")
        for path in webm_paths
    ]
    await bot.create_new_sticker_set(
        user_id=user_id,
        name=pack_name,
        title=title,
        stickers=stickers,
        sticker_type="regular",
    )
    return f"https://t.me/addstickers/{pack_name}"
