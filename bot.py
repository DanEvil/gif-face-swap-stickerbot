import asyncio
import os
import random
import time

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ChatType
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
)

import config
import db
import packs
import subs
from faceswap import (
    extract_source_face,
    ffmpeg_available,
    generate_stickers,
    list_packs,
    pick_pack_files,
)

router = Router()
router.message.filter(F.chat.type == ChatType.PRIVATE)

_job_sem: asyncio.Semaphore = asyncio.Semaphore(config.JOBS_CONCURRENCY)
_job_waiting = 0

CHECK = {"ok": "\N{LARGE GREEN CIRCLE}", "fail": "\N{LARGE RED CIRCLE}"}

BTN_OURS = "\N{PERFORMING ARTS} Наши стикеры"
BTN_OWN = "\N{GRINNING FACE} Свои стикеры"
BTN_STATS = "\N{BAR CHART} Статистика"
BTN_MAIN = "\N{HOUSE BUILDING} Главное меню"
BTN_RANDOM_PACK = "\N{GAME DIE} Случайный пак"


class Flow(StatesGroup):
    choose_pack = State()
    waiting_own_sticker = State()
    waiting_photo = State()


def menu_kb() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        resize_keyboard=True,
        keyboard=[
            [KeyboardButton(text=BTN_OURS), KeyboardButton(text=BTN_OWN)],
            [KeyboardButton(text=BTN_STATS)],
        ],
    )


def packs_kb(pack_names: list[str]) -> ReplyKeyboardMarkup:
    rows: list[list[KeyboardButton]] = [[KeyboardButton(text=BTN_RANDOM_PACK)]]
    row: list[KeyboardButton] = []
    for name in pack_names:
        title = name.removesuffix(".webm").replace("_", " ")
        row.append(KeyboardButton(text=f"\N{CLAPPER BOARD} {title}"))
        if len(row) == 2:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([KeyboardButton(text=BTN_MAIN)])
    return ReplyKeyboardMarkup(resize_keyboard=True, keyboard=rows)


def gate_keyboard() -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    for ch in config.REQUIRED_CHANNELS:
        handle = ch.lstrip("@")
        rows.append([InlineKeyboardButton(
            text=f"\N{ROUND PUSHPIN} Канал {handle}",
            url=f"https://t.me/{handle}",
        )])
    if config.GATEWAY_REQUIRED and config.GATEWAY_BOT_USERNAME:
        handle = config.GATEWAY_BOT_USERNAME.lstrip("@")
        rows.append([InlineKeyboardButton(
            text=f"\N{ROBOT FACE} Стартовать @{handle}",
            url=f"https://t.me/{handle}",
        )])
    rows.append([InlineKeyboardButton(
        text="\N{WHITE HEAVY CHECK MARK} Проверить",
        callback_data="check_gate",
    )])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def checks_text(ch: dict) -> str:
    lines = []
    for handle, ok in ch["channels"].items():
        lines.append(f"{CHECK['ok' if ok else 'fail']} Подписка на канал {handle}")
    if ch["gateway"]["required"]:
        lines.append(
            f"{CHECK['ok' if ch['gateway']['ok'] else 'fail']} "
            f"Старт бота @{config.GATEWAY_BOT_USERNAME.lstrip('@')}"
        )
    return "\n".join(lines) + "\n\nВыполни условия, затем нажми «Проверить»."


def menu_text() -> str:
    return (
        f"\N{GRINNING FACE WITH SMILING EYES} Привет!\n\n"
        "Я создам тебе персональный стикерпак по фильмам и сериалам.\n\n"
        f"\N{PERFORMING ARTS} <Наши стикеры> — выберу тематический пак из библиотеки\n"
        f"\N{GRINNING FACE} <Свои стикеры> — пришли стикер из своего пака, и я возьму его за основу\n\n"
        "Для любого варианта потом пришли фото лица "
        "(\N{BUST IN SILHOUETTE} анфас, крупно, хорошее освещение)."
    )


def greeting_caption() -> str:
    return (
        "<b>Что умеет этот бот?</b>\n\n"
        "Привет! Я создам тебе персональный стикерпак по фильмам и сериалам.\n\n"
        f"\N{PERFORMING ARTS} «Наши стикеры» — тематические паки из библиотеки\n"
        f"\N{GRINNING FACE} «Свои стикеры» — пришли стикер из своего пака, и я возьму его за основу\n\n"
        "Для любого варианта потом пришли фото лица "
        "(\N{BUST IN SILHOUETTE} анфас, крупно, хорошее освещение).\n\n"
        "Нажми «Начать» и выбирай!"
    )


def _greeting_image_bytes() -> bytes | None:
    import cv2
    import numpy as np

    cache_path = os.path.join(config.CACHE_DIR, "greeting.jpg")
    try:
        if os.path.isfile(cache_path):
            with open(cache_path, "rb") as f:
                return f.read()
    except Exception:
        pass
    packs = list_packs()
    if not packs:
        return None
    random.shuffle(packs)
    frames = []
    for pack in packs:
        files = pick_pack_files(pack, 10)
        if not files:
            continue
        for src in random.sample(files, min(2, len(files))):
            cap = cv2.VideoCapture(src)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 3))
            ok, frame = cap.read()
            cap.release()
            if ok:
                frames.append(frame)
            if len(frames) >= 6:
                break
        if len(frames) >= 6:
            break
    if not frames:
        return None
    frames = frames[:6]
    while len(frames) < 6:
        frames.append(frames[-1])
    tiles = [cv2.resize(f, (512, 512)) for f in frames]
    try:
        grid = np.vstack([np.hstack(tiles[0:3]), np.hstack(tiles[3:6])])
    except Exception:
        return None
    ok, buf = cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        return None
    data = buf.tobytes()
    try:
        with open(cache_path, "wb") as f:
            f.write(data)
    except Exception:
        pass
    return data


async def send_greeting(message: Message):
    img = await asyncio.get_running_loop().run_in_executor(None, _greeting_image_bytes)
    if img:
        try:
            from aiogram.types import BufferedInputFile

            await message.answer_photo(
                BufferedInputFile(img, filename="start.jpg"),
                caption=greeting_caption(),
                reply_markup=menu_kb(),
                parse_mode="HTML",
            )
            return
        except Exception:
            pass
    await message.answer(menu_text(), reply_markup=menu_kb(), disable_web_page_preview=True)


async def checks_or_gate(message: Message, state: FSMContext, bot: Bot) -> bool:
    ch = await subs.checks(bot, message.from_user.id)
    if subs.flattened(ch):
        db.remember(message.from_user.id, message.from_user.first_name)
        return True
    await state.clear()
    await message.answer(
        f"\N{GRINNING FACE WITH SMILING EYES} Сначала выполни условия:\n\n" + checks_text(ch),
        reply_markup=gate_keyboard(),
        disable_web_page_preview=True,
    )
    return False


@router.message(F.text == BTN_MAIN)
async def btn_main(message: Message, state: FSMContext, bot: Bot):
    await cmd_start(message, state, bot)


@router.message(F.text == BTN_OURS)
async def btn_ours(message: Message, state: FSMContext, bot: Bot):
    await on_ours(message, state, bot)


@router.message(F.text == BTN_OWN)
async def btn_own(message: Message, state: FSMContext, bot: Bot):
    await on_own(message, state, bot)


@router.message(F.text == BTN_STATS)
async def btn_stats(message: Message, bot: Bot):
    await cmd_stats(message, bot)


@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext, bot: Bot):
    await state.clear()
    db.started(message.from_user.id)
    if not ffmpeg_available():
        await message.answer("\N{WARNING SIGN} ffmpeg не найден, генерация недоступна")
        return
    if not list_packs():
        await message.answer(
            f"\N{WARNING SIGN} Библиотека стикеров пуста: папка {config.STICKER_SRC_DIR}"
        )
        return
    if not await checks_or_gate(message, state, bot):
        return
    await send_greeting(message)


@router.callback_query(F.data == "check_gate")
async def cb_check(call: CallbackQuery, state: FSMContext, bot: Bot):
    ch = await subs.checks(bot, call.from_user.id)
    if subs.flattened(ch):
        db.remember(call.from_user.id, call.from_user.first_name)
        try:
            await call.message.edit_reply_markup()
        except TelegramBadRequest:
            pass
        await call.answer()
        await send_greeting(call.message)
        return
    await call.answer("Условия пока не выполнены", show_alert=True)


async def on_ours(message: Message, state: FSMContext, bot: Bot):
    if not await checks_or_gate(message, state, bot):
        return
    pack_names = list_packs()
    if not pack_names:
        await message.answer(f"\N{WARNING SIGN} Библиотека пуста: {config.STICKER_SRC_DIR}")
        return
    await state.set_state(Flow.choose_pack)
    await state.update_data(pack_names=pack_names)
    await message.answer(
        "\N{CLAPPER BOARD} Выбери пак из библиотеки — или жми «Случайный пак» (первая кнопка):",
        reply_markup=packs_kb(pack_names[:30]),
    )


@router.message(Flow.choose_pack, F.text == BTN_RANDOM_PACK)
async def on_random_pack(message: Message, state: FSMContext):
    data = await state.get_data()
    names = data.get("pack_names") or list_packs()
    await start_ours_pack(message, state, random.choice(names))


@router.message(Flow.choose_pack, F.text)
async def on_pick_pack(message: Message, state: FSMContext):
    data = await state.get_data()
    names = data.get("pack_names") or list_packs()
    wanted = message.text.removeprefix(f"\N{CLAPPER BOARD} ").lower()
    for name in names:
        if name.removesuffix(".webm").replace("_", " ").lower() == wanted:
            await start_ours_pack(message, state, name)
            return
    await message.answer("\N{NO ENTRY SIGN} Не понял выбор — нажми кнопку с названием пака")


def _preview_frames(srcs: list[str], count: int = 3) -> list[bytes]:
    import cv2

    out = []
    for src in srcs[:count]:
        cap = cv2.VideoCapture(src)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            continue
        if frame.shape[:2] != (512, 512):
            frame = cv2.resize(frame, (512, 512))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            out.append(buf.tobytes())
    return out


async def send_pack_preview(bot: Bot, chat_id: int, srcs: list[str]):
    frames = await asyncio.get_running_loop().run_in_executor(None, _preview_frames, srcs)
    if not frames:
        return
    from aiogram.types import BufferedInputFile, InputMediaPhoto

    media = [InputMediaPhoto(media=BufferedInputFile(b, filename=f"preview{i}.jpg")) for i, b in enumerate(frames)]
    try:
        await bot.send_media_group(chat_id, media=media)
    except Exception:
        pass


async def start_ours_pack(message: Message, state: FSMContext, pack_name: str):
    srcs = pick_pack_files(pack_name, config.MAX_STICKERS)
    if not srcs:
        await message.answer("\N{NO ENTRY SIGN} В этом паке нет стикеров — выбери другой")
        return
    await state.update_data(srcs=srcs, pack=pack_name)
    await state.set_state(Flow.waiting_photo)
    title = pack_name.removesuffix(".webm").replace("_", " ")
    await message.answer(
        f"\N{FILM FRAMES} Пак «{title}» — {len(srcs)} стикеров.\n"
        "Вот примеры исходников:"
    )
    await send_pack_preview(message.bot, message.chat.id, srcs)
    await message.answer(
        "Теперь пришли фото лица (\N{BUST IN SILHOUETTE} анфас, крупно, хорошее освещение)."
    )


async def on_own(message: Message, state: FSMContext, bot: Bot):
    if not await checks_or_gate(message, state, bot):
        return
    await state.set_state(Flow.waiting_own_sticker)
    await message.answer(
        "\N{INCOMING ENVELOPE} Пришли любой стикер из своего стикерпака "
        "(лучше видео-стикер)."
    )


@router.message(Flow.waiting_own_sticker, F.sticker)
async def on_own_sticker(message: Message, state: FSMContext, bot: Bot):
    sticker = message.sticker
    if sticker.is_animated:
        await message.answer("\N{NO ENTRY SIGN} Анимированные (TGS) стикеры не поддерживаются. Пришли видео-стикер")
        return
    if not sticker.set_name:
        await message.answer("\N{NO ENTRY SIGN} Это одиночный стикер, а не из набора. Пришли стикер из своего пака")
        return
    try:
        st_set = await bot.get_sticker_set(sticker.set_name)
    except TelegramBadRequest:
        await message.answer("\N{NO ENTRY SIGN} Не удалось получить набор. Пришли стикер из своего пака")
        return
    stickers = list(st_set.stickers)
    usable = [s for s in stickers if not s.is_animated]
    if not usable:
        await message.answer("\N{NO ENTRY SIGN} В наборе нет подходящих стикеров")
        return
    if len(usable) > config.MAX_STICKERS:
        usable = random.sample(usable, config.MAX_STICKERS)
    srcs_dir = os.path.join(config.CACHE_DIR, f"own{message.from_user.id}")
    os.makedirs(srcs_dir, exist_ok=True)
    srcs = []
    for i, s in enumerate(usable):
        f = await bot.get_file(s.file_id)
        ext = os.path.splitext(f.file_path or "")[1] or (".webm" if s.is_video else ".webp")
        dst = os.path.join(srcs_dir, f"s{i}{ext}")
        await bot.download(f, destination=dst)
        srcs.append(dst)
    title = (st_set.title or sticker.set_name)[:48]
    await state.update_data(srcs=srcs, pack=title)
    await state.set_state(Flow.waiting_photo)
    await message.answer(
        f"\N{FILM FRAMES} Взял твой набор «{title}» ({len(srcs)} стикеров).\n\n"
        "Теперь пришли фото лица (\N{BUST IN SILHOUETTE} анфас, крупно, хорошее освещение)."
    )


@router.message(Flow.waiting_own_sticker)
async def on_own_other(message: Message):
    await message.answer("\N{INCOMING ENVELOPE} Жду стикер из твоего пака — отправь его стикером, не файлом")


def photo_path(message: Message) -> tuple | None:
    if message.photo:
        file = message.photo[-1]
        ext = ".jpg"
    elif message.document and message.document.mime_type:
        file = message.document
        mime = message.document.mime_type
        ext = {
            "image/png": ".png",
            "image/webp": ".webp",
            "image/jpeg": ".jpg",
            "image/gif": ".gif",
        }.get(mime)
        if ext is None:
            return None
    else:
        return None
    return file, ext


@router.message(Flow.waiting_photo, F.photo | F.document)
async def on_image(message: Message, state: FSMContext, bot: Bot):
    data = await state.get_data()
    if data.get("busy"):
        await message.answer("\N{HOURGLASS WITH FLOWING SAND} Уже работаю над твоим набором")
        return
    got = photo_path(message)
    if got is None:
        await message.answer("\N{NO ENTRY SIGN} Это не фото. Пришли снимок лица")
        return
    if not data.get("srcs"):
        await state.clear()
        await message.answer(
            "\N{NO ENTRY SIGN} Не знаю, на какие стикеры менять лицо. "
            "Начни заново кнопкой «Заново»."
        )
        return
    file, ext = got
    await state.update_data(busy=True)
    await state.set_state(None)
    asyncio.create_task(run_job(message, bot, file, ext, data.get("srcs"), state))


async def run_job(message: Message, bot: Bot, file, ext: str, srcs: list[str], state: FSMContext):
    global _job_waiting
    position = None
    if _job_sem.locked():
        _job_waiting += 1
        position = _job_waiting
    try:
        await _job_sem.acquire()
    finally:
        if position:
            _job_waiting -= 1
    try:
        await process(message, bot, file, ext, srcs, queue_position=position)
    finally:
        _job_sem.release()
        await state.clear()


@router.message(Flow.waiting_photo)
async def on_other(message: Message):
    await message.answer(
        "\N{BUST IN SILHOUETTE} Пришли фото лица — обычное фото снимком, без текста."
    )


@router.message(Command("stats"))
async def cmd_stats(message: Message, bot: Bot):
    if not subs.is_admin(message.from_user.id):
        await message.answer("\N{LOCK} Доступно только администраторам")
        return
    s = db.stats()
    await message.answer(
        f"\N{CHART WITH UPWARDS TREND} Статистика\n\n"
        f"\N{BUST IN SILHOUETTE} Запустили бота: {s['users']}\n"
        f"\N{WHITE HEAVY CHECK MARK} Прошли гейт: {s['gated']}\n"
        f"\N{PAGE FACING UP} Стикерпаков создано: {s['packs']}"
    )


@router.message(F.sticker)
async def on_stray_sticker(message: Message, state: FSMContext, bot: Bot):
    await on_own_sticker(message, state, bot)


async def process(message: Message, bot: Bot, file, ext: str, srcs: list[str], queue_position: int | None = None):
    photos_dir = os.path.join(config.CACHE_DIR, "photos")
    os.makedirs(photos_dir, exist_ok=True)
    path = os.path.join(photos_dir, f"{message.from_user.id}_{round(time.time())}{ext}")
    try:
        await bot.download(file, destination=path)
    except Exception:
        await message.answer("\N{CROSS MARK} Не удалось скачать фото")
        return
    loop = asyncio.get_running_loop()
    try:
        source_face = await loop.run_in_executor(None, extract_source_face, path)
    except Exception as e:
        await message.answer(f"\N{CROSS MARK} Ошибка: {e}")
        return
    if source_face is None:
        await message.answer(
            "\N{FACE WITHOUT MOUTH} Не нашёл лицо на фото. "
            "Пришли фото с одним лицом в анфас и хорошим светом."
        )
        return
    os.remove(path)
    if queue_position:
        await message.answer(f"\N{HOURGLASS} Ты {queue_position}-й в очереди — жди результатов")
    status = await message.answer("\N{FERRIS WHEEL} Генерирую... Это займёт несколько минут")

    async def progress(done: int, total: int):
        try:
            await status.edit_text(f"\N{FERRIS WHEEL} Обработано стикеров: {done}/{total}")
        except TelegramBadRequest:
            pass

    try:
        final = await subs.checks(bot, message.from_user.id)
        if not subs.flattened(final):
            await status.edit_text(
                "\N{ROUND PUSHPIN} Условия больше не выполнены:\n\n"
                + checks_text(final)
                + "\n\nВыполни их и отправь фото ещё раз."
            )
            return
        webms = await generate_stickers(source_face, srcs, progress_cb=progress)
        link = await packs.create_sticker_pack(bot, message.from_user.id, message.from_user.first_name or "", webms)
        db.remember(message.from_user.id, message.from_user.first_name)
        db.mark_pack_created(message.from_user.id, link)
    except Exception as e:
        await status.edit_text(f"\N{CROSS MARK} Не получилось: {e}")
        return
    kb = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="\N{SPARKLES} Открыть стикерпак", url=link),
    ]])
    try:
        sample = random.choice(webms)
        await message.answer_sticker(FSInputFile(sample))
    except Exception:
        pass
    await status.edit_text(
        f"\N{GRINNING FACE WITH SMILING EYES} Готово!\n{link}\n\n"
        "\N{SLIGHTLY SMILING FACE} Если стикеры не применились сразу — закрой и открой Telegram.",
        reply_markup=kb,
        disable_web_page_preview=True,
    )


@router.message()
async def fallback(message: Message, state: FSMContext, bot: Bot):
    await cmd_start(message, state, bot)


async def main():
    bot = make_bot()
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)
    await bot.delete_webhook(drop_pending_updates=True)
    me = await bot.get_me()
    print(f"Бот запущен: @{me.username}"
          + (f" через прокси" if config.TELEGRAM_PROXY else ""))
    try:
        await dp.start_polling(bot, handle_signals=True)
    finally:
        await bot.session.close()


def make_bot() -> Bot:
    if config.TELEGRAM_PROXY:
        from aiogram.client.session.aiohttp import AiohttpSession

        session = AiohttpSession(proxy=config.TELEGRAM_PROXY)
        return Bot(token=config.BOT_TOKEN, session=session, default=DefaultBotProperties(parse_mode=None))
    if config.TELEGRAM_API_BASE_URL:
        return Bot(
            token=config.BOT_TOKEN,
            base_url=config.TELEGRAM_API_BASE_URL.rstrip("/"),
            default=DefaultBotProperties(parse_mode=None),
        )
    return Bot(token=config.BOT_TOKEN, default=DefaultBotProperties(parse_mode=None))


if __name__ == "__main__":
    asyncio.run(main())
