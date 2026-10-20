import asyncio
import os
import random
import shutil
import subprocess

import cv2
import numpy as np
from PIL import Image

from config import CACHE_DIR, MAX_STICKERS, STICKER_SRC_DIR

_ffmpeg_env = os.getenv("FFMPEG_PATH", "")


def _find_ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    if _ffmpeg_env and os.path.isfile(_ffmpeg_env):
        return _ffmpeg_env
    winget_pkgs = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Packages")
    if os.path.isdir(winget_pkgs):
        for dirpath, dirnames, filenames in os.walk(winget_pkgs):
            if "ffmpeg.exe" in filenames:
                return os.path.join(dirpath, "ffmpeg.exe")
    return ""


_ffmpeg = _find_ffmpeg()
_app = None
_swapper = None
INSWAPPER_URL = os.getenv(
    "INSWAPPER_URL",
    "https://github.com/deepinsight/insightface/releases/download/model-zoo/inswapper_128.onnx",
)
INSWAPPER_SHA256 = "e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af"


def _ensure_inswapper_file() -> str:
    try:
        from insightface.utils.storage import get_root

        root = get_root()
    except Exception:
        root = os.path.join(os.path.expanduser("~"), ".insightface")
    path = os.path.join(root, "models", "inswapper_128.onnx")
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Скачиваю inswapper_128.onnx в {path}...")
    import hashlib
    import urllib.request

    tmp = path + ".part"
    urllib.request.urlretrieve(INSWAPPER_URL, tmp)
    h = hashlib.sha256()
    with open(tmp, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != INSWAPPER_SHA256:
        os.remove(tmp)
        raise RuntimeError("Контрольная сумма inswapper_128.onnx не совпала")
    os.replace(tmp, path)
    print("Готово")
    return path


def ffmpeg_available() -> bool:
    return _ffmpeg is not None


def _init_models():
    global _app, _swapper
    if _app is None:
        from insightface.app import FaceAnalysis
        from insightface.model_zoo.model_zoo import get_model

        _app = FaceAnalysis(providers=["CPUExecutionProvider"])
        _app.prepare(ctx_id=0, det_size=(640, 640))
        _swapper = get_model(_ensure_inswapper_file())
    return _app


def extract_source_face(image_path: str):
    app = _init_models()
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Не удалось прочитать изображение")
    faces = app.get(img)
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def list_sources() -> list[str]:
    if not os.path.isdir(STICKER_SRC_DIR):
        return []
    exts = (".gif", ".png", ".webp", ".webm")
    files = [f for f in os.listdir(STICKER_SRC_DIR) if f.lower().endswith(exts)]
    return [os.path.join(STICKER_SRC_DIR, f) for f in sorted(files)]


def list_packs() -> list[str]:
    if not os.path.isdir(STICKER_SRC_DIR):
        return []
    packs = []
    for name in sorted(os.listdir(STICKER_SRC_DIR)):
        path = os.path.join(STICKER_SRC_DIR, name)
        if os.path.isdir(path) and os.listdir(path):
            packs.append(name)
    return packs


def pick_pack_files(pack_name: str, need: int = MAX_STICKERS) -> list[str]:
    pack_dir = os.path.join(STICKER_SRC_DIR, pack_name)
    if not os.path.isdir(pack_dir):
        return []
    exts = (".gif", ".png", ".webp", ".webm")
    files = [f for f in os.listdir(pack_dir) if f.lower().endswith(exts)]
    files = [os.path.join(pack_dir, f) for f in sorted(files)]
    if len(files) <= need:
        return files
    return random.sample(files, need)


def pick_sources(need: int = MAX_STICKERS) -> list[str]:
    srcs = list_sources()
    if len(srcs) <= need:
        return srcs
    return random.sample(srcs, need)


def _read_gif_frames(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть {os.path.basename(path)}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 1 else 10.0
    frames = []
    ok, frame = cap.read()
    while ok:
        frames.append(frame)
        ok, frame = cap.read()
    cap.release()
    if not frames:
        raise ValueError(f"GIF пустой: {os.path.basename(path)}")
    return frames, fps


def _read_static_frames(path: str):
    img = Image.open(path)
    if getattr(img, "is_animated", False):
        frames = []
        for i in range(img.n_frames):
            img.seek(i)
            frames.append(cv2.cvtColor(np.array(img.convert("RGBA")), cv2.COLOR_RGBA2BGR))
    else:
        frames = [cv2.cvtColor(np.array(img.convert("RGBA")), cv2.COLOR_RGBA2BGR)]
    return frames, 10.0


def _read_media(path: str):
    if path.lower().endswith((".webp", ".png")):
        return _read_static_frames(path)
    return _read_gif_frames(path)


def _swap_frames(frames, source_face):
    app = _init_models()
    out = []
    for frame in frames:
        faces = app.get(frame)
        if not faces:
            out.append(frame)
            continue
        target = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        out.append(_swapper.get(frame, target, source_face))
    return out


def _frames_to_gif(frames, fps, out_path: str):
    imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)


def _swap_media(src: str, source_face, workdir: str, idx: int) -> str:
    frames, fps = _read_media(src)
    swapped = _swap_frames(frames, source_face)
    gif_path = os.path.join(workdir, f"swap_{idx}.gif")
    _frames_to_gif(swapped, fps, gif_path)
    webm_path = os.path.join(workdir, f"sticker_{idx}.webm")
    if not _ffmpeg:
        raise RuntimeError("ffmpeg не найден в PATH")
    cmd = [
        _ffmpeg, "-y", "-loglevel", "error", "-i", gif_path,
        "-c:v", "libvpx-vp9", "-crf", "30", "-b:v", "0", "-pix_fmt", "yuva420p",
        "-vf", "fps=30,scale=512:512:force_original_aspect_ratio=decrease,pad=512:512:(ow-iw)/2:(oh-ih)/2:white",
        "-loop", "0", "-an", webm_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    os.remove(gif_path)
    return webm_path


async def generate_stickers(source_face, srcs: list[str], progress_cb=None) -> list[str]:
    if not srcs:
        raise RuntimeError(f"Не выбраны исходные стикеры")
    out_dir = os.path.join(CACHE_DIR, f"job{random.randrange(10**9)}")
    os.makedirs(out_dir, exist_ok=True)
    loop = asyncio.get_running_loop()
    results = []
    for i, src in enumerate(srcs):
        webm = await loop.run_in_executor(None, _swap_media, src, source_face, out_dir, i)
        results.append(webm)
        if progress_cb:
            await progress_cb(i + 1, len(srcs))
    return results
