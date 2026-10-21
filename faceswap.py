import asyncio
import hashlib
import json
import os
import random
import shutil
import subprocess
import threading

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
_restore_sess = None
INSWAPPER_URL = os.getenv(
    "INSWAPPER_URL",
    "https://github.com/deepinsight/insightface/releases/download/model-zoo/inswapper_128.onnx",
)
INSWAPPER_SHA256 = "e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af"

RESTORE_ENABLED = os.getenv("FS_RESTORE", "1").strip().lower() in ("1", "true", "yes", "on")
RESTORE_BLEND = max(0.0, min(1.0, float(os.getenv("FS_RESTORE_BLEND", "80")) / 100))
CODEFORMER_URL = os.getenv(
    "FS_CODEFORMER_URL",
    "https://huggingface.co/facefusion/models-3.0.0/resolve/main/codeformer.onnx",
)
FFHQ_512 = np.array(
    [
        [0.37691676, 0.46864664],
        [0.62285697, 0.46912813],
        [0.50123859, 0.61331904],
        [0.39308822, 0.72541100],
        [0.61150205, 0.72490465],
    ],
    dtype="float32",
)


def _providers_list() -> list[str]:
    try:
        import onnxruntime as _ort

        avail = _ort.get_available_providers()
    except Exception:
        avail = []
    if "CUDAExecutionProvider" in avail:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _codeformer_file() -> str | None:
    env = os.getenv("FS_CODEFORMER_PATH", "")
    if env and os.path.isfile(env):
        return env
    if os.path.isfile("/opt/models/codeformer.onnx"):
        return "/opt/models/codeformer.onnx"
    try:
        from insightface.utils.storage import get_root

        root = get_root()
    except Exception:
        root = os.path.join(os.path.expanduser("~"), ".insightface")
    path = os.path.join(root, "models", "codeformer.onnx")
    if os.path.isfile(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Скачиваю codeformer.onnx в {path}...")
    import urllib.request

    tmp = path + ".part"
    try:
        urllib.request.urlretrieve(CODEFORMER_URL, tmp)
        os.replace(tmp, path)
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        print(f"Не удалось скачать CodeFormer: {e}")
        return None
    return path


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


_MODEL_LOCK = threading.RLock()


def _init_models():
    global _app, _swapper
    if _app is None:
        with _MODEL_LOCK:
            if _app is None:
                from insightface.app import FaceAnalysis
                from insightface.model_zoo.model_zoo import get_model

                provs = _providers_list()
                _app = FaceAnalysis(providers=provs)
                _app.prepare(ctx_id=0, det_size=(640, 640))
                _swapper = get_model(_ensure_inswapper_file())
            if _restore_sess is None and RESTORE_ENABLED:
                _init_restore()
    return _app


def _init_restore():
    global _restore_sess
    with _MODEL_LOCK:
        if _restore_sess is None:
            path = _codeformer_file()
            if not path:
                return None
            import onnxruntime as ort

            _restore_sess = ort.InferenceSession(path, providers=_providers_list())
    return _restore_sess


def _restore_face(frame, target_face):
    sess = _restore_sess
    if sess is None:
        return frame
    kps = np.asarray(target_face.kps, dtype="float32")
    template = FFHQ_512 * 512
    M = cv2.estimateAffinePartial2D(kps, template, method=cv2.RANSAC, ransacReprojThreshold=100)[0]
    if M is None:
        return frame
    crop = cv2.warpAffine(frame, M, (512, 512), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
    x = crop[:, :, ::-1] / 255.0
    x = (x - 0.5) / 0.5
    x = np.expand_dims(x.transpose(2, 0, 1), axis=0).astype(np.float32)
    feed = {"input": x}
    for out_input in sess.get_inputs():
        if out_input.name == "weight":
            feed["weight"] = np.array([1]).astype(np.float64)
    y = sess.run(None, feed)[0][0]
    y = np.clip(y, -1, 1)
    y = ((y + 1) / 2 * 255.0).round().astype(np.uint8)[:, :, ::-1]
    mask = np.ones((512, 512), dtype="float32")
    mask = cv2.GaussianBlur(mask, (0, 0), 20)
    mask = np.maximum(mask, 0)
    Minv = cv2.invertAffineTransform(M)
    h, w = frame.shape[:2]
    wmask = cv2.warpAffine(mask, Minv, (w, h)).clip(0, 1)
    wface = cv2.warpAffine(y, Minv, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    pasted = (
        frame * (1 - wmask[..., None]) + wface * wmask[..., None]
    ).astype(np.uint8)
    return cv2.addWeighted(frame, 1 - RESTORE_BLEND, pasted, RESTORE_BLEND, 0)


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


_SRC_FACES_DIR = os.path.join(CACHE_DIR, "src_faces")


def _src_faces_file(path: str) -> str:
    st = os.stat(path)
    h = hashlib.sha1()
    h.update(os.path.abspath(path).encode("utf-8", "ignore"))
    h.update(str(int(st.st_mtime)).encode())
    h.update(str(st.st_size).encode())
    return os.path.join(_SRC_FACES_DIR, h.hexdigest() + ".json")


def _faces_for_frames(path: str, frames):
    app = _init_models()
    cache = _src_faces_file(path)
    data = None
    if os.path.isfile(cache):
        try:
            with open(cache, encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list) and len(loaded) == len(frames):
                data = loaded
        except Exception:
            data = None
    if data is None:
        data = []
        for frame in frames:
            faces = app.get(frame)
            if faces:
                f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                data.append([f.bbox.tolist(), f.kps.tolist(), float(f.det_score)])
            else:
                data.append(None)
        os.makedirs(_SRC_FACES_DIR, exist_ok=True)
        tmp = cache + ".part"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, cache)
    out = []
    for entry in data:
        if entry is None:
            out.append(None)
            continue
        from insightface.app.common import Face

        out.append(
            Face(
                bbox=np.array(entry[0], dtype="float32"),
                kps=np.array(entry[1], dtype="float32"),
                det_score=entry[2],
            )
        )
    return out


def _swap_frames(frames, faces, source_face):
    out = []
    for frame, target in zip(frames, faces):
        if target is None:
            out.append(frame)
            continue
        swapped = _swapper.get(frame, target, source_face)
        out.append(_restore_face(swapped, target))
    return out


def _begin_swap_media(src: str, source_face, workdir: str, idx: int):
    frames, fps = _read_media(src)
    faces = _faces_for_frames(src, frames)
    h, w = frames[0].shape[:2]
    webm_path = os.path.join(workdir, f"sticker_{idx}.webm")
    if not _ffmpeg:
        raise RuntimeError("ffmpeg не найден в PATH")
    cmd = [
        _ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
        "-r", f"{fps:.4f}", "-i", "-",
        "-vf", "fps=30,scale=512:512:force_original_aspect_ratio=decrease,pad=512:512:(ow-iw)/2:(oh-ih)/2:white",
        "-c:v", "libvpx-vp9", "-crf", "26", "-b:v", "0",
        "-deadline", "good", "-cpu-used", "4", "-row-mt", "1",
        "-threads", "4",
        "-pix_fmt", "yuva420p", "-loop", "0", "-an", webm_path,
    ]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    try:
        for frame in _swap_frames(frames, faces, source_face):
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        proc.stdin.close()
    except Exception:
        proc.kill()
        raise
    return proc, webm_path


def _encode_from_webm(src_webm: str, out_webm: str, crf: int, extra: list[str] = ()):
    cmd = [
        _ffmpeg, "-y", "-loglevel", "error", "-i", src_webm,
        "-vf", "fps=30,scale=512:512:force_original_aspect_ratio=decrease,pad=512:512:(ow-iw)/2:(oh-ih)/2:white",
        "-c:v", "libvpx-vp9", "-crf", str(crf), "-b:v", "0",
        "-deadline", "good", "-cpu-used", "4", "-row-mt", "1",
        "-threads", "4",
        "-pix_fmt", "yuva420p", "-loop", "0", "-an",
        *extra, out_webm,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


MAX_WEBM_BYTES = 250 * 1024  # лимит Telegram для видео-стикера 256 КБ, с запасом


def _finalize_webm(proc, webm_path: str) -> str:
    rc = proc.wait()
    if rc != 0:
        err = (proc.stderr.read() or b"").decode("utf-8", "ignore")[-400:]
        raise RuntimeError(f"ffmpeg завершился с кодом {rc}: {err}")
    proc.stderr.close()
    if os.path.getsize(webm_path) <= MAX_WEBM_BYTES:
        return webm_path
    tmp = webm_path + ".tmp.webm"
    try:
        for crf in (30, 34, 38, 42):
            _encode_from_webm(webm_path, tmp, crf)
            if os.path.getsize(tmp) <= MAX_WEBM_BYTES:
                os.replace(tmp, webm_path)
                return webm_path
        _encode_from_webm(webm_path, tmp, 42, extra=["-t", "2.0"])
        if os.path.getsize(tmp) > MAX_WEBM_BYTES:
            raise RuntimeError(
                f"Стикер не влезает в лимит Telegram 256 КБ ({os.path.getsize(tmp) // 1024} КБ)"
            )
        os.replace(tmp, webm_path)
        return webm_path
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


async def generate_stickers(source_face, srcs: list[str], progress_cb=None) -> list[str]:
    if not srcs:
        raise RuntimeError(f"Не выбраны исходные стикеры")
    out_dir = os.path.join(CACHE_DIR, f"job{random.randrange(10**9)}")
    os.makedirs(out_dir, exist_ok=True)
    loop = asyncio.get_running_loop()
    results = []
    jobs = []
    for i, src in enumerate(srcs):
        proc, webm = await loop.run_in_executor(None, _begin_swap_media, src, source_face, out_dir, i)
        jobs.append((proc, webm))
        if progress_cb:
            await progress_cb(i + 1, len(srcs))
    for proc, webm in jobs:
        results.append(await loop.run_in_executor(None, _finalize_webm, proc, webm))
    return results
