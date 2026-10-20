import asyncio
import os

import cv2
import numpy as np
from PIL import Image

import faceswap


def make_test_gif(out: str, img: np.ndarray):
    h, w = img.shape[:2]
    frames = []
    for dx in range(0, 3):
        m = np.roll(img, dx * 10, axis=1)
        frames.append(Image.fromarray(cv2.cvtColor(m, cv2.COLOR_BGR2RGB)))
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=120, loop=0)


async def main():
    from insightface.data import get_image

    img = get_image("t1")
    os.makedirs(faceswap.STICKER_SRC_DIR, exist_ok=True)
    gif_path = os.path.join(faceswap.STICKER_SRC_DIR, "test_source.gif")
    make_test_gif(gif_path, img)

    tmp_photo = os.path.join(faceswap.CACHE_DIR, "t1.jpg")
    os.makedirs(faceswap.CACHE_DIR, exist_ok=True)
    cv2.imwrite(tmp_photo, img)
    face = faceswap.extract_source_face(tmp_photo)
    assert face is not None, "лицо не найдено на сэмпле"
    print("OK: face extracted")

    async def progress(done, total):
        print(f"  {done}/{total}")

    webms = await faceswap.generate_stickers(face, [gif_path], progress_cb=progress)
    for w in webms:
        size = os.path.getsize(w)
        cap = cv2.VideoCapture(w)
        ok, frame = cap.read()
        cap.release()
        assert ok and size > 1000, f"битый webm: {w}"
        print(f"OK: {os.path.basename(w)} {size} bytes")
    print("PIPELINE OK")


if __name__ == "__main__":
    asyncio.run(main())
