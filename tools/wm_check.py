from pathlib import Path

import cv2

OUT = Path(r"C:\Users\nenas\.local\share\opencode\wm")
for tag, p in [
    ("after", Path(r"C:\Users\nenas\Desktop\Stickers_clean\IAmLegendFreeTopsticker.webm\IAmLegendFreeTopsticker_AgADDC8AAvuBsEo.webm")),
    ("before", Path(r"C:\Users\nenas\Desktop\Stickers\IAmLegendFreeTopsticker.webm\IAmLegendFreeTopsticker_AgADDC8AAvuBsEo.webm")),
]:
    cap = cv2.VideoCapture(str(p))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))
    ok, frame = cap.read()
    cap.release()
    print(tag, "ok", ok, "size", None if frame is None else frame.shape)
    if ok:
        crop = frame[430:512, 0:260]
        cv2.imwrite(str(OUT / f"chk_{tag}.png"), cv2.resize(crop, (520, 164)))
