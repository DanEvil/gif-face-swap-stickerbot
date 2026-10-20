from pathlib import Path

import cv2

OUT = Path(r"C:\Users\nenas\.local\share\opencode\wm")
SRC = Path(r"C:\Users\nenas\Desktop\Stickers")
DST = Path(r"C:\Users\nenas\Desktop\Stickers_clean")
PACK = "IAmLegendFreeTopsticker.webm"
NAMES = [
    "IAmLegendFreeTopsticker_AgAD-DAAAsAesUo.webm",
    "IAmLegendFreeTopsticker_AgADPCsAAonPsUo.webm",
    "IAmLegendFreeTopsticker_AgADoy8AAsfGsUo.webm",
]
tiles = []
for n in NAMES:
    for root in (SRC, DST):
        cap = cv2.VideoCapture(str(root / PACK / n))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            continue
        if frame.shape[:2] != (512, 512):
            frame = cv2.resize(frame, (512, 512))
        crop = frame[440:512, 0:280]
        tiles.append(cv2.resize(crop, (560, 144)))
img = np.vstack(tiles) if False else None
import numpy as np
img = np.vstack(tiles)
cv2.imwrite(str(OUT / "chk2.png"), img)
print("saved", [t.shape for t in tiles])
