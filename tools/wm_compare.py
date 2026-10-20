import sys
from pathlib import Path

import cv2
import numpy as np

SRC = Path(r"C:\Users\nenas\Desktop\Stickers")
DST = Path(r"C:\Users\nenas\Desktop\Stickers_clean")
OUT = Path(r"C:\Users\nenas\.local\share\opencode\wm")


def mid(path: Path) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    if frame.shape[:2] != (512, 512):
        frame = cv2.resize(frame, (512, 512))
    return cv2.resize(frame, (256, 256))


def main():
    pack = sys.argv[1] if len(sys.argv) > 1 else "IAmLegendFreeTopsticker.webm"
    rows = []
    files = sorted((SRC / pack).glob("*.webm"))[::10][:5]
    for f in files:
        a = mid(f)
        b = mid(DST / pack / f.name)
        if a is None or b is None:
            continue
        sep = np.full((256, 4, 3), 128, np.uint8)
        rows.append(np.hstack([a, sep, b, sep]))
    img = np.vstack([np.hstack([r, np.full((r.shape[0], 4, 3), 128, np.uint8)]) for r in rows[:5]])
    cv2.putText(img, f"{pack} before|after", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(OUT / f"cmp_{pack}.png"), img)
    print("saved")


if __name__ == "__main__":
    main()
