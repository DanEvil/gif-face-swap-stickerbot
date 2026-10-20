import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"C:\Users\nenas\Desktop\Stickers")
OUT = Path(r"C:\Users\nenas\.local\share\opencode\wm")
PACKS = sys.argv[1:]
BAND = 40  # px of bottom/top strip


def strip_frames(pack: Path, top: bool, n_files: int = 6):
    files = sorted(pack.glob("*.webm"))
    step = max(1, len(files) // n_files)
    files = files[::step][:n_files]
    strips = []
    for f in files:
        cap = cv2.VideoCapture(str(f))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            continue
        h, w = frame.shape[:2]
        sy = 0 if top else h - BAND
        s = frame[sy:sy + BAND]
        if s.shape[0] != BAND:
            continue
        if s.shape[1] != w:
            continue
        strips.append(s)
    if not strips:
        return None
    wmax = max(s.shape[1] for s in strips)
    strips = [cv2.copyMakeBorder(cv2.resize(s, (int(wmax * s.shape[0] / BAND), BAND)), 0, 0, 0, 2, cv2.BORDER_CONSTANT) for s in strips]
    return np.hstack(strips)


def main():
    for i, name in enumerate(PACKS):
        pack = ROOT / name
        bot = strip_frames(pack, False)
        top = strip_frames(pack, True)
        parts = []
        for s in (top, bot):
            if s is not None:
                parts.append(s)
        if not parts:
            continue
        wmax = max(p.shape[1] for p in parts)
        img = np.vstack([cv2.copyMakeBorder(p, 0, 4, 0, wmax - p.shape[1], cv2.BORDER_CONSTANT) for p in parts])
        img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
        cv2.putText(img, name, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite(str(OUT / f"strip_{name}.png"), img)
        print("saved", name)


if __name__ == "__main__":
    main()
