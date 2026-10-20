import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"C:\Users\nenas\Desktop\Stickers")
OUT = Path(r"C:\Users\nenas\.local\share\opencode\wm")

CASES = [
    ("Brat2FreeTopsticker.webm", 10),
    ("DuneFreetopsticker.webm", 10),
    ("TmeinHollywoodFreetopsticker.webm", 10),
    ("Fightclub12.webm", 10),
    ("IAmLegendFreeTopsticker.webm", 5),
    ("MrBean112.webm", 5),
    ("Boys1123.webm", 12),
]
PARAMS = [(6.0, 24.0), (10.0, 20.0), (12.0, 12.0)]


def detect(path: Path, std_t: float, edge_t: float):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[:2] != (512, 512):
            frame = cv2.resize(frame, (512, 512))
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.release()
    if len(frames) < 2:
        return None
    stack = np.stack(frames)
    std = stack.std(axis=0)
    med = np.median(stack, axis=0).astype(np.uint8)
    sob = np.sqrt(cv2.Sobel(med, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(med, cv2.CV_32F, 0, 1) ** 2)
    mask = ((std < std_t) & (sob > edge_t)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    h, w = mask.shape
    boxes = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 100 or area > 9000 or bh > 45 or bw > 400:
            continue
        if not (y + bh >= 470 or y <= 42 or x <= 35 or x + bw >= w - 35):
            continue
        boxes.append([int(x), int(y), int(bw), int(bh)])
    vis = cv2.cvtColor(med, cv2.COLOR_GRAY2BGR)
    for x, y, bw, bh in boxes:
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
    return boxes, vis


def main():
    for pack, idx in CASES:
        f = sorted((ROOT / pack).glob("*.webm"))[idx]
        for si, (std_t, edge_t) in enumerate(PARAMS):
            res = detect(f, std_t, edge_t)
            if res is None:
                continue
            boxes, vis = res
            cv2.imwrite(str(OUT / f"tune_{pack.split('Free')[0].split('.')[0]}_{si}.png"), vis)
            print(f"{pack} std<{std_t} edge>{edge_t}: {boxes}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
