import sys
from pathlib import Path

import cv2
import numpy as np

STD_T = 6.0
EDGE_T = 12.0
MAX_H = 40
MAX_W = 350
MIN_AREA = 60
MAX_AREA = 8000
EDGE_MARGIN = 45  # box must be this close to a frame border

OUT = Path(r"C:\Users\nenas\.local\share\opencode\wm")
OUT.mkdir(parents=True, exist_ok=True)


def detect_boxes(path: Path, size: tuple[int, int] = (512, 512)) -> tuple[list[list[int]], np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    grays = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[:2] != (512, 512):
            frame = cv2.resize(frame, (512, 512))
        grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.release()
    if len(grays) < 2:
        return [], None
    stack = np.stack(grays)
    std = stack.std(axis=0)
    static = std < STD_T
    med = np.median(stack, axis=0).astype(np.uint8)
    sob = cv2.Sobel(med, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(med, cv2.CV_32F, 0, 1) ** 2
    sob = np.sqrt(sob)
    contrast = sob > EDGE_T * 4
    mask = (static & contrast).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    boxes = []
    h, w = mask.shape
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < MIN_AREA or area > MAX_AREA:
            continue
        if bh > MAX_H or bw > MAX_W:
            continue
        near = y + bh >= h - EDGE_MARGIN or y <= EDGE_MARGIN or x <= EDGE_MARGIN or x + bw >= w - EDGE_MARGIN
        if not near:
            continue
        boxes.append([int(x), int(y), int(bw), int(bh), int(area)])
    return boxes, med


def main():
    root = Path(r"C:\Users\nenas\Desktop\Stickers")
    packs = sys.argv[1:] or sorted(p.name for p in root.iterdir() if p.is_dir())
    for name in packs:
        pack = root / name
        files = sorted(pack.glob("*.webm"))
        if not files:
            continue
        sample = files[len(files) // 2]
        boxes, med = detect_boxes(sample)
        if med is not None:
            vis = cv2.cvtColor(med, cv2.COLOR_GRAY2BGR)
            for x, y, bw, bh, _ in boxes:
                cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
            cv2.putText(vis, name, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imwrite(str(OUT / f"pf_{name}.png"), vis)
        print(f"{name}: {len(boxes)} {boxes}")


if __name__ == "__main__":
    main()
