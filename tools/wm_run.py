import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"C:\Users\nenas\Desktop\Stickers")
OUT = Path(r"C:\Users\nenas\.local\share\opencode\wm")

STD_T = 10.0
EDGE_T = 20.0
MAX_H = 45
MAX_W = 400
MIN_AREA = 100
MAX_AREA = 9000
EDGE_MARGIN = 42


def detect_file(args: tuple[str, str]) -> tuple[str, list[list[int]]]:
    pack, fname = args
    path = ROOT / pack / fname
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
        return f"{pack}/{fname}", []
    stack = np.stack(frames)
    std = stack.std(axis=0)
    med = np.median(stack, axis=0).astype(np.uint8)
    sob = np.sqrt(cv2.Sobel(med, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(med, cv2.CV_32F, 0, 1) ** 2)
    mask = ((std < STD_T) & (sob > EDGE_T)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    h, w = mask.shape
    boxes = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < MIN_AREA or area > MAX_AREA or bh > MAX_H or bw > MAX_W:
            continue
        if not (y + bh >= h - EDGE_MARGIN or y <= EDGE_MARGIN or x <= EDGE_MARGIN or x + bw >= w - EDGE_MARGIN):
            continue
        fill = area / float(bw * bh)
        if fill < 0.12 or (bw / bh < 1.0 and bh > 30):
            continue
        boxes.append([int(x), int(y), int(bw), int(bh)])
    return f"{pack}/{fname}", boxes


def main():
    jobs = []
    for pack in sorted(p.name for p in ROOT.iterdir() if p.is_dir()):
        for f in sorted((ROOT / pack).glob("*.webm")):
            jobs.append((pack, f.name))
    print(f"{len(jobs)} files", flush=True)
    with Pool(processes=8) as pool:
        results = dict(pool.imap_unordered(detect_file, jobs, chunksize=8))
    per_pack: dict[str, dict[str, list]] = {}
    union: dict[str, list] = {}
    for key, boxes in results.items():
        pack, fname = key.split("/", 1)
        per_pack.setdefault(pack, {})[fname] = boxes
    for pack, files in per_pack.items():
        u = []
        for boxes in files.values():
            for b in boxes:
                if b not in u:
                    u.append(b)
        union[pack] = u
    (OUT / "file_boxes.json").write_text(json.dumps({"per_file": per_pack, "union": union}, indent=1), encoding="utf-8")
    for pack in sorted(per_pack):
        nb = len(union[pack])
        print(f"{pack}: union {nb} boxes")
    print("done")


if __name__ == "__main__":
    main()
