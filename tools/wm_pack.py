import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"C:\Users\nenas\Desktop\Stickers")
OUT = Path(r"C:\Users\nenas\.local\share\opencode\wm")

STD_T = 6.0
FRAC_T = 0.8
BOTTOM_BAND = 70
TOP_BAND = 50
SIDE_BAND = 40
MAX_W = 400
MAX_H = 150
MIN_AREA = 60


def std_map(path: Path) -> np.ndarray | None:
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
    return np.stack(frames).std(axis=0)


def band_mask() -> np.ndarray:
    h, w = 512, 512
    m = np.zeros((h, w), dtype=bool)
    m[h - BOTTOM_BAND:, :] = True
    m[:TOP_BAND, :] = True
    m[:, :SIDE_BAND] = True
    m[:, w - SIDE_BAND:] = True
    return m


def pack_boxes(pack: Path):
    files = sorted(pack.glob("*.webm"))
    acc = np.zeros((512, 512), dtype=np.float32)
    count = 0
    sample = None
    for f in files:
        s = std_map(f)
        if s is None:
            continue
        acc += (s < STD_T).astype(np.float32)
        count += 1
        if sample is None:
            cap = cv2.VideoCapture(str(f))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))
            _, sample = cap.read()
            if sample is not None and sample.shape[:2] != (512, 512):
                sample = cv2.resize(sample, (512, 512))
            cap.release()
    if count == 0:
        return [], None
    frac = acc / count
    band = band_mask()
    mask = ((frac > FRAC_T) & band).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    boxes = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < MIN_AREA:
            continue
        if bw > MAX_W or bh > MAX_H:
            continue
        boxes.append([int(x), int(y), int(bw), int(bh)])
    return boxes, sample


def main():
    report = {}
    packs = sys.argv[1:] or sorted(p.name for p in ROOT.iterdir() if p.is_dir())
    for name in packs:
        try:
            boxes, sample = pack_boxes(ROOT / name)
            report[name] = boxes
            if sample is not None and boxes:
                vis = sample.copy()
                for x, y, bw, bh in boxes:
                    cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
                cv2.putText(vis, name, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.imwrite(str(OUT / f"pk_{name}.png"), vis)
            print(f"{name}: {len(boxes)} {boxes}")
        except Exception as e:  # noqa: BLE001
            print(f"{name}: ERROR {e!r}")
        sys.stdout.flush()
    (OUT / "pack_boxes.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("json saved")


if __name__ == "__main__":
    main()
