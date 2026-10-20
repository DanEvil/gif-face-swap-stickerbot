import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"C:\Users\nenas\Desktop\Stickers")
OUT = Path(r"C:\Users\nenas\.local\share\opencode\wm")
OUT.mkdir(parents=True, exist_ok=True)

STD_T = 6.0        # pixel considered "static" if temporal std < this
FRAC_T = 0.8       # pixel is watermark if static in >80% of pack files
MIN_AREA = 80      # ignore tiny specks
MAX_FRAC = 0.35    # component bigger than 35% of frame -> not a watermark


def temporal_std(path: Path) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        frames.append(gray)
    cap.release()
    if len(frames) < 2:
        return None
    stack = np.stack(frames)
    return stack.std(axis=0)


def components_boxes(mask: np.ndarray) -> list[list[int]]:
    m = (mask * 255).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    boxes = []
    h, w = mask.shape
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < MIN_AREA:
            continue
        if bw * bh > MAX_FRAC * w * h:
            continue
        boxes.append([int(x), int(y), int(bw), int(bh), int(area)])
    return boxes


def main():
    report = {}
    packs = sorted(p for p in ROOT.iterdir() if p.is_dir())
    for pack in packs:
        try:
            files = sorted(pack.glob("*.webm"))
            acc = np.zeros((512, 512), dtype=np.float32)
            count = 0
            sample_frame = None
            for f in files:
                s = temporal_std(f)
                if s is None:
                    continue
                acc += (s < STD_T).astype(np.float32)
                count += 1
                if sample_frame is None:
                    cap = cv2.VideoCapture(str(f))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2))
                    _, sample_frame = cap.read()
                    cap.release()
            if count == 0:
                report[pack.name] = []
                continue
            frac = acc / count
            mask = frac > FRAC_T
            boxes = components_boxes(mask)
            report[pack.name] = boxes
            if sample_frame is not None and boxes:
                vis = sample_frame.copy()
                for x, y, bw, bh, area in boxes:
                    cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
                cv2.imwrite(str(OUT / f"{pack.name}.png"), vis)
                print(f"{pack.name}: {len(boxes)} box(es) {boxes}")
            else:
                print(f"{pack.name}: clean")
        except Exception as e:  # noqa: BLE001
            print(f"{pack.name}: ERROR {e!r}")
        sys.stdout.flush()
    (OUT / "wm_boxes.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("json saved")


if __name__ == "__main__":
    main()
