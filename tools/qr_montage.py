import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\nenas\Desktop\Stickers")
OUT = Path(r"C:\Users\nenas\.local\share\opencode\montage")
OUT.mkdir(parents=True, exist_ok=True)

THUMB = 256
PER_PACK = 4  # 2 different stickers x middle frames


def mid_frame(path: Path, size: int = THUMB):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)


def main():
    packs = sorted(p for p in ROOT.iterdir() if p.is_dir())
    for m, pack in enumerate(packs):
        tiles = []
        files = sorted(pack.glob("*.webm"))[:: max(1, len(list(pack.glob('*.webm'))) // 2)]
        for f in files[:2]:
            for _ in range(2):
                fr = mid_frame(f)
                if fr is None:
                    break
                tiles.append(fr)
                if len(tiles) >= PER_PACK:
                    break
        if not tiles:
            print(f"skip {pack.name}")
            continue
        row = np.hstack(tiles)
        cv2.putText(row, pack.name, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite(str(OUT / f"m{m:02d}_{pack.name}.png"), row)
    print("done")


if __name__ == "__main__":
    main()
