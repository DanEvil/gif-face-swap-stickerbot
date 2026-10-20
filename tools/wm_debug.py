from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"C:\Users\nenas\Desktop\Stickers")
OUT = Path(r"C:\Users\nenas\.local\share\opencode\wm")

for pack, idx in [("Brat2FreeTopsticker.webm", 10), ("DuneFreetopsticker.webm", 10), ("TmeinHollywoodFreetopsticker.webm", 10), ("Fightclub12.webm", 10)]:
    files = sorted((ROOT / pack).glob("*.webm"))
    f = files[idx]
    cap = cv2.VideoCapture(str(f))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[:2] != (512, 512):
            frame = cv2.resize(frame, (512, 512))
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.release()
    std = np.stack(frames).std(axis=0)
    vis_std = np.clip(std * 10, 0, 255).astype(np.uint8)
    med = np.median(np.stack(frames), axis=0).astype(np.uint8)
    band = np.zeros((512, 512), np.uint8)
    band[442:] = 255
    sel = (std[442:] < 15).astype(np.uint8) * 255
    dbg = cv2.cvtColor(med, cv2.COLOR_GRAY2BGR)
    dbg[442:][sel > 0] = (0, 0, 255)
    cv2.imwrite(str(OUT / f"dbg_{pack}.png"), dbg)
    region = std[442:512, 0:220]
    print(pack, f"idx{idx} nframes={len(frames)} band-std min={region.min():.1f} max={region.max():.1f} median={np.median(region):.1f} frac<6={(region < 6).mean():.2f} frac<12={(region < 12).mean():.2f}")
