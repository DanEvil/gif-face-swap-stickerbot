import sys
from pathlib import Path

import cv2

MAX_FRAMES = 8
SAMPLE_STEP_RATIO = 0.2


def iter_frames(path: Path, count: int = MAX_FRAMES):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or count
    step = max(1, int(total * SAMPLE_STEP_RATIO))
    for i in range(0, total, step):
        if len(range(0, total, step)) > count and i >= count * step:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if ok:
            yield frame
    cap.release()


def scan_file(path: Path, detector) -> list[tuple[str, str]]:
    hits = []
    for idx, frame in enumerate(iter_frames(path)):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        retval, _, _ = detector.detectAndDecode(gray)
        if retval:
            hits.append((f"frame{idx}", retval))
            continue
        bgr_retval, _, _ = detector.detectAndDecode(frame)
        if bgr_retval:
            hits.append((f"frame{idx}", bgr_retval))
    return hits


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\nenas\Desktop\Stickers")
    detector = cv2.QRCodeDetector()
    files = sorted(root.rglob("*.webm"))
    print(f"scanning {len(files)} files in {root}")
    flagged = 0
    for i, f in enumerate(files, 1):
        try:
            hits = scan_file(f, detector)
        except Exception as e:  # noqa: BLE001
            print(f"[ERR ] {f.relative_to(root)}: {e}")
            continue
        if hits:
            flagged += 1
            for fr, data in hits:
                print(f"[QR  ] {f.relative_to(root)} {fr}: {data!r}")
        if i % 25 == 0:
            print(f"... {i}/{len(files)} scanned, {flagged} flagged")
    print(f"DONE: {len(files)} scanned, {flagged} with QR")


if __name__ == "__main__":
    main()
