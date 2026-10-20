from pathlib import Path

import cv2

SRC = Path(r"C:\Users\nenas\Desktop\Stickers\TheGreenMiles2Freetopsticker.webm\TheGreenMiles2Freetopsticker_AgADCjcAAl5GuUo.webm")
OUT = Path(r"C:\Users\nenas\.local\share\opencode\qr_all")
OUT.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(SRC))
det = cv2.QRCodeDetector()
i = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    try:
        retval, points, _ = det.detectAndDecode(frame)
    except Exception:  # noqa: BLE001
        retval = "?"
    if retval:
        print(f"frame {i}: {retval!r} points={None if points is None else points.reshape(-1, 2).astype(int).tolist()}")
        cv2.imwrite(str(OUT / f"gm2_f{i}.png"), frame)
    i += 1
cap.release()
print("total", i)
