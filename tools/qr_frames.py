from pathlib import Path

import cv2

SRC = Path(r"C:\Users\nenas\Desktop\Stickers\BetterCallSaul112.webm\BetterCallSaul112_AgADCUYAAo5O-Eo.webm")
OUT = Path(r"C:\Users\nenas\.local\share\opencode")
OUT.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(SRC))
det = cv2.QRCodeDetector()
i = 0
last_qr = None
while True:
    ok, frame = cap.read()
    if not ok:
        break
    try:
        retval, _, _ = det.detectAndDecode(frame)
    except Exception:  # noqa: BLE001
        retval = None
    if retval:
        last_qr = i
        print(f"frame {i}: QR present")
    i += 1
cap.release()
print(f"total frames: {i}, last frame with QR: {last_qr}")
