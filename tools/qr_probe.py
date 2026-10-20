from pathlib import Path

import cv2

SRC = Path(r"C:\Users\nenas\Desktop\Stickers\BetterCallSaul112.webm\BetterCallSaul112_AgADCUYAAo5O-Eo.webm")
OUT = Path(r"C:\Users\nenas\.local\share\opencode")
OUT.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(SRC))
ok, frame = cap.read()
cap.release()
det = cv2.QRCodeDetector()
retval, points, _ = det.detectAndDecode(frame)
print("decoded:", retval)
pts = points.reshape(-1, 2).astype(int)
print("points:", pts.tolist())
x0, y0 = pts.min(axis=0)
x1, y1 = pts.max(axis=0)
print(f"bbox: x={x0} y={y0} w={x1 - x0} h={y1 - y0}")
cv2.imwrite(str(OUT / "qr_frame.png"), frame)
cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 2)
cv2.imwrite(str(OUT / "qr_frame_bbox.png"), frame)
