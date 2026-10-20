from pathlib import Path

import cv2

ROOT = Path(r"C:\Users\nenas\Desktop\Stickers")
TARGETS = [
    "BetterCallSaul112.webm/BetterCallSaul112_AgADCUYAAo5O-Eo.webm",
    "Brat2FreeTopsticker.webm/Brat2FreeTopsticker_AgADo0YAAn-t-Uo.webm",
    "BratFreeTopsticker.webm/BratFreeTopsticker_AgAD3E4AAo3--Uo.webm",
    "DuneFreetopsticker.webm/DuneFreetopsticker_AgADqE4AAr9o-Eo.webm",
    "Fightclub12.webm/Fightclub12_AgAD-lEAAonr8Eo.webm",
    "InterstellarFreeTopsticker.webm/InterstellarFreeTopsticker_AgADokcAAnmc-Uo.webm",
    "JasePinkmanFreetopsticker.webm/JasePinkmanFreetopsticker_AgADHVUAApz58Uo.webm",
    "Sumerki11.webm/Sumerki11_AgADhUcAAgab-Eo.webm",
    "TheBatmanDarkKnight.webm/TheBatmanDarkKnight_AgADHkkAAiYwCEo.webm",
    "TheGreenMileFreetopsticker.webm/TheGreenMileFreetopsticker_AgADs0gAAicX-Eo.webm",
    "TheGreenMiles2Freetopsticker.webm/TheGreenMiles2Freetopsticker_AgADCjcAAl5GuUo.webm",
    "TmeinHollywoodFreetopsticker.webm/TmeinHollywoodFreetopsticker_AgADmUsAAiVg-Eo.webm",
]
OUT = Path(r"C:\Users\nenas\.local\share\opencode\qr_all")
OUT.mkdir(parents=True, exist_ok=True)

det = cv2.QRCodeDetector()
for rel in TARGETS:
    src = ROOT / rel
    cap = cv2.VideoCapture(str(src))
    total = 0
    qr_frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        try:
            retval, _, _ = det.detectAndDecode(frame)
        except Exception:  # noqa: BLE001
            retval = "?"
        if retval:
            qr_frames.append(total)
        total += 1
        if ok and total == 1:
            safe = rel.replace("/", "_").replace("\\", "_").replace(".webm", "")
            cv2.imwrite(str(OUT / f"{safe}_f0.png"), frame)
    cap.release()
    pct = 100.0 * len(qr_frames) / total if total else 0
    print(f"{rel}: {total} frames, QR on {len(qr_frames)} ({pct:.0f}%)")
