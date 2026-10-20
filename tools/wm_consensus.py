import json
from pathlib import Path

WM_PATH = Path(r"C:\Users\nenas\.local\share\opencode\wm\file_boxes.json")
OUT_PATH = Path(r"C:\Users\nenas\.local\share\opencode\wm\consensus_boxes.json")

MANUAL = {
    "Brat2FreeTopsticker.webm": [[0, 478, 220, 34]],
    "BratFreeTopsticker.webm": [[0, 478, 220, 34]],
    "DuneFreetopsticker.webm": [[0, 478, 200, 34]],
    "TheBatmanDarkKnight.webm": [[0, 478, 220, 34]],
    "Sumerki11.webm": [[0, 478, 230, 34]],
    "TmeinHollywoodFreetopsticker.webm": [[0, 472, 512, 40]],
    "TheGreenMileFreetopsticker.webm": [[0, 478, 200, 34]],
    "TheGreenMiles2Freetopsticker.webm": [[0, 478, 220, 34]],
    "JasePinkmanFreetopsticker.webm": [[0, 478, 220, 34], [330, 480, 180, 32]],
    "Fightclub12.webm": [[360, 494, 124, 18]],
    "Boys1123.webm": [[0, 5, 175, 40], [420, 0, 92, 30]],
    "BesstiJiE11.webm": [[306, 452, 200, 48]],
    "SexEducation112.webm": [[0, 470, 170, 42]],
    "MrBean112.webm": [[0, 468, 512, 44]],
    "IRobotStickersFTS.webm": [[0, 478, 190, 34]],
    "KAkjawstretilmamu.webm": [[414, 470, 98, 42], [180, 474, 200, 38]],
}


def overlaps(a: list[int], b: list[int], pad: int = 6) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw + pad <= bx or bx + bw + pad <= ax or ay + ah + pad <= by or by + bh + pad <= ay)


def consensus_for_band(boxes: list[list[int]], n_files: int) -> list[int] | None:
    groups: dict[tuple[int, int], list[list[int]]] = {}
    for b in boxes:
        key = (b[2] // 24, b[3] // 10)
        groups.setdefault(key, []).append(b)
    need = max(3, round(0.12 * n_files))
    best = None
    for members in groups.values():
        if len(members) >= need and (best is None or len(members) > len(best)):
            best = members
    if best is None:
        return None
    xs = sorted(b[0] for b in best)
    ys = sorted(b[1] for b in best)
    ws = sorted(b[2] for b in best)
    hs = sorted(b[3] for b in best)
    mid = len(best) // 2
    return [xs[mid], ys[mid], ws[mid], hs[mid]]


def main():
    data = json.loads(WM_PATH.read_text(encoding="utf-8"))
    per_file = data["per_file"]
    final: dict[str, dict[str, list]] = {}
    stats = []
    for pack, files in per_file.items():
        bottom = [b for boxes in files.values() for b in boxes if b[1] + b[3] >= 455]
        top = [b for boxes in files.values() for b in boxes if b[1] <= 55]
        cons = []
        for band in (bottom, top):
            c = consensus_for_band(band, len(files))
            if c:
                cons.append(c)
        for m in MANUAL.get(pack, []):
            pass
        pf = {}
        n_boost = 0
        for fname, boxes in files.items():
            fb = list(boxes)
            for m in MANUAL.get(pack, []):
                fb = [b for b in fb if not overlaps(b, m, 4)]
                fb.append(list(m))
            for c in cons:
                if not any(overlaps(b, c, 4) for b in fb):
                    fb.append(list(c))
                    n_boost += 1
            pf[fname] = fb
        final[pack] = pf
        stats.append(f"{pack}: {cons} (+{n_boost})" if cons else f"{pack}: none")
    OUT_PATH.write_text(json.dumps(final, indent=1), encoding="utf-8")
    print("\n".join(stats))
    print("saved", OUT_PATH)


if __name__ == "__main__":
    main()
