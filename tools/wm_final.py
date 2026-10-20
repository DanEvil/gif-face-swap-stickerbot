import json
from pathlib import Path

WM_PATH = Path(r"C:\Users\nenas\.local\share\opencode\wm\file_boxes.json")
OUT_PATH = Path(r"C:\Users\nenas\.local\share\opencode\wm\final_boxes.json")

PAD = 4
SUPPORT = 0.15
MIN_SUPPORT_N = 3


def overlaps(a: list[int], b: list[int], pad: int = 0) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw + pad <= bx or bx + bw + pad <= ax or ay + ah + pad <= by or by + bh + pad <= ay)


def cluster_boxes(boxes: list[list[int]]) -> list[tuple[list[int], int]]:
    cl = [(b[:], 1) for b in boxes]
    changed = True
    while changed:
        changed = False
        out = []
        skip = [False] * len(cl)
        for i, (box, n) in enumerate(cl):
            if skip[i]:
                continue
            for j in range(i + 1, len(cl)):
                if skip[j]:
                    continue
                ob, on = cl[j]
                if overlaps(box, ob, PAD):
                    x0, y0 = min(box[0], ob[0]), min(box[1], ob[1])
                    x1 = max(box[0] + box[2], ob[0] + ob[2])
                    y1 = max(box[1] + box[3], ob[1] + ob[3])
                    box = [x0, y0, x1 - x0, y1 - y0]
                    n += on
                    skip[j] = True
                    skip[i] = True
                    changed = True
            out.append((box, n))
        cl = out
    return cl


def main():
    data = json.loads(WM_PATH.read_text(encoding="utf-8"))
    per_file = data["per_file"]
    final: dict[str, dict[str, list]] = {}
    stats = []
    for pack, files in per_file.items():
        all_boxes = [b for boxes in files.values() for b in boxes]
        cl = cluster_boxes(all_boxes)
        need = max(MIN_SUPPORT_N, round(SUPPORT * len(files)))
        consensus = [box for box, n in cl if n >= need]
        pf = {}
        n_boost = 0
        for fname, boxes in files.items():
            fb = list(boxes)
            for c in consensus:
                if not any(overlaps(b, c, 4) for b in fb):
                    fb.append(list(c))
                    n_boost += 1
            pf[fname] = fb
        final[pack] = pf
        if consensus:
            stats.append(f"{pack}: {consensus} (+{n_boost})")
        else:
            stats.append(f"{pack}: none")
    OUT_PATH.write_text(json.dumps(final, indent=1), encoding="utf-8")
    print("\n".join(stats))
    print("saved", OUT_PATH)


if __name__ == "__main__":
    main()
