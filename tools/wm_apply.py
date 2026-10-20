import json
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(r"C:\Users\nenas\Desktop\Stickers")
DST = Path(r"C:\Users\nenas\Desktop\Stickers_clean")
JSON_PATH = Path(r"C:\Users\nenas\.local\share\opencode\wm\consensus_boxes.json")
FFMPEG = Path(r"C:\Users\nenas\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin\ffmpeg.exe")


def clamp_box(b: list[int], w: int = 512, h: int = 512) -> tuple[int, int, int, int] | None:
    x, y, bw, bh = b
    pad = 3
    x0 = max(1, x - pad)
    y0 = max(1, y - pad)
    x1 = min(w - 1, x + bw + pad)
    y1 = min(h - 1, y + bh + pad)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None
    return x0, y0, x1 - x0, y1 - y0


def process(args) -> tuple[str, str]:
    pack, fname, boxes = args
    src = ROOT / pack / fname
    dst = DST / pack / fname
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not boxes:
        import shutil

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return (str(dst), "copy")
    chain = []
    for b in boxes:
        c = clamp_box(b)
        if c:
            x, y, bw, bh = c
            chain.append(f"delogo=x={x}:y={y}:w={bw}:h={bh}")
    if not chain:
        return (str(dst), "copy")
    vf = "scale=512:512," + ",".join(chain)
    cmd = [
        str(FFMPEG), "-y", "-loglevel", "error", "-i", str(src),
        "-vf", vf, "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p",
        "-crf", "18", "-b:v", "0", "-deadline", "realtime", "-cpu-used", "5", "-row-mt", "1",
        "-an", str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return (str(dst), "ok" if r.returncode == 0 else r.stderr.strip()[:200])


def main():
    only = sys.argv[1:] or None
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    per_file = data if "per_file" not in data else data["per_file"]
    jobs = []
    for pack in sorted(per_file):
        if only and pack not in only:
            continue
        for fname, boxes in per_file[pack].items():
            jobs.append((pack, fname, boxes))
    print(f"{len(jobs)} files", flush=True)
    with Pool(processes=12) as pool:
        results = pool.map(process, jobs, chunksize=4)
    ok = sum(1 for _, s in results if s in ("ok", "copy"))
    errs = [r for r in results if r[1] not in ("ok", "copy")]
    print(f"done: {ok} ok, {len(errs)} errors")
    for path, err in errs[:20]:
        print("ERR", path, err)


if __name__ == "__main__":
    main()
