"""Extract review frames from a rendered clip and tile them into one sheet.

Usage (from video/):
    uv run build/review.py s03_n_blocks NBlocks --times 0,0.3,0.6,1.0,1.5,2.6,5,end
    uv run build/review.py s03_n_blocks NBlocks --every 1.0 [--layout landscape] [--quality 480p15]

Frames land in media/review/<stem>/t_<sec>.png and a contact sheet in
media/review/<stem>/sheet.png (frames scaled to 640 px wide, 3 per row).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import av
from PIL import Image, ImageDraw

VIDEO = Path(__file__).resolve().parents[1]
FFMPEG = VIDEO / ".venv" / "bin" / "ffmpeg"


def newest_render(layout: str, stem: str, cls: str, quality: str | None) -> Path:
    folder = VIDEO / "media" / layout / "videos" / stem
    pattern = f"{quality}/{cls}.mp4" if quality else f"*/{cls}.mp4"
    hits = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not hits:
        raise SystemExit(f"no render for {stem}/{cls} under {folder}")
    return hits[-1]


def duration(path: Path) -> float:
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        return float(s.duration * s.time_base) if s.duration else c.duration / 1e6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("cls")
    ap.add_argument("--times", default=None, help="comma list of seconds; 'end' = last frame")
    ap.add_argument("--every", type=float, default=None)
    ap.add_argument("--layout", default="landscape")
    ap.add_argument("--quality", default=None, help="e.g. 480p15 or 1080p30; default newest")
    ap.add_argument("--cols", type=int, default=3)
    args = ap.parse_args()

    clip = newest_render(args.layout, args.stem, args.cls, args.quality)
    dur = duration(clip)
    if args.times:
        times = [dur - 0.02 if t.strip() == "end" else float(t) for t in args.times.split(",")]
    else:
        step = args.every or 1.0
        times, t = [], 0.0
        while t < dur:
            times.append(t)
            t += step
        times.append(dur - 0.02)
    out = VIDEO / "media" / "review" / args.stem
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    paths = []
    for t in times:
        t = max(0.0, min(t, dur - 0.02))
        p = out / f"t_{t:06.2f}.png"
        subprocess.run([str(FFMPEG), "-y", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", str(clip),
                        "-frames:v", "1", str(p)], check=True)
        if not p.exists():   # past the last decodable timestamp: take the true last frame
            subprocess.run([str(FFMPEG), "-y", "-loglevel", "error", "-sseof", "-0.2", "-i", str(clip),
                            "-update", "1", "-frames:v", "5", str(p)], check=True)
        paths.append((t, p))
    w = 640
    tiles = []
    for t, p in paths:
        im = Image.open(p).convert("RGB")
        im = im.resize((w, round(im.height * w / im.width)))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 90, 18], fill=(0, 0, 0))
        d.text((4, 3), f"t={t:.2f}s", fill=(255, 255, 255))
        tiles.append(im)
    cols = args.cols
    rows = (len(tiles) + cols - 1) // cols
    th = tiles[0].height
    sheet = Image.new("RGB", (cols * w + (cols - 1) * 6, rows * th + (rows - 1) * 6), (40, 40, 40))
    for i, im in enumerate(tiles):
        sheet.paste(im, ((i % cols) * (w + 6), (i // cols) * (th + 6)))
    sheet_path = out / "sheet.png"
    sheet.save(sheet_path)
    print(f"{clip.name}: {dur:.2f} s, {len(paths)} frames -> {sheet_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
