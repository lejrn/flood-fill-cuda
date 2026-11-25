"""Compare every GL clip of the chain cut with its Cairo clip.

Usage (from video/):
    uv run build/gl_parity.py --layout landscape [--only r06_diagonals ...] [--min-ssim 0.98]

For each clip: the newest Cairo render (media/chain_<look>/<layout>) and
the newest GL render (media/chain_<look>_gl/<layout>, from
`assemble.py --renderer gl`). Checks that both have the same frame count
and the same snap time (<clip>.snap.json), and measures SSIM on every
frame pair with ffmpeg's ssim filter (luma and chroma, "All"). Both clips
are lossy (x264 CRF 23 and NVENC CQ 23), so a perfect match reads about
0.99, not 1. The worst frame pair goes to
media/review/gl_parity_<look>_<layout>/<clip>.png (Cairo | GL | 8x diff) for
a look. Writes the table to the same folder as parity.json. Exit 1 when a
frame count or snap time differs, or a frame falls under --min-ssim.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import av
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import assemble  # noqa: E402

VIDEO = assemble.VIDEO
FFMPEG = Path(os.environ.get("GL_FFMPEG", VIDEO / ".tools" / "ffmpeg-n8.1" / "bin" / "ffmpeg"))


def newest(layout: str, stem: str, cls: str, look: str, renderer: str) -> Path:
    """The newest full-quality clip (Manim names the folder by height: 1080p30, or 1920p30 when vertical)."""
    os.environ["VIDEO_RENDERER"] = renderer
    try:
        return assemble.newest_render(layout, stem, cls, "chain", look,
                                      quality="1920p30" if layout == "vertical" else "1080p30")
    finally:
        os.environ.pop("VIDEO_RENDERER", None)


def frame_count(path: Path) -> int:
    with av.open(str(path)) as c:
        return sum(1 for _ in c.decode(video=0))


def snap(path: Path) -> float | None:
    p = Path(str(path) + ".snap.json")
    return json.loads(p.read_text())["snap_s"] if p.exists() else None


def ssim_per_frame(a: Path, b: Path) -> np.ndarray:
    with tempfile.NamedTemporaryFile("r", suffix=".log") as log:
        subprocess.run([str(FFMPEG), "-hide_banner", "-loglevel", "error", "-i", str(a), "-i", str(b),
                        "-lavfi", f"[0:v][1:v]ssim=stats_file={log.name}", "-f", "null", "-"], check=True)
        vals = [float(m.group(1)) for m in re.finditer(r"All:([0-9.]+)", Path(log.name).read_text())]
    return np.array(vals)


def frame_at(path: Path, n: int) -> np.ndarray:
    with av.open(str(path)) as c:
        for i, f in enumerate(c.decode(video=0)):
            if i == n:
                return f.to_ndarray(format="rgb24")
    raise IndexError(f"{path}: no frame {n}")


def sheet(a: np.ndarray, b: np.ndarray, out: Path) -> None:
    d = np.clip(np.abs(a.astype(int) - b.astype(int)) * 8, 0, 255).astype(np.uint8)
    Image.fromarray(np.concatenate([a, b, d], axis=1)).save(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="landscape", choices=["landscape", "vertical"])
    ap.add_argument("--look", default="iso")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--min-ssim", type=float, default=0.98)
    args = ap.parse_args()

    review = VIDEO / "media" / "review" / f"gl_parity_{args.look}_{args.layout}"
    review.mkdir(parents=True, exist_ok=True)
    rows, bad = [], []
    for stem, cls, _ in assemble.clip_list("chain"):
        name = Path(stem).name
        if args.only and name not in args.only:
            continue
        cairo = newest(args.layout, stem, cls, args.look, "cairo")
        glc = newest(args.layout, stem, cls, args.look, "gl")
        na, nb = frame_count(cairo), frame_count(glc)
        sa, sb = snap(cairo), snap(glc)
        s = ssim_per_frame(cairo, glc)
        worst = int(np.argmin(s))
        sheet(frame_at(cairo, worst), frame_at(glc, worst), review / f"{name}.png")
        row = {"clip": name, "frames_cairo": na, "frames_gl": nb, "snap_cairo": sa, "snap_gl": sb,
               "ssim_mean": round(float(s.mean()), 5), "ssim_min": round(float(s.min()), 5), "worst_frame": worst,
               "frames_under": int((s < args.min_ssim).sum())}
        problems = []
        if na != nb:
            problems.append(f"frames {na} vs {nb}")
        if (sa is None) != (sb is None) or (sa is not None and abs(sa - sb) > 1e-6):
            problems.append(f"snap {sa} vs {sb}")
        if row["frames_under"]:
            problems.append(f"{row['frames_under']} frames under SSIM {args.min_ssim}")
        row["problems"] = problems
        rows.append(row)
        if problems:
            bad.append(name)
        print(f"{name:16s} frames {na:5d}/{nb:5d}  snap {sa}/{sb}  SSIM mean {row['ssim_mean']:.4f} "
              f"min {row['ssim_min']:.4f} @ {worst:5d}  {'; '.join(problems) or 'ok'}", flush=True)
    (review / "parity.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"{len(rows) - len(bad)}/{len(rows)} clips ok; sheets in {review.relative_to(VIDEO)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
