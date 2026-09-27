"""Verify the cuts between consecutive clips are invisible.

For each consecutive pair of rendered clips, decode the last frame of clip
k and the first frame of clip k+1 (never whole clips: 1080p frames are
big) and compare. The last frame of a clip is a P-frame deep in a GOP and
the first frame of the next is a fresh I-frame, so libx264 leaves about
one level of noise over text and noisy pictures even when the pictures
are identical. The verdict therefore uses 4x4 box-averaged frames, where
encoder noise averages out and a moved or missing element does not: mean
below 1.5 (preview-quality speckle images reach 0.95) and fewer than
0.05% of averaged pixels above 16 levels.
`--intra` also checks the last frames of every clip for a jump (a bad
`snap()`); the outro fades out, so its jump is expected.

Usage (from video/):
    uv run build/seam_check.py [--layout landscape] [--quality 1080p30] [--intra]
Writes media/review/seams/<k>_<k+1>.png (A | B | 8x diff) for every pair.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import av
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from assemble import BEATS, newest_render, duration  # noqa: E402

VIDEO = Path(__file__).resolve().parents[1]
MEAN_MAX, FRAC_MAX, LEVELS = 1.5, 0.0005, 16
BOX = 4


def frames_at_edges(path: Path, n_tail: int = 1) -> tuple:
    """(first frame, last n_tail frames) as uint8 arrays, decoding once."""
    first, tail = None, []
    with av.open(str(path)) as c:
        for fr in c.decode(video=0):
            arr = fr.to_ndarray(format="rgb24")
            if first is None:
                first = arr
            tail.append(arr)
            if len(tail) > n_tail:
                tail.pop(0)
    return first, tail


def box_avg(a: np.ndarray) -> np.ndarray:
    h, w = (a.shape[0] // BOX) * BOX, (a.shape[1] // BOX) * BOX
    return a[:h, :w].astype(np.float32).reshape(h // BOX, BOX, w // BOX, BOX, 3).mean(axis=(1, 3))


def diff_stats(a: np.ndarray, b: np.ndarray) -> tuple:
    """(raw mean, box-averaged mean, fraction of averaged pixels > LEVELS, raw diff)."""
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    db = np.abs(box_avg(a) - box_avg(b))
    return float(d.mean()), float(db.mean()), float((db.max(axis=2) > LEVELS).mean()), d


def save_sheet(a: np.ndarray, b: np.ndarray, d: np.ndarray, out: Path) -> None:
    amp = np.clip(d.astype(np.int32) * 8, 0, 255).astype(np.uint8)
    sheet = np.concatenate([a, b, amp], axis=1)
    Image.fromarray(sheet).resize((sheet.shape[1] // 2, sheet.shape[0] // 2)).save(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="landscape")
    ap.add_argument("--quality", default=None)
    ap.add_argument("--intra", action="store_true")
    args = ap.parse_args()

    out = VIDEO / "media" / "review" / "seams"
    out.mkdir(parents=True, exist_ok=True)
    clips = []
    for stem, cls, beat in BEATS:
        folder = VIDEO / "media" / args.layout / "videos" / stem
        pattern = f"{args.quality}/{cls}.mp4" if args.quality else f"*/{cls}.mp4"
        hits = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime)
        if not hits:
            raise SystemExit(f"no render for {stem}/{cls}")
        clips.append((beat, hits[-1]))

    n_tail = 12 if args.intra else 1
    edges = [frames_at_edges(p, n_tail) for _, p in clips]
    ok = True
    for i in range(len(clips) - 1):
        a = edges[i][1][-1]
        b = edges[i + 1][0]
        raw, mean, frac, d = diff_stats(a, b)
        verdict = "OK" if (mean < MEAN_MAX and frac < FRAC_MAX) else "SEAM"
        ok &= verdict == "OK"
        save_sheet(a, b, d, out / f"{i}_{i + 1}.png")
        print(f"{clips[i][0]:14s} -> {clips[i + 1][0]:14s} raw {raw:6.3f}  box {mean:6.3f}  "
              f">{LEVELS}: {frac * 100:6.3f}%  {verdict}")
    if args.intra:
        for (beat, _), (_, tail) in zip(clips[:-1], edges[:-1]):
            jumps = [diff_stats(tail[j], tail[j + 1])[1] for j in range(len(tail) - 1)]
            worst = max(jumps) if jumps else 0.0
            flag = "" if worst < MEAN_MAX else "  JUMP"
            ok &= worst < MEAN_MAX
            print(f"{beat:14s} largest jump in the last {len(tail)} frames: {worst:6.3f}{flag}")
    total = sum(duration(p) for _, p in clips)
    print(f"total {total:.2f} s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
