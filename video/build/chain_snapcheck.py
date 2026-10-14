"""Check one chain clip: does its last animated frame equal the next state?

Usage (from video/):
    uv run build/chain_snapcheck.py r03_giant Giant [--look iso] [--layout vertical] [--quality 854p15]

Reads <clip>.snap.json (written by ChainScene) and compares the frames on
both sides of the snap, with seam_check's measure (4x4 box-averaged mean,
fail at 1.5). Exit 1 on a jump. At preview quality a value near 1 can be
encoder noise (the frame after the snap starts a new partial movie, an
I-frame); a missing element shows as a much bigger jump.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from assemble import newest_render  # noqa: E402
from seam_check import MEAN_MAX, diff_stats, frames_around  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("cls")
    ap.add_argument("--look", default="iso")
    ap.add_argument("--layout", default="vertical")
    ap.add_argument("--quality", default=None)
    a = ap.parse_args()
    clip = newest_render(a.layout, "chain/" + a.stem, a.cls, "chain", a.look, a.quality)
    side = Path(str(clip) + ".snap.json")
    if not side.exists():
        print(f"{clip.name}: no snap.json (the last beat fades out instead)")
        return 0
    before, after = frames_around(clip, json.loads(side.read_text())["snap_s"])
    raw, mean, frac, _ = diff_stats(before, after)
    ok = mean < MEAN_MAX
    print(f"{a.stem} {a.layout}: jump across the snap {mean:.3f} (raw {raw:.3f}) {'OK' if ok else 'JUMP'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
