"""Captions and transcript of the explainer video, from the video's own files.

The page's explainer is the chain cut, v5 (see make_explainer.sh). This
script writes:

- static/videos/explainer.en.vtt: one cue per sentence or clause, timed
  on the narration's own word times;
- the transcript in index.html, between the markers
  <!-- transcript:start --> and <!-- transcript:end -->: one paragraph
  per beat, a line on what the screen shows (SHOWS below, written by
  hand from video/HANDOFF.md) and then the narration.

Nothing is transcribed by ear. The words come from
video/narration/script_chain.md, the word times from the paced Kokoro
take, and each beat's start from the lengths of the rendered clips, the
way video/build/assemble.py lays the narration.

Inputs (the last three are gitignored, in the main checkout's video/):

    video/narration/script_chain.md
    video/out/kokoro_paced_chain/words.json     word times per beat
    video/out/kokoro_paced_chain/timing.json    beat lengths, the gap
    video/out/work_chain_iso_gl_landscape_kokoro_paced/concat.txt

Run from the repository root:

    FFMPEG=video/.venv/bin/ffmpeg .venv/bin/python project-page/tools/make_captions.py

LEAD_IN (default 0.5 s) must match make_explainer.sh.
"""

from __future__ import annotations

import html
import json
import os
import re
import subprocess
import sys
from pathlib import Path

PAGE = Path(__file__).resolve().parents[1]
REPO = PAGE.parent
VIDEO = Path(os.environ.get("VIDEO_DIR", REPO / "video"))
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
LEAD_IN = float(os.environ.get("LEAD_IN", "0.5"))
FPS = 30

SCRIPT = VIDEO / "narration" / "script_chain.md"
VOICE = VIDEO / "out" / "kokoro_paced_chain"
CONCAT = VIDEO / "out" / "work_chain_iso_gl_landscape_kokoro_paced" / "concat.txt"
VTT = PAGE / "static" / "videos" / "explainer.en.vtt"
INDEX = PAGE / "index.html"

# The clip that holds two beats; every other clip holds the beat it is named after.
CLIP_BEATS = {"intro": ["i0_problem", "i1_compare"]}

MAX_CHARS = 80      # a cue is at most two lines of about 40 characters
TAIL = 0.25         # a cue stays this long after its last word
MIN_LEN = 1.0

# What the screen shows in each beat. Numbers as in video/HANDOFF.md, section 11.
SHOWS = {
    "i0_problem": "Footage of a drone light show, then its filter output beside it: white blobs on "
                  "black. A stopwatch shows the frame budget, 33 ms.",
    "i1_compare": "A card for the drone frames: the CPU's bar at 0.679 ms, far inside the budget, and "
                  "a question mark for the GPU. Then a card for an 81-megapixel frame: the CPU's "
                  "stopwatch runs past 33 ms to 225 ms.",
    "r00_zoom": "The picture freezes and zooms into one drone until its pixels show as squares.",
    "r01_cpu": "Pixels turn from red to painted one at a time, on a CPU board with one core lit. "
               "Beside it a GPU board with 24 SM tiles, all dark.",
    "r02_one_block": "One SM tile lights up and a whole ring of pixels is painted in one step. A race "
                     "bar: on one small blob the CPU wins, 4.3× (0.121 against 0.520 ms).",
    "r03_giant": "A giant blob. The block's queue outgrows its 48 KB of shared memory and the rest "
                 "spills into global memory.",
    "r03b_one_sm": "One block is 1.1× faster than the CPU. One of the 24 SM tiles is lit.",
    "r04_two_blocks": "Two blocks, blue and green, on two SMs. The queue moves into global memory. "
                      "Twice as fast as one block, and 22 SM tiles stay dark.",
    "r05_all_blocks": "All 24 SMs light up with 48 blocks, one colour per block, and the colours mix "
                      "into speckle on the blob. 8.5× the CPU. A sweep of other launch shapes gains "
                      "15% at best.",
    "r05b_snake": "A thin winding snake. One block beats 48: a gate closes twice per ring, about 2 "
                  "microseconds each time.",
    "r06_diagonals": "The wave becomes a square and needs half the rings. The scoreboard's second "
                     "row: one huge blob, GPU 11×.",
    "r07_real_frame": "The 81-megapixel frame with its 2,522 blobs. One launch per blob, about 1,462 "
                      "ms, an estimate. The CPU, at 202 ms, wins by 7.2×.",
    "r08_tags": "A queue entry with one spare bit as a tag, and two blobs filling in two colours in "
                "one launch. About 850 ms: the CPU still wins, by 4.2×.",
    "r09_seeds": "Waves start all over the frame at once. 53.9 ms: 3.8× faster than the CPU, but over "
                 "the 33 ms budget.",
    "r10_tiles": "The picture is cut into tiles of 32 by 32 pixels. A close-up of one tile: 32 "
                 "threads, one row each, fill along the rows, the diagonals and the columns until "
                 "nothing changes.",
    "r11_rounds": "Border bits pass into the neighbouring tiles and the wave crosses the giant square "
                  "tile by tile. 4,001 rings become 127 rounds, and 101 ms becomes 3.12 ms. Row two: "
                  "GPU 340×.",
    "r12_pieces": "A 384-pixel crop of the frame: 135 pieces inside its tiles, 36 of them seeds, 8 "
                  "blobs. Coloured waves spread from piece to piece and meet.",
    "r13_frame": "The view pulls back to the whole frame: 11,609 seeds, 6 rounds, 2,522 blobs, 2.21 "
                 "ms. Row three: GPU 100×.",
    "r14_reveal": "The scoreboard. One small blob: 0.192 against 0.044 ms, GPU 4.4× in kernel time; "
                  "with the copy to the card the CPU wins, 3.5×. One huge blob: 1,065 against 3.12 "
                  "ms, 340×. Many blobs: 225 against 2.21 ms, 100×. The drones return, then the title "
                  "card: flood fill, from pixels to tiles.",
}


def beat_texts() -> list[tuple[str, str]]:
    text = SCRIPT.read_text(encoding="utf-8")
    return re.findall(r"^## beat (\S+)\n\n```text\n(.*?)\n```", text, re.S | re.M)


def clip_seconds(path: str) -> float:
    """A clip's length, snapped to whole frames."""
    err = subprocess.run([FFMPEG, "-hide_banner", "-i", path], capture_output=True, text=True).stderr
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", err)
    if not m:
        sys.exit(f"no duration for {path}")
    h, mnt, s = m.groups()
    return round((int(h) * 3600 + int(mnt) * 60 + float(s)) * FPS) / FPS


def beat_starts(lengths: dict[str, float], gap: float) -> dict[str, float]:
    """Each beat's start in the final cut, as assemble.py lays it."""
    starts, t0 = {}, 0.0
    for line in CONCAT.read_text(encoding="utf-8").splitlines():
        path = line.split("'")[1]
        stem = Path(path).parents[1].name
        off = 0.0
        for beat in CLIP_BEATS.get(stem, [stem]):
            starts[beat] = t0 + off
            off += lengths[beat] + gap
        t0 += clip_seconds(path)
    return starts


def split_cue(tokens: list[str]) -> list[list[str]]:
    """Sentences first, then clauses, until every cue fits MAX_CHARS."""
    def fits(ts):
        return len(" ".join(ts)) <= MAX_CHARS

    def at(ts, marks):
        return [i + 1 for i, t in enumerate(ts[:-1]) if t[-1] in marks]

    def cut(ts):
        if fits(ts):
            return [ts]
        for marks in (":;", ","):
            points = at(ts, marks)
            if points:
                mid = len(" ".join(ts)) / 2
                p = min(points, key=lambda i: abs(len(" ".join(ts[:i])) - mid))
                return cut(ts[:p]) + cut(ts[p:])
        p = len(ts) // 2
        return cut(ts[:p]) + cut(ts[p:])

    cues, sentence = [], []
    for t in tokens:
        sentence.append(t)
        if t[-1] in ".?!":
            cues += cut(sentence)
            sentence = []
    if sentence:
        cues += cut(sentence)
    return cues


def stamp(t: float) -> str:
    ms = round(t * 1000)
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d}.{ms % 1000:03d}"


def main() -> None:
    words = json.loads((VOICE / "words.json").read_text())["beats"]
    timing = json.loads((VOICE / "timing.json").read_text())
    lengths = {b["beat"]: float(b["seconds"]) for b in timing["beats"]}
    beats = beat_texts()
    starts = beat_starts(lengths, float(timing["gap"]))

    cues = []
    for name, text in beats:
        tokens = text.split()
        if len(tokens) != len(words[name]):
            sys.exit(f"{name}: {len(tokens)} script words, {len(words[name])} timed words")
        i = 0
        for part in split_cue(tokens):
            first, last = words[name][i], words[name][i + len(part) - 1]
            t0 = LEAD_IN + starts[name] + float(first[1])
            t1 = LEAD_IN + starts[name] + float(last[2]) + TAIL
            cues.append([t0, max(t1, t0 + MIN_LEN), " ".join(part)])
            i += len(part)
    for a, b in zip(cues, cues[1:]):
        a[1] = min(a[1], b[0] - 0.02)

    lines = ["WEBVTT", ""]
    for n, (t0, t1, text) in enumerate(cues, 1):
        lines += [str(n), f"{stamp(t0)} --> {stamp(t1)}", text, ""]
    VTT.write_text("\n".join(lines), encoding="utf-8")
    print(f"{VTT.relative_to(REPO)}: {len(cues)} cues, {stamp(cues[0][0])} to {stamp(cues[-1][1])}")

    paragraphs = []
    for name, text in beats:
        if name not in SHOWS:
            sys.exit(f"no SHOWS line for beat {name}")
        paragraphs.append(f"          <p><em>{html.escape(SHOWS[name], quote=False)}</em> "
                          f"{html.escape(text, quote=False)}</p>")
    page = INDEX.read_text(encoding="utf-8")
    pattern = re.compile(r"(<!-- transcript:start -->\n)(.*?)(<!-- transcript:end -->)", re.S)
    if not pattern.search(page):
        sys.exit("index.html has no transcript markers")
    block = "\n".join(paragraphs) + "\n"
    INDEX.write_text(pattern.sub(lambda m: m.group(1) + block + m.group(3), page), encoding="utf-8")
    print(f"index.html: transcript of {len(paragraphs)} beats")


if __name__ == "__main__":
    main()
