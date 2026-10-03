#!/usr/bin/env bash
# Build the chapter 6 "runs" explainer: the 45 s video, its poster, and the
# 9 s square carousel loop with its poster.
#
# Run from the repo root (the directory that holds src/ and project-page/):
#
#   FFMPEG=/path/to/ffmpeg PYTHON=/path/to/python \
#     bash project-page/tools/make_runs_explainer.sh [video|carousel]
#
# FFMPEG    ffmpeg with libx264 (default: ffmpeg on PATH)
# PYTHON    python with numpy and Pillow (Pillow built with WebP support)
#           (default: VPYTHON if set, else python)
# FONT, FONT_BOLD, FONT_MONO
#           optional TTF files (default: DejaVu Sans, Bold and Mono Bold)
#
# Outputs (all under project-page/static/):
#   videos/runs_explainer.mp4            1280x720, 30 fps, H.264 CRF 23, no audio
#   images/runs_explainer_poster.webp    lossless, the run table filled in
#   videos/carousel/runs.mp4             720x720 loop, 9 s, H.264 CRF 22
#   images/carousel/runs.webp            lossless poster of the loop
#
# The scene is drawn with Pillow and numpy, not Manim. There are no per-cell
# objects and no Manim render, so the job needs well under 1 GB of RAM and no
# scratch folder: frames go to ffmpeg as raw RGB on a pipe, the same pattern as
# make_race.py. Both clips are tagged BT.709 (limited range) and use +faststart.
#
# The blob comes from project-page/tools/runs_blob.json. Before a frame is
# drawn the script recomputes the runs, the row counts, the 8-connectivity
# partners and both merge rounds from the mask and asserts them against that
# file, and it asserts the 58.51 ms, 2.96 ms and 19.8x figures against
# src/flood_fill_cuda/results/ch06_gpu_nblob_runs/benchmark_results/
# runs_20260725T161448Z.json.
#
# For a single review frame, without encoding (the PNG must go outside
# project-page, the repo gitignores .png there):
#   "$PYTHON" project-page/tools/runs_explainer.py --still 33.7 --out /some/scratch/frame.png
# (--still T is video seconds; add --design to give design seconds instead.)

set -euo pipefail

FFMPEG=${FFMPEG:-ffmpeg}
PYTHON=${PYTHON:-${VPYTHON:-python}}
ONLY=${1:-}

if [[ ! -f project-page/tools/runs_explainer.py || ! -d src/flood_fill_cuda/results ]]; then
    echo "make_runs_explainer.sh: run me from the repo root (no project-page/tools or src/ here)" >&2
    exit 1
fi
if [[ -n $ONLY && $ONLY != video && $ONLY != carousel ]]; then
    echo "make_runs_explainer.sh: the optional argument is 'video' or 'carousel'" >&2
    exit 1
fi

export FFMPEG
[[ -n ${FONT:-} ]] && export FONT
[[ -n ${FONT_BOLD:-} ]] && export FONT_BOLD
[[ -n ${FONT_MONO:-} ]] && export FONT_MONO

if [[ -n $ONLY ]]; then
    "$PYTHON" project-page/tools/runs_explainer.py --only "$ONLY"
else
    "$PYTHON" project-page/tools/runs_explainer.py
fi

echo
echo "done:"
for f in project-page/static/videos/runs_explainer.mp4 \
         project-page/static/images/runs_explainer_poster.webp \
         project-page/static/videos/carousel/runs.mp4 \
         project-page/static/images/carousel/runs.webp; do
    if [[ -f $f ]]; then
        printf '  %9d  %s\n' "$(stat -c %s "$f")" "$f"
    fi
done
