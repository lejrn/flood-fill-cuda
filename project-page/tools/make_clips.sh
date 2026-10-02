#!/usr/bin/env bash
# Build the project page teaser, the carousel clips, their posters and the
# ch06 before/after still.
#
# Run from the repo root (the directory that holds src/ and project-page/):
#
#   FFMPEG=/path/to/ffmpeg PYTHON=/path/to/python VIDEO_DIR=/path/to/video \
#     bash project-page/tools/make_clips.sh
#
# FFMPEG    ffmpeg with libx264 (default: ffmpeg on PATH)
# PYTHON    python with Pillow built with WebP support (default: python)
# VIDEO_DIR the video/ folder that holds the gitignored Manim renders
#           (default: video). Only the ch06 "runs" clip needs it.
#
# Outputs (all under project-page/static/):
#   videos/teaser.mp4, images/teaser_poster.webp
#   videos/carousel/<name>.mp4, images/carousel/<name>.webp
#   images/before_after.webp
#   videos/two_blobs_ab.mp4, videos/merge_ab.mp4 and their posters
#
# Notes on the settings:
# - The GIFs are pixel art on white. They are converted with an fps=30
#   filter, so the 60 ms frames and the final hold keep their timing.
# - Every scale of a GIF uses flags=neighbor and an integer factor. The
#   192 px scenes (stored at 3x = 576 px) are scaled 2x to 1152 px, so each
#   logical pixel is a 6x6 block that lines up with the 2x2 chroma grid of
#   yuv420p. At 576 px the 3x3 blocks straddle that grid and the colours bleed.
#   Three 512 px clips with pixel-level hue noise are also scaled 2x (see 2.).
# - RGB is converted to yuv420p with the BT.709 matrix (limited range) and
#   the stream is tagged BT.709, so browsers decode the colours as intended.
# - Runs.mp4 is an untagged Manim render encoded with the BT.601 matrix, so
#   it is decoded as BT.601 before it is re-encoded as tagged BT.709.
# - Posters are written by Pillow as lossless WebP. No .png or .jpg is ever
#   written into the repo (the repo gitignores them).

set -euo pipefail

FFMPEG=${FFMPEG:-ffmpeg}
PYTHON=${PYTHON:-python}
VIDEO_DIR=${VIDEO_DIR:-video}

RES=src/flood_fill_cuda/results
VID=project-page/static/videos
IMG=project-page/static/images
RUNS_SRC=$VIDEO_DIR/media/landscape/videos/s07_runs/1080p30/Runs.mp4

if [[ ! -d $RES || ! -d project-page ]]; then
    echo "make_clips.sh: run me from the repo root (no $RES or project-page here)" >&2
    exit 1
fi

mkdir -p "$VID/carousel" "$IMG/carousel"

# H.264 output shared by every clip: yuv420p, BT.709 tags, faststart, no audio.
X264=(-c:v libx264 -preset slow -pix_fmt yuv420p
      -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv
      -movflags +faststart -an)

# RGB -> yuv420p with the BT.709 matrix. flags=area averages each 2x2 chroma
# block exactly, so 2x pixel-art blocks keep their exact colour.
TO_YUV="scale=out_color_matrix=bt709:out_range=tv:flags=area+accurate_rnd,format=yuv420p"

ff() {
    "$FFMPEG" -hide_banner -loglevel error -nostdin -y "$@"
}

# gif_clip SRC OUT CRF [UPSCALE]
gif_clip() {
    local src=$1 out=$2 crf=$3 up=${4:-1} sc=""
    if (( up > 1 )); then
        sc=",scale=iw*$up:ih*$up:flags=neighbor"
    fi
    echo "clip   $out  <- $src (crf $crf, x$up)"
    ff -i "$src" -vf "fps=30$sc,$TO_YUV" "${X264[@]}" -crf "$crf" "$out"
}

# gif_poster SRC OUT [UPSCALE]: the GIF's last frame as lossless WebP.
gif_poster() {
    local src=$1 out=$2 up=${3:-1}
    echo "poster $out  <- last frame of $src (x$up)"
    "$PYTHON" - "$src" "$out" "$up" <<'PY'
import sys
from PIL import Image

src, out, up = sys.argv[1], sys.argv[2], int(sys.argv[3])
im = Image.open(src)
im.seek(getattr(im, "n_frames", 1) - 1)
rgb = im.convert("RGB")
if up > 1:
    rgb = rgb.resize((rgb.width * up, rgb.height * up), Image.NEAREST)
rgb.save(out, lossless=True, quality=100, method=6)
PY
}

# png_stdin_to_webp OUT: read one PNG frame from stdin (a pipe, never a file
# in the repo) and save it as lossless WebP.
# The code goes in -c, not a heredoc, because stdin carries the PNG.
png_stdin_to_webp() {
    "$PYTHON" -c '
import io
import sys
from PIL import Image

Image.open(io.BytesIO(sys.stdin.buffer.read())).convert("RGB").save(
    sys.argv[1], lossless=True, quality=100, method=6
)
' "$1"
}

# 1. Teaser: the ch05 wavefront over 2,522 blobs, 900x900 native.
TEASER_SRC=$RES/ch05_gpu_nblob_nblock/wavefront/input_blobs_final.gif
gif_clip "$TEASER_SRC" "$VID/teaser.mp4" 20
gif_poster "$TEASER_SRC" "$IMG/teaser_poster.webp"

# 2. Carousel clips from the chapter GIFs: name, source, integer upscale.
# two_blocks, n_blocks and conn8 change hue from one logical pixel to the
# next. At 512 px each logical pixel is a single chroma sample, and the
# bilinear chroma upsampling of the browser greys those hues out (mean error
# 7-18 per channel). At 2x (1024 px) the error halves and the clip looks
# like the GIF. The smooth-gradient clips stay at their native size.
CAROUSEL=(
    "cpu_walk   ch01_gpu_1blob_1block/wavefront/square256_cpu_order.gif          1"
    "one_block  ch01_gpu_1blob_1block/wavefront/square256_b1_t256.gif            1"
    "two_blocks ch02_gpu_1blob_2block/wavefront/global_square256.gif             2"
    "n_blocks   ch03_gpu_1blob_nblock/wavefront/square256_b8_t32.gif             2"
    "conn8      ch03_gpu_1blob_nblock/wavefront/square256_b8_t32_conn8.gif       2"
    "two_blobs  ch04_gpu_2blob_nblock/wavefront/asym384_b8_t32_multisource.gif   1"
    "merge_u    ch05_gpu_nblob_nblock/wavefront/u192_merge_prov.gif              2"
    "noise      ch05_gpu_nblob_nblock/wavefront/random192_ccl_final.gif          2"
)
for row in "${CAROUSEL[@]}"; do
    read -r name rel up <<<"$row"
    gif_clip "$RES/$rel" "$VID/carousel/$name.mp4" 18 "$up"
    gif_poster "$RES/$rel" "$IMG/carousel/$name.webp" "$up"
done

# 3. ch06 "runs": a square crop of the centre pane of the Manim scene.
# The box (600x600 at x=796, y=316 of 1920x1080) holds only the row of
# pixel cells, the teal run bars and the recoloured crop. The tables, the
# thumbnail labels and the caption with timings all sit outside it.
# Window 2.0-7.5 s: empty pane, cells wipe in (2.5 s), cells collapse into
# runs left to right (3.8-5.0 s), the recoloured crop appears (6.0 s) and holds.
# Poster: source frame 134 (4.467 s, clip frame 74), mid-collapse. The first
# two runs are already teal bars and the last run still shows crisp red
# pixel cells, so the still shows "cells become runs" on its own. For the
# finished state instead (run bars plus the recoloured crop), use frame 210.
RUNS_POSTER_FRAME=134
RUNS_CROP="crop=600:600:796:316"
RUNS_RGB="scale=in_color_matrix=bt601:in_range=tv:flags=lanczos+accurate_rnd+full_chroma_int,format=gbrp,scale=720:720:flags=lanczos+accurate_rnd"
if [[ -f $RUNS_SRC ]]; then
    echo "clip   $VID/carousel/runs.mp4  <- $RUNS_SRC 2.0-7.5 s (crf 22)"
    ff -ss 2.0 -t 5.5 -i "$RUNS_SRC" \
        -vf "$RUNS_CROP,$RUNS_RGB,$TO_YUV" -r 30 "${X264[@]}" -crf 22 \
        "$VID/carousel/runs.mp4"
    echo "poster $IMG/carousel/runs.webp  <- $RUNS_SRC frame $RUNS_POSTER_FRAME"
    "$FFMPEG" -hide_banner -loglevel error -nostdin -i "$RUNS_SRC" \
        -vf "select='eq(n\,$RUNS_POSTER_FRAME)',$RUNS_CROP,$RUNS_RGB,format=rgb24" \
        -fps_mode passthrough -frames:v 1 -f image2pipe -c:v png - \
        | png_stdin_to_webp "$IMG/carousel/runs.webp"
else
    echo "skip   runs: $RUNS_SRC not found (set VIDEO_DIR to the main checkout's video/)" >&2
fi

# 4. Still: the ch06 before/after figure, one 1536x380 frame, lossless.
gif_poster "$RES/ch06_gpu_nblob_runs/figures/before_after.gif" "$IMG/before_after.webp"

# 5. Side-by-side clips (Nerfies' "stacked" videos). Each pair was rendered
# by one generator on one clock with the same frame count and delays, so
# frame i of the left GIF and frame i of the right GIF line up.
# stack_clip LEFT RIGHT OUT UPSCALE GAP
stack_clip() {
    local left=$1 right=$2 out=$3 up=$4 gap=$5 sc=""
    if (( up > 1 )); then
        sc=",scale=iw*$up:ih*$up:flags=neighbor"
    fi
    echo "stack  $out  <- $left | $right (x$up, gap $gap)"
    ff -i "$left" -i "$right" -filter_complex \
        "[0:v]fps=30,format=rgb24$sc,pad=iw+$gap:ih:0:0:white[l];[1:v]fps=30,format=rgb24$sc[r];[l][r]hstack=inputs=2,$TO_YUV" \
        "${X264[@]}" -crf 18 "$out"
}

# stack_poster LEFT RIGHT OUT UPSCALE GAP: both last frames, side by side.
stack_poster() {
    echo "poster $3  <- last frames of $1 | $2"
    "$PYTHON" - "$@" <<'PY'
import sys
from PIL import Image

left, right, out, up, gap = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
frames = []
for src in (left, right):
    im = Image.open(src)
    im.seek(getattr(im, "n_frames", 1) - 1)
    rgb = im.convert("RGB")
    if up > 1:
        rgb = rgb.resize((rgb.width * up, rgb.height * up), Image.NEAREST)
    frames.append(rgb)
w = frames[0].width + gap + frames[1].width
h = max(f.height for f in frames)
sheet = Image.new("RGB", (w, h), (255, 255, 255))
sheet.paste(frames[0], (0, 0))
sheet.paste(frames[1], (frames[0].width + gap, 0))
sheet.save(out, lossless=True, quality=100, method=6)
PY
}

# ch04: two launches one after the other (left) vs one shared launch (right).
# The sequential GIF replays the multisource depth map on a sequential clock.
SEQ=$RES/ch04_gpu_2blob_nblock/wavefront/asym384_b8_t32_sequential.gif
MULTI=$RES/ch04_gpu_2blob_nblock/wavefront/asym384_b8_t32_multisource.gif
stack_clip "$SEQ" "$MULTI" "$VID/two_blobs_ab.mp4" 1 16
stack_poster "$SEQ" "$MULTI" "$IMG/two_blobs_ab.webp" 1 16

# ch05: provisional labels (left) vs final labels after the merge (right).
PROV=$RES/ch05_gpu_nblob_nblock/wavefront/u192_merge_prov.gif
FINAL=$RES/ch05_gpu_nblob_nblock/wavefront/u192_merge_final.gif
stack_clip "$PROV" "$FINAL" "$VID/merge_ab.mp4" 2 32
stack_poster "$PROV" "$FINAL" "$IMG/merge_ab.webp" 2 32

echo
echo "done:"
for f in "$VID/teaser.mp4" "$IMG/teaser_poster.webp" \
         "$VID"/carousel/*.mp4 "$IMG"/carousel/*.webp "$IMG/before_after.webp" \
         "$VID/two_blobs_ab.mp4" "$IMG/two_blobs_ab.webp" "$VID/merge_ab.mp4" "$IMG/merge_ab.webp"; do
    if [[ -f $f ]]; then
        printf '  %9d  %s\n' "$(stat -c %s "$f")" "$f"
    fi
done
