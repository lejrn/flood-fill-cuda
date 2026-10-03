#!/usr/bin/env bash
# Build the project page teaser, the carousel clips, their posters and the
# ch06 before/after still.
#
# Run from the repo root (the directory that holds src/ and project-page/):
#
#   FFMPEG=/path/to/ffmpeg PYTHON=/path/to/python \
#     bash project-page/tools/make_clips.sh
#
# FFMPEG    ffmpeg with libx264 (default: ffmpeg on PATH)
# PYTHON    python with Pillow built with WebP support (default: python)
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
# - Posters are written by Pillow as lossless WebP. No .png or .jpg is ever
#   written into the repo (the repo gitignores them).

set -euo pipefail

FFMPEG=${FFMPEG:-ffmpeg}
PYTHON=${PYTHON:-python}

RES=src/flood_fill_cuda/results
VID=project-page/static/videos
IMG=project-page/static/images

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

# 3. The ch06 "runs" carousel loop is not built here: it comes from
#    tools/make_runs_explainer.sh, with the 45 s runs explainer.

# 4. Still: the ch06 before/after figure, one 1536x380 frame, lossless.
gif_poster "$RES/ch06_gpu_nblob_runs/figures/before_after.gif" "$IMG/before_after.webp"

# 5. Side-by-side clips (Nerfies' "stacked" videos). Each pair was rendered
# by one generator on one clock with the same frame count and delays, so
# frame i of the left GIF and frame i of the right GIF line up.
# stack_clip LEFT RIGHT OUT UPSCALE GAP [RIGHT_PTS]
# RIGHT_PTS rescales the right clip's timestamps (e.g. 180/241 plays it
# faster). When the right clip ends first, hstack holds its last frame.
stack_clip() {
    local left=$1 right=$2 out=$3 up=$4 gap=$5 pts=${6:-1} sc="" rt=""
    if (( up > 1 )); then
        sc=",scale=iw*$up:ih*$up:flags=neighbor"
    fi
    if [[ $pts != 1 ]]; then
        rt="setpts=PTS*$pts,"
    fi
    echo "stack  $out  <- $left | $right (x$up, gap $gap, right pts x$pts)"
    ff -i "$left" -i "$right" -filter_complex \
        "[0:v]fps=30,format=rgb24$sc,pad=iw+$gap:ih:0:0:white[l];[1:v]${rt}fps=30,format=rgb24$sc[r];[l][r]hstack=inputs=2,$TO_YUV" \
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
# Each GIF spreads its whole clock over 96 frames: 242 ticks (levels
# 181 + 61) on the left, 181 on the right. Frames sample ticks 0..n-1, so
# playing the right clip at 180/241 of its length gives one BFS level the
# same screen time on both sides, and the one-launch side finishes first.
SEQ=$RES/ch04_gpu_2blob_nblock/wavefront/asym384_b8_t32_sequential.gif
MULTI=$RES/ch04_gpu_2blob_nblock/wavefront/asym384_b8_t32_multisource.gif
stack_clip "$SEQ" "$MULTI" "$VID/two_blobs_ab.mp4" 1 16 180/241
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
