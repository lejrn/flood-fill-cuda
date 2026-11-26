#!/usr/bin/env bash
# Build the web cut of the explainer video and its poster for the project page.
#
# Run from the repository root:
#
#   FFMPEG=video/.venv/bin/ffmpeg PYTHON=.venv/bin/python VIDEO_DIR=video \
#       bash project-page/tools/make_explainer.sh
#
# Input (gitignored, made by video/build/assemble.py in the main checkout):
#   ${VIDEO_DIR}/out/final_chain_iso_gl_landscape_kokoro_paced.mp4
#       the chain cut, v5: "flood fill, from pixels to tiles".
#       1920x1080, 30 fps, 278.43 s, H.264 + AAC mono 48 kHz.
#
# Outputs:
#   project-page/static/videos/explainer.mp4         1280x720 H.264 + AAC 64k mono
#   project-page/static/images/explainer_poster.webp 1280x720, quality 85
#
# The whole cut goes on the page, from its first frame. It opens and closes
# on drone-show footage by a third party.
# (Until 2026-11 the page carried the older three-pane cut and started it at
# 44 s to leave that footage out.)
#
# LEAD_IN (default 0.5 s) holds the first frame, with silence, before the
# narration starts, so a browser that starts audio a little late does not
# clip the first word. LEAD_IN=0 gives the bare cut. make_captions.py reads
# the same default, so change both together.
#
# The poster is source frame POSTER_FRAME (default 7230, 241.0 s, beat r12:
# the tile pieces of a crop, each wave in its own colour).
#
# Encoding is CPU only; run nothing else heavy at the same time.

set -euo pipefail

FFMPEG="${FFMPEG:-ffmpeg}"
PYTHON="${PYTHON:-python}"
VIDEO_DIR="${VIDEO_DIR:-video}"

SRC="${VIDEO_DIR}/out/final_chain_iso_gl_landscape_kokoro_paced.mp4"
OUT_VIDEO="project-page/static/videos/explainer.mp4"
OUT_POSTER="project-page/static/images/explainer_poster.webp"

FPS=30
POSTER_FRAME="${POSTER_FRAME:-7230}"
LEAD_IN="${LEAD_IN:-0.5}"

if [[ ! -d project-page/tools ]]; then
    echo "run this from the repository root (project-page/tools not found)" >&2
    exit 1
fi
if [[ ! -f "$SRC" ]]; then
    echo "source not found: $SRC (set VIDEO_DIR to the main checkout's video/)" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUT_VIDEO")" "$(dirname "$OUT_POSTER")"

# 1. The video: scale to 720p, optional lead-in hold. The lead-in is a whole
#    number of frames, and the audio gets the same delay in samples.
vf="[0:v]fps=${FPS},scale=1280:720:flags=lanczos"
af="[0:a]asetpts=PTS-STARTPTS"
lead_frames=$("$PYTHON" -c "print(round(float('${LEAD_IN}') * ${FPS}))")
if (( lead_frames > 0 )); then
    vf+=",tpad=start=${lead_frames}:start_mode=clone"
    af+=",adelay=delays=$(( lead_frames * 48000 / FPS ))S:all=1"
fi
vf+=",format=yuv420p[v]"
af+="[a]"

echo "encoding $OUT_VIDEO (lead-in ${LEAD_IN} s)"
"$FFMPEG" -hide_banner -loglevel warning -stats -y -i "$SRC" \
    -filter_complex "${vf};${af}" \
    -map "[v]" -map "[a]" -map_metadata -1 \
    -c:v libx264 -preset slow -crf 28 -pix_fmt yuv420p -r "$FPS" \
    -c:a aac -b:a 64k -ac 1 -ar 48000 \
    -movflags +faststart \
    "$OUT_VIDEO"

# 2. The poster: one frame, the same scaling, lossy WebP.
echo "writing $OUT_POSTER (source frame $POSTER_FRAME)"
"$FFMPEG" -hide_banner -loglevel warning -y -i "$SRC" -an \
    -vf "trim=start_frame=${POSTER_FRAME}:end_frame=$((POSTER_FRAME + 1)),setpts=PTS-STARTPTS,scale=1280:720:flags=lanczos" \
    -frames:v 1 -map_metadata -1 \
    -c:v libwebp -quality 85 -compression_level 6 \
    "$OUT_POSTER"

# 3. Report what was written.
"$PYTHON" - "$OUT_VIDEO" "$OUT_POSTER" <<'EOF'
import os
import sys
from PIL import Image

video, poster = sys.argv[1:3]
print(f"{video}: {os.path.getsize(video):,} bytes")
with Image.open(poster) as im:
    frames = getattr(im, "n_frames", 1)
    print(f"{poster}: {os.path.getsize(poster):,} bytes, {im.width}x{im.height}, "
          f"{im.format}, {frames} frame(s)")
EOF
"$FFMPEG" -hide_banner -i "$OUT_VIDEO" 2>&1 | grep -E "Duration|Stream" || true
v_frames=$("$FFMPEG" -hide_banner -i "$OUT_VIDEO" -map 0:v -f null - 2>&1 |
    tr '\r' '\n' | grep -oE 'frame= *[0-9]+' | tail -1 | grep -oE '[0-9]+')
a_time=$("$FFMPEG" -hide_banner -i "$OUT_VIDEO" -map 0:a -f null - 2>&1 |
    tr '\r' '\n' | grep -oE 'time=[0-9:.]+' | tail -1)
echo "video: ${v_frames} frames = $("$PYTHON" -c "print(f'{${v_frames} / ${FPS}:.3f}')") s; audio: ${a_time#time=}"
