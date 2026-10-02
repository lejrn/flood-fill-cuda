#!/usr/bin/env bash
# Build the web cut of the explainer video and its poster for the project page.
#
# Run from the repository root:
#
#   FFMPEG=video/.venv/bin/ffmpeg PYTHON=.venv/bin/python VIDEO_DIR=video \
#       bash project-page/tools/make_explainer.sh
#
# Inputs (gitignored, made by video/build/assemble.py in the main checkout):
#   ${VIDEO_DIR}/out/final_landscape_elevenlabs.mp4
#       1920x1080, 30 fps, 185.73 s, H.264 + AAC mono 48 kHz.
#
# Outputs:
#   project-page/static/videos/explainer.mp4         1280x720 H.264 + AAC 64k mono
#   project-page/static/images/explainer_poster.webp 1280x720, quality 85
#
# Why the cut starts at 44.000 s
#   The first 44.0 s are the intro (clip s_intro, exactly 1320 frames). It is
#   built on third-party drone-show footage with no recorded licence, so the
#   public page must not carry it. The web cut starts at source frame 1320,
#   the first frame of the s00_cpu clip:
#     - frame 1319 is the last intro frame and still shows faint residue;
#       frame 1320 is the flat background (#090d12) and nothing else,
#     - the audio is digital silence from 43.601 s to 44.000 s, and the
#       00_cpu voice ("A CPU fills a blob ...") rises above -60 dBFS at
#       44.056 s, so the cut lands in silence before the first word.
#   Both cut and poster use frame-exact trim and sample-exact atrim filters,
#   not input seeking, so the first frame is clean.
#   The script re-checks both facts before it encodes and stops if a
#   re-rendered source no longer matches them.
#
# LEAD_IN (default 0.5 s) holds that flat first frame, with silence, before
# stage 00 starts, so a browser that starts audio a little late does not
# clip the first word. LEAD_IN=0 gives the bare cut.
#
# The poster is source frame 2070 (69.0 s, 25.0 s into the cut, stage 01
# "one block, one SM"): matrix, blob and GPU panes all populated, the blue
# frontier diamond on red reads well as a thumbnail.
#
# Encoding is CPU only; run nothing else heavy at the same time (6 GB RAM).

set -euo pipefail

FFMPEG="${FFMPEG:-ffmpeg}"
PYTHON="${PYTHON:-python}"
VIDEO_DIR="${VIDEO_DIR:-video}"

SRC="${VIDEO_DIR}/out/final_landscape_elevenlabs.mp4"
OUT_VIDEO="project-page/static/videos/explainer.mp4"
OUT_POSTER="project-page/static/images/explainer_poster.webp"

FPS=30
CUT_FRAME=1320          # first frame of s00_cpu = 44.000 s
CUT_S="44.000"          # the same point for the audio
POSTER_FRAME=2070       # 69.0 s in the source
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

# 1. Check the cut point on this source: the cut frame must be flat
#    background, and the audio from 43.62 s to the cut must be silent.
echo "checking the cut point (frame $CUT_FRAME, $CUT_S s)"
"$FFMPEG" -hide_banner -loglevel error -i "$SRC" -an \
    -vf "trim=start_frame=${CUT_FRAME}:end_frame=$((CUT_FRAME + 1)),setpts=PTS-STARTPTS,scale=480:270" \
    -frames:v 1 -f rawvideo -pix_fmt rgb24 - |
    "$PYTHON" -c '
import sys
import numpy as np
a = np.frombuffer(sys.stdin.buffer.read(), dtype=np.uint8).reshape(270, 480, 3).astype(int)
bg = np.median(a.reshape(-1, 3), axis=0)
off = int((np.abs(a - bg).max(axis=2) > 3).sum())
print(f"  cut frame: background {bg.astype(int).tolist()}, {off} pixels off it")
sys.exit(1 if off > 0 or bg.max() > 40 else 0)
' || { echo "the cut frame is not the flat background: re-check CUT_FRAME" >&2; exit 1; }

"$FFMPEG" -hide_banner -loglevel error -i "$SRC" -vn \
    -af "atrim=start=43.62:end=${CUT_S}" -ac 1 -f s16le - |
    "$PYTHON" -c '
import sys
import numpy as np
x = np.frombuffer(sys.stdin.buffer.read(), dtype=np.int16)
peak = int(np.abs(x.astype(int)).max()) if x.size else 0
print(f"  audio 43.62 s to the cut: {x.size} samples, peak {peak}")
sys.exit(1 if x.size == 0 or peak > 33 else 0)
' || { echo "the audio is not silent before the cut: re-check CUT_S" >&2; exit 1; }

# 2. The video: trim at the cut, scale to 720p, optional lead-in hold.
#    setpts drops the link's frame rate, so fps=30 sets it again (a no-op on
#    this 30 fps source); without it tpad pads zero frames. The lead-in is a
#    whole number of frames, and the audio gets the same delay in samples.
vf="[0:v]trim=start_frame=${CUT_FRAME},setpts=PTS-STARTPTS,fps=${FPS},scale=1280:720:flags=lanczos"
af="[0:a]atrim=start=${CUT_S},asetpts=PTS-STARTPTS"
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

# 3. The poster: one frame, the same scaling, lossy WebP.
echo "writing $OUT_POSTER (source frame $POSTER_FRAME)"
"$FFMPEG" -hide_banner -loglevel warning -y -i "$SRC" -an \
    -vf "trim=start_frame=${POSTER_FRAME}:end_frame=$((POSTER_FRAME + 1)),setpts=PTS-STARTPTS,scale=1280:720:flags=lanczos" \
    -frames:v 1 -map_metadata -1 \
    -c:v libwebp -quality 85 -compression_level 6 \
    "$OUT_POSTER"

# 4. Report what was written.
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
