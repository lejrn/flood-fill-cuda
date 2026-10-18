"""libx264 against NVENC on the video's own frames: wall time, CPU time,
size and quality (SSIM and PSNR against the source), one encode at a time.

Usage (from video/):
    uv run build/nvenc_bench.py SRC.mp4 --out DIR --mode clip   # Manim's writer: RGBA frames in, CRF 23
    uv run build/nvenc_bench.py SRC.mp4 --out DIR --mode cut    # assemble.py's concat: decode + CRF 18

--mode clip decodes SRC once to raw RGBA (the frames Manim hands its
writer), then feeds that file to each encoder with the conversion Manim's
PyAV writer does (swscale, BT.601, limited range, bilinear). --mode cut
re-encodes SRC as the concat step does (decode included, plus a
decode-only run to subtract). The ffmpeg used is a static build with NVENC
(video/.tools/, git-ignored; --ffmpeg to override). CPU time is the
encoder process's user + sys (getrusage of the child). Quality: ffmpeg's
ssim and psnr filters, encode against source, all frames.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import resource
import subprocess
import sys
import time
from pathlib import Path

import av

VIDEO = Path(__file__).resolve().parents[1]
FFMPEG = VIDEO / ".tools" / "ffmpeg-n8.1" / "bin" / "ffmpeg"
CSC = "scale=flags=bilinear:out_color_matrix=bt601:out_range=tv,format=yuv420p"

CLIP = {    # Manim's partial movie files: libx264 CRF 23, preset medium (PyAV's default)
    "x264_crf23": ["-vf", CSC, "-c:v", "libx264", "-preset", "medium", "-crf", "23"],
    "nvenc_h264_cq23": ["-vf", CSC, "-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr",
                        "-cq", "23", "-b:v", "0"],
    "nvenc_h264_cq20": ["-vf", CSC, "-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr",
                        "-cq", "20", "-b:v", "0"],
    "nvenc_h264_cq20_gpucsc": ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "20",
                               "-b:v", "0"],
    "nvenc_hevc_cq23": ["-vf", CSC, "-c:v", "hevc_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr",
                        "-cq", "23", "-b:v", "0"],
}
CUT = {     # assemble.py's concat: libx264 CRF 18, preset medium, yuv420p
    "x264_crf18": ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p"],
    "nvenc_h264_cq18": ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "18",
                        "-b:v", "0", "-pix_fmt", "yuv420p"],
    "nvenc_h264_cq16": ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "16",
                        "-b:v", "0", "-pix_fmt", "yuv420p"],
}


def timed(cmd: list[str]) -> tuple[float, float, str]:
    """(wall s, child CPU s, stderr) of one run."""
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    if p.returncode:
        raise SystemExit(f"failed ({p.returncode}): {' '.join(cmd)}\n{p.stderr[-2000:]}")
    return wall, (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime), p.stderr


def quality(ff: str, enc: Path, ref_args: list[str]) -> dict:
    """SSIM (Y, all) and PSNR (average) of enc against the reference input."""
    _, _, err = timed([ff, "-hide_banner", "-nostats", "-i", str(enc), *ref_args,
                       "-lavfi", "[0:v][1:v]ssim;[0:v][1:v]psnr", "-f", "null", "-"])
    ssim = re.search(r"SSIM Y:([\d.]+).*All:([\d.]+)", err)
    psnr = re.search(r"PSNR .*average:([\d.inf]+)", err)
    return {"ssim_y": float(ssim.group(1)), "ssim_all": float(ssim.group(2)), "psnr_avg": float(psnr.group(1))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=("clip", "cut"), default="clip")
    ap.add_argument("--ffmpeg", default=str(FFMPEG))
    ap.add_argument("--only", nargs="*", help="config names to run")
    args = ap.parse_args()
    ff = args.ffmpeg
    args.out.mkdir(parents=True, exist_ok=True)
    s = av.open(str(args.src)).streams.video[0]
    W, H, fps = s.width, s.height, s.average_rate
    print(f"source {args.src.name}: {W}x{H} @ {fps}, load {open('/proc/loadavg').read().split()[:3]}")
    rows = []
    if args.mode == "clip":
        raw = args.out / "src.rgba"
        if not raw.exists():
            timed([ff, "-hide_banner", "-loglevel", "error", "-y", "-i", str(args.src), "-f", "rawvideo",
                   "-pix_fmt", "rgba", str(raw)])
        n = raw.stat().st_size // (W * H * 4)
        src_args = ["-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}", "-r", str(fps), "-i", str(raw)]
        configs = CLIP
    else:
        n = s.frames
        src_args = ["-i", str(args.src)]
        configs = CUT
        wall, cpu, _ = timed([ff, "-hide_banner", "-loglevel", "error", "-i", str(args.src), "-f", "null", "-"])
        rows.append({"config": "decode_only", "wall_s": wall, "cpu_s": cpu, "fps": n / wall, "mb": 0.0})
        print(f"{'decode_only':24s} wall {wall:6.2f} s  cpu {cpu:6.2f} s")
    for name, enc in configs.items():
        if args.only and name not in args.only:
            continue
        dst = args.out / f"{name}.mp4"
        wall, cpu, _ = timed([ff, "-hide_banner", "-loglevel", "error", "-y", *src_args, "-an", *enc, str(dst)])
        q = quality(ff, dst, src_args)
        row = {"config": name, "wall_s": wall, "cpu_s": cpu, "fps": n / wall, "mb": dst.stat().st_size / 1e6, **q}
        rows.append(row)
        print(f"{name:24s} wall {wall:6.2f} s  cpu {cpu:6.2f} s  {n / wall:7.1f} fps  {row['mb']:7.2f} MB  "
              f"SSIM {q['ssim_all']:.5f}  PSNR {q['psnr_avg']:.2f} dB", flush=True)
    meta = {"src": args.src.name, "mode": args.mode, "frames": n, "size": [W, H], "fps": float(fps),
            "ffmpeg": subprocess.run([ff, "-version"], capture_output=True, text=True).stdout.splitlines()[0],
            "load_after": open("/proc/loadavg").read().split()[:3], "rows": rows}
    (args.out / f"bench_{args.mode}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
