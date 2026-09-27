"""Footage for the intro: three frame sets, rendered from data.

  drones_sky     the camera
  drones_mask    the motion filter: |frame - static background| above a
                 threshold, white on black
  drones_labels  the mask labelled and coloured by the repo's Chapter 6
                 run-table kernel (real GPU output through `recolor`),
                 drawn on black

Two sources:
  default        a simulation: a clear sky with soft clouds, five
                 quadcopters and one missile (procedural, seeded), 640 x 360,
                 240 frames at 30 fps; the background is the known sky
  --source FILE  real footage (e.g. the drone-show Short downloaded with
                 yt-dlp into assets/source/); frames [--start, --start +
                 --frames) at native resolution, extracted once with the
                 video venv's ffmpeg into assets/source/<stem>_frames/.
                 The Short is one continuous take from a moving camera,
                 so no background model holds (--filter median is kept
                 for a static camera); the default --filter tophat is a
                 white top-hat: gray minus its 9x9 morphological opening,
                 which keeps small bright blobs against the sky whatever
                 the camera does. Blobs under MIN_AREA px or wider than
                 MAX_BBOX px (court lines, reflections) are dropped.

Tracking. The kernel labels each frame on its own, in scan order, so a
blob's label (and palette colour) would change from frame to frame. The
labels set therefore colours each blob by a TRACK id: blobs are matched
to the previous frame's tracks by nearest centroid within GATE pixels
(greedy, closest pairs first), unmatched blobs open new tracks, and a
track survives MISS_LIMIT frames without a match. Each track keeps one
colour for its life. This is the classic centroid tracker; the labelling
itself is still the ch06 kernel's.

Run from the repo root with the ROOT venv (numba + CUDA):
    .venv/bin/python video/assets/make_drone_frames.py
    .venv/bin/python video/assets/make_drone_frames.py --source video/assets/source/drones_short.mp4 --start 1020 --frames 240
Writes video/assets/drones_*/frame_NNN.png + meta.json (gitignored).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from flood_fill_cuda.chapters.ch06_gpu_nblob_runs.recolor import RunRecolor, recolor  # noqa: E402

W, H, N, FPS, SS = 640, 360, 240, 30, 2
THRESH = 24
rng = np.random.default_rng(7)

BODY = (58, 60, 68)
MISSILE = (74, 76, 84)


def sky_background() -> np.ndarray:
    top, bottom = np.array([118, 168, 228]), np.array([196, 218, 240])
    t = np.linspace(0, 1, H)[:, None, None]
    bg = top * (1 - t) + bottom * t
    noise = rng.random((H // 20 + 1, W // 20 + 1)).astype(np.float32)
    clouds = Image.fromarray((noise * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)
    clouds = clouds.filter(ImageFilter.GaussianBlur(14))
    c = np.asarray(clouds).astype(np.float32) / 255.0
    c = np.clip((c - 0.45) * 2.2, 0, 1) * 26.0        # a few soft white patches
    bg = bg + c[:, :, None]
    return np.clip(bg, 0, 255).astype(np.uint8)


class Drone:
    def __init__(self, x, y, vx, vy, size, phase):
        self.x, self.y, self.vx, self.vy, self.size, self.phase = x, y, vx, vy, size, phase

    def pos(self, f):
        return (self.x + self.vx * f + 6 * math.sin(0.05 * f + self.phase),
                self.y + self.vy * f + 4 * math.cos(0.037 * f + self.phase))

    def draw(self, d: ImageDraw.ImageDraw, f: int):
        cx, cy = self.pos(f)
        cx, cy, s = cx * SS, cy * SS, self.size * SS
        arm = 0.95 * s
        for ang in (45, 135, 225, 315):
            ex, ey = cx + arm * math.cos(math.radians(ang)), cy + arm * math.sin(math.radians(ang)) * 0.55
            d.line([(cx, cy), (ex, ey)], fill=BODY, width=max(1, int(0.16 * s)))
            r = 0.34 * s
            d.ellipse([ex - r, ey - r * 0.55, ex + r, ey + r * 0.55], fill=BODY)
        d.rounded_rectangle([cx - 0.5 * s, cy - 0.24 * s, cx + 0.5 * s, cy + 0.24 * s],
                            radius=0.12 * s, fill=BODY)


class Missile:
    def __init__(self, f0, f1, p0, p1, size):
        self.f0, self.f1, self.p0, self.p1, self.size = f0, f1, np.array(p0), np.array(p1), size

    def draw(self, d: ImageDraw.ImageDraw, f: int):
        if not (self.f0 <= f <= self.f1):
            return
        t = (f - self.f0) / (self.f1 - self.f0)
        c = (self.p0 * (1 - t) + self.p1 * t) * SS
        v = self.p1 - self.p0
        ang = math.atan2(v[1], v[0])
        s = self.size * SS
        pts = []
        for px, py in [(-1.6, -0.22), (1.3, -0.22), (1.7, 0.0), (1.3, 0.22), (-1.6, 0.22)]:
            x, y = px * s, py * s
            pts.append((c[0] + x * math.cos(ang) - y * math.sin(ang),
                        c[1] + x * math.sin(ang) + y * math.cos(ang)))
        d.polygon(pts, fill=MISSILE)
        for sign in (-1, 1):
            fin = [(-1.6, sign * 0.22), (-1.2, sign * 0.22), (-1.35, sign * 0.62)]
            d.polygon([(c[0] + px * s * math.cos(ang) - py * s * math.sin(ang),
                        c[1] + px * s * math.sin(ang) + py * s * math.cos(ang)) for px, py in fin],
                      fill=MISSILE)


def objects():
    drones = [
        Drone(60, 70, 1.6, 0.25, 15, 0.0),
        Drone(520, 40, -1.1, 0.55, 21, 1.3),
        Drone(300, 210, 0.9, -0.45, 28, 2.1),
        Drone(120, 300, 1.3, -0.15, 13, 3.7),
        Drone(600, 250, -1.9, -0.35, 18, 5.0),
    ]
    missile = Missile(50, 190, (-50, 300), (700, 90), 19)
    return drones, missile


def to_display(painted: np.ndarray) -> np.ndarray:  # kept for the palette-painted variant
    """(H, W, 3) painted image on white -> the same blobs on black."""
    out = painted.copy()
    white = np.all(painted == 255, axis=2)
    out[white] = 0
    return out


# ------------------------------------------------------------- tracking
GATE = 18          # px: a blob further than this from every track's prediction is a new drone
MISS_LIMIT = 8     # frames a track may go unmatched before it is closed
MIN_AREA = 4       # px: smaller blobs in real footage are compression noise
MAX_BBOX = 26      # px: wider blobs in real footage are lines or reflections, not drones
TOPHAT = 9         # px: the opening size of the top-hat filter
MAX_STEP = 12      # px: a larger frame-to-frame shift is a correlation glitch, not the camera
STATIC_BAND = 0.35  # the bottom fraction of the frame used to measure camera motion
TRACK_PALETTE = np.array([
    (76, 143, 214), (63, 185, 80), (217, 154, 43), (163, 113, 247), (42, 161, 152),
    (226, 100, 90), (240, 228, 66), (255, 122, 182), (86, 180, 233), (196, 111, 47),
    (154, 208, 245), (184, 233, 134), (255, 176, 59), (128, 222, 234), (233, 150, 122),
], dtype=np.uint8)


class Tracker:
    """Centroid tracker: each track predicts its next position with its
    last velocity (a constant-velocity model, the simplest Kalman-style
    prediction), blobs are matched to predictions by nearest distance
    within GATE (greedy, closest pairs first)."""

    def __init__(self):
        self.tracks = {}          # id -> (centroid xy, velocity xy, misses)
        self.next_id = 0

    def update(self, centroids: np.ndarray) -> np.ndarray:
        """centroids (n, 2) for this frame's blobs -> track id per blob."""
        n = len(centroids)
        ids = np.full(n, -1, dtype=np.int64)
        if self.tracks and n:
            tids = list(self.tracks.keys())
            prev = np.array([self.tracks[t][0] + self.tracks[t][1] for t in tids])   # predicted
            d = np.sqrt(((centroids[:, None, :] - prev[None, :, :]) ** 2).sum(axis=2))
            order = np.argsort(d, axis=None)
            used_b, used_t = set(), set()
            for flat in order:
                b, t = divmod(int(flat), len(tids))
                if d[b, t] > GATE:
                    break
                if b in used_b or t in used_t:
                    continue
                ids[b] = tids[t]
                used_b.add(b)
                used_t.add(t)
        for b in range(n):
            if ids[b] < 0:
                ids[b] = self.next_id
                self.next_id += 1
        matched = set(int(i) for i in ids)
        for t in list(self.tracks.keys()):
            if t not in matched:
                c, v, miss = self.tracks[t]
                if miss + 1 > MISS_LIMIT:
                    del self.tracks[t]
                else:
                    self.tracks[t] = (c + v, v, miss + 1)             # coast along the prediction
        for b in range(n):
            t = int(ids[b])
            if t in self.tracks:
                c, v, _ = self.tracks[t]
                v = 0.6 * (centroids[b] - c) + 0.4 * v                # smoothed velocity
            else:
                v = np.zeros(2)
            self.tracks[t] = (centroids[b], v, 0)
        return ids


def blob_centroids(label_hw: np.ndarray, n_blobs: int) -> np.ndarray:
    """(n_blobs, 2) centroid (x, y) per label 0..n_blobs-1 of a (H, W) label map."""
    ys, xs = np.nonzero(label_hw >= 0)
    lab = label_hw[ys, xs]
    cnt = np.bincount(lab, minlength=n_blobs).astype(np.float64)
    sx = np.bincount(lab, weights=xs, minlength=n_blobs)
    sy = np.bincount(lab, weights=ys, minlength=n_blobs)
    cnt[cnt == 0] = 1
    return np.stack([sx / cnt, sy / cnt], axis=1)


def paint_tracks(label_hw: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """(H, W, 3) on black: every blob in its track's colour."""
    out = np.zeros(label_hw.shape + (3,), dtype=np.uint8)
    on = label_hw >= 0
    out[on] = TRACK_PALETTE[ids[label_hw[on]] % len(TRACK_PALETTE)]
    return out


def gray_of(arr: np.ndarray) -> np.ndarray:
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]


def simulated_frames():
    """Yields (frame RGB (H, W, 3) uint8, background gray (H, W)) for the simulation."""
    bg = sky_background()
    bg_gray = gray_of(bg)
    drones, missile = objects()
    for f in range(N):
        overlay = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        for dr in drones:
            dr.draw(d, f)
        missile.draw(d, f)
        overlay = overlay.resize((W, H), Image.LANCZOS)
        frame = Image.alpha_composite(Image.fromarray(bg).convert("RGBA"), overlay).convert("RGB")
        yield np.asarray(frame), bg_gray


def extract_frames(path: Path, start: int, count: int) -> list:
    """Frames [start, start + count) of a video as PNGs, via the video
    venv's ffmpeg (this script runs in the root venv, which has no PyAV)."""
    import subprocess

    ffmpeg = HERE.parent / ".venv" / "bin" / "ffmpeg"
    out = path.parent / f"{path.stem}_frames"
    out.mkdir(parents=True, exist_ok=True)
    want = [out / f"{i:05d}.png" for i in range(start, start + count)]
    if not all(p.exists() for p in want):
        for old in out.glob("*.png"):
            old.unlink()
        subprocess.run([str(ffmpeg), "-y", "-loglevel", "error", "-i", str(path),
                        "-vf", f"select='between(n\\,{start}\\,{start + count - 1})'", "-vsync", "0",
                        "-start_number", str(start), str(out / "%05d.png")], check=True)
    if not all(p.exists() for p in want):
        raise SystemExit(f"ffmpeg did not produce frames {start}..{start + count - 1} in {out}")
    return want


def stabilise(frames: list) -> list:
    """Cancel camera translation: the shift between consecutive frames by
    phase correlation on the gray image (whole pixels), accumulated to a
    per-frame offset against the first frame; every frame is moved by
    the opposite offset and all are cropped to the area they share."""
    # correlate only the bottom part of the frame: the court and buildings
    # are static structure, while the drone field above moves as one body
    # and would be mistaken for camera motion
    h, w = frames[0].shape[:2]
    y0 = int(h * (1 - STATIC_BAND))
    grays = [gray_of(f[y0:]).astype(np.float32) for f in frames]
    hb, wb = grays[0].shape
    offs = [(0, 0)]
    for a, b in zip(grays[:-1], grays[1:]):
        F = np.fft.fft2(a) * np.conj(np.fft.fft2(b))
        r = np.fft.ifft2(F / (np.abs(F) + 1e-6)).real
        dy, dx = np.unravel_index(int(np.argmax(r)), r.shape)
        dy = dy - hb if dy > hb // 2 else dy
        dx = dx - wb if dx > wb // 2 else dx
        if abs(dx) > MAX_STEP or abs(dy) > MAX_STEP:      # a spurious peak, not camera motion
            dx, dy = 0, 0
        offs.append((offs[-1][0] + dx, offs[-1][1] + dy))
    mx = max(abs(o[0]) for o in offs)
    my = max(abs(o[1]) for o in offs)
    out = []
    for f, (dx, dy) in zip(frames, offs):
        # content moved by (dx, dy) since frame 0: move it back, then crop the shared window
        canvas = np.zeros_like(f)
        ys, ye = max(0, -dy), min(h, h - dy)
        xs, xe = max(0, -dx), min(w, w - dx)
        canvas[ys:ye, xs:xe] = f[ys + dy:ye + dy, xs + dx:xe + dx]
        out.append(canvas[my:h - my, mx:w - mx])
    print(f"stabilised: max drift {mx} x {my} px, kept {out[0].shape[1]}x{out[0].shape[0]}")
    return out


def tophat_background(arr: np.ndarray) -> np.ndarray:
    """Gray background by a 9x9 morphological opening: everything smaller
    than 9 px (the drone lights) is removed, the rest stays."""
    g = Image.fromarray(gray_of(arr).astype(np.uint8))
    g = g.filter(ImageFilter.MinFilter(TOPHAT)).filter(ImageFilter.MaxFilter(TOPHAT))
    return np.asarray(g).astype(np.float32)


def source_frames(path: Path, start: int, count: int, stabilised: bool = True, filt: str = "tophat"):
    """Frames [start, start + count) of a video with a background per frame:
    `tophat` (spatial: the 9x9 opening of the frame itself, camera-proof) or
    `median` (temporal: the per-pixel median of the window, for a steady
    camera; computed in row chunks, the stack is too big to median in one
    go on this laptop)."""
    frames = [np.asarray(Image.open(p).convert("RGB")) for p in extract_frames(path, start, count)]
    if filt == "tophat":
        for f in frames:
            yield f, tophat_background(f)
        return
    if stabilised:
        frames = stabilise(frames)
    h, w = frames[0].shape[:2]
    bg_gray = np.empty((h, w), dtype=np.float32)
    step = 32
    for y in range(0, h, step):
        chunk = np.stack([gray_of(f[y:y + step]) for f in frames]).astype(np.float32)
        bg_gray[y:y + step] = np.median(chunk, axis=0)
    for f in frames:
        yield f, bg_gray


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=None, help="real footage instead of the simulation")
    ap.add_argument("--start", type=int, default=0, help="first source frame")
    ap.add_argument("--frames", type=int, default=N, help="frames to use from the source")
    ap.add_argument("--thresh", type=float, default=None, help="mask threshold (gray levels)")
    ap.add_argument("--fps", type=float, default=FPS)
    ap.add_argument("--no-stabilise", action="store_true", help="median filter: skip alignment")
    ap.add_argument("--filter", choices=["tophat", "median"], default="tophat")
    args = ap.parse_args()

    if args.source:
        gen = source_frames(args.source, args.start, args.frames,
                            stabilised=not args.no_stabilise, filt=args.filter)
        thresh = args.thresh if args.thresh is not None else (60.0 if args.filter == "tophat" else 40.0)
        source = f"{args.source.name} frames {args.start}..{args.start + args.frames - 1}"
    else:
        gen = simulated_frames()
        thresh = args.thresh if args.thresh is not None else THRESH
        source = "make_drone_frames.py (simulation)"

    folders = {k: HERE / f"drones_{k}" for k in ("sky", "mask", "labels")}
    for p in folders.values():
        p.mkdir(parents=True, exist_ok=True)
        for old in p.glob("frame_*.png"):
            old.unlink()
    engine = None
    tracker = Tracker()
    blobs, size = [], None
    for f, (arr, bg_gray) in enumerate(gen):
        h, w = arr.shape[:2]
        if engine is None:
            engine = RunRecolor(w, h, run_capacity=max(65536, w * h // 8))
            size = [w, h]
        if args.source and args.filter == "tophat":
            mask = (gray_of(arr) - bg_gray) > thresh                    # brighter than surroundings
        else:
            mask = np.abs(gray_of(arr) - bg_gray) > thresh             # (H, W)
        # the repo's images are (width, height, 3), indexed [x, y], red on white
        img = np.full((w, h, 3), 255, dtype=np.uint8)
        img[mask.T] = (255, 0, 0)
        r = recolor(img, contract="rgb", engine=engine, emit_seeds=False, emit_label=True, copy_img=False)
        label_hw = np.ascontiguousarray(r.label.T)                      # (H, W), -1 off-blob
        # the kernel's labels are canonical run indices (sparse); make them dense 0..n-1
        on = label_hw >= 0
        uniq, inv = np.unique(label_hw[on], return_inverse=True)
        label_hw = np.full(label_hw.shape, -1, dtype=np.int64)
        label_hw[on] = inv
        n_blobs = len(uniq)
        if args.source and n_blobs:
            # real footage: drop specks under MIN_AREA px (noise) and blobs wider
            # than MAX_BBOX px (court lines, reflections: not point targets)
            lab = label_hw[on]
            ys, xs = np.nonzero(on)
            area = np.bincount(lab, minlength=n_blobs)
            x0 = np.full(n_blobs, 10 ** 9); x1 = np.full(n_blobs, -1)
            y0 = np.full(n_blobs, 10 ** 9); y1 = np.full(n_blobs, -1)
            np.minimum.at(x0, lab, xs); np.maximum.at(x1, lab, xs)
            np.minimum.at(y0, lab, ys); np.maximum.at(y1, lab, ys)
            small = (area < MIN_AREA) | (x1 - x0 + 1 > MAX_BBOX) | (y1 - y0 + 1 > MAX_BBOX)
            if small.any():
                drop = small[lab]
                idx = np.nonzero(on)
                label_hw[idx[0][drop], idx[1][drop]] = -1
                mask = label_hw >= 0
                on = mask
                uniq, inv = np.unique(label_hw[on], return_inverse=True)
                label_hw = np.full(label_hw.shape, -1, dtype=np.int64)
                label_hw[on] = inv
                n_blobs = len(uniq)
        # the kernel's labels are canonical per frame; the tracker makes them persistent
        ids = tracker.update(blob_centroids(label_hw, n_blobs))
        blobs.append(n_blobs)
        Image.fromarray(arr).save(folders["sky"] / f"frame_{f:03d}.png")
        Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB").save(folders["mask"] / f"frame_{f:03d}.png")
        Image.fromarray(paint_tracks(label_hw, ids)).save(folders["labels"] / f"frame_{f:03d}.png")
    n = len(blobs)
    for k, p in folders.items():
        meta = {"source": source, "frames": n, "size": size,
                "delay_ms": round(1000 / args.fps), "kind": k, "thresh": thresh}
        if k == "labels":
            meta["blobs_min"], meta["blobs_max"] = min(blobs), max(blobs)
            meta["tracks"] = tracker.next_id
            meta["tracking"] = f"centroid + constant velocity, gate {GATE} px, miss limit {MISS_LIMIT}"
        (p / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"{n} frames x 3 sets, {size[0]}x{size[1]}; blobs per frame {min(blobs)}..{max(blobs)}; "
          f"{tracker.next_id} tracks; source: {source}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
