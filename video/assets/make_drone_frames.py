"""Simulated defence-camera footage for the intro, rendered from data.

Three frame sets of 240 frames (8 s at 30 fps, 640 x 360):
  drones_sky     a clear sky with soft clouds; five quadcopters and one
                 missile fly through (procedural, seeded, deterministic)
  drones_mask    the motion filter: |frame - static background| above a
                 threshold, white on black
  drones_labels  the mask labelled and coloured by the repo's Chapter 6
                 run-table kernel (real GPU output through `recolor`),
                 drawn on black

Run from the repo root with the ROOT venv (numba + CUDA):
    .venv/bin/python video/assets/make_drone_frames.py
Writes video/assets/drones_*/frame_NNN.png + meta.json (gitignored).
"""
from __future__ import annotations

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


def to_display(painted: np.ndarray) -> np.ndarray:
    """(H, W, 3) painted image on white -> the same blobs on black."""
    out = painted.copy()
    white = np.all(painted == 255, axis=2)
    out[white] = 0
    return out


def main() -> int:
    bg = sky_background()
    bg_gray = (0.299 * bg[..., 0] + 0.587 * bg[..., 1] + 0.114 * bg[..., 2])
    drones, missile = objects()
    folders = {k: HERE / f"drones_{k}" for k in ("sky", "mask", "labels")}
    for p in folders.values():
        p.mkdir(parents=True, exist_ok=True)
        for old in p.glob("frame_*.png"):
            old.unlink()
    engine = RunRecolor(W, H, run_capacity=65536)
    blobs = []
    for f in range(N):
        overlay = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        for dr in drones:
            dr.draw(d, f)
        missile.draw(d, f)
        overlay = overlay.resize((W, H), Image.LANCZOS)
        frame = Image.alpha_composite(Image.fromarray(bg).convert("RGBA"), overlay).convert("RGB")
        arr = np.asarray(frame)
        gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        mask = np.abs(gray - bg_gray) > THRESH                       # (H, W)
        # the repo's images are (width, height, 3), indexed [x, y], red on white
        img = np.full((W, H, 3), 255, dtype=np.uint8)
        img[mask.T] = (255, 0, 0)
        r = recolor(img, contract="rgb", engine=engine, emit_seeds=False)
        painted = np.transpose(r.img, (1, 0, 2))                     # back to (H, W, 3)
        blobs.append(int(r.n_blobs))
        frame.save(folders["sky"] / f"frame_{f:03d}.png")
        Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB").save(folders["mask"] / f"frame_{f:03d}.png")
        Image.fromarray(to_display(painted)).save(folders["labels"] / f"frame_{f:03d}.png")
    for k, p in folders.items():
        meta = {"source": "make_drone_frames.py", "frames": N, "size": [W, H],
                "delay_ms": round(1000 / FPS), "kind": k}
        if k == "labels":
            meta["blobs_min"], meta["blobs_max"] = min(blobs), max(blobs)
        (p / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"{N} frames x 3 sets, {W}x{H}; blobs per frame {min(blobs)}..{max(blobs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
