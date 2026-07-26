"""FastAPI app: POST /api/fill, GET /healthz, and the static paint frontend.

Response format for /api/fill (application/octet-stream), chosen over
JSON+base64 for size (no +33% overhead) and zero-copy decoding on the
browser side (new Uint16Array(buf, 16) needs no parsing):

    offset 0   uint32 LE  magic 0x46494C4C ("FILL")
    offset 4   uint32 LE  width  (crop px)
    offset 8   uint32 LE  height (crop px)
    offset 12  uint32 LE  levels
    offset 16  width*height uint16 LE, row-major (i = y*width + x):
               0 = not part of the blob, else min(depth+1, 65535)

?mode=cpu|gpu (default gpu) picks the engine — see engine.py's module
docstring for why the two are a fair side-by-side comparison. X-Mode on
the response echoes back which one actually ran. X-Kernel-Ms/X-Total-Ms
and X-Amplified-Filled report the *amplified*-scale run (see engine.py's
module docstring) -- honest timing/pixel-count at the scale where the
GPU's advantage is real, even though the returned depth map itself is at
the size actually painted.

GZipMiddleware is worthwhile here specifically because the encoding makes
the background all-zeros: a real stroke's payload compresses hard.
"""

import asyncio
import io
import struct
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Request, Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from . import engine
from . import survey

STATIC_DIR = Path(__file__).parent / "static"

MAGIC = 0x46494C4C
HEADER_FMT = "<IIII"   # magic, width, height, levels
HEADER_SIZE = struct.calcsize(HEADER_FMT)

# /api/scan wire format v3 (ch06): header, then sweep uint16[w*h], then
# track uint16[w*h] (dense label ids, 0=background), then n_blobs pairs of
# uint32 (x, y) — the GPU-chosen canonical seeds, row i belongs to track
# i+1.
#
# Two v2 fields are GONE because ch06 cannot produce them, and inventing
# them would be a lie the rest of this repo doesn't tell:
#   depth  -> sweep. ch06 is not a BFS, so no pixel has a "level". The
#             sweep field is the order the kernel SCANS in (row-major,
#             which is left-to-right on the canvas), used to drive the
#             reveal. It is a spatial ordering, not a timeline.
#   prov   -> dropped. Provisional labels were a ch05 seed_merge artifact
#             (colliding waves before the union settled). ch06 merges runs
#             in one data-independent pass; there is no intermediate state
#             to show. Its only consumer was SKYWATCH, now removed.
SCAN_MAGIC = 0x5343414E   # "SCAN"
SCAN_HEADER_FMT = "<IIIIII"   # magic, width, height, steps, n_blobs, flags
SCAN_FLAG_AMPLIFIED = 1       # an amplified-scale timing run was included

MAX_BODY_BYTES = 25 * 1024 * 1024   # PNG upload cap; engine.MAX_PIXELS is
                                     # the real (much tighter) size guard
ALPHA_THRESHOLD = 128
BACKLOG_CAP = 8                     # queued-for-GPU cap; fills run 5-50ms
                                     # so this is invisible in practice

# The ONLY thing in this process that ever touches the GPU. Concurrent
# cooperative-kernel launches nondeterministically wedge under WSL2 (see
# chapters/README.md "Finding 2") -- a single-worker executor makes
# concurrent launches structurally impossible, not just discouraged. Never
# run this app with --workers > 1: extra workers are extra processes, each
# with its own single-worker executor, which recreates the exact hazard
# this guards against.
_gpu_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")
# CPU-mode fills never touch CUDA, so they carry none of the above hazard
# and get their own pool -- a slow CPU fill (that's the point of the mode)
# never blocks GPU-mode requests waiting behind it.
_cpu_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cpu-fill")
_inflight = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_gpu_executor, engine.warmup)
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1024)


@app.get("/healthz")
async def healthz():
    from numba import cuda
    device = cuda.get_current_device()
    name = device.name
    if isinstance(name, bytes):
        name = name.decode()
    name = name.split("\x00", 1)[0].strip()   # numba pads this to a fixed C-array width
    return {"status": "ok", "warm": True, "device": name}


def _parse_optional_float(request, name):
    raw = request.query_params.get(name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}")


@app.post("/api/fill")
async def fill(request: Request):
    global _inflight

    mode = request.query_params.get("mode", "gpu").lower()
    if mode not in engine.MODES:
        return JSONResponse(
            {"detail": f"mode must be one of {engine.MODES}, got {mode!r}"},
            status_code=400)

    try:
        # Crop-local coordinates of where the user released the pointer;
        # the seed the fill spreads from. Falls back to the mask's center
        # of mass (engine.run_fill's default) if omitted.
        seed_x = _parse_optional_float(request, "seed_x")
        seed_y = _parse_optional_float(request, "seed_y")
    except ValueError as e:
        return JSONResponse({"detail": str(e)}, status_code=400)

    body = await request.body()
    if not body:
        return JSONResponse({"detail": "empty request body"}, status_code=400)
    if len(body) > MAX_BODY_BYTES:
        return JSONResponse({"detail": "request body too large"}, status_code=413)

    try:
        img = Image.open(io.BytesIO(body))
        img.load()
    except (UnidentifiedImageError, OSError):
        return JSONResponse({"detail": "body is not a decodable PNG"},
                            status_code=400)

    rgba = np.array(img.convert("RGBA"))
    mask = rgba[:, :, 3] >= ALPHA_THRESHOLD

    # Check-then-increment with no `await` between them: the event loop is
    # single-threaded, so this pair can't race even without a lock.
    if _inflight >= BACKLOG_CAP:
        return JSONResponse({"detail": "server busy, try again shortly"},
                            status_code=503)
    _inflight += 1
    try:
        loop = asyncio.get_running_loop()
        executor = _gpu_executor if mode == "gpu" else _cpu_executor
        outcome = await loop.run_in_executor(
            executor, engine.run_fill, mask, mode, seed_x, seed_y)
    except engine.MaskTooLargeError as e:
        return JSONResponse({"detail": str(e)}, status_code=413)
    except ValueError as e:
        return JSONResponse({"detail": str(e)}, status_code=400)
    except RuntimeError as e:
        return JSONResponse({"detail": str(e)}, status_code=500)
    finally:
        _inflight -= 1

    header = struct.pack(HEADER_FMT, MAGIC, outcome.width, outcome.height,
                         outcome.levels)
    payload = header + outcome.depth_u16.tobytes()
    return Response(
        content=payload,
        media_type="application/octet-stream",
        headers={
            "X-Filled": str(outcome.filled),
            "X-Amplified-Filled": str(outcome.amplified_filled),
            "X-Kernel-Ms": f"{outcome.kernel_ms:.3f}",
            "X-Total-Ms": f"{outcome.total_ms:.3f}",
            "X-Mode": outcome.mode,
        },
    )


@app.post("/api/scan")
async def scan(request: Request):
    """Seedless discovery over the WHOLE canvas. No seeds, no mode — the
    GPU finds every blob (ch06's run-table connected components), and the
    njit seedless reference runs concurrently on the CPU pool for the race
    bar. An empty canvas is valid (n_blobs=0).

    ?amp=1 additionally times the same shape upscaled ~250x, so a small
    painted stroke can report a kernel time that isn't dominated by six
    launch overheads — the paint page's RUNS mode asks for this."""
    global _inflight

    want_amp = request.query_params.get("amp", "0") == "1"

    body = await request.body()
    if not body:
        return JSONResponse({"detail": "empty request body"}, status_code=400)
    if len(body) > MAX_BODY_BYTES:
        return JSONResponse({"detail": "request body too large"},
                            status_code=413)
    try:
        img = Image.open(io.BytesIO(body))
        img.load()
    except (UnidentifiedImageError, OSError):
        return JSONResponse({"detail": "body is not a decodable PNG"},
                            status_code=400)

    rgba = np.array(img.convert("RGBA"))
    mask = rgba[:, :, 3] >= ALPHA_THRESHOLD

    if _inflight >= BACKLOG_CAP:
        return JSONResponse({"detail": "server busy, try again shortly"},
                            status_code=503)
    _inflight += 1
    try:
        loop = asyncio.get_running_loop()
        gpu_task = loop.run_in_executor(_gpu_executor, engine.discover,
                                        mask, want_amp)
        njit_task = loop.run_in_executor(_cpu_executor,
                                         engine.njit_reference_ms, mask)
        outcome = await gpu_task
        njit_ms = await njit_task
    except engine.MaskTooLargeError as e:
        return JSONResponse({"detail": str(e)}, status_code=413)
    except ValueError as e:
        return JSONResponse({"detail": str(e)}, status_code=400)
    except RuntimeError as e:
        return JSONResponse({"detail": str(e)}, status_code=500)
    finally:
        _inflight -= 1

    flags = SCAN_FLAG_AMPLIFIED if outcome.amplified_filled else 0
    header = struct.pack(SCAN_HEADER_FMT, SCAN_MAGIC, outcome.width,
                         outcome.height, outcome.steps, outcome.n_blobs,
                         flags)
    seeds_bytes = outcome.seeds.astype("<u4").tobytes()
    payload = (header + outcome.sweep_u16.tobytes()
               + outcome.track_u16.tobytes() + seeds_bytes)
    phase = ",".join(f"{k}:{v:.3f}" for k, v in outcome.phase_ms.items())
    headers = {
        "X-Filled": str(outcome.filled),
        "X-Runs": str(outcome.n_runs),
        "X-Unions": str(outcome.unions),
        "X-Kernel-Ms": f"{outcome.kernel_ms:.3f}",
        "X-Total-Ms": f"{outcome.total_ms:.3f}",
        "X-Njit-Ms": f"{njit_ms:.3f}",
        "X-Phase-Ms": phase,
        # 1 = first launch after an idle GPU, so X-Kernel-Ms is an
        # over-estimate (this laptop drops to ~700 MHz of 3105 and does
        # not spin up for millisecond kernels). Reported rather than
        # papered over with a keep-warm loop.
        "X-Cold": "1" if outcome.cold else "0",
    }
    if outcome.amplified_filled:
        headers["X-Amplified-Filled"] = str(outcome.amplified_filled)
        headers["X-Amplified-Runs"] = str(outcome.amplified_runs)
        headers["X-Amplified-Kernel-Ms"] = f"{outcome.amplified_kernel_ms:.3f}"
    return Response(content=payload,
                    media_type="application/octet-stream", headers=headers)


def _challenge_response(challenge):
    png = survey.field_png()
    return Response(
        content=png, media_type="image/png",
        headers={
            "X-Challenge-Id": challenge["id"],
            "X-Width": str(challenge["width"]),
            "X-Height": str(challenge["height"]),
            "X-Difficulty": challenge["difficulty"],
            "Cache-Control": "no-store",
        })


@app.get("/api/challenge")
async def get_challenge():
    """DEEP FIELD: the current shared star field as a PNG (no truth in the
    response). Generates one on first call."""
    loop = asyncio.get_running_loop()
    challenge = await loop.run_in_executor(_gpu_executor,
                                           survey.ensure_challenge)
    return _challenge_response(challenge)


@app.post("/api/challenge/new")
async def post_challenge_new(request: Request):
    """Roll a fresh field for everyone; resets the leaderboard to it."""
    difficulty = request.query_params.get("difficulty",
                                          survey.DEFAULT_DIFFICULTY)
    if difficulty not in survey.DIFFICULTIES:
        return JSONResponse(
            {"detail": f"difficulty must be one of "
             f"{list(survey.DIFFICULTIES)}"}, status_code=400)
    loop = asyncio.get_running_loop()
    challenge = await loop.run_in_executor(
        _gpu_executor, survey.new_challenge, difficulty)
    return _challenge_response(challenge)


@app.post("/api/guess")
async def post_guess(request: Request):
    """Score a guess against the current field's true count; returns the
    reveal (truth, your error, your rank) and the updated board."""
    try:
        body = await request.json()
        name = body.get("name", "")
        guess = body["guess"]
    except (ValueError, KeyError, TypeError):
        return JSONResponse({"detail": "body must be JSON with a 'guess'"},
                            status_code=400)
    try:
        result = survey.submit_guess(name, guess)
    except (ValueError, TypeError) as e:
        return JSONResponse({"detail": str(e)}, status_code=400)
    return JSONResponse(result)


@app.get("/api/leaderboard")
async def get_leaderboard():
    return JSONResponse(survey.leaderboard())


# Mounted LAST: routes registered above (/api/fill, /api/scan, the survey
# routes, /healthz) win over this catch-all, and html=True serves
# static/index.html at "/" with no separate redirect route needed.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
