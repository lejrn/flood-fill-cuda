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

STATIC_DIR = Path(__file__).parent / "static"

MAGIC = 0x46494C4C
HEADER_FMT = "<IIII"   # magic, width, height, levels
HEADER_SIZE = struct.calcsize(HEADER_FMT)

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


@app.post("/api/fill")
async def fill(request: Request):
    global _inflight

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
        outcome = await loop.run_in_executor(_gpu_executor, engine.run_fill,
                                             mask)
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
            "X-Kernel-Ms": f"{outcome.kernel_ms:.3f}",
            "X-Total-Ms": f"{outcome.total_ms:.3f}",
        },
    )


# Mounted LAST: routes registered above (/api/fill, /healthz) win over this
# catch-all, and html=True serves static/index.html at "/" with no
# separate redirect route needed.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
