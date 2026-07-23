# Paint-and-fill web service

A webpage where anyone can paint a thick blob with the mouse (or a finger);
on release, ch03's single-seed cooperative GPU kernel floods it — seeded at
the blob's center of mass — and the browser replays the frontier spreading
outward from the depth timeline the kernel returns, then the finished shape
falls off-screen like a feather.

No websockets, no frame streaming: the kernel runs the *entire* BFS in one
cooperative launch and returns a per-pixel `depth` array (the level each
pixel was filled at). The server ships that array once; the browser
animates it locally by thresholding `depth <= t`, the same trick
[`ch04/benchmarks/wavefront.py`](../chapters/ch04_gpu_2blob_nblock/benchmarks/wavefront.py)
uses to render its GIFs.

## Run (Phase 1 — local)

```bash
uv sync
uv run python -m flood_fill_cuda.service
# or, equivalently:
uv run uvicorn flood_fill_cuda.service.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/`. First request may lag a second or two if
the lifespan warmup hasn't finished JIT-compiling the kernel yet; `GET
/healthz` reports `"warm": true` once it has.

**Never run with `--workers > 1` or behind a multi-process manager.**
Concurrent cooperative-kernel launches nondeterministically wedge under
WSL2 (`chapters/README.md`'s "Finding 2" — even an 8+8 block pair hung mid
test suite once). `app.py`'s single-worker `ThreadPoolExecutor` makes
concurrent launches structurally impossible *within* one process; extra
worker processes each get their own executor and recreate the hazard.
Likewise, don't run a chapter's benchmark suite at the same time as the
service — that's a second, independent source of concurrent cooperative
launches on the same GPU.

## Test

```bash
uv run pytest src/flood_fill_cuda/service/ -v
```

`test_engine.py` exercises the real GPU directly (seed placement,
orientation, depth math against the CPU oracle). `test_api.py` drives the
same GPU through a `TestClient` (wire format, error-code mapping).

Manual smoke test:

```bash
curl localhost:8000/healthz
curl -X POST --data-binary @stroke.png localhost:8000/api/fill -D- -o depth.bin
```

## API

`POST /api/fill` — body is a PNG (the painted stroke's bounding-box crop,
RGBA; alpha ≥ 128 is "painted"). Response is `application/octet-stream`:
a 16-byte header (`magic, width, height, levels`, all `uint32` LE) followed
by `width*height` `uint16` LE values, row-major: `0` = not part of the
blob, else `min(depth+1, 65535)`. See `app.py`'s module docstring for the
exact byte layout and the reasoning for this framing over JSON.

`GET /healthz` → `{"status", "warm", "device"}`.

## Phase 2 — public via Cloudflare Tunnel

No code changes; run the service locally (above), then in a second
terminal:

```bash
cloudflared tunnel --url http://127.0.0.1:8000
```

That prints an ephemeral `https://*.trycloudflare.com` URL — share it, it
proxies straight to your local GPU. For a stable hostname instead of a
random one each run, use a named tunnel:

```bash
cloudflared tunnel login
cloudflared tunnel create flood-fill
cloudflared tunnel route dns flood-fill flood-fill.yourdomain.com
cloudflared tunnel run --url http://127.0.0.1:8000 flood-fill
```

The service's only abuse guard right now is the 503 backlog cap
(`app.py`'s `BACKLOG_CAP`) — treat quick-tunnel URLs as ephemeral / share
narrowly until something stronger (rate limiting, auth) is added.

## Phase 3 — future: cloud GPU (not built yet)

Notes for whoever picks this up — no Dockerfile exists yet on purpose
(an untested one just rots):

- Needs a GPU with cooperative-launch support — T4 / A10G / L4 are all
  fine; the WSL2 wedge finding is WSL2-specific, but keep the
  one-concurrent-fill-per-GPU discipline regardless (cooperative launches
  are a shared hazard class, not a WSL2-only one).
- Set `NUMBA_CUDA_USE_NVIDIA_BINDING=1` in the container image/env — see
  `engine.py`'s and `kernels.py`'s top-of-file comment on why.
- Wire the lifespan warmup to a container readiness probe on `/healthz`
  (`"warm": true`) so orchestrators don't route traffic before the JIT
  compile finishes.
- Candidate targets: Modal (`@app.cls(gpu=...)` + `@modal.enter` warmup,
  serverless, scales to zero) or a Hugging Face Space on the Docker SDK
  with a GPU runtime.
