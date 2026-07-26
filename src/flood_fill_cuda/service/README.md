# Paint-and-fill web service

A webpage where anyone can paint a thick blob with the mouse (or a finger)
using one of a few brush shapes; on release, the fill spreads from wherever
the pointer was let go, computed on either ch03's single-seed cooperative
GPU kernel or ch01's sequential CPU oracle (the frontend's CPU/GPU toggle).
Every fill actually runs twice: once on the blob exactly as painted (what's
drawn on screen — the visual size never changes), and once on the same
shape amplified to ~250x the pixel count (see `engine.py`'s module
docstring), purely so the reported timing and pixel counts — and the
animation's real-time pacing — are honest at the scale where the GPU's
advantage actually shows up; a Full HD brush stroke alone never gets close.
The browser replays the frontier as a brightness wave spreading outward
from the depth timeline the engine returns — brightest at the leading
edge, cooling to a dark, saturated resting color behind it — paced to the
engine's own real elapsed compute time at that amplified scale, not a
stylized pace. The finished shape fades out over about a second.

No websockets, no frame streaming: the engine runs the *entire* BFS in one
call and returns a per-pixel `depth` array (the level each pixel was filled
at). The server ships that array once; the browser animates it locally by
thresholding `depth <= t`, the same trick
[`ch04/benchmarks/wavefront.py`](../chapters/ch04_gpu_2blob_nblock/benchmarks/wavefront.py)
uses to render its GIFs.

A third mode, **RUNS**, is a different job on a different kernel. It takes
no seed at all: the stroke goes to `/api/scan`, and **ch06** finds and
recolours *every* blob in it at once, each in its own canonical-label
colour. There is no wavefront to replay — ch06 is not a BFS and no pixel
has a level — so the reveal is a left-to-right scan bar following the
kernel's own row-major scan order, and the readout reports what ch06
actually counts: blobs and runs. See the `/api/scan` notes below for why
the wire format changed with it.

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
launches on the same GPU. CPU-mode fills (`?mode=cpu`) never touch CUDA, so
they run on their own small executor and aren't subject to any of this.

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

`POST /api/fill?mode=cpu|gpu&seed_x=<num>&seed_y=<num>` — body is a PNG
(the painted stroke's bounding-box crop, RGBA; alpha ≥ 128 is "painted").
`mode` defaults to `gpu`; `seed_x`/`seed_y` (crop-local, typically where the
pointer was released) default to the mask's center of mass if omitted —
either way the seed is snapped to the nearest actual painted pixel. Response
is `application/octet-stream`: a 16-byte header (`magic, width, height,
levels`, all `uint32` LE) followed by `width*height` `uint16` LE values,
row-major: `0` = not part of the blob, else `min(depth+1, 65535)`. See
`app.py`'s module docstring for the exact byte layout and the reasoning for
this framing over JSON. `X-Mode` on the response echoes back which engine
actually ran. `X-Filled` is the real, painted-mask pixel count (matches the
returned depth array exactly); `X-Amplified-Filled`, `X-Kernel-Ms`, and
`X-Total-Ms` report the *amplified*-scale run instead — see `engine.py`'s
module docstring for why the response mixes real-size depth data with
amplified-scale timing.

`POST /api/scan?amp=0|1` — the seedless endpoint, backing both DEEP
FIELD and the paint page's RUNS mode: multi-blob discovery over the
WHOLE canvas via **ch06's run-table connected components** (no seeds, no
mode — that is the point). Body: RGBA PNG, alpha ≥ 128 is "painted"; an
empty canvas is valid (`n_blobs=0`). Response format v3 (`app.py`'s
docstring has the layout): a 24-byte header (`magic "SCAN", width,
height, steps, n_blobs, flags`), then `sweep` uint16[w·h], `track`
uint16[w·h] (dense per-blob label ids in canonical order, 0 =
background), then `n_blobs` uint32 (x, y) pairs — the GPU-chosen
canonical seeds.

`?amp=1` additionally times the same shape upscaled ~250×, so a small
painted stroke can report a kernel time that isn't dominated by six
launch overheads; it adds `X-Amplified-Filled`, `X-Amplified-Runs` and
`X-Amplified-Kernel-Ms` and sets bit 0 of `flags`. The payload itself is
always the real-size scan.

Headers: `X-Kernel-Ms`, `X-Total-Ms`, `X-Njit-Ms` (the @njit seedless
reference, run concurrently on the CPU pool — the honest race bar),
`X-Filled`, `X-Runs`, `X-Unions`, `X-Phase-Ms`, `X-Cold`.

**What changed when ch06 replaced ch05 here (v2 → v3):** `depth` became
`sweep`, and `prov` is gone. ch06 is not a BFS, so no pixel has a level
and there is no provisional-label stage to replay — the run merge is one
data-independent pass. `sweep` is the order the kernel *scans* in
(row-major, which is left-to-right on the canvas), used to drive the
reveal; it is a spatial ordering, not a timeline, and the code says so
in both places it appears. Canonical labels, `track` ids and `seeds` are
bit-for-bit what ch05 produced — the same CPU oracle still judges them
in `test_scan.py`, which is exactly why the swap was safe.

`X-Cold: 1` marks the first launch after an idle GPU. This laptop drops
to ~700 MHz of 3105 and will not spin up for a millisecond kernel, so
that first scan reads several times high (measured through this
endpoint: 10.1 ms against 2.1 ms back-to-back for the same amplified
run). Chapter 6 hit the same effect on the bench and solved it there by
spinning the clock up before timing; a web service cannot honestly burn
the GPU to flatter its own number, so it reports the condition instead
and the UI prints "(cold clock)".

`GET /healthz` → `{"status", "warm", "device"}`.

`GET /api/challenge` → the current shared star field as a PNG (headers
`X-Challenge-Id`, `X-Width`, `X-Height`, `X-Difficulty`; no truth in the
response). `POST /api/challenge/new?difficulty=scout|surveyor|deepfield`
rolls a fresh field and resets the board. `POST /api/guess` (JSON
`{name, guess}`) scores against the current field's TRUE count — which
is `engine.discover(...).n_blobs`, the real kernel on the field, not the
generator's placed count — and returns `{true_count, n_placed,
your_error, your_rank, players, leaderboard}`. `GET /api/leaderboard` →
the current board (names, guesses, errors) and player count. State
persists to `results/service/state/survey_state.json`.

## DEEP FIELD — the star-count game

`http://127.0.0.1:8000/skysurvey.html` — a shared, generated star field
(varied sizes, some touching). Estimate the count, press SURVEY: the
real ch06 scan sweeps the frame, every star lights in its
own colour, a counter spins up to the true number, and your guess climbs
a persistent leaderboard of names. The honest twist: stars that touch
merge into one component, so the true count is below the number placed —
the exact mistake a human eye makes. Astronomy source extraction
(SExtractor is literally threshold → connected-component labelling →
measure) turned into a party game.

## SHOOT — the splatter tool

Pick the **✳** tool on the paint page. Click to fire a shot: a shotgun
scatter of hundreds-to-thousands of small drops at the cursor, with a
minority flung wide. Hold and drag to keep spraying. Pick drops/shot from
the toolbar (200 … 20,000); **CLEAR** wipes the canvas and the counters.

The canvas is deliberately **never cleared between shots**, so drops pile
up and start touching — and the blob count does something worth watching.
Measured on a 1600×900 canvas:

| drops fired | painted px | blobs found | merged away | runs | GPU kernel | @njit |
|---|---|---|---|---|---|---|
| 500 | 14k | 311 | 189 | 2.8k | 0.68 ms | 3.9 ms |
| 5,000 | 137k | 2,761 | 2,239 | 25k | 0.86 ms | 9.6 ms |
| 10,000 | 258k | 4,068 | 5,932 | 44k | 0.68 ms | 20.9 ms |
| 20,000 | 499k | 4,630 | 15,370 | 74k | 0.88 ms | 22.1 ms |
| 50,000 | 893k | **1,670** | 48,330 | 92k | 0.77 ms | 36.4 ms |

Two things the tool exists to show. **The kernel time does not move** —
0.7–0.9 ms whether you fire 500 drops or 50,000; only the CPU reference
climbs, so the speedup grows from 5.8× to 47×. And the blob count
**peaks around 20k drops and then collapses** to 1,670 at 50k: that is
percolation, live — separate drops fusing into continents faster than new
ones can land.

Each shot re-scans the WHOLE canvas through `/api/scan?amp=0` (a full
canvas is ~1.4 Mpx of real work, so it needs no amplification to report
an honest number) and recolours it in place, blob by blob, following the
kernel's scan order. The reveal is a fixed-length slow-motion replay and
says so: the real kernel finishes in well under one frame, so pacing it
truthfully would mean showing nothing at all. The HUD prints the true
kernel time next to the round trip, which is the more interesting pair —
the GPU is around 1% of the wall clock, and everything else is PNG
decode, numpy glue and transfer.

**On that round trip:** it was 806 ms per shot at 20k drops until two
fixes landed, neither of them in a kernel. `GZipMiddleware` defaults to
`compresslevel=9`, which spent **677 ms** compressing 5.76 MB of
mostly-zeros (level 1: 24 ms, for 682 KB against 436 KB). And the wire
format was shipping a per-pixel scan-order field that is a pure function
of the column — 2.88 MB per shot of data the client can compute in a
loop. Both fixed; 20k drops is now ~120 ms end to end.

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
