// SKYWATCH: paint a raid -> POST /api/scan -> the REAL ch05 cooperative
// kernel discovers, labels and fills every contact, no seeds given ->
// acquisition replay, lock-on boxes, engage phase, and an honest
// GPU-vs-@njit race measured on this very scan. No build step, no deps.
(() => {
  "use strict";

  const stage = document.getElementById("stage");
  const paint = document.getElementById("paint");
  const fx = document.getElementById("fx");
  const pctx = paint.getContext("2d", { willReadFrequently: true });
  const fctx = fx.getContext("2d");

  const statusEl = document.getElementById("hud-status");
  const scoreEl = document.getElementById("hud-score");
  const statLine = document.getElementById("stat-line");
  const phaseBar = document.getElementById("phase-bar");
  const banner = document.getElementById("wave-banner");
  const scanBtn = document.getElementById("scan-btn");
  const racePanel = document.getElementById("race-panel");
  const raceTog = document.getElementById("race-tog");
  const mergeTog = document.getElementById("merge-tog");

  const BRUSH = 44;
  const BRUSH_STEP = 10;
  const THREAT = "rgb(255, 74, 60)";
  const MAX_BACKING_PIXELS = 2.0e6;   // njit race stays sub-second
  const MIN_REPLAY_MS = 900;          // a 2 ms kernel is invisible at 1:1
  const GOLDEN = 137.508;

  let state = "arm";                  // arm | scanning | replay | engage
  let painting = false;
  let lastX = 0, lastY = 0;
  let currentShape = "circle";
  let score = 0;

  // last scan, for hit-testing and pixel erasure during ENGAGE
  let scan = null;   // {width,height,depth,track,seeds,tracks,order,...}
  let engageTimer = null;

  // ---- canvas sizing --------------------------------------------------
  function sizeCanvas() {
    const rect = stage.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    let w = Math.round(rect.width * dpr);
    let h = Math.round(rect.height * dpr);
    if (w * h > MAX_BACKING_PIXELS) {
      const s = Math.sqrt(MAX_BACKING_PIXELS / (w * h));
      w = Math.max(1, Math.round(w * s));
      h = Math.max(1, Math.round(h * s));
    }
    for (const c of [paint, fx]) {
      c.width = w; c.height = h;
      c.style.width = rect.width + "px";
      c.style.height = rect.height + "px";
    }
    pctx.fillStyle = THREAT;
    pctx.strokeStyle = THREAT;
    pctx.lineCap = "round";
    endEngage(false);
  }
  window.addEventListener("resize", sizeCanvas);
  sizeCanvas();

  function toCanvasXY(e) {
    const rect = paint.getBoundingClientRect();
    return [
      (e.clientX - rect.left) * paint.width / rect.width,
      (e.clientY - rect.top) * paint.height / rect.height,
    ];
  }

  // ---- brushes (ported from paint.js; eraser splits blobs) ------------
  function stampCircle(x, y) {
    pctx.beginPath(); pctx.arc(x, y, BRUSH / 2, 0, Math.PI * 2); pctx.fill();
  }
  function stampSquare(x, y) {
    const s = BRUSH * 0.86;
    pctx.save(); pctx.translate(x, y);
    pctx.rotate((Math.random() - 0.5) * 0.5);
    pctx.fillRect(-s / 2, -s / 2, s, s); pctx.restore();
  }
  function stampStar(x, y) {
    const outer = BRUSH * 0.62, inner = outer * 0.42;
    pctx.save(); pctx.translate(x, y);
    pctx.rotate(Math.random() * Math.PI * 2);
    pctx.beginPath();
    for (let i = 0; i < 10; i++) {
      const r = i % 2 === 0 ? outer : inner;
      const a = (Math.PI / 5) * i - Math.PI / 2;
      const px = Math.cos(a) * r, py = Math.sin(a) * r;
      if (i === 0) pctx.moveTo(px, py); else pctx.lineTo(px, py);
    }
    pctx.closePath(); pctx.fill(); pctx.restore();
  }
  function stampScratchy(x, y) {
    const r = BRUSH / 2, prev = pctx.lineWidth;
    pctx.lineWidth = Math.max(3, BRUSH * 0.16);
    for (let i = 0; i < 6; i++) {
      const a = Math.random() * Math.PI * 2;
      const len = r * (0.6 + Math.random() * 0.7);
      const cx = x + (Math.random() * 2 - 1) * r * 0.35;
      const cy = y + (Math.random() * 2 - 1) * r * 0.35;
      pctx.beginPath();
      pctx.moveTo(cx - Math.cos(a) * len / 2, cy - Math.sin(a) * len / 2);
      pctx.lineTo(cx + Math.cos(a) * len / 2, cy + Math.sin(a) * len / 2);
      pctx.stroke();
    }
    pctx.lineWidth = prev;
  }
  function stampEraser(x, y) {
    pctx.save();
    pctx.globalCompositeOperation = "destination-out";
    pctx.beginPath(); pctx.arc(x, y, BRUSH * 0.45, 0, Math.PI * 2);
    pctx.fill(); pctx.restore();
  }
  const BRUSHES = { circle: stampCircle, square: stampSquare,
                    star: stampStar, scratchy: stampScratchy,
                    eraser: stampEraser };

  function strokeSegment(x0, y0, x1, y1) {
    const dist = Math.hypot(x1 - x0, y1 - y0);
    const steps = Math.max(1, Math.ceil(dist / BRUSH_STEP));
    for (let i = 1; i <= steps; i++) {
      const t = i / steps;
      BRUSHES[currentShape](x0 + (x1 - x0) * t, y0 + (y1 - y0) * t);
    }
  }

  document.querySelectorAll("#toolbar .brush-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#toolbar .brush-btn")
        .forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      currentShape = btn.dataset.shape;
    });
  });

  paint.addEventListener("contextmenu", (e) => e.preventDefault());
  paint.addEventListener("pointerdown", (e) => {
    if (state !== "arm") return;
    paint.setPointerCapture(e.pointerId);
    painting = true;
    const [x, y] = toCanvasXY(e);
    lastX = x; lastY = y;
    BRUSHES[currentShape](x, y);
  });
  paint.addEventListener("pointermove", (e) => {
    if (!painting) return;
    const [x, y] = toCanvasXY(e);
    strokeSegment(lastX, lastY, x, y);
    lastX = x; lastY = y;
  });
  const stopPaint = () => { painting = false; };
  paint.addEventListener("pointerup", stopPaint);
  paint.addEventListener("pointercancel", stopPaint);

  // ---- raid generator (procedural, but still genuine kernel input) ----
  document.getElementById("raid-btn").addEventListener("click", () => {
    if (state !== "arm") return;
    const shapes = ["circle", "square", "star", "scratchy"];
    const keep = currentShape;
    const bursts = 6 + Math.floor(Math.random() * 9);
    for (let i = 0; i < bursts; i++) {
      currentShape = shapes[Math.floor(Math.random() * shapes.length)];
      let x = paint.width * (0.06 + Math.random() * 0.88);
      let y = paint.height * (0.06 + Math.random() * 0.88);
      BRUSHES[currentShape](x, y);
      const drags = Math.floor(Math.random() * 4);
      for (let d = 0; d < drags; d++) {
        const nx = x + (Math.random() - 0.5) * BRUSH * 3;
        const ny = y + (Math.random() - 0.5) * BRUSH * 3;
        strokeSegment(x, y, nx, ny);
        x = nx; y = ny;
      }
    }
    currentShape = keep;
  });

  document.getElementById("clear-btn").addEventListener("click", () => {
    if (state !== "arm") return;
    pctx.clearRect(0, 0, paint.width, paint.height);
    fctx.clearRect(0, 0, fx.width, fx.height);
    statLine.textContent = "scope clear";
  });

  // ---- scan ------------------------------------------------------------
  scanBtn.addEventListener("click", () => {
    if (state === "arm") doScan();
  });

  async function requestScan() {
    const blob = await new Promise((r) => paint.toBlob(r, "image/png"));
    const prov = mergeTog.checked ? 1 : 0;
    const resp = await fetch(`/api/scan?prov=${prov}`,
                             { method: "POST", body: blob });
    if (!resp.ok) {
      const detail = await resp.json().catch(() => ({}));
      throw new Error(`scan failed (${resp.status}): `
                      + (detail.detail || resp.statusText));
    }
    const buf = await resp.arrayBuffer();
    const dv = new DataView(buf);
    const width = dv.getUint32(4, true);
    const height = dv.getUint32(8, true);
    const levels = dv.getUint32(12, true);
    const nBlobs = dv.getUint32(16, true);
    const flags = dv.getUint32(20, true);
    const n = width * height;
    let off = 24;
    const depth = new Uint16Array(buf, off, n); off += 2 * n;
    const track = new Uint16Array(buf, off, n); off += 2 * n;
    const seeds = new Uint32Array(buf, off, 2 * nBlobs); off += 8 * nBlobs;
    const prov16 = (flags & 1) ? new Uint16Array(buf, off, n) : null;
    const ph = {};
    for (const part of (resp.headers.get("x-phase-ms") || "").split(",")) {
      const [k, v] = part.split(":");
      if (k) ph[k] = parseFloat(v) || 0;
    }
    return {
      width, height, levels, nBlobs, depth, track, seeds, prov: prov16,
      filled: parseInt(resp.headers.get("x-filled"), 10) || 0,
      candidates: parseInt(resp.headers.get("x-candidates"), 10) || 0,
      unions: parseInt(resp.headers.get("x-unions"), 10) || 0,
      kernelMs: parseFloat(resp.headers.get("x-kernel-ms")) || 0,
      njitMs: parseFloat(resp.headers.get("x-njit-ms")) || 0,
      phaseMs: ph,
    };
  }

  // One pass over the track map: bbox, size, centroid per track id.
  function trackInfo(s) {
    const t = [];
    for (let id = 1; id <= s.nBlobs; id++) {
      t.push({ id, size: 0, minX: 1e9, minY: 1e9, maxX: -1, maxY: -1,
               sx: 0, sy: 0, alive: true });
    }
    const { track, width } = s;
    for (let i = 0; i < track.length; i++) {
      const id = track[i];
      if (!id) continue;
      const rec = t[id - 1];
      const x = i % width, y = (i / width) | 0;
      rec.size++;
      if (x < rec.minX) rec.minX = x;
      if (x > rec.maxX) rec.maxX = x;
      if (y < rec.minY) rec.minY = y;
      if (y > rec.maxY) rec.maxY = y;
      rec.sx += x; rec.sy += y;
    }
    for (const rec of t) {
      rec.cx = rec.size ? rec.sx / rec.size : 0;
      rec.cy = rec.size ? rec.sy / rec.size : 0;
    }
    return t;
  }

  const hue = (id) => (id * GOLDEN) % 360;

  async function doScan() {
    state = "scanning";
    scanBtn.disabled = true;
    statusEl.textContent = "SCANNING — one cooperative launch, zero seeds";
    fctx.clearRect(0, 0, fx.width, fx.height);
    banner.hidden = true;
    try {
      const s = await requestScan();
      scan = s;
      s.tracks = trackInfo(s);
      s.order = s.tracks.slice().sort((a, b) => b.size - a.size);

      showStats(s);
      showPhases(s.phaseMs);
      if (raceTog.checked && s.filled > 0) runRace(s);

      if (s.nBlobs === 0) {
        statusEl.textContent = "0 tracks — clear sky. ARM again.";
        state = "arm"; scanBtn.disabled = false;
        return;
      }
      state = "replay";
      await acquisitionReplay(s);
      drawLockOns(s);
      startEngage(s);
    } catch (err) {
      console.error(err);
      statusEl.textContent = String(err.message || err);
      state = "arm"; scanBtn.disabled = false;
    }
  }

  function showStats(s) {
    const slow = Math.max(1, MIN_REPLAY_MS / Math.max(s.kernelMs, 0.001));
    statLine.textContent =
      `${s.nBlobs.toLocaleString()} tracks · acquisition ` +
      `${s.kernelMs.toFixed(2)} ms (kernel) · ${s.candidates} candidates · ` +
      `${s.unions} unions · ${s.filled.toLocaleString()} px · replay ` +
      `slowed ×${slow.toFixed(0)}`;
  }

  function showPhases(ph) {
    const order = ["init", "scan", "fill", "flatten"];
    const total = order.reduce((a, k) => a + (ph[k] || 0), 0);
    phaseBar.innerHTML = "";
    if (!total) { phaseBar.hidden = true; return; }
    for (const k of order) {
      if (!ph[k]) continue;
      const seg = document.createElement("div");
      seg.className = "ph-" + k;
      seg.style.width = (100 * ph[k] / total) + "%";
      seg.title = `${k}: ${ph[k].toFixed(2)} ms`;
      phaseBar.appendChild(seg);
    }
    phaseBar.hidden = false;
  }

  // ---- acquisition replay ---------------------------------------------
  // Level-bucketed wavefront reveal over the fx overlay (the paint.js
  // replay, full-canvas): each pixel settles to its track's hue the
  // level it was filled at; a bright leading edge glides ahead. If the
  // merge toggle was on, the settle hues are the PROVISIONAL labels,
  // then snap to canonical at the end — the atomicMin merge, visible.
  function bucketByLevel(depth, levels) {
    const buckets = Array.from({ length: levels }, () => []);
    for (let i = 0; i < depth.length; i++) {
      const v = depth[i];
      if (v > 0) buckets[v - 1].push(i);
    }
    return buckets;
  }

  function hslToRgb(h, s, l) {
    h = (((h % 360) + 360) % 360) / 360;
    const k = (n) => (n + h * 12) % 12;
    const a = s * Math.min(l, 1 - l);
    const f = (n) =>
      l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
    return [Math.round(f(0) * 255), Math.round(f(8) * 255),
            Math.round(f(4) * 255)];
  }

  function paintIds(out, s, ids, lightness) {
    // repaint every filled pixel from an id map (track or prov)
    const palette = new Map();
    for (let i = 0; i < ids.length; i++) {
      const id = ids[i];
      if (!id) continue;
      let c = palette.get(id);
      if (!c) { c = hslToRgb(hue(id), 0.75, lightness); palette.set(id, c); }
      const p = i * 4;
      out.data[p] = c[0]; out.data[p + 1] = c[1];
      out.data[p + 2] = c[2]; out.data[p + 3] = 235;
    }
  }

  function acquisitionReplay(s) {
    return new Promise((resolve) => {
      const buckets = bucketByLevel(s.depth, s.levels);
      const out = fctx.createImageData(s.width, s.height);
      const ids = s.prov || s.track;
      const palette = new Map();
      const colorOf = (id) => {
        let c = palette.get(id);
        if (!c) { c = hslToRgb(hue(id), 0.75, 0.42); palette.set(id, c); }
        return c;
      };
      const duration = Math.max(s.kernelMs, MIN_REPLAY_MS);
      let start = null, prevLevel = -1;

      function frame(ts) {
        if (start === null) start = ts;
        const elapsed = ts - start;
        let t = Math.floor((elapsed / duration) * s.levels);
        if (t > s.levels - 1) t = s.levels - 1;
        for (let lvl = prevLevel + 1; lvl <= t; lvl++) {
          for (const i of buckets[lvl]) {
            const c = colorOf(ids[i]);
            const p = i * 4;
            out.data[p] = c[0]; out.data[p + 1] = c[1];
            out.data[p + 2] = c[2]; out.data[p + 3] = 235;
          }
        }
        const frontier = t + 1;
        if (frontier < s.levels) {
          for (const i of buckets[frontier]) {
            const p = i * 4;
            out.data[p] = 220; out.data[p + 1] = 255;
            out.data[p + 2] = 230; out.data[p + 3] = 255;
          }
        }
        fctx.putImageData(out, 0, 0);
        prevLevel = t;
        if (elapsed < duration) { requestAnimationFrame(frame); return; }
        if (s.prov) {
          // the merge, made visible: snap provisional hues -> canonical
          setTimeout(() => {
            paintIds(out, s, s.track, 0.42);
            fctx.putImageData(out, 0, 0);
            resolve();
          }, 350);
        } else resolve();
      }
      requestAnimationFrame(frame);
    });
  }

  // ---- lock-ons + engage ----------------------------------------------
  function drawLockOns(s) {
    fctx.save();
    fctx.font = "11px ui-monospace, Menlo, Consolas, monospace";
    fctx.textBaseline = "bottom";
    s.order.forEach((rec, rank) => {
      if (!rec.alive) return;
      const pad = 4;
      fctx.strokeStyle = rank === 0 ? "#f0b429" : "rgba(70,224,138,0.9)";
      fctx.lineWidth = rank === 0 ? 2 : 1;
      fctx.strokeRect(rec.minX - pad, rec.minY - pad,
                      rec.maxX - rec.minX + 2 * pad,
                      rec.maxY - rec.minY + 2 * pad);
      fctx.fillStyle = fctx.strokeStyle;
      fctx.fillText(`#${rank + 1} · ${rec.size}px`,
                    rec.minX - pad, rec.minY - pad - 2);
      // the GPU-chosen canonical seed
      const sx = s.seeds[(rec.id - 1) * 2], sy = s.seeds[(rec.id - 1) * 2 + 1];
      fctx.beginPath();
      fctx.moveTo(sx - 6, sy); fctx.lineTo(sx + 6, sy);
      fctx.moveTo(sx, sy - 6); fctx.lineTo(sx, sy + 6);
      fctx.stroke();
    });
    fctx.restore();
  }

  function redrawEngage(s) {
    const out = fctx.createImageData(s.width, s.height);
    const alive = new Set(s.order.filter((r) => r.alive).map((r) => r.id));
    const idsAlive = new Uint16Array(s.track.length);
    for (let i = 0; i < s.track.length; i++) {
      if (alive.has(s.track[i])) idsAlive[i] = s.track[i];
    }
    paintIds(out, s, idsAlive, 0.42);
    fctx.putImageData(out, 0, 0);
    s.order = s.order.filter((r) => r.alive)
      .concat(s.order.filter((r) => !r.alive));
    drawLockOns(s);
  }

  function startEngage(s) {
    state = "engage";
    stage.classList.add("engaging");
    const total = Math.min(25000, 4000 + 600 * s.nBlobs);
    const deadline = performance.now() + total;
    statusEl.textContent =
      `ENGAGE — destroy tracks in priority order (#1 first). ` +
      `${(total / 1000).toFixed(0)}s`;
    engageTimer = setInterval(() => {
      const left = deadline - performance.now();
      if (left <= 0) { endEngage(true); return; }
      statusEl.textContent =
        `ENGAGE — priority order. ${(left / 1000).toFixed(1)}s · ` +
        `${s.order.filter((r) => r.alive).length} tracks left`;
    }, 100);
  }

  fx.addEventListener("pointerdown", (e) => {
    if (state !== "engage" || !scan) return;
    const rect = fx.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) * fx.width / rect.width);
    const y = Math.floor((e.clientY - rect.top) * fx.height / rect.height);
    const id = scan.track[y * scan.width + x];
    if (!id) return;
    const rec = scan.tracks[id - 1];
    if (!rec.alive) return;
    const aliveOrder = scan.order.filter((r) => r.alive);
    if (rec === aliveOrder[0]) {
      score += 100 + Math.round(rec.size / 50);
      destroyTrack(scan, rec);
      if (!scan.order.some((r) => r.alive)) endEngage(true);
      else redrawEngage(scan);
    } else {
      score = Math.max(0, score - 150);
      statusEl.textContent = "ENGAGE — wrong priority! #1 first.";
    }
    scoreEl.textContent = `SCORE ${score.toLocaleString()}`;
  });

  function destroyTrack(s, rec) {
    rec.alive = false;
    // erase the track's pixels from the paint canvas via the track map
    const bx = rec.minX, by = rec.minY;
    const bw = rec.maxX - rec.minX + 1, bh = rec.maxY - rec.minY + 1;
    const im = pctx.getImageData(bx, by, bw, bh);
    for (let y = 0; y < bh; y++) {
      for (let x = 0; x < bw; x++) {
        if (s.track[(by + y) * s.width + (bx + x)] === rec.id) {
          im.data[(y * bw + x) * 4 + 3] = 0;
        }
      }
    }
    pctx.putImageData(im, bx, by);
  }

  function endEngage(showBanner) {
    if (engageTimer) { clearInterval(engageTimer); engageTimer = null; }
    stage.classList.remove("engaging");
    if (showBanner && scan) {
      const left = scan.order.filter((r) => r.alive).length;
      banner.textContent = left === 0
        ? `WAVE CLEAR\nSCORE ${score.toLocaleString()}`
        : `TIME UP — ${left} LEAKERS\nSCORE ${score.toLocaleString()}`;
      banner.hidden = false;
      setTimeout(() => { banner.hidden = true; }, 2600);
      fctx.clearRect(0, 0, fx.width, fx.height);
      statusEl.textContent = left === 0
        ? "ARM — wave clear. Paint the next raid."
        : "ARM — leakers still on scope. Reinforce and SCAN again.";
    }
    state = "arm";
    scanBtn.disabled = false;
  }

  // ---- the race --------------------------------------------------------
  function runRace(s) {
    racePanel.hidden = false;
    const gpuFill = document.getElementById("race-gpu");
    const cpuFill = document.getElementById("race-cpu");
    const gpuMs = document.getElementById("race-gpu-ms");
    const cpuMs = document.getElementById("race-cpu-ms");
    // both bars run in REAL time: the GPU one completes in kernelMs
    // (often a single frame — that is the point), the CPU one crawls
    // for the njit reference's actual wall time on this same canvas.
    for (const [el, ms, lab] of [[gpuFill, s.kernelMs, gpuMs],
                                 [cpuFill, s.njitMs, cpuMs]]) {
      el.style.transition = "none";
      el.style.width = "0%";
      void el.offsetWidth;                    // reflow: restart transition
      el.style.transition = `width ${Math.max(ms, 16) / 1000}s linear`;
      el.style.width = "100%";
      lab.textContent = "";
      setTimeout(() => {
        let note = "";
        if (el === cpuFill) {
          const ratio = s.njitMs / Math.max(s.kernelMs, 0.001);
          // honest both ways: a sparse scope is too little work for 48
          // cooperative blocks, and the CPU genuinely wins those
          note = ratio >= 1
            ? ` · ${ratio.toFixed(ratio >= 10 ? 0 : 1)}× slower`
            : ` · ${(1 / ratio).toFixed(1)}× FASTER — paint a denser raid`;
        }
        lab.textContent = `${ms.toFixed(2)} ms${note}`;
      }, Math.max(ms, 16));
    }
  }

  scoreEl.textContent = "SCORE 0";
})();
