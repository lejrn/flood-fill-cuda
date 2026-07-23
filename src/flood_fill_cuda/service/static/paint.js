// Paint-and-fill frontend: thick-brush strokes -> POST /api/fill -> replay
// the returned depth timeline as a spreading brightness wave -> fade away.
// No build step, no dependencies.
(() => {
  "use strict";

  const stage = document.getElementById("stage");
  const canvas = document.getElementById("paint");
  const ctx = canvas.getContext("2d", { willReadFrequently: true });

  const BRUSH = 48;              // backing-store px; a thick blob-forming brush
  // Covers the largest stamp's extent from its center (star/square corners
  // reach further than BRUSH/2), plus a small buffer, so onStrokeEnd's crop
  // never clips a stamp.
  const BRUSH_PAD = Math.ceil(BRUSH * 0.65) + 4;
  const BRUSH_STEP = 10;         // backing-store px between stamps along a drag
  const PAINT_COLOR = "rgba(70, 74, 92, 0.94)";   // neutral "wet chalk" stroke
  const MAX_BACKING_PIXELS = 3.5e6;
  const MAX_LIVE_SHAPES = 24;
  const STATS_LIFETIME_MS = 5000;   // results readout: shown, then just gone, no fade

  // Wave animation: frontier pixels are brightest, cooling to a dark,
  // saturated resting shade as the wave passes them. BRIGHT_L/DARK_L are
  // HSL lightness; DECAY_FRACTION is how much of the total level range
  // the cooldown takes (5% -- a fast, tight trailing glow).
  const BRIGHT_L = 0.80;
  const DARK_L = 0.26;
  const WAVE_SAT = 0.72;
  const DECAY_FRACTION = 0.05;

  let painting = false;
  let lastX = 0, lastY = 0;
  let bbox = null;
  let liveShapes = [];
  let currentShape = "circle";
  let currentMode = "gpu";

  // ---- canvas sizing -------------------------------------------------
  // Backing-store resolution is CSS px * devicePixelRatio (capped at 1.5
  // so a 3x phone doesn't triple the pixel count for no visual gain),
  // then uniformly scaled down further if that would still exceed a
  // sane pixel budget.
  function sizeCanvas() {
    const rect = stage.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    let w = Math.round(rect.width * dpr);
    let h = Math.round(rect.height * dpr);
    if (w * h > MAX_BACKING_PIXELS) {
      const scale = Math.sqrt(MAX_BACKING_PIXELS / (w * h));
      w = Math.max(1, Math.round(w * scale));
      h = Math.max(1, Math.round(h * scale));
    }
    canvas.width = w;
    canvas.height = h;
    canvas.style.width = rect.width + "px";
    canvas.style.height = rect.height + "px";
    ctx.fillStyle = PAINT_COLOR;
    ctx.strokeStyle = PAINT_COLOR;
    ctx.lineCap = "round";
  }
  window.addEventListener("resize", sizeCanvas);
  sizeCanvas();

  // The one coordinate mapping used everywhere: CSS px -> backing-store
  // px. Every place that needs a canvas-space point goes through this, so
  // devicePixelRatio / downscaling never has more than one place to get
  // wrong.
  function toCanvasXY(e) {
    const rect = canvas.getBoundingClientRect();
    return [
      (e.clientX - rect.left) * canvas.width / rect.width,
      (e.clientY - rect.top) * canvas.height / rect.height,
    ];
  }

  function growBBox(x, y) {
    const p = BRUSH_PAD;
    if (!bbox) {
      bbox = { minX: x - p, minY: y - p, maxX: x + p, maxY: y + p };
    } else {
      bbox.minX = Math.min(bbox.minX, x - p);
      bbox.minY = Math.min(bbox.minY, y - p);
      bbox.maxX = Math.max(bbox.maxX, x + p);
      bbox.maxY = Math.max(bbox.maxY, y + p);
    }
  }

  function clearMain() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    bbox = null;
  }

  // ---- brush shapes ---------------------------------------------------
  // Each stamp draws one dab centered at (x, y) in backing-store px, using
  // the already-set fillStyle/strokeStyle. Dragging calls these repeatedly
  // along the path (see strokeSegment), so consecutive dabs must overlap
  // enough to stay one connected blob for the kernel.

  function stampCircle(x, y) {
    ctx.beginPath();
    ctx.arc(x, y, BRUSH / 2, 0, Math.PI * 2);
    ctx.fill();
  }

  function stampSquare(x, y) {
    const s = BRUSH * 0.86;
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate((Math.random() - 0.5) * 0.5);
    ctx.fillRect(-s / 2, -s / 2, s, s);
    ctx.restore();
  }

  function stampStar(x, y) {
    const outer = BRUSH * 0.62;
    const inner = outer * 0.42;
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(Math.random() * Math.PI * 2);
    ctx.beginPath();
    for (let i = 0; i < 10; i++) {
      const r = i % 2 === 0 ? outer : inner;
      const a = (Math.PI / 5) * i - Math.PI / 2;
      const px = Math.cos(a) * r, py = Math.sin(a) * r;
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  // A cluster of short jittered strokes per dab, so a drag builds up a
  // rough, hand-scratched hatch texture instead of a smooth fill.
  function stampScratchy(x, y) {
    const r = BRUSH / 2;
    const prevWidth = ctx.lineWidth;
    ctx.lineWidth = Math.max(3, BRUSH * 0.16);
    for (let i = 0; i < 6; i++) {
      const a = Math.random() * Math.PI * 2;
      const len = r * (0.6 + Math.random() * 0.7);
      const ox = (Math.random() * 2 - 1) * r * 0.35;
      const oy = (Math.random() * 2 - 1) * r * 0.35;
      const cx = x + ox, cy = y + oy;
      ctx.beginPath();
      ctx.moveTo(cx - Math.cos(a) * len / 2, cy - Math.sin(a) * len / 2);
      ctx.lineTo(cx + Math.cos(a) * len / 2, cy + Math.sin(a) * len / 2);
      ctx.stroke();
    }
    ctx.lineWidth = prevWidth;
  }

  const BRUSH_SHAPES = {
    circle: stampCircle,
    square: stampSquare,
    star: stampStar,
    scratchy: stampScratchy,
  };

  function stampBrush(x, y) {
    BRUSH_SHAPES[currentShape](x, y);
  }

  // Dabs the current brush shape at fixed spacing along a segment, so
  // fast drags don't leave gaps and every shape (not just round strokes)
  // gets continuous coverage.
  function strokeSegment(x0, y0, x1, y1) {
    const dist = Math.hypot(x1 - x0, y1 - y0);
    const steps = Math.max(1, Math.ceil(dist / BRUSH_STEP));
    for (let i = 1; i <= steps; i++) {
      const t = i / steps;
      stampBrush(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t);
    }
  }

  const shapeButtons = document.querySelectorAll("#toolbar .brush-btn");
  shapeButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      shapeButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      currentShape = btn.dataset.shape;
    });
  });

  const modeButtons = document.querySelectorAll("#toolbar .mode-btn");
  modeButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      modeButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      currentMode = btn.dataset.mode;
    });
  });

  // ---- painting --------------------------------------------------------
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());

  canvas.addEventListener("pointerdown", (e) => {
    canvas.setPointerCapture(e.pointerId);
    painting = true;
    bbox = null;
    const [x, y] = toCanvasXY(e);
    lastX = x; lastY = y;
    stampBrush(x, y);
    growBBox(x, y);
  });

  canvas.addEventListener("pointermove", (e) => {
    if (!painting) return;
    const [x, y] = toCanvasXY(e);
    strokeSegment(lastX, lastY, x, y);
    growBBox(x, y);
    lastX = x; lastY = y;
  });

  canvas.addEventListener("pointerup", onStrokeEnd);
  canvas.addEventListener("pointercancel", onStrokeEnd);

  function onStrokeEnd(e) {
    if (!painting) return;
    painting = false;
    try { canvas.releasePointerCapture(e.pointerId); } catch (_) {}
    if (!bbox) return;

    // Where the pointer was released — becomes the fill's seed, in
    // crop-local coordinates (below) once the bbox origin is known.
    const [releaseX, releaseY] = toCanvasXY(e);

    const bx = Math.max(0, Math.floor(bbox.minX));
    const by = Math.max(0, Math.floor(bbox.minY));
    const bw = Math.min(canvas.width, Math.ceil(bbox.maxX)) - bx;
    const bh = Math.min(canvas.height, Math.ceil(bbox.maxY)) - by;
    if (bw <= 0 || bh <= 0) { clearMain(); return; }

    const shapeCanvas = document.createElement("canvas");
    shapeCanvas.width = bw;
    shapeCanvas.height = bh;
    shapeCanvas.getContext("2d").drawImage(canvas, bx, by, bw, bh, 0, 0, bw, bh);

    // Screen placement in CSS px, so the falling shape appears exactly
    // where the stroke was painted regardless of backing-store scale.
    const rect = canvas.getBoundingClientRect();
    const scaleX = rect.width / canvas.width;
    const scaleY = rect.height / canvas.height;
    const screenX = rect.left + bx * scaleX;
    const screenY = rect.top + by * scaleY;
    const screenW = bw * scaleX;
    const screenH = bh * scaleY;

    const seedX = releaseX - bx;
    const seedY = releaseY - by;

    clearMain();   // user can paint the next blob immediately
    spawnShape(shapeCanvas, screenX, screenY, screenW, screenH, currentMode, seedX, seedY);
  }

  // ---- server round trip -------------------------------------------
  async function requestFill(shapeCanvas, mode, seedX, seedY) {
    const blob = await new Promise((res) => shapeCanvas.toBlob(res, "image/png"));
    const params = new URLSearchParams({ mode, seed_x: seedX, seed_y: seedY });
    const resp = await fetch(`/api/fill?${params}`, { method: "POST", body: blob });
    if (!resp.ok) {
      const detail = await resp.json().catch(() => ({}));
      throw new Error(`fill failed (${resp.status}): ${detail.detail || resp.statusText}`);
    }
    const buf = await resp.arrayBuffer();
    const dv = new DataView(buf);
    const width = dv.getUint32(4, true);
    const height = dv.getUint32(8, true);
    const levels = dv.getUint32(12, true);
    const depth = new Uint16Array(buf, 16);
    const stats = {
      filled: parseInt(resp.headers.get("x-filled"), 10) || 0,
      amplifiedFilled: parseInt(resp.headers.get("x-amplified-filled"), 10) || 0,
      kernelMs: parseFloat(resp.headers.get("x-kernel-ms")) || 0,
      mode: resp.headers.get("x-mode") || mode,
    };
    return { width, height, levels, depth, stats };
  }

  // Bucket pixel indices by BFS level once, so the animation loop only
  // ever touches pixels newly crossed this frame instead of rescanning
  // the whole depth array every tick. Mirrors the depth-threshold replay
  // in chapters/ch04_gpu_2blob_nblock/benchmarks/wavefront.py::render_timeline.
  function bucketByLevel(depth, levels) {
    const buckets = Array.from({ length: levels }, () => []);
    for (let i = 0; i < depth.length; i++) {
      const v = depth[i];
      if (v > 0) buckets[v - 1].push(i);   // encoded v = depth+1
    }
    return buckets;
  }

  function hslToRgb(h, s, l) {
    h = (((h % 360) + 360) % 360) / 360;
    const k = (n) => (n + h * 12) % 12;
    const a = s * Math.min(l, 1 - l);
    const f = (n) => l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
    return [Math.round(f(0) * 255), Math.round(f(8) * 255), Math.round(f(4) * 255)];
  }

  // Wave color as a function of normalized BFS depth (0 at the seed, 1 at
  // the outermost level, hue purple -> blue -> green) and how far behind
  // the frontier this pixel currently is, in levels: bright right at the
  // frontier, cooling linearly to a dark stable shade over decayLevels.
  function waveColor(levelNorm, distanceBehindFrontier, decayLevels) {
    const hue = 270 - 130 * levelNorm;
    const t = Math.min(1, Math.max(0, distanceBehindFrontier / decayLevels));
    const light = BRIGHT_L + (DARK_L - BRIGHT_L) * t;
    return hslToRgb(hue, WAVE_SAT, light);
  }

  // durationMs is the engine's own reported compute time: the replay
  // plays at the fill's actual real-world speed rather than a stylized
  // pace, so a 32ms GPU fill visibly snaps in ~32ms and a slower CPU fill
  // on the same blob visibly crawls for as long as it really took.
  function animateFill(sctx, origImageData, depth, levels, durationMs) {
    return new Promise((resolve) => {
      const buckets = bucketByLevel(depth, levels);
      const out = sctx.createImageData(origImageData.width, origImageData.height);
      out.data.set(origImageData.data);

      const duration = Math.max(durationMs, 1);
      const decayLevels = Math.max(1, Math.round(levels * DECAY_FRACTION));
      let start = null;

      function paintLevel(idx, color) {
        const p = idx * 4;
        out.data[p] = color[0];
        out.data[p + 1] = color[1];
        out.data[p + 2] = color[2];
        // Preserve the original (possibly antialiased) alpha so stroke
        // edges keep their softness instead of gaining a hard outline.
        out.data[p + 3] = origImageData.data[p + 3];
      }

      function paintWindow(currentLevel) {
        const hi = Math.floor(currentLevel);
        const lo = Math.max(0, Math.floor(currentLevel - decayLevels));
        for (let lvl = lo; lvl <= hi; lvl++) {
          const levelNorm = levels > 1 ? lvl / (levels - 1) : 0;
          const color = waveColor(levelNorm, currentLevel - lvl, decayLevels);
          for (const idx of buckets[lvl]) paintLevel(idx, color);
        }
      }

      function frame(ts) {
        if (start === null) start = ts;
        const elapsed = ts - start;
        const currentLevel = Math.min(levels - 1, (elapsed / duration) * levels);
        paintWindow(currentLevel);
        sctx.putImageData(out, 0, 0);

        if (elapsed < duration) {
          requestAnimationFrame(frame);
        } else {
          // Final settle pass: every level fully cooled, so the finished
          // shape never freezes mid-brighten at its outermost ring.
          for (let lvl = 0; lvl < levels; lvl++) {
            const levelNorm = levels > 1 ? lvl / (levels - 1) : 0;
            const color = waveColor(levelNorm, decayLevels, decayLevels);
            for (const idx of buckets[lvl]) paintLevel(idx, color);
          }
          sctx.putImageData(out, 0, 0);
          resolve();
        }
      }
      requestAnimationFrame(frame);
    });
  }

  // ---- falling shapes ------------------------------------------------
  function evictOldestIfNeeded() {
    while (liveShapes.length > MAX_LIVE_SHAPES) {
      liveShapes.shift().remove();
    }
  }

  // The painted pixel count and the amplified (real GPU/CPU-scale) pixel
  // count are shown side by side: the shape you see stays exactly the
  // size you painted, but the numbers -- and the pacing below -- are
  // honest at the scale where CPU vs GPU actually differs.
  function showStatsLabel(wrap, stats, levels) {
    const label = document.createElement("div");
    label.className = "shape-stats";
    label.textContent =
      `${stats.mode.toUpperCase()} · ${stats.filled.toLocaleString()} px → ` +
      `${stats.amplifiedFilled.toLocaleString()} px @ scale · ` +
      `${levels} levels · ${stats.kernelMs.toFixed(2)} ms`;
    wrap.appendChild(label);
    return label;
  }

  // The stats label and the blob fade independently: the label just
  // disappears outright (no transition) after STATS_LIFETIME_MS, while
  // only the blob (the inner canvas) gets the CSS fade.
  function scheduleStatsRemoval(label) {
    return new Promise((resolve) => {
      setTimeout(() => {
        label.remove();
        resolve();
      }, STATS_LIFETIME_MS);
    });
  }

  function startFadeOut(inner) {
    return new Promise((resolve) => {
      inner.addEventListener("animationend", () => resolve(), { once: true });
      inner.classList.add("fading");
    });
  }

  async function spawnShape(shapeCanvas, screenX, screenY, screenW, screenH, mode, seedX, seedY) {
    const wrap = document.createElement("div");
    wrap.className = "shape-wrap";
    wrap.style.left = screenX + "px";
    wrap.style.top = screenY + "px";
    wrap.style.width = screenW + "px";
    wrap.style.height = screenH + "px";

    const inner = document.createElement("div");
    inner.className = "shape-inner";
    inner.appendChild(shapeCanvas);

    wrap.appendChild(inner);
    stage.appendChild(wrap);
    liveShapes.push(wrap);
    evictOldestIfNeeded();

    const sctx = shapeCanvas.getContext("2d");
    const origImageData = sctx.getImageData(0, 0, shapeCanvas.width, shapeCanvas.height);

    let labelDone = Promise.resolve();
    try {
      const { levels, depth, stats } = await requestFill(shapeCanvas, mode, seedX, seedY);
      // Shown the instant the fill computation's result is known, right
      // above the blob, so the timing is legible exactly when it matters.
      // Its own lifetime (STATS_LIFETIME_MS) runs independently of the
      // blob's fade below.
      const label = showStatsLabel(wrap, stats, levels);
      labelDone = scheduleStatsRemoval(label);
      if (levels > 0) {
        await animateFill(sctx, origImageData, depth, levels, stats.kernelMs);
      }
    } catch (err) {
      // Network hiccup or server error: let the shape fade as painted
      // rather than stranding it on screen.
      console.error(err);
    }

    // Shape stays fully still on screen throughout painting AND the fill
    // animation above; only now does the blob start fading. wrap itself
    // stays in the DOM until both the blob's fade and the label's own
    // lifetime are done, so the label keeps its anchor even after the
    // blob underneath it has faded away.
    await Promise.all([startFadeOut(inner), labelDone]);
    wrap.remove();
    liveShapes = liveShapes.filter((s) => s !== wrap);
  }
})();
