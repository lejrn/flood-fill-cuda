// Paint-and-fill frontend: thick-brush strokes -> POST /api/fill -> replay
// the returned depth timeline as a spreading recolor -> feather-fall away.
// No build step, no dependencies.
(() => {
  "use strict";

  const stage = document.getElementById("stage");
  const canvas = document.getElementById("paint");
  const statsEl = document.getElementById("stats");
  const ctx = canvas.getContext("2d", { willReadFrequently: true });

  const BRUSH = 48;              // backing-store px; a thick blob-forming brush
  const BRUSH_PAD = BRUSH / 2 + 2;
  const PAINT_COLOR = "rgba(70, 74, 92, 0.94)";   // neutral "wet chalk" stroke
  const FILL_PALETTE = [
    [255, 183, 3], [33, 158, 188], [251, 133, 0],
    [6, 214, 160], [239, 71, 111], [131, 56, 236],
  ];
  const MAX_BACKING_PIXELS = 3.5e6;
  const MAX_LIVE_SHAPES = 24;

  let painting = false;
  let lastX = 0, lastY = 0;
  let bbox = null;
  let liveShapes = [];

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
    ctx.lineWidth = BRUSH;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.fillStyle = PAINT_COLOR;
    ctx.strokeStyle = PAINT_COLOR;
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

  // ---- painting --------------------------------------------------------
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());

  canvas.addEventListener("pointerdown", (e) => {
    canvas.setPointerCapture(e.pointerId);
    painting = true;
    bbox = null;
    const [x, y] = toCanvasXY(e);
    lastX = x; lastY = y;
    ctx.beginPath();
    ctx.arc(x, y, BRUSH / 2, 0, Math.PI * 2);
    ctx.fill();
    growBBox(x, y);
  });

  canvas.addEventListener("pointermove", (e) => {
    if (!painting) return;
    const [x, y] = toCanvasXY(e);
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();
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

    clearMain();   // user can paint the next blob immediately
    spawnShape(shapeCanvas, screenX, screenY, screenW, screenH);
  }

  // ---- server round trip -------------------------------------------
  async function requestFill(shapeCanvas) {
    const blob = await new Promise((res) => shapeCanvas.toBlob(res, "image/png"));
    const resp = await fetch("/api/fill", { method: "POST", body: blob });
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
    statsEl.textContent =
      `${resp.headers.get("x-filled")} px · ${levels} levels · ` +
      `kernel ${parseFloat(resp.headers.get("x-kernel-ms")).toFixed(2)} ms`;
    return { width, height, levels, depth };
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

  function animateFill(sctx, origImageData, depth, levels, fillColor) {
    return new Promise((resolve) => {
      const buckets = bucketByLevel(depth, levels);
      const out = sctx.createImageData(origImageData.width, origImageData.height);
      out.data.set(origImageData.data);
      const bandColor = fillColor.map((c) => Math.min(255, c + 90));

      const duration = Math.min(2000, Math.max(600, levels * 12));
      let start = null;
      let prevLevel = -1;

      function paintLevel(idx, color) {
        const p = idx * 4;
        out.data[p] = color[0];
        out.data[p + 1] = color[1];
        out.data[p + 2] = color[2];
        // Preserve the original (possibly antialiased) alpha so stroke
        // edges keep their softness instead of gaining a hard outline.
        out.data[p + 3] = origImageData.data[p + 3];
      }

      function frame(ts) {
        if (start === null) start = ts;
        const elapsed = ts - start;
        let t = Math.floor((elapsed / duration) * levels);
        if (t > levels - 1) t = levels - 1;

        for (let lvl = prevLevel + 1; lvl <= t; lvl++) {
          for (const idx of buckets[lvl]) paintLevel(idx, fillColor);
        }
        for (let b = 1; b <= 2; b++) {
          const lvl = t + b;
          if (lvl >= levels) break;
          for (const idx of buckets[lvl]) paintLevel(idx, bandColor);
        }
        sctx.putImageData(out, 0, 0);
        prevLevel = t;

        if (elapsed < duration) requestAnimationFrame(frame);
        else resolve();
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

  function startFall(wrap, inner) {
    const fallDur = (6 + Math.random() * 3).toFixed(2);
    const swayDur = (1.2 + Math.random() * 0.6).toFixed(2);
    const drift = (4 + Math.random() * 5).toFixed(1);
    const rot = (5 + Math.random() * 6).toFixed(1);
    wrap.style.setProperty("--fall-dur", fallDur + "s");
    inner.style.setProperty("--sway-dur", swayDur + "s");
    inner.style.setProperty("--drift", drift + "vw");
    inner.style.setProperty("--rot", rot + "deg");
    wrap.addEventListener("animationend", () => {
      wrap.remove();
      liveShapes = liveShapes.filter((s) => s !== wrap);
    }, { once: true });
    wrap.classList.add("falling");
    inner.classList.add("swaying");
  }

  async function spawnShape(shapeCanvas, screenX, screenY, screenW, screenH) {
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
    const fillColor = FILL_PALETTE[Math.floor(Math.random() * FILL_PALETTE.length)];

    try {
      const { levels, depth } = await requestFill(shapeCanvas);
      if (levels > 0) {
        await animateFill(sctx, origImageData, depth, levels, fillColor);
      }
    } catch (err) {
      // Network hiccup or server error: let the shape fall as painted
      // rather than stranding it on screen.
      console.error(err);
    }

    // Shape stays fully still on screen throughout painting AND the fill
    // animation above; only now does it start falling.
    startFall(wrap, inner);
  }
})();
