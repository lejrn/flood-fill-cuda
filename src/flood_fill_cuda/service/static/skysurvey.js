// DEEP FIELD: a shared star field, you estimate the count, then the REAL
// ch06 kernel surveys it — a scan bar sweeps the frame, every
// star lights in its own color, a counter spins up to the true number, and
// the closest guess climbs the leaderboard. No build step, no deps.
(() => {
  "use strict";

  const stage = document.getElementById("stage");
  const field = document.getElementById("field");
  const fx = document.getElementById("fx");
  const fctx = field.getContext("2d");
  const xctx = fx.getContext("2d");

  const guessEl = document.getElementById("guess");
  const nameEl = document.getElementById("name");
  const surveyBtn = document.getElementById("survey-btn");
  const newBtn = document.getElementById("new-btn");
  const diffEl = document.getElementById("difficulty");
  const playersEl = document.getElementById("players");
  const revealEl = document.getElementById("reveal");
  const counterEl = document.getElementById("counter");
  const resultEl = document.getElementById("result");
  const boardBody = document.querySelector("#board tbody");
  const boardHint = document.getElementById("board-hint");

  const REVEAL_MS = 2600;           // dramatic sweep, independent of kernel ms
  const GOLDEN = 137.508;

  let challenge = null;             // {id, width, height}
  let fieldBlob = null;             // exact PNG bytes to re-scan
  let busy = false;

  // ---- load / draw the field ------------------------------------------
  function fitCanvases(w, h) {
    const rect = stage.getBoundingClientRect();
    const scale = Math.min(rect.width / w, rect.height / h) * 0.94;
    for (const c of [field, fx]) {
      c.width = w; c.height = h;
      c.style.width = w * scale + "px";
      c.style.height = h * scale + "px";
    }
  }

  async function loadChallenge(makeNew) {
    busy = true; surveyBtn.disabled = true;
    revealEl.hidden = true;
    resultEl.innerHTML = "";
    boardBody.innerHTML = "";
    boardHint.hidden = false;
    playersEl.textContent = "loading field…";
    try {
      const url = makeNew
        ? `/api/challenge/new?difficulty=${encodeURIComponent(diffEl.value)}`
        : "/api/challenge";
      const resp = await fetch(url, { method: makeNew ? "POST" : "GET" });
      if (!resp.ok) throw new Error(`field load failed (${resp.status})`);
      challenge = {
        id: resp.headers.get("x-challenge-id"),
        width: parseInt(resp.headers.get("x-width"), 10),
        height: parseInt(resp.headers.get("x-height"), 10),
        difficulty: resp.headers.get("x-difficulty"),
      };
      fieldBlob = await resp.blob();
      const bmp = await createImageBitmap(fieldBlob);
      fitCanvases(challenge.width, challenge.height);
      fctx.clearRect(0, 0, field.width, field.height);
      fctx.drawImage(bmp, 0, 0);
      xctx.clearRect(0, 0, fx.width, fx.height);
      field.style.opacity = "1";
      await refreshPlayers();
    } catch (err) {
      playersEl.textContent = String(err.message || err);
    } finally {
      busy = false; surveyBtn.disabled = false;
    }
  }

  async function refreshPlayers() {
    try {
      const r = await fetch("/api/leaderboard");
      const d = await r.json();
      playersEl.textContent = d.players
        ? `${d.players} astronomer${d.players === 1 ? "" : "s"} surveyed · ${challenge.difficulty}`
        : `be the first to survey · ${challenge.difficulty}`;
    } catch (_) { playersEl.textContent = challenge.difficulty || ""; }
  }

  window.addEventListener("resize", () => {
    if (challenge) fitCanvases(challenge.width, challenge.height);
  });

  // ---- survey (the reveal) --------------------------------------------
  surveyBtn.addEventListener("click", doSurvey);
  newBtn.addEventListener("click", () => { if (!busy) loadChallenge(true); });
  guessEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter") doSurvey();
  });

  function hslToRgb(h, s, l) {
    h = (((h % 360) + 360) % 360) / 360;
    const k = (n) => (n + h * 12) % 12;
    const a = s * Math.min(l, 1 - l);
    const f = (n) =>
      l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
    return [Math.round(f(0) * 255), Math.round(f(8) * 255), Math.round(f(4) * 255)];
  }
  const hue = (id) => (id * GOLDEN) % 360;

  // Bucket pixels by the kernel's SCAN ORDER, not by time: ch06 has no
  // BFS levels, because it has no BFS. Every row is counted, emitted and
  // merged at once and the whole scan is over in about a millisecond, so
  // the reveal animates the one ordering the algorithm really has — its
  // row-major sweep, which is left-to-right on the canvas.
  //
  // Derived from the column rather than received: the bucket is a pure
  // function of x (see app.py's wire-format note), so the server sends
  // only `steps`.
  function bucketByColumn(track, width, height, steps) {
    const buckets = Array.from({ length: Math.max(steps, 1) }, () => []);
    for (let col = 0; col < width; col++) {
      const b = buckets[Math.min(((col * steps) / width) | 0, steps - 1)];
      for (let row = 0, i = col; row < height; row++, i += width) {
        if (track[i]) b.push(i);
      }
    }
    return buckets;
  }

  async function scanField() {
    const resp = await fetch("/api/scan", { method: "POST", body: fieldBlob });
    if (!resp.ok) throw new Error(`scan failed (${resp.status})`);
    const buf = await resp.arrayBuffer();
    const dv = new DataView(buf);
    const width = dv.getUint32(4, true);
    const height = dv.getUint32(8, true);
    const steps = dv.getUint32(12, true);
    const nBlobs = dv.getUint32(16, true);
    const n = width * height;
    const track = new Uint16Array(buf, 24, n);
    return { width, height, steps, nBlobs, track,
             kernelMs: parseFloat(resp.headers.get("x-kernel-ms")) || 0 };
  }

  function revealAnimation(s, trueCount) {
    return new Promise((resolve) => {
      const buckets = bucketByColumn(s.track, s.width, s.height, s.steps);
      const out = xctx.createImageData(s.width, s.height);
      const palette = new Map();
      const colorOf = (id) => {
        let c = palette.get(id);
        if (!c) { c = hslToRgb(hue(id), 0.72, 0.62); palette.set(id, c); }
        return c;
      };
      const seen = new Set();
      let start = null, prevLevel = -1;
      field.style.transition = "opacity 0.4s"; field.style.opacity = "0.18";

      function frame(ts) {
        if (start === null) start = ts;
        const p = Math.min((ts - start) / REVEAL_MS, 1);
        const t = Math.min(Math.floor(p * s.steps), s.steps - 1);
        for (let lvl = prevLevel + 1; lvl <= t; lvl++) {
          for (const i of buckets[lvl]) {
            const id = s.track[i];
            if (id) seen.add(id);
            const c = colorOf(id);
            const q = i * 4;
            out.data[q] = c[0]; out.data[q + 1] = c[1];
            out.data[q + 2] = c[2]; out.data[q + 3] = 255;
          }
        }
        xctx.putImageData(out, 0, 0);
        prevLevel = t;
        // counter tracks discovery, but eases to the true total at the end
        const shown = p < 1 ? seen.size
                            : trueCount;
        counterEl.textContent = shown.toLocaleString();
        if (p < 1) requestAnimationFrame(frame);
        else { counterEl.textContent = trueCount.toLocaleString(); resolve(); }
      }
      revealEl.hidden = false;
      counterEl.textContent = "0";
      resultEl.innerHTML = "";
      requestAnimationFrame(frame);
    });
  }

  async function doSurvey() {
    if (busy || !challenge) return;
    const guess = parseInt(guessEl.value, 10);
    if (!Number.isFinite(guess) || guess < 0) {
      guessEl.focus(); return;
    }
    busy = true; surveyBtn.disabled = true; newBtn.disabled = true;
    try {
      // submit the guess and scan the field concurrently; the guess is
      // recorded server-side even if the animation is cut short
      const [scan, verdict] = await Promise.all([
        scanField(),
        fetch("/api/guess", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: nameEl.value, guess }),
        }).then((r) => r.json()),
      ]);
      await revealAnimation(scan, verdict.true_count);
      showVerdict(verdict);
      renderBoard(verdict.leaderboard, nameEl.value.trim() || "ANON", guess);
      await refreshPlayers();
    } catch (err) {
      resultEl.textContent = String(err.message || err);
      revealEl.hidden = false;
    } finally {
      busy = false; surveyBtn.disabled = false; newBtn.disabled = false;
    }
  }

  function showVerdict(v) {
    const merged = v.n_placed - v.true_count;
    const mergeNote = merged > 0
      ? `<div class="merge">${v.n_placed.toLocaleString()} stars placed — `
        + `${merged.toLocaleString()} lost to touching neighbours, so the `
        + `true component count is ${v.true_count.toLocaleString()}.</div>`
      : "";
    resultEl.innerHTML =
      `<span class="you">you guessed ${v.your_guess.toLocaleString()} — `
      + `off by ${v.your_error.toLocaleString()}</span> · rank #${v.your_rank} `
      + `of ${v.players}` + mergeNote;
  }

  function renderBoard(board, myName, myGuess) {
    boardHint.hidden = true;
    boardBody.innerHTML = "";
    let mineMarked = false;
    board.forEach((row, i) => {
      const tr = document.createElement("tr");
      const mine = !mineMarked && row.name === myName && row.guess === myGuess;
      if (mine) { tr.className = "you"; mineMarked = true; }
      tr.innerHTML =
        `<td class="rank">#${i + 1}</td>`
        + `<td>${escapeHtml(row.name)}</td>`
        + `<td>${row.guess.toLocaleString()}</td>`
        + `<td class="err">±${row.error.toLocaleString()}</td>`;
      boardBody.appendChild(tr);
    });
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"]/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }

  loadChallenge(false);
})();
