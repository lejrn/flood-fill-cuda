"""Word times for narration that comes without them, anchored on its pauses.

A voice made outside this repo (the ElevenLabs connector) has no
timestamps; spreading the words by character count drifts by up to a
second over a long sentence. TTS voices pause clearly at punctuation, so
the pauses in the audio pin the sentence and clause ends: each pause is
matched to the punctuation mark whose character-count estimate is
nearest (in order, strong marks first), and the words between two pinned
ends are spread by character count over the speech between them.

Also used by pace.py to find where a sentence's pause is.
"""
from __future__ import annotations

import re

import numpy as np

HOP = 0.01                       # seconds per energy frame
STRONG, MEDIUM, WEAK = 3.0, 2.0, 1.0
_PUNCT = {".": STRONG, "?": STRONG, "!": STRONG, ":": MEDIUM, ";": MEDIUM, ",": WEAK}


def boundaries(text: str) -> list[tuple[int, str]]:
    """(index of the word the mark follows, the mark) for every clause end
    inside the text (the last word's mark is the end of the beat)."""
    words = text.split()
    out = []
    for i, w in enumerate(words[:-1]):
        m = re.search(r"[.?!:;,]+$", w.rstrip("\"')"))
        if m:
            out.append((i, m.group(0)[0]))
    return out


def energy(audio: np.ndarray, sr: int) -> np.ndarray:
    """RMS per HOP frame."""
    n = int(sr * HOP)
    k = len(audio) // n
    if k == 0:
        return np.zeros(1)
    a = audio[: k * n].reshape(k, n)
    return np.sqrt((a.astype(np.float64) ** 2).mean(axis=1))


def quiet_mask(audio: np.ndarray, sr: int, rel_db: float = -36.0) -> np.ndarray:
    e = energy(audio, sr)
    return e < e.max() * 10 ** (rel_db / 20)


def pauses(audio: np.ndarray, sr: int, min_len: float = 0.1, rel_db: float = -36.0) -> list[tuple[float, float]]:
    """(start, end) of every quiet run of at least min_len, inside the speech."""
    q = quiet_mask(audio, sr, rel_db)
    loud = np.flatnonzero(~q)
    if loud.size == 0:
        return []
    first, last = loud[0], loud[-1]
    out, i = [], first
    while i <= last:
        if q[i]:
            j = i
            while j <= last and q[j]:
                j += 1
            if (j - i) * HOP >= min_len:
                out.append((i * HOP, j * HOP))
            i = j
        else:
            i += 1
    return out


def speech_span(audio: np.ndarray, sr: int, rel_db: float = -36.0) -> tuple[float, float]:
    q = quiet_mask(audio, sr, rel_db)
    loud = np.flatnonzero(~q)
    if loud.size == 0:
        return 0.0, len(audio) / sr
    return loud[0] * HOP, (loud[-1] + 1) * HOP


def _spread(words: list[str], t0: float, t1: float) -> list[list]:
    weights = [len(w) + 1 for w in words]
    total = float(sum(weights)) or 1.0
    rows, t = [], t0
    for w, n in zip(words, weights):
        d = (t1 - t0) * n / total
        rows.append([w, round(t, 3), round(t + d, 3)])
        t += d
    return rows


def align_words(text: str, audio: np.ndarray, sr: int, tol: float = 1.2) -> list[list]:
    """[word, start, end] for every word of `text` in `audio`."""
    words = text.split()
    s0, s1 = speech_span(audio, sr)
    bnds = boundaries(text)
    ps = pauses(audio, sr)
    if not bnds or not ps:
        return _spread(words, s0, s1)
    # character-count estimate of each boundary's time
    weights = np.array([len(w) + 1 for w in words], float)
    cum = np.cumsum(weights) / weights.sum()
    est = [s0 + (s1 - s0) * cum[i] for i, _ in bnds]
    strength = [_PUNCT[m] for _, m in bnds]
    centres = [(a + b) / 2 for a, b in ps]
    # monotonic matching that maximises sum(strength - distance / tol)
    nb, npz = len(bnds), len(ps)
    best = np.zeros((nb + 1, npz + 1))
    move = np.zeros((nb + 1, npz + 1), int)
    for i in range(1, nb + 1):
        for j in range(1, npz + 1):
            opts = [best[i - 1, j], best[i, j - 1]]
            d = abs(centres[j - 1] - est[i - 1])
            gain = strength[i - 1] * (1.0 - d / tol)
            opts.append(best[i - 1, j - 1] + gain if d < tol and gain > 0 else -1e9)
            k = int(np.argmax(opts))
            best[i, j], move[i, j] = opts[k], k
    pairs = []
    i, j = nb, npz
    while i > 0 and j > 0:
        k = move[i, j]
        if k == 2:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif k == 0:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    # pin each matched boundary: its word ends where the pause starts
    rows: list[list] = []
    w0, t0 = 0, s0
    for bi, pj in pairs:
        wi = bnds[bi][0]
        rows += _spread(words[w0:wi + 1], t0, ps[pj][0])
        w0, t0 = wi + 1, ps[pj][1]
    rows += _spread(words[w0:], t0, s1)
    return rows
