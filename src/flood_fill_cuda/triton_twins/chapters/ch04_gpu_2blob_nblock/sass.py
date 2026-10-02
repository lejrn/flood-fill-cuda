"""Machine-code evidence for the ENQ switch (twin-only helper).

The per-lane enqueue (ENQ="lane") is written as one relaxed atomic per
winning lane, and the PTX says exactly that. ptxas then rewrites each of
those uniform-address atomics into the warp-aggregation pattern of Numba's
hand-written helper. These helpers read that back from the compiled
binary:

- ``ptx_barriers(ptx)``: CTA barriers (``bar.sync`` / ``barrier.sync``) in
  the PTX. For these kernels the count equals the SASS ``BAR.SYNC`` count,
  so it needs no disassembler.
- ``disassemble(cubin, path)``: SASS text through the nvdisasm Triton
  bundles. nvdisasm cannot read a pipe, so the cubin is written to
  ``path`` first; the caller owns that file.
- ``enqueue_report(sass)``: per enqueue atomic, whether ptxas
  warp-aggregated a per-lane +1 into it, and where the CTA barriers sit.

An enqueue atomic is a 32-bit ``ATOMG.E.ADD`` (the rear counter): the
visited claims are ``ATOMG.E.EXCH`` and the exit counters are 64-bit.

Two tests per atomic, from strict to loose:

- ``warp_aggregated`` (strict): a per-lane +1, aggregated over the warp.
  A ``VOTEU.ANY`` (``VOTE.ANY`` in a regular register) takes the
  active-lane mask, the atomic's operand is the ``POPC`` of that very mask
  (one slot per active lane), the atomic is predicated (one leader lane
  issues it) and a ``SHFL.IDX`` (the leader's result broadcast to the
  warp) follows. That is the pattern of Numba's ``_warp_enqueue_global``
  (activemask, popc, leader atomic, shfl).
- ``vote_wrapped`` (loose): the same vote, leader predicate and
  ``SHFL.IDX``, whatever the operand. ptxas wraps any uniform-address
  atomic this way. In the ENQ="program" binaries the atomic sits behind
  the program's one-thread branch and adds the program's count, so ptxas
  multiplies that count by the POPC of a one-lane mask: wrapped, but not
  a per-lane aggregation.
"""

import re
import subprocess

_INSTR = re.compile(r"/\*[0-9a-f]{4,}\*/\s+(.*?)\s*;")
_ENQ_ATOM = re.compile(r"\bATOMG\.E\.ADD\.STRONG\.GPU\b")
_PTX_BAR = re.compile(r"\b(?:bar|barrier)\.sync\b")
_VOTE = re.compile(r"\bVOTEU?\.ANY\s+(U?R\d+)")
_GUARD = re.compile(r"^@!?U?P\w+\s+")
_REG = re.compile(r"^(U?R)(\d+)$")
# opcodes whose first destination is a predicate, the register second
_PRED_FIRST = ("SHFL", "ATOMG", "ATOM", "RED")


def ptx_barriers(ptx):
    """Number of CTA barriers in a kernel's PTX."""
    return len(_PTX_BAR.findall(ptx))


def nvdisasm_path():
    from triton import knobs

    return knobs.nvidia.nvdisasm.path


def disassemble(cubin, path):
    """SASS of ``cubin`` (bytes), written to ``path`` for nvdisasm."""
    with open(path, "wb") as f:
        f.write(cubin)
    return subprocess.run([nvdisasm_path(), "-c", str(path)],
                          capture_output=True, text=True, check=True).stdout


def instructions(sass):
    """The instruction text of every SASS line, in address order."""
    return [m.group(1) for m in _INSTR.finditer(sass)]


def _split(ins):
    """(opcode, operands) of one instruction, its guard predicate removed."""
    body = _GUARD.sub("", ins)
    op, _, rest = body.partition(" ")
    return op, [o.strip() for o in rest.split(",")] if rest else []


def _writes(ins, reg):
    """Whether ``ins`` writes register ``reg`` (a 64-bit op writes a pair)."""
    op, ops = _split(ins)
    if not ops:
        return False
    dst = ops[1] if op.startswith(_PRED_FIRST) and len(ops) > 1 else ops[0]
    if dst == reg:
        return True
    m, r = _REG.match(dst), _REG.match(reg)
    return bool(".64" in op and m and r and m.group(1) == r.group(1)
                and int(r.group(2)) == int(m.group(2)) + 1)


def _per_lane_count(ins, i, window):
    """Whether atomic ``ins[i]`` adds the POPC of a VOTE(U).ANY mask, i.e.
    one slot per active lane: walks back to the nearest writer of its
    operand register, which must be ``POPC <operand>, <vote mask>``."""
    data = _split(ins[i])[1][-1]
    for j in range(i - 1, max(-1, i - 1 - window), -1):
        if not _writes(ins[j], data):
            continue
        op, ops = _split(ins[j])
        if op != "POPC" or len(ops) != 2:
            return False
        masks = {m.group(1) for s in ins[max(0, j - window):j]
                 for m in [_VOTE.search(s)] if m}
        return ops[1] in masks
    return False


def enqueue_report(sass, window=12):
    """Enqueue atomics, how many ptxas warp-aggregated, and the barriers.

    ``warp_aggregated`` counts the per-lane +1 atomics ptxas aggregated per
    warp (strict test); ``vote_wrapped`` counts every atomic ptxas wrapped
    in a vote / leader / shuffle sequence (loose test, see the module
    docstring). ``bar_sync_between`` lists, for each pair of consecutive
    enqueue atomics, how many BAR.SYNC sit between them: all zeros means
    no CTA barrier inside the enqueue sequence of a tile.
    """
    ins = instructions(sass)
    sites = [i for i, s in enumerate(ins) if _ENQ_ATOM.search(s)]
    bars = [i for i, s in enumerate(ins) if "BAR.SYNC" in s]
    wrapped = aggregated = 0
    for i in sites:
        before = ins[max(0, i - window):i]
        after = ins[i + 1:i + 1 + window]
        if (ins[i].startswith("@")
                and any(_VOTE.search(s) for s in before)
                and any("SHFL.IDX" in s for s in after)):
            wrapped += 1
            aggregated += _per_lane_count(ins, i, window)
    between = [sum(a < b < c for b in bars) for a, c in zip(sites, sites[1:])]
    return {"enqueue_atomics": len(sites), "warp_aggregated": aggregated,
            "vote_wrapped": wrapped, "bar_sync": len(bars),
            "bar_sync_between": between}
