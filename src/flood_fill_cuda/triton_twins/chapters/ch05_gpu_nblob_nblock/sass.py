"""Machine-code evidence for the ENQ switch (twin-only helper).

The per-lane enqueue (ENQ="lane") is written as one relaxed atomic_add of
1 per winning lane on the rear counter, and the PTX says exactly that. ptxas
then rewrites each of those uniform-address atomics into the warp
aggregation of Numba's hand-written _warp_enqueue_global. These helpers
read that back from the compiled binary:

- ``ptx_barriers(ptx)``: CTA barriers (``bar.sync`` / ``barrier.sync``) in
  the PTX. For these kernels the count equals the SASS ``BAR.SYNC`` count,
  so it needs no disassembler.
- ``nvdisasm_path()``: the nvdisasm Triton bundles (or the CUDA toolkit's),
  or None.
- ``disassemble(cubin, path)``: SASS text of a cubin. nvdisasm cannot read
  a pipe, so the cubin is written to ``path`` first; the caller owns that
  file.
- ``enqueue_report(sass)``: per rear atomic, whether ptxas warp-aggregated
  a per-lane +1 into it, and the CTA barrier count.

A rear atomic is a 32-bit ``ATOMG.E.ADD`` (q_state is int32): the visited
claims are ``ATOMG.E.EXCH``, the links ``ATOMG.E.MIN``, the exit counters
and the grid barrier 64-bit. Two tests per atomic, from strict to loose:

- ``warp_aggregated`` (strict): a per-lane +1, aggregated over the warp. A
  ``VOTEU.ANY`` (``VOTE.ANY`` in a regular register) takes the active-lane
  mask, the atomic's operand is the ``POPC`` of that very mask (one slot
  per active lane), the atomic is predicated (one leader lane issues it)
  and a ``SHFL.IDX`` (the leader's result broadcast to the warp) follows.
  That is Numba's activemask / popc / leader atomic / shfl.
- ``vote_wrapped`` (loose): the same vote, leader predicate and
  ``SHFL.IDX``, whatever the operand. ptxas wraps any uniform-address
  atomic this way: in the ENQ="program" binaries the atomic sits behind
  the program's one-thread branch and adds the program's count, so it is
  wrapped but not a per-lane aggregation.
"""

import os
import re
import shutil
import subprocess

_INSTR = re.compile(r"/\*[0-9a-f]{4,}\*/\s+(.*?)\s*;")
_PTX_BAR = re.compile(r"\b(?:bar|barrier)\.sync\b")
_PTX_ATOM32 = re.compile(r"\batom(?:\.\w+)*\.add\.u32\b")
_VOTE = re.compile(r"\bVOTEU?\.ANY\s+(U?R\d+)")
_GUARD = re.compile(r"^@!?U?P\w+\s+")
_REG = re.compile(r"^(U?R)(\d+)$")
# opcodes whose first destination is a predicate, the register second
_PRED_FIRST = ("SHFL", "ATOMG", "ATOM", "RED")


def ptx_barriers(ptx):
    """Number of CTA barriers in a kernel's PTX."""
    return len(_PTX_BAR.findall(ptx))


def ptx_rear_atomics(ptx):
    """Number of 32-bit atomic adds in a kernel's PTX: one per enqueue
    site (the grid barrier and the exit counters add 64 bits). ptxas may
    emit more than one SASS atomic per site: in a one-warp program it
    compiles the ENQ="program" atomic into two paths (per-lane atomics
    when the warp is partial, a SHFL.UP scan and one atomic when it is
    full)."""
    return len(_PTX_ATOM32.findall(ptx))


def nvdisasm_path():
    """Triton's bundled nvdisasm, else the toolkit's, else None."""
    try:
        import triton.backends.nvidia as nv
        bundled = os.path.join(os.path.dirname(nv.__file__), "bin",
                               "nvdisasm")
    except ImportError:
        bundled = ""
    for path in (bundled, shutil.which("nvdisasm") or "",
                 "/usr/local/cuda/bin/nvdisasm"):
        if path and os.access(path, os.X_OK):
            return path
    return None


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


def is_rear_atomic(ins):
    """A 32-bit global atomic add: a queue-rear ticket. Checks the opcode
    only (an address operand such as [R12.64] is not a 64-bit add)."""
    op = _split(ins)[0]
    return op.startswith("ATOMG.E.ADD.") and ".64" not in op


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
    """Rear atomics, how many ptxas warp-aggregated, and the barriers.

    ``warp_aggregated`` counts the per-lane +1 atomics ptxas aggregated per
    warp (strict test); ``vote_wrapped`` counts every rear atomic ptxas
    wrapped in a vote / leader / shuffle sequence (loose test, see the
    module docstring); ``bar_sync`` counts the CTA barriers of the kernel.
    """
    ins = instructions(sass)
    sites = [i for i, s in enumerate(ins) if is_rear_atomic(s)]
    wrapped = aggregated = 0
    for i in sites:
        before = ins[max(0, i - window):i]
        after = ins[i + 1:i + 1 + window]
        if (ins[i].startswith("@")
                and any(_VOTE.search(s) for s in before)
                and any("SHFL.IDX" in s for s in after)):
            wrapped += 1
            aggregated += _per_lane_count(ins, i, window)
    return {"enqueue_atomics": len(sites), "warp_aggregated": aggregated,
            "vote_wrapped": wrapped,
            "bar_sync": sum("BAR.SYNC" in s for s in ins),
            "instructions": len(ins)}
