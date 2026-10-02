"""Machine-code evidence for the ENQ switch (twin-only helper).

The per-lane enqueue (ENQ="lane") is written as one relaxed atomic per
winning lane, and the PTX says exactly that. ptxas then rewrites each of
those uniform-address atomics into Numba's warp-aggregated pattern. These
helpers read that back from the compiled binary:

- ``ptx_barriers(ptx)``: CTA barriers (``bar.sync`` / ``barrier.sync``) in
  the PTX. For these kernels the count equals the SASS ``BAR.SYNC`` count,
  so it needs no disassembler.
- ``disassemble(cubin, path)``: SASS text through the nvdisasm Triton
  bundles. nvdisasm cannot read a pipe, so the cubin is written to
  ``path`` first; the caller owns that file.
- ``enqueue_report(sass)``: per enqueue atomic, whether ptxas
  warp-aggregated it, and where the CTA barriers sit.

An enqueue atomic is a 32-bit ``ATOMG.E.ADD`` (the rear counter): the
visited claims are ``ATOMG.E.EXCH`` and the exit counters are 64-bit. It
counts as warp-aggregated when a ``VOTEU.ANY`` (the active-lane mask;
``VOTE.ANY`` when it lands in a regular register) comes shortly before
it, the atomic itself is predicated (one leader lane issues it) and a
``SHFL.IDX`` (the leader's result broadcast to the warp) comes shortly
after it. That is the SASS of Numba's
``_warp_enqueue_global`` (activemask, popc, leader atomic, shfl).
"""

import re
import subprocess

_INSTR = re.compile(r"/\*[0-9a-f]{4,}\*/\s+(.*?)\s*;")
_ENQ_ATOM = re.compile(r"\bATOMG\.E\.ADD\.STRONG\.GPU\b")
_PTX_BAR = re.compile(r"\b(?:bar|barrier)\.sync\b")
_VOTE = re.compile(r"\bVOTEU?\.ANY\b")


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


def enqueue_report(sass, window=12):
    """Enqueue atomics, how many ptxas warp-aggregated, and the barriers.

    ``bar_sync_between`` lists, for each pair of consecutive enqueue
    atomics, how many BAR.SYNC sit between them: all zeros means no CTA
    barrier inside the enqueue sequence of a tile.
    """
    ins = instructions(sass)
    sites = [i for i, s in enumerate(ins) if _ENQ_ATOM.search(s)]
    bars = [i for i, s in enumerate(ins) if "BAR.SYNC" in s]
    aggregated = 0
    for i in sites:
        before = ins[max(0, i - window):i]
        after = ins[i + 1:i + 1 + window]
        if (ins[i].startswith("@")
                and any(_VOTE.search(s) for s in before)
                and any("SHFL.IDX" in s for s in after)):
            aggregated += 1
    between = [sum(a < b < c for b in bars) for a, c in zip(sites, sites[1:])]
    return {"enqueue_atomics": len(sites), "warp_aggregated": aggregated,
            "bar_sync": len(bars), "bar_sync_between": between}
