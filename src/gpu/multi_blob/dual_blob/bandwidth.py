"""Bandwidth model + measured peak, re-exported from the multi_block stage.

The model is IDENTICAL for the dual-blob kernels — and that identity is
itself a finding worth stating: the blob label rides inside the queue
entry's spare bits (both entry formats, see kernels.py), so labeling two
blobs moves ZERO extra bytes. Same per-dequeue traffic, same probe term,
same CAS term, same enqueue term as the single-blob multi_block kernels.
"""
import importlib.util
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_MB = os.path.abspath(os.path.join(
    _HERE, os.pardir, os.pardir, "single_blob", "multi_block"))


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_bw = _load_by_path("_mb_bandwidth", os.path.join(_MB, "bandwidth.py"))

MODEL_NOTE = _bw.MODEL_NOTE
model_bytes = _bw.model_bytes
model_gb_s = _bw.model_gb_s
measure_peak_bandwidth = _bw.measure_peak_bandwidth
