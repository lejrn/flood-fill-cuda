"""Bandwidth model + measured peak, re-exported from shared/bandwidth.py.

The model is IDENTICAL for the dual-blob kernels — and that identity is
itself a finding worth stating: the blob label rides inside the queue
entry's spare bits (both entry formats, see kernels.py), so labeling two
blobs moves ZERO extra bytes. Same per-dequeue traffic, same probe term,
same CAS term, same enqueue term as ch03's single-blob kernels.
"""

from ....shared.bandwidth import (
    MODEL_NOTE, model_bytes, model_gb_s, measure_peak_bandwidth,
)
