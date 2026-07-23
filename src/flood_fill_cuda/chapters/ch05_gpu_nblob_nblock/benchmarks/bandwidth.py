"""Bandwidth model + measured peak for the seed-discovery stage.

The measured D2D peak comes from shared/bandwidth.py unchanged. The MODEL
does not: ch04 could claim "labeling moves zero extra bytes" because the
label rode inside the queue entry — this chapter retires that format
(see kernels.py), so the model gains explicit discovery and label_map
terms. The chapter-specific formula and its itemized note live in
flood_fill.py (model_bytes_ch05 / MODEL_NOTE); this shim re-exports both
next to the shared peak measurement so benchmark code has one import
site, like every other chapter.
"""

from ....shared.bandwidth import model_gb_s, measure_peak_bandwidth
from ..flood_fill import model_bytes_ch05, MODEL_NOTE
