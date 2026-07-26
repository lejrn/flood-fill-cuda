"""Re-exports the promoted bandwidth instrumentation from shared/bandwidth.py.

This chapter introduced bandwidth instrumentation; the canonical
implementation now lives in shared/ since ch04_gpu_2blob_nblock reuses it
too.
"""

from ....shared.bandwidth import (
    MODEL_NOTE, model_bytes, model_gb_s, measure_peak_bandwidth,
)
