"""4-connectivity CPU reference, re-exported from ch01_gpu_1blob_1block.

@njit(cache=True) keys its cache on the source file path, so importing the
same compiled function object here shares one compiled reference with
ch01 rather than recompiling.
"""

from ..ch01_gpu_1blob_1block.cpu_oracle import cpu_flood_fill
