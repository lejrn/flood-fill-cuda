"""CPU references: 4-connectivity from ch01, 8-connectivity from shared/.

@njit(cache=True) keys its cache on the source file path, so importing
these function objects directly shares one compiled reference with their
origin module rather than recompiling.
"""

from ..ch01_gpu_1blob_1block.cpu_oracle import cpu_flood_fill
from ...shared.cpu_oracle import cpu_flood_fill_8
