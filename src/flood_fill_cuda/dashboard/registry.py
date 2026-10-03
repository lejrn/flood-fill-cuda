"""
Ordered list of chapter renderer modules the dashboard assembles.

Explicit opt-in, not a plugin/discovery system: the project's chapters
have a fixed narrative order (1.1 -> 1.2 -> 1.3 -> 1.4 -> 1.5 -> 2 -> 3
-> 4), so this list IS that order. Adding a new chapter's dashboard
section means adding its renderer module here (and wiring its section(s)
into assemble.py's page template, since section counts per chapter vary).
"""
from ..chapters.ch01_gpu_1blob_1block.benchmarks import visualize as ch01_viz
from ..chapters.ch02_gpu_1blob_2block.benchmarks import visualize as ch02_viz
from ..chapters.ch03_gpu_1blob_nblock.benchmarks import visualize as ch03_viz
from ..chapters.ch04_gpu_2blob_nblock.benchmarks import visualize as ch04_viz
from ..chapters.ch05_gpu_nblob_nblock.benchmarks import visualize as ch05_viz
from ..chapters.ch06_gpu_nblob_runs.benchmarks import visualize as ch06_viz
from ..triton_twins.compare import visualize as triton_viz

CHAPTERS = [ch01_viz, ch02_viz, ch03_viz, ch04_viz, ch05_viz, ch06_viz]

# Sections after the chapters: the Triton twins of all of them (section 5).
EXTRAS = [triton_viz]
