"""
Optimized CUDA Flood Fill - Multi-Block Implementation

This package provides an optimized flood fill implementation that utilizes
all streaming multiprocessors (SMs) of modern NVIDIA GPUs for maximum performance.

Key Features:
- Multi-SM parallelization (24 SMs on RTX 4060)
- Global memory queue with L2 cache optimization
- Optimal thread/block configuration
- Comprehensive performance profiling
- Detailed logging for debugging and analysis

Modules:
- kernels: CUDA kernel implementations
- setup: Scene generation and configuration
- profiling: Performance measurement utilities
- utils: Constants and helper functions
- main: Main execution script
"""

