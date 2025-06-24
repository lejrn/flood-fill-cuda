# CUDA Flood Fill Implementation

This project implements GPU-accelerated flood fill algorithms using CUDA and Numba, optimized for the **NVIDIA GeForce RTX 4060 8GB Laptop GPU**. The implementation includes both CPU sequential versions and high-performance GPU parallel versions for processing large images efficiently.

## Hardware Specifications - RTX 4060 8GB Laptop GPU

This project is optimized for the **NVIDIA GeForce RTX 4060 8GB Laptop GPU** with Ada Lovelace architecture. Understanding these specifications helps optimize flood fill algorithms for maximum performance.

### Core Architecture
- **Compute Capability**: 8.9 (Ada Lovelace)
  - *Latest NVIDIA architecture with enhanced efficiency and features*
- **Multiprocessors**: 24 SMs (Streaming Multiprocessors)
  - *Each SM can execute multiple thread blocks simultaneously*
- **CUDA Cores**: ~3072 (128 cores per SM × 24 SMs)
  - *Parallel processing units for general computation*

### Memory Hierarchy
- **Global Memory**: 8.00 GB GDDR6
  - *Main GPU memory for storing large datasets (images, arrays)*
- **Memory Bandwidth**: ~256 GB/s (128-bit bus @ 8001 MHz)
  - *Speed at which data moves between GPU cores and memory*
- **L2 Cache**: 32 MB (33,554,432 bytes)
  - *Fast intermediate storage reducing global memory access*
- **Shared Memory per Block**: 48 KB (49,152 bytes)
  - *Ultra-fast memory shared among threads in a block*
- **Shared Memory per SM**: 100 KB (102,400 bytes)
  - *Total shared memory available per multiprocessor*
- **Constant Memory**: 64 KB (65,536 bytes)
  - *Read-only memory for constants accessible by all threads*

### Thread Organization
- **Max Threads per Block**: 1,024
  - *Maximum number of threads that can work together in a block*
- **Warp Size**: 32 threads
  - *Smallest unit of parallel execution (threads execute in lockstep)*
- **Max Block Dimensions**: (1024, 1024, 64)
  - *Maximum size in X, Y, Z dimensions for thread blocks*
- **Max Grid Dimensions**: (2,147,483,647, 65,535, 65,535)
  - *Maximum number of blocks in X, Y, Z dimensions*

### Performance Characteristics
- **Core Clock**: 1,470 MHz
  - *Base frequency of GPU cores*
- **Memory Clock**: 8,001 MHz
  - *Frequency of memory operations*
- **Concurrent Kernels**: Supported
  - *Ability to run multiple GPU programs simultaneously*

### Flood Fill Optimization Guidelines

#### Memory Usage Recommendations
- **Image Size Limits**: Up to 4096×4096 pixels (fits comfortably in 8GB VRAM)
- **Batch Processing**: Process ~170 simultaneous 4K images
- **BFS Queue Capacity**: ~6,016 coordinate pairs per block in shared memory

#### Thread Block Configuration
```cuda
// Recommended thread block sizes for flood fill
dim3 blockSize(16, 16);    // 256 threads - good for small kernels
dim3 blockSize(16, 32);    // 512 threads - balanced performance
dim3 blockSize(32, 32);    // 1024 threads - maximum parallelism
```

#### Grid Configuration
```cuda
// Optimal grid sizing for 24 multiprocessors
dim3 gridSize(
    (width + blockSize.x - 1) / blockSize.x,
    (height + blockSize.y - 1) / blockSize.y
);
```

#### Memory Access Patterns
- **Coalesced Memory Access**: Align memory reads to 128-byte boundaries
- **Shared Memory Usage**: Store frequently accessed pixels in shared memory
- **L2 Cache Optimization**: Leverage 32MB cache for repeated memory access

## Algorithm Overview

### CPU Sequential Implementation
The sequential version (located in `src/cpu/sequential.py`) uses a breadth-first search (BFS) algorithm to fill connected regions of red pixels with random colors. This serves as a baseline for performance comparison.

### GPU Parallel Implementation
The CUDA implementation leverages the RTX 4060's parallel processing capabilities:
- **Parallel BFS**: Multiple threads process different regions simultaneously
- **Shared Memory Queues**: Each thread block maintains a local BFS queue
- **Atomic Operations**: Coordinate access to shared visited arrays
- **Memory Coalescing**: Optimize memory access patterns for maximum bandwidth

## Implementation Details

### Red Detection Algorithm
A pixel qualifies as red if its red channel is above a threshold while its green and blue channels are sufficiently low:
```python
def is_red_pixel(r, g, b, threshold=100):
    return r > threshold and g < threshold/2 and b < threshold/2
```

### BFS Flood Fill Process
1. **Initialization**: Mark all red pixels as unvisited
2. **Seed Detection**: Find unvisited red pixels as flood fill starting points
3. **Queue Processing**: Use BFS to explore connected components
4. **Color Assignment**: Assign unique colors to each connected region
5. **Output Generation**: Save processed image with colored blobs

## Environment Setup

### Prerequisites
- **Operating System**: WSL2 on Windows 11 (or native Linux)
- **GPU**: NVIDIA GeForce RTX 4060 8GB Laptop GPU
- **CUDA**: Version 12.9 (verified compatible)
- **Python**: 3.10+ 
- **Poetry**: Package management and virtual environments

### CUDA Environment Configuration
The project requires specific CUDA environment variables for RTX 4060 8GB Laptop GPU:

```bash
# Add to ~/.zshrc or ~/.bashrc
export CUDA_PATH=/usr/local/cuda-12.9
export CUDA_HOME=/usr/local/cuda-12.9
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda-12.9/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-12.9/bin:$PATH
```

### Installation Steps

1. **Install Poetry** ([Poetry Installation](https://python-poetry.org/docs/#installation)):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Configure Poetry for project-local virtual environments**:
   ```bash
   poetry config virtualenvs.in-project true
   ```

3. **Install project dependencies**:
   ```bash
   poetry install
   ```

4. **Verify CUDA setup**:
   ```bash
   poetry run python utils/cuda_diagnostic.py
   ```

5. **Test GPU information**:
   ```bash
   poetry run python utils/get_info.py
   ```
## Usage

### Running Flood Fill Algorithms

1. **Place your input image** at `images/input/input_blobs.png`

2. **CPU Sequential Version**:
   ```bash
   poetry run python src/cpu/sequential.py
   ```

3. **GPU CUDA Version** (optimized for RTX 4060 8GB):
   ```bash
   poetry run python src/gpu/flood_fill_cuda.py
   ```

4. **Performance Comparison**:
   ```bash
   poetry run python benchmarks/compare_performance.py
   ```

### Output
- **Processed images**: Saved in `images/output/`
- **Performance metrics**: Execution time and throughput
- **Blob count**: Number of connected components found

### Diagnostic Tools

- **GPU Information**: `poetry run python utils/get_info.py`
- **CUDA Diagnostics**: `poetry run python utils/cuda_diagnostic.py`
- **Memory Usage**: Monitor VRAM usage during processing

## Performance Expectations (RTX 4060 8GB Laptop GPU)

### Image Processing Capabilities
- **Small images (512×512)**: ~1000 FPS
- **Medium images (1024×1024)**: ~250 FPS  
- **Large images (2048×2048)**: ~60 FPS
- **Ultra-large (4096×4096)**: ~15 FPS

### Memory Utilization
- **Single 4K image**: ~48 MB VRAM
- **Batch processing**: Up to 170 simultaneous 4K images
- **Memory bandwidth**: ~256 GB/s effective throughput

## Project Structure

```
flood-fill-cuda/
├── src/
│   ├── cpu/                 # CPU implementations
│   │   └── sequential.py    # Sequential BFS flood fill
│   └── gpu/                 # GPU implementations  
│       ├── flood_fill_cuda.py   # Main CUDA implementation
│       ├── bfs_only.py          # BFS-focused version
│       └── scan_only.py         # Scanning-focused version
├── utils/
│   ├── get_info.py          # RTX 4060 8GB GPU information
│   ├── cuda_diagnostic.py   # CUDA environment diagnostics
│   └── generate_*.py        # Test image generators
├── images/
│   ├── input/               # Input test images
│   └── output/              # Processed results
├── notebooks/               # Jupyter notebooks for development
├── benchmarks/              # Performance testing
└── docs/                    # Documentation
```

## Dependencies

### Core Libraries
- **Python**: 3.10+ (Compatible with CUDA 12.9)
- **NumPy**: 1.24.0+ (Numerical computing and array operations)
- **Numba**: 0.61.0+ (JIT compilation and CUDA kernel generation)
- **CUDA-Python**: 12.0.0+ (Low-level CUDA API access)
- **Pillow**: 10.0.0+ (Image loading and saving)
- **OpenCV**: 4.11.0+ (Advanced image processing)
- **Matplotlib**: 3.10.0+ (Visualization and plotting)

### Development Tools
- **Poetry**: Dependency management and virtual environments
- **Jupyter**: Interactive development and experimentation
- **Rich**: Enhanced terminal output and progress bars
- **NVTX**: NVIDIA profiling and debugging

### System Requirements
- **CUDA Toolkit**: 12.9 (for compilation and runtime)
- **NVIDIA Driver**: 576.80+ (WSL2 compatible)
- **Memory**: 16GB+ RAM recommended (for large image processing)
- **Storage**: 2GB+ free space (for dependencies and results)

## Optimization Techniques

### Memory Optimization
- **Shared Memory Utilization**: Store BFS queues in fast 48KB shared memory
- **Memory Coalescing**: Align pixel access to 128-byte boundaries
- **Texture Memory**: Use texture cache for repeated image access
- **Constant Memory**: Store algorithm parameters in 64KB constant memory

### Thread Organization
- **Warp Efficiency**: Organize threads to minimize divergence
- **Occupancy Optimization**: Balance shared memory vs. thread count
- **Load Balancing**: Distribute work evenly across 24 multiprocessors

### Algorithm Efficiency
- **Atomic Operations**: Minimize contention in visited arrays
- **Early Termination**: Stop processing when queues are empty
- **Batch Processing**: Process multiple images simultaneously
- **Pipeline Optimization**: Overlap computation and memory transfers

## Troubleshooting

### Common Issues
1. **CUDA Not Available**: 
   - Run `poetry run python utils/cuda_diagnostic.py`
   - Verify CUDA environment variables
   - Restart WSL2: `wsl --shutdown`

2. **Out of Memory Errors**:
   - Reduce image size or batch count
   - Monitor VRAM usage with `nvidia-smi`
   - Optimize shared memory usage

3. **Performance Issues**:
   - Check GPU utilization with `nvidia-smi`
   - Verify optimal thread block sizes
   - Profile with NVIDIA Nsight

### Performance Verification
```bash
# Test GPU accessibility
poetry run python -c "from numba import cuda; print('CUDA Available:', cuda.is_available())"

# Check memory bandwidth
poetry run python benchmarks/memory_bandwidth_test.py

# Profile kernel execution
poetry run python benchmarks/profile_kernels.py
```

## Contributing

This project demonstrates high-performance GPU computing techniques for image processing. Contributions focusing on algorithm optimization, memory efficiency, or extended CUDA features are welcome.

### Development Setup
1. Fork the repository
2. Set up RTX 4060 8GB development environment
3. Run existing tests to verify setup
4. Implement optimizations or new features
5. Profile performance improvements
6. Submit pull request with benchmarks

## License

This project is intended for educational and research purposes, demonstrating GPU-accelerated flood fill algorithms optimized for modern NVIDIA hardware.

---

**Hardware Optimized For**: NVIDIA GeForce RTX 4060 8GB Laptop GPU  
**CUDA Version**: 12.9  
**Architecture**: Ada Lovelace (Compute Capability 8.9)  
**Maximum Performance**: 8GB VRAM, 256 GB/s memory bandwidth, 24 multiprocessors
