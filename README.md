# CUDA Flood Fill - Evolution Roadmap

> **Vision**: Demonstrate the evolution from simple sequential CPU algorithms to complex parallel GPU implementations

## Project Philosophy

This project showcases the journey from simple to complex, sequential to parallel:

1. **Start Simple**: Begin with a basic CPU BFS that anyone can understand
2. **Add Complexity**: Show how more sophisticated algorithms (scan) work on CPU
3. **Introduce Parallelism**: Move the simple algorithm to GPU
4. **Optimize**: Combine the best of both worlds (GPU + advanced algorithms)

## Evolution Path (The Main Story)

### Step 1: `cpu-bfs-flood-fill` - The Foundation
**Goal**: Create the simplest possible flood-fill implementation

- **Algorithm**: Basic Breadth-First Search (BFS)
- **Platform**: CPU (Sequential)
- **Focus**: Clarity and correctness
- **Deliverables**:
  - Simple, readable BFS implementation
  - Basic visualization
  - Performance baseline
  - Documentation explaining the algorithm

**Key Learning**: Understanding the core flood-fill problem

### Step 2: `cpu-scan-flood-fill` - Algorithm Evolution
**Goal**: Introduce more sophisticated algorithms while staying on CPU

- **Algorithm**: Connected Component Labeling with scanning techniques
- **Platform**: CPU (Still sequential, but more advanced)
- **Focus**: Algorithm sophistication
- **Deliverables**:
  - Scan-based flood fill implementation
  - Comparison with BFS approach
  - Performance analysis (algorithm complexity)
  - Documentation of scan techniques

**Key Learning**: How algorithm choice affects performance, even on the same hardware

### Step 3: `gpu-bfs-flood-fill` - Platform Evolution
**Goal**: Take the simple algorithm and make it parallel

- **Algorithm**: BFS (same as Step 1)
- **Platform**: GPU (Parallel)
- **Focus**: Parallelization strategies
- **Deliverables**:
  - CUDA kernel for parallel BFS
  - Memory management strategies
  - Thread block optimization
  - Performance comparison with CPU version

**Key Learning**: How parallelization can speed up the same algorithm

### Step 4: `gpu-scan-flood-fill` - Full Optimization
**Goal**: Combine the best algorithms with the best hardware

- **Algorithm**: Advanced scanning techniques
- **Platform**: GPU (Fully optimized)
- **Focus**: Maximum performance
- **Deliverables**:
  - Optimized CUDA kernels
  - Advanced memory usage patterns
  - Benchmark suite
  - Performance analysis report

**Key Learning**: How hardware and algorithms work together for optimal performance

## Supporting Development

### Development Branches (`dev-*`)
- `dev-profiling`: Performance measurement and analysis tools
- `dev-optimization`: General optimization experiments
- `dev-memory-optimization`: Memory usage optimization
- `dev-documentation`: Documentation improvements
- `dev-testing`: Test suite development

### Experimental Branches (`exp-*`)
- `exp-pycuda`: Alternative CUDA binding experiments
- `exp-cupy`: CuPy-based implementations
- `exp-two-pass-algorithm`: Alternative algorithm approaches

### Infrastructure Branches (`infra-*`)
- `infra-project-setup`: Build system and project configuration
- `infra-poetry-setup`: Dependency management
- `infra-ci-cd`: Continuous integration setup

## Benchmarking Strategy

### Performance Metrics to Track
1. **Execution Time**: How fast each approach runs
2. **Memory Usage**: Peak and average memory consumption (out of 8 GB)
3. **GPU Utilization**: SM occupancy and throughput efficiency
4. **Memory Bandwidth**: Utilization of 128-bit GDDR6 bus
5. **Scalability**: Performance with different image sizes
6. **Thread Efficiency**: Warp utilization and divergence analysis
7. **Shared Memory Usage**: Efficiency of 48 KB per block allocation

## Target Hardware: NVIDIA RTX 4060 Laptop GPU

### GPU Specifications
- **Device**: NVIDIA GeForce RTX 4060 Laptop GPU
- **Architecture**: Ada Lovelace
- **Compute Capability**: 8.9
- **Total Memory**: 8.00 GB GDDR6
- **Memory Bus Width**: 128-bit
- **Memory Clock**: 8001 MHz
- **Base Clock**: 1470 MHz

### Technical Architecture Details

#### Streaming Multiprocessors (SMs)
- **SM Count**: 24 Streaming Multiprocessors
- **Max Threads per Block**: 1024
- **Max Block Dimensions**: (1024, 1024, 64)
- **Warp Size**: 32 threads
- **Max Registers per Block**: 65,536
- **Max Registers per SM**: 65,536

#### Memory Hierarchy
```
┌─────────────────────────────────────────────────────────────┐
│                    Global Memory (8 GB)                     │
├─────────────────────────────────────────────────────────────┤
│                   L2 Cache (32 MB)                         │
├─────────────────────────────────────────────────────────────┤
│  Shared Memory per Block: 48 KB (up to 99 KB with opt-in)  │
├─────────────────────────────────────────────────────────────┤
│          Constant Memory: 64 KB (cached)                   │
├─────────────────────────────────────────────────────────────┤
│               Registers: 65,536 per block                  │
└─────────────────────────────────────────────────────────────┘
```

#### Optimization Guidelines for RTX 4060
- **Optimal Thread Block Sizes**: 256, 512, or 1024 threads
- **Shared Memory Strategy**: Up to 48 KB per block (99 KB with opt-in)
- **Grid Sizing**: Consider 24 SMs for optimal occupancy
- **Memory Bandwidth**: 128-bit bus optimized for GDDR6
- **Concurrent Kernels**: Supported for overlapping execution

### CUDA Programming Model Context

#### Thread Hierarchy
```
Grid (up to 2^31-1 × 65,535 × 65,535)
├── Block (up to 1024 × 1024 × 64)
    ├── Warp (32 threads - execution unit)
    └── Thread (individual processing unit)
```

#### Execution Hierarchy with Constraints and Memory Levels

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                               GRID LEVEL                                    │
│  Memory: Global Memory (8 GB), Constant Memory (64 KB)                     │
│  Constraints: Max dimensions (2^31-1 × 65,535 × 65,535)                    │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                         SM LEVEL (24 SMs)                            │ │
│  │  Memory: L2 Cache (32 MB shared across all SMs)                      │ │
│  │  Constraints: Max blocks per SM depends on resource usage             │ │
│  │  Resource Limits:                                                     │ │
│  │    • 65,536 registers per SM                                          │ │
│  │    • 102,400 bytes (100 KB) shared memory per SM                     │ │
│  │    • Max 1024 threads per block                                       │ │
│  │                                                                       │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │ │
│  │  │                    BLOCK LEVEL                                  │ │ │
│  │  │  Memory: Shared Memory (48 KB per block, up to 99 KB opt-in)   │ │ │
│  │  │  Constraints:                                                   │ │ │
│  │  │    • Max 1024 threads per block                                 │ │ │
│  │  │    • Max dimensions (1024 × 1024 × 64)                         │ │ │
│  │  │    • Max 65,536 registers per block                            │ │ │
│  │  │    • Shared memory allocation affects occupancy               │ │ │
│  │  │                                                                 │ │ │
│  │  │  ┌───────────────────────────────────────────────────────────┐ │ │ │
│  │  │  │                   WARP LEVEL (32 threads)                 │ │ │ │
│  │  │  │  Memory: L1 Cache, Texture Cache                          │ │ │ │
│  │  │  │  Constraints:                                              │ │ │ │
│  │  │  │    • Fixed size: exactly 32 threads                       │ │ │ │
│  │  │  │    • SIMD execution (lock-step)                           │ │ │ │
│  │  │  │    • Branch divergence causes serialization               │ │ │ │
│  │  │  │    • Memory coalescing requirements                       │ │ │ │
│  │  │  │                                                            │ │ │ │
│  │  │  │  ┌─────────────────────────────────────────────────────┐ │ │ │ │
│  │  │  │  │              THREAD LEVEL                           │ │ │ │ │
│  │  │  │  │  Memory: Registers (per-thread private)             │ │ │ │ │
│  │  │  │  │  Constraints:                                        │ │ │ │ │
│  │  │  │  │    • Max registers limited by total per block       │ │ │ │ │
│  │  │  │  │    • Local memory spillover for excess variables    │ │ │ │ │
│  │  │  │  │    • Private register space                         │ │ │ │ │
│  │  │  │  └─────────────────────────────────────────────────────┘ │ │ │ │
│  │  │  └───────────────────────────────────────────────────────────┘ │ │ │
│  │  └─────────────────────────────────────────────────────────────────┘ │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Resource Allocation Examples for RTX 4060

**Scenario 1: Memory-Bound Configuration**
```
Block Configuration: 512 threads, 48 KB shared memory
├── SM Capacity: 2 blocks per SM (48 KB × 2 = 96 KB < 100 KB limit)
├── Warp Count: 16 warps per block (512 ÷ 32)
└── Register Usage: ~127 registers per thread (65,536 ÷ 512)
```

**Scenario 2: Register-Bound Configuration**
```
Block Configuration: 1024 threads, 24 KB shared memory  
├── SM Capacity: 4 blocks per SM (24 KB × 4 = 96 KB < 100 KB limit)
├── Warp Count: 32 warps per block (1024 ÷ 32)
└── Register Usage: ~64 registers per thread (65,536 ÷ 1024)
```

**Scenario 3: Thread-Bound Configuration**
```
Block Configuration: 256 threads, 12 KB shared memory
├── SM Capacity: 8 blocks per SM (limited by other factors)
├── Warp Count: 8 warps per block (256 ÷ 32)  
└── Register Usage: ~256 registers per thread (65,536 ÷ 256)
```

#### Memory Access Patterns
- **Coalesced Access**: Maximize memory throughput with aligned, contiguous access
- **Shared Memory Banks**: 32 banks, avoid bank conflicts
- **Texture Cache**: Available for 2D spatial locality
- **Constant Cache**: Optimized for uniform access across warps

### GPU Execution Model Deep Dive

#### How SMs Manage Blocks

Each Streaming Multiprocessor (SM) is responsible for executing one or more thread blocks. The GPU scheduler distributes blocks across the 24 available SMs based on resource availability:

```
SM Resource Allocation:
┌─────────────────────────────────────────┐
│ SM 0: Block 0, Block 4, Block 8, ...   │
│ SM 1: Block 1, Block 5, Block 9, ...   │ 
│ SM 2: Block 2, Block 6, Block 10, ...  │
│ SM 3: Block 3, Block 7, Block 11, ...  │
│ ...                                     │
│ SM 23: Block 23, Block 47, ...         │
└─────────────────────────────────────────┘
```

**Block Assignment Constraints:**
- **Shared Memory**: Each block needs ≤48 KB, so max 2 blocks per SM if using full allocation
- **Registers**: 65,536 registers per SM limits concurrent blocks based on register usage per thread
- **Threads**: Max 1024 threads per block, with SM supporting multiple blocks simultaneously

#### How Blocks Manage Warps

Within each block, threads are organized into warps of 32 threads each. The warp scheduler executes warps in a round-robin fashion:

```cuda
// Example: 512-thread block organization
Block (16×32 threads):
├── Warp 0:  threads [0-31]     (row 0: columns 0-31)
├── Warp 1:  threads [32-63]    (row 1: columns 0-31) 
├── Warp 2:  threads [64-95]    (row 2: columns 0-31)
├── ...
└── Warp 15: threads [480-511]  (row 15: columns 0-31)
```

#### Spatial Distribution Examples for Flood Fill

**Example 1: 2D Image Processing Distribution**
```cuda
// Grid: (25, 25) blocks for 400×400 image
// Block: (16, 16) threads per block
__global__ void flood_fill_kernel(int* image, int width, int height) {
    // SM assignment based on block coordinates
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int smId = blockId % 24;  // Round-robin across 24 SMs
    
    // Thread coordinates within image
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Warp assignment within block
    int warpId = (threadIdx.y * blockDim.x + threadIdx.x) / 32;
    int laneId = (threadIdx.y * blockDim.x + threadIdx.x) % 32;
}
```

**Example 2: Load Balancing for Irregular Workloads**
```cuda
// Dynamic work distribution for variable blob sizes
__global__ void dynamic_flood_fill(int* work_queue, int queue_size) {
    // Global thread ID for work stealing
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    
    // Each thread processes multiple work items
    for (int work_id = tid; work_id < queue_size; work_id += total_threads) {
        // Process work_queue[work_id]
        // This ensures load balancing across all threads
    }
}
```

#### Problem-Solving Applications

**1. SM-Level Problem Solving:**
- **Parallel Processing**: Multiple independent regions processed simultaneously
- **Resource Management**: Balancing memory and compute resources across blocks
- **Fault Tolerance**: If one SM stalls, others continue processing

**2. Block-Level Problem Solving:**
- **Shared Memory Coordination**: BFS queue shared among threads in a block
- **Synchronization**: `__syncthreads()` coordinates within-block operations
- **Local Communication**: Fast inter-thread communication via shared memory

**3. Warp-Level Problem Solving:**
- **SIMD Execution**: All 32 threads execute same instruction simultaneously
- **Memory Coalescing**: Contiguous memory access across warp lanes
- **Branch Divergence Handling**: Different execution paths within warps

#### Real-World Flood Fill Distribution Strategies

**Strategy 1: Spatial Decomposition**
```
Image divided into 16×16 pixel tiles:
┌─────┬─────┬─────┬─────┐
│ B0  │ B1  │ B2  │ B3  │  ← Blocks 0-3 → SM 0-3
├─────┼─────┼─────┼─────┤
│ B4  │ B5  │ B6  │ B7  │  ← Blocks 4-7 → SM 0-3
├─────┼─────┼─────┼─────┤
│ B8  │ B9  │ B10 │ B11 │  ← Blocks 8-11 → SM 0-3
└─────┴─────┴─────┴─────┘
```

**Strategy 2: Work Queue Distribution**
```
Global work queue with blob starting points:
Queue: [blob1, blob2, blob3, blob4, ...]
       ↓      ↓      ↓      ↓
     SM0    SM1    SM2    SM3   (round-robin assignment)
```

**Strategy 3: Hierarchical Processing**
```
Level 1: SMs process different image regions
Level 2: Blocks within SM handle sub-regions  
Level 3: Warps process scanlines or pixel groups
Level 4: Threads handle individual pixels
```

### Implementation Considerations for Flood Fill

#### Thread Block Design
```cuda
// Optimal configurations for RTX 4060
Block Size Options:
├── 16×16 = 256 threads (good for 2D problems)
├── 32×16 = 512 threads (balanced approach)  
└── 32×32 = 1024 threads (maximum occupancy)
```

#### Memory Usage Strategy
- **Global Memory**: Store input/output images (utilize 8 GB capacity)
- **Shared Memory**: Queue storage for BFS (48 KB per block)
- **Constant Memory**: Direction vectors and small lookup tables (64 KB)
- **Registers**: Temporary variables and loop counters (65K per block)

#### Parallelization Challenges
1. **Load Balancing**: Different blob sizes create uneven work distribution
2. **Memory Coalescing**: 2D image access patterns can cause uncoalesced reads
3. **Thread Divergence**: Conditional processing based on pixel colors
4. **Synchronization**: Coordinating queue operations across threads
5. **Occupancy**: Balancing register usage vs. thread count per SM

## Implementation Phases

### Phase 1: Foundation
- [ ] Implement `cpu-bfs-flood-fill`
- [ ] Create basic test images
- [ ] Set up benchmarking framework
- [ ] Document the BFS algorithm

### Phase 2: Algorithm Evolution
- [ ] Implement `cpu-scan-flood-fill`
- [ ] Compare algorithms on CPU
- [ ] Document scanning techniques
- [ ] Create performance comparison

### Phase 3: Parallelization
- [ ] Implement `gpu-bfs-flood-fill`
- [ ] Optimize CUDA kernels for RTX 4060
- [ ] Memory management optimization
- [ ] GPU vs CPU performance analysis

### Phase 4: Full Optimization
- [ ] Implement `gpu-scan-flood-fill`
- [ ] Advanced CUDA optimizations for Ada Lovelace
- [ ] Complete benchmark suite
- [ ] Final performance report

### Phase 5: Polish & Documentation
- [ ] Complete documentation
- [ ] Create demo notebooks
- [ ] Performance visualization
- [ ] Project presentation materials

## Educational Value

### For Beginners
- Start with `cpu-bfs-flood-fill` to understand the problem
- See clear, simple code that solves the problem
- Learn basic flood-fill concepts

### For Intermediate Developers
- Progress through `cpu-scan-flood-fill` to see algorithm alternatives
- Understand how algorithm choice affects performance
- Learn about connected component labeling

### For Advanced Developers
- Dive into `gpu-bfs-flood-fill` for parallelization techniques
- Master `gpu-scan-flood-fill` for high-performance computing
- Understand memory optimization and CUDA best practices

## Learning Outcomes

By the end of this evolution path, developers will understand:

1. **Problem Decomposition**: How to break down a complex problem
2. **Algorithm Selection**: When to use different algorithms
3. **Platform Migration**: How to move from CPU to GPU
4. **Performance Optimization**: Memory, threading, and hardware optimization
5. **Benchmarking**: How to measure and compare performance
6. **Real-world Trade-offs**: Complexity vs. performance vs. maintainability

## Branch Workflow

```mermaid
graph LR
    A[main] --> B[cpu/bfs-flood-fill]
    B --> C[cpu/scanning+bfs-flood-fill]
    C --> D[gpu/bfs-flood-fill]
    D --> E[gpu/scanning+bfs-flood-fill]
```

## Success Criteria

- [ ] All four evolution steps implemented and documented
- [ ] Clear performance progression demonstrated
- [ ] Educational materials created for each step
- [ ] Comprehensive benchmark results
- [ ] Working demo for each evolution step
- [ ] Documentation that tells the complete story

---

*This roadmap ensures that every step builds logically on the previous one, creating a compelling narrative of optimization and scaling.*
