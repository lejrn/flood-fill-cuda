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

#### Concurrency vs True Parallelism in CUDA Hierarchy

Understanding the difference between **concurrency** (time-sliced execution) and **true parallelism** (simultaneous execution) is crucial for CUDA optimization:

##### **1. Threads in a Warp**
**Type**: **True Parallelism** (SIMD)

**Explanation**:
- All **32 threads** execute the **same instruction simultaneously**
- Each thread processes different data (SIMD: Single Instruction, Multiple Data)
- Hardware executes them in **lockstep** on separate ALUs
- True parallel execution at the instruction level
- **No time-slicing** - all threads execute together

**Example**:
```cuda
// All 32 threads in warp execute this simultaneously
int pixel_value = image[threadIdx.x];  // Different data per thread
```

##### **2. Warps in a Block**
**Type**: **Concurrency** (Time-sliced)

**Explanation**:
- Only **one warp executes at a time** on an SM
- Warps are **scheduled sequentially** by the warp scheduler
- When one warp stalls (memory access), another warp is scheduled
- Multiple warps share the same execution units through **time-multiplexing**
- **Resource sharing**: Same compute cores, same shared memory

**Example**:
```cuda
// Timeline on one SM:
// Cycle 1-10:  Warp 0 executes arithmetic
// Cycle 11:    Warp 0 stalls on memory access
// Cycle 12-20: Warp 1 executes (hides Warp 0's latency)
// Cycle 21:    Warp 0 resumes (memory arrived)
```

##### **3. Blocks in a Streaming Multiprocessor (SM)**
**Type**: **Concurrency** (Resource sharing)

**Explanation**:
- Multiple blocks **share the same SM resources** (cores, shared memory, registers)
- Blocks are scheduled concurrently but compete for resources
- Limited by shared memory (100KB) and register count (64K)
- **Time-sliced execution**: When one block's warps stall, another block's warps execute
- In RTX 4060: up to 6 blocks per SM with optimal resource allocation

**Example**:
```cuda
// SM Resource Allocation:
// Block 0: Uses 48KB shared memory, 8 warps
// Block 1: Uses 48KB shared memory, 8 warps  
// Total: 96KB < 100KB limit, so both blocks fit
// Execution: Block 0 and Block 1 warps interleave
```

##### **4. SMs in a Grid (GPU)**
**Type**: **True Parallelism** (Physical separation)

**Explanation**:
- Each of the **24 SMs** is a physically separate processor
- All SMs can execute **simultaneously and independently**
- No resource sharing between SMs (except global memory and L2 cache)
- True parallel execution at the hardware level
- **Maximum parallelism**: 24 blocks can execute truly in parallel

**Example**:
```cuda
// All execute simultaneously on different SMs:
// SM 0:  Block 0 processes image region (0,0) to (16,16)
// SM 1:  Block 1 processes image region (16,0) to (32,16)
// SM 2:  Block 2 processes image region (32,0) to (48,16)
// ...
// SM 23: Block 23 processes image region (368,0) to (384,16)
```

#### Summary: Concurrency vs Parallelism in CUDA

| **Execution Level** | **Type** | **Execution Model** | **Key Characteristics** |
|---------------------|----------|---------------------|-------------------------|
| **Threads in Warp** | **True Parallelism** | SIMD (32 threads) | Same instruction, parallel ALUs, lockstep execution |
| **Warps in Block** | **Concurrency** | Time-sliced scheduling | Warp scheduler, zero-overhead context switch, resource sharing |
| **Blocks in SM** | **Concurrency** | Resource sharing | Compete for shared memory/registers, interleaved execution |
| **SMs in GPU** | **True Parallelism** | Physical separation | Independent processors, simultaneous execution |

#### Memory Access Hierarchy and Latencies

Understanding memory access patterns and latencies is critical for optimizing CUDA applications:

##### **Memory Hierarchy Overview**
```
┌─────────────────────────────────────────────────────────────┐
│                    Global Memory (8 GB)                     │
│                   Latency: 400-800 cycles                   │
│                   Bandwidth: 128-bit @ 8001 MHz            │
├─────────────────────────────────────────────────────────────┤
│                   L2 Cache (32 MB)                         │
│                   Latency: 200-400 cycles                   │
│                   Shared across all 24 SMs                 │
├─────────────────────────────────────────────────────────────┤
│  L1 Cache + Shared Memory (per SM)                         │
│  L1 Cache Latency: 80-120 cycles                           │
│  Shared Memory Latency: 1-32 cycles                        │
│  Combined: 128KB (configurable split)                      │
├─────────────────────────────────────────────────────────────┤
│          Constant Memory: 64 KB (cached)                   │
│          Latency: 1-2 cycles (if cached)                   │
├─────────────────────────────────────────────────────────────┤
│               Registers: 65,536 per SM                     │
│               Latency: 1 cycle (immediate)                 │
└─────────────────────────────────────────────────────────────┘
```

##### **Memory Access by Execution Unit**

**Threads (within a warp) - Memory Access**:
- **Registers**: 1 cycle (immediate, private per thread)
- **Shared Memory**: 1-32 cycles (potential bank conflicts)
- **L1 Cache**: 80-120 cycles (if hit)
- **L2 Cache**: 200-400 cycles (if L1 miss)
- **Global Memory**: 400-800 cycles (if L2 miss)

**Key Insight**: Memory coalescing across 32 threads in a warp is crucial

**Warps (within a block) - Memory Access**:
- **Shared Memory Contention**: When multiple warps access same banks
- **Memory Coalescing**: Optimal when 32 threads access contiguous memory
- **Cache Locality**: L1 cache shared among warps in same block
- **Latency Hiding**: While one warp waits for memory, others execute

**Blocks (within an SM) - Memory Access**:
- **Shared Memory**: Isolated (each block has own allocation)
- **L1 Cache**: Shared among all blocks in same SM
- **Resource Competition**: Multiple blocks compete for cache space
- **Memory Bandwidth**: Blocks share SM's memory controllers

**SMs (across GPU) - Memory Access**:
- **L2 Cache**: Shared resource, potential contention across 24 SMs
- **Global Memory**: All SMs compete for memory bandwidth
- **Memory Controllers**: Multiple controllers distribute access
- **NUMA Effects**: Memory locality can affect performance

##### **Latency Hiding Through Concurrency**

**The Power of Multiple Warps**:
```cuda
// Timeline showing latency hiding:
// Cycle 1-10:  Warp 0 executes arithmetic operations
// Cycle 11:    Warp 0 issues global memory load (400+ cycle latency)
// Cycle 12-20: Warp 1 executes (hides Warp 0's memory latency)
// Cycle 21-30: Warp 2 executes (continues hiding latency)
// Cycle 31-40: Warp 3 executes (latency still hidden)
// ...
// Cycle 400+:  Warp 0's memory data arrives, ready to execute
```

**Optimal Configuration for RTX 4060**:
```cuda
// Maximize latency hiding:
threads_per_block = 256;  // 8 warps per block
blocks_per_grid = 24;     // All SMs utilized
// Total: 192 warps across GPU for maximum latency hiding
```

##### **Resource Sharing and Time-Slicing Details**

**Within a Block (Warp Concurrency)**:
- **Shared Resources**: 
  - Compute cores (CUDA cores, special function units)
  - Shared memory (48KB divided among all warps)
  - Register file (65,536 registers shared by all threads)
- **Time-Slicing Mechanism**:
  - Warp scheduler maintains ready queue of warps
  - Round-robin scheduling with priority for ready warps
  - Stalled warps (memory access) moved to wait queue
  - Context switching between warps has zero overhead

**Within an SM (Block Concurrency)**:
- **Shared Resources**:
  - All execution units (schedulers, cores, memory controllers)
  - L1 cache/shared memory complex (128KB total)
  - Register file (65,536 total across all blocks)
- **Resource Allocation**:
  - Blocks compete for shared memory allocation
  - Register usage per thread affects max blocks per SM
  - Memory bandwidth shared among all resident blocks

#### Summary: Memory Access Latencies

| **Memory Type** | **Latency (cycles)** | **Latency (ns)** | **Access Scope** | **Size (RTX 4060)** |
|-----------------|---------------------|------------------|------------------|---------------------|
| **Registers** | 1 | ~0.7 | Per-thread | 65,536 per SM |
| **Shared Memory** | 1-32 | ~0.7-22 | Per-block | 48 KB per block |
| **L1 Cache** | 80-120 | ~55-82 | Per-SM | Part of 128KB |
| **L2 Cache** | 200-400 | ~136-272 | GPU-wide | 32 MB |
| **Global Memory** | 400-800 | ~272-545 | GPU-wide | 8 GB |

#### Performance Optimization Principles

**1. Maximize True Parallelism**:
- Use all 24 SMs simultaneously
- Configure blocks to utilize full GPU capacity
- Prefer `blocks_per_grid = 24` or multiples

**2. Leverage Concurrency for Latency Hiding**:
- Use 8+ warps per block for effective latency hiding
- Balance register usage to maximize occupancy
- Optimize shared memory access patterns

**3. Optimize Memory Access Hierarchy**:
- Minimize global memory access through data locality
- Use shared memory for frequently accessed data
- Ensure coalesced memory access patterns
- Avoid bank conflicts in shared memory

**4. Balance Resource Utilization**:
- Monitor occupancy vs. resource usage trade-offs
- Use profiling tools to identify bottlenecks
- Consider memory bandwidth vs. compute intensity

**For Your Flood Fill Implementation**:
```cuda
// Optimal configuration for RTX 4060:
threads_per_block = 256;  // 8 warps (good concurrency)
blocks_per_grid = 24;     // Full SM utilization (true parallelism)
shared_memory_per_block = 24576;  // 24KB (half of available)

// This gives you:
// - 6,144 threads total (good utilization)
// - 192 warps across GPU (excellent latency hiding)
// - 24 independent processing units (maximum parallelism)
// - Balanced resource usage (registers, shared memory, occupancy)
```

---

#### Performance Implications

1. **Memory Coalescing**: 
   - Coalesced access: 1 transaction for 32 threads
   - Non-coalesced: Up to 32 transactions for 32 threads
   - **Performance difference**: Up to 32x

2. **Bank Conflicts in Shared Memory**:
   - No conflicts: 1 cycle access
   - 2-way conflict: 2 cycles (serialized)
   - 32-way conflict: 32 cycles (fully serialized)

3. **Occupancy vs Resource Usage**:
   - More registers per thread → fewer threads per SM
   - More shared memory per block → fewer blocks per SM
   - **Sweet spot**: Balance occupancy with resource needs

**Flood Fill Optimization Strategy**:

```cuda
// Resource allocation for optimal performance:
Block Configuration:
├── 256 threads per block (8 warps for latency hiding)
├── 24KB shared memory per block (queue storage)
├── ~100 registers per thread (balanced usage)
└── 4 blocks per SM (good occupancy)

Memory Strategy:
├── Coalesced global memory reads for image data
├── Shared memory for BFS queue (low latency)
├── Register spilling minimized (keep < 100 registers/thread)
└── Atomic operations on shared memory (avoid global atomics)
```
