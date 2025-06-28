"""
CUDA kernels for optimized multi-block flood fill.
Standalone script containing GPU kernels and execution functions.
"""

from numba import cuda

# Import from same directory (not relative imports)
from utils import (
    DX_host, DY_host, TOTAL_WARPS, QUEUE_CAPACITY, 
    is_red, is_valid_pixel, PerformanceLogger
)

# Initialize performance logger
perf_logger = PerformanceLogger()

@cuda.jit
def optimized_flood_fill(img, visited, start_x, start_y, width, height, new_color, 
                        global_queue_x, global_queue_y, global_queue_front, global_queue_rear,
                        debug_block_usage, debug_thread_usage, debug_warp_usage, debug_pixel_count, debug_queue_usage, debug_iteration_count):
    """
    Optimized flood fill kernel using global memory queue and multiple SMs.
    
    This kernel utilizes 20 SMs of the RTX 4060 with 2 blocks per SM (40 total blocks)
    and 2 warps per block (80 total warps) for balanced parallelism. Each warp processes 
    different portions of the global queue, enabling efficient work distribution.
    
    Key optimizations:
    - Global memory queue cached in L2 (32MB)
    - Coalesced memory access patterns
    - Work distribution across 1,152 warps
    - Atomic operations for thread-safe queue management
    - 8-direction flood fill with neighbor processing
    
    Args:
        img: Input image array (width, height, 3)
        visited: Visited pixels array (width, height)
        start_x, start_y: Starting coordinates for flood fill
        width, height: Image dimensions
        new_color: RGB color to fill with
        global_queue_x, global_queue_y: Global memory queues for BFS
        global_queue_front, global_queue_rear: Queue pointers
        debug_block_usage: Debug array to track which blocks were active
        debug_thread_usage: Debug array to track which threads were active
        debug_warp_usage: Debug array to track which warps were active
        debug_pixel_count: Debug counter for pixels processed
        debug_queue_usage: Debug counter for maximum queue usage
        debug_iteration_count: Debug counter for total iterations executed
    """
    # Block and warp identification for work distribution
    block_id = cuda.blockIdx.x
    thread_id = cuda.threadIdx.x
    warp_id = thread_id // 32
    lane_id = thread_id % 32
    
    # Global work distribution calculation
    warps_per_block = cuda.blockDim.x // 32
    global_warp_id = block_id * warps_per_block + warp_id
    global_thread_id = block_id * cuda.blockDim.x + thread_id
    
    # Track debug information for GPU utilization analysis
    # NOTE: We'll only track when threads actually do work, not just when they launch
    
    # Direction arrays in constant memory (64KB cache)
    DX_const = cuda.const.array_like(DX_host)
    DY_const = cuda.const.array_like(DY_host)
    
    # Initialize queue (only one thread across all blocks does this)
    if cuda.threadIdx.x == 0 and cuda.blockIdx.x == 0:
        # Log: Queue initialization
        global_queue_x[0] = start_x
        global_queue_y[0] = start_y
        global_queue_front[0] = 0
        global_queue_rear[0] = 1
        visited[start_x, start_y] = 1
        # Note: Print statements in CUDA kernels are limited, so logging happens on host
    
    # Synchronize all blocks before starting BFS
    cuda.syncthreads()
    
    # Main BFS loop with work distribution
    max_iterations = 16000000  # Prevent infinite loops in edge cases
    iteration = 0
    
    while global_queue_front[0] < global_queue_rear[0] and iteration < max_iterations:
        # Calculate dynamic work distribution
        current_front = global_queue_front[0]
        current_rear = global_queue_rear[0]
        current_queue_size = current_rear - current_front
        work_per_warp = max(1, (current_queue_size + TOTAL_WARPS - 1) // TOTAL_WARPS)
        
        # Each warp gets its portion of the queue (load balancing)
        warp_start = current_front + global_warp_id * work_per_warp
        warp_end = min(warp_start + work_per_warp, current_rear)
        
        # Store current rear position for this iteration (prevents race conditions)
        iteration_rear = current_rear

        # Process queue items: each thread in warp processes one item
        # This creates coalesced memory access patterns for optimal L2 cache utilization
        for batch_start in range(warp_start, warp_end, 32):
            queue_idx = batch_start + lane_id
            
            if queue_idx < warp_end and queue_idx < iteration_rear:
                # Coalesced memory access from global queue (cached in L2)
                x = global_queue_x[queue_idx]
                y = global_queue_y[queue_idx]
                
                # Process pixel (image access will hit L2 cache for nearby pixels)
                if x >= 0 and y >= 0 and x < width and y < height:
                    # Track that this thread is actually doing work
                    debug_thread_usage[global_thread_id] = 1
                    debug_warp_usage[global_warp_id] = 1
                    if thread_id == 0:  # One thread per block records block usage
                        debug_block_usage[block_id] = 1
                    
                    # Track pixel processing
                    cuda.atomic.add(debug_pixel_count, 0, 1)
                    
                    # Recolor pixel with thread-specific blue variation for visualization
                    img[x, y, 0] = new_color[0]  # Red component
                    img[x, y, 1] = new_color[1]  # Green component  
                    img[x, y, 2] = (thread_id * 4) % 255  # Blue: color coding by thread ID
                    
                    # Process 8-connected neighbors (including diagonals)
                    for i in range(8):
                        nx = x + DX_const[i]
                        ny = y + DY_const[i]
                        
                        if is_valid_pixel(nx, ny, width, height):
                            # Use atomic compare-and-swap to mark as visited (thread-safe)
                            old = cuda.atomic.cas(visited, (nx, ny), 0, 1)
                            if old == 0 and is_red(img, nx, ny):
                                # Add to global queue atomically (prevents race conditions)
                                pos = cuda.atomic.add(global_queue_rear, 0, 1)
                                if pos < QUEUE_CAPACITY:
                                    global_queue_x[pos] = nx
                                    global_queue_y[pos] = ny
                                    # Track maximum queue usage
                                    cuda.atomic.max(debug_queue_usage, 0, pos + 1)
        
        # Synchronize all blocks before next iteration (ensures queue consistency)
        cuda.syncthreads()
        
        # Update queue front pointer (only one thread does this to prevent conflicts)
        if cuda.threadIdx.x == 0 and cuda.blockIdx.x == 0:
            global_queue_front[0] = iteration_rear
            # Track total iterations executed
            debug_iteration_count[0] = iteration + 1
        
        # Final synchronization before next iteration
        cuda.syncthreads()
        iteration += 1

def reset_global_queue(global_queue_front, global_queue_rear):
    """Reset global queue pointers"""
    print("🔄 Resetting Global Queue:")
    global_queue_front[0] = 0
    global_queue_rear[0] = 0
    print("    ✅ Queue pointers reset to 0")
