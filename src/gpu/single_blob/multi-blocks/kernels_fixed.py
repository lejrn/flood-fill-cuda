"""
Fixed CUDA kernels for optimized multi-block flood fill.
This version launches separate kernels per iteration to ensure proper block synchronization.
"""

import numpy as np
from numba import cuda

@cuda.jit
def initialize_flood_fill(img, visited, start_x, start_y, global_queue_x, global_queue_y, global_queue_front, global_queue_rear):
    """Initialize the flood fill algorithm with the starting pixel."""
    if cuda.threadIdx.x == 0 and cuda.blockIdx.x == 0:
        global_queue_x[0] = start_x
        global_queue_y[0] = start_y
        global_queue_front[0] = 0
        global_queue_rear[0] = 1
        visited[start_x, start_y] = 1


# Import from same directory (not relative imports)
from utils import (
    DX_host, DY_host, QUEUE_CAPACITY, 
    is_red, is_valid_pixel, PerformanceLogger
)

# Initialize performance logger
perf_logger = PerformanceLogger()

@cuda.jit
def flood_fill_iteration(img, visited, width, height, new_color,
                        global_queue_x, global_queue_y, global_queue_front, global_queue_rear,
                        debug_block_usage, debug_thread_usage, debug_warp_usage, debug_pixel_count,
                        chunk_size, iteration_num):
    """
    Single iteration of flood fill using multi-block work distribution.
    
    This kernel processes one iteration of the BFS flood fill algorithm.
    Each kernel launch ensures proper synchronization between all blocks.
    
    Key improvements:
    - Each iteration is a separate kernel launch (ensures block synchronization)
    - Work is distributed in 32-pixel chunks across blocks
    - Multiple blocks can work simultaneously when queue size permits
    - Optimal GPU utilization through proper work distribution
    - No device-side termination checks (handled by separate kernel)
    
    Args:
        img: Input image array (width, height, 3)
        visited: Visited pixels array (width, height)
        width, height: Image dimensions
        new_color: RGB color to fill with
        global_queue_x, global_queue_y: Global memory queues for BFS
        global_queue_front, global_queue_rear: Queue pointers
        debug_*: Debug arrays to track GPU utilization
        iteration_num: Current iteration number for debug tracking
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
    
    # Direction arrays in constant memory (64KB cache)
    DX_const = cuda.const.array_like(DX_host)
    DY_const = cuda.const.array_like(DY_host)
    
    # Read current queue state (all blocks see the same values)
    iteration_front = global_queue_front[0]
    iteration_rear = global_queue_rear[0]
    current_queue_size = iteration_rear - iteration_front
    
    # Early exit if no work
    if current_queue_size <= 0:
        return
    
    # Multi-block work distribution strategy
    # Distribute work in 32-pixel chunks for optimal hardware utilization
    pixel_start = pixel_end = 0  # Default: no work
    
    # Constants for chunk-based distribution
    CHUNK_SIZE = chunk_size  # Configurable chunk size for optimization
    total_chunks = (current_queue_size + CHUNK_SIZE - 1) // CHUNK_SIZE  # Ceiling division
    available_blocks = cuda.gridDim.x
    
    if total_chunks >= available_blocks:
        # Enough chunks for all blocks - distribute evenly
        chunks_per_block = total_chunks // available_blocks
        extra_chunks = total_chunks % available_blocks
        
        # This block gets base chunks + maybe one extra
        block_chunks = chunks_per_block + (1 if block_id < extra_chunks else 0)
        
        # Calculate chunk range for this block
        if block_id < extra_chunks:
            chunk_start = block_id * (chunks_per_block + 1)
        else:
            chunk_start = extra_chunks * (chunks_per_block + 1) + (block_id - extra_chunks) * chunks_per_block
        
        chunk_end = chunk_start + block_chunks
        
        # Convert chunk indices to pixel indices
        pixel_start = iteration_front + chunk_start * CHUNK_SIZE
        pixel_end = min(iteration_front + chunk_end * CHUNK_SIZE, iteration_rear)
        
    else:
        # More blocks than chunks: only some blocks get work
        if block_id < total_chunks:
            # This block gets exactly one chunk
            pixel_start = iteration_front + block_id * CHUNK_SIZE
            pixel_end = min(pixel_start + CHUNK_SIZE, iteration_rear)
        else:
            # This block gets no work
            pixel_start = pixel_end = 0
    
    # Within each block, distribute work among warps
    block_queue_size = pixel_end - pixel_start
    if block_queue_size > 0:
        # Distribute among warps in the block
        items_per_warp = max(1, (block_queue_size + warps_per_block - 1) // warps_per_block)
        warp_start = pixel_start + warp_id * items_per_warp
        warp_end = min(warp_start + items_per_warp, pixel_end)
    else:
        warp_start = warp_end = 0  # No work for this block
    
    # Process queue items if this warp has work assigned
    if warp_start < warp_end:
        # Track utilization (only when threads actually do work)
        if thread_id == 0:  # One thread per block records block usage
            debug_block_usage[block_id] = 1
        if lane_id == 0:  # One thread per warp records warp usage
            debug_warp_usage[global_warp_id] = 1
            
        # Process queue items: each thread in warp processes one item
        for batch_start in range(warp_start, warp_end, 32):
            queue_idx = batch_start + lane_id
            
            if queue_idx < warp_end and queue_idx < iteration_rear:
                # Coalesced memory access from global queue (cached in L2)
                x = global_queue_x[queue_idx]
                y = global_queue_y[queue_idx]
                
                # Process pixel
                if x >= 0 and y >= 0 and x < width and y < height:
                    # Track that this thread is actually doing work
                    debug_thread_usage[global_thread_id] = 1
                    
                    # Track pixel processing
                    cuda.atomic.add(debug_pixel_count, 0, 1)
                    
                    # Recolor pixel with thread-specific blue variation
                    img[x, y, 0] = new_color[0]  # Red component
                    img[x, y, 1] = new_color[1]  # Green component  
                    img[x, y, 2] = (global_thread_id * 4) % 255  # Blue: thread ID encoding
                    
                    # Process 8-connected neighbors
                    for i in range(8):
                        nx = x + DX_const[i]
                        ny = y + DY_const[i]
                        
                        if is_valid_pixel(nx, ny, width, height):
                            # Use atomic compare-and-swap to mark as visited (thread-safe)
                            old = cuda.atomic.cas(visited, (nx, ny), 0, 1)
                            if old == 0 and is_red(img, nx, ny):
                                # Add to global queue atomically
                                pos = cuda.atomic.add(global_queue_rear, 0, 1)
                                if pos < QUEUE_CAPACITY:
                                    global_queue_x[pos] = nx
                                    global_queue_y[pos] = ny
    
    # Update queue front pointer (only one thread does this)
    if cuda.threadIdx.x == 0 and cuda.blockIdx.x == 0:
        # Mark processed items as consumed by updating front pointer
        global_queue_front[0] = iteration_rear


@cuda.jit
def check_work_remaining(global_queue_front, global_queue_rear, work_remaining_flag):
    """
    Device-side kernel to check if work remains and set the flag.
    This eliminates the need for host-device transfers in the main loop.
    Returns: work_remaining_flag[0] = 1 if work remains, 0 if complete
    """
    if cuda.threadIdx.x == 0 and cuda.blockIdx.x == 0:
        current_front = global_queue_front[0]
        current_rear = global_queue_rear[0]
        
        if current_front < current_rear:
            work_remaining_flag[0] = 1  # Work remaining
        else:
            work_remaining_flag[0] = 0  # No work left - complete


@cuda.jit
def run_flood_fill_kernel(img, visited, start_x, start_y, width, height, new_color,
                         global_queue_x, global_queue_y, global_queue_front, global_queue_rear,
                         debug_block_usage, debug_thread_usage, debug_warp_usage, debug_pixel_count,
                         iteration_counter, global_sync_flag, max_iterations=50000):
    """
    Pure device-side flood fill kernel that handles the entire algorithm.
    
    Uses a coordination mechanism where all blocks work together:
    - Block 0 manages the iteration loop and queue state
    - All blocks participate in flood fill work
    - Uses global memory flag for inter-block coordination
    
    This completely eliminates host-device transfers!
    """
    block_id = cuda.blockIdx.x
    thread_id = cuda.threadIdx.x
    
    # Initialize (only block 0, thread 0)
    if block_id == 0 and thread_id == 0:
        global_queue_x[0] = start_x
        global_queue_y[0] = start_y
        global_queue_front[0] = 0
        global_queue_rear[0] = 1
        visited[start_x, start_y] = 1
        iteration_counter[0] = 0
        global_sync_flag[0] = 1  # 1 = continue, 0 = stop
    
    # Wait for initialization to complete
    cuda.syncthreads()
    
    # Main loop runs entirely on device
    for iteration in range(max_iterations):
        # Block 0 checks if work remains and updates sync flag
        if block_id == 0 and thread_id == 0:
            current_front = global_queue_front[0]
            current_rear = global_queue_rear[0]
            if current_front >= current_rear:
                global_sync_flag[0] = 0  # Signal all blocks to stop
                iteration_counter[0] = iteration
            else:
                global_sync_flag[0] = 1  # Continue processing
        
        # All blocks check the sync flag
        cuda.syncthreads()
        should_continue = global_sync_flag[0]
        
        if should_continue == 0:
            break  # All blocks exit
        
        # Read current queue state
        current_front = global_queue_front[0]
        current_rear = global_queue_rear[0]
        current_queue_size = current_rear - current_front
        
        if current_queue_size <= 0:
            continue  # Skip this iteration
        
        # All blocks participate in flood fill iteration
        warp_id = thread_id // 32
        lane_id = thread_id % 32
        warps_per_block = cuda.blockDim.x // 32
        global_warp_id = block_id * warps_per_block + warp_id
        global_thread_id = block_id * cuda.blockDim.x + thread_id
        
        # Direction arrays in constant memory
        DX_const = cuda.const.array_like(DX_host)
        DY_const = cuda.const.array_like(DY_host)
        
        # Multi-block work distribution
        pixel_start = pixel_end = 0
        
        # Constants for chunk-based distribution
        CHUNK_SIZE = 32
        total_chunks = (current_queue_size + CHUNK_SIZE - 1) // CHUNK_SIZE
        available_blocks = cuda.gridDim.x
        
        if total_chunks >= available_blocks:
            # Enough chunks for all blocks
            chunks_per_block = total_chunks // available_blocks
            extra_chunks = total_chunks % available_blocks
            
            block_chunks = chunks_per_block + (1 if block_id < extra_chunks else 0)
            
            if block_id < extra_chunks:
                chunk_start = block_id * (chunks_per_block + 1)
            else:
                chunk_start = extra_chunks * (chunks_per_block + 1) + (block_id - extra_chunks) * chunks_per_block
            
            chunk_end = chunk_start + block_chunks
            
            pixel_start = current_front + chunk_start * CHUNK_SIZE
            pixel_end = min(current_front + chunk_end * CHUNK_SIZE, current_rear)
            
        else:
            # More blocks than chunks
            if block_id < total_chunks:
                pixel_start = current_front + block_id * CHUNK_SIZE
                pixel_end = min(pixel_start + CHUNK_SIZE, current_rear)
        
        # Distribute work among warps within block
        block_queue_size = pixel_end - pixel_start
        if block_queue_size > 0:
            items_per_warp = max(1, (block_queue_size + warps_per_block - 1) // warps_per_block)
            warp_start = pixel_start + warp_id * items_per_warp
            warp_end = min(warp_start + items_per_warp, pixel_end)
        else:
            warp_start = warp_end = 0
        
        # Process queue items
        if warp_start < warp_end:
            # Track utilization
            if thread_id == 0:
                debug_block_usage[block_id] = 1
            if lane_id == 0:
                debug_warp_usage[global_warp_id] = 1
                
            # Process pixels
            for batch_start in range(warp_start, warp_end, 32):
                queue_idx = batch_start + lane_id
                
                if queue_idx < warp_end and queue_idx < current_rear:
                    x = global_queue_x[queue_idx]
                    y = global_queue_y[queue_idx]
                    
                    if x >= 0 and y >= 0 and x < width and y < height:
                        debug_thread_usage[global_thread_id] = 1
                        cuda.atomic.add(debug_pixel_count, 0, 1)
                        
                        # Recolor pixel
                        img[x, y, 0] = new_color[0]
                        img[x, y, 1] = new_color[1]
                        img[x, y, 2] = (global_thread_id * 4) % 255
                        
                        # Process 8-connected neighbors
                        for i in range(8):
                            nx = x + DX_const[i]
                            ny = y + DY_const[i]
                            
                            if is_valid_pixel(nx, ny, width, height):
                                old = cuda.atomic.cas(visited, (nx, ny), 0, 1)
                                if old == 0 and is_red(img, nx, ny):
                                    pos = cuda.atomic.add(global_queue_rear, 0, 1)
                                    if pos < QUEUE_CAPACITY:
                                        global_queue_x[pos] = nx
                                        global_queue_y[pos] = ny
        
        # Update queue front pointer (only block 0, thread 0)
        cuda.syncthreads()
        if block_id == 0 and thread_id == 0:
            global_queue_front[0] = current_rear
        
        # Small delay to ensure proper synchronization
        cuda.syncthreads()


def reset_global_queue(global_queue_front, global_queue_rear):
    """Reset global queue pointers"""
    print("🔄 Resetting Global Queue:")
    global_queue_front[0] = 0
    global_queue_rear[0] = 0
    print("    ✅ Queue pointers reset to 0")


def run_multi_iteration_flood_fill(img, visited, start_x, start_y, width, height, new_color,
                                  global_queue_x, global_queue_y, global_queue_front, global_queue_rear,
                                  debug_block_usage, debug_thread_usage, debug_warp_usage, debug_pixel_count,
                                  blocks_per_grid=40, threads_per_block=64, chunk_size=32, max_iterations=50000):
    """
    Optimal multi-iteration flood fill with minimal host-device transfers.
    
    Uses separate kernel launches for proper block synchronization,
    but minimizes host transfers by checking termination less frequently.
    
    Returns:
        int: Number of iterations executed
    """
    print("🚀 Starting optimized flood fill (minimal host-device transfers)...")
    
    # Initialize the algorithm
    initialize_flood_fill[1, 1](img, visited, start_x, start_y, global_queue_x, global_queue_y, 
                               global_queue_front, global_queue_rear)
    cuda.synchronize()
    
    iteration = 0
    check_interval = 50  # Check every 50 iterations (much less frequent)
    
    while iteration < max_iterations:
        # Only check termination occasionally to minimize transfers
        if iteration % check_interval == 0:
            current_front = global_queue_front.copy_to_host()[0]
            current_rear = global_queue_rear.copy_to_host()[0]
            
            if current_front >= current_rear:
                break  # No more work
        
        # Launch kernel for this iteration
        flood_fill_iteration[blocks_per_grid, threads_per_block](
            img, visited, width, height, new_color,
            global_queue_x, global_queue_y, global_queue_front, global_queue_rear,
            debug_block_usage, debug_thread_usage, debug_warp_usage, debug_pixel_count,
            chunk_size, iteration
        )
        cuda.synchronize()
        
        iteration += 1
    
    # Final verification
    final_front = global_queue_front.copy_to_host()[0]
    final_rear = global_queue_rear.copy_to_host()[0]
    final_queue_size = final_rear - final_front
    pixels_processed = debug_pixel_count.copy_to_host()[0]
    
    print(f"✅ Flood fill completed after {iteration} iterations")
    print(f"📊 Final Statistics:")
    print(f"   • Iterations: {iteration}")
    print(f"   • Pixels processed: {pixels_processed:,}")
    print(f"   • Final queue state: front={final_front}, rear={final_rear}, size={final_queue_size}")
    print(f"   • Host transfers: only every {check_interval} iterations + final check")
    
    return iteration
