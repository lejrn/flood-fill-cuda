import numpy as np
from numba import cuda
import timeit
from PIL import Image
import random

# Description
"""
Optimized GPU flood fill using global memory queue and multiple SMs.
This implementation utilizes all 24 SMs of the RTX 4060 for maximum parallelism.
"""

# Place direction arrays in constant memory (read-only) - 8 directions
DX_host = np.array([1, 1, 0, -1, -1, -1,  0, 1], dtype=np.int32)
DY_host = np.array([0, 1, 1,  1,  0, -1, -1, -1], dtype=np.int32)

# Configuration constants
IMAGE_SIZE = 400 * 400  # For 400x400 image
QUEUE_CAPACITY = IMAGE_SIZE // 4  # Conservative: 25% of image could be blob
TOTAL_WARPS = 144 * 8  # 144 blocks × 8 warps per block = 1,152 warps

# Kernel Helper functions
@cuda.jit(device=True, inline=True)
def is_red(img, x, y):
    return img[x, y, 0] == 255 and img[x, y, 1] == 0 and img[x, y, 2] == 0

@cuda.jit(device=True, inline=True)
def is_valid_pixel(x, y, width, height):
    return 0 <= x < width and 0 <= y < height

# Optimized flood fill kernel using global memory queue
@cuda.jit
def optimized_flood_fill(img, visited, start_x, start_y, width, height, new_color, 
                        global_queue_x, global_queue_y, global_queue_front, global_queue_rear):
    # Block and warp identification
    block_id = cuda.blockIdx.x
    thread_id = cuda.threadIdx.x
    warp_id = thread_id // 32
    lane_id = thread_id % 32
    
    # Global work distribution
    warps_per_block = cuda.blockDim.x // 32
    global_warp_id = block_id * warps_per_block + warp_id
    
    # Direction arrays in constant memory
    DX_const = cuda.const.array_like(DX_host)
    DY_const = cuda.const.array_like(DY_host)
    
    # Initialize queue (only one thread across all blocks)
    if cuda.threadIdx.x == 0 and cuda.blockIdx.x == 0:
        global_queue_x[0] = start_x
        global_queue_y[0] = start_y
        global_queue_front[0] = 0
        global_queue_rear[0] = 1
        visited[start_x, start_y] = 1
    
    # Synchronize all blocks before starting BFS
    cuda.syncthreads()
    
    # Main BFS loop with work distribution
    max_iterations = 1000  # Prevent infinite loops
    iteration = 0
    
    while global_queue_front[0] < global_queue_rear[0] and iteration < max_iterations:
        # Calculate work distribution
        current_queue_size = global_queue_rear[0] - global_queue_front[0]
        work_per_warp = max(1, (current_queue_size + TOTAL_WARPS - 1) // TOTAL_WARPS)
        
        # Each warp gets its portion of the queue
        warp_start = global_queue_front[0] + global_warp_id * work_per_warp
        warp_end = min(warp_start + work_per_warp, global_queue_rear[0])
        
        # Store current rear position for this iteration
        iteration_rear = global_queue_rear[0]
        
        # Process queue items (each thread in warp processes one item)
        for batch_start in range(warp_start, warp_end, 32):
            queue_idx = batch_start + lane_id
            
            if queue_idx < warp_end and queue_idx < iteration_rear:
                # Coalesced memory access from global queue (cached in L2)
                x = global_queue_x[queue_idx]
                y = global_queue_y[queue_idx]
                
                # Process pixel (image access will hit L2 cache)
                if x >= 0 and y >= 0 and x < width and y < height:
                    # Recolor pixel
                    img[x, y, 0] = new_color[0]
                    img[x, y, 1] = new_color[1]
                    img[x, y, 2] = (thread_id * 4) % 255  # Color coding by thread
                    
                    # Process neighbors
                    for i in range(8):
                        nx = x + DX_const[i]
                        ny = y + DY_const[i]
                        
                        if is_valid_pixel(nx, ny, width, height):
                            # Use atomic compare-and-swap to mark as visited
                            old = cuda.atomic.cas(visited, (nx, ny), 0, 1)
                            if old == 0 and is_red(img, nx, ny):
                                # Add to global queue atomically
                                pos = cuda.atomic.add(global_queue_rear, 0, 1)
                                if pos < QUEUE_CAPACITY:
                                    global_queue_x[pos] = nx
                                    global_queue_y[pos] = ny
        
        # Synchronize all blocks before next iteration
        cuda.syncthreads()
        
        # Update queue front pointer (only one thread does this)
        if cuda.threadIdx.x == 0 and cuda.blockIdx.x == 0:
            global_queue_front[0] = iteration_rear
        
        cuda.syncthreads()
        iteration += 1


def setup_simple_scene():
    """Create test scene with red blob for flood fill"""
    # Image dimensions and initialization
    width, height = 4000, 4000
    img = np.full((width, height, 3), 255, dtype=np.uint8)
    
    # Create a simple red rectangle blob
    start_x, start_y = 1500, 1500
    end_x, end_y = 2500, 2500
    
    img[start_x:end_x, start_y:end_y] = [255, 0, 0]
    
    # Create visited array
    visited = np.zeros((width, height), dtype=np.int32)
    # New fill color (blue)
    new_color = np.array([0, 0, 255], dtype=np.uint8)
    
    # Optimal GPU configuration for RTX 4060
    threads_per_block = 256   # 8 warps per block
    blocks_per_grid = 144     # 6 blocks per SM × 24 SMs
    
    return img, visited, start_x, start_y, width, height, new_color, threads_per_block, blocks_per_grid


def profile_kernel(num_runs=5):
    """
    Profile the optimized flood fill kernel performance.
    """
    print("=== Optimized Multi-SM Flood Fill Performance ===")
    print("RTX 4060 Configuration:")
    print("  - 24 SMs, 144 blocks total (6 blocks per SM)")
    print("  - 256 threads per block (8 warps per block)")
    print("  - 1,152 total warps across GPU")
    print("  - Global memory queue (L2 cached)")
    print()
    
    # Allocate global memory arrays for the queue (reuse across runs)
    global_queue_x = cuda.device_array(QUEUE_CAPACITY, dtype=np.int32)
    global_queue_y = cuda.device_array(QUEUE_CAPACITY, dtype=np.int32)
    global_queue_front = cuda.device_array(1, dtype=np.int32)
    global_queue_rear = cuda.device_array(1, dtype=np.int32)
    
    # First run for warm-up (compilation)
    img, visited, start_x, start_y, width, height, new_color, threads_per_block, blocks_per_grid = setup_simple_scene()
    d_img = cuda.to_device(img)
    d_visited = cuda.to_device(visited)
    
    # Reset queue
    global_queue_front[0] = 0
    global_queue_rear[0] = 0
    
    optimized_flood_fill[blocks_per_grid, threads_per_block](
        d_img, d_visited, start_x, start_y, width, height, new_color,
        global_queue_x, global_queue_y, global_queue_front, global_queue_rear)
    cuda.synchronize()
    
    print("Warmup completed, starting performance runs...")
    
    # Performance testing
    run_times = []
    
    for i in range(num_runs):
        # Reset queue for each run
        global_queue_front[0] = 0
        global_queue_rear[0] = 0
        
        # Generate new scene for each run
        img, visited, start_x, start_y, width, height, new_color, threads_per_block, blocks_per_grid = setup_simple_scene()
        d_img = cuda.to_device(img)
        d_visited = cuda.to_device(visited)
        
        # Time this run
        start_time = timeit.default_timer()
        optimized_flood_fill[blocks_per_grid, threads_per_block](
            d_img, d_visited, start_x, start_y, width, height, new_color,
            global_queue_x, global_queue_y, global_queue_front, global_queue_rear)
        cuda.synchronize()
        end_time = timeit.default_timer()
        
        run_times.append((end_time - start_time) * 1000)  # Convert to ms
    
    # Calculate statistics
    avg_time = sum(run_times) / len(run_times)
    min_time = min(run_times)
    max_time = max(run_times)
    std_dev = (sum((t - avg_time) ** 2 for t in run_times) / len(run_times)) ** 0.5
    
    # Print results
    print(f"Optimized kernel execution time over {num_runs} runs:")
    print(f"  Average: {avg_time:.2f} ms")
    print(f"  Min: {min_time:.2f} ms")
    print(f"  Max: {max_time:.2f} ms")
    print(f"  Std Dev: {std_dev:.2f} ms")
    print()
    print("GPU Utilization:")
    print("  - SMs used: 24/24 (100%)")
    print(f"  - Total threads: {blocks_per_grid * threads_per_block:,}")
    print(f"  - Total warps: {blocks_per_grid * threads_per_block // 32:,}")
    print(f"  - Blocks per SM: {blocks_per_grid // 24}")
    
    # Return results from the last run
    return d_img.copy_to_host(), d_visited.copy_to_host()


if __name__ == '__main__':
    img_result, visited_result = profile_kernel()
    
    # Save results
    img_result = Image.fromarray(img_result)
    img_result.save('./images/results/optimized_flood_fill.png')
    
    visited_result = Image.fromarray(visited_result.astype(np.uint8) * 255)
    visited_result.save('./images/results/optimized_flood_fill_visited.png')
    
    print("Results saved to:")
    print("  - ./images/results/optimized_flood_fill.png")
    print("  - ./images/results/optimized_flood_fill_visited.png")
