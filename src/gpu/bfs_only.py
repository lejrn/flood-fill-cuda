import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
import numba as nb
import random
import timeit
from PIL import Image
# Description
"""
This script will call a GPU kernel that is a flood fill.

The GPU Kernel is named "flood_fill", and it gets a few paremeters, but the most important ones are:
1. img: the image to be processed
2. visited: an array to store the visited pixels
3. start_x: the x coordinate of the pixel to start the flood fill
4. start_y: the y coordinate of the pixel to start the flood
5. width: the width of the image
6. height: the height of the image
7. new_color: the blue color to be recolored 

The flood fill kernel will recolor the red pixels and its neighbors to blue, using a BFS algorithm.
The background is white (255,255,255).
We can assume that each blob size is no bigger than 100x100 pixels.

The kernel will use the Shared Memory to store the queue of neighbors pixels to be traversed.
"""


# # Place direction arrays in constant memory (read-only) # 4 directions
# DX_host = np.array([1, 0, -1, 0], dtype=np.int32)
# DY_host = np.array([0, 1, 0, -1], dtype=np.int32)

# Place direction arrays in constant memory (read-only) # 8 directions
DX_host = np.array([1, 1, 0, -1, -1, -1,  0, 1], dtype=np.int32)
DY_host = np.array([0, 1, 1,  1,  0, -1, -1, -1], dtype=np.int32)


# Kernel Helper functions
@cuda.jit(device=True,inline=True)
def is_red(img, x, y):
    return img[x, y, 0] == 255 and img[x, y, 1] == 0 and img[x, y, 2] == 0

@cuda.jit(device=True,inline=True)
def is_white(img, x, y):
    return img[x, y, 0] == 255 and img[x, y, 1] == 255 and img[x, y, 2] == 255

@cuda.jit(device=True,inline=True)
def is_not_visited(visited, x, y):
    return visited[x, y] == 0

@cuda.jit(device=True,inline=True)
def is_valid_pixel(x, y, width, height):
    return 0 <= x < width and 0 <= y < height

# Kernel functions
@cuda.jit
def flood_fill(img, visited, start_x, start_y, width, height, new_color):
    # this kernel gets a red pixel in the image, then using BFS algorithm, it recolors the red pixel and its neighbors to new_color
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    block_size = cuda.blockDim.x
    
    # Compute global thread ID
    global_tid = bid * block_size + tid

    # shared memory for queue (each element stores an x-y coordinate)
    queue_x = cuda.shared.array(shape=6000, dtype=nb.int32)
    queue_y = cuda.shared.array(shape=6000, dtype=nb.int32)
    queue_front = cuda.shared.array(shape=1, dtype=nb.int32)
    queue_rear = cuda.shared.array(shape=1, dtype=nb.int32)
    
    DX_const = cuda.const.array_like(DX_host)
    DY_const = cuda.const.array_like(DY_host)
    
    # Initialization using x,y indexing for visited
    if global_tid == 0:
        queue_x[0] = start_x
        queue_y[0] = start_y
        queue_front[0] = 0
        queue_rear[0] = 1
        visited[start_x, start_y] = 1
    cuda.syncthreads()
    
    # while the queue is not empty
    while queue_front[0] < queue_rear[0]:
        cuda.syncthreads()
        
        # Check if queue is empty
        if queue_front[0] >= queue_rear[0]:
            # print(queue_rear[0])
            # print(queue_front[0])
            break
        
        current_size = queue_rear[0] - queue_front[0]

        # # Each thread processes some queue items
        # items_per_thread = (current_size + block_size - 1) // block_size
        # start_idx = queue_front[0] + tid * items_per_thread
        # end_idx = start_idx + items_per_thread if start_idx + items_per_thread < queue_rear[0] else queue_rear[0]
        
                # Calculate items per thread
        items_per_thread = max(1, (current_size + block_size - 1) // block_size)
        start_idx = queue_front[0] + tid * items_per_thread
        end_idx = min(start_idx + items_per_thread, queue_rear[0])
        
        for idx in range(start_idx, end_idx):
            # Retrieve the pixel from the queue
            x = queue_x[idx]
            y = queue_y[idx]
            # Mark the pixel with new_color
            img[x, y, 0] = new_color[0]
            img[x, y, 1] = new_color[1]
            # img[x, y, 2] = new_color[2]
            img[x, y, 2] = (tid*8) % 255
            
            # Process 4-connected neighbors (up, down, left, right) using global direction arrays
            for i in range(4):
                # nx = x + DX[i]
                # ny = y + DY[i]
                nx = x + DX_const[i]
                ny = y + DY_const[i]
                
                if is_valid_pixel(nx, ny, width, height):
                    if is_not_visited(visited, nx, ny):
                        old = cuda.atomic.cas(visited, (nx, ny), 0, 1)
                        if is_red(img, nx, ny):
                            # Safely mark as visited
                            if old == 0:
                                # Add to queue
                                pos = cuda.atomic.add(queue_rear, 0, 1)
                                # Check queue capacity
                                if pos < 6000:  # Using the queue capacity
                                    queue_x[pos] = nx
                                    queue_y[pos] = ny
        
        cuda.syncthreads()
        # Update queue front pointer once all threads complete processing
        if global_tid == 0:
            queue_front[0] += current_size
            # print(current_size)
    
        cuda.syncthreads()

    if global_tid == 0:
        print(queue_front[0])


def setup_scene():
    # Image dimensions and initialization
    width, height = 400, 400
    img = np.full((width, height, 3), 255, dtype=np.uint8)
    # Create a red blob (~100x100) in the image
    # blob_center_x = random.randint(100, 300)
    # blob_center_y = random.randint(100, 300)
    # num_walks = 10
    # walk_length = 1000
    # directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    # for _ in range(num_walks):
    #     x, y = blob_center_x, blob_center_y
    #     img[x, y] = [255, 0, 0]
    #     for i in range(walk_length):
    #         dx, dy = random.choice(directions)
    #         new_x = x + dx
    #         new_y = y + dy
    #         if 0 <= new_x < width and 0 <= new_y < height:
    #             if ((new_x - blob_center_x)**2 + (new_y - blob_center_y)**2) < (60**2):
    #                 x, y = new_x, new_y
    #                 img[x, y] = [255, 0, 0]
    
    x = np.random.randint(100, 300)
    y = np.random.randint(100, 300)
    w = np.random.randint(50, 60)
    h = np.random.randint(50, 60)
    img[y:y+h, x:x+w] = [255, 0, 0]
    
    # Find starting red pixel
    start_x, start_y = -1, -1
    for x in range(width):
        for y in range(height):
            if (img[x, y, 0] == 255 and img[x, y, 1] == 0 and img[x, y, 2] == 0):
                start_x, start_y = x, y
                break
        if start_x != -1:
            break
    
    # Create visited array
    visited = np.zeros((width, height), dtype=np.int32)
    # New fill color (blue)
    new_color = np.array([0, 0, 255], dtype=np.uint8)
    # GPU setup
    threads_per_block = 32
    blocks_per_grid = 1
    return img, visited, start_x, start_y, width, height, new_color, threads_per_block, blocks_per_grid


def profile_kernel(num_runs=100, explore_configs=False):
    """
    Profile the flood fill kernel performance.
    
    Args:
        num_runs: Number of times to run the kernel for averaging
    
    Returns:
        The processed image and visited array from the last run
    """
    # First run for warm-up (compilation)
    img, visited, start_x, start_y, width, height, new_color, threads_per_block, blocks_per_grid = setup_scene()
    d_img = cuda.to_device(img)
    d_visited = cuda.to_device(visited)
    flood_fill[blocks_per_grid, threads_per_block](d_img, d_visited, start_x, start_y, width, height, new_color)
    cuda.synchronize()
    
    # For accurate timing, create fresh data for each run
    run_times = []
    
    for i in range(num_runs):
        # Generate new scene for each run
        img, visited, start_x, start_y, width, height, new_color, _, _ = setup_scene()
        d_img = cuda.to_device(img)
        d_visited = cuda.to_device(visited)
        
        # Time this run
        start_time = timeit.default_timer()
        flood_fill[blocks_per_grid, threads_per_block](d_img, d_visited, start_x, start_y, width, height, new_color)
        cuda.synchronize()
        end_time = timeit.default_timer()
        
        run_times.append((end_time - start_time) * 1000)  # Convert to ms
    
    # Calculate statistics
    avg_time = sum(run_times) / len(run_times)
    min_time = min(run_times)
    max_time = max(run_times)
    std_dev = (sum((t - avg_time) ** 2 for t in run_times) / len(run_times)) ** 0.5
    
    # Print results
    print(f"Kernel execution time over {num_runs} runs:")
    print(f"  Average: {avg_time:.2f} ms")
    print(f"  Min: {min_time:.2f} ms")
    print(f"  Max: {max_time:.2f} ms")
    print(f"  Std Dev: {std_dev:.2f} ms")
    
    # Return results from the last run
    return d_img.copy_to_host(), d_visited.copy_to_host()

if __name__ == '__main__':
    img_result, visited_result = profile_kernel()
    img_result = Image.fromarray(img_result)
    img_result.save('./images/results/bfs_only_timeit.png')
    
    # fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # # Transpose image for proper orientation
    # axs[0].imshow(np.transpose(img_result, (1, 0, 2)))
    # axs[0].set_title("Image with Recolored Blob")
    # axs[1].imshow(np.transpose(visited_result, (1, 0)), cmap='gray')
    # axs[1].set_title("Visited Pixels")
    # plt.show()