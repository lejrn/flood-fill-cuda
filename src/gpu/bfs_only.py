import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
import numba as nb
import random

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

# Define direction arrays as regular numpy arrays
DX = np.array([0, 0, -1, 1], dtype=np.int32)
DY = np.array([-1, 1, 0, 0], dtype=np.int32)

# Kernel Helper functions
@cuda.jit(device=True)
def is_red(img, x, y, width, height):
    # Check if pixel (x,y) is red: (255,0,0)
    if x < 0 or x >= width or y < 0 or y >= height:
        return False
    return img[x, y, 0] == 255 and img[x, y, 1] == 0 and img[x, y, 2] == 0

@cuda.jit(device=True)
def is_white(img, x, y, width, height):
    if x < 0 or x >= width or y < 0 or y >= height:
        return False
    return img[x, y, 0] == 255 and img[x, y, 1] == 255 and img[x, y, 2] == 255

@cuda.jit(device=True)
def is_not_visited(visited, x, y, width):
    # Use x,y order for visited array
    return visited[x, y] == 0

@cuda.jit(device=True)
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
    queue_x = cuda.shared.array(shape=4096, dtype=nb.int32)
    queue_y = cuda.shared.array(shape=4096, dtype=nb.int32)
    queue_front = cuda.shared.array(shape=1, dtype=nb.int32)
    queue_rear = cuda.shared.array(shape=1, dtype=nb.int32)
    
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
            img[x, y, 2] = (tid*16) % 255
            
            # Process 4-connected neighbors (up, down, left, right) using global direction arrays
            for i in range(4):
                nx = x + DX[i]
                ny = y + DY[i]
                if is_valid_pixel(nx, ny, width, height):
                    if is_not_visited(visited, nx, ny, width):
                        old = cuda.atomic.cas(visited, (nx, ny), 0, 1)
                        if is_red(img, nx, ny, width, height):
                            # Safely mark as visited
                            if old == 0:
                                # Add to queue
                                pos = cuda.atomic.add(queue_rear, 0, 1)
                                # Check queue capacity
                                if pos < 4096:  # Using the queue capacity
                                    queue_x[pos] = nx
                                    queue_y[pos] = ny
        
        cuda.syncthreads()
        # Update queue front pointer once all threads complete processing
        if global_tid == 0:
            queue_front[0] += current_size
    
        cuda.syncthreads()

    # print(queue_front[0])
    # print(queue_rear[0])

# Main function
def main():
    # Image dimensions and initialization
    width, height = 400, 400
    # Create image with x,y indexing (width, height)
    img = np.full((width, height, 3), 255, dtype=np.uint8)
    
    # Create a red blob (~100x100) in the image
    blob_center_x = random.randint(100, 300)
    blob_center_y = random.randint(100, 300)
    
    # Create an irregular, continuous red blob using multiple random walks
    num_walks = 10
    walk_length = 1000
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    for _ in range(num_walks):
        x, y = blob_center_x, blob_center_y
        img[x, y] = [255, 0, 0]
        for i in range(walk_length):
            dx, dy = random.choice(directions)
            new_x = x + dx
            new_y = y + dy
            # Ensure the new pixel stays within image bounds and near the blob center
            if 0 <= new_x < width and 0 <= new_y < height:
                # Constrain the blob's spread to a rough radius (e.g., 60 pixels) around the center
                if ((new_x - blob_center_x)**2 + (new_y - blob_center_y)**2) < (60**2):
                    x, y = new_x, new_y
                    img[x, y] = [255, 0, 0]

    # Find a starting red pixel
    start_x, start_y = -1, -1
    for x in range(width):
        for y in range(height):
            if (img[x, y, 0] == 255 and img[x, y, 1] == 0 and img[x, y, 2] == 0):
                start_x, start_y = x, y
                break
        if start_x != -1:
            break

    # print(f"{start_x=},{start_y=}")

    # Create visited as a 2D array with x,y indexing
    visited = np.zeros((width, height), dtype=np.int32)
    
    # Copy arrays to device
    d_img = cuda.to_device(img)
    d_visited = cuda.to_device(visited)
    
    # New color to fill with (blue)
    new_color = np.array([0, 0, 255], dtype=np.uint8)
    
    # Launch kernel with multiple blocks to better utilize the GPU
    threads_per_block = 256
    blocks_per_grid = 20  # Using all 20 multiprocessors on your RTX 3050
    flood_fill[blocks_per_grid, threads_per_block](d_img, d_visited, start_x, start_y, width, height, new_color)
    
    # Copy result back to host
    img_result = d_img.copy_to_host()
    visited_result = d_visited.copy_to_host()
    
    # Display results
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # Need to transpose for imshow since we've changed to x,y indexing
    axs[0].imshow(np.transpose(img_result, (1, 0, 2)))
    axs[0].set_title("Image with Recolored Blob")
    axs[0].scatter([start_x], [start_y], color='green', s=50, marker='o')  # Flipped for scatter plot
    axs[0].text(start_x+1, start_y+1, "Start", color='green')
    axs[1].imshow(np.transpose(visited_result, (1, 0)), cmap='gray')
    axs[1].set_title("Visited Pixels")
    axs[1].scatter([start_x], [start_y], color='red', s=50, marker='x')  # Flipped for scatter plot
    plt.show()

if __name__ == '__main__':
    main()