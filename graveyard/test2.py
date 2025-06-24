import numpy as np
from numba import cuda, int32, uint8
import matplotlib.pyplot as plt
from PIL import Image
import math

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
PATCH_SIZE = 100             # Each patch is 100x100 pixels
QUEUE_CAPACITY = 5000        # BFS queue capacity

# ------------------------------------------------------------------------------
# Device function: BFS flood fill for Pass 1.
# This function processes a single blob within a patch.
# The BFS queue is stored in shared memory.
# The global visited array (visited) is in global memory (1D int32 array, size=height*width).
# Pixels on the patch border are skipped.
# ------------------------------------------------------------------------------
@cuda.jit(device=True)
def flood_fill_iterative_pass1(image, visited, width, patch_start_x, patch_start_y,
                               start_local_x, start_local_y,
                               new_r, new_g, new_b,
                               queue):
    head = 0
    tail = 0
    # Enqueue the starting pixel (patch-local coordinates)
    queue[tail, 0] = start_local_x
    queue[tail, 1] = start_local_y
    tail = (tail + 1) % QUEUE_CAPACITY

    while head != tail:
        lx = queue[head, 0]
        ly = queue[head, 1]
        head = (head + 1) % QUEUE_CAPACITY

        # Compute absolute coordinates
        x = patch_start_x + lx
        y = patch_start_y + ly

        # Skip if on patch border
        if (x == patch_start_x or x == patch_start_x + PATCH_SIZE - 1 or
            y == patch_start_y or y == patch_start_y + PATCH_SIZE - 1):
            continue

        # Recolor the pixel
        image[y, x, 0] = new_r
        image[y, x, 1] = new_g
        image[y, x, 2] = new_b

        # Enqueue 4-connected neighbors.
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx = lx + dx
            ny = ly + dy
            if 0 <= nx < PATCH_SIZE and 0 <= ny < PATCH_SIZE:
                abs_x = patch_start_x + nx
                abs_y = patch_start_y + ny
                global_idx = abs_y * width + abs_x
                if (visited[global_idx] == 0 and
                    not (image[abs_y, abs_x, 0] == 255 and
                         image[abs_y, abs_x, 1] == 255 and
                         image[abs_y, abs_x, 2] == 255)):
                    next_tail = (tail + 1) % QUEUE_CAPACITY
                    if next_tail == head:
                        break  # Queue full.
                    queue[tail, 0] = nx
                    queue[tail, 1] = ny
                    tail = next_tail
                    # Mark neighbor as visited.
                    visited[global_idx] = 1

# ------------------------------------------------------------------------------
# Kernel: Process Pass 1 – Each block processes one PATCH_SIZE x PATCH_SIZE patch.
# Multiple threads scan the patch. For each pixel that is a candidate (red),
# an atomic compare-and-swap on the global visited array ensures that only one thread
# launches BFS for that pixel.
# Border pixels are stored in a global boundary array.
# ------------------------------------------------------------------------------
@cuda.jit
def process_patch_kernel_pass1(image, visited, patch_size, global_blob_count,
                               boundary_pixels, boundary_count, local_blob_id, width):
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    patch_start_x = bx * patch_size
    patch_start_y = by * patch_size

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bd_x = cuda.blockDim.x
    bd_y = cuda.blockDim.y

    # Allocate shared BFS queue.
    queue = cuda.shared.array(shape=(QUEUE_CAPACITY, 2), dtype=uint8)

    local_blob_count = 0
    # Partition the patch among threads.
    for j in range(ty, patch_size, bd_y):
        for i in range(tx, patch_size, bd_x):
            x = patch_start_x + i
            y = patch_start_y + j
            global_idx = y * width + x
            # If pixel is non-white (blob candidate)
            if not (image[y, x, 0] == 255 and image[y, x, 1] == 255 and image[y, x, 2] == 255):
                # If pixel is on patch border, store its coordinate.
                if (i == 0 or i == patch_size - 1 or j == 0 or j == patch_size - 1):
                    idxb = cuda.atomic.add(boundary_count, 0, 1)
                    if idxb < boundary_pixels.shape[0]:
                        boundary_pixels[idxb, 0] = x
                        boundary_pixels[idxb, 1] = y
                    continue
                # If pixel is red (initial blob color)
                if image[y, x, 0] == 255 and image[y, x, 1] == 0 and image[y, x, 2] == 0:
                    # Atomically claim the pixel using global visited array.
                    # Cast global_idx to int in case it is not already.
                    old = cuda.atomic.compare_and_swap(visited, int(global_idx), 0, 1)
                    if old == 0:
                        # Only one thread wins, so this thread launches BFS.
                        my_id = cuda.atomic.add(local_blob_id, 0, 1)
                        new_r = (my_id * 53) % 254
                        new_g = (my_id * 101) % 254
                        new_b = (my_id * 197) % 254
                        flood_fill_iterative_pass1(image, visited, width,
                                                   patch_start_x, patch_start_y,
                                                   i, j, new_r, new_g, new_b,
                                                   queue)
                        local_blob_count += 1
    # Accumulate blob count per patch.
    if tx == 0 and ty == 0:
        cuda.atomic.add(global_blob_count, 0, local_blob_count)

# # ------------------------------------------------------------------------------
# # Modified BFS that uses the global visited array.
# # ------------------------------------------------------------------------------
# @cuda.jit(device=True)
# def flood_fill_iterative_pass1(image, visited, width, patch_start_x, patch_start_y,
#                                start_local_x, start_local_y, new_r, new_g, new_b,
#                                queue):
#     head = 0
#     tail = 0
#     queue[tail, 0] = start_local_x
#     queue[tail, 1] = start_local_y
#     tail = (tail + 1) % QUEUE_CAPACITY

#     while head != tail:
#         lx = queue[head, 0]
#         ly = queue[head, 1]
#         head = (head + 1) % QUEUE_CAPACITY

#         x = patch_start_x + lx
#         y = patch_start_y + ly

#         if (x == patch_start_x or x == patch_start_x + PATCH_SIZE - 1 or
#             y == patch_start_y or y == patch_start_y + PATCH_SIZE - 1):
#             continue

#         image[y, x, 0] = new_r
#         image[y, x, 1] = new_g
#         image[y, x, 2] = new_b

#         for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
#             nx = lx + dx
#             ny = ly + dy
#             if 0 <= nx < PATCH_SIZE and 0 <= ny < PATCH_SIZE:
#                 abs_x = patch_start_x + nx
#                 abs_y = patch_start_y + ny
#                 global_idx = abs_y * width + abs_x
#                 if (visited[global_idx] == 0 and
#                     not (image[abs_y, abs_x, 0] == 255 and 
#                          image[abs_y, abs_x, 1] == 255 and 
#                          image[abs_y, abs_x, 2] == 255)):
#                     next_tail = (tail + 1) % QUEUE_CAPACITY
#                     if next_tail == head:
#                         break
#                     queue[tail, 0] = nx
#                     queue[tail, 1] = ny
#                     tail = next_tail
#                     visited[global_idx] = 1

# ------------------------------------------------------------------------------
# Host function: Test Pass 1 with multi-thread scanning and global visited.
# ------------------------------------------------------------------------------
def test_pass1(image, patch_size=PATCH_SIZE):
    height, width, _ = image.shape
    global_blob_count = np.array([0], dtype=np.int32)
    MAX_BOUNDARIES = 100000
    boundary_pixels = np.zeros((MAX_BOUNDARIES, 2), dtype=np.uint16)
    boundary_count = np.array([0], dtype=np.int32)
    local_blob_id = np.array([0], dtype=np.int32)

    # Allocate global visited array as a 1D int32 array of size (height*width,)
    visited_host = np.zeros((height * width,), dtype=np.int32)
    d_visited = cuda.to_device(visited_host)

    d_image = cuda.to_device(image)
    d_global_blob_count = cuda.to_device(global_blob_count)
    d_boundary_pixels = cuda.to_device(boundary_pixels)
    d_boundary_count = cuda.to_device(boundary_count)
    d_local_blob_id = cuda.to_device(local_blob_id)

    grid_dim_x = width // patch_size
    grid_dim_y = height // patch_size
    blockdim = (4,4,1)
    griddim = (grid_dim_x, grid_dim_y)

    process_patch_kernel_pass1[griddim, blockdim](d_image, d_visited, patch_size,
                                                  d_global_blob_count, d_boundary_pixels,
                                                  d_boundary_count, d_local_blob_id, width)
    result_image = d_image.copy_to_host()
    blob_count = d_global_blob_count.copy_to_host()[0]
    boundaries = d_boundary_count.copy_to_host()[0]
    print("Pass 1: Blob count =", blob_count, "Boundary count =", boundaries)
    return result_image

# ------------------------------------------------------------------------------
# Example usage.
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    height, width = 9000, 9000
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    np.random.seed(0)
    for _ in range(1000):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        w = np.random.randint(20, 200)
        h = np.random.randint(20, 200)
        image[y:y+h, x:x+w] = [255, 0, 0]

    result_pass1 = test_pass1(image, patch_size=PATCH_SIZE)
    plt.imshow(result_pass1)
    plt.title("Result After Pass 1 (Multi-thread scanning with global visited)")
    plt.axis("off")
    plt.show()
