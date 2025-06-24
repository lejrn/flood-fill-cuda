import os
# os.environ["NUMBA_ENABLE_CUDASIM"] = "1"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np
from numba import cuda, uint8, int32, uint32
import matplotlib.pyplot as plt
from PIL import Image
import random
import os
from datetime import datetime



# Helper device functions for pixel checking
@cuda.jit(device=True, inline=True)
def is_red(image, y, x):
    return image[y, x, 0] == 255 and image[y, x, 1] == 0 and image[y, x, 2] == 0

@cuda.jit(device=True, inline=True)
def is_white(image, y, x):
    return image[y, x, 0] == 255 and image[y, x, 1] == 255 and image[y, x, 2] == 255

@cuda.jit(device=True, inline=True)
def is_not_visited(visited, y, x):
    return visited[y, x] == 0

@cuda.jit(device=True, inline=True)
def is_visited(visited, y, x):
    return visited[y, x] == 1

@cuda.jit(device=True, inline=True)
def is_waiting_handle(visited, y, x, value=2):
    return visited[y, x] == value

@cuda.jit(device=True, inline=True)
def try_mark_visited(visited, y, x):
    """
    Atomically try to mark a pixel as visited.
    Returns True if this thread was the one to mark it (it wasn't visited before).
    Returns False if another thread has already marked it.
    """
    # Use atomic compare-and-swap to safely check and update
    # If visited[y,x] is 0, set it to 1 and return the old value (0)
    # If visited[y,x] is not 0, return its current value (non-zero)
    old_value = cuda.atomic.compare_and_swap(visited, y * visited.shape[1] + x, 0, 1)
    return old_value == 0

@cuda.jit(device=True, inline=True)
def try_mark_for_later(visited, y, x, value):
    """
    Atomically try to mark a pixel with a value for later processing.
    Returns True if this thread was the one to mark it (it wasn't visited before).
    Returns False if another thread has already marked it.
    """
    old_value = cuda.atomic.compare_and_swap(visited, y * visited.shape[1] + x, 0, value)
    return old_value == 0

# @cuda.jit(device=True, inline=True)
# def try_mark_visited(visited, y, x):
#     """
#     Atomically try to mark a pixel as visited.
#     Returns True if this thread was the one to mark it (it wasn't visited before).
#     Returns False if another thread has already marked it.
#     """
#     # In CUDA simulator, atomic operations work differently
#     old_value = visited[y, x]
#     if old_value == 0:
#         visited[y, x] = 1
#         return True
#     return False

# @cuda.jit(device=True, inline=True)
# def try_mark_for_later(visited, y, x, value):
#     """
#     Atomically try to mark a pixel with a value for later processing.
#     Returns True if this thread was the one to mark it (it wasn't visited before).
#     Returns False if another thread has already marked it.
#     """
#     # In CUDA simulator, atomic operations work differently
#     old_value = visited[y, x]
#     if old_value == 0:
#         visited[y, x] = value
#         return True
#     return False

# @cuda.jit(device=True, inline=True)
# def try_mark_visited(visited, y, x):
#     """
#     Atomically try to mark a pixel as visited.
#     Returns True if this thread was the one to mark it (it wasn't visited before).
#     Returns False if another thread has already marked it.
#     """
#     # Use atomic compare-and-swap to safely check and update
#     # For CUDA simulator mode compatibility
#     if cuda.config.ENABLE_CUDASIM:
#         old_value = visited[y, x]
#         if old_value == 0:
#             visited[y, x] = 1
#             return True
#         return False
#     else:
#         # Actual CUDA mode
#         old_value = cuda.atomic.compare_and_swap(visited, y * visited.shape[1] + x, 0, 1)
#         return old_value == 0

# @cuda.jit(device=True, inline=True)
# def try_mark_for_later(visited, y, x, value):
#     """
#     Atomically try to mark a pixel with a value for later processing.
#     Returns True if this thread was the one to mark it (it wasn't visited before).
#     Returns False if another thread has already marked it.
#     """
#     # For CUDA simulator mode compatibility
#     if cuda.config.ENABLE_CUDASIM:
#         old_value = visited[y, x]
#         if old_value == 0:
#             visited[y, x] = value
#             return True
#         return False
#     else:
#         # Actual CUDA mode
#         old_value = cuda.atomic.compare_and_swap(visited, y * visited.shape[1] + x, 0, value)
#         return old_value == 0

# @cuda.jit(device=True, inline=True)
# def try_mark_visited(visited, y, x):
#     """
#     Atomically try to mark a pixel as visited.
#     Returns True if this thread was the one to mark it (it wasn't visited before).
#     Returns False if another thread has already marked it.
#     """
#     # Use atomic compare-and-swap directly on the array element
#     old_value = cuda.atomic.compare_and_swap(visited[y, x], 0, 1)
#     return old_value == 0

# @cuda.jit(device=True, inline=True)
# def try_mark_for_later(visited, y, x, value):
#     """
#     Atomically try to mark a pixel with a value for later processing.
#     Returns True if this thread was the one to mark it (it wasn't visited before).
#     Returns False if another thread has already marked it.
#     """
#     old_value = cuda.atomic.compare_and_swap(visited[y, x], 0, value)
#     return old_value == 0

@cuda.jit(device=True, inline=True)
def is_valid_pixel(height, width, y, x):
    return 0 <= y < height and 0 <= x < width

@cuda.jit(device=True, inline=True)
def is_on_patch_border(patch_start_x, patch_start_y, patch_size, x, y):
    return (x == patch_start_x or x == patch_start_x + patch_size - 1 or
            y == patch_start_y or y == patch_start_y + patch_size - 1)

@cuda.jit(device=True)
def flood_fill_iterative_using_BFS(image, visited, patch_start_x, patch_start_y, 
                                  start_local_x, start_local_y, new_r, new_g, new_b,
                                  queue, height, width, handle_value=2):
    """
    BFS flood fill that recolors connected pixels with a new color.
    Used for blobs fully contained within a patch.
    """
    head = 0
    tail = 0
    iterations = 0
    
    # Get global coordinates
    x = patch_start_x + start_local_x
    y = patch_start_y + start_local_y

    # Skip if out of bounds
    assert is_valid_pixel(height, width, y, x), "BFS start point out of bounds"
    
    # Skip if out of bounds
    if not is_valid_pixel(height, width, y, x):
        return
    
    # Enqueue starting pixel
    queue[tail, 0] = x
    queue[tail, 1] = y
    tail = (tail + 1) % QUEUE_CAPACITY
    visited[y, x] = 1  # Mark as visited
    
    # Apply new color to starting pixel
    image[y, x, 0] = new_r
    image[y, x, 1] = new_g
    image[y, x, 2] = new_b
    
    
    while head != tail:
        
        # Track queue size for debugging
        queue_size = (tail - head) if tail > head else (QUEUE_CAPACITY - head + tail)
        assert queue_size < QUEUE_CAPACITY, f"Queue size approaching capacity"
        assert iterations < 1000000, f"Excessive BFS iterations: queue size"
        
        iterations += 1
        
        cx = queue[head, 0]
        cy = queue[head, 1]
        head = (head + 1) % QUEUE_CAPACITY
        
        # Check 4-connected neighbors
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx = cx + dx
            ny = cy + dy
            
            # Check if pixel is valid and either unvisited (0) or marked for later handling (handle_value)
            if is_valid_pixel(height, width, ny, nx) and (not is_visited(visited, ny, nx) or is_waiting_handle(visited, ny, nx, handle_value)):
                # Check if it's part of the blob (not white)
                if not is_white(image, ny, nx):
                    # Mark as visited
                    visited[ny, nx] = 1
                    
                    # Apply the new color
                    image[ny, nx, 0] = new_r
                    image[ny, nx, 1] = new_g
                    image[ny, nx, 2] = new_b
                    
                    # Add to queue
                    next_tail = (tail + 1) % QUEUE_CAPACITY
                    if next_tail != head:  # Ensure queue is not full
                        queue[tail, 0] = nx
                        queue[tail, 1] = ny
                        tail = next_tail
                    else:
                        # Force assertion failure if queue is full
                        assert False, f"BFS queue overflow: head tail QUEUE_CAPACITY"

@cuda.jit(device=True)
def flood_fill_iterative_using_BFS_to_handle_later(image, visited, patch_start_x, patch_start_y, 
                                                 start_local_x, start_local_y, handle_value,
                                                 queue, height, width):
    """
    BFS flood fill that marks connected pixels with a handle_value in visited array.
    Used for blobs that cross patch boundaries to be handled later.
    """
    head = 0
    tail = 0
    iterations = 0
    
    # Get global coordinates
    x = patch_start_x + start_local_x
    y = patch_start_y + start_local_y

    # Skip if out of bounds
    assert is_valid_pixel(height, width, y, x), "BFS_to_handle_later start point out of bounds"
    
    # Skip if out of bounds
    if not is_valid_pixel(height, width, y, x):
        return
    
    # Enqueue starting pixel
    queue[tail, 0] = x
    queue[tail, 1] = y
    tail = (tail + 1) % QUEUE_CAPACITY
    visited[y, x] = handle_value  # Mark with handle_value for later processing
    
    while head != tail:
        # Track queue size for debugging
        queue_size = (tail - head) if tail > head else (QUEUE_CAPACITY - head + tail)
        assert queue_size < QUEUE_CAPACITY, f"Queue size approaching capacity"
        assert iterations < 1000000, f"Excessive BFS_to_handle_later iterations:"
        
        iterations += 1
        
        cx = queue[head, 0]
        cy = queue[head, 1]
        
        assert is_valid_pixel(height, width, cy, cx), f"Invalid dequeued pixel:"
        
        
        head = (head + 1) % QUEUE_CAPACITY
        
        # Check 4-connected neighbors
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx = cx + dx
            ny = cy + dy
            
            if is_valid_pixel(height, width, ny, nx) and is_not_visited(visited, ny, nx):
                # Check if it's part of the blob (not white)
                if not is_white(image, ny, nx):
                    # Mark with handle_value for later processing
                    visited[ny, nx] = handle_value
                    
                    # Add to queue
                    next_tail = (tail + 1) % QUEUE_CAPACITY
                    if next_tail != head:  # Ensure queue is not full
                        queue[tail, 0] = nx
                        queue[tail, 1] = ny
                        tail = next_tail
                    else:
                        # Force assertion failure if queue is full
                        assert False, f"BFS_to_handle_later queue overflow:"

@cuda.jit
def First_process_all_patches_kernel(image, visited, patch_size, global_blob_count,
                                   boundary_pixels, boundary_count, local_blob_id):
    """
    First pass: Process 100x100 patches to color blobs fully contained within patches
    and mark boundary blobs for later processing.
    """
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    
    block_dim_x = cuda.blockDim.x
    block_dim_y = cuda.blockDim.y
    thread_id = ty * block_dim_x + tx
    total_threads = block_dim_x * block_dim_y
    
    patch_start_x = bx * patch_size
    patch_start_y = by * patch_size
    height, width = image.shape[:2]
    
    # Allocate shared memory for BFS queue - shared across all threads in block
    queue = cuda.shared.array(shape=(QUEUE_CAPACITY, 2), dtype=int32)
    
    # Shared memory for local blob counting
    shared_local_blob_count = cuda.shared.array(shape=(1,), dtype=int32)
    
    # Initialize shared counter
    if thread_id == 0:
        shared_local_blob_count[0] = 0
    cuda.syncthreads()
    
    # Process patch from outer frame to inner frame
    for layer in range(patch_size // 2 + 1):  # +1 to ensure we cover the center
        # Skip processing if we've reached center
        if layer >= patch_size // 2:
            break
        
        # Calculate work distribution parameters for each edge of the layer
        edge_pixels = 4 * (patch_size - 2 * layer) - 4  # Total pixels in this layer's frame
                # Safety check to avoid negative edge_pixels
        if edge_pixels <= 0:
            break
        
        # pixels_per_thread = (edge_pixels + total_threads - 1) // total_threads
        pixels_per_thread = max(1, (edge_pixels + total_threads - 1) // total_threads)
        start_idx = thread_id * pixels_per_thread
        end_idx = min(start_idx + pixels_per_thread, edge_pixels)
        
        # Skip if no work for this thread
        if start_idx >= edge_pixels:
            continue
        
        for idx in range(start_idx, end_idx):
            # Convert linear idx to x,y position in this layer's frame
            side_length = patch_size - 2 * layer
            
            # Top edge (left to right)
            if idx < side_length:
                i = layer + idx
                j = layer
            # Right edge (top to bottom)
            elif idx < side_length * 2 - 1:
                i = patch_size - layer - 1
                j = layer + (idx - (side_length - 1))
            # Bottom edge (right to left)
            elif idx < side_length * 3 - 2:
                i = patch_size - layer - 1 - (idx - (side_length * 2 - 2))
                j = patch_size - layer - 1
            # Left edge (bottom to top)
            else:
                i = layer
                j = patch_size - layer - 1 - (idx - (side_length * 3 - 3))
            
            x = patch_start_x + i
            y = patch_start_y + j
            
            if is_valid_pixel(height, width, y, x):
                # Check if it's a border pixel of the patch
                if layer == 0:
                    if is_red(image, y, x) and try_mark_for_later(visited, y, x, 2):
                        # Add to boundary pixels queue
                        idx = cuda.atomic.add(boundary_count, 0, 1)
                        if idx < boundary_pixels.shape[0]:
                            boundary_pixels[idx, 0] = x
                            boundary_pixels[idx, 1] = y
                            
                        # Mark for later handling
                        flood_fill_iterative_using_BFS_to_handle_later(
                            image, visited, patch_start_x, patch_start_y, 
                            i, j, 2, queue, height, width)
                # Process inner pixel (not on patch border)
                elif is_red(image, y, x) and try_mark_visited(visited, y, x): # and is_not_visited(visited, y, x):
                    # Generate unique color for this blob
                    my_id = cuda.atomic.add(local_blob_id, 0, 1)
                    new_r = (my_id * 53 + 7) % 253 + 1  # Avoid 0 and 255
                    new_g = (my_id * 101 + 11) % 253 + 1
                    new_b = (my_id * 197 + 13) % 253 + 1
                    
                    # Color the blob
                    flood_fill_iterative_using_BFS(
                        image, visited, patch_start_x, patch_start_y,
                        i, j, new_r, new_g, new_b, queue, height, width)
                    cuda.atomic.add(shared_local_blob_count, 0, 1)
                else:
                    # Mark as visited to avoid duplicate processing
                    visited[y, x] = 1
    
    # Wait for all threads to finish processing
    cuda.syncthreads()
    
    # Add local blob count to global counter
    if thread_id == 0:
        cuda.atomic.add(global_blob_count, 0, shared_local_blob_count[0])
 
@cuda.jit
def Second_process_all_patches_kernel(image, visited, patch_size, global_blob_count,
                                    boundary_pixels_input, boundary_count_input,
                                    boundary_pixels_output, boundary_count_output,
                                    local_blob_id):
    """
    Second/third pass: Process larger patches (400x400 or 800x800) to handle
    blobs crossing patch boundaries from previous passes.
    """
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    
    patch_start_x = bx * patch_size
    patch_start_y = by * patch_size
    height, width = image.shape[:2]
    
    # Use one thread per block to handle the patch
    if tx == 0 and ty == 0:
        local_blob_count = 0
        # Allocate shared memory for BFS queue
        queue = cuda.shared.array(shape=(QUEUE_CAPACITY, 2), dtype=int32)
        
        # Step 1: First scan patch boundaries to mark border-crossing blobs with value 3
        # Only checking the outer frame of the patch
        for i in range(patch_size):
            # Top edge
            if patch_start_y < height and patch_start_x + i < width:
                x, y = patch_start_x + i, patch_start_y
                if not is_white(image, y, x) and is_waiting_handle(visited, y, x, 2):
                    flood_fill_iterative_using_BFS_to_handle_later(
                        image, visited, patch_start_x, patch_start_y,
                        i, 0, 3, queue, height, width)
                    
                    # Add to new boundary pixels queue
                    idx = cuda.atomic.add(boundary_count_output, 0, 1)
                    if idx < boundary_pixels_output.shape[0]:
                        boundary_pixels_output[idx, 0] = x
                        boundary_pixels_output[idx, 1] = y
            
            # Bottom edge
            if patch_start_y + patch_size - 1 < height and patch_start_x + i < width:
                x, y = patch_start_x + i, patch_start_y + patch_size - 1
                if not is_white(image, y, x) and is_waiting_handle(visited, y, x, 2):
                    flood_fill_iterative_using_BFS_to_handle_later(
                        image, visited, patch_start_x, patch_start_y,
                        i, patch_size - 1, 3, queue, height, width)
                    
                    # Add to new boundary pixels queue
                    idx = cuda.atomic.add(boundary_count_output, 0, 1)
                    if idx < boundary_pixels_output.shape[0]:
                        boundary_pixels_output[idx, 0] = x
                        boundary_pixels_output[idx, 1] = y
            
            # Left edge (excluding corners already checked)
            if i > 0 and i < patch_size - 1 and patch_start_x < width and patch_start_y + i < height:
                x, y = patch_start_x, patch_start_y + i
                if not is_white(image, y, x) and is_waiting_handle(visited, y, x, 2):
                    flood_fill_iterative_using_BFS_to_handle_later(
                        image, visited, patch_start_x, patch_start_y,
                        0, i, 3, queue, height, width)
                    
                    # Add to new boundary pixels queue
                    idx = cuda.atomic.add(boundary_count_output, 0, 1)
                    if idx < boundary_pixels_output.shape[0]:
                        boundary_pixels_output[idx, 0] = x
                        boundary_pixels_output[idx, 1] = y
            
            # Right edge (excluding corners already checked)
            if i > 0 and i < patch_size - 1 and patch_start_x + patch_size - 1 < width and patch_start_y + i < height:
                x, y = patch_start_x + patch_size - 1, patch_start_y + i
                if not is_white(image, y, x) and is_waiting_handle(visited, y, x, 2):
                    flood_fill_iterative_using_BFS_to_handle_later(
                        image, visited, patch_start_x, patch_start_y,
                        patch_size - 1, i, 3, queue, height, width)
                    
                    # Add to new boundary pixels queue
                    idx = cuda.atomic.add(boundary_count_output, 0, 1)
                    if idx < boundary_pixels_output.shape[0]:
                        boundary_pixels_output[idx, 0] = x
                        boundary_pixels_output[idx, 1] = y
        
        # Step 2: Process boundary pixels from previous pass that are fully contained in this patch
        for idx in range(boundary_count_input[0]):
            x = boundary_pixels_input[idx, 0]
            y = boundary_pixels_input[idx, 1]
            
            # Check if pixel is within this patch and not on the patch boundary
            if (patch_start_x <= x < patch_start_x + patch_size and
                patch_start_y <= y < patch_start_y + patch_size):
                
                # Not on the patch boundary
                if not is_on_patch_border(patch_start_x, patch_start_y, patch_size, x, y):
                    # If marked with value 2 (waiting to be handled)
                    if is_waiting_handle(visited, y, x, 2):
                        # Generate unique color for this blob
                        my_id = cuda.atomic.add(local_blob_id, 0, 1)
                        new_r = (my_id * 53 + 7) % 253 + 1
                        new_g = (my_id * 101 + 11) % 253 + 1
                        new_b = (my_id * 197 + 13) % 253 + 1
                        
                        # Color the blob
                        flood_fill_iterative_using_BFS(
                            image, visited, patch_start_x, patch_start_y,
                            x - patch_start_x, y - patch_start_y, 
                            new_r, new_g, new_b, queue, height, width, 2)
                        local_blob_count += 1
        
        # Add local blob count to global counter
        cuda.atomic.add(global_blob_count, 0, local_blob_count)
        
import os
from datetime import datetime

def visualize_boundaries(boundary_pixels_array, boundary_count, image_width=9000, image_height=9000, output_folder="boundary_images"):
    """
    Create a blank image and visualize boundary pixels marked in dark green.
    
    Args:
        boundary_pixels_array: Array of boundary pixel coordinates [[x1, y1], [x2, y2], ...]
        boundary_count: Number of valid boundary pixels
        image_width: Width of the output image
        image_height: Height of the output image
        output_folder: Folder to save the image
    """
    # Create a blank black image
    vis_image = np.full((image_height, image_width, 3), 255, dtype=np.uint8)
    
    # Dark green color for boundaries
    dark_green = [0, 255, 0]
    
    # Mark each boundary pixel
    for i in range(min(boundary_count, len(boundary_pixels_array))):
        x, y = boundary_pixels_array[i]
        
        # Check if coordinates are within image bounds
        if 0 <= x < image_width and 0 <= y < image_height:
            vis_image[y, x] = dark_green
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"boundaries_{timestamp}.png"
    filepath = os.path.join(output_folder, filename)
    
    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(vis_image)
    plt.title(f"Boundary visualization with {boundary_count} pixels")
    plt.axis('off')
    plt.show()
    
    # Save the image
    plt.imsave(filepath, vis_image)
    print(f"Image saved to {filepath}")
    
    return vis_image

def detect_blobs_cuda(image):
    """
    Main function to detect and color blobs in an image using multi-pass CUDA approach.
    """
    height, width, channels = image.shape
    
    # Initialize arrays
    global_blob_count = np.array([0], dtype=np.int32)
    visited = np.zeros((height, width), dtype=np.uint8)
    local_blob_id = np.array([0], dtype=np.int32)
    
    # Boundary pixel arrays for each pass
    boundary_pixels_pass1 = np.zeros((MAX_BOUNDARIES, 2), dtype=np.int32)
    boundary_count_pass1 = np.array([0], dtype=np.int32)
    
    boundary_pixels_pass2 = np.zeros((MAX_BOUNDARIES, 2), dtype=np.int32)
    boundary_count_pass2 = np.array([0], dtype=np.int32)
    
    # Copy data to device
    d_image = cuda.to_device(image)
    d_visited = cuda.to_device(visited)
    d_global_blob_count = cuda.to_device(global_blob_count)
    d_local_blob_id = cuda.to_device(local_blob_id)
    
    d_boundary_pixels_pass1 = cuda.to_device(boundary_pixels_pass1)
    d_boundary_count_pass1 = cuda.to_device(boundary_count_pass1)
    
    d_boundary_pixels_pass2 = cuda.to_device(boundary_pixels_pass2)
    d_boundary_count_pass2 = cuda.to_device(boundary_count_pass2)
    
    # Define thread and block dimensions
    threads_per_block = (16, 16)
    
    # First pass with 100x100 patches
    blocks_pass1 = ((width + INITIAL_PATCH_SIZE - 1) // INITIAL_PATCH_SIZE, 
                   (height + INITIAL_PATCH_SIZE - 1) // INITIAL_PATCH_SIZE)
    
    First_process_all_patches_kernel[blocks_pass1, threads_per_block](d_image, 
                                                                      d_visited, 
                                                                      INITIAL_PATCH_SIZE, 
                                                                      d_global_blob_count, 
                                                                      d_boundary_pixels_pass1, d_boundary_count_pass1, 
                                                                      d_local_blob_id
    )
    
    # Get results after first pass
    boundary_count_after_pass1 = d_boundary_count_pass1.copy_to_host()[0]
    blob_count_after_pass1 = d_global_blob_count.copy_to_host()[0]
    
    print(f"Pass 1: Detected {blob_count_after_pass1} blobs, {boundary_count_after_pass1} boundary pixels")
    
    # visualize_boundaries(d_boundary_pixels_pass1.copy_to_host(), boundary_count_after_pass1)
    
    final_blob_count = blob_count_after_pass1
    result_image = d_image.copy_to_host()
    # Second pass with 400x400 patches
    # blocks_pass2 = ((width + SECOND_PATCH_SIZE - 1) // SECOND_PATCH_SIZE, 
    #                (height + SECOND_PATCH_SIZE - 1) // SECOND_PATCH_SIZE)
    
    
    # threads_per_block = (1,1)
    # Second_process_all_patches_kernel[blocks_pass2, threads_per_block](
    #     d_image, d_visited, SECOND_PATCH_SIZE, d_global_blob_count,
    #     d_boundary_pixels_pass1, d_boundary_count_pass1,
    #     d_boundary_pixels_pass2, d_boundary_count_pass2, d_local_blob_id
    # )
    
    # # Get results after second pass
    # boundary_count_after_pass2 = d_boundary_count_pass2.copy_to_host()[0]
    # blob_count_after_pass2 = d_global_blob_count.copy_to_host()[0]
    
    # print(f"Pass 2: Detected {blob_count_after_pass2 - blob_count_after_pass1} additional blobs, "
    #       f"{boundary_count_after_pass2} boundary pixels")
    
    # final_blob_count = blob_count_after_pass2
    # result_image = d_image.copy_to_host()
    
    # visualize_boundaries(d_boundary_pixels_pass2.copy_to_host(), boundary_count_after_pass2)
    
    # # Check if we need a third pass
    # if boundary_count_after_pass2 > 0:
    #     blocks_pass3 = ((width + THIRD_PATCH_SIZE - 1) // THIRD_PATCH_SIZE, 
    #                    (height + THIRD_PATCH_SIZE - 1) // THIRD_PATCH_SIZE)
        
    #     Second_process_all_patches_kernel[blocks_pass3, threads_per_block](
    #         d_image, d_visited, THIRD_PATCH_SIZE, d_global_blob_count,
    #         d_boundary_pixels_pass2, d_boundary_count_pass2,
    #         d_boundary_pixels_pass1, d_boundary_count_pass1, d_local_blob_id
    #     )
        
    #     # Get final results
    #     final_blob_count = d_global_blob_count.copy_to_host()[0]
    #     print(f"Pass 3: Detected {final_blob_count - blob_count_after_pass2} additional blobs")
    # else:
    #     final_blob_count = blob_count_after_pass2
    
    # # Copy processed image back to host
    # result_image = d_image.copy_to_host()
    
    # print(f"Total blobs detected: {final_blob_count}")
    
    return result_image, final_blob_count

# Constants
INITIAL_PATCH_SIZE = 100
SECOND_PATCH_SIZE = 400
THIRD_PATCH_SIZE = 800
QUEUE_CAPACITY = 4000  # Smaller than patch size to fit in shared memory
MAX_BOUNDARIES = 1000000  # Maximum number of boundary pixels to track
IMAGE_SIZE = 900

# Example usage
if __name__ == "__main__":
    # Create a synthetic 9000x9000 white image
    image = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 255, dtype=np.uint8)
    
    # Draw some random red blobs
    np.random.seed(0)
    for _ in range(1000):
        x = np.random.randint(0, IMAGE_SIZE)
        y = np.random.randint(0, IMAGE_SIZE)
        w = np.random.randint(20, 50)
        h = np.random.randint(20, 50)
        image[y:y+h, x:x+w] = [255, 0, 0]
    
    # Process the image
    result_image, blob_count = detect_blobs_cuda(image)
    
    # Visualize a portion of the result
    plt.figure(figsize=(12, 10))
    plt.imshow(result_image)
    plt.title(f"Detected {blob_count} Blobs")
    plt.axis('off')
    plt.show()