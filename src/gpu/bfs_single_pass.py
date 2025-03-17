import numpy as np
from numba import cuda, uint8, int32
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg')  # Or 'Qt5Agg' if you have Qt installed
import matplotlib.pyplot as plt

# Device function: Iterative BFS flood fill using a shared memory queue.
@cuda.jit(device=True)
def flood_fill_iterative(image, patch_start_x, patch_start_y, patch_width, patch_height,
                         start_local_x, start_local_y,
                         orig_r, orig_g, orig_b,
                         new_r, new_g, new_b,
                         queue, queue_count):
    # Initialize the BFS queue.
    head = 0
    queue[0, 0] = start_local_x  # local x coordinate within patch
    queue[0, 1] = start_local_y  # local y coordinate within patch
    queue_count[0] = 1  # one element in the queue

    # Process the queue.
    while head < queue_count[0]:
        lx = queue[head, 0]
        ly = queue[head, 1]
        head += 1

        # Convert local coordinates to absolute image coordinates.
        x = patch_start_x + lx
        y = patch_start_y + ly

        # Check if the pixel still has the original color.
        if image[y, x, 0] == orig_r and image[y, x, 1] == orig_g and image[y, x, 2] == orig_b:
            # Fill with the new color.
            image[y, x, 0] = new_r
            image[y, x, 1] = new_g
            image[y, x, 2] = new_b

            # Explore neighbors in 4 directions.
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx = lx + dx
                ny = ly + dy
                # Check bounds within the patch (local coordinates).
                if nx >= 0 and nx < patch_width and ny >= 0 and ny < patch_height:
                    abs_x = patch_start_x + nx
                    abs_y = patch_start_y + ny
                    # Only add neighbor if it has the original color.
                    if image[abs_y, abs_x, 0] == orig_r and image[abs_y, abs_x, 1] == orig_g and image[abs_y, abs_x, 2] == orig_b:
                        idx = queue_count[0]
                        # Ensure we do not exceed the fixed capacity.
                        if idx < 10000:
                            queue[idx, 0] = nx
                            queue[idx, 1] = ny
                            queue_count[0] = idx + 1

# Kernel: Each block processes a 100x100 patch using a single thread.
@cuda.jit
def process_patch_kernel(image, patch_width, patch_height, global_blob_count):
    # Determine which patch this block is responsible for.
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    patch_start_x = bx * patch_width
    patch_start_y = by * patch_height

    # Only one thread per block does the work.
    if cuda.threadIdx.x == 0 and cuda.threadIdx.y == 0 and cuda.threadIdx.z == 0:
        local_blob_count = 0

        # Allocate shared memory for the BFS queue.
        # The queue has 10,000 entries (for a 100x100 patch), each with 2 uint8 values.
        queue = cuda.shared.array(shape=(10000, 2), dtype=uint8)
        # Allocate a shared counter for the queue.
        queue_count = cuda.shared.array(shape=1, dtype=int32)
        queue_count[0] = 0

        # Scan through every pixel in the patch (using nested loops over local coordinates).
        for local_y in range(patch_height):
            for local_x in range(patch_width):
                x = patch_start_x + local_x  # Absolute x coordinate.
                y = patch_start_y + local_y  # Absolute y coordinate.
                # Detect a blob: here we assume the blob is red (255, 0, 0).
                if image[y, x, 0] == 255 and image[y, x, 1] == 0 and image[y, x, 2] == 0:
                    # Compute a new color based on block indices (for demonstration).
                    new_r = (bx * 37) % 254
                    new_g = (by * 73) % 254
                    new_b = ((bx + by) * 19) % 254
                    # The original color is red: (255, 0, 0).
                    flood_fill_iterative(image, patch_start_x, patch_start_y, patch_width, patch_height,
                                         local_x, local_y,
                                         255, 0, 0,
                                         new_r, new_g, new_b,
                                         queue, queue_count)
                    local_blob_count += 1

        # Atomically add the number of blobs processed in this patch to the global counter.
        cuda.atomic.add(global_blob_count, 0, local_blob_count)

# Host function: sets up the image, launches the kernel, and retrieves the results.
def gpu_process_patches(image, patch_width=100, patch_height=100):
    # Image is expected to be a numpy array of shape (height, width, 3) with dtype uint8.
    height, width, channels = image.shape
    assert channels == 3, "Image must have 3 color channels"

    # Global blob count (initialize to 0).
    global_blob_count = np.array([0], dtype=np.int32)

    # Copy the image and blob counter to device memory.
    d_image = cuda.to_device(image)
    d_blob_count = cuda.to_device(global_blob_count)

    # Calculate grid dimensions based on patch size.
    grid_dim_x = width // patch_width
    grid_dim_y = height // patch_height

    # Launch the kernel with one thread per block.
    blockdim = (1, 1, 1)
    griddim = (grid_dim_x, grid_dim_y)

    process_patch_kernel[griddim, blockdim](d_image, patch_width, patch_height, d_blob_count)

    # Retrieve the results.
    result_image = d_image.copy_to_host()
    result_blob_count = d_blob_count.copy_to_host()
    return result_image, result_blob_count[0]

# ---------------------
# Example usage:
if __name__ == '__main__':
    # Create a 1000x1000 white image (all pixels white: 255,255,255).
    image = np.full((1000, 1000, 3), 255, dtype=np.uint8)

    # Draw a red blob in patch at grid position (2, 3).
    patch_w, patch_h = 100, 100
    block_x, block_y = 2, 3
    patch_start_x = block_x * patch_w
    patch_start_y = block_y * patch_h
    # Draw a red square in the center of that patch.
    for y in range(patch_start_y + 30, patch_start_y + 70):
        for x in range(patch_start_x + 30, patch_start_x + 70):
            image[y, x] = np.array([255, 0, 0], dtype=np.uint8)

    # Draw another red blob in patch (5, 6).
    block_x2, block_y2 = 5, 6
    patch_start_x2 = block_x2 * patch_w
    patch_start_y2 = block_y2 * patch_h
    for y in range(patch_start_y2 + 20, patch_start_y2 + 50):
        for x in range(patch_start_x2 + 20, patch_start_x2 + 50):
            image[y, x] = np.array([255, 0, 0], dtype=np.uint8)

     # Display the result image.
    plt.imshow(image)
    plt.title("Before Processing")
    plt.savefig('before_processing.png')
    plt.close()  # Close the figure to free memory

    # Process the image patches with flood fill.
    result_image, blob_count = gpu_process_patches(image, patch_width=100, patch_height=100)
    print("Detected and processed blobs:", blob_count)

    # Display the result image.
    plt.imshow(result_image)
    plt.title("Result Image with Flood Filled Blobs")
    plt.savefig('after_processing.png')
    plt.close()
