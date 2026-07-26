# Imports 


# Kernel Helper functions

# is_red

# is_white

# is_visited

# is_not_visited

# is_valid_pixel

# is_on_patch_boundaries


# Kernel functions

# flood_fill kernel

    # this kernel gets a red pixel in the image, then using BFS algorithm, it recolors the red pixel and its neighbors (until it reaches the blob or image boundaries) to blue
    # the kernel will use a queue to store the pixels to be visited
    # the kernel will use a visited array to store the pixels that have been visited
    # the kernel will use a given X threads to process the pixels in the queue, so it's important to avoid race condition

    # variables

    # get the current pixel
    
    # BFS algorithm
    
    # insert the first pixel into the queue
    
    # mark it as visited

    # while the queue is not empty

        # pop the pixel from the queue
        
        # mark the popped pixel as blue
    
        # for loop to process the neighbors of the current pixel    

            # check if the pixel is valid
            
                # check if the pixel is not visited
                
                    # check if the pixel is red
                        
                    # insert the pixel into the queue
                    
                    # mark the pixel as visited



# process image kernel

    # this kernel will process the image and recolor the blobs
    # the kernel will scan the image for red pixels
    # the kernel will use the flood_fill kernel to recolor the blobs
    # the kernel will return the number of blobs found in the image
    
    # define shared memory queue
    
    # define shared memory visited array
    
    # get the image dimensions
    
    # scan the image pixels
    
        # check if the pixel is valid
        
            # check if the pixel is red
            
                # check if the pixel is not visited
                                                            
                    # call the flood_fill kernel with that pixel to recolor the blob
                    
                    # increment the blob count

# Main function

# Imports 
import numpy as np
from numba import cuda
import math

# Kernel Helper functions

@cuda.jit(device=True)
def is_red(pixel):
    return pixel[0] == 255 and pixel[1] == 0 and pixel[2] == 0

@cuda.jit(device=True)
def is_white(pixel):
    return pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255

@cuda.jit(device=True)
def is_visited(visited, x, y, width):
    return visited[y * width + x] == 1

@cuda.jit(device=True)
def is_not_visited(visited, x, y, width):
    return visited[y * width + x] == 0

@cuda.jit(device=True)
def is_valid_pixel(x, y, width, height):
    return 0 <= x < width and 0 <= y < height

@cuda.jit(device=True)
def is_on_patch_boundaries(img, x, y, width, height):
    if not is_valid_pixel(x, y, width, height):
        return False
    
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            nx, ny = x + dx, y + dy
            if dx == 0 and dy == 0:
                continue
            if not is_valid_pixel(nx, ny, width, height) or not is_red(img[ny, nx]):
                return True
    return False

# Kernel functions

@cuda.jit
def flood_fill(img, visited, start_x, start_y, width, height):
    # this kernel gets a red pixel in the image, then using BFS algorithm, it recolors the red pixel and its neighbors to blue
    # variables
    tid = cuda.threadIdx.x
    block_size = cuda.blockDim.x
    
    # shared memory for queue
    queue_x = cuda.shared.array(shape=1024, dtype=int32)
    queue_y = cuda.shared.array(shape=1024, dtype=int32)
    queue_front = cuda.shared.array(shape=1, dtype=int32)
    queue_rear = cuda.shared.array(shape=1, dtype=int32)
    
    # get the current pixel
    if tid == 0:
        queue_x[0] = start_x
        queue_y[0] = start_y
        queue_front[0] = 0
        queue_rear[0] = 1
        
        # mark it as visited
        visited[start_y * width + start_x] = 1
    
    cuda.syncthreads()
    
    # while the queue is not empty
    while queue_front[0] < queue_rear[0]:
        cuda.syncthreads()
        
        # Calculate current queue size
        current_size = queue_rear[0] - queue_front[0]
        
        # Each thread processes some queue items
        items_per_thread = max(1, (current_size + block_size - 1) // block_size)
        start_idx = queue_front[0] + tid * items_per_thread
        end_idx = min(start_idx + items_per_thread, queue_rear[0])
        
        # Process assigned items
        for idx in range(start_idx, end_idx):
            # pop the pixel from the queue
            x = queue_x[idx]
            y = queue_y[idx]
            
            # mark the popped pixel as blue
            img[y, x, 0] = 0
            img[y, x, 1] = 0
            img[y, x, 2] = 255
            
            # for loop to process the neighbors (4-connected: up, down, left, right)
            dx = [0, 0, -1, 1]
            dy = [-1, 1, 0, 0]
            
            for i in range(4):
                nx, ny = x + dx[i], y + dy[i]
                
                # check if the pixel is valid
                if is_valid_pixel(nx, ny, width, height):
                    # check if the pixel is not visited
                    if is_not_visited(visited, nx, ny, width):
                        # check if the pixel is red
                        if is_red(img[ny, nx]):
                            # Atomically mark the pixel as visited to avoid race conditions
                            old = cuda.atomic.compare_and_swap(visited, ny * width + nx, 0, 1)
                            
                            if old == 0:  # Only if we were the one to mark it
                                # insert the pixel into the queue
                                pos = cuda.atomic.add(queue_rear, 0, 1)
                                queue_x[pos] = nx
                                queue_y[pos] = ny
        
        # Update queue front pointer
        cuda.syncthreads()
        if tid == 0:
            queue_front[0] = queue_rear[0]
        
        cuda.syncthreads()

@cuda.jit
def process_image_kernel(img, blob_count):
    # this kernel will process the image and recolor the blobs
    # define shared memory queue
    
    # define shared memory visited array - for large images, we'll use global memory
    height, width = img.shape[0], img.shape[1]
    visited = cuda.device_array((height * width), dtype=np.int8)
    
    # get the image dimensions
    x = cuda.grid(1)  # Flattened grid position
    
    # Initialize visited array
    for i in range(x, width * height, cuda.gridsize(1)):
        visited[i] = 0
    
    cuda.syncthreads()
    
    # scan the image pixels
    for y in range(cuda.blockIdx.y, height, cuda.gridDim.y):
        for x in range(cuda.blockIdx.x, width, cuda.gridDim.x):
            # check if the pixel is valid
            if is_valid_pixel(x, y, width, height):
                # check if the pixel is red
                if is_red(img[y, x]):
                    # check if the pixel is not visited
                    if is_not_visited(visited, x, y, width):
                        # Atomically mark this pixel to prevent other threads from processing it
                        old = cuda.atomic.compare_and_swap(visited, y * width + x, 0, 1)
                        
                        if old == 0:
                            # call the flood_fill kernel with that pixel to recolor the blob
                            flood_fill[1, 256](img, visited, x, y, width, height)
                            
                            # increment the blob count
                            cuda.atomic.add(blob_count, 0, 1)

def main(input_image_path, output_image_path=None):
    """
    Main function to process an image and count/recolor blobs
    
    Args:
        input_image_path: Path to input image
        output_image_path: Path to save output image (optional)
    
    Returns:
        Number of blobs found in the image
    """
    # Load the image
    from PIL import Image
    img = np.array(Image.open(input_image_path))
    
    # Prepare CUDA arrays
    height, width = img.shape[0], img.shape[1]
    d_img = cuda.to_device(img)
    d_blob_count = cuda.device_array(1, dtype=np.int32)
    
    # Set up grid and block dimensions
    threads_per_block = (16, 16)
    blocks_per_grid_x = math.ceil(width / threads_per_block[0])
    blocks_per_grid_y = math.ceil(height / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    # Launch kernel
    process_image_kernel[blocks_per_grid, threads_per_block](d_img, d_blob_count)
    
    # Copy results back
    result_img = d_img.copy_to_host()
    blob_count = d_blob_count.copy_to_host()[0]
    
    # Save the result if path is provided
    if output_image_path:
        result_image = Image.fromarray(result_img)
        result_image.save(output_image_path)
    
    print(f"Found {blob_count} blobs in the image.")
    return blob_count

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else "output.png"
        main(input_path, output_path)
    else:
        print("Usage: python bfs.py input_image.png [output_image.png]")
