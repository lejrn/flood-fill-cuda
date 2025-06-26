import numpy as np
from numba import cuda, uint8, int32, float32
import matplotlib.pyplot as plt

@cuda.jit(device=True)
def atomicRGB(image, x, y, new_r, new_g, new_b):
    """Atomically update all three color channels"""
    # Calculate linear index for each channel
    size_y = image.shape[0]
    size_x = image.shape[1]
    base_idx = y * size_x + x
    
    # Update each channel using linear indexing
    cuda.atomic.exch(image.reshape(-1), base_idx * 3 + 0, new_r)
    cuda.atomic.exch(image.reshape(-1), base_idx * 3 + 1, new_g)
    cuda.atomic.exch(image.reshape(-1), base_idx * 3 + 2, new_b)

@cuda.jit
def parallel_flood_kernel(image, labels, changes_made, patch_size=32):
    """
    Kernel that processes image in parallel using multiple threads and shared memory
    Each thread block handles a patch_size x patch_size region
    """
    # Shared memory for local patch processing
    shared_patch = cuda.shared.array(shape=(34, 34, 3), dtype=uint8)  # 32x32 + 1 pixel border
    shared_labels = cuda.shared.array(shape=(34, 34), dtype=int32)
    
    # Block and thread info
    bx, by = cuda.blockIdx.x, cuda.blockIdx.y
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    
    # Global coordinates of patch start
    patch_start_x = bx * patch_size
    patch_start_y = by * patch_size
    
    # Load patch into shared memory including borders
    for dy in range(0, patch_size + 2, cuda.blockDim.y):
        y = ty + dy
        if y < patch_size + 2:
            for dx in range(0, patch_size + 2, cuda.blockDim.x):
                x = tx + dx
                if x < patch_size + 2:
                    # Global coordinates
                    gx = patch_start_x + x - 1
                    gy = patch_start_y + y - 1
                    
                    # Boundary check
                    if (gx >= 0 and gx < image.shape[1] and 
                        gy >= 0 and gy < image.shape[0]):
                        # Copy image data
                        for c in range(3):
                            shared_patch[y, x, c] = image[gy, gx, c]
                        shared_labels[y, x] = labels[gy, gx]
    
    cuda.syncthreads()
    
    # Process inner pixels
    if (tx < patch_size and ty < patch_size):
        # Local coordinates (add 1 to account for border)
        lx = tx + 1
        ly = ty + 1
        
        # Global coordinates
        gx = patch_start_x + tx
        gy = patch_start_y + ty
        
        if (gx < image.shape[1] and gy < image.shape[0]):
            # Check if this is an unprocessed red pixel
            if (shared_patch[ly, lx, 0] == 255 and 
                shared_patch[ly, lx, 1] == 0 and 
                shared_patch[ly, lx, 2] == 0 and 
                shared_labels[ly, lx] == 0):
                
                # Check neighbors
                for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ny, nx = ly + dy, lx + dx
                    
                    # If neighbor has a label, propagate it
                    if shared_labels[ny, nx] > 0:
                        label = shared_labels[ny, nx]
                        if cuda.atomic.cas(labels, (gy, gx), 0, label) == 0:
                            # Generate color based on label
                            new_r = (label * 37) % 254
                            new_g = (label * 73) % 254
                            new_b = (label * 19) % 254
                            
                            atomicRGB(image, gx, gy, new_r, new_g, new_b)
                            changes_made[0] = 1
                            break
                    
                # If no labeled neighbors, create new label
                if labels[gy, gx] == 0:
                    new_label = cuda.atomic.add(labels, (0, 0), 1) + 1
                    if cuda.atomic.cas(labels, (gy, gx), 0, new_label) == 0:
                        # Generate color based on new label
                        new_r = (new_label * 37) % 254
                        new_g = (new_label * 73) % 254
                        new_b = (new_label * 19) % 254
                        
                        atomicRGB(image, gx, gy, new_r, new_g, new_b)
                        changes_made[0] = 1

def optimized_flood_fill(image):
    """
    Host function that manages the parallel flood fill process
    Uses an iterative approach with a label propagation technique
    """
    height, width = image.shape[:2]
    
    # Allocate device arrays
    d_image = cuda.to_device(image)
    d_labels = cuda.to_device(np.zeros((height, width), dtype=np.int32))
    d_changes = cuda.to_device(np.array([1], dtype=np.int32))
    
    # Configure kernel
    patch_size = 32  # Size of patch processed by each thread block
    threads_per_block = (16, 16)  # 16x16 threads per block
    blocks_x = (width + patch_size - 1) // patch_size
    blocks_y = (height + patch_size - 1) // patch_size
    blocks_per_grid = (blocks_x, blocks_y)
    
    # Iteratively process until no changes are made
    max_iterations = 100  # Safety limit
    iteration = 0
    
    while iteration < max_iterations:
        # Reset changes flag
        d_changes.copy_to_device(np.array([0], dtype=np.int32))
        
        # Launch kernel
        parallel_flood_kernel[blocks_per_grid, threads_per_block](
            d_image, d_labels, d_changes, patch_size
        )
        
        # Check if any changes were made
        changes = d_changes.copy_to_host()
        if changes[0] == 0:
            break
            
        iteration += 1
    
    # Copy result back to host
    result_image = d_image.copy_to_host()
    return result_image

if __name__ == '__main__':
    # Create test image (1000x1000 white with some red blobs)
    image = np.full((1000, 1000, 3), 255, dtype=np.uint8)
    
    # Draw some test blobs
    def draw_blob(image, center_x, center_y, radius):
        for y in range(center_y - radius, center_y + radius):
            for x in range(center_x - radius, center_x + radius):
                if (x - center_x)**2 + (y - center_y)**2 <= radius**2:
                    if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                        image[y, x] = [255, 0, 0]
    
    # Draw several blobs of different sizes
    blob_centers = [
        (200, 200, 30),
        (500, 500, 40),
        (700, 300, 25),
        (300, 700, 35),
        (800, 800, 45)
    ]
    
    for x, y, r in blob_centers:
        draw_blob(image, x, y, r)
    
    # Save original image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title("Original Image")
    plt.savefig('original_optimized.png')
    plt.close()
    
    # Process image
    result = optimized_flood_fill(image)
    
    # Save result
    plt.figure(figsize=(10, 10))
    plt.imshow(result)
    plt.title("Processed Image")
    plt.savefig('result_optimized.png')
    plt.close() 