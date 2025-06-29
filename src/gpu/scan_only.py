from numba import cuda
import numba as nb
import numpy as np
import random
import timeit
from PIL import Image
import matplotlib.pyplot as plt

# Kernel Helper functions
@cuda.jit(device=True, inline=True)
def is_red(img, x, y):
    """Check if a pixel is red"""
    return img[x, y, 0] == 255 and img[x, y, 1] == 0 and img[x, y, 2] == 0

@cuda.jit(device=True, inline=True)
def is_white(img, x, y):
    """Check if a pixel is white"""
    return img[x, y, 0] == 255 and img[x, y, 1] == 255 and img[x, y, 2] == 255

@cuda.jit(device=True, inline=True)
def is_not_visited(visited, x, y):
    """Check if a pixel has not been visited"""
    return visited[x, y, 0] == 0 and visited[x, y, 1] == 0 and visited[x, y, 2] == 0

@cuda.jit(device=True, inline=True)
def is_valid_pixel(x, y, width, height):
    """Check if a pixel is within the image boundaries"""
    return 0 <= x < width and 0 <= y < height

@cuda.jit
def scan_image_small_example(img, visited, width, height, found_flag):
    """
    Kernel to scan a smaller 100x100 image with 10x10 patches.
    Each thread processes exactly one 10x10 patch with clear visualization
    of parallel execution using dark-to-light gradients.
    
    Parameters:
    - img: Input image (width x height x 3)
    - visited: RGB visited array (width x height x 3)
    - width, height: Image dimensions
    - found_flag: Flag to indicate if a red pixel was found
    """
    # Get thread ID
    tid = cuda.threadIdx.x
    
    # Define patch size
    patch_size = 10  # Each patch is 10x10
    patches_per_row = width // patch_size
    
    # Calculate which patch this thread should process
    patch_idx = tid
    patch_x = (patch_idx % patches_per_row) * patch_size
    patch_y = (patch_idx // patches_per_row) * patch_size
    
    # Shared memory for stop flag and global clock
    stop_scanning = cuda.shared.array(shape=1, dtype=nb.int32)
    clock = cuda.shared.array(shape=1, dtype=nb.int32)
    
    # Initialize shared memory
    if tid == 0:
        stop_scanning[0] = 0
        clock[0] = 0
    
    cuda.syncthreads()
    
    # Generate a unique hue for this thread using golden ratio method
    # This ensures maximum visual distinction between threads
    golden_ratio = 0.618033988749895
    h = (tid * golden_ratio) % 1.0
    
    # Convert HSV to RGB for base color (simplified HSV->RGB conversion)
    # Using saturation=1.0 to get vibrant colors
    h_i = int(h * 6)
    f = h * 6 - h_i
    
    # Define thread's base color
    if h_i == 0:
        base_r, base_g, base_b = 255, int(255 * f), 0
    elif h_i == 1:
        base_r, base_g, base_b = int(255 * (1 - f)), 255, 0
    elif h_i == 2:
        base_r, base_g, base_b = 0, 255, int(255 * f)
    elif h_i == 3:
        base_r, base_g, base_b = 0, int(255 * (1 - f)), 255
    elif h_i == 4:
        base_r, base_g, base_b = int(255 * f), 0, 255
    else:
        base_r, base_g, base_b = 255, 0, int(255 * (1 - f))
    
    # Process assigned patch one pixel at a time
    for sum_idx in range(2 * patch_size - 1):  # Process diagonals
        for i in range(patch_size):
            j = sum_idx - i
            
            # Check if this diagonal element is within bounds
            if 0 <= j < patch_size:
                # Convert to actual image coordinates
                x = patch_x + i
                y = patch_y + j
                
                # Force a tiny delay to make timing more visible
                # This simulates real-world processing time differences
                for _ in range(5):  # Small delay - adjust as needed
                    pass  # Just burn cycles
                
                # Get the current global clock value and increment it
                # This is the key to visualizing the true timeline
                tick = cuda.atomic.add(clock, 0, 1)
                
                # Check if we should stop scanning
                if stop_scanning[0] == 1:
                    break
                
                # Check if this pixel is not visited yet
                if is_not_visited(visited, x, y):
                    # Calculate brightness based on global timeline position
                    # This creates a dark-to-light effect as execution progresses
                    # Normalize to max expected ticks (100 threads * 100 pixels)
                    max_ticks = 10000  # 100 threads * 100 pixels
                    normalized_tick = min(1.0, tick / max_ticks)
                    
                    # Apply brightness to thread's base color
                    # Start very dark (20%) and progress to full brightness (100%)
                    brightness = 0.2 + 0.8 * normalized_tick
                    
                    # Calculate pixel color based on thread's hue and global timestamp
                    r_val = min(255, int(base_r * brightness))
                    g_val = min(255, int(base_g * brightness))
                    b_val = min(255, int(base_b * brightness))
                    
                    # Check if the pixel is red
                    if is_red(img, x, y):
                        # For red pixels - use a bright magenta with brightness variation
                        visited[x, y, 0] = min(255, int(255 * brightness))
                        visited[x, y, 1] = 0
                        visited[x, y, 2] = min(255, int(255 * brightness))
                        
                        # Set found flag and notify other threads to stop
                        old = cuda.atomic.cas(found_flag, 0, 0, 1)
                        if old == 0:  # If we're the first to find a red pixel
                            found_flag[1] = x
                            found_flag[2] = y
                            stop_scanning[0] = 1  # Signal all threads to stop
                    else:
                        # For non-red pixels - use thread's hue with global timestamp brightness
                        visited[x, y, 0] = r_val
                        visited[x, y, 1] = g_val
                        visited[x, y, 2] = b_val
        
        # Check again if we should stop scanning
        if stop_scanning[0] == 1:
            break
    
    cuda.syncthreads()

@cuda.jit
def simple_scan_kernel(img, visited, width, height, found_flag):
    """
    Simple kernel that scans an image for red pixels without early termination.
    Each thread processes one 10x10 patch of the image.
    Visualization shows thread execution pattern with unique colors per thread
    and brightness indicating chronological progress.
    
    Parameters:
    - img: Input image (width x height x 3)
    - visited: RGB visited array (width x height x 3)
    - width, height: Image dimensions
    - found_flag: Flag to indicate if a red pixel was found
    """
    # Get thread ID within the block
    tid = cuda.threadIdx.x
    
    # Define patch size
    patch_size = 10  # Each patch is 10x10
    patches_per_row = width // patch_size
    
    # Calculate which patch this thread should process
    patch_idx = tid
    patch_x = (patch_idx % patches_per_row) * patch_size
    patch_y = (patch_idx // patches_per_row) * patch_size
    
    # Shared memory for global clock (visualization only)
    clock = cuda.shared.array(shape=1, dtype=nb.int32)
    
    # Initialize shared memory
    if tid == 0:
        clock[0] = 0
    
    cuda.syncthreads()
    
    # Generate a unique hue for this thread using golden ratio
    golden_ratio = 0.618033988749895
    h = (tid * golden_ratio) % 1.0
    
    # Convert HSV to RGB (simplified conversion)
    h_i = int(h * 6)
    f = h * 6 - h_i
    
    # Define thread's base color
    if h_i == 0:
        base_r, base_g, base_b = 255, int(255 * f), 0
    elif h_i == 1:
        base_r, base_g, base_b = int(255 * (1 - f)), 255, 0
    elif h_i == 2:
        base_r, base_g, base_b = 0, 255, int(255 * f)
    elif h_i == 3:
        base_r, base_g, base_b = 0, int(255 * (1 - f)), 255
    elif h_i == 4:
        base_r, base_g, base_b = int(255 * f), 0, 255
    else:
        base_r, base_g, base_b = 255, 0, int(255 * (1 - f))
    
    # Process assigned patch row by row
    for j in range(patch_size):
        for i in range(patch_size):
            # Calculate image coordinates
            x = patch_x + i
            y = patch_y + j
            
            # Get current global timestamp for visualization
            tick = cuda.atomic.add(clock, 0, 1)
            
            # Check if this pixel is not visited yet
            if is_not_visited(visited, x, y):
                # Calculate brightness based on global timeline position
                max_ticks = 10000  # 100 threads * 100 pixels
                normalized_tick = min(1.0, tick / max_ticks)
                
                # Apply brightness to thread's base color (dark to light)
                brightness = 0.2 + 0.8 * normalized_tick
                
                # Calculate pixel color
                r_val = min(255, int(base_r * brightness))
                g_val = min(255, int(base_g * brightness))
                b_val = min(255, int(base_b * brightness))
                
                # Check if the pixel is red
                if is_red(img, x, y):
                    # Mark red pixels distinctively
                    visited[x, y, 0] = min(255, int(255 * brightness))
                    visited[x, y, 1] = 0
                    visited[x, y, 2] = min(255, int(255 * brightness))
                    
                    # Set found flag (no early termination)
                    cuda.atomic.max(found_flag, 0, 1)
                    found_flag[1] = x
                    found_flag[2] = y
                else:
                    # Regular non-red pixel
                    visited[x, y, 0] = r_val
                    visited[x, y, 1] = g_val
                    visited[x, y, 2] = b_val
    
    # No final synchronization needed since we don't update shared memory after this

def setup_scene_small_example():
    """Set up a smaller 100x100 test scene with a red square."""
    # Image dimensions
    width, height = 100, 100
    img = np.full((width, height, 3), 255, dtype=np.uint8)
    
    # Create a 20x20 red square
    square_x = random.randint(30, 60)
    square_y = random.randint(30, 60)
    square_size = 20
    
    img[square_x:square_x+square_size, square_y:square_y+square_size] = [255, 0, 0]
    
    # Create RGB visited array (initialized to all zeros)
    visited = np.zeros((width, height, 3), dtype=np.uint8)
    
    # Create found flag array [found, x, y]
    found_flag = np.zeros(3, dtype=np.int32)
    
    return img, visited, width, height, found_flag

def process_image_small_example(img, visited, width, height, found_flag):
    """Process the small example image."""
    # Copy data to device
    d_img = cuda.to_device(img)
    d_visited = cuda.to_device(visited)
    d_found_flag = cuda.to_device(found_flag)
    
    # Configure thread block - exactly 100 threads for 100 patches (10x10 each)
    threads_per_block = 100
    
    # Launch kernel with a single block
    print(f"Launching with single block of {threads_per_block} threads")
    scan_image_small_example[1, threads_per_block](d_img, d_visited, width, height, d_found_flag)
    cuda.synchronize()
    
    # Get results
    result_visited = d_visited.copy_to_host()
    result_found = d_found_flag.copy_to_host()
    
    return result_visited, result_found

def main_small_example():
    """Main function for the small example."""
    # Setup the scene
    img, visited, width, height, found_flag = setup_scene_small_example()
    
    # Save original image
    orig_img = Image.fromarray(img)
    orig_img.save('./images/scan_small_original.png')
    
    # Process on GPU
    print("Processing smaller example image...")
    start_time = timeit.default_timer()
    processed_visited, found_info = process_image_small_example(img, visited, width, height, found_flag)
    end_time = timeit.default_timer()
    
    elapsed_time = (end_time - start_time) * 1000  # Convert to ms
    print(f"Processing time: {elapsed_time:.2f} ms")
    
    if found_info[0] == 1:
        print(f"Found red pixel at ({found_info[1]}, {found_info[2]})")
    else:
        print("No red pixels found")
    
    # Save visited visualization
    visited_img = Image.fromarray(processed_visited)
    visited_img.save('./images/results/scan_small_visited.png')
    
    # Display results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(img)
    ax1.set_title("Original Image")
    ax2.imshow(processed_visited)
    ax2.set_title("Visited Pixels")
    if found_info[0] == 1:
        ax2.plot(found_info[2], found_info[1], 'ro', markersize=10, markeredgecolor='yellow')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main_small_example()