import numpy as np
from PIL import Image
import random
from collections import deque
from pathlib import Path
import timeit
import time

# Constants
RED_THRESHOLD = 200  # Threshold for red detection
WHITE_BG = (255, 255, 255)

def is_red(pixel):
    r, g, b = pixel
    return r > RED_THRESHOLD and g < 50 and b < 50

def bfs_flood_fill(image, visited, start_x, start_y, new_color):
    height, width = image.shape[:2]
    queue = deque([(start_x, start_y)])
    visited[start_x, start_y] = True
    
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    while queue:
        x, y = queue.popleft()
        image[x, y] = new_color
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < height and 0 <= ny < width and 
                not visited[nx, ny] and is_red(image[nx, ny])):
                visited[nx, ny] = True
                queue.append((nx, ny))

def process_image(image_path):
    # Load image
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img, dtype=np.uint8)
    height, width = img_array.shape[:2]
    
    # Initialize visited array and output
    visited = np.zeros((height, width), dtype=bool)
    output = img_array.copy()
    blob_count = 0
    
    # Scan for blobs
    print(f"Starting to scan {height}x{width} image for red blobs...")
    scan_start = time.time()
    
    for x in range(height):
        # Progress indicator every 1000 rows
        if x % 1000 == 0:
            elapsed = time.time() - scan_start
            progress = (x / height) * 100
            print(f"Progress: {progress:.1f}% (row {x}/{height}) - {blob_count} blobs found so far - {elapsed:.1f}s elapsed")
        
        for y in range(width):
            if is_red(output[x, y]) and not visited[x, y]:
                # Generate random color (0-254 ensures not white or bright red)
                new_color = (random.randint(0, 254), random.randint(0, 254), random.randint(0, 254))
                bfs_flood_fill(output, visited, x, y, new_color)
                blob_count += 1
                print(f"Found blob #{blob_count} at position ({x}, {y})")
    
    print(f"Scan complete! Found {blob_count} total blobs.")
    
    return Image.fromarray(output), blob_count


HOME = Path.cwd()
image_path = HOME / "images" / "input" / "input_blobs.png"
output_path = HOME / "images" / "output" / "colored_blobs_cpu_mvp.png"

print(f"Starting flood fill processing...")
print(f"Input image: {image_path}")
print(f"Output image: {output_path}")

start_time = timeit.default_timer()
result_img, blob_count = process_image(image_path)
elapsed = timeit.default_timer() - start_time
print(f"\n=== FINAL RESULTS ===")
print(f"Detected {blob_count} blobs in {elapsed:.2f} seconds")
print(f"Saving result to: {output_path}")
result_img.save(output_path)
print("Done!")