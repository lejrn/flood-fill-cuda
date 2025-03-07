import numpy as np
from PIL import Image
import random
from collections import deque

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
    for x in range(height):
        for y in range(width):
            if is_red(output[x, y]) and not visited[x, y]:
                # Generate random color (0-254 ensures not white or bright red)
                new_color = (random.randint(0, 254), random.randint(0, 254), random.randint(0, 254))
                bfs_flood_fill(output, visited, x, y, new_color)
                blob_count += 1
    
    return Image.fromarray(output), blob_count


from pathlib import Path
HOME = Path.cwd()
image_path = HOME / "images" / "input" / "input_blobs.png"
output_path = HOME / "images" / "output" / "colored_blobs_cpu_mvp.png"
result_img, blob_count = process_image(image_path)
print(f"Detected {blob_count} blobs")
result_img.save(output_path)