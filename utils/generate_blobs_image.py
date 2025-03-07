import random
import numpy as np
from PIL import Image
from collections import deque
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn
from rich.console import Console

# A helper to convert an (R, G, B) color to a single int.
def color_to_int(color):
    r, g, b = color
    return (r << 16) | (g << 8) | b

# Generate a random nonzero color (so that 0 means "empty")
def next_color():
    while True:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        col_int = color_to_int(color)
        if col_int != 0:  # avoid using 0 as a valid color
            return col_int
        
# Get the red color value (now we'll use direct RGB instead of int encoding)
def get_red():
    return np.array([255, 0, 0], dtype=np.uint8)

# This class represents a blob to draw.
class BlobObject:
    def __init__(self, max_size, color):
        self.max_size = max_size  # maximum radius of the blob
        self.color = color  # color as an int
        self.growth_factor = random.uniform(0.3, 0.7)  # controls blob shape irregularity

# Check if the point is within the image bounds and not already filled
def is_valid_point(x, y, array, img_width, img_height):
    return (0 <= x < img_width and 
            0 <= y < img_height and 
            np.array_equal(array[y, x], [255, 255, 255]))  # Check if it's white/background

# Generate a continuous blob starting at (seed_x, seed_y)
def generate_blob(seed_x, seed_y, max_size, array, img_width, img_height):
    # Always use red color for all blobs
    red_color = get_red()
    
    # Queue for growing the blob (breadth-first)
    queue = deque([(seed_x, seed_y)])
    visited = set([(seed_x, seed_y)])
    blob_pixels = set()
    
    # Determine blob size - vary between min_points and max_points
    area_factor = random.uniform(0.4, 0.9)  # How much of the max area to fill
    max_points = int(np.pi * max_size * max_size * area_factor)  # Approximate circle area
    min_points = max(5, int(max_points * 0.3))  # At least 5 pixels, or 30% of max
    target_size = random.randint(min_points, max_points)
    
    # Only use cardinal directions to ensure blob connectivity
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    while queue and len(blob_pixels) < target_size:
        x, y = queue.popleft()
        
        # Skip if already colored or out of bounds
        if not is_valid_point(x, y, array, img_width, img_height):
            continue
            
        # Add to blob
        blob_pixels.add((x, y))
        
        # Randomize direction order for more organic growth
        random.shuffle(directions)
        
        # Use distance-based probability to maintain somewhat circular shape
        distance = max(abs(x - seed_x), abs(y - seed_y))
        base_prob = 1.0 - (distance / max_size) ** 1.5  # Steeper falloff for more compact blobs
        
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            
            if ((new_x, new_y) not in visited and 
                random.random() < base_prob and
                is_valid_point(new_x, new_y, array, img_width, img_height)):
                queue.append((new_x, new_y))
                visited.add((new_x, new_y))
    
    # Only color the pixels if we've met the minimum size requirement
    if len(blob_pixels) >= min_points:
        for x, y in blob_pixels:
            array[y, x] = red_color
        return len(blob_pixels)
    
    return 0  # Return 0 if blob couldn't grow enough

# Check if a new blob can be placed with adequate separation from existing blobs
def can_place_blob(seed_x, seed_y, separation, array, img_width, img_height):
    # Check a larger area around the seed for separation
    for y in range(max(0, seed_y - separation), min(img_height, seed_y + separation + 1)):
        for x in range(max(0, seed_x - separation), min(img_width, seed_x + separation + 1)):
            if not np.array_equal(array[y, x], [255, 255, 255]):  # If not background color
                return False
    return True

# Main function
def main():
    console = Console()
    
    # Image dimensions and separation parameters
    img_width = 9000
    img_height = 9000
    separation = 20  # Minimum pixels between blobs
    
    # Blob parameters
    min_blob_size = 5    # Minimum radius
    max_blob_size = 100   # Maximum radius
    num_blobs = 3000    # Target number of blobs
    
    # Create an RGB array (white background)
    array = np.full((img_height, img_width, 3), 255, dtype=np.uint8)

    # Use rich progress bar
    console.print("[bold blue]Generating random red blobs...")
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[green]Creating blobs...", total=num_blobs)
        
        blobs_created = 0
        max_attempts = num_blobs * 20  # Increased attempts for better placement
        attempts = 0
        
        while blobs_created < num_blobs and attempts < max_attempts:
            # Generate a random position for the blob
            seed_x = random.randint(separation + max_blob_size, img_width - separation - max_blob_size - 1)
            seed_y = random.randint(separation + max_blob_size, img_height - separation - max_blob_size - 1)
            
            if can_place_blob(seed_x, seed_y, separation, array, img_width, img_height):
                # Generate a random blob size
                blob_max_size = random.randint(min_blob_size, max_blob_size)
                
                # Generate the blob
                blob_size = generate_blob(seed_x, seed_y, blob_max_size, array, img_width, img_height)
                
                if blob_size > 0:
                    blobs_created += 1
                    progress.update(task, advance=1)
            
            attempts += 1
            if attempts % 100 == 0:
                progress.update(task, description=f"[yellow]Creating blobs... ({blobs_created}/{num_blobs}, {attempts} attempts)")
        
        progress.update(task, completed=blobs_created)

    # Save the image
    img = Image.fromarray(array)
    img.save(HOME / "images" / "input" / "input_blobs.png")
    console.print(f"[bold green]✓ Image saved as input_blobs.png with {blobs_created} red blobs")

from pathlib import Path
HOME = Path.cwd()

if __name__ == '__main__':
    main()