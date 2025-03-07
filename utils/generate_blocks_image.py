import random
import numpy as np
from PIL import Image
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
        
# Return a specific turquoise color as an integer
def get_red():
    return color_to_int((255, 0, 0))

# This class represents a block to draw.
class FillObject:
    def __init__(self, width, color):
        self.width = width  # block size (in pixels)
        self.color = color  # color as an int

# Choose one object from the list whose width is less than or equal to max_size.
def get_object(objects, max_size):
    valid = [obj for obj in objects if obj.width <= max_size]
    return random.choice(valid) if valid else None

# This function determines if a block can be placed starting at (x,y)
# and returns the maximum size available.
def get_max_size(x, y, img_width, img_height, array, pad, largest):
    # First, ensure that the surrounding pad region is free.
    for y2 in range(y - pad, y + pad):
        for x2 in range(x - pad, x + pad):
            if (x2 < 0 or y2 < 0 or x2 >= img_width or y2 >= img_height or array[y2, x2] != 0):
                return False, 0

    max_size = 0
    initial_max = largest + pad
    # For rows from y-pad up through y, count how many consecutive pixels (to the right) are free.
    for y2 in range(y - pad, y + 1):
        current_max = 0
        # Use the previous max_size (if any) as a limit
        limit = initial_max if max_size <= 0 else max_size
        for x2 in range(x, min(img_width, x + limit)):
            if array[y2, x2] != 0:
                break
            current_max += 1
        max_size = current_max

    # Adjust max_size by the pad and image boundaries.
    max_size = min(max_size - pad, img_width - x - pad, img_height - y - pad)
    return (max_size > 0), max_size

# This function "draws" blocks on the array.
def generate(img_width, img_height, array, pad, objects, largest):
    y = pad
    while y < img_height:
        x = pad
        while x < img_width:
            valid, max_size = get_max_size(x, y, img_width, img_height, array, pad, largest)
            if not valid:
                x += 1
                continue

            obj = get_object(objects, max_size)
            if obj is None:
                x += 1
                continue

            # Draw the block: fill a square of size obj.width starting at (x, y)
            for y2 in range(y, y + obj.width):
                for x2 in range(x, x + obj.width):
                    array[y2, x2] = obj.color

            # Move x to the right of the drawn block plus padding.
            x += obj.width + pad
        y += 1

# Convert our integer array into an RGB image.
def int_array_to_rgb_image(array):
    height, width = array.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    # Set background for "empty" pixels (0) to white.
    mask = (array == 0)
    rgb[mask] = [255, 255, 255]
    
    # For nonzero entries, decode the color.
    nonzero_indices = np.where(~mask)
    values = array[nonzero_indices]
    # Extract red, green, blue components
    r = (values >> 16) & 0xFF
    g = (values >> 8) & 0xFF
    b = values & 0xFF
    rgb[nonzero_indices[0], nonzero_indices[1]] = np.stack([r, g, b], axis=-1)
    return Image.fromarray(rgb, 'RGB')

# Main function: set up parameters, generate the image, and display/save it.
def main():
    # Import rich for progress display

    console = Console()
    
    # Image dimensions and padding
    img_width = 1000
    img_height = 1000
    pad = 3  # You can adjust padding as needed

    with console.status("[bold green]Creating objects..."):
        # Create a list of FillObjects.
        # (Note: starting from 1 to avoid a block of width 0.)
        num_objects = 40
        objects = [FillObject(width, get_red()) for width in range(1, num_objects + 1)]
        largest = max(obj.width for obj in objects)

        # Create an array to represent the image (0 means unfilled).
        array = np.zeros((img_height, img_width), dtype=np.int32)

    # Use rich progress bar for the image generation
    console.print("[bold blue]Generating pixelated pattern...")
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[green]Processing rows...", total=img_height-pad)
        
        # Custom version of generate function to update progress bar
        y = pad
        while y < img_height:
            x = pad
            while x < img_width:
                valid, max_size = get_max_size(x, y, img_width, img_height, array, pad, largest)
                if not valid:
                    x += 1
                    continue

                obj = get_object(objects, max_size)
                if obj is None:
                    x += 1
                    continue

                # Draw the block: fill a square of size obj.width starting at (x, y)
                for y2 in range(y, y + obj.width):
                    for x2 in range(x, x + obj.width):
                        array[y2, x2] = obj.color

                # Move x to the right of the drawn block plus padding.
                x += obj.width + pad
            progress.update(task, advance=1)
            y += 1

    # Convert the integer array to an RGB image.
    with console.status("[bold green]Converting to RGB image..."):
        img = int_array_to_rgb_image(array)
    
    # Save the image
    img.save(HOME / "images" / "input" / "input_blocks.png")
    console.print("[bold green]✓ Image saved as output.png")

from pathlib import Path
HOME = Path.cwd()

if __name__ == '__main__':
    main()
