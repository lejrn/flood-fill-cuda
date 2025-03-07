# Flood Fill Sequential Implementation

This README provides details on the sequential flood fill algorithm implemented in `src/cpu/sequential.py`.

## Overview

The sequential version uses a breadth-first search (BFS) algorithm to fill connected regions of red pixels with random colors. It detects red pixels based on specific RGB thresholds and fills each contiguous blob uniquely.

## How It Works

- **Red Detection:** A pixel qualifies as red if its red channel is above a threshold while its green and blue channels are sufficiently low.
- **BFS Flood Fill:** Starting from an unvisited red pixel, the algorithm queues adjacent pixels meeting the red criteria and applies a new random color to the entire blob.
- **Output:** The processed image highlights each red blob with a distinct color.

## Usage

1. Ensure Poetry is installed ([Poetry Installation](https://python-poetry.org/docs/#installation)).
2. Install dependencies:
   ```
   poetry install
   ```
3. Place your input image at `images/input/input_blobs.png`.
4. Run the script:
   ```
   poetry run python3 src/cpu/sequential.py
   ```
5. The processed image is saved as `images/output/colored_blobs_cpu_mvp.png` and the blob count is printed to the console.

## Dependencies

- Python 3
- numpy
- Pillow
- numba

## Notes

- This implementation is intended for demonstration and educational purposes.
- For enhanced performance on larger images, consider exploring GPU-based implementations.

Enjoy experimenting with flood fill!
