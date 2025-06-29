"""
Test script for the fixed multi-block flood fill implementation.
This version uses separate kernel launches per iteration to ensure proper multi-block execution.
"""

import numpy as np
from numba import cuda
import time

# Import the fixed kernels
from kernels_fixed import run_multi_iteration_flood_fill, reset_global_queue
from utils import RTX_4060_CONFIG

def test_fixed_flood_fill():
    """Test the fixed multi-block flood fill with proper synchronization."""
    print("🧪 Testing Fixed Multi-Block Flood Fill")
    print("=" * 50)
    
    # Create a smaller test scene for quick verification
    width, height = 2000, 2000
    print(f"📏 Test scene: {width}x{height} pixels")
    
    # Create test image with red blob
    img_host = np.full((width, height, 3), 255, dtype=np.uint8)  # White background
    blob_size = 500
    start_x, start_y = width // 2, height // 2
    
    # Create red blob
    for x in range(start_x - blob_size//2, start_x + blob_size//2):
        for y in range(start_y - blob_size//2, start_y + blob_size//2):
            if 0 <= x < width and 0 <= y < height:
                img_host[x, y] = [255, 0, 0]  # Red pixels
    
    print(f"🔴 Red blob: {blob_size}x{blob_size} at ({start_x}, {start_y})")
    
    # Setup GPU arrays
    img = cuda.to_device(img_host)
    visited = cuda.device_array((width, height), dtype=np.int32)
    visited[:] = 0
    
    # Queue arrays
    queue_capacity = 500000
    global_queue_x = cuda.device_array(queue_capacity, dtype=np.int32)
    global_queue_y = cuda.device_array(queue_capacity, dtype=np.int32)
    global_queue_front = cuda.device_array(1, dtype=np.int32)
    global_queue_rear = cuda.device_array(1, dtype=np.int32)
    
    # Debug arrays
    debug_block_usage = cuda.device_array(40, dtype=np.int32)
    debug_thread_usage = cuda.device_array(40 * 64, dtype=np.int32)
    debug_warp_usage = cuda.device_array(40 * 2, dtype=np.int32)
    debug_pixel_count = cuda.device_array(1, dtype=np.int32)
    
    # Initialize debug arrays
    debug_block_usage[:] = 0
    debug_thread_usage[:] = 0
    debug_warp_usage[:] = 0
    debug_pixel_count[:] = 0
    
    # Reset queue
    reset_global_queue(global_queue_front, global_queue_rear)
    
    # Set fill color
    new_color = np.array([0, 0, 255], dtype=np.uint8)  # Blue
    
    print("🚀 Starting fixed multi-block flood fill...")
    start_time = time.time()
    
    # Run the fixed algorithm
    iterations = run_multi_iteration_flood_fill(
        img, visited, start_x, start_y, width, height, new_color,
        global_queue_x, global_queue_y, global_queue_front, global_queue_rear,
        debug_block_usage, debug_thread_usage, debug_warp_usage, debug_pixel_count,
        blocks_per_grid=40, threads_per_block=64, max_iterations=1000
    )
    
    end_time = time.time()
    execution_time = (end_time - start_time) * 1000  # Convert to ms
    
    print(f"✅ Algorithm completed in {execution_time:.2f} ms")
    print(f"🔄 Iterations executed: {iterations}")
    
    # Get debug statistics
    debug_block_usage_host = debug_block_usage.copy_to_host()
    debug_thread_usage_host = debug_thread_usage.copy_to_host()
    debug_warp_usage_host = debug_warp_usage.copy_to_host()
    debug_pixel_count_host = debug_pixel_count.copy_to_host()
    
    # Calculate utilization
    blocks_used = np.sum(debug_block_usage_host > 0)
    threads_used = np.sum(debug_thread_usage_host > 0)
    warps_used = np.sum(debug_warp_usage_host > 0)
    pixels_processed = debug_pixel_count_host[0]
    
    print("\n📊 GPU Utilization Analysis:")
    print(f"  🧱 Blocks Used: {blocks_used}/40 ({blocks_used/40*100:.1f}%)")
    print(f"  🧵 Threads Used: {threads_used}/2560 ({threads_used/2560*100:.1f}%)")
    print(f"  🌊 Warps Used: {warps_used}/80 ({warps_used/80*100:.1f}%)")
    print(f"  🎨 Pixels Processed: {pixels_processed:,}")
    
    if blocks_used > 1:
        print("🎉 SUCCESS: Multiple blocks participated!")
        print(f"   Block utilization improved from 2.5% to {blocks_used/40*100:.1f}%")
    else:
        print("❌ ISSUE: Still only using 1 block")
    
    # Show which specific blocks were used
    active_blocks = [i for i, usage in enumerate(debug_block_usage_host) if usage > 0]
    print(f"  🔧 Active blocks: {active_blocks}")
    
    return blocks_used > 1


if __name__ == "__main__":
    success = test_fixed_flood_fill()
    if success:
        print("\n✅ Test PASSED: Multi-block execution achieved!")
    else:
        print("\n❌ Test FAILED: Still single-block execution")
