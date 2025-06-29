"""
Main script for the optimized multi-block flood fill algorithm.
Uses the fixed kernels with minimal host-device transfers.
"""

import os
import sys
import numpy as np
import time
from PIL import Image
from numba import cuda

# Import our optimized implementation
from kernels_fixed import run_multi_iteration_flood_fill, reset_global_queue
from debug_logging import create_global_queue_arrays
from utils import log_gpu_memory_info, log_configuration_details

def ensure_output_dir():
    """Ensure output directory exists"""
    print("📁 Setting up output directory...")
    output_dir = './images/results'
    os.makedirs(output_dir, exist_ok=True)
    print(f"   ✅ Output directory ready: {output_dir}")
    return output_dir

def create_large_scene():
    """Create 8000x8000 scene with 4000x4000 red blob"""
    print("🎨 Creating large scene: 8000x8000 with 4000x4000 red blob...")
    
    width, height = 8000, 8000
    blob_size = 4000
    
    # Create white background
    img = np.full((width, height, 3), 255, dtype=np.uint8)
    
    # Create centered red blob
    start_x = width // 2
    start_y = height // 2
    blob_start_x = start_x - blob_size // 2
    blob_end_x = start_x + blob_size // 2
    blob_start_y = start_y - blob_size // 2
    blob_end_y = start_y + blob_size // 2
    
    # Fill red blob
    for x in range(blob_start_x, blob_end_x):
        for y in range(blob_start_y, blob_end_y):
            if 0 <= x < width and 0 <= y < height:
                img[x, y] = [255, 0, 0]  # Red pixels
    
    print(f"   ✅ Scene created: {width}x{height} pixels")
    print(f"   🔴 Red blob: {blob_size}x{blob_size} at center ({start_x}, {start_y})")
    print(f"   📊 Red pixels: {blob_size * blob_size:,}")
    
    return img, width, height, start_x, start_y

def save_results(img_result, visited_result, output_dir, prefix='optimized_8000x8000'):
    """Save results to image files"""
    print(f"💾 Saving results with prefix '{prefix}'...")
    
    # Save processed image
    img_path = f'{output_dir}/{prefix}_flood_fill.png'
    img_result_pil = Image.fromarray(img_result)
    img_result_pil.save(img_path)
    print(f"   ✅ Processed image saved: {img_path}")
    
    # Save visited mask
    visited_path = f'{output_dir}/{prefix}_visited.png'
    visited_result_pil = Image.fromarray(visited_result.astype(np.uint8) * 255)
    visited_result_pil.save(visited_path)
    print(f"   ✅ Visited mask saved: {visited_path}")
    
    # Log file sizes
    img_size = os.path.getsize(img_path) / (1024 * 1024)  # MB
    visited_size = os.path.getsize(visited_path) / (1024 * 1024)  # MB
    print(f"   📁 File sizes: image={img_size:.1f}MB, visited={visited_size:.1f}MB")

def run_optimized_large_scene():
    """Run the optimized flood fill on 8000x8000 scene"""
    print("🚀 Running Optimized Flood Fill on Large Scene")
    print("=" * 60)
    
    # Create large scene
    img_host, width, height, start_x, start_y = create_large_scene()
    
    # GPU configuration for large scene
    blocks_per_grid = 48
    threads_per_block = 128
    print(f"🔧 GPU Configuration: {blocks_per_grid} blocks, {threads_per_block} threads/block")
    
    # Setup GPU arrays
    print("🔄 Setting up GPU arrays...")
    img = cuda.to_device(img_host)
    visited = cuda.device_array((width, height), dtype=np.int32)
    visited[:] = 0
    
    # Global queue arrays
    global_queue_x, global_queue_y, global_queue_front, global_queue_rear = create_global_queue_arrays()
    
    # Debug arrays
    debug_block_usage = cuda.device_array(blocks_per_grid, dtype=np.int32)
    debug_thread_usage = cuda.device_array(blocks_per_grid * threads_per_block, dtype=np.int32)
    debug_warp_usage = cuda.device_array(blocks_per_grid * 2, dtype=np.int32)
    debug_pixel_count = cuda.device_array(1, dtype=np.int32)
    
    # Initialize debug arrays
    debug_block_usage[:] = 0
    debug_thread_usage[:] = 0
    debug_warp_usage[:] = 0
    debug_pixel_count[:] = 0
    
    # Reset queue
    reset_global_queue(global_queue_front, global_queue_rear)
    
    # Set fill color (blue)
    new_color = np.array([0, 0, 255], dtype=np.uint8)
    new_color_gpu = cuda.to_device(new_color)
    
    print("🚀 Starting optimized large-scene flood fill...")
    start_time = time.time()
    
    # Run the optimized algorithm
    iterations = run_multi_iteration_flood_fill(
        img, visited, start_x, start_y, width, height, new_color_gpu,
        global_queue_x, global_queue_y, global_queue_front, global_queue_rear,
        debug_block_usage, debug_thread_usage, debug_warp_usage, debug_pixel_count,
        blocks_per_grid=blocks_per_grid, threads_per_block=threads_per_block, max_iterations=10000
    )
    
    end_time = time.time()
    execution_time = (end_time - start_time) * 1000  # Convert to ms
    
    print(f"✅ Large scene processing completed in {execution_time:.2f} ms ({execution_time/1000:.2f} seconds)")
    
    # Get results
    img_result = img.copy_to_host()
    visited_result = visited.copy_to_host()
    
    # Get utilization statistics
    debug_block_usage_host = debug_block_usage.copy_to_host()
    debug_thread_usage_host = debug_thread_usage.copy_to_host()
    debug_warp_usage_host = debug_warp_usage.copy_to_host()
    debug_pixel_count_host = debug_pixel_count.copy_to_host()
    
    # Calculate utilization
    blocks_used = np.sum(debug_block_usage_host > 0)
    threads_used = np.sum(debug_thread_usage_host > 0)
    warps_used = np.sum(debug_warp_usage_host > 0)
    pixels_processed = debug_pixel_count_host[0]
    
    print("\n📊 Large Scene Performance Analysis:")
    print(f"  🧱 Blocks Used: {blocks_used}/{blocks_per_grid} ({blocks_used/blocks_per_grid*100:.1f}%)")
    print(f"  🧵 Threads Used: {threads_used}/{blocks_per_grid*threads_per_block} ({threads_used/(blocks_per_grid*threads_per_block)*100:.1f}%)")
    print(f"  🌊 Warps Used: {warps_used}/{blocks_per_grid*2} ({warps_used/(blocks_per_grid*2)*100:.1f}%)")
    print(f"  🎨 Pixels Processed: {pixels_processed:,}")
    print(f"  ⚡ Performance: {pixels_processed/(execution_time/1000):.0f} pixels/second")
    print(f"  🔄 Iterations: {iterations}")
    print(f"  ⏱️  Time per iteration: {execution_time/iterations:.2f} ms")
    
    return img_result, visited_result, {
        'execution_time': execution_time,
        'iterations': iterations,
        'pixels_processed': pixels_processed,
        'blocks_used': blocks_used,
        'threads_used': threads_used,
        'warps_used': warps_used,
        'block_utilization': blocks_used/blocks_per_grid*100,
        'thread_utilization': threads_used/(blocks_per_grid*threads_per_block)*100,
        'warp_utilization': warps_used/(blocks_per_grid*2)*100
    }

def main():
    """Main execution function"""
    print("🎯 CUDA Flood Fill - Optimized Large Scene Implementation")
    print("=" * 70)
    
    log_gpu_memory_info()
    log_configuration_details()
    
    output_dir = ensure_output_dir()
    print()
    
    try:
        # Run optimized flood fill on large scene
        img_result, visited_result, stats = run_optimized_large_scene()
        
        # Save results
        print()
        save_results(img_result, visited_result, output_dir, 'optimized_8000x8000')
        
        print()
        print("🎉 Large Scene Execution Summary:")
        print(f"   • Scene Size: 8000x8000 pixels (64M pixels)")
        print(f"   • Red Blob: 4000x4000 pixels (16M pixels)")
        print(f"   • Execution Time: {stats['execution_time']/1000:.2f} seconds")
        print(f"   • GPU Utilization: {stats['block_utilization']:.1f}% blocks, "
              f"{stats['thread_utilization']:.1f}% threads, {stats['warp_utilization']:.1f}% warps")
        print(f"   • Pixels Processed: {stats['pixels_processed']:,}")
        print(f"   • Performance: {stats['pixels_processed']/(stats['execution_time']/1000):.0f} pixels/second")
        print(f"   • Iterations: {stats['iterations']}")
        print(f"   • Host-Device Transfers: Minimized (every 50 iterations)")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    print("🎯 Starting Large Scene CUDA Flood Fill")
    print(f"⏰ Execution started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    exit_code = main()
        
    print()
    print(f"⏰ Execution finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎯 Large scene processing completed!")
    
    sys.exit(exit_code)
