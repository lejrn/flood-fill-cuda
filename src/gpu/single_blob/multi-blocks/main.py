"""
Main script for optimized multi-SM flood fill algorithm.
Standalone script that can be run directly with: poetry run python main.py
"""

import os
import sys
import numpy as np
import time
import traceback
from PIL import Image
from numba import cuda
import time

# Import from same directory (not relative imports)
from utils import (log_gpu_memory_info, log_configuration_details, 
                   PerformanceLogger, format_memory_size)
from setup import setup_large_scene
from kernels import optimized_flood_fill
from debug_logging import (log_kernel_launch_info, create_global_queue_arrays, 
                         analyze_debug_arrays, log_debug_array_creation)

# Initialize performance logger
perf_logger = PerformanceLogger()

def ensure_output_dir():
    """Ensure output directory exists"""
    print("📁 Setting up output directory...")
    output_dir = './images/results'
    os.makedirs(output_dir, exist_ok=True)
    print(f"   ✅ Output directory ready: {output_dir}")
    return output_dir

def save_results(img_result, visited_result, output_dir, prefix='optimized'):
    """Save results to image files with detailed logging"""
    print(f"💾 Saving results with prefix '{prefix}'...")
    
    # Save processed image
    img_path = f'{output_dir}/{prefix}_flood_fill.png'
    img_result_pil = Image.fromarray(img_result)
    img_result_pil.save(img_path)
    print(f"   ✅ Processed image saved: {img_path}")
    
    # Save visited mask
    visited_path = f'{output_dir}/{prefix}_flood_fill_visited.png'
    visited_result_pil = Image.fromarray(visited_result.astype(np.uint8) * 255)
    visited_result_pil.save(visited_path)
    print(f"   ✅ Visited mask saved: {visited_path}")
    
    # Log file sizes
    img_size = os.path.getsize(img_path)
    visited_size = os.path.getsize(visited_path)
    print(f"   📊 File sizes: {format_memory_size(img_size)} + {format_memory_size(visited_size)}")
    
    return img_path, visited_path

def run_optimized_flood_fill_with_debug():
    """Run optimized flood fill with comprehensive debug tracking"""
    print("🎯 Running Optimized Multi-Block Flood Fill with Debug Tracking")
    print("=" * 70)
    
    # Setup scene
    print("🎨 Setting up test scene...")
    scene_data = setup_large_scene()
    img, visited, start_x, start_y, width, height, new_color, threads_per_block, blocks_per_grid = scene_data
    
    print(f"   • Scene: {width}x{height} pixels")
    print(f"   • Start: ({start_x}, {start_y})")
    print(f"   • GPU: {blocks_per_grid} blocks, {threads_per_block} threads/block")
    
    # Create global queue arrays
    global_queue_x, global_queue_y, global_queue_front, global_queue_rear = create_global_queue_arrays()
    
    # Create debug arrays
    total_threads = blocks_per_grid * threads_per_block
    total_warps = total_threads // 32
    
    log_debug_array_creation(blocks_per_grid, threads_per_block)
    
    # Initialize debug arrays to zero
    debug_block_usage = cuda.device_array(blocks_per_grid, dtype=np.int32)
    debug_thread_usage = cuda.device_array(total_threads, dtype=np.int32)
    debug_warp_usage = cuda.device_array(total_warps, dtype=np.int32)
    debug_pixel_count = cuda.device_array(1, dtype=np.int32)
    debug_queue_usage = cuda.device_array(1, dtype=np.int32)
    debug_iteration_count = cuda.device_array(1, dtype=np.int32)

    # Copy image data to GPU
    img_gpu = cuda.to_device(img)
    visited_gpu = cuda.to_device(visited)
    new_color_gpu = cuda.to_device(new_color)

    print()
    log_kernel_launch_info("Optimized Multi-Block Flood Fill", blocks_per_grid, threads_per_block)
    
    # Launch kernel with debug arrays
    perf_logger.start_timer("Kernel Execution")
    
    optimized_flood_fill[blocks_per_grid, threads_per_block](
        img_gpu, visited_gpu, start_x, start_y, width, height, new_color_gpu,
        global_queue_x, global_queue_y, global_queue_front, global_queue_rear,
        debug_block_usage, debug_thread_usage, debug_warp_usage, debug_pixel_count, debug_queue_usage, debug_iteration_count
    )
    
    cuda.synchronize()
    kernel_time = perf_logger.end_timer("Kernel Execution")
    
    print(f"✅ Kernel execution completed in {kernel_time:.2f} ms")
    
    # Copy results back to host
    img_result = img_gpu.copy_to_host()
    visited_result = visited_gpu.copy_to_host()
    
    # Copy debug arrays back for analysis
    debug_block_usage_host = debug_block_usage.copy_to_host()
    debug_thread_usage_host = debug_thread_usage.copy_to_host()
    debug_warp_usage_host = debug_warp_usage.copy_to_host()
    debug_pixel_count_host = debug_pixel_count.copy_to_host()
    debug_queue_usage_host = debug_queue_usage.copy_to_host()
    debug_iteration_count_host = debug_iteration_count.copy_to_host()
    
    # Analyze debug arrays
    debug_stats = analyze_debug_arrays(
        debug_block_usage_host, debug_thread_usage_host, debug_warp_usage_host,
        debug_pixel_count_host, debug_queue_usage_host, debug_iteration_count_host, blocks_per_grid, threads_per_block
    )
    
    return img_result, visited_result, debug_stats


def main():
    """Main execution with debug tracking and result saving"""
    print("🚀 CUDA Flood Fill - Multi-Block Implementation with Debug Tracking")
    print("=" * 70)
    
    log_gpu_memory_info()
    log_configuration_details()
    
    output_dir = ensure_output_dir()
    print()
    
    try:
        # Run optimized flood fill with debug tracking
        img_result, visited_result, debug_stats = run_optimized_flood_fill_with_debug()
        
        # Save results
        print()
        save_results(img_result, visited_result, output_dir, 'optimized_debug')
        
        print()
        print("📊 Execution Summary:")
        print(f"   • GPU Utilization: {debug_stats['block_utilization']:.1f}% blocks, "
              f"{debug_stats['thread_utilization']:.1f}% threads, {debug_stats['warp_utilization']:.1f}% warps")
        print(f"   • Pixels Processed: {debug_stats['pixels_processed']:,}")
        print(f"   • Queue Peak Usage: {debug_stats['max_queue_usage']:,} ({debug_stats['queue_utilization']:.1f}%)")
        print(f"   • Iterations Executed: {debug_stats['iterations_executed']:,}")
        print(f"   • Efficiency: {debug_stats['pixels_processed']/debug_stats['threads_used']:.1f} pixels/thread")
        
        perf_logger.log_counters()
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    print("🎯 Starting CUDA Flood Fill Multi-Block Implementation")
    print(f"⏰ Execution started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    exit_code = main()
        
    print()
    print(f"⏰ Execution finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎯 Script execution completed!")
    
    sys.exit(exit_code)
