#!/usr/bin/env python3
"""
Test the newly discovered optimal configuration (24×128, chunk=32) on large scenes.
"""

import os
import sys
import numpy as np
import time
import json
from PIL import Image

from kernels_fixed import run_multi_iteration_flood_fill, reset_global_queue
from debug_logging import create_global_queue_arrays
from utils import log_gpu_memory_info
from numba import cuda

def create_test_scene(width, height, blob_size):
    """Create test scene with specified dimensions"""
    print(f"🎨 Creating large test scene: {width}x{height} with {blob_size}x{blob_size} red blob")
    
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
    red_pixels = 0
    for x in range(blob_start_x, blob_end_x):
        for y in range(blob_start_y, blob_end_y):
            if 0 <= x < width and 0 <= y < height:
                img[x, y] = [255, 0, 0]  # Red pixels
                red_pixels += 1
    
    print(f"   ✅ Scene created with {red_pixels:,} red pixels")
    return img, start_x, start_y, red_pixels

def test_new_optimal_config(width, height, blob_size):
    """Test the new optimal configuration: 24×128, chunk=32"""
    
    # New optimal configuration
    blocks_per_grid = 24
    threads_per_block = 128
    chunk_size = 32
    max_iterations = 3000
    
    print(f"🧪 Testing NEW optimal configuration:")
    print(f"   • {blocks_per_grid} blocks per grid")
    print(f"   • {threads_per_block} threads per block") 
    print(f"   • {chunk_size} chunk size")
    print(f"   • Total threads: {blocks_per_grid * threads_per_block:,}")
    
    # Create test scene
    img_host, start_x, start_y, expected_pixels = create_test_scene(width, height, blob_size)
    
    # Setup GPU arrays
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
    
    # Set fill color
    new_color = np.array([0, 100, 255], dtype=np.uint8)
    new_color_gpu = cuda.to_device(new_color)
    
    print("⏱️  Starting flood fill...")
    start_time = time.time()
    
    iterations = run_multi_iteration_flood_fill(
        img, visited, start_x, start_y, width, height, new_color_gpu,
        global_queue_x, global_queue_y, global_queue_front, global_queue_rear,
        debug_block_usage, debug_thread_usage, debug_warp_usage, debug_pixel_count,
        blocks_per_grid=blocks_per_grid, 
        threads_per_block=threads_per_block,
        chunk_size=chunk_size,
        max_iterations=max_iterations
    )
    
    end_time = time.time()
    execution_time = (end_time - start_time) * 1000  # ms
    
    # Get results
    debug_block_usage_host = debug_block_usage.copy_to_host()
    debug_thread_usage_host = debug_thread_usage.copy_to_host()
    debug_warp_usage_host = debug_warp_usage.copy_to_host()
    debug_pixel_count_host = debug_pixel_count.copy_to_host()
    
    # Calculate metrics
    blocks_used = np.sum(debug_block_usage_host > 0)
    threads_used = np.sum(debug_thread_usage_host > 0)
    warps_used = np.sum(debug_warp_usage_host > 0)
    pixels_processed = debug_pixel_count_host[0]
    
    total_blocks = blocks_per_grid
    total_threads = blocks_per_grid * threads_per_block
    total_warps = blocks_per_grid * (threads_per_block // 32)
    
    pixels_per_second = pixels_processed / (execution_time / 1000) if execution_time > 0 else 0
    block_utilization = (blocks_used / total_blocks * 100) if total_blocks > 0 else 0
    thread_utilization = (threads_used / total_threads * 100) if total_threads > 0 else 0
    memory_efficiency = pixels_processed / (width * height) if (width * height) > 0 else 0
    
    print(f"✅ Results for {width}x{height} scene:")
    print(f"   • Execution time: {execution_time:.2f} ms")
    print(f"   • Iterations: {iterations}")
    print(f"   • Pixels processed: {pixels_processed:,}")
    print(f"   • Performance: {pixels_per_second:,.0f} pixels/second")
    print(f"   • GPU utilization: {block_utilization:.1f}% blocks, {thread_utilization:.1f}% threads")
    print(f"   • Time per iteration: {execution_time/iterations:.3f} ms")
    print(f"   • Memory efficiency: {memory_efficiency:.3f} (processed/total pixels)")
    
    # Save output images for large scenes
    if width >= 6000:
        print("💾 Saving output images...")
        img_result = img.copy_to_host()
        visited_result = visited.copy_to_host()
        
        # Ensure output directory exists
        os.makedirs('./images/results', exist_ok=True)
        
        # Save flood filled image
        img_pil = Image.fromarray(img_result, 'RGB')
        output_path = f'./images/results/new_optimal_{width}x{height}_flood_fill.png'
        img_pil.save(output_path)
        print(f"   ✅ Flood filled image saved: {output_path}")
        
        # Save visited mask
        visited_normalized = (visited_result > 0).astype(np.uint8) * 255
        visited_pil = Image.fromarray(visited_normalized, 'L')
        visited_path = f'./images/results/new_optimal_{width}x{height}_visited.png'
        visited_pil.save(visited_path)
        print(f"   ✅ Visited mask saved: {visited_path}")
    
    return {
        'scene_size': f'{width}x{height}',
        'total_pixels': width * height,
        'blob_pixels': expected_pixels,
        'execution_time_ms': execution_time,
        'iterations': iterations,
        'pixels_processed': pixels_processed,
        'pixels_per_second': pixels_per_second,
        'block_utilization_pct': block_utilization,
        'thread_utilization_pct': thread_utilization,
        'time_per_iteration_ms': execution_time / iterations if iterations > 0 else 0,
        'memory_efficiency': memory_efficiency
    }

def main():
    """Test the new optimal configuration on multiple scene sizes"""
    
    print("🧪 NEW OPTIMAL CONFIGURATION VALIDATION")
    print("=" * 55)
    print("⏰ Started at:", time.strftime('%Y-%m-%d %H:%M:%S'))
    print()
    
    print("🚀 TESTING NEW OPTIMAL CUDA FLOOD FILL CONFIGURATION")
    print("=" * 60)
    print("🎯 Configuration: 24 blocks × 128 threads, chunk size 32")
    print("📈 Expected improvement: +4.2% over previous best")
    print()
    
    log_gpu_memory_info()
    
    # Test scenes
    test_scenes = [
        (2000, 2000, 1000),   # Original benchmark scene
        (4000, 4000, 2000),   # Medium scene
        (6000, 6000, 3000),   # Large scene  
        (8000, 8000, 4000),   # Very large scene
    ]
    
    results = []
    
    for width, height, blob_size in test_scenes:
        print(f"\\n🧪 Testing {width}x{height} scene with {blob_size}x{blob_size} blob")
        print("-" * 50)
        
        try:
            result = test_new_optimal_config(width, height, blob_size)
            results.append(result)
            
        except Exception as e:
            print(f"❌ Error in {width}x{height} test: {e}")
            continue
    
    # Performance summary
    if results:
        print(f"\\n📊 PERFORMANCE SUMMARY:")
        print("=" * 40)
        print(f"{'Scene Size':<12} {'Pixels/s':<15} {'Time (ms)':<10} {'Util %':<8} {'Iterations':<10}")
        print("-" * 65)
        
        total_perf = 0
        for result in results:
            scene = result['scene_size']
            perf = result['pixels_per_second']
            time_ms = result['execution_time_ms']
            util = result['block_utilization_pct']
            iters = result['iterations']
            total_perf += perf
            
            print(f"{scene:<12} {perf:>13,.0f} {time_ms:>8.1f} {util:>6.1f} {iters:>9}")
        
        avg_perf = total_perf / len(results) if results else 0
        max_perf = max(r['pixels_per_second'] for r in results)
        
        print(f"\\nAverage performance: {avg_perf:,.0f} pixels/s")
        print(f"Maximum performance: {max_perf:,.0f} pixels/s")
        
        # Scalability analysis
        if len(results) > 1:
            first_util = results[0]['block_utilization_pct']
            last_util = results[-1]['block_utilization_pct']
            first_time_per_iter = results[0]['time_per_iteration_ms']
            last_time_per_iter = results[-1]['time_per_iteration_ms']
            
            print(f"\\n📈 SCALABILITY ANALYSIS:")
            print(f"   • Performance generally {'increases' if max_perf > results[0]['pixels_per_second'] else 'decreases'} with scene size")
            print(f"   • GPU utilization: {first_util:.1f}% → {last_util:.1f}%")
            print(f"   • Time per iteration: {first_time_per_iter:.3f}ms → {last_time_per_iter:.3f}ms")
        
        print(f"\\n🎯 NEW OPTIMAL CONFIGURATION VALIDATION COMPLETE!")
        
        # Save detailed results
        detailed_results = {
            'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'configuration': {
                'blocks_per_grid': 24,
                'threads_per_block': 128,
                'chunk_size': 32,
                'total_threads': 24 * 128
            },
            'improvement_claim': '+4.2% over previous best',
            'test_results': results,
            'performance_summary': {
                'average_pixels_per_second': avg_perf,
                'maximum_pixels_per_second': max_perf,
                'scenes_tested': len(results)
            }
        }
        
        # Convert numpy types to regular Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        detailed_results = convert_numpy_types(detailed_results)
        
        with open('./benchmark_results/new_optimal_config_validation.json', 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"💾 Detailed results saved to: ./benchmark_results/new_optimal_config_validation.json")
        
    else:
        print("❌ No successful tests completed!")
        return 1
    
    print(f"\\n⏰ Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    return 0

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
