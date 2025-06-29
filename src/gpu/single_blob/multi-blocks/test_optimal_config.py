#!/usr/bin/env python3
"""
Test the optimal CUDA flood fill configuration on a large scene.
Uses the best configuration found in benchmarking.
"""

import time
import numpy as np
from PIL import Image

# Import our optimized implementation
from kernels_fixed import run_multi_iteration_flood_fill, reset_global_queue
from debug_logging import create_global_queue_arrays
from utils import log_gpu_memory_info
from numba import cuda

def create_large_test_scene(width, height, blob_size):
    """Create a large test scene with a central blob"""
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

def test_optimal_configuration():
    """Test the optimal configuration found in benchmarking"""
    
    print("🚀 TESTING OPTIMAL CUDA FLOOD FILL CONFIGURATION")
    print("=" * 60)
    
    # Optimal configuration from benchmarking
    OPTIMAL_BLOCKS = 40
    OPTIMAL_THREADS = 128
    OPTIMAL_CHUNK = 128
    
    print(f"🎯 Using optimal configuration:")
    print(f"   • {OPTIMAL_BLOCKS} blocks per grid")
    print(f"   • {OPTIMAL_THREADS} threads per block")
    print(f"   • {OPTIMAL_CHUNK} chunk size")
    print(f"   • Total threads: {OPTIMAL_BLOCKS * OPTIMAL_THREADS:,}")
    print()
    
    log_gpu_memory_info()
    
    # Test on multiple scene sizes
    test_scenes = [
        (4000, 4000, 2000),  # 16M pixels, 4M blob
        (6000, 6000, 3000),  # 36M pixels, 9M blob
        (8000, 8000, 4000),  # 64M pixels, 16M blob
    ]
    
    results = []
    
    for width, height, blob_size in test_scenes:
        print(f"\\n🧪 Testing {width}x{height} scene with {blob_size}x{blob_size} blob")
        print("-" * 50)
        
        try:
            # Create test scene
            img_host, start_x, start_y, expected_pixels = create_large_test_scene(width, height, blob_size)
            
            # Setup GPU arrays
            img = cuda.to_device(img_host)
            visited = cuda.device_array((width, height), dtype=np.int32)
            visited[:] = 0
            
            # Global queue arrays
            global_queue_x, global_queue_y, global_queue_front, global_queue_rear = create_global_queue_arrays()
            
            # Debug arrays
            debug_block_usage = cuda.device_array(OPTIMAL_BLOCKS, dtype=np.int32)
            debug_thread_usage = cuda.device_array(OPTIMAL_BLOCKS * OPTIMAL_THREADS, dtype=np.int32)
            debug_warp_usage = cuda.device_array(OPTIMAL_BLOCKS * 2, dtype=np.int32)
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
            
            # Run flood fill with optimal configuration
            print(f"⏱️  Starting flood fill...")
            start_time = time.time()
            
            iterations = run_multi_iteration_flood_fill(
                img, visited, start_x, start_y, width, height, new_color_gpu,
                global_queue_x, global_queue_y, global_queue_front, global_queue_rear,
                debug_block_usage, debug_thread_usage, debug_warp_usage, debug_pixel_count,
                blocks_per_grid=OPTIMAL_BLOCKS,
                threads_per_block=OPTIMAL_THREADS,
                chunk_size=OPTIMAL_CHUNK,
                max_iterations=5000
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
            
            total_blocks = OPTIMAL_BLOCKS
            total_threads = OPTIMAL_BLOCKS * OPTIMAL_THREADS
            total_warps = OPTIMAL_BLOCKS * (OPTIMAL_THREADS // 32)
            
            pixels_per_second = pixels_processed / (execution_time / 1000) if execution_time > 0 else 0
            
            # Results
            result = {
                'scene_size': f"{width}x{height}",
                'total_pixels': width * height,
                'blob_pixels': expected_pixels,
                'execution_time_ms': execution_time,
                'iterations': iterations,
                'pixels_processed': pixels_processed,
                'pixels_per_second': pixels_per_second,
                'blocks_used': blocks_used,
                'threads_used': threads_used,
                'warps_used': warps_used,
                'block_utilization_pct': (blocks_used / total_blocks * 100) if total_blocks > 0 else 0,
                'thread_utilization_pct': (threads_used / total_threads * 100) if total_threads > 0 else 0,
                'warp_utilization_pct': (warps_used / total_warps * 100) if total_warps > 0 else 0,
                'time_per_iteration_ms': execution_time / iterations if iterations > 0 else 0,
                'memory_efficiency': pixels_processed / (width * height) if width * height > 0 else 0
            }
            
            results.append(result)
            
            print(f"✅ Results for {width}x{height} scene:")
            print(f"   • Execution time: {execution_time:.2f} ms")
            print(f"   • Iterations: {iterations}")
            print(f"   • Pixels processed: {pixels_processed:,}")
            print(f"   • Performance: {pixels_per_second:,.0f} pixels/second")
            print(f"   • GPU utilization: {result['block_utilization_pct']:.1f}% blocks, {result['thread_utilization_pct']:.1f}% threads")
            print(f"   • Time per iteration: {result['time_per_iteration_ms']:.3f} ms")
            print(f"   • Memory efficiency: {result['memory_efficiency']:.3f} (processed/total pixels)")
            
            # Save output images for the largest scene
            if width == 8000:
                print(f"💾 Saving output images for {width}x{height} scene...")
                
                # Copy results back to host
                img_result = img.copy_to_host()
                visited_result = visited.copy_to_host()
                
                # Save flood filled image
                img_pil = Image.fromarray(img_result)
                output_path = f'./images/results/optimal_config_{width}x{height}_flood_fill.png'
                img_pil.save(output_path)
                print(f"   ✅ Flood filled image saved: {output_path}")
                
                # Save visited mask
                visited_normalized = (visited_result > 0).astype(np.uint8) * 255
                visited_pil = Image.fromarray(visited_normalized, mode='L')
                visited_path = f'./images/results/optimal_config_{width}x{height}_visited.png'
                visited_pil.save(visited_path)
                print(f"   ✅ Visited mask saved: {visited_path}")
            
        except Exception as e:
            print(f"❌ Error testing {width}x{height} scene: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print(f"\\n📊 PERFORMANCE SUMMARY:")
    print("=" * 40)
    print(f"{'Scene Size':<12} {'Pixels/s':<15} {'Time (ms)':<10} {'Util %':<8} {'Iterations':<10}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['scene_size']:<12} {result['pixels_per_second']:>13,.0f} "
              f"{result['execution_time_ms']:>8.1f} {result['block_utilization_pct']:>6.1f} "
              f"{result['iterations']:>9}")
    
    if results:
        avg_speed = sum(r['pixels_per_second'] for r in results) / len(results)
        max_speed = max(r['pixels_per_second'] for r in results)
        print(f"\\nAverage performance: {avg_speed:,.0f} pixels/s")
        print(f"Maximum performance: {max_speed:,.0f} pixels/s")
    
    # Scalability analysis
    if len(results) >= 2:
        print(f"\\n📈 SCALABILITY ANALYSIS:")
        print(f"   • Performance generally {'increases' if results[-1]['pixels_per_second'] > results[0]['pixels_per_second'] else 'decreases'} with scene size")
        print(f"   • GPU utilization: {results[0]['block_utilization_pct']:.1f}% → {results[-1]['block_utilization_pct']:.1f}%")
        print(f"   • Time per iteration: {results[0]['time_per_iteration_ms']:.3f}ms → {results[-1]['time_per_iteration_ms']:.3f}ms")
    
    print(f"\\n🎯 OPTIMAL CONFIGURATION VALIDATION COMPLETE!")
    return results

def main():
    """Main test function"""
    print("🧪 CUDA Flood Fill Optimal Configuration Test")
    print("=" * 50)
    print(f"⏰ Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        results = test_optimal_configuration()
        
        # Save results
        import json
        with open('./benchmark_results/optimal_config_test.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\\n💾 Results saved to: ./benchmark_results/optimal_config_test.json")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print(f"\\n⏰ Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("✅ Optimal configuration test completed successfully!")
    return 0

if __name__ == '__main__':
    exit_code = main()
    import sys
    sys.exit(exit_code)
