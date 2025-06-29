"""
Comprehensive benchmarking script to find optimal CUDA flood fill parameters.
Tests different combinations of blocks_per_grid, threads_per_block, and chunk_size.
"""

import os
import sys
import numpy as np
import time
import csv
from itertools import product
from numba import cuda

# Import our optimized implementation
from kernels_fixed import run_multi_iteration_flood_fill, reset_global_queue
from debug_logging import create_global_queue_arrays
from utils import log_gpu_memory_info

class BenchmarkConfig:
    """Configuration for benchmark parameters"""
    def __init__(self):
        # Test parameters - focused on most promising ranges
        self.blocks_per_grid_options = [24, 32, 40, 48, 96, 128]
        self.threads_per_block_options = [64, 128, 256, 512]
        self.chunk_size_options = [16, 32, 64, 128]
        
        # Scene configuration - smaller for faster testing
        self.width = 2000
        self.height = 2000
        self.blob_size = 1000
        
        # Test settings
        self.max_iterations = 3000
        self.warmup_runs = 1
        self.benchmark_runs = 2

def create_test_scene(width, height, blob_size):
    """Create test scene with specified dimensions"""
    print(f"🎨 Creating test scene: {width}x{height} with {blob_size}x{blob_size} red blob")
    
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

def run_single_benchmark(img_host, start_x, start_y, width, height, 
                        blocks_per_grid, threads_per_block, chunk_size, max_iterations):
    """Run a single benchmark configuration"""
    
    # Import here to ensure fresh imports
    from kernels_fixed import run_multi_iteration_flood_fill, reset_global_queue
    from debug_logging import create_global_queue_arrays
    
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
    
    # Run benchmark
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
    
    return {
        'blocks_per_grid': blocks_per_grid,
        'threads_per_block': threads_per_block,
        'chunk_size': chunk_size,
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
        'time_per_iteration_ms': execution_time / iterations if iterations > 0 else 0
    }

def run_comprehensive_benchmark():
    """Run comprehensive benchmark across all parameter combinations"""
    config = BenchmarkConfig()
    
    print("🚀 Starting Comprehensive CUDA Flood Fill Benchmarks")
    print("=" * 70)
    
    log_gpu_memory_info()
    
    # Create test scene
    img_host, start_x, start_y, expected_pixels = create_test_scene(
        config.width, config.height, config.blob_size
    )
    
    # Generate all parameter combinations
    combinations = list(product(
        config.blocks_per_grid_options,
        config.threads_per_block_options,
        config.chunk_size_options
    ))
    
    total_tests = len(combinations) * config.benchmark_runs
    print(f"\n📊 Running {len(combinations)} configurations × {config.benchmark_runs} runs = {total_tests} total tests")
    print(f"📐 Test scene: {config.width}×{config.height} with {expected_pixels:,} red pixels")
    
    results = []
    test_count = 0
    
    # Ensure output directory exists
    os.makedirs('./benchmark_results', exist_ok=True)
    
    # Run benchmarks
    for blocks_per_grid, threads_per_block, chunk_size in combinations:
        test_count += 1
        
        # Skip invalid configurations
        if threads_per_block > 1024:  # Max threads per block
            continue
        if blocks_per_grid * threads_per_block > 8192:  # Reasonable total thread limit
            continue
        
        print(f"\n🔧 Test {test_count}/{len(combinations)}: blocks={blocks_per_grid}, threads={threads_per_block}, chunk={chunk_size}")
        
        run_results = []
        
        # Run multiple times for averaging
        for run in range(config.benchmark_runs):
            try:
                result = run_single_benchmark(
                    img_host, start_x, start_y, config.width, config.height,
                    blocks_per_grid, threads_per_block, chunk_size, config.max_iterations
                )
                run_results.append(result)
                
                if run == 0:  # Print first run details
                    print(f"   ⏱️  {result['execution_time_ms']:.2f}ms, "
                          f"{result['pixels_per_second']:,.0f} pixels/s, "
                          f"{result['iterations']} iters, "
                          f"{result['block_utilization_pct']:.1f}% blocks")
                
            except Exception as e:
                print(f"   ❌ Error in run {run+1}: {e}")
                continue
        
        if run_results:
            # Average the results
            avg_result = {}
            for key in run_results[0].keys():
                if isinstance(run_results[0][key], (int, float)):
                    avg_result[key] = np.mean([r[key] for r in run_results])
                else:
                    avg_result[key] = run_results[0][key]  # Keep non-numeric values from first run
            
            results.append(avg_result)
    
    return results

def analyze_and_save_results(results):
    """Analyze results and save to CSV"""
    if not results:
        print("❌ No results to analyze!")
        return
    
    print(f"\n📊 Analyzing {len(results)} benchmark results...")
    
    # Save to CSV
    csv_file = './benchmark_results/cuda_flood_fill_benchmark.csv'
    fieldnames = list(results[0].keys())
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"💾 Results saved to: {csv_file}")
    
    # Find best configurations
    best_by_speed = max(results, key=lambda x: x['pixels_per_second'])
    best_by_utilization = max(results, key=lambda x: x['block_utilization_pct'])
    best_by_efficiency = max(results, key=lambda x: x['pixels_per_second'] * x['block_utilization_pct'] / 100)
    
    print("\n🏆 TOP CONFIGURATIONS:")
    print("\n🚀 FASTEST (pixels/second):")
    print(f"   • {best_by_speed['blocks_per_grid']} blocks, {best_by_speed['threads_per_block']} threads, chunk={best_by_speed['chunk_size']}")
    print(f"   • {best_by_speed['pixels_per_second']:,.0f} pixels/s")
    print(f"   • {best_by_speed['execution_time_ms']:.2f}ms, {best_by_speed['iterations']} iterations")
    print(f"   • GPU: {best_by_speed['block_utilization_pct']:.1f}% blocks, {best_by_speed['thread_utilization_pct']:.1f}% threads")
    
    print("\n🎯 BEST GPU UTILIZATION:")
    print(f"   • {best_by_utilization['blocks_per_grid']} blocks, {best_by_utilization['threads_per_block']} threads, chunk={best_by_utilization['chunk_size']}")
    print(f"   • {best_by_utilization['block_utilization_pct']:.1f}% blocks, {best_by_utilization['thread_utilization_pct']:.1f}% threads")
    print(f"   • {best_by_utilization['pixels_per_second']:,.0f} pixels/s")
    print(f"   • {best_by_utilization['execution_time_ms']:.2f}ms, {best_by_utilization['iterations']} iterations")
    
    print("\n⚖️  BEST OVERALL EFFICIENCY (speed × utilization):")
    print(f"   • {best_by_efficiency['blocks_per_grid']} blocks, {best_by_efficiency['threads_per_block']} threads, chunk={best_by_efficiency['chunk_size']}")
    print(f"   • {best_by_efficiency['pixels_per_second']:,.0f} pixels/s")
    print(f"   • GPU: {best_by_efficiency['block_utilization_pct']:.1f}% blocks, {best_by_efficiency['thread_utilization_pct']:.1f}% threads")
    print(f"   • {best_by_efficiency['execution_time_ms']:.2f}ms, {best_by_efficiency['iterations']} iterations")
    
    # Show parameter trends
    print("\n📈 PARAMETER ANALYSIS:")
    
    # Group by parameter and show trends
    chunk_performance = {}
    for result in results:
        chunk = result['chunk_size']
        if chunk not in chunk_performance:
            chunk_performance[chunk] = []
        chunk_performance[chunk].append(result['pixels_per_second'])
    
    print("   📐 Chunk Size Performance:")
    for chunk in sorted(chunk_performance.keys()):
        avg_speed = np.mean(chunk_performance[chunk])
        print(f"      • Chunk {chunk}: {avg_speed:,.0f} pixels/s average")

def main():
    """Main benchmark execution"""
    print("🎯 CUDA Flood Fill Parameter Optimization Benchmark")
    print("=" * 60)
    print(f"⏰ Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        results = run_comprehensive_benchmark()
        analyze_and_save_results(results)
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print(f"\n⏰ Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎯 Benchmark completed successfully!")
    return 0

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
