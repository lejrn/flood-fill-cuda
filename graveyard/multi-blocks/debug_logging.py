"""
Logging utilities for CUDA kernel execution and performance analysis.

This module contains all logging and printing functions for kernel launches,
debug analysis, and performance metrics.
"""

import numpy as np
from utils import QUEUE_CAPACITY


def log_kernel_launch_info(kernel_name, blocks_per_grid, threads_per_block):
    """Log detailed information about kernel launch configuration"""
    total_threads = blocks_per_grid * threads_per_block
    total_warps = total_threads // 32
    
    print(f"🚀 Launching {kernel_name} Kernel:")
    print("    📋 Launch Configuration:")
    print(f"       • Blocks: {blocks_per_grid}")
    print(f"       • Threads per Block: {threads_per_block}")
    print(f"       • Total Threads: {total_threads:,}")
    print(f"       • Total Warps: {total_warps:,}")
    print(f"       • Grid Dimensions: ({blocks_per_grid}, 1, 1)")
    print(f"       • Block Dimensions: ({threads_per_block}, 1, 1)")


def create_global_queue_arrays():
    """Create and initialize global memory arrays for the queue"""
    print("🔧 Creating Global Memory Queue Arrays:")
    print("    💾 Allocating queue arrays:")
    print(f"       • Queue X: {QUEUE_CAPACITY:,} int32 elements")
    print(f"       • Queue Y: {QUEUE_CAPACITY:,} int32 elements") 
    print("       • Queue Pointers: 2 int32 elements")
    
    total_memory = QUEUE_CAPACITY * 8 + 8  # 2 arrays + 2 pointers
    print(f"       • Total Memory: {total_memory:,} bytes")
    
    try:
        from numba import cuda
        global_queue_x = cuda.device_array(QUEUE_CAPACITY, dtype=np.int32)
        global_queue_y = cuda.device_array(QUEUE_CAPACITY, dtype=np.int32)
        global_queue_front = cuda.device_array(1, dtype=np.int32)
        global_queue_rear = cuda.device_array(1, dtype=np.int32)
        
        print("    ✅ Global queue arrays created successfully")
        return global_queue_x, global_queue_y, global_queue_front, global_queue_rear
        
    except Exception as e:
        print(f"    ❌ Failed to create global queue arrays: {e}")
        raise


def analyze_debug_arrays(debug_block_usage, debug_thread_usage, debug_warp_usage, 
                        debug_pixel_count, debug_queue_usage, debug_iteration_count, blocks_per_grid, threads_per_block):
    """Analyze debug arrays to provide GPU utilization statistics"""
    print("\n📊 GPU Utilization Analysis:")
    
    # Block utilization
    blocks_used = np.sum(debug_block_usage)
    block_utilization = (blocks_used / blocks_per_grid) * 100
    print(f"    🧱 Block Utilization:")
    print(f"       • Blocks Used: {blocks_used}/{blocks_per_grid}")
    print(f"       • Utilization: {block_utilization:.1f}%")
    
    # Thread utilization
    total_threads = blocks_per_grid * threads_per_block
    threads_used = np.sum(debug_thread_usage)
    thread_utilization = (threads_used / total_threads) * 100
    print(f"    🧵 Thread Utilization:")
    print(f"       • Threads Used: {threads_used}/{total_threads}")
    print(f"       • Utilization: {thread_utilization:.1f}%")
    
    # Warp utilization
    total_warps = total_threads // 32
    warps_used = np.sum(debug_warp_usage)
    warp_utilization = (warps_used / total_warps) * 100
    print(f"    🌊 Warp Utilization:")
    print(f"       • Warps Used: {warps_used}/{total_warps}")
    print(f"       • Utilization: {warp_utilization:.1f}%")
    
    # Pixel processing
    pixels_processed = debug_pixel_count[0]
    print(f"    🎨 Pixel Processing:")
    print(f"       • Pixels Processed: {pixels_processed:,}")
    
    # Queue usage
    max_queue_usage = debug_queue_usage[0]
    queue_utilization = (max_queue_usage / QUEUE_CAPACITY) * 100
    print(f"    📦 Queue Usage:")
    print(f"       • Maximum Queue Size: {max_queue_usage:,}")
    print(f"       • Queue Utilization: {queue_utilization:.1f}%")
    
    # Iteration count
    iterations_executed = debug_iteration_count[0]
    print(f"    🔄 Algorithm Progress:")
    print(f"       • Iterations Executed: {iterations_executed:,}")
    print(f"       • Pixels per Iteration: {pixels_processed/max(iterations_executed, 1):.1f}")
    
    # Overall efficiency metrics
    print(f"    ⚡ Efficiency Metrics:")
    print(f"       • Pixels per Thread: {pixels_processed/threads_used:.1f}")
    print(f"       • Pixels per Warp: {pixels_processed/warps_used:.1f}")
    
    return {
        'blocks_used': blocks_used,
        'block_utilization': block_utilization,
        'threads_used': threads_used,
        'thread_utilization': thread_utilization,
        'warps_used': warps_used,
        'warp_utilization': warp_utilization,
        'pixels_processed': pixels_processed,
        'max_queue_usage': max_queue_usage,
        'queue_utilization': queue_utilization,
        'iterations_executed': iterations_executed
    }


def log_memory_allocation(array_name, size, dtype):
    """Log memory allocation for debug arrays"""
    memory_size = size * np.dtype(dtype).itemsize
    print(f"    📊 {array_name}: {size:,} elements ({memory_size:,} bytes)")


def log_debug_array_creation(blocks_per_grid, threads_per_block):
    """Log creation of debug arrays for GPU utilization tracking"""
    total_threads = blocks_per_grid * threads_per_block
    total_warps = total_threads // 32
    
    print("🔧 Creating Debug Arrays for GPU Utilization Tracking:")
    log_memory_allocation("Block Usage Array", blocks_per_grid, np.int32)
    log_memory_allocation("Thread Usage Array", total_threads, np.int32)
    log_memory_allocation("Warp Usage Array", total_warps, np.int32)
    log_memory_allocation("Pixel Count Array", 1, np.int32)
    log_memory_allocation("Queue Usage Array", 1, np.int32)
    
    total_debug_memory = (blocks_per_grid + total_threads + total_warps + 2) * 4
    print(f"    💾 Total Debug Memory: {total_debug_memory:,} bytes")
