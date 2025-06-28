"""
GPU Configuration utilities for CUDA flood fill.
Focused on grid, SM, block, and thread configuration for optimal GPU utilization.
"""

import numpy as np
from numba import cuda

# Import from same directory
from utils import RTX_4060_CONFIG, format_memory_size, PerformanceLogger

# Initialize performance logger
perf_logger = PerformanceLogger()


def calculate_optimal_grid_config(width, height, target_utilization=0.75):
    """
    Calculate optimal grid configuration for the given image dimensions.
    
    Args:
        width, height: Image dimensions
        target_utilization: Target GPU utilization (0.0 to 1.0)
        
    Returns:
        Dictionary with optimal configuration
    """
    print(f"� Calculating Optimal Grid Configuration for {width}x{height} image:")
    
    # RTX 4060 constraints
    max_sms = RTX_4060_CONFIG['num_sms']  # 24
    max_threads_per_sm = RTX_4060_CONFIG['max_threads_per_sm']  # 1536
    max_threads_per_block = RTX_4060_CONFIG['max_threads_per_block']  # 1024
    
    total_pixels = width * height
    
    # Calculate different configuration options
    configs = []
    
    # Option 1: Maximum utilization (24 blocks, 1024 threads each)
    config_max = {
        'blocks_per_grid': 24,
        'threads_per_block': 1024,
        'total_threads': 24 * 1024,
        'sms_used': 24,
        'occupancy': 1.0,
        'name': 'Maximum Utilization'
    }
    configs.append(config_max)
    
    # Option 2: Balanced approach (144 blocks, 256 threads each)
    config_balanced = {
        'blocks_per_grid': 144,
        'threads_per_block': 256,
        'total_threads': 144 * 256,
        'sms_used': 24,
        'blocks_per_sm': 6,
        'occupancy': (6 * 256) / max_threads_per_sm,
        'name': 'Balanced High Parallelism'
    }
    configs.append(config_balanced)
    
    # Option 3: High block count (192 blocks, 128 threads each)
    config_high_blocks = {
        'blocks_per_grid': 192,
        'threads_per_block': 128,
        'total_threads': 192 * 128,
        'sms_used': 24,
        'blocks_per_sm': 8,
        'occupancy': (8 * 128) / max_threads_per_sm,
        'name': 'High Block Count'
    }
    configs.append(config_high_blocks)
    
    print("    📊 Configuration Options:")
    for config in configs:
        print(f"       • {config['name']}:")
        print(f"         - Blocks: {config['blocks_per_grid']}")
        print(f"         - Threads/Block: {config['threads_per_block']}")
        print(f"         - Total Threads: {config['total_threads']:,}")
        print(f"         - Occupancy: {config['occupancy']:.1%}")
        if 'blocks_per_sm' in config:
            print(f"         - Blocks/SM: {config['blocks_per_sm']}")
        print()
    
    # Select optimal configuration (currently using balanced approach)
    optimal = config_balanced
    print(f"    ✅ Selected: {optimal['name']}")
    
    return optimal


def analyze_memory_requirements(blocks_per_grid, threads_per_block, queue_capacity):
    """Analyze memory requirements for the given configuration"""
    print("💾 Memory Requirements Analysis:")
    
    total_threads = blocks_per_grid * threads_per_block
    total_warps = total_threads // 32
    
    # Global queue memory
    queue_memory = queue_capacity * 2 * 4 + 2 * 4  # 2 arrays + 2 pointers
    
    # Debug arrays memory
    debug_memory = (blocks_per_grid + total_threads + total_warps + 2) * 4
    
    # Image memory (example for 400x400 RGB image)
    image_memory = 400 * 400 * 3 * 1  # uint8
    visited_memory = 400 * 400 * 4    # int32
    
    total_memory = queue_memory + debug_memory + image_memory + visited_memory
    
    print(f"    📦 Global Queue: {format_memory_size(queue_memory)}")
    print(f"    🔍 Debug Arrays: {format_memory_size(debug_memory)}")
    print(f"    🎨 Image Data: {format_memory_size(image_memory + visited_memory)}")
    print(f"    💎 Total Memory: {format_memory_size(total_memory)}")
    
    # Check against RTX 4060 memory (8GB)
    rtx_memory = 8 * 1024 * 1024 * 1024
    utilization = (total_memory / rtx_memory) * 100
    
    print(f"    📊 Memory Utilization: {utilization:.2f}% of RTX 4060 (8GB)")
    
    if utilization < 1.0:
        print("    ✅ Memory usage is excellent")
    elif utilization < 5.0:
        print("    ⚡ Memory usage is good")
    else:
        print("    ⚠️  Consider optimizing memory usage")
    
    return {
        'queue_memory': queue_memory,
        'debug_memory': debug_memory,
        'image_memory': image_memory + visited_memory,
        'total_memory': total_memory,
        'utilization_percent': utilization
    }


def get_rtx_4060_config():
    """Get the optimal RTX 4060 configuration"""
    print("� RTX 4060 Optimal Configuration:")
    print(f"    • Streaming Multiprocessors: {RTX_4060_CONFIG['num_sms']}")
    print(f"    • Max Threads per SM: {RTX_4060_CONFIG['max_threads_per_sm']}")
    print(f"    • Max Threads per Block: {RTX_4060_CONFIG['max_threads_per_block']}")
    print(f"    • Warp Size: 32 threads")
    print(f"    • L2 Cache: 32MB (excellent for global queue)")
    print(f"    • Memory: 8GB GDDR6")
    
    return RTX_4060_CONFIG


if __name__ == '__main__':
    """Test GPU configuration functions when run directly"""
    print("🧪 Testing GPU configuration functions...")
    print()
    
    try:
        # Test optimal grid calculation
        config = calculate_optimal_grid_config(400, 400)
        print()
        
        # Test memory analysis
        memory_info = analyze_memory_requirements(
            config['blocks_per_grid'], 
            config['threads_per_block'], 
            1000000  # 1M queue capacity
        )
        print()
        
        # Test RTX 4060 config
        rtx_config = get_rtx_4060_config()
        
        print("\n✅ All GPU configuration functions tested successfully!")
        
    except Exception as e:
        print(f"❌ GPU configuration test failed: {e}")
        import traceback
        traceback.print_exc()
