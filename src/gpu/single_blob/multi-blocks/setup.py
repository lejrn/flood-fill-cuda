"""
Scene setup functions for flood fill testing.
Standalone script for generating test scenes.
"""

import numpy as np
from PIL import Image
import random
import os
import time
from datetime import datetime
from typing import Tuple, List

# Import from same directory (not relative imports)  
from utils import (RTX_4060_CONFIG, log_timing, format_memory_size, PerformanceLogger)

# Initialize performance logger
perf_logger = PerformanceLogger()


def setup_large_scene(width=8000, height=8000):
    """
    Create large test scene with a single large blob for performance testing.
    
    Args:
        width: Image width in pixels (default 8000)
        height: Image height in pixels (default 8000)
        
    Returns:
        Tuple containing scene data for the large blob scene
    """
    print(f"🎨 Setting up large scene ({width}x{height})...")
    print("   ⚠️  Large scene - this will use significant memory...")
    start_time = time.time()
    
    # Image initialization with white background
    print("   📄 Creating white background...")
    img = np.full((width, height, 3), 255, dtype=np.uint8)
    
    # Create a large red blob in the center (4000x4000)
    blob_size = 4000
    start_x = width // 2 - blob_size // 2  # 2000
    start_y = height // 2 - blob_size // 2  # 2000
    end_x = start_x + blob_size  # 6000
    end_y = start_y + blob_size  # 6000
    
    print(f"   🔴 Creating large red blob: {blob_size}x{blob_size} pixels")
    print(f"      Position: ({start_x},{start_y}) to ({end_x},{end_y})")
    print(f"      Total red pixels: {blob_size * blob_size:,}")
    img[start_x:end_x, start_y:end_y] = [255, 0, 0]
    
    # Create visited array
    print("   📋 Initializing visited array...")
    visited = np.zeros((width, height), dtype=np.int32)
    
    # New fill color (blue)
    new_color = np.array([0, 0, 255], dtype=np.uint8)
    print("   🔵 Fill color set to blue (0, 0, 255)")
    
    # Get optimal GPU configuration
    threads_per_block = RTX_4060_CONFIG['threads_per_block']
    blocks_per_grid = RTX_4060_CONFIG['blocks_per_grid']
    
    # Log memory usage
    img_memory = img.nbytes
    visited_memory = visited.nbytes
    total_memory = img_memory + visited_memory
    
    print(f"   💾 Memory allocation:")
    print(f"      • Image: {format_memory_size(img_memory)}")
    print(f"      • Visited: {format_memory_size(visited_memory)}")
    print(f"      • Total: {format_memory_size(total_memory)}")
    
    setup_time = log_timing(start_time, time.time(), "Scene setup")
    
    print(f"   ✅ Large scene ready! ({setup_time:.2f} ms)")
    
    return (img, visited, start_x, start_y, width, height, 
            new_color, threads_per_block, blocks_per_grid)
if __name__ == '__main__':
    """Test the setup functions when run directly"""
    print("🧪 Testing setup functions...")
    print()
    
    # Test large scene  
    print("Testing large scene...")
    large_data = setup_large_scene(width=800, height=600)
    print()
    
    print("✅ Setup function tested successfully!")
    
    # Log performance summary
    perf_logger.log_counters()
