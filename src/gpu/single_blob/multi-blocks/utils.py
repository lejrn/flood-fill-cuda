import numpy as np
from numba import cuda
import time

# Configuration constants
IMAGE_SIZE = 8000 * 8000  # For 8000x8000 large scene
QUEUE_CAPACITY = 16000000  # 16M queue capacity for large blobs (adjust based on available memory)
TOTAL_WARPS = 40 * 2  # 40 blocks × 2 warps per block = 80 warps

# Place direction arrays in constant memory (read-only) - 8 directions
DX_host = np.array([1, 1, 0, -1, -1, -1,  0, 1], dtype=np.int32)
DY_host = np.array([0, 1, 1,  1,  0, -1, -1, -1], dtype=np.int32)

# DX_host = np.array([1,-1, 0,  0, 1, -1, -1, 1], dtype=np.int32)
# DY_host = np.array([0, 0, 1, -1, 1, -1,  1, -1], dtype=np.int32)


# Simplified RTX 4060 Configuration:
RTX_4060_CONFIG = {
    'threads_per_block': 512,   # 2 warps per block (64 threads = 2 × 32)
    'blocks_per_grid': 96,     # 2 blocks per SM × 24 SMs
    'num_sms': 24,             # Using 24 out of 24 available SMs
    'blocks_per_sm': 4,        # 4 blocks per SM
    'warps_per_block': 16,      # 16 warps per block
    'total_warps': 640,          # 96 blocks × 16 warps = 640 warps
    'total_threads': 49152,     # 96 blocks × 512 threads = 49,152 threads
    'shared_memory_per_block': 48 * 1024,  # 48KB
    'max_threads_per_sm': 1536,
    'max_threads_per_block': 1024,  # Hardware limit for RTX 4060
    'max_shared_memory_per_sm': 100 * 1024  # 100KB
}

@cuda.jit(device=True, inline=True)
def is_red(img, x, y):
    """Check if pixel is red (255, 0, 0)"""
    return img[x, y, 0] == 255 and img[x, y, 1] == 0 and img[x, y, 2] == 0

@cuda.jit(device=True, inline=True)
def is_valid_pixel(x, y, width, height):
    """Check if pixel coordinates are within image bounds"""
    return 0 <= x < width and 0 <= y < height

def format_memory_size(bytes_size):
    """Format memory size in human readable format"""
    if bytes_size < 1024:
        return f"{bytes_size} B"
    elif bytes_size < 1024 * 1024:
        return f"{bytes_size / 1024:.2f} KB"
    elif bytes_size < 1024 * 1024 * 1024:
        return f"{bytes_size / (1024 * 1024):.2f} MB"
    else:
        return f"{bytes_size / (1024 * 1024 * 1024):.2f} GB"

def log_gpu_memory_info():
    """Log current GPU memory usage"""
    try:
        meminfo = cuda.current_context().get_memory_info()
        free_bytes, total_bytes = meminfo
        used_bytes = total_bytes - free_bytes
        
        print(f"    📊 GPU Memory Status:")
        print(f"       • Total: {format_memory_size(total_bytes)}")
        print(f"       • Used:  {format_memory_size(used_bytes)} ({used_bytes/total_bytes*100:.1f}%)")
        print(f"       • Free:  {format_memory_size(free_bytes)} ({free_bytes/total_bytes*100:.1f}%)")
    except Exception as e:
        print(f"    ⚠️  Could not get GPU memory info: {e}")

def log_configuration_details():
    """Log detailed RTX 4060 configuration"""
    print("    🔧 RTX 4060 Hardware Configuration:")
    print(f"       • Streaming Multiprocessors: {RTX_4060_CONFIG['num_sms']}")
    print(f"       • Blocks per SM: {RTX_4060_CONFIG['blocks_per_sm']}")
    print(f"       • Total Blocks: {RTX_4060_CONFIG['blocks_per_grid']}")
    print(f"       • Threads per Block: {RTX_4060_CONFIG['threads_per_block']}")
    print(f"       • Warps per Block: {RTX_4060_CONFIG['warps_per_block']}")
    print(f"       • Total Warps: {RTX_4060_CONFIG['total_warps']:,}")
    print(f"       • Max Threads per SM: {RTX_4060_CONFIG['max_threads_per_sm']}")
    print(f"       • Shared Memory per Block: {format_memory_size(RTX_4060_CONFIG['shared_memory_per_block'])}")
    print(f"       • Max Shared Memory per SM: {format_memory_size(RTX_4060_CONFIG['max_shared_memory_per_sm'])}")

def calculate_occupancy(threads_per_block, blocks_per_sm):
    """Calculate theoretical occupancy"""
    threads_per_sm = threads_per_block * blocks_per_sm
    max_threads_per_sm = RTX_4060_CONFIG['max_threads_per_sm']
    occupancy = threads_per_sm / max_threads_per_sm
    
    print(f"    📈 Theoretical Occupancy Analysis:")
    print(f"       • Threads per SM: {threads_per_sm}")
    print(f"       • Max Threads per SM: {max_threads_per_sm}")
    print(f"       • Occupancy: {occupancy:.1%}")
    
    if occupancy >= 0.75:
        print(f"       ✅ Excellent occupancy!")
    elif occupancy >= 0.5:
        print(f"       ⚡ Good occupancy")
    else:
        print(f"       ⚠️  Low occupancy - consider optimization")

def log_timing(start_time, end_time, operation_name):
    """Log timing information with detailed breakdown"""
    duration_ms = (end_time - start_time) * 1000
    
    if duration_ms < 1:
        print(f"    ⏱️  {operation_name}: {duration_ms:.3f} ms")
    elif duration_ms < 1000:
        print(f"    ⏱️  {operation_name}: {duration_ms:.2f} ms")
    else:
        print(f"    ⏱️  {operation_name}: {duration_ms/1000:.2f} seconds")
    
    return duration_ms

def log_queue_info(queue_capacity, image_size):
    """Log queue configuration details"""
    queue_memory = queue_capacity * 8  # 2 int32 arrays
    queue_percentage = (queue_capacity / image_size) * 100
    
    print(f"    🔄 Global Queue Configuration:")
    print(f"       • Capacity: {queue_capacity:,} pixels")
    print(f"       • Memory: {format_memory_size(queue_memory)} (2 x int32 arrays)")
    print(f"       • Coverage: {queue_percentage:.1f}% of image pixels")
    print(f"       • Access Pattern: L2 cached, coalesced reads")

def log_work_distribution(total_warps, queue_size):
    """Log work distribution across warps"""
    if queue_size > 0:
        work_per_warp = max(1, (queue_size + total_warps - 1) // total_warps)
        total_work_items = total_warps * work_per_warp
        
        print(f"    🎯 Work Distribution:")
        print(f"       • Queue Size: {queue_size:,} pixels")
        print(f"       • Work per Warp: {work_per_warp} pixels")
        print(f"       • Total Work Items: {total_work_items:,}")
        print(f"       • Parallel Efficiency: {(queue_size/total_work_items)*100:.1f}%")

class PerformanceLogger:
    """Class to track and log performance metrics"""
    
    def __init__(self):
        self.timings = {}
        self.counters = {}
    
    def start_timer(self, name):
        """Start timing an operation"""
        self.timings[name] = time.time()
        print(f"    ⏳ Starting: {name}")
    
    def end_timer(self, name):
        """End timing an operation and log results"""
        if name in self.timings:
            duration = log_timing(self.timings[name], time.time(), name)
            return duration
        else:
            print(f"    ❌ Timer '{name}' was not started")
            return 0
    
    def increment_counter(self, name, value=1):
        """Increment a performance counter"""
        if name not in self.counters:
            self.counters[name] = 0
        self.counters[name] += value
    
    def log_counters(self):
        """Log all performance counters"""
        if self.counters:
            print(f"    📊 Performance Counters:")
            for name, value in self.counters.items():
                print(f"       • {name}: {value:,}")

# Global performance logger instance
perf_logger = PerformanceLogger()
