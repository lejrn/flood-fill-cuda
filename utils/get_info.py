from numba import cuda
import numpy as np

def print_gpu_info():
    """Print comprehensive GPU information for CUDA development"""
    
    # Check if CUDA is available
    if not cuda.is_available():
        print("CUDA is not available on this system")
        return
    
    # Get the current device
    device = cuda.get_current_device()
    
    print("----- BASIC DEVICE INFORMATION -----")
    print(f"Device Name: {device.name}")
    print(f"Compute Capability: {device.compute_capability[0]}.{device.compute_capability[1]}")
    print()
    
    print("----- MEMORY INFORMATION -----")
    # Get memory info
    meminfo = cuda.current_context().get_memory_info()
    free_memory = meminfo.free
    total_memory = meminfo.total
    used_memory = total_memory - free_memory
    
    print(f"Total Global Memory: {total_memory} bytes ({total_memory/1024/1024:.2f} MB, {total_memory/1024/1024/1024:.2f} GB)")
    print(f"Free Global Memory: {free_memory} bytes ({free_memory/1024/1024:.2f} MB, {free_memory/1024/1024/1024:.2f} GB)")
    print(f"Used Global Memory: {used_memory} bytes ({used_memory/1024/1024:.2f} MB, {used_memory/1024/1024/1024:.2f} GB)")
    print()
    
    print("----- THREAD AND BLOCK LIMITS -----")
    print(f"Max Threads Per Block: {device.MAX_THREADS_PER_BLOCK}")
    print(f"Max Block Dimensions: {device.MAX_BLOCK_DIM_X, device.MAX_BLOCK_DIM_Y, device.MAX_BLOCK_DIM_Z}")
    print(f"Max Grid Dimensions: {device.MAX_GRID_DIM_X, device.MAX_GRID_DIM_Y, device.MAX_GRID_DIM_Z}")
    print(f"Warp Size: {device.WARP_SIZE}")
    print(f"Max Shared Memory per Block: {device.MAX_SHARED_MEMORY_PER_BLOCK} bytes ({device.MAX_SHARED_MEMORY_PER_BLOCK/1024:.2f} KB)")
    print()
    
    print("----- HARDWARE DETAILS -----")
    print(f"Max Registers Per Block: {device.MAX_REGISTERS_PER_BLOCK}")
    print(f"Max Registers Per Multiprocessor: {device.MAX_REGISTERS_PER_MULTIPROCESSOR}")
    print(f"Number of Multiprocessors: {device.MULTIPROCESSOR_COUNT}")
    print(f"Clock Rate: {device.CLOCK_RATE} KHz ({device.CLOCK_RATE/1000:.2f} MHz)")
    print(f"Memory Clock Rate: {device.MEMORY_CLOCK_RATE} KHz ({device.MEMORY_CLOCK_RATE/1000:.2f} MHz)")
    print(f"Global Memory Bus Width: {device.GLOBAL_MEMORY_BUS_WIDTH} bits")
    print(f"Concurrent Kernels Supported: {bool(device.CONCURRENT_KERNELS)}")
    print(f"Max Shared Memory per Multiprocessor: {device.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR} bytes ({device.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR/1024:.2f} KB)")
    print(f"Total Constant Memory: {device.TOTAL_CONSTANT_MEMORY} bytes ({device.TOTAL_CONSTANT_MEMORY/1024:.2f} KB)")
    print(f"L2 Cache Size: {device.L2_CACHE_SIZE} bytes ({device.L2_CACHE_SIZE/1024:.2f} KB)")
    
    # These might not be available on all devices
    try:
        print(f"Max Shared Memory Per Block (Opt-in): {device.MAX_SHARED_MEMORY_PER_BLOCK_OPTIN} bytes ({device.MAX_SHARED_MEMORY_PER_BLOCK_OPTIN/1024:.2f} KB)")
    except AttributeError:
        print("Max Shared Memory Per Block (Opt-in): Not available")
    
    try:
        print(f"Local Memory Per Thread: {device.LOCAL_MEMORY_PER_THREAD} bytes ({device.LOCAL_MEMORY_PER_THREAD/1024:.2f} KB)")
    except AttributeError:
        print("Local Memory Per Thread: Not available")
    
    print()
    
    print("----- SUMMARY OF KEY MEMORY LIMITS -----")
    print(f"Global Memory: {total_memory/1024/1024/1024:.2f} GB")
    print(f"Shared Memory Per Block: {device.MAX_SHARED_MEMORY_PER_BLOCK/1024:.2f} KB")
    print(f"Shared Memory Per MP: {device.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR/1024:.2f} KB")
    print(f"Constant Memory: {device.TOTAL_CONSTANT_MEMORY/1024:.2f} KB")
    print(f"L2 Cache: {device.L2_CACHE_SIZE/1024:.2f} KB")
    print(f"Threads Per Block: {device.MAX_THREADS_PER_BLOCK}")
    print()
    
    print("----- OPTIMIZATION RECOMMENDATIONS FOR RTX 4060 -----")
    print(f"• Optimal thread block sizes: 256, 512, or 1024 threads")
    print(f"• Use up to {device.MAX_SHARED_MEMORY_PER_BLOCK/1024:.0f} KB shared memory per block")
    print(f"• Consider {device.MULTIPROCESSOR_COUNT} multiprocessors for grid sizing")
    print(f"• Memory bandwidth: {device.GLOBAL_MEMORY_BUS_WIDTH}-bit bus at {device.MEMORY_CLOCK_RATE/1000:.0f} MHz")
    print(f"• RTX 4060 8GB: Ada Lovelace architecture, compute capability 8.9")
    print(f"• Enhanced RT cores and Tensor cores available")
    
    # Calculate some useful derived values
    max_blocks_per_mp = device.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR // device.MAX_SHARED_MEMORY_PER_BLOCK
    print(f"• Max blocks per MP (shared memory limited): {max_blocks_per_mp}")
    
    # For your flood fill algorithm
    print()
    print("----- FLOOD FILL ALGORITHM RECOMMENDATIONS FOR RTX 4060 -----")
    # Queue capacity calculation for your BFS implementation
    queue_element_size = 8  # 2 int32 values (x, y coordinates)
    available_shared_mem = device.MAX_SHARED_MEMORY_PER_BLOCK - 1024  # Reserve some space
    max_queue_capacity = available_shared_mem // queue_element_size
    print(f"• BFS Queue capacity (shared memory): ~{max_queue_capacity} elements")
    print(f"• Recommended patch sizes for 8GB VRAM: 512x512, 1024x1024, 2048x2048")
    print(f"• Use atomic operations for visited array with {device.MAX_THREADS_PER_BLOCK} threads")
    print(f"• Consider larger batch sizes due to 8GB memory capacity")

if __name__ == "__main__":
    print_gpu_info()