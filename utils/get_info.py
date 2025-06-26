from numba import cuda

def format_memory_size(size_bytes):
    """Format memory size in bytes to appropriate unit (B, KB, MB, GB, TB)"""
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    size = float(size_bytes)
    unit_index = 0
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    # If the result has more than 3 digits, move to next unit if possible
    if size >= 1000 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    # Format with appropriate decimal places
    if size >= 100:
        return f"{size:.0f} {units[unit_index]}"
    elif size >= 10:
        return f"{size:.1f} {units[unit_index]}"
    else:
        return f"{size:.2f} {units[unit_index]}"

def print_gpu_info():
    """Print comprehensive GPU information for CUDA development"""
    
    # Check if CUDA is available
    if not cuda.is_available():
        print("CUDA is not available on this system")
        return
    
    # Get the current device
    device: cuda.cudadrv.device.Device

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
    
    print(f"Total Global Memory: {total_memory} bytes ({format_memory_size(total_memory)})")
    print(f"Free Global Memory: {free_memory} bytes ({format_memory_size(free_memory)})")
    print(f"Used Global Memory: {used_memory} bytes ({format_memory_size(used_memory)})")
    print()
    
    print("----- MULTIPROCESSOR ATTRIBUTES -----")
    max_threads_per_mp = getattr(device, 'MAX_THREADS_PER_MULTI_PROCESSOR', 2048)
    print(f"Number of Multiprocessors: {device.MULTIPROCESSOR_COUNT}")
    print(f"Max Threads Per Multiprocessor: {max_threads_per_mp}")
    print(f"Max Shared Memory per Multiprocessor: {device.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR} bytes ({format_memory_size(device.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)})")
    print(f"Max Registers Per Multiprocessor: {device.MAX_REGISTERS_PER_MULTIPROCESSOR}")
    print()
    
    print("----- BLOCK ATTRIBUTES -----")
    print(f"Max Threads Per Block: {device.MAX_THREADS_PER_BLOCK}")
    print(f"Max Block Dimensions: {device.MAX_BLOCK_DIM_X, device.MAX_BLOCK_DIM_Y, device.MAX_BLOCK_DIM_Z}")
    print(f"Max Shared Memory per Block: {device.MAX_SHARED_MEMORY_PER_BLOCK} bytes ({format_memory_size(device.MAX_SHARED_MEMORY_PER_BLOCK)})")
    print(f"Max Registers Per Block: {device.MAX_REGISTERS_PER_BLOCK}")
    # These might not be available on all devices
    try:
        print(f"Max Shared Memory Per Block (Opt-in): {device.MAX_SHARED_MEMORY_PER_BLOCK_OPTIN} bytes ({format_memory_size(device.MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)})")
    except AttributeError:
        print("Max Shared Memory Per Block (Opt-in): Not available")
    print()
    
    print("----- THREAD ATTRIBUTES -----")
    print(f"Warp Size: {device.WARP_SIZE}")
    print(f"Max Threads Per Block: {device.MAX_THREADS_PER_BLOCK}")
    print(f"Max Threads Per Multiprocessor: {max_threads_per_mp}")
    print()
    
    print("----- GRID ATTRIBUTES -----")
    print(f"Max Grid Dimensions: {device.MAX_GRID_DIM_X, device.MAX_GRID_DIM_Y, device.MAX_GRID_DIM_Z}")
    print()
    
    print("----- MEMORY HARDWARE ATTRIBUTES -----")
    print(f"Total Constant Memory: {device.TOTAL_CONSTANT_MEMORY} bytes ({format_memory_size(device.TOTAL_CONSTANT_MEMORY)})")
    print(f"L2 Cache Size: {device.L2_CACHE_SIZE} bytes ({format_memory_size(device.L2_CACHE_SIZE)})")
    print(f"Global Memory Bus Width: {device.GLOBAL_MEMORY_BUS_WIDTH} bits")
    print(f"Memory Clock Rate: {device.MEMORY_CLOCK_RATE} KHz ({device.MEMORY_CLOCK_RATE/1000:.2f} MHz)")
    print()
    
    print("----- PROCESSOR HARDWARE ATTRIBUTES -----")
    print(f"Clock Rate: {device.CLOCK_RATE} KHz ({device.CLOCK_RATE/1000:.2f} MHz)")
    print(f"Concurrent Kernels Supported: {bool(device.CONCURRENT_KERNELS)}")
    print()
    
    print("----- SUMMARY OF KEY MEMORY LIMITS -----")
    print(f"Global Memory: {format_memory_size(total_memory)}")
    print(f"Shared Memory Per Block: {format_memory_size(device.MAX_SHARED_MEMORY_PER_BLOCK)}")
    print(f"Shared Memory Per MP: {format_memory_size(device.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)}")
    print(f"Constant Memory: {format_memory_size(device.TOTAL_CONSTANT_MEMORY)}")
    print(f"L2 Cache: {format_memory_size(device.L2_CACHE_SIZE)}")
    print(f"Threads Per Block: {device.MAX_THREADS_PER_BLOCK}")
    print()
    
    print("----- OPTIMIZATION RECOMMENDATIONS FOR RTX 4060 -----")
    print("• Optimal thread block sizes: 256, 512, or 1024 threads")
    print(f"• Use up to {format_memory_size(device.MAX_SHARED_MEMORY_PER_BLOCK)} shared memory per block")
    print(f"• Consider {device.MULTIPROCESSOR_COUNT} multiprocessors for grid sizing")
    print(f"• Memory bandwidth: {device.GLOBAL_MEMORY_BUS_WIDTH}-bit bus at {device.MEMORY_CLOCK_RATE/1000:.0f} MHz")
    print("• RTX 4060 8GB: Ada Lovelace architecture, compute capability 8.9")
    print("• Enhanced RT cores and Tensor cores available")
    
    # Calculate some useful derived values
    max_blocks_per_mp_if_max_shared = device.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR // device.MAX_SHARED_MEMORY_PER_BLOCK
    print(f"• Max blocks per MP (if using max shared memory): {max_blocks_per_mp_if_max_shared}")
    
    # Show examples with different shared memory usage
    print("• Blocks per MP with different shared memory and thread block sizes:")
    
    print(f"  Max threads per MP: {max_threads_per_mp}")
    print(f"  Max shared memory per MP: {device.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR} bytes ({format_memory_size(device.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)})")
    print()
    
    # Show combinations of thread block sizes and shared memory usage
    thread_block_sizes = [256, 512, 1024]
    shared_memory_kb = [0, 1, 4, 8, 16, 32, 48]
    
    for threads_per_block in thread_block_sizes:
        print(f"  With {threads_per_block} threads per block:")
        max_blocks_by_threads = max_threads_per_mp // threads_per_block
        
        for shared_kb in shared_memory_kb:
            shared_bytes = shared_kb * 1024
            if shared_bytes <= device.MAX_SHARED_MEMORY_PER_BLOCK:
                if shared_bytes == 0:
                    blocks_per_mp = min(max_blocks_by_threads, 32)  # 32 is hardware limit
                    limit_reason = f"threads ({max_blocks_by_threads})" if max_blocks_by_threads < 32 else "hardware (32)"
                    print(f"    - {shared_kb:2d} KB shared memory: {blocks_per_mp:2d} blocks/MP (limited by {limit_reason})")
                else:
                    blocks_by_shared_mem = device.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR // shared_bytes
                    blocks_per_mp = min(blocks_by_shared_mem, max_blocks_by_threads, 32)
                    
                    # Determine limiting factor
                    if blocks_per_mp == max_blocks_by_threads:
                        limit_reason = "threads"
                    elif blocks_per_mp == blocks_by_shared_mem:
                        limit_reason = "shared memory"
                    else:
                        limit_reason = "hardware limit"
                    
                    print(f"    - {shared_kb:2d} KB shared memory: {blocks_per_mp:2d} blocks/MP (limited by {limit_reason})")
            else:
                break
        print()

if __name__ == "__main__":
    print_gpu_info()