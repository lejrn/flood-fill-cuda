import os
import sys
import subprocess

def run_cuda_diagnostics():
    print("=== RTX 4060 8GB Laptop GPU CUDA Diagnostic (WSL2) ===\n")
    
    # Auto-set CUDA environment for RTX 4060 8GB Laptop GPU (CUDA 12.9 - Latest)
    os.environ['CUDA_PATH'] = '/usr/local/cuda-12.9'
    os.environ['CUDA_HOME'] = '/usr/local/cuda-12.9'
    wsl_cuda_path = '/usr/lib/wsl/lib'
    cuda_lib_path = '/usr/local/cuda-12.9/targets/x86_64-linux/lib'
    system_lib_path = '/usr/lib/x86_64-linux-gnu'
    os.environ['LD_LIBRARY_PATH'] = f"{wsl_cuda_path}:{cuda_lib_path}:{system_lib_path}:" + os.environ.get('LD_LIBRARY_PATH', '')
    
    # 1. Environment check
    print("1. Environment Variables (RTX 4060 8GB Optimized):")
    cuda_vars = ['CUDA_PATH', 'CUDA_HOME', 'LD_LIBRARY_PATH', 'PATH']
    for var in cuda_vars:
        value = os.environ.get(var, 'Not set')
        if var == 'LD_LIBRARY_PATH' and len(value) > 100:
            # Truncate long LD_LIBRARY_PATH for readability
            value = value[:100] + "... (truncated)"
        print(f"   {var}: {value}")
    
    # 2. CUDA installations
    print("\n2. CUDA Installations:")
    try:
        result = subprocess.run(['ls', '-la', '/usr/local/'], capture_output=True, text=True)
        cuda_dirs = [line for line in result.stdout.split('\n') if 'cuda' in line.lower()]
        for dir_info in cuda_dirs:
            print(f"   {dir_info}")
    except Exception as e:
        print(f"   Error checking installations: {e}")
    
    # 3. RTX 4060 8GB Laptop GPU Driver Check
    print("\n3. RTX 4060 8GB Laptop GPU Driver (WSL2):")
    try:
        # Use WSL2-specific nvidia-smi
        result = subprocess.run(['/usr/lib/wsl/lib/nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✓ WSL2 nvidia-smi working")
            # Extract GPU info
            lines = result.stdout.split('\n')
            for line in lines:
                if 'RTX 4060' in line and '8GB' in line:
                    print(f"   ✓ RTX 4060 8GB Laptop GPU detected: {line.strip()}")
                    break
            else:
                print(f"   ⚠ RTX 4060 8GB not found in output")
        else:
            print(f"   ✗ WSL2 nvidia-smi failed: {result.stderr}")
            
        # Fallback to system nvidia-smi
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            gpu_info = result.stdout.strip()
            print(f"   System nvidia-smi: {gpu_info}")
        else:
            print(f"   System nvidia-smi failed")
            
    except Exception as e:
        print(f"   Error: {e}")
    
    # 4. nvcc version
    print("\n4. NVCC Compiler:")
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        version_line = [line for line in result.stdout.split('\n') if 'release' in line.lower()]
        print(f"   {version_line[0] if version_line else 'Version not found'}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 5. Python CUDA packages for RTX 4060 8GB Laptop GPU
    print("\n5. Python CUDA Packages (RTX 4060 8GB Laptop GPU):")
    try:
        import numba
        print(f"   Numba version: {numba.__version__}")
        
        from numba import cuda
        print(f"   CUDA available: {cuda.is_available()}")
        
        if cuda.is_available():
            device = cuda.get_current_device()
            device_name = device.name.decode() if hasattr(device.name, 'decode') else str(device.name)
            print(f"   ✓ RTX 4060 8GB Laptop GPU detected: {device_name}")
            print(f"   Compute capability: {device.compute_capability}")
            print(f"   Multiprocessors: {device.MULTIPROCESSOR_COUNT}")
            print(f"   Max threads/block: {device.MAX_THREADS_PER_BLOCK}")
            print(f"   Shared memory/block: {device.MAX_SHARED_MEMORY_PER_BLOCK/1024:.0f} KB")
            
            # Memory info for RTX 4060 8GB Laptop GPU
            meminfo = cuda.current_context().get_memory_info()
            total_gb = meminfo.total / (1024**3)
            free_gb = meminfo.free / (1024**3)
            print(f"   GPU Memory: {total_gb:.1f} GB total, {free_gb:.1f} GB free")
            
            # Verify 8GB specification
            if 7.0 <= total_gb <= 8.5:  # Account for system reservation
                print("   ✓ Memory confirms RTX 4060 8GB Laptop GPU")
            else:
                print(f"   ⚠ Unexpected memory size: {total_gb:.1f} GB")
                
            print(f"   CUDA devices: {len(cuda.gpus)}")
        else:
            print("   ✗ RTX 4060 8GB Laptop GPU not accessible")
            try:
                cuda.detect()
            except Exception as e:
                print(f"   CUDA detection error: {e}")
                
                # WSL2-specific troubleshooting for RTX 4060 8GB Laptop GPU
                print("\n   WSL2 RTX 4060 8GB Laptop GPU Troubleshooting:")
                print("   1. Restart WSL2: wsl --shutdown && wsl")
                print("   2. Update Windows NVIDIA driver (>=470.76 for WSL2)")
                print("   3. Check Windows NVIDIA Control Panel")
                print("   4. Verify Windows GPU is not in power saving mode")
                
    except ImportError as e:
        print(f"   Import error: {e}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 6. CUDA Library Check (RTX 4060 8GB Laptop GPU)
    print("\n6. CUDA Library Check (RTX 4060 8GB Laptop GPU):")
    cuda_libs = [
        '/usr/local/cuda-12.9/targets/x86_64-linux/lib/libcudart.so',
        '/usr/local/cuda-12.2/targets/x86_64-linux/lib/libcudart.so',
        '/usr/lib/x86_64-linux-gnu/libcudart.so'
    ]
    
    for lib in cuda_libs:
        if os.path.exists(lib):
            print(f"   ✓ Found: {lib}")
            try:
                result = subprocess.run(['ldd', lib], capture_output=True, text=True)
                if 'not found' in result.stdout:
                    print(f"     ⚠ Missing dependencies in {lib}")
            except:
                pass
        else:
            print(f"   ✗ Missing: {lib}")
    
    # 7. RTX 4060 8GB Laptop GPU Flood Fill Optimization Recommendations
    print("\n7. RTX 4060 8GB Laptop GPU Flood Fill Recommendations:")
    print("   • Optimal block sizes: 256-512 threads (Ada Lovelace architecture)")
    print("   • Max image dimensions: 4096x4096 pixels (fits in 8GB VRAM)")
    print("   • Shared memory queue capacity: ~6144 coordinates per block")
    print("   • Warp size: 32 threads (use for memory coalescing)")
    print("   • Memory bandwidth: ~288 GB/s (laptop variant)")
    print("   • RT Cores: 3rd gen for accelerated raytracing")
    print("   • Tensor Cores: 4th gen for AI workloads")
    print("   • Consider dynamic parallelism for complex flood fill patterns")

if __name__ == "__main__":
    run_cuda_diagnostics()