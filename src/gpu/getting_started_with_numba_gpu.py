# from numba import cuda
# print(hasattr(cuda, 'nvtx_range_push'))


# import numpy as np
# import time
# import timeit
# from numba import cuda
# import nvtx  # For NVTX annotations

# # Create a custom NVTX domain
# NVTX_DOMAIN = nvtx.Domain("MyApp")

# # Input data
# x = np.arange(100).reshape(10, 10).astype(np.float32)  # Ensure float32 for GPU

# # Transfer data to GPU
# with nvtx.annotate("Data Transfer to GPU", domain=NVTX_DOMAIN, color="orange"):
#     d_x = cuda.to_device(x)
#     d_result = cuda.device_array_like(x)  # Output array on GPU

# @cuda.jit
# def go_fast(a, result):
#     # Get thread index
#     i = cuda.grid(1)
#     if i < a.shape[0]:
#         # Compute trace (diagonal elements)
#         trace = 0.0
#         with nvtx.annotate("go_fast Loop", domain=NVTX_DOMAIN, color="blue"):
#             trace = cuda.atomic.add(result, (0, 0), np.tanh(a[i, i]))
#         # Add trace to all elements in the row
#         for j in range(a.shape[1]):
#             result[i, j] = a[i, j] + trace

# # Configure grid and block sizes
# threads_per_block = 32
# blocks_per_grid = (x.shape[0] + (threads_per_block - 1)) // threads_per_block

# # First run (with compilation)
# with nvtx.annotate("First Run (with Compilation)", domain=NVTX_DOMAIN, color="red"):
#     start = time.time()
#     go_fast[blocks_per_grid, threads_per_block](d_x, d_result)
#     cuda.synchronize()  # Ensure kernel completes
#     result = d_result.copy_to_host()  # Transfer result back
#     end = time.time()
#     print("Elapsed (with compilation) = %s" % (end - start))

# # Second run (from cache)
# with nvtx.annotate("Second Run (after Compilation)", domain=NVTX_DOMAIN, color="green"):
#     start = time.time()
#     go_fast[blocks_per_grid, threads_per_block](d_x, d_result)
#     cuda.synchronize()
#     result = d_result.copy_to_host()
#     end = time.time()
#     print("Elapsed (after compilation) = %s" % (end - start))

# # Timeit run
# def timeit_wrapper():
#     with nvtx.annotate("Timeit Iteration", domain=NVTX_DOMAIN, color="yellow"):
#         go_fast[blocks_per_grid, threads_per_block](d_x, d_result)
#         cuda.synchronize()

# with nvtx.annotate("Timeit 1000 Runs", domain=NVTX_DOMAIN, color="purple"):
#     print(timeit.timeit(timeit_wrapper, number=1000))

# import numpy as np
# import time
# import timeit
# from numba import cuda
# import nvtx  # Use the nvtx package for annotations

# # Input data
# x = np.arange(100).reshape(10, 10).astype(np.float32)  # Ensure float32 for GPU

# # Transfer data to GPU
# with nvtx.annotate("Data Transfer to GPU", color="orange"):
#     d_x = cuda.to_device(x)
#     d_result = cuda.device_array_like(x)  # Output array on GPU

# @cuda.jit
# def go_fast(a, result):
#     # Get thread index
#     i = cuda.grid(1)
#     if i < a.shape[0]:
#         # Compute trace (diagonal elements)
#         trace = 0.0
#         # Note: No NVTX inside the kernel
#         trace = cuda.atomic.add(result, (0, 0), np.tanh(a[i, i]))
#         # Add trace to all elements in the row
#         for j in range(a.shape[1]):
#             result[i, j] = a[i, j] + trace

# # Configure grid and block sizes
# threads_per_block = 32
# blocks_per_grid = (x.shape[0] + (threads_per_block - 1)) // threads_per_block

# # First run (with compilation)
# with nvtx.annotate("First Run (with Compilation)", color="red"):
#     start = time.time()
#     with nvtx.annotate("go_fast Kernel Launch", color="blue"):
#         go_fast[blocks_per_grid, threads_per_block](d_x, d_result)
#     cuda.synchronize()  # Ensure kernel completes
#     result = d_result.copy_to_host()  # Transfer result back
#     end = time.time()
#     print("Elapsed (with compilation) = %s" % (end - start))

# # Second run (from cache)
# with nvtx.annotate("Second Run (after Compilation)", color="green"):
#     start = time.time()
#     with nvtx.annotate("go_fast Kernel Launch", color="blue"):
#         go_fast[blocks_per_grid, threads_per_block](d_x, d_result)
#     cuda.synchronize()
#     result = d_result.copy_to_host()
#     end = time.time()
#     print("Elapsed (after compilation) = %s" % (end - start))

# # Timeit run
# def timeit_wrapper():
#     with nvtx.annotate("Timeit Iteration", color="yellow"):
#         with nvtx.annotate("go_fast Kernel Launch", color="blue"):
#             go_fast[blocks_per_grid, threads_per_block](d_x, d_result)
#         cuda.synchronize()

# with nvtx.annotate("Timeit 1000 Runs", color="purple"):
#     print(timeit.timeit(timeit_wrapper, number=1000))


import numpy as np
from numba import cuda
import nvtx  # For NVTX annotations

# Input data
N = 100000000  # 1M elements
a = np.ones(N, dtype=np.float32)
b = np.ones(N, dtype=np.float32)
c = np.zeros(N, dtype=np.float32)

# Transfer data to GPU
with nvtx.annotate("Data Transfer to GPU", color="orange"):
    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_c = cuda.device_array_like(c)

@cuda.jit
def vector_add(a, b, c):
    i = cuda.grid(1)
    if i < a.size:
        c[i] = a[i] + b[i]

# Configure grid and block sizes
threads_per_block = 256
blocks_per_grid = (N + (threads_per_block - 1)) // threads_per_block

# Run the kernel
with nvtx.annotate("Vector Add Kernel Launch", color="blue"):
    vector_add[blocks_per_grid, threads_per_block](d_a, d_b, d_c)
    cuda.synchronize()

# Transfer result back to host
with nvtx.annotate("Data Transfer to Host", color="green"):
    c = d_c.copy_to_host()

# Verify result
print("First few elements of result:", c[:10])