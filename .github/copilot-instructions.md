This is my GPU limitations:

----- BASIC DEVICE INFORMATION -----
Device Name: b'NVIDIA GeForce RTX 4060 Laptop GPU'
Compute Capability: 8.9

----- MEMORY INFORMATION -----
Total Global Memory: 8585216000 bytes (8187.50 MB, 8.00 GB)
Free Global Memory: 7443841024 bytes (7099.00 MB, 6.93 GB)
Used Global Memory: 1141374976 bytes (1088.50 MB, 1.06 GB)

----- THREAD AND BLOCK LIMITS -----
Max Threads Per Block: 1024
Max Block Dimensions: (1024, 1024, 64)
Max Grid Dimensions: (2147483647, 65535, 65535)
Warp Size: 32
Max Shared Memory per Block: 49152 bytes (48.00 KB)

----- HARDWARE DETAILS -----
Max Registers Per Block: 65536
Max Registers Per Multiprocessor: 65536
Number of Multiprocessors: 24
Clock Rate: 1470000 KHz (1470.00 MHz)
Memory Clock Rate: 8001000 KHz (8001.00 MHz)
Global Memory Bus Width: 128 bits
Concurrent Kernels Supported: True
Max Shared Memory per Multiprocessor: 102400 bytes (100.00 KB)
Total Constant Memory: 65536 bytes (64.00 KB)
L2 Cache Size: 33554432 bytes (32768.00 KB)
Max Shared Memory Per Block (Opt-in): 101376 bytes (99.00 KB)
Local Memory Per Thread: Not available

----- SUMMARY OF KEY MEMORY LIMITS -----
Global Memory: 8.00 GB
Shared Memory Per Block: 48.00 KB
Shared Memory Per MP: 100.00 KB
Constant Memory: 64.00 KB
L2 Cache: 32768.00 KB
Threads Per Block: 1024

----- OPTIMIZATION RECOMMENDATIONS FOR RTX 4060 -----
• Optimal thread block sizes: 256, 512, or 1024 threads
• Use up to 48 KB shared memory per block
• Consider 24 multiprocessors for grid sizing
• Memory bandwidth: 128-bit bus at 8001 MHz
• RTX 4060 8GB: Ada Lovelace architecture, compute capability 8.9
• Enhanced RT cores and Tensor cores available
• Max blocks per MP (shared memory limited): 2