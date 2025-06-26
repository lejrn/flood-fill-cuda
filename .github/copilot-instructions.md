
# Terminal Python Copilot Instructions
to run scripts in the terminal, use the following command:
```bash
poetry run python <path/from/pwd/script_file.py>
```
where `<script_file>` is the name of the Python module you want to run.

No need to use `python3` or `python`, just use `poetry run python`.

No need to use `poetry run` if you are already in a poetry shell.

No need to check "cd to the directory" in the terminal, as the command will automatically run in the correct directory.


# Constraints for CUDA code generation

----- BASIC DEVICE INFORMATION -----
Device Name: b'NVIDIA GeForce RTX 4060 Laptop GPU'
Compute Capability: 8.9

----- MEMORY INFORMATION -----
Total Global Memory: 8585216000 bytes (8.00 GB)
Free Global Memory: 7443841024 bytes (6.93 GB)
Used Global Memory: 1141374976 bytes (1.06 GB)

----- MULTIPROCESSOR ATTRIBUTES -----
Number of Multiprocessors: 24
Max Threads Per Multiprocessor: 1536
Max Shared Memory per Multiprocessor: 102400 bytes (100 KB)
Max Registers Per Multiprocessor: 65536

----- BLOCK ATTRIBUTES -----
Max Threads Per Block: 1024
Max Block Dimensions: (1024, 1024, 64)
Max Shared Memory per Block: 49152 bytes (48.0 KB)
Max Registers Per Block: 65536
Max Shared Memory Per Block (Opt-in): 101376 bytes (99.0 KB)

----- THREAD ATTRIBUTES -----
Warp Size: 32
Max Threads Per Block: 1024
Max Threads Per Multiprocessor: 1536

----- GRID ATTRIBUTES -----
Max Grid Dimensions: (2147483647, 65535, 65535)

----- MEMORY HARDWARE ATTRIBUTES -----
Total Constant Memory: 65536 bytes (64.0 KB)
L2 Cache Size: 33554432 bytes (32.0 MB)
Global Memory Bus Width: 128 bits
Memory Clock Rate: 8001000 KHz (8001.00 MHz)

----- PROCESSOR HARDWARE ATTRIBUTES -----
Clock Rate: 1470000 KHz (1470.00 MHz)
Concurrent Kernels Supported: True

----- SUMMARY OF KEY MEMORY LIMITS -----
Global Memory: 8.00 GB
Shared Memory Per Block: 48.0 KB
Shared Memory Per MP: 100 KB
Constant Memory: 64.0 KB
L2 Cache: 32.0 MB
Threads Per Block: 1024

----- OPTIMIZATION RECOMMENDATIONS FOR RTX 4060 -----
• Optimal thread block sizes: 256, 512, or 1024 threads
• Use up to 48.0 KB shared memory per block
• Consider 24 multiprocessors for grid sizing
• Memory bandwidth: 128-bit bus at 8001 MHz
• RTX 4060 8GB: Ada Lovelace architecture, compute capability 8.9
• Enhanced RT cores and Tensor cores available
• Max blocks per MP (if using max shared memory): 2
• Blocks per MP with different shared memory and thread block sizes:
  Max threads per MP: 1536
  Max shared memory per MP: 102400 bytes (100 KB)

  With 256 threads per block:
    -  0 KB shared memory:  6 blocks/MP (limited by threads (6))
    -  1 KB shared memory:  6 blocks/MP (limited by threads)
    -  4 KB shared memory:  6 blocks/MP (limited by threads)
    -  8 KB shared memory:  6 blocks/MP (limited by threads)
    - 16 KB shared memory:  6 blocks/MP (limited by threads)
    - 32 KB shared memory:  3 blocks/MP (limited by shared memory)
    - 48 KB shared memory:  2 blocks/MP (limited by shared memory)

  With 512 threads per block:
    -  0 KB shared memory:  3 blocks/MP (limited by threads (3))
    -  1 KB shared memory:  3 blocks/MP (limited by threads)
    -  4 KB shared memory:  3 blocks/MP (limited by threads)
    -  8 KB shared memory:  3 blocks/MP (limited by threads)
    - 16 KB shared memory:  3 blocks/MP (limited by threads)
    - 32 KB shared memory:  3 blocks/MP (limited by threads)
    - 48 KB shared memory:  2 blocks/MP (limited by shared memory)

  With 1024 threads per block:
    -  0 KB shared memory:  1 blocks/MP (limited by threads (1))
    -  1 KB shared memory:  1 blocks/MP (limited by threads)
    -  4 KB shared memory:  1 blocks/MP (limited by threads)
    -  8 KB shared memory:  1 blocks/MP (limited by threads)
    - 16 KB shared memory:  1 blocks/MP (limited by threads)
    - 32 KB shared memory:  1 blocks/MP (limited by threads)
    - 48 KB shared memory:  1 blocks/MP (limited by threads)