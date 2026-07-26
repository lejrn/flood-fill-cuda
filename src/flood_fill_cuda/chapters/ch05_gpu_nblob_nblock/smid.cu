// %smid reader for Numba CUDA-C linking.
// Numba's declare_device C ABI: int fn(RetType* ret, args...)
extern "C" __device__ int get_smid(unsigned int* ret) {
    unsigned int r;
    asm("mov.u32 %0, %%smid;" : "=r"(r));
    *ret = r;
    return 0;
}

// %globaltimer: device-wide wall clock in NANOSECONDS — comparable across
// SMs, so tid-0 stamps at grid.sync boundaries give per-phase wall times.
extern "C" __device__ int get_globaltimer(unsigned long long* ret) {
    unsigned long long r;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(r));
    *ret = r;
    return 0;
}

// %clock64: per-SM CYCLE counter. Deltas are only meaningful within one
// thread (a block never migrates SMs), which is exactly how the in-flight
// union timing uses it: each thread accumulates its own union cycles.
extern "C" __device__ int get_clock64(unsigned long long* ret) {
    unsigned long long r;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(r));
    *ret = r;
    return 0;
}
