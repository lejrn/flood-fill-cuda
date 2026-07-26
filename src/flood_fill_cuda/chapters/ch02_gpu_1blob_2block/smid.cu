// %smid reader for Numba CUDA-C linking.
// Numba's declare_device C ABI: int fn(RetType* ret, args...)
extern "C" __device__ int get_smid(unsigned int* ret) {
    unsigned int r;
    asm("mov.u32 %0, %%smid;" : "=r"(r));
    *ret = r;
    return 0;
}
