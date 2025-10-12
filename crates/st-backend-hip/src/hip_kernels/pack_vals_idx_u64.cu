#include <hip/hip_runtime.h>
#include <stdint.h>

__device__ __forceinline__ uint64_t pack_f32_i32(float v, int32_t i){
    uint32_t fv = *((uint32_t*)&v);
    return ((uint64_t)fv << 32) | (uint64_t)((uint32_t)i);
}

extern "C" __global__
void pack_vals_idx_u64_kernel(const float* __restrict__ vals, const int32_t* __restrict__ idx, uint64_t* __restrict__ out, int total){
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < total){
        out[t] = pack_f32_i32(vals[t], idx[t]);
    }
}

extern "C"
hipError_t st_pack_vals_idx_u64(const float* vals, const int32_t* idx, uint64_t* out, int total, hipStream_t stream){
    int block=256; int grid=(total + block - 1)/block;
    hipLaunchKernelGGL(pack_vals_idx_u64_kernel, dim3(grid), dim3(block), 0, stream, vals, idx, out, total);
    return hipGetLastError();
}
