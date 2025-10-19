#include <hip/hip_runtime.h>
#include <stdint.h>
#include <float.h>

__device__ __forceinline__ float unpack_f(uint64_t p){
    union { uint32_t u; float f; } cvt;
    cvt.u = static_cast<uint32_t>(p >> 32);
    return cvt.f;
}

__device__ __forceinline__ int32_t unpack_i(uint64_t p){
    return static_cast<int32_t>(p & 0xffffffffu);
}

__device__ __forceinline__ uint64_t pack_fi(float v, int32_t i){
    union { float f; uint32_t u; } cvt;
    cvt.f = v;
    return (static_cast<uint64_t>(cvt.u) << 32) | static_cast<uint32_t>(i);
}

extern "C" __global__
void hip_topk_tile_bitonic_u64_kernel(const uint64_t* __restrict__ cand,
                                      int rows,
                                      int total,
                                      int k_final,
                                      uint64_t* __restrict__ out)
{
    int r = blockIdx.x;
    if (r >= rows) return;
    extern __shared__ unsigned char smem[];
    float* vals = reinterpret_cast<float*>(smem);
    int32_t* idx = reinterpret_cast<int32_t*>(smem + total * sizeof(float));
    int tid = threadIdx.x;

    for (int i = tid; i < total; i += blockDim.x) {
        uint64_t packed = cand[r * total + i];
        vals[i] = unpack_f(packed);
        idx[i] = unpack_i(packed);
    }
    __syncthreads();

    int n = 1;
    while (n < total) n <<= 1;
    for (int i = tid + total; i < n; i += blockDim.x) {
        vals[i] = -INFINITY;
        idx[i] = -1;
    }
    __syncthreads();

    for (int k = n; k > 1; k >>= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < n; i += blockDim.x) {
                int ixj = i ^ j;
                if (ixj > i) {
                    float a = vals[i];
                    float b = vals[ixj];
                    int32_t ai = idx[i];
                    int32_t bi = idx[ixj];
                    if (a < b) {
                        vals[i] = b;
                        vals[ixj] = a;
                        idx[i] = bi;
                        idx[ixj] = ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (int j = tid; j < k_final; j += blockDim.x) {
        out[r * k_final + j] = pack_fi(vals[j], idx[j]);
    }
}

extern "C"
hipError_t st_topk_tile_bitonic_u64(const uint64_t* cand,
                                    int rows,
                                    int total,
                                    int k_final,
                                    uint64_t* out,
                                    hipStream_t stream)
{
    dim3 grid(rows);
    const int BLOCK = 256;
    dim3 block(BLOCK);
    size_t shared = static_cast<size_t>(total) * sizeof(float)
                  + static_cast<size_t>(total) * sizeof(int32_t);
    hipLaunchKernelGGL(hip_topk_tile_bitonic_u64_kernel, grid, block, shared, stream,
                       cand, rows, total, k_final, out);
    return hipGetLastError();
}
