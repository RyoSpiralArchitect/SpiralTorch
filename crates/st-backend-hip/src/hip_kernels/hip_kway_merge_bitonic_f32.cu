#include <hip/hip_runtime.h>
#include <stdint.h>

extern "C" __global__
void hip_kway_merge_bitonic_f32_kernel(
    const float* __restrict__ cand_vals,  // [rows, total]
    const int32_t* __restrict__ cand_idx, // [rows, total]
    int rows, int total, int k_final,
    float* __restrict__ out_vals, int32_t* __restrict__ out_idx)
{
    int r = blockIdx.x;
    if (r>=rows) return;
    extern __shared__ unsigned char smem[];
    float* vals = (float*)smem;
    int32_t* idx = (int32_t*)(smem + total * sizeof(float));
    int tid = threadIdx.x;
    for (int i=tid; i<total; i+=blockDim.x){
        vals[i] = cand_vals[r*total + i];
        idx[i]  = cand_idx [r*total + i];
    }
    __syncthreads();
    int n = 1; while(n<total) n<<=1;
    for (int i=tid+total; i<n; i+=blockDim.x){ vals[i] = -INFINITY; idx[i] = -1; }
    __syncthreads();
    for (int k=n; k>1; k>>=1){
        for (int j=k>>1; j>0; j>>=1){
            for (int i=tid; i<n; i+=blockDim.x){
                int ixj = i ^ j;
                if (ixj > i){
                    float a = vals[i], b = vals[ixj];
                    int32_t ai = idx[i], bi = idx[ixj];
                    if (a < b){
                        vals[i] = b; vals[ixj] = a;
                        idx[i] = bi; idx[ixj] = ai;
                    }
                }
            }
            __syncthreads();
        }
    }
    for (int j=tid; j<k_final; j+=blockDim.x){
        out_vals[r*k_final + j] = vals[j];
        out_idx [r*k_final + j] = idx[j];
    }
}

extern "C"
hipError_t st_kway_merge_bitonic_f32(
    const float* cand_vals, const int32_t* cand_idx,
    int rows, int total, int k_final,
    float* out_vals, int32_t* out_idx,
    hipStream_t stream)
{
    dim3 grid(rows);
    const int BLOCK=256;
    dim3 block(BLOCK);
    size_t sh = total*sizeof(float) + total*sizeof(int32_t);
    hipLaunchKernelGGL(hip_kway_merge_bitonic_f32_kernel, grid, block, sh, stream,
        cand_vals, cand_idx, rows, total, k_final, out_vals, out_idx);
    return hipGetLastError();
}
