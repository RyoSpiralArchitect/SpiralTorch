#include <hip/hip_runtime.h>
#include <stdint.h>
extern "C" __global__
void hip_topk_pass1_f32_kernel(const float* __restrict__ X, int rows, int cols, int stride, int k, float* __restrict__ out_vals, int32_t* __restrict__ out_idx){
    int r = blockIdx.x;
    if (r >= rows) return;
    __shared__ float sh_vals[1024 * 8];
    __shared__ int   sh_idx [1024 * 8];
    int tid = threadIdx.x;
    const int T = blockDim.x;
    float lv[8]; int li[8];
    #pragma unroll
    for (int i=0;i<8;i++){ lv[i] = -INFINITY; li[i]=-1; }
    const float* row = X + r * stride;
    for (int c = tid; c < cols; c += T){
        float v = row[c];
        float m=v; int id=c;
        #pragma unroll
        for (int i=0;i<8;i++){
            if (m > lv[i]){ float t=lv[i]; int ti=li[i]; lv[i]=m; li[i]=id; m=t; id=ti; }
        }
    }
    for (int i=0;i<8;i++){ sh_vals[tid*8 + i] = lv[i]; sh_idx[tid*8 + i] = li[i]; }
    __syncthreads();
    if (tid==0){
        int total=T*8;
        for (int j=0;j<k && j<total; j++){
            int best=0; float bv=sh_vals[0];
            for (int t=1;t<total;t++){ if (sh_vals[t]>bv){ bv=sh_vals[t]; best=t; } }
            out_vals[r*k + j]=bv; out_idx[r*k + j]=sh_idx[best];
            sh_vals[best]=-INFINITY;
        }
    }
}
extern "C"
hipError_t st_topk_pass1_f32(const float* dX, int rows, int cols, int stride, int k, float* dVals, int32_t* dIdx, hipStream_t stream){
    dim3 grid(rows); dim3 block(1024);
    hipLaunchKernelGGL(hip_topk_pass1_f32_kernel, grid, block, 0, stream, dX, rows, cols, stride, k, dVals, dIdx);
    return hipGetLastError();
}
