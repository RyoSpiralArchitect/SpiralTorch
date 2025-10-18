#include <hip/hip_runtime.h>
extern "C" __global__
void hip_compaction_scan_pass(const float* __restrict__ vin,
                              unsigned int* __restrict__ pos,
                              int rows, int cols, float low, float high, int tile)
{
    __shared__ unsigned int flags[256];
    int r = blockIdx.x; if (r>=rows) return;
    int tid = threadIdx.x;
    int row_off = r*cols;
    unsigned base = 0;
    for (int c0=0; c0<cols; c0+=tile){
        int cend = min(cols, c0+tile);
        unsigned keep = 0;
        int c = c0 + tid;
        if (c < cend) {
            float v = vin[row_off + c];
            if (v>=low && v<=high) keep = 1;
        }
        flags[tid] = keep;
        __syncthreads();
        // scan
        unsigned offset=1;
        for (int d=256>>1; d>0; d>>=1){
            if (tid<d){ unsigned ai=offset*(2*tid+1)-1; unsigned bi=offset*(2*tid+2)-1; flags[bi]+=flags[ai]; }
            offset<<=1; __syncthreads();
        }
        if (tid==0) flags[255]=0; __syncthreads();
        for (int d=1; d<256; d<<=1){
            offset>>=1;
            if (tid<d){ unsigned ai=offset*(2*tid+1)-1; unsigned bi=offset*(2*tid+2)-1; unsigned t=flags[ai]; flags[ai]=flags[bi]; flags[bi]+=t; }
            __syncthreads();
        }
        if (c < cend) {
            if (keep) pos[row_off + c] = base + flags[tid]; else pos[row_off + c] = 0xffffffffu;
        }
        if (tid==255){
            // recompute last keep
            unsigned last_keep = 0;
            if (cend-1 >= c0){
                float v = vin[row_off + (cend-1)];
                last_keep = (v>=low && v<=high) ? 1u : 0u;
            }
            flags[255]+=last_keep;
        }
        __syncthreads();
        base += flags[255];
        __syncthreads();
    }
}

extern "C"
hipError_t st_compaction_scan_pass(const float* vin,
                                   unsigned int* pos,
                                   int rows,
                                   int cols,
                                   float low,
                                   float high,
                                   int tile,
                                   hipStream_t stream)
{
    if (rows <= 0 || cols <= 0) {
        return hipSuccess;
    }

    if (tile <= 0) {
        tile = 256;
    }

    dim3 grid(rows);
    dim3 block(256);
    hipLaunchKernelGGL(hip_compaction_scan_pass, grid, block, 0, stream,
                       vin, pos, rows, cols, low, high, tile);
    return hipGetLastError();
}
