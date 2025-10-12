#include <hip/hip_runtime.h>
extern "C" __global__
void hip_compaction_scan_kernel(const float* __restrict__ vin, int rows, int cols, float low, float high,
                                unsigned* __restrict__ flags, unsigned* __restrict__ tilecnt, int tiles_per_row)
{
    __shared__ unsigned s[256];
    int gid = blockIdx.x; // linear tile id
    int r = gid / tiles_per_row;
    int tile = gid % tiles_per_row;
    if (r >= rows) return;
    int base = r * cols + tile * 256;
    int tid = threadIdx.x;

    unsigned f = 0;
    if (tile*256 + tid < cols) {
        float v = vin[base + tid];
        if (v >= low && v <= high) f = 1;
    }
    s[tid] = f;
    __syncthreads();
    // Blelloch exclusive scan
    for (int d = 128; d>0; d >>= 1) {
        if (tid < d) {
            int ai = (2*tid+1)-1;
            int bi = (2*tid+2)-1;
            s[bi] += s[ai];
        }
        __syncthreads();
    }
    if (tid == 255) s[255] = 0;
    __syncthreads();
    for (int d = 1; d < 256; d <<= 1) {
        if (tid < d) {
            int ai = (2*tid+1)-1;
            int bi = (2*tid+2)-1;
            unsigned t = s[ai];
            s[ai] = s[bi];
            s[bi] += t;
        }
        __syncthreads();
    }
    if (tile*256 + tid < cols) flags[base + tid] = s[tid];
    if (tid == 255) tilecnt[r*tiles_per_row + tile] = s[255] + f;
}
