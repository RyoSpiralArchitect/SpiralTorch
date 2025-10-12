#include <hip/hip_runtime.h>
#include <stdint.h>

extern "C" __global__
void hip_compaction_1ce_kernel(const float* __restrict__ vin,
                               const int32_t* __restrict__ iin,
                               int rows, int cols, float low, float high,
                               float* __restrict__ vout,
                               int32_t* __restrict__ iout)
{
    __shared__ unsigned int flags[256];
    int r = blockIdx.x;
    if (r >= rows) return;
    int tid = threadIdx.x;
    int row_off = r * cols;

    unsigned f = 0;
    if (tid < cols) {
        float v = vin[row_off + tid];
        if (v >= low && v <= high) f = 1;
    }
    flags[tid] = f;
    __syncthreads();

    // Blelloch scan (exclusive) in-place over flags[256]
    unsigned offset = 1;
    for (unsigned d = 256 >> 1; d > 0; d >>= 1) {
        if (tid < d) {
            unsigned ai = offset*(2*tid+1) - 1;
            unsigned bi = offset*(2*tid+2) - 1;
            flags[bi] += flags[ai];
        }
        offset <<= 1;
        __syncthreads();
    }
    if (tid == 0) flags[255] = 0;
    __syncthreads();
    for (unsigned d = 1; d < 256; d <<= 1) {
        offset >>= 1;
        if (tid < d) {
            unsigned ai = offset*(2*tid+1) - 1;
            unsigned bi = offset*(2*tid+2) - 1;
            unsigned t = flags[ai];
            flags[ai] = flags[bi];
            flags[bi] += t;
        }
        __syncthreads();
    }

    // scatter
    if (tid < cols) {
        float v = vin[row_off + tid];
        if (v >= low && v <= high) {
            unsigned pos = flags[tid];
            vout[row_off + pos] = v;
            iout[row_off + pos] = iin[row_off + tid];
        }
    }
}
