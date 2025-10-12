#include <hip/hip_runtime.h>
extern "C" __global__
void hip_compaction_apply_kernel(const float* __restrict__ vin, const int* __restrict__ iin,
                                 int rows, int cols, float low, float high,
                                 const unsigned* __restrict__ flags,
                                 const unsigned* __restrict__ tilecnt, int tiles_per_row,
                                 float* __restrict__ vout, int* __restrict__ iout)
{
    int gid = blockIdx.x;
    int r = gid / tiles_per_row;
    int tile = gid % tiles_per_row;
    if (r >= rows) return;
    int base = r * cols + tile * 256;
    int tid = threadIdx.x;

    unsigned row_off = 0;
    for (int t=0; t<tile; ++t) row_off += tilecnt[r*tiles_per_row + t];

    if (tile*256 + tid < cols) {
        float v = vin[base + tid];
        if (v >= low && v <= high) {
            unsigned pos = flags[base + tid] + row_off;
            vout[r*cols + pos] = v;
            iout[r*cols + pos] = iin[base + tid];
        }
    }
}
