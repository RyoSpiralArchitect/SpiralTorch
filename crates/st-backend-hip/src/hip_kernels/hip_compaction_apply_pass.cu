#include <hip/hip_runtime.h>
extern "C" __global__
void hip_compaction_apply_pass(const float* __restrict__ vin,
                               const int*   __restrict__ iin,
                               const unsigned int* __restrict__ pos,
                               int rows, int cols, float low, float high,
                               float* __restrict__ vout,
                               int*   __restrict__ iout)
{
    int r = blockIdx.x; if (r>=rows) return;
    int tid = threadIdx.x;
    int row_off = r*cols;
    for (int c=tid; c<cols; c+=blockDim.x){
        float v = vin[row_off + c];
        if (v>=low && v<=high){
            unsigned p = pos[row_off + c];
            if (p != 0xffffffffu){
                vout[row_off + p] = v;
                iout[row_off + p] = iin[row_off + c];
            }
        }
    }
}
