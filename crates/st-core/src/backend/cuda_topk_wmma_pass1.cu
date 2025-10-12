// Scaffold for CUDA TopK Pass1 (candidate generation) using half2 + WMMA (mma.sync).
// Host should compile via NVRTC/PTX. Tile/block sizes are placeholders to be tuned.
extern "C" __global__
void topk_wmma_pass1(const float* __restrict__ X, float* __restrict__ CVAL, int* __restrict__ CIDX,
                     int rows, int cols, int stride, int cand_cols) {
    int row = blockIdx.y;
    if (row >= rows) return;
    int lane = threadIdx.x;
    // TODO: load to shared, convert to half2, feed WMMA fragments, compute candidate maxima per lane.
    // Write best values into CVAL/CI... following index scheme compatible with pass2.
}
