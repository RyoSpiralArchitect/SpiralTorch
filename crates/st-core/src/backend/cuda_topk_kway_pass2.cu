// Scaffold for CUDA TopK Pass2 (K-way merge) with shared heap/bitonic skeleton and multi-CTA hooks.
extern "C" __global__
void topk_kway_pass2(float* __restrict__ CVAL, const int* __restrict__ CIDX,
                     float* __restrict__ OVAL, int* __restrict__ OIDX,
                     int rows, int k, int cand_cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    // TODO: shared buffer for candidate values; iteratively select max K items;
    // For multi-CTA, stage outputs across CTAs and merge; or assign per-row CTAs.
}
