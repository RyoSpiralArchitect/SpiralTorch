#include <metal_stdlib>
using namespace metal;
struct Meta { uint rows, cols, k, k_lane, chunk_cols, cand_cols; };
kernel void topk_kway_1ce(
    device const float* X [[buffer(0)]],
    device float* OUTV [[buffer(1)]],
    device int*   OUTI [[buffer(2)]],
    constant Meta& meta [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]
) {
    if (row >= meta.rows) return;
    threadgroup float cand_vals[256*32];
    threadgroup int   cand_idxs[256*32];
    const uint stride = 256;
    const uint k_lane = meta.k_lane;
    const uint base = row * meta.cols;
    float local_vals[32]; int local_idxs[32];
    for (uint i=0;i<k_lane;i++){ local_vals[i] = -INFINITY; local_idxs[i] = -1; }
    const uint chunk = (meta.chunk_cols==0) ? meta.cols : meta.chunk_cols;
    uint pos = 0;
    while (true){
        uint start_c = pos * chunk;
        if (start_c >= meta.cols) break;
        uint end_c = min(start_c + chunk, meta.cols);
        uint c = start_c + tid;
        while (c < end_c){
            float v = X[base + c];
            uint minp = 0u; float minv = local_vals[0];
            for (uint p=1;p<k_lane;p++){ if (local_vals[p] < minv){ minv = local_vals[p]; minp = p; } }
            if (v > minv){ local_vals[minp]=v; local_idxs[minp]=int(c); }
            c += stride;
        }
        pos++;
        if (meta.chunk_cols==0) break;
    }
    uint offset = tid*k_lane;
    for (uint i=0;i<k_lane;i++){ cand_vals[offset+i]=local_vals[i]; cand_idxs[offset+i]=local_idxs[i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint total = 256u * k_lane;
    for (uint size=2u; size<=total; size<<=1u){
        for (uint stride2=size>>1u; stride2>0u; stride2>>=1u){
            for (uint i=tid; i<total; i+=256u){
                uint j = i ^ stride2;
                if (j > i){
                    bool up = ((i & size) == 0u);
                    float vi = cand_vals[i]; float vj = cand_vals[j];
                    if ((up && vi < vj) || (!up && vi > vj)){
                        float ti = vi; int ii = cand_idxs[i];
                        float tj = vj; int ij = cand_idxs[j];
                        cand_vals[i] = tj; cand_idxs[i] = ij;
                        cand_vals[j] = ti; cand_idxs[j] = ii;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (tid < meta.k){
        OUTV[row*meta.k + tid] = cand_vals[tid];
        OUTI[row*meta.k + tid] = cand_idxs[tid];
    }
}
