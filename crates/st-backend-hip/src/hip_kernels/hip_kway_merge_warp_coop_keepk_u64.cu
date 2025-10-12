#include <hip/hip_runtime.h>
#include <stdint.h>
#include <float.h>
#include <string.h>

__device__ __forceinline__ float unpack_f(uint64_t p){ uint32_t u = (uint32_t)(p >> 32); float f; memcpy(&f, &u, sizeof(float)); return f; }
__device__ __forceinline__ int   unpack_i(uint64_t p){ return (int)(p & 0xffffffffu); }

template<int M>
__device__ __forceinline__ void keepk_insert_desc(float vals[], int idxs[], float v, int i){
    int pos = M-1;
    if (v <= vals[pos]) return;
    while (pos>0 && v > vals[pos-1]) { vals[pos] = vals[pos-1]; idxs[pos] = idxs[pos-1]; pos--; }
    vals[pos] = v; idxs[pos] = i;
}

// Shuffle-based pairwise merge of two tiny descending lists (M)
// Bring partner's list via shfl and locally merge (host picks M so that registers fit).
template<int M>
__device__ __forceinline__ void warp_pair_merge(float av[M], int ai[M], unsigned mask, int lane, int partner){
    // Pull partner list elements via shuffles
    float bv[M]; int bi[M];
    #pragma unroll
    for (int j=0;j<M;++j){
        int p_index = partner; // lane id
        // HIP equivalent of __shfl_sync: __shfl
        float v = __shfl(av[j], partner, 32);
        int   i = __shfl(ai[j], partner, 32);
        bv[j] = v; bi[j] = i;
    }
    // merge av/ai with bv/bi into av/ai keep top-M
    int pa=0, pb=0;
    float outv[M]; int outi[M];
    #pragma unroll
    for (int j=0;j<M;++j){
        float va = (pa<M)? av[pa]: -INFINITY;
        float vb = (pb<M)? bv[pb]: -INFINITY;
        bool take_a = va>=vb;
        outv[j] = take_a? va: vb;
        outi[j] = take_a? ai[pa]: bi[pb];
        if (take_a) pa++; else pb++;
    }
    #pragma unroll
    for (int j=0;j<M;++j){ av[j]=outv[j]; ai[j]=outi[j]; }
}

extern "C" __global__
void hip_kway_merge_warp_coop_keepk_u64_kernel(
    const uint64_t* __restrict__ cand_packed,
    int rows, int total, int k_final,
    float* __restrict__ out_vals, int32_t* __restrict__ out_idx)
{
    const int M = 4; // per-lane keep-k
    const int WARP = 32;
    const int BLOCK = blockDim.x;
    int r = blockIdx.x; if (r>=rows) return;
    int tid = threadIdx.x;
    int lane = tid % WARP;
    int warp = tid / WARP;

    // per-lane tiny keep-k
    float kv[M]; int ki[M];
    #pragma unroll
    for (int j=0;j<M;++j){ kv[j] = -INFINITY; ki[j] = -1; }
    for (int i=tid; i<total; i+=BLOCK){
        uint64_t p = cand_packed[r*total + i];
        float v = unpack_f(p);
        int   ix= unpack_i(p);
        keepk_insert_desc<M>(kv, ki, v, ix);
    }

    // warp-coop reduction: lanes pairwise merge down to lane0
    unsigned mask = 0xffffffffu;
    for (int d=1; d<32; d<<=1){
        if ((lane % (2*d))==0){
            int partner = lane + d;
            if (partner < 32){
                warp_pair_merge<M>(kv, ki, mask, lane, partner);
            }
        }
    }

    // lane0 writes its M to shared; others idle
    extern __shared__ unsigned char smem_raw[];
    float* svals = (float*)smem_raw;
    int32_t* sidx = (int32_t*)(smem_raw + (BLOCK/WARP)*M*sizeof(float));
    if (lane==0){
        int o = warp*M;
        #pragma unroll
        for (int j=0;j<M;++j){ svals[o+j] = kv[j]; sidx[o+j] = ki[j]; }
    }
    __syncthreads();

    // block-level: now we have warps*M items; select top-k via simple selection (k is small)
    if (tid == 0){
        int W = BLOCK/WARP;
        int pool = W*M;
        for (int j=0;j<k_final; ++j){
            int best = j;
            for (int t=j+1; t<pool; ++t){
                if (svals[t] > svals[best]) best = t;
            }
            // swap
            float tv = svals[best]; int32_t ti = sidx[best];
            svals[best] = svals[j]; sidx[best] = sidx[j];
            svals[j] = tv; sidx[j] = ti;
            out_vals[r*k_final + j] = svals[j];
            out_idx [r*k_final + j] = sidx[j];
        }
    }
}

extern "C"
hipError_t st_kway_merge_warp_coop_keepk_u64(
    const uint64_t* cand_packed,
    int rows, int total, int k_final,
    float* out_vals, int32_t* out_idx,
    hipStream_t stream)
{
    const int BLOCK = 256;
    dim3 grid(rows);
    dim3 block(BLOCK);
    size_t shared = (size_t)(BLOCK/32)*4*sizeof(float) + (size_t)(BLOCK/32)*4*sizeof(int32_t);
    hipLaunchKernelGGL(hip_kway_merge_warp_coop_keepk_u64_kernel, grid, block, shared, stream,
        cand_packed, rows, total, k_final, out_vals, out_idx);
    return hipGetLastError();
}
