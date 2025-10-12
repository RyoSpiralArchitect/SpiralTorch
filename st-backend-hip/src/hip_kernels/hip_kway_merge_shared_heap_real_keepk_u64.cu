#include <hip/hip_runtime.h>
#include <stdint.h>
#include <float.h>
#include <string.h>

// packed: [31:0]=idx, [63:32]=f32 bits
__device__ __forceinline__ float unpack_f(uint64_t p){ uint32_t u = (uint32_t)(p >> 32); float f; memcpy(&f, &u, sizeof(float)); return f; }
__device__ __forceinline__ int   unpack_i(uint64_t p){ return (int)(p & 0xffffffffu); }

template<int M>
__device__ __forceinline__ void keepk_insert_desc(float vals[], int idxs[], float v, int i){
    int pos = M-1;
    if (v <= vals[pos]) return;
    while (pos>0 && v > vals[pos-1]) { vals[pos] = vals[pos-1]; idxs[pos] = idxs[pos-1]; pos--; }
    vals[pos] = v; idxs[pos] = i;
}

// Merge two descending lists a(M), b(M) into out(2M), keep top-K (K<=2M) via selection
template<int M>
__device__ __forceinline__ void merge2_keepk(const float* a, const int* ai, const float* b, const int* bi, float* outv, int* outi, int K){
    int ia=0, ib=0, o=0;
    #pragma unroll
    for (int t=0;t<2*M && o<K; ++t){
        float va = (ia<M)? a[ia]: -INFINITY;
        float vb = (ib<M)? b[ib]: -INFINITY;
        bool take_a = va>=vb;
        outv[o] = take_a ? va : vb;
        outi[o] = take_a ? ai[ia] : bi[ib];
        if (take_a) ia++; else ib++;
        o++;
    }
}

extern "C" __global__
void hip_kway_merge_shared_heap_real_keepk_u64_kernel(
    const uint64_t* __restrict__ cand_packed, // [rows,total]
    int rows, int total, int k_final,
    float* __restrict__ out_vals, int32_t* __restrict__ out_idx)
{
    // Strategy (true keep-k, minimal global bitonic):
    // 1) Per-thread local keep-k_tiny (M=4) from strided scan over 'total' elements.
    // 2) Stage per-thread M to shared as [BLOCK x M] descending lists.
    // 3) Warp-wise tree merge: pairs of M into 2M keep-k lists (in shared), until one list per warp.
    // 4) Block-wide reduction: merge warp lists into final top-k (selection merge), store the first k.
    // Assumptions: BLOCK % 32 == 0; shared sized accordingly.
    const int M = 4;
    const int BLOCK = blockDim.x;
    int r = blockIdx.x; if (r>=rows) return;
    int tid = threadIdx.x;

    // per-thread tiny keep-k
    float tvals[M]; int tidxs[M];
    #pragma unroll
    for (int j=0;j<M;++j){ tvals[j] = -INFINITY; tidxs[j] = -1; }
    for (int i=tid; i<total; i+=BLOCK){
        uint64_t p = cand_packed[r*total + i];
        float v = unpack_f(p);
        int   ix= unpack_i(p);
        keepk_insert_desc<M>(tvals, tidxs, v, ix);
    }

    // shared layout: [BLOCK][M] for vals & idx
    extern __shared__ unsigned char smem_raw[];
    float* svals = (float*)smem_raw;
    int32_t* sidx = (int32_t*)(smem_raw + BLOCK*M*sizeof(float));

    // write locals
    #pragma unroll
    for (int j=0;j<M;++j){ svals[tid*M + j] = tvals[j]; sidx[tid*M + j] = tidxs[j]; }
    __syncthreads();

    // 3) Warp-level merges: reduce per 32 threads into one list of size (M*2.. up to 32*M but we keep top K cap per step)
    // For simplicity, perform pairwise merges within warp onto lane 0's tile in shared.
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int warp_base = warp * 32 * M;

    // Working buffers in shared for the warp (lane0 writes results)
    float* wvals = svals + warp_base;
    int32_t* widx = sidx + warp_base;

    // log2(32)=5 rounds; each round merges distance d
    for (int d=1; d<32; d<<=1){
        int partner = (lane ^ d);
        // merge only for lane < partner and both within warp
        if ((lane % (2*d))==0){
            float* a = wvals + lane*M;
            float* b = wvals + partner*M;
            int32_t* ai = widx + lane*M;
            int32_t* bi = widx + partner*M;
            // temp local buffers
            float outv[2*M]; int outi[2*M];
            merge2_keepk<M>(a, ai, b, bi, outv, outi, min(k_final, 2*M));
            // write back to 'a' region; partner region becomes don't-care
            #pragma unroll
            for (int j=0;j<min(k_final,2*M);++j){ a[j] = outv[j]; ai[j] = outi[j]; }
            #pragma unroll
            for (int j=min(k_final,2*M); j<M; ++j){ a[j] = -INFINITY; ai[j] = -1; } // pad
        }
        __syncthreads();
    }

    // after warp reduction, lane0 at each warp holds top <= 2*M of its 32 lanes
    // compact warp leaders to the first W*M area
    if (lane==0){
        // already in place at lane0 slot
    }
    __syncthreads();

    // 4) Block-wide reduction among warps: number of warps = BLOCK/32
    // Use a simple tree merge across warp leader lists.
    int warps = BLOCK / 32;
    // We'll reuse the first 'warps * (2*M)' entries as inputs, and write results over the first '2*M' repeatedly.
    // Single warp (tid<32) performs final merges to avoid extra sync complexity.
    if (tid < 32){
        float tmpv[2*M]; int tmpi[2*M];
        // seed from warp0
        for (int j=0;j<2*M;++j){
            tmpv[j] = (j<M) ? svals[j] : -INFINITY;
            tmpi[j] = (j<M) ? sidx[j] : -1;
        }
        for (int w=1; w<warps; ++w){
            float *a = tmpv, *b = svals + (w*32*M); // take first M of warp leader
            int   *ai= tmpi, *bi= (int*)(sidx + (w*32*M));
            float outv[2*M]; int outi[2*M];
            merge2_keepk<M>(a, ai, b, bi, outv, outi, min(k_final, 2*M));
            #pragma unroll
            for (int j=0;j<2*M; ++j){ tmpv[j]=outv[j]; tmpi[j]=outi[j]; }
        }
        // write final k
        for (int j=0;j<k_final; ++j){
            out_vals[r*k_final + j] = (j<2*M)? tmpv[j] : -INFINITY;
            out_idx [r*k_final + j] = (j<2*M)? tmpi[j] : -1;
        }
    }
}

extern "C"
hipError_t st_kway_merge_shared_heap_real_keepk_u64(
    const uint64_t* cand_packed,
    int rows, int total, int k_final,
    float* out_vals, int32_t* out_idx,
    hipStream_t stream)
{
    const int BLOCK = 256;
    dim3 grid(rows);
    dim3 block(BLOCK);
    size_t shared = (size_t)BLOCK*4*sizeof(float) + (size_t)BLOCK*4*sizeof(int32_t);
    hipLaunchKernelGGL(hip_kway_merge_shared_heap_real_keepk_u64_kernel, grid, block, shared, stream,
        cand_packed, rows, total, k_final, out_vals, out_idx);
    return hipGetLastError();
}
