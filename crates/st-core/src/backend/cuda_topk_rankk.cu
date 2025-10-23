// cuda_topk_rankk.cu
// Rowwise TopK kernels: warp-heap / warp-bitonic (float32).
// Compile to PTX and load via cust/cudarc. K up to 256 per row in single pass (extend for larger).

#include <cuda_runtime.h>
#include <math_constants.h>

extern "C" {

__inline__ __device__ float warp_reduce_max(float v) {
  for (int offset=16; offset>0; offset/=2)
    v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
  return v;
}

__global__ void topk_warp_heap_rowwise_kernel(
    const float* __restrict__ X, int rows, int cols, int k,
    float* __restrict__ out_vals, int* __restrict__ out_idx)
{
  int row = blockIdx.y;
  if (row >= rows) return;
  int lane = threadIdx.x & 31;
  // lane-local keep-k (linear insert). For brevity KLANE=8 fixed; generalize as template if needed.
  const int KLANE = 8;
  float vbuf[KLANE]; int ibuf[KLANE];
  #pragma unroll
  for (int i=0;i<KLANE;i++){ vbuf[i] = -CUDART_INF_F; ibuf[i]=-1; }

  // strided scan over columns
  for (int c = lane; c < cols; c += 32) {
    float v = X[row*cols + c];
    // insert if better
    #pragma unroll
    for (int pos=0; pos<KLANE; ++pos) {
      if (v > vbuf[pos]) {
        for (int q=KLANE-1; q>pos; --q) { vbuf[q]=vbuf[q-1]; ibuf[q]=ibuf[q-1]; }
        vbuf[pos]=v; ibuf[pos]=c;
        break;
      }
    }
  }

  // Write lane candidates to shared
  extern __shared__ unsigned char smem[];
  float* s_vals = (float*)smem;
  int*   s_idx  = (int*)(s_vals + 32*KLANE);
  int base = lane*KLANE;
  #pragma unroll
  for (int i=0;i<KLANE;i++){ s_vals[base+i]=vbuf[i]; s_idx[base+i]=ibuf[i]; }
  __syncthreads();

  // lane 0 of warp 0 selects global top-k (naive partial sort)
  if (threadIdx.x == 0) {
    int total = 32*KLANE;
    for (int oi=0; oi<k; ++oi) {
      float best_v = -CUDART_INF_F; int best_j=0;
      for (int j=0; j<total; ++j) {
        if (s_vals[j]>best_v) { best_v=s_vals[j]; best_j=j; }
      }
      out_vals[row*k + oi] = best_v;
      out_idx[row*k + oi]  = s_idx[best_j];
      s_vals[best_j] = -CUDART_INF_F;
    }
  }
}

__global__ void bottomk_warp_heap_rowwise_kernel(
    const float* __restrict__ X, int rows, int cols, int k,
    float* __restrict__ out_vals, int* __restrict__ out_idx)
{
  int row = blockIdx.y;
  if (row >= rows) return;
  int lane = threadIdx.x & 31;
  const int KLANE = 8;
  float vbuf[KLANE]; int ibuf[KLANE];
  #pragma unroll
  for (int i=0;i<KLANE;i++){ vbuf[i] = CUDART_INF_F; ibuf[i]=-1; }

  for (int c = lane; c < cols; c += 32) {
    float v = X[row*cols + c];
    #pragma unroll
    for (int pos=0; pos<KLANE; ++pos) {
      if (v < vbuf[pos]) {
        for (int q=KLANE-1; q>pos; --q) { vbuf[q]=vbuf[q-1]; ibuf[q]=ibuf[q-1]; }
        vbuf[pos]=v; ibuf[pos]=c;
        break;
      }
    }
  }

  extern __shared__ unsigned char smem[];
  float* s_vals = (float*)smem;
  int*   s_idx  = (int*)(s_vals + 32*KLANE);
  int base = lane*KLANE;
  #pragma unroll
  for (int i=0;i<KLANE;i++){ s_vals[base+i]=vbuf[i]; s_idx[base+i]=ibuf[i]; }
  __syncthreads();

  if (threadIdx.x == 0) {
    int total = 32*KLANE;
    for (int oi=0; oi<k; ++oi) {
      float best_v = CUDART_INF_F; int best_j=0;
      for (int j=0; j<total; ++j) {
        if (s_vals[j]<best_v) { best_v=s_vals[j]; best_j=j; }
      }
      out_vals[row*k + oi] = best_v;
      out_idx[row*k + oi]  = s_idx[best_j];
      s_vals[best_j] = CUDART_INF_F;
    }
  }
}

__global__ void topk_warp_bitonic_rowwise_kernel(
    const float* __restrict__ X, int rows, int cols, int k,
    float* __restrict__ out_vals, int* __restrict__ out_idx)
{
  int row = blockIdx.y;
  if (row >= rows) return;
  int lane = threadIdx.x & 31;
  // simple chunk max then bitonic across lanes (illustrative; tune as needed)
  float best = -CUDART_INF_F; int bestc=-1;
  for (int c = lane; c < cols; c += 32) {
    float v = X[row*cols + c];
    if (v>best){ best=v; bestc=c; }
  }
  // bitonic reduce (max) to lane0
  for (int offset=16; offset>0; offset/=2) {
    float ov = __shfl_down_sync(0xffffffff, best, offset);
    int   oc = __shfl_down_sync(0xffffffff, bestc, offset);
    if (ov>best){ best=ov; bestc=oc; }
  }
  if (lane==0) {
    out_vals[row*k + 0] = best;
    out_idx[row*k + 0]  = bestc;
    // NOTE: For full K, extend to keep-k network; this shows pattern only.
  }
}

} // extern "C"
