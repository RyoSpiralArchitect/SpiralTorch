// cuda_topk_rankk.cu
// Rowwise TopK kernels: warp-heap / warp-bitonic (float32).
// Compile to PTX and load via cust/cudarc. K up to 1024 per row in single pass (extend for larger).

#include <cuda_runtime.h>
#include <math_constants.h>

extern "C" {

constexpr int WARP_LANES = 32;
constexpr int BLOCK_WARPS = 4;
constexpr int THREADS_PER_BLOCK = WARP_LANES * BLOCK_WARPS;
constexpr int KEEP_PER_THREAD = 8;
static_assert(THREADS_PER_BLOCK % WARP_LANES == 0, "blockDim.x must be warp-aligned");
static_assert(BLOCK_WARPS * WARP_LANES == THREADS_PER_BLOCK, "block warp geometry mismatch");

struct HeapEntry {
  float value;
  int column;
  int slot;
  int tid;
};

__device__ __forceinline__ HeapEntry reduce_top_warp(HeapEntry entry) {
  unsigned mask = 0xffffffffu;
  int lane = threadIdx.x & (WARP_LANES - 1);
  for (int offset = WARP_LANES / 2; offset > 0; offset >>= 1) {
    float other_value = __shfl_down_sync(mask, entry.value, offset);
    int other_col = __shfl_down_sync(mask, entry.column, offset);
    int other_slot = __shfl_down_sync(mask, entry.slot, offset);
    int other_tid = __shfl_down_sync(mask, entry.tid, offset);
    bool take_other = (lane + offset) < WARP_LANES && other_value > entry.value;
    if (take_other) {
      entry.value = other_value;
      entry.column = other_col;
      entry.slot = other_slot;
      entry.tid = other_tid;
    }
  }
  return entry;
}

__device__ __forceinline__ HeapEntry reduce_bottom_warp(HeapEntry entry) {
  unsigned mask = 0xffffffffu;
  int lane = threadIdx.x & (WARP_LANES - 1);
  for (int offset = WARP_LANES / 2; offset > 0; offset >>= 1) {
    float other_value = __shfl_down_sync(mask, entry.value, offset);
    int other_col = __shfl_down_sync(mask, entry.column, offset);
    int other_slot = __shfl_down_sync(mask, entry.slot, offset);
    int other_tid = __shfl_down_sync(mask, entry.tid, offset);
    bool take_other = (lane + offset) < WARP_LANES && other_value < entry.value;
    if (take_other) {
      entry.value = other_value;
      entry.column = other_col;
      entry.slot = other_slot;
      entry.tid = other_tid;
    }
  }
  return entry;
}

__global__ void topk_warp_heap_rowwise_kernel(
    const float* __restrict__ X, int rows, int cols, int k,
    float* __restrict__ out_vals, int* __restrict__ out_idx)
{
  int row = blockIdx.y;
  if (row >= rows) return;
  int tid = threadIdx.x;
  int stride = blockDim.x;
  if (stride != THREADS_PER_BLOCK) return;

  float vbuf[KEEP_PER_THREAD];
  int ibuf[KEEP_PER_THREAD];
  #pragma unroll
  for (int i = 0; i < KEEP_PER_THREAD; ++i) {
    vbuf[i] = -CUDART_INF_F;
    ibuf[i] = -1;
  }

  for (int c = tid; c < cols; c += stride) {
    float v = X[row * cols + c];
    #pragma unroll
    for (int pos = 0; pos < KEEP_PER_THREAD; ++pos) {
      if (v > vbuf[pos]) {
        for (int q = KEEP_PER_THREAD - 1; q > pos; --q) {
          vbuf[q] = vbuf[q - 1];
          ibuf[q] = ibuf[q - 1];
        }
        vbuf[pos] = v;
        ibuf[pos] = c;
        break;
      }
    }
  }

  extern __shared__ unsigned char smem[];
  float* s_vals = (float*)smem;
  int* s_idx = (int*)(s_vals + stride * KEEP_PER_THREAD);
  int base = tid * KEEP_PER_THREAD;
  #pragma unroll
  for (int i = 0; i < KEEP_PER_THREAD; ++i) {
    s_vals[base + i] = vbuf[i];
    s_idx[base + i] = ibuf[i];
  }

  __shared__ float warp_vals[BLOCK_WARPS];
  __shared__ int warp_cols[BLOCK_WARPS];
  __shared__ int warp_slots[BLOCK_WARPS];
  __shared__ int warp_tids[BLOCK_WARPS];
  __shared__ float block_value;
  __shared__ int block_col;
  __shared__ int block_slot;
  __shared__ int block_tid;

  int warp = tid / WARP_LANES;
  int lane = tid & (WARP_LANES - 1);

  for (int oi = 0; oi < k; ++oi) {
    float best_v = -CUDART_INF_F;
    int best_slot = -1;
    int best_col = -1;
    #pragma unroll
    for (int s = 0; s < KEEP_PER_THREAD; ++s) {
      float v = s_vals[base + s];
      if (v > best_v) {
        best_v = v;
        best_slot = s;
        best_col = s_idx[base + s];
      }
    }

    HeapEntry entry{best_v, best_col, best_slot, tid};
    entry = reduce_top_warp(entry);
    if (lane == 0) {
      warp_vals[warp] = entry.value;
      warp_cols[warp] = entry.column;
      warp_slots[warp] = entry.slot;
      warp_tids[warp] = entry.tid;
    }
    __syncthreads();

    if (warp == 0) {
      HeapEntry block_entry;
      if (lane < BLOCK_WARPS) {
        block_entry.value = warp_vals[lane];
        block_entry.column = warp_cols[lane];
        block_entry.slot = warp_slots[lane];
        block_entry.tid = warp_tids[lane];
      } else {
        block_entry.value = -CUDART_INF_F;
        block_entry.column = -1;
        block_entry.slot = -1;
        block_entry.tid = -1;
      }
      block_entry = reduce_top_warp(block_entry);
      if (lane == 0) {
        block_value = block_entry.value;
        block_col = block_entry.column;
        block_slot = block_entry.slot;
        block_tid = block_entry.tid;
      }
    }
    __syncthreads();

    if (tid == block_tid && block_slot >= 0) {
      s_vals[base + block_slot] = -CUDART_INF_F;
      s_idx[base + block_slot] = -1;
    }
    if (tid == 0) {
      out_vals[row * k + oi] = block_value;
      out_idx[row * k + oi] = block_col;
    }
    __syncthreads();
  }
}

__global__ void bottomk_warp_heap_rowwise_kernel(
    const float* __restrict__ X, int rows, int cols, int k,
    float* __restrict__ out_vals, int* __restrict__ out_idx)
{
  int row = blockIdx.y;
  if (row >= rows) return;
  int tid = threadIdx.x;
  int stride = blockDim.x;
  if (stride != THREADS_PER_BLOCK) return;

  float vbuf[KEEP_PER_THREAD];
  int ibuf[KEEP_PER_THREAD];
  #pragma unroll
  for (int i = 0; i < KEEP_PER_THREAD; ++i) {
    vbuf[i] = CUDART_INF_F;
    ibuf[i] = -1;
  }

  for (int c = tid; c < cols; c += stride) {
    float v = X[row * cols + c];
    #pragma unroll
    for (int pos = 0; pos < KEEP_PER_THREAD; ++pos) {
      if (v < vbuf[pos]) {
        for (int q = KEEP_PER_THREAD - 1; q > pos; --q) {
          vbuf[q] = vbuf[q - 1];
          ibuf[q] = ibuf[q - 1];
        }
        vbuf[pos] = v;
        ibuf[pos] = c;
        break;
      }
    }
  }

  extern __shared__ unsigned char smem[];
  float* s_vals = (float*)smem;
  int* s_idx = (int*)(s_vals + stride * KEEP_PER_THREAD);
  int base = tid * KEEP_PER_THREAD;
  #pragma unroll
  for (int i = 0; i < KEEP_PER_THREAD; ++i) {
    s_vals[base + i] = vbuf[i];
    s_idx[base + i] = ibuf[i];
  }

  __shared__ float warp_vals[BLOCK_WARPS];
  __shared__ int warp_cols[BLOCK_WARPS];
  __shared__ int warp_slots[BLOCK_WARPS];
  __shared__ int warp_tids[BLOCK_WARPS];
  __shared__ float block_value;
  __shared__ int block_col;
  __shared__ int block_slot;
  __shared__ int block_tid;

  int warp = tid / WARP_LANES;
  int lane = tid & (WARP_LANES - 1);

  for (int oi = 0; oi < k; ++oi) {
    float best_v = CUDART_INF_F;
    int best_slot = -1;
    int best_col = -1;
    #pragma unroll
    for (int s = 0; s < KEEP_PER_THREAD; ++s) {
      float v = s_vals[base + s];
      if (v < best_v) {
        best_v = v;
        best_slot = s;
        best_col = s_idx[base + s];
      }
    }

    HeapEntry entry{best_v, best_col, best_slot, tid};
    entry = reduce_bottom_warp(entry);
    if (lane == 0) {
      warp_vals[warp] = entry.value;
      warp_cols[warp] = entry.column;
      warp_slots[warp] = entry.slot;
      warp_tids[warp] = entry.tid;
    }
    __syncthreads();

    if (warp == 0) {
      HeapEntry block_entry;
      if (lane < BLOCK_WARPS) {
        block_entry.value = warp_vals[lane];
        block_entry.column = warp_cols[lane];
        block_entry.slot = warp_slots[lane];
        block_entry.tid = warp_tids[lane];
      } else {
        block_entry.value = CUDART_INF_F;
        block_entry.column = -1;
        block_entry.slot = -1;
        block_entry.tid = -1;
      }
      block_entry = reduce_bottom_warp(block_entry);
      if (lane == 0) {
        block_value = block_entry.value;
        block_col = block_entry.column;
        block_slot = block_entry.slot;
        block_tid = block_entry.tid;
      }
    }
    __syncthreads();

    if (tid == block_tid && block_slot >= 0) {
      s_vals[base + block_slot] = CUDART_INF_F;
      s_idx[base + block_slot] = -1;
    }
    if (tid == 0) {
      out_vals[row * k + oi] = block_value;
      out_idx[row * k + oi] = block_col;
    }
    __syncthreads();
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
