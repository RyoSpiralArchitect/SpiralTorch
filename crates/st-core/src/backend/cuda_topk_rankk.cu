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

__device__ __forceinline__ int linear_row_index() {
  long long gx = static_cast<long long>(gridDim.x);
  long long gy = static_cast<long long>(gridDim.y);
  long long x = static_cast<long long>(blockIdx.x);
  long long y = static_cast<long long>(blockIdx.y);
  long long z = static_cast<long long>(blockIdx.z);
  long long row = x + y * gx + z * gx * gy;
  return static_cast<int>(row);
}

struct HeapEntry {
  float value;
  int column;
  int slot;
  int tid;
};

template <typename Comparator>
__device__ __forceinline__ bool prefer_entry(
    const HeapEntry& candidate,
    const HeapEntry& current,
    Comparator cmp) {
  if (candidate.column < 0) {
    return false;
  }
  if (current.column < 0) {
    return true;
  }
  if (cmp(candidate.value, current.value)) {
    return true;
  }
  if (cmp(current.value, candidate.value)) {
    return false;
  }
  if (candidate.column < current.column) {
    return true;
  }
  if (candidate.column > current.column) {
    return false;
  }
  if (candidate.tid < current.tid) {
    return true;
  }
  if (candidate.tid > current.tid) {
    return false;
  }
  return candidate.slot < current.slot;
}

template <typename Comparator>
__device__ __forceinline__ HeapEntry reduce_warp(HeapEntry entry, Comparator cmp) {
  unsigned mask = __activemask();
  int lane = threadIdx.x & (WARP_LANES - 1);
  for (int offset = WARP_LANES / 2; offset > 0; offset >>= 1) {
    int src_lane = lane + offset;
    bool other_active = (src_lane < WARP_LANES) && ((mask >> src_lane) & 1u);
    float other_value = __shfl_down_sync(mask, entry.value, offset);
    int other_col = __shfl_down_sync(mask, entry.column, offset);
    int other_slot = __shfl_down_sync(mask, entry.slot, offset);
    int other_tid = __shfl_down_sync(mask, entry.tid, offset);
    HeapEntry other{other_value, other_col, other_slot, other_tid};
    if (other_active && prefer_entry(other, entry, cmp)) {
      entry = other;
    }
  }
  return entry;
}

struct GreaterThan {
  __device__ bool operator()(float lhs, float rhs) const { return lhs > rhs; }
};

struct LessThan {
  __device__ bool operator()(float lhs, float rhs) const { return lhs < rhs; }
};

template <typename Comparator>
struct HeapTraits;

template <>
struct HeapTraits<GreaterThan> {
  static __device__ __forceinline__ float sentinel() { return -CUDART_INF_F; }
};

template <>
struct HeapTraits<LessThan> {
  static __device__ __forceinline__ float sentinel() { return CUDART_INF_F; }
};

template <typename Comparator>
__device__ __forceinline__ void heap_select_rowwise_kernel_impl(
    const float* __restrict__ X,
    int rows,
    int cols,
    int k,
    float* __restrict__ out_vals,
    int* __restrict__ out_idx,
    float* s_vals,
    int* s_idx,
    HeapEntry* warp_entries,
    HeapEntry* block_choice) {
  int row = linear_row_index();
  if (row >= rows) return;
  int tid = threadIdx.x;
  int stride = blockDim.x;
  if (stride != THREADS_PER_BLOCK) return;

  Comparator cmp;
  float sentinel = HeapTraits<Comparator>::sentinel();

  if (cols <= 0 || k <= 0) {
    if (tid == 0 && k > 0) {
      size_t out_base = static_cast<size_t>(row) * static_cast<size_t>(k);
      for (int oi = 0; oi < k; ++oi) {
        out_vals[out_base + oi] = CUDART_NAN_F;
        out_idx[out_base + oi] = -1;
      }
    }
    return;
  }

  float vbuf[KEEP_PER_THREAD];
  int ibuf[KEEP_PER_THREAD];
  #pragma unroll
  for (int i = 0; i < KEEP_PER_THREAD; ++i) {
    vbuf[i] = sentinel;
    ibuf[i] = -1;
  }

  const float* row_ptr = X + static_cast<size_t>(row) * static_cast<size_t>(cols);

  for (int c = tid; c < cols; c += stride) {
    float v = row_ptr[c];
    #pragma unroll
    for (int pos = 0; pos < KEEP_PER_THREAD; ++pos) {
      if (cmp(v, vbuf[pos]) || (v == vbuf[pos] && (ibuf[pos] < 0 || c < ibuf[pos]))) {
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

  int base = tid * KEEP_PER_THREAD;
  #pragma unroll
  for (int i = 0; i < KEEP_PER_THREAD; ++i) {
    s_vals[base + i] = vbuf[i];
    s_idx[base + i] = ibuf[i];
  }

  int warp = tid / WARP_LANES;
  int lane = tid & (WARP_LANES - 1);

  int take = k < cols ? k : cols;

  for (int oi = 0; oi < take; ++oi) {
    HeapEntry thread_best{sentinel, -1, -1, -1};
    #pragma unroll
    for (int s = 0; s < KEEP_PER_THREAD; ++s) {
      HeapEntry candidate{s_vals[base + s], s_idx[base + s], s, tid};
      if (prefer_entry(candidate, thread_best, cmp)) {
        thread_best = candidate;
      }
    }

    HeapEntry entry = reduce_warp(thread_best, cmp);
    if (lane == 0) {
      warp_entries[warp] = entry;
    }
    __syncthreads();

    if (warp == 0) {
      HeapEntry block_entry;
      if (lane < BLOCK_WARPS) {
        block_entry = warp_entries[lane];
      } else {
        block_entry = HeapEntry{sentinel, -1, -1, -1};
      }
      block_entry = reduce_warp(block_entry, cmp);
      if (lane == 0) {
        *block_choice = block_entry;
      }
    }
    __syncthreads();

    HeapEntry chosen = *block_choice;
    if (tid == chosen.tid && chosen.slot >= 0) {
      s_vals[base + chosen.slot] = sentinel;
      s_idx[base + chosen.slot] = -1;
    }
    if (tid == 0) {
      size_t out_base = static_cast<size_t>(row) * static_cast<size_t>(k);
      out_vals[out_base + oi] = chosen.value;
      out_idx[out_base + oi] = chosen.column;
    }
    __syncthreads();
  }

  if (tid == 0 && take < k) {
    size_t out_base = static_cast<size_t>(row) * static_cast<size_t>(k);
    for (int oi = take; oi < k; ++oi) {
      out_vals[out_base + oi] = CUDART_NAN_F;
      out_idx[out_base + oi] = -1;
    }
    __syncthreads();

    HeapEntry chosen = *block_choice;
    if (tid == chosen.tid && chosen.slot >= 0) {
      s_vals[base + chosen.slot] = sentinel;
      s_idx[base + chosen.slot] = -1;
    }
    if (tid == 0) {
      size_t out_base = static_cast<size_t>(row) * static_cast<size_t>(k);
      out_vals[out_base + oi] = chosen.value;
      out_idx[out_base + oi] = chosen.column;
    }
    __syncthreads();
  }

  if (tid == 0 && take < k) {
    size_t out_base = static_cast<size_t>(row) * static_cast<size_t>(k);
    for (int oi = take; oi < k; ++oi) {
      out_vals[out_base + oi] = CUDART_NAN_F;
      out_idx[out_base + oi] = -1;
    }
    __syncthreads();
  }
}

__global__ void topk_warp_heap_rowwise_kernel(
    const float* __restrict__ X, int rows, int cols, int k,
    float* __restrict__ out_vals, int* __restrict__ out_idx)
{
  extern __shared__ unsigned char smem[];
  float* s_vals = (float*)smem;
  int* s_idx = (int*)(s_vals + blockDim.x * KEEP_PER_THREAD);
  __shared__ HeapEntry warp_entries[BLOCK_WARPS];
  __shared__ HeapEntry block_choice;
  heap_select_rowwise_kernel_impl<GreaterThan>(
      X, rows, cols, k, out_vals, out_idx, s_vals, s_idx, warp_entries, &block_choice);
}

__global__ void bottomk_warp_heap_rowwise_kernel(
    const float* __restrict__ X, int rows, int cols, int k,
    float* __restrict__ out_vals, int* __restrict__ out_idx)
{
  extern __shared__ unsigned char smem[];
  float* s_vals = (float*)smem;
  int* s_idx = (int*)(s_vals + blockDim.x * KEEP_PER_THREAD);
  __shared__ HeapEntry warp_entries[BLOCK_WARPS];
  __shared__ HeapEntry block_choice;
  heap_select_rowwise_kernel_impl<LessThan>(
      X, rows, cols, k, out_vals, out_idx, s_vals, s_idx, warp_entries, &block_choice);
}

__global__ void topk_warp_bitonic_rowwise_kernel(
    const float* __restrict__ X, int rows, int cols, int k,
    float* __restrict__ out_vals, int* __restrict__ out_idx)
{
  int row = linear_row_index();
  if (row >= rows) return;
  if (k <= 0) return;

  unsigned mask = __activemask();
  int lane = threadIdx.x & (WARP_LANES - 1);
  const float* row_ptr = X + static_cast<size_t>(row) * static_cast<size_t>(cols);

  if (cols <= 0) {
    if (lane == 0) {
      size_t out_base = static_cast<size_t>(row) * static_cast<size_t>(k);
      for (int oi = 0; oi < k; ++oi) {
        out_vals[out_base + oi] = CUDART_NAN_F;
        out_idx[out_base + oi] = -1;
      }
    }
    return;
  }

  float best = -CUDART_INF_F;
  int bestc = -1;
  for (int c = lane; c < cols; c += WARP_LANES) {
    float v = row_ptr[c];
    if (v > best || (v == best && (bestc < 0 || c < bestc))) {
      best = v;
      bestc = c;
    }
  }

  for (int offset = WARP_LANES / 2; offset > 0; offset >>= 1) {
    float ov = __shfl_down_sync(mask, best, offset);
    int oc = __shfl_down_sync(mask, bestc, offset);
    if (ov > best || (ov == best && (oc >= 0 && (bestc < 0 || oc < bestc)))) {
      best = ov;
      bestc = oc;
    }
  }

  if (lane == 0) {
    size_t out_base = static_cast<size_t>(row) * static_cast<size_t>(k);
    out_vals[out_base + 0] = best;
    out_idx[out_base + 0] = bestc;
    for (int oi = 1; oi < k; ++oi) {
      out_vals[out_base + oi] = CUDART_NAN_F;
      out_idx[out_base + oi] = -1;
    }
  }
}

__global__ void bottomk_warp_bitonic_rowwise_kernel(
    const float* __restrict__ X, int rows, int cols, int k,
    float* __restrict__ out_vals, int* __restrict__ out_idx)
{
  int row = linear_row_index();
  if (row >= rows) return;
  if (k <= 0) return;

  unsigned mask = __activemask();
  int lane = threadIdx.x & (WARP_LANES - 1);
  const float* row_ptr = X + static_cast<size_t>(row) * static_cast<size_t>(cols);

  if (cols <= 0) {
    if (lane == 0) {
      size_t out_base = static_cast<size_t>(row) * static_cast<size_t>(k);
      for (int oi = 0; oi < k; ++oi) {
        out_vals[out_base + oi] = CUDART_NAN_F;
        out_idx[out_base + oi] = -1;
      }
    }
    return;
  }

  float best = CUDART_INF_F;
  int bestc = -1;
  for (int c = lane; c < cols; c += WARP_LANES) {
    float v = row_ptr[c];
    if (v < best || (v == best && (bestc < 0 || c < bestc))) {
      best = v;
      bestc = c;
    }
  }

  for (int offset = WARP_LANES / 2; offset > 0; offset >>= 1) {
    float ov = __shfl_down_sync(mask, best, offset);
    int oc = __shfl_down_sync(mask, bestc, offset);
    if (ov < best || (ov == best && (oc >= 0 && (bestc < 0 || oc < bestc)))) {
      best = ov;
      bestc = oc;
    }
  }

  if (lane == 0) {
    size_t out_base = static_cast<size_t>(row) * static_cast<size_t>(k);
    out_vals[out_base + 0] = best;
    out_idx[out_base + 0] = bestc;
    for (int oi = 1; oi < k; ++oi) {
      out_vals[out_base + oi] = CUDART_NAN_F;
      out_idx[out_base + oi] = -1;
    }
  }
}

} // extern "C"
