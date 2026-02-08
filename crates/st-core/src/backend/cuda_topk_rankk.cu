// cuda_topk_rankk.cu
// Rowwise TopK / BottomK / MidK kernels for CUDA (float32).

#include <cuda_runtime.h>
#include <cstddef>
#include <math_constants.h>

extern "C" {

constexpr int WARP_LANES = 32;
constexpr int KEEP_PER_THREAD = 8;

__device__ __forceinline__ int linear_row_index() {
  long long gx = static_cast<long long>(gridDim.x);
  long long gy = static_cast<long long>(gridDim.y);
  long long x = static_cast<long long>(blockIdx.x);
  long long y = static_cast<long long>(blockIdx.y);
  long long z = static_cast<long long>(blockIdx.z);
  long long row = x + y * gx + z * gx * gy;
  return static_cast<int>(row);
}

__device__ __forceinline__ bool prefer_desc(
    float cand_v,
    int cand_i,
    float best_v,
    int best_i) {
  if (cand_i < 0) {
    return false;
  }
  if (best_i < 0) {
    return true;
  }
  if (cand_v > best_v) {
    return true;
  }
  if (cand_v < best_v) {
    return false;
  }
  return cand_i < best_i;
}

__device__ __forceinline__ bool prefer_asc(
    float cand_v,
    int cand_i,
    float best_v,
    int best_i) {
  if (cand_i < 0) {
    return false;
  }
  if (best_i < 0) {
    return true;
  }
  if (cand_v < best_v) {
    return true;
  }
  if (cand_v > best_v) {
    return false;
  }
  return cand_i < best_i;
}

__device__ __forceinline__ bool asc_out_of_order(
    float left_v,
    int left_i,
    float right_v,
    int right_i) {
  if (left_v > right_v) {
    return true;
  }
  if (left_v < right_v) {
    return false;
  }
  return left_i > right_i;
}

__device__ __forceinline__ void fill_row_with_nan(
    int row,
    int k,
    float* out_vals,
    int* out_idx) {
  if (k <= 0) {
    return;
  }
  size_t out_base = static_cast<size_t>(row) * static_cast<size_t>(k);
  for (int oi = 0; oi < k; ++oi) {
    out_vals[out_base + oi] = CUDART_NAN_F;
    out_idx[out_base + oi] = -1;
  }
}

__device__ __forceinline__ void fill_tail_with_nan(
    int row,
    int start,
    int k,
    float* out_vals,
    int* out_idx) {
  if (start >= k) {
    return;
  }
  size_t out_base = static_cast<size_t>(row) * static_cast<size_t>(k);
  for (int oi = start; oi < k; ++oi) {
    out_vals[out_base + oi] = CUDART_NAN_F;
    out_idx[out_base + oi] = -1;
  }
}

__device__ __forceinline__ void init_desc(float* vals, int* idx) {
#pragma unroll
  for (int i = 0; i < KEEP_PER_THREAD; ++i) {
    vals[i] = -CUDART_INF_F;
    idx[i] = -1;
  }
}

__device__ __forceinline__ void init_asc(float* vals, int* idx) {
#pragma unroll
  for (int i = 0; i < KEEP_PER_THREAD; ++i) {
    vals[i] = CUDART_INF_F;
    idx[i] = -1;
  }
}

__device__ __forceinline__ void insert_desc(float v, int col, float* vals, int* idx) {
#pragma unroll
  for (int pos = 0; pos < KEEP_PER_THREAD; ++pos) {
    if (prefer_desc(v, col, vals[pos], idx[pos])) {
      for (int q = KEEP_PER_THREAD - 1; q > pos; --q) {
        vals[q] = vals[q - 1];
        idx[q] = idx[q - 1];
      }
      vals[pos] = v;
      idx[pos] = col;
      return;
    }
  }
}

__device__ __forceinline__ void insert_asc(float v, int col, float* vals, int* idx) {
#pragma unroll
  for (int pos = 0; pos < KEEP_PER_THREAD; ++pos) {
    if (prefer_asc(v, col, vals[pos], idx[pos])) {
      for (int q = KEEP_PER_THREAD - 1; q > pos; --q) {
        vals[q] = vals[q - 1];
        idx[q] = idx[q - 1];
      }
      vals[pos] = v;
      idx[pos] = col;
      return;
    }
  }
}

__global__ void topk_warp_heap_rowwise_kernel(
    const float* __restrict__ X,
    int rows,
    int cols,
    int k,
    float* __restrict__ out_vals,
    int* __restrict__ out_idx) {
  int row = linear_row_index();
  if (row >= rows) {
    return;
  }
  if (k <= 0) {
    return;
  }

  int tid = threadIdx.x;
  int stride = blockDim.x;

  if (cols <= 0) {
    if (tid == 0) {
      fill_row_with_nan(row, k, out_vals, out_idx);
    }
    return;
  }

  float local_vals[KEEP_PER_THREAD];
  int local_idx[KEEP_PER_THREAD];
  init_desc(local_vals, local_idx);

  const float* row_ptr = X + static_cast<size_t>(row) * static_cast<size_t>(cols);
  for (int c = tid; c < cols; c += stride) {
    insert_desc(row_ptr[c], c, local_vals, local_idx);
  }

  extern __shared__ unsigned char smem[];
  float* s_vals = reinterpret_cast<float*>(smem);
  int* s_idx = reinterpret_cast<int*>(s_vals + static_cast<size_t>(stride) * KEEP_PER_THREAD);

  int base = tid * KEEP_PER_THREAD;
#pragma unroll
  for (int i = 0; i < KEEP_PER_THREAD; ++i) {
    s_vals[base + i] = local_vals[i];
    s_idx[base + i] = local_idx[i];
  }
  __syncthreads();

  if (tid == 0) {
    int take = k < cols ? k : cols;
    int total = stride * KEEP_PER_THREAD;
    size_t out_base = static_cast<size_t>(row) * static_cast<size_t>(k);

    for (int oi = 0; oi < take; ++oi) {
      float best_v = -CUDART_INF_F;
      int best_i = -1;
      int best_slot = -1;

      for (int slot = 0; slot < total; ++slot) {
        float cand_v = s_vals[slot];
        int cand_i = s_idx[slot];
        if (prefer_desc(cand_v, cand_i, best_v, best_i)) {
          best_v = cand_v;
          best_i = cand_i;
          best_slot = slot;
        }
      }

      if (best_slot >= 0) {
        out_vals[out_base + oi] = best_v;
        out_idx[out_base + oi] = best_i;
        s_vals[best_slot] = -CUDART_INF_F;
        s_idx[best_slot] = -1;
      } else {
        out_vals[out_base + oi] = CUDART_NAN_F;
        out_idx[out_base + oi] = -1;
      }
    }

    fill_tail_with_nan(row, take, k, out_vals, out_idx);
  }
}

__global__ void bottomk_warp_heap_rowwise_kernel(
    const float* __restrict__ X,
    int rows,
    int cols,
    int k,
    float* __restrict__ out_vals,
    int* __restrict__ out_idx) {
  int row = linear_row_index();
  if (row >= rows) {
    return;
  }
  if (k <= 0) {
    return;
  }

  int tid = threadIdx.x;
  int stride = blockDim.x;

  if (cols <= 0) {
    if (tid == 0) {
      fill_row_with_nan(row, k, out_vals, out_idx);
    }
    return;
  }

  float local_vals[KEEP_PER_THREAD];
  int local_idx[KEEP_PER_THREAD];
  init_asc(local_vals, local_idx);

  const float* row_ptr = X + static_cast<size_t>(row) * static_cast<size_t>(cols);
  for (int c = tid; c < cols; c += stride) {
    insert_asc(row_ptr[c], c, local_vals, local_idx);
  }

  extern __shared__ unsigned char smem[];
  float* s_vals = reinterpret_cast<float*>(smem);
  int* s_idx = reinterpret_cast<int*>(s_vals + static_cast<size_t>(stride) * KEEP_PER_THREAD);

  int base = tid * KEEP_PER_THREAD;
#pragma unroll
  for (int i = 0; i < KEEP_PER_THREAD; ++i) {
    s_vals[base + i] = local_vals[i];
    s_idx[base + i] = local_idx[i];
  }
  __syncthreads();

  if (tid == 0) {
    int take = k < cols ? k : cols;
    int total = stride * KEEP_PER_THREAD;
    size_t out_base = static_cast<size_t>(row) * static_cast<size_t>(k);

    for (int oi = 0; oi < take; ++oi) {
      float best_v = CUDART_INF_F;
      int best_i = -1;
      int best_slot = -1;

      for (int slot = 0; slot < total; ++slot) {
        float cand_v = s_vals[slot];
        int cand_i = s_idx[slot];
        if (prefer_asc(cand_v, cand_i, best_v, best_i)) {
          best_v = cand_v;
          best_i = cand_i;
          best_slot = slot;
        }
      }

      if (best_slot >= 0) {
        out_vals[out_base + oi] = best_v;
        out_idx[out_base + oi] = best_i;
        s_vals[best_slot] = CUDART_INF_F;
        s_idx[best_slot] = -1;
      } else {
        out_vals[out_base + oi] = CUDART_NAN_F;
        out_idx[out_base + oi] = -1;
      }
    }

    fill_tail_with_nan(row, take, k, out_vals, out_idx);
  }
}

__global__ void topk_warp_bitonic_rowwise_kernel(
    const float* __restrict__ X,
    int rows,
    int cols,
    int k,
    float* __restrict__ out_vals,
    int* __restrict__ out_idx) {
  int row = linear_row_index();
  if (row >= rows || k <= 0) {
    return;
  }

  int lane = threadIdx.x & (WARP_LANES - 1);
  unsigned mask = __activemask();

  if (cols <= 0) {
    if (lane == 0) {
      fill_row_with_nan(row, k, out_vals, out_idx);
    }
    return;
  }

  const float* row_ptr = X + static_cast<size_t>(row) * static_cast<size_t>(cols);
  float best = -CUDART_INF_F;
  int best_idx = -1;

  for (int c = lane; c < cols; c += WARP_LANES) {
    float v = row_ptr[c];
    if (prefer_desc(v, c, best, best_idx)) {
      best = v;
      best_idx = c;
    }
  }

  for (int offset = WARP_LANES / 2; offset > 0; offset >>= 1) {
    float other_v = __shfl_down_sync(mask, best, offset);
    int other_i = __shfl_down_sync(mask, best_idx, offset);
    if (prefer_desc(other_v, other_i, best, best_idx)) {
      best = other_v;
      best_idx = other_i;
    }
  }

  if (lane == 0) {
    size_t out_base = static_cast<size_t>(row) * static_cast<size_t>(k);
    out_vals[out_base] = best;
    out_idx[out_base] = best_idx;
    fill_tail_with_nan(row, 1, k, out_vals, out_idx);
  }
}

__global__ void bottomk_warp_bitonic_rowwise_kernel(
    const float* __restrict__ X,
    int rows,
    int cols,
    int k,
    float* __restrict__ out_vals,
    int* __restrict__ out_idx) {
  int row = linear_row_index();
  if (row >= rows || k <= 0) {
    return;
  }

  int lane = threadIdx.x & (WARP_LANES - 1);
  unsigned mask = __activemask();

  if (cols <= 0) {
    if (lane == 0) {
      fill_row_with_nan(row, k, out_vals, out_idx);
    }
    return;
  }

  const float* row_ptr = X + static_cast<size_t>(row) * static_cast<size_t>(cols);
  float best = CUDART_INF_F;
  int best_idx = -1;

  for (int c = lane; c < cols; c += WARP_LANES) {
    float v = row_ptr[c];
    if (prefer_asc(v, c, best, best_idx)) {
      best = v;
      best_idx = c;
    }
  }

  for (int offset = WARP_LANES / 2; offset > 0; offset >>= 1) {
    float other_v = __shfl_down_sync(mask, best, offset);
    int other_i = __shfl_down_sync(mask, best_idx, offset);
    if (prefer_asc(other_v, other_i, best, best_idx)) {
      best = other_v;
      best_idx = other_i;
    }
  }

  if (lane == 0) {
    size_t out_base = static_cast<size_t>(row) * static_cast<size_t>(k);
    out_vals[out_base] = best;
    out_idx[out_base] = best_idx;
    fill_tail_with_nan(row, 1, k, out_vals, out_idx);
  }
}

__global__ void midk_shared_odd_even_rowwise_kernel(
    const float* __restrict__ X,
    int rows,
    int cols,
    int k,
    float* __restrict__ out_vals,
    int* __restrict__ out_idx) {
  int row = linear_row_index();
  if (row >= rows || k <= 0) {
    return;
  }

  int tid = threadIdx.x;

  if (cols <= 0) {
    if (tid == 0) {
      fill_row_with_nan(row, k, out_vals, out_idx);
    }
    return;
  }

  extern __shared__ unsigned char smem[];
  float* s_vals = reinterpret_cast<float*>(smem);
  int* s_idx = reinterpret_cast<int*>(s_vals + cols);

  const float* row_ptr = X + static_cast<size_t>(row) * static_cast<size_t>(cols);
  for (int c = tid; c < cols; c += blockDim.x) {
    s_vals[c] = row_ptr[c];
    s_idx[c] = c;
  }
  __syncthreads();

  for (int phase = 0; phase < cols; ++phase) {
    int start = phase & 1;
    for (int c = start + 2 * tid; c + 1 < cols; c += 2 * blockDim.x) {
      float left_v = s_vals[c];
      int left_i = s_idx[c];
      float right_v = s_vals[c + 1];
      int right_i = s_idx[c + 1];
      if (asc_out_of_order(left_v, left_i, right_v, right_i)) {
        s_vals[c] = right_v;
        s_vals[c + 1] = left_v;
        s_idx[c] = right_i;
        s_idx[c + 1] = left_i;
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    int take = k < cols ? k : cols;
    int start = (cols - take) / 2;
    size_t out_base = static_cast<size_t>(row) * static_cast<size_t>(k);
    for (int oi = 0; oi < take; ++oi) {
      out_vals[out_base + oi] = s_vals[start + oi];
      out_idx[out_base + oi] = s_idx[start + oi];
    }
    fill_tail_with_nan(row, take, k, out_vals, out_idx);
  }
}

}  // extern "C"
