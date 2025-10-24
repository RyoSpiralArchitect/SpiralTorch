// cuda_attention.cu
// Scaled dot-product attention kernel with optional Z-space and per-query biases.

#include <cuda_runtime.h>
#include <math_constants.h>

extern "C" {

constexpr int WARP_LANES = 32;
constexpr int THREADS_PER_BLOCK = 128;
constexpr int SCRATCH_SIZE = THREADS_PER_BLOCK / WARP_LANES;

__device__ __forceinline__ float warp_reduce_sum(float value) {
  for (int offset = WARP_LANES / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xFFFFFFFF, value, offset);
  }
  return value;
}

__device__ __forceinline__ float warp_reduce_max(float value) {
  for (int offset = WARP_LANES / 2; offset > 0; offset >>= 1) {
    float other = __shfl_down_sync(0xFFFFFFFF, value, offset);
    value = value > other ? value : other;
  }
  return value;
}

__device__ __forceinline__ float block_reduce_sum(float value, float* scratch, int warp_count) {
  int lane = threadIdx.x & (WARP_LANES - 1);
  int warp = threadIdx.x / WARP_LANES;

  value = warp_reduce_sum(value);
  __syncthreads();

  if (lane == 0) {
    scratch[warp] = value;
  }
  __syncthreads();

  float result = 0.0f;
  if (warp == 0) {
    float lane_value = (lane < warp_count) ? scratch[lane] : 0.0f;
    lane_value = warp_reduce_sum(lane_value);
    if (lane == 0) {
      scratch[0] = lane_value;
    }
  }
  __syncthreads();
  result = scratch[0];
  return result;
}

__device__ __forceinline__ float block_reduce_max(float value, float* scratch, int warp_count) {
  int lane = threadIdx.x & (WARP_LANES - 1);
  int warp = threadIdx.x / WARP_LANES;

  value = warp_reduce_max(value);
  __syncthreads();

  if (lane == 0) {
    scratch[warp] = value;
  }
  __syncthreads();

  float result = -CUDART_INF_F;
  if (warp == 0) {
    float lane_value = (lane < warp_count) ? scratch[lane] : -CUDART_INF_F;
    lane_value = warp_reduce_max(lane_value);
    if (lane == 0) {
      scratch[0] = lane_value;
    }
  }
  __syncthreads();
  result = scratch[0];
  return result;
}

__global__ void scaled_dot_attention_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const unsigned int* __restrict__ context_lengths,
    const float* __restrict__ z_bias,
    const float* __restrict__ attn_bias,
    float* __restrict__ attn_probs,
    float* __restrict__ out,
    int contexts,
    int seq_len,
    int head_dim,
    float scale,
    int use_z_bias,
    int use_context_lengths,
    int use_attn_bias,
    int causal_mask,
    int use_attn_probs) {
  int context = blockIdx.y + blockIdx.z * gridDim.y;
  int query = blockIdx.x;
  if (context >= contexts || query >= seq_len) {
    return;
  }

  extern __shared__ float shared[];
  float* scores = shared;
  float* scratch = shared + seq_len;

  int warp_count = (blockDim.x + WARP_LANES - 1) / WARP_LANES;
  int context_offset = context * seq_len;
  size_t q_offset = static_cast<size_t>(context_offset + query) * static_cast<size_t>(head_dim);
  size_t kv_offset = static_cast<size_t>(context_offset) * static_cast<size_t>(head_dim);
  size_t probs_offset = (static_cast<size_t>(context_offset) + static_cast<size_t>(query)) *
                        static_cast<size_t>(seq_len);

  int context_length = seq_len;
  if (use_context_lengths) {
    context_length = static_cast<int>(context_lengths[context]);
    if (context_length > seq_len) {
      context_length = seq_len;
    }
    if (context_length < 0) {
      context_length = 0;
    }
  }
  if (context_length <= 0) {
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
      out[q_offset + d] = 0.0f;
    }
    if (use_attn_probs) {
      for (int key = threadIdx.x; key < seq_len; key += blockDim.x) {
        attn_probs[probs_offset + key] = 0.0f;
      }
    }
    return;
  }
  if (query >= context_length) {
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
      out[q_offset + d] = 0.0f;
    }
    if (use_attn_probs) {
      for (int key = threadIdx.x; key < seq_len; key += blockDim.x) {
        attn_probs[probs_offset + key] = 0.0f;
      }
    }
    return;
  }

  int key_limit = context_length;
  if (causal_mask) {
    int causal_limit = query + 1;
    key_limit = key_limit < causal_limit ? key_limit : causal_limit;
  }

  if (key_limit <= 0) {
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
      out[q_offset + d] = 0.0f;
    }
    if (use_attn_probs) {
      for (int key = threadIdx.x; key < seq_len; key += blockDim.x) {
        attn_probs[probs_offset + key] = 0.0f;
      }
    }
    return;
  }

  const float* q_ptr = q + q_offset;
  const float* k_ptr = k + kv_offset;
  const float* v_ptr = v + kv_offset;

  for (int key = 0; key < key_limit; ++key) {
    const float* key_vec = k_ptr + static_cast<size_t>(key) * static_cast<size_t>(head_dim);
    float dot = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
      dot += q_ptr[d] * key_vec[d];
    }
    float reduced = block_reduce_sum(dot, scratch, warp_count);
    if (threadIdx.x == 0) {
      float biased = reduced * scale;
      if (use_z_bias) {
        biased += z_bias[context_offset + key];
      }
      if (use_attn_bias) {
        size_t bias_idx = (static_cast<size_t>(context_offset) + static_cast<size_t>(query)) *
                          static_cast<size_t>(seq_len) + static_cast<size_t>(key);
        biased += attn_bias[bias_idx];
      }
      scores[key] = biased;
    }
    __syncthreads();
  }

  for (int key = key_limit + threadIdx.x; key < seq_len; key += blockDim.x) {
    scores[key] = -CUDART_INF_F;
  }
  __syncthreads();

  float local_max = -CUDART_INF_F;
  for (int idx = threadIdx.x; idx < seq_len; idx += blockDim.x) {
    float value = scores[idx];
    local_max = value > local_max ? value : local_max;
  }
  float max_score = block_reduce_max(local_max, scratch, warp_count);
  __syncthreads();

  float local_sum = 0.0f;
  for (int idx = threadIdx.x; idx < seq_len; idx += blockDim.x) {
    float value = __expf(scores[idx] - max_score);
    scores[idx] = value;
    local_sum += value;
  }
  float denom = block_reduce_sum(local_sum, scratch, warp_count);
  __syncthreads();

  float inv_denom = denom > 0.0f ? 1.0f / denom : 0.0f;
  for (int idx = threadIdx.x; idx < seq_len; idx += blockDim.x) {
    scores[idx] *= inv_denom;
  }
  __syncthreads();

  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float acc = 0.0f;
    for (int key = 0; key < key_limit; ++key) {
      const float weight = scores[key];
      const float* value_vec = v_ptr + static_cast<size_t>(key) * static_cast<size_t>(head_dim);
      acc += weight * value_vec[d];
    }
    out[q_offset + d] = acc;
  }

  if (use_attn_probs) {
    for (int key = threadIdx.x; key < seq_len; key += blockDim.x) {
      attn_probs[probs_offset + key] = (key < key_limit) ? scores[key] : 0.0f;
    }
  }
}

} // extern "C"
