// hip_topk_rankk.hip.cpp
// Rowwise TopK kernels for HIP (MI300X etc.). wavefront=64 assumed.
// Build with hipcc → HSACO. Shared-heap and warp-heap illustrate structure.

#include <hip/hip_runtime.h>
#include <cmath>
#include <limits>

static constexpr int kWavefront = 64;
static constexpr int kLaneKeep = 8;

namespace {

__device__ __host__ inline int lane_keep_for_k(int k) {
  if (k <= 0) {
    return 1;
  }
  int keep = (k + kWavefront - 1) / kWavefront;
  if (keep < 1) {
    keep = 1;
  }
  if (keep > kLaneKeep) {
    keep = kLaneKeep;
  }
  return keep;
}

__device__ inline void initialize_lane_buffers(float* vals, int* idx) {
  #pragma unroll
  for (int i = 0; i < kLaneKeep; ++i) {
    vals[i] = -INFINITY;
    idx[i] = -1;
  }
}

__device__ inline void lane_insert(float v, int id, float* vals, int* idx,
                                    int lane_keep) {
  const int last = lane_keep - 1;
  #pragma unroll
  for (int pos = 0; pos < kLaneKeep; ++pos) {
    if (pos >= lane_keep) {
      break;
    }
    if (v > vals[pos]) {
      for (int q = last; q > pos; --q) {
        vals[q] = vals[q - 1];
        idx[q] = idx[q - 1];
      }
      vals[pos] = v;
      idx[pos] = id;
      break;
    }
  }
}

__device__ inline int compute_row_index() {
  return blockIdx.y + blockIdx.x * gridDim.y;
}

}  // namespace

extern "C" {

__device__ inline float wf_reduce_max(float v) {
  for (int offset=32; offset>0; offset/=2)
    v = fmaxf(v, __shfl_down(v, offset, 64));
  return v;
}

__global__ void topk_shared_heap_rowwise_kernel(
    const float* __restrict__ X, int rows, int cols, int k,
    float* __restrict__ out_vals, int* __restrict__ out_idx)
{
  int row = compute_row_index();
  if (row >= rows) return;
  int lane = threadIdx.x & 63;
  int lane_keep = lane_keep_for_k(k);
  __shared__ float s_vals[64*kLaneKeep];
  __shared__ int   s_idx [64*kLaneKeep];

  float vbuf[kLaneKeep];
  int ibuf[kLaneKeep];
  initialize_lane_buffers(vbuf, ibuf);

  for (int c=lane; c<cols; c+=64) {
    float v = X[row*cols + c];
    lane_insert(v, c, vbuf, ibuf, lane_keep);
  }
  int base = lane*kLaneKeep;
  #pragma unroll
  for (int i=0;i<lane_keep;i++){ s_vals[base+i]=vbuf[i]; s_idx[base+i]=ibuf[i]; }
  __syncthreads();

  if (threadIdx.x==0){
    int total = 64*lane_keep;
    for (int oi=0; oi<k; ++oi) {
      float best_v = -INFINITY; int best_j=0;
      for (int j=0; j<total; ++j) {
        if (s_vals[j]>best_v) { best_v=s_vals[j]; best_j=j; }
      }
      out_vals[row*k + oi] = best_v;
      out_idx[row*k + oi]  = s_idx[best_j];
      s_vals[best_j] = -INFINITY;
    }
  }
}

__global__ void topk_warp_heap_rowwise_kernel(
    const float* __restrict__ X, int rows, int cols, int k,
    float* __restrict__ out_vals, int* __restrict__ out_idx)
{
  int row = compute_row_index();
  if (row >= rows) return;
  int lane = threadIdx.x & 63;
  int lane_keep = lane_keep_for_k(k);
  float vbuf[kLaneKeep];
  int ibuf[kLaneKeep];
  initialize_lane_buffers(vbuf, ibuf);
  for (int c=lane; c<cols; c+=64) {
    float v = X[row*cols + c];
    lane_insert(v, c, vbuf, ibuf, lane_keep);
  }
  extern __shared__ unsigned char smem[];
  float* s_vals = (float*)smem;
  int lane_stride = lane_keep;
  int*   s_idx  = (int*)(s_vals + 64*lane_stride);
  int base = lane*lane_stride;
  #pragma unroll
  for (int i=0;i<lane_keep;i++){ s_vals[base+i]=vbuf[i]; s_idx[base+i]=ibuf[i]; }
  __syncthreads();

  if (threadIdx.x==0){
    int total = 64*lane_keep;
    for (int oi=0; oi<k; ++oi) {
      float best_v = -INFINITY; int best_j=0;
      for (int j=0; j<total; ++j) { if (s_vals[j]>best_v) { best_v=s_vals[j]; best_j=j; } }
      out_vals[row*k + oi] = best_v;
      out_idx[row*k + oi]  = s_idx[best_j];
      s_vals[best_j] = -INFINITY;
    }
  }
}

extern "C" hipError_t st_hip_topk_rowwise_launch(
    const float* host_input,
    int rows,
    int cols,
    int k,
    float* host_out_vals,
    int* host_out_idx)
{
  if (rows < 0 || cols < 0 || k < 0) {
    return hipErrorInvalidValue;
  }

  if (rows == 0 || cols == 0 || k == 0) {
    return hipSuccess;
  }

  if (k > kWavefront * kLaneKeep) {
    return hipErrorInvalidValue;
  }

  auto checked_mul = [](size_t a, size_t b, size_t& out) {
    if (a == 0 || b == 0) {
      out = 0;
      return true;
    }
    if (a > std::numeric_limits<size_t>::max() / b) {
      return false;
    }
    out = a * b;
    return true;
  };

  size_t rows_sz = static_cast<size_t>(rows);
  size_t cols_sz = static_cast<size_t>(cols);
  size_t k_sz = static_cast<size_t>(k);

  size_t elems = 0;
  if (!checked_mul(rows_sz, cols_sz, elems)) {
    return hipErrorInvalidValue;
  }
  size_t input_bytes = 0;
  if (!checked_mul(elems, sizeof(float), input_bytes)) {
    return hipErrorInvalidValue;
  }
  size_t out_elems = 0;
  if (!checked_mul(rows_sz, k_sz, out_elems)) {
    return hipErrorInvalidValue;
  }
  size_t out_val_bytes = 0;
  if (!checked_mul(out_elems, sizeof(float), out_val_bytes)) {
    return hipErrorInvalidValue;
  }
  size_t out_idx_bytes = 0;
  if (!checked_mul(out_elems, sizeof(int), out_idx_bytes)) {
    return hipErrorInvalidValue;
  }

  float* d_input = nullptr;
  float* d_vals = nullptr;
  int* d_idx = nullptr;

  auto cleanup = [&](hipError_t status) {
    if (d_idx) { hipFree(d_idx); d_idx = nullptr; }
    if (d_vals) { hipFree(d_vals); d_vals = nullptr; }
    if (d_input) { hipFree(d_input); d_input = nullptr; }
    return status;
  };

  hipError_t err = hipMalloc(reinterpret_cast<void**>(&d_input), input_bytes);
  if (err != hipSuccess) return cleanup(err);
  err = hipMalloc(reinterpret_cast<void**>(&d_vals), out_val_bytes);
  if (err != hipSuccess) return cleanup(err);
  err = hipMalloc(reinterpret_cast<void**>(&d_idx), out_idx_bytes);
  if (err != hipSuccess) return cleanup(err);

  err = hipMemcpy(d_input, host_input, input_bytes, hipMemcpyHostToDevice);
  if (err != hipSuccess) return cleanup(err);

  dim3 block(kWavefront, 1, 1);
  constexpr unsigned int kMaxGridY = 65535u;
  unsigned int total_rows = static_cast<unsigned int>(rows);
  unsigned int grid_y = total_rows < kMaxGridY ? total_rows : kMaxGridY;
  if (grid_y == 0) {
    grid_y = 1;
  }
  unsigned int grid_x = (total_rows + grid_y - 1) / grid_y;
  dim3 grid(grid_x, grid_y, 1);

  int lane_keep = lane_keep_for_k(k);
  size_t shared = static_cast<size_t>(kWavefront) * static_cast<size_t>(lane_keep) *
                  (sizeof(float) + sizeof(int));

  bool use_warp_heap = false;
  int device = 0;
  hipDeviceProp_t props{};
  if (hipGetDevice(&device) == hipSuccess &&
      hipGetDeviceProperties(&props, device) == hipSuccess) {
    size_t shared_limit = static_cast<size_t>(props.sharedMemPerBlock);
    if (shared <= shared_limit) {
      use_warp_heap = (cols >= kWavefront);
    }
  }

  if (use_warp_heap) {
    hipLaunchKernelGGL(topk_warp_heap_rowwise_kernel, grid, block, shared, 0,
                       d_input, rows, cols, k, d_vals, d_idx);
  } else {
    hipLaunchKernelGGL(topk_shared_heap_rowwise_kernel, grid, block, 0, 0,
                       d_input, rows, cols, k, d_vals, d_idx);
  }

  err = hipGetLastError();
  if (err != hipSuccess) return cleanup(err);

  err = hipDeviceSynchronize();
  if (err != hipSuccess) return cleanup(err);

  err = hipMemcpy(host_out_vals, d_vals, out_val_bytes, hipMemcpyDeviceToHost);
  if (err != hipSuccess) return cleanup(err);
  err = hipMemcpy(host_out_idx, d_idx, out_idx_bytes, hipMemcpyDeviceToHost);
  if (err != hipSuccess) return cleanup(err);

  return cleanup(hipSuccess);
}

extern "C" const char* st_hip_error_string(int code) {
  return hipGetErrorString(static_cast<hipError_t>(code));
}

} // extern "C"
