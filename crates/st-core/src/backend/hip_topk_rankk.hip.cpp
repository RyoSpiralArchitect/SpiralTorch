// hip_topk_rankk.hip.cpp
// Rowwise TopK kernels for HIP (MI300X etc.). wavefront=64 assumed.
// Build with hipcc → HSACO. Shared-heap and warp-heap illustrate structure.

#include <hip/hip_runtime.h>

static constexpr int kWavefront = 64;
static constexpr int kLaneKeep = 8;

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
  int row = blockIdx.y;
  if (row >= rows) return;
  int lane = threadIdx.x & 63;
  const int KLANE = 8;
  __shared__ float s_vals[64*KLANE];
  __shared__ int   s_idx [64*KLANE];

  float vbuf[KLANE]; int ibuf[KLANE];
  #pragma unroll
  for (int i=0;i<KLANE;i++){ vbuf[i]=-INFINITY; ibuf[i]=-1; }

  for (int c=lane; c<cols; c+=64) {
    float v = X[row*cols + c];
    #pragma unroll
    for (int pos=0; pos<KLANE; ++pos) {
      if (v > vbuf[pos]) {
        for (int q=KLANE-1; q>pos; --q) { vbuf[q]=vbuf[q-1]; ibuf[q]=ibuf[q-1]; }
        vbuf[pos]=v; ibuf[pos]=c;
        break;
      }
    }
  }
  int base = lane*KLANE;
  #pragma unroll
  for (int i=0;i<KLANE;i++){ s_vals[base+i]=vbuf[i]; s_idx[base+i]=ibuf[i]; }
  __syncthreads();

  if (threadIdx.x==0){
    int total = 64*KLANE;
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
  int row = blockIdx.y;
  if (row >= rows) return;
  int lane = threadIdx.x & 63;
  const int KLANE = 8;
  float vbuf[KLANE]; int ibuf[KLANE];
  #pragma unroll
  for (int i=0;i<KLANE;i++){ vbuf[i]=-INFINITY; ibuf[i]=-1; }
  for (int c=lane; c<cols; c+=64) {
    float v = X[row*cols + c];
    #pragma unroll
    for (int pos=0; pos<KLANE; ++pos) {
      if (v > vbuf[pos]) {
        for (int q=KLANE-1; q>pos; --q) { vbuf[q]=vbuf[q-1]; ibuf[q]=ibuf[q-1]; }
        vbuf[pos]=v; ibuf[pos]=c;
        break;
      }
    }
  }
  extern __shared__ unsigned char smem[];
  float* s_vals = (float*)smem;
  int*   s_idx  = (int*)(s_vals + 64*KLANE);
  int base = lane*KLANE;
  #pragma unroll
  for (int i=0;i<KLANE;i++){ s_vals[base+i]=vbuf[i]; s_idx[base+i]=ibuf[i]; }
  __syncthreads();

  if (threadIdx.x==0){
    int total = 64*KLANE;
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
  if (k > kWavefront * kLaneKeep) {
    return hipErrorInvalidValue;
  }

  size_t input_bytes = static_cast<size_t>(rows) * static_cast<size_t>(cols) * sizeof(float);
  size_t out_val_bytes = static_cast<size_t>(rows) * static_cast<size_t>(k) * sizeof(float);
  size_t out_idx_bytes = static_cast<size_t>(rows) * static_cast<size_t>(k) * sizeof(int);

  float* d_input = nullptr;
  float* d_vals = nullptr;
  int* d_idx = nullptr;

  auto cleanup = [&](hipError_t status) {
    if (d_idx) hipFree(d_idx);
    if (d_vals) hipFree(d_vals);
    if (d_input) hipFree(d_input);
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

  dim3 grid(1, rows, 1);
  dim3 block(kWavefront, 1, 1);
  size_t shared = static_cast<size_t>(kWavefront) * kLaneKeep * (sizeof(float) + sizeof(int));
  hipLaunchKernelGGL(topk_warp_heap_rowwise_kernel, grid, block, shared, 0,
                     d_input, rows, cols, k, d_vals, d_idx);

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
