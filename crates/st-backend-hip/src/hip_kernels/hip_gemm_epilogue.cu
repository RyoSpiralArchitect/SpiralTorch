#include <hip/hip_runtime.h>
#include <math.h>

namespace {

enum Activation : int {
    kRelu = 1,
    kGelu = 2,
};

__device__ __forceinline__ float gelu_approx(float value) {
    constexpr float kCoeff = 0.044715f;
    constexpr float kSqrt2OverPi = 0.7978846f;
    const float cubed = value * value * value;
    return 0.5f * value
        * (1.0f + tanhf(kSqrt2OverPi * (value + kCoeff * cubed)));
}

__global__ void gemm_epilogue_f32_kernel(
    float* __restrict__ output,
    const float* __restrict__ bias,
    const float* __restrict__ residual,
    int total,
    int cols,
    int activation,
    int has_residual) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) {
        return;
    }

    float value = output[index] + bias[index % cols];
    if (has_residual != 0) {
        value += residual[index];
    }

    if (activation == kRelu) {
        output[index] = value > 0.0f ? value : 0.0f;
    } else if (activation == kGelu) {
        output[index] = gelu_approx(value);
    }
}

}  // namespace

extern "C" hipError_t st_gemm_epilogue_f32(
    float* output,
    const float* bias,
    const float* residual,
    int total,
    int cols,
    int activation,
    int has_residual,
    hipStream_t stream) {
    constexpr int kBlockSize = 256;
    const int grid = 1 + (total - 1) / kBlockSize;
    hipLaunchKernelGGL(
        gemm_epilogue_f32_kernel,
        dim3(grid),
        dim3(kBlockSize),
        0,
        stream,
        output,
        bias,
        residual,
        total,
        cols,
        activation,
        has_residual);
    return hipGetLastError();
}
