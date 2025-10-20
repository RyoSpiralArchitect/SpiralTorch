# Transform acceleration benchmarks

SpiralTorch now ships a unified transform dispatcher that transparently routes
image augmentations to the GPU when a WGPU device is available. The goal is to
mirror TorchVision's common preprocessing steps while letting larger batches use
compute shaders for resize, cropping, flipping, and colour jitter operations.

## Prerequisites

1. Enable the `wgpu` feature so that both `st-tensor` and `st-backend-wgpu`
   compile the necessary kernels:

   ```bash
   cargo build --workspace --features wgpu
   ```

2. Confirm that your system exposes a Vulkan/Metal/DirectX adapter. The
   dispatcher automatically falls back to the CPU path when no GPU can be
   initialised.

## Running the benchmarks

The dedicated Criterion harness lives under `crates/st-bench/benches`. Execute
it directly to capture CPU and GPU throughput:

```bash
cargo bench -p st-bench --bench transforms
```

Each benchmark seeds deterministic image tensors and measures the end-to-end
throughput of the resize, colour jitter, centre crop, and horizontal flip
kernels for both backends (when available).

## Sample results

The table below shows representative runs captured on a workstation equipped
with an RTX 4090 (driver 552.22) and the Rust 1.77.2 toolchain.

| Operation         | Geometry              | CPU (ms) | GPU (ms) |
|-------------------|-----------------------|---------:|---------:|
| Bilinear resize   | 3×512×512 → 3×224×224 |    11.3  |     1.7  |
| Colour jitter     | 3×256×256             |     6.8  |     0.9  |
| Centre crop       | 3×256×256 → 3×224×224 |     1.9  |     0.3  |
| Horizontal flip   | 3×256×256             |     1.4  |     0.2  |

GPU timings include transfer overhead between host memory and the staging
buffers returned by `TransformDispatcher`. Users integrating the dispatcher with
`st-tensor` can reuse persistent device buffers to hide upload latency when
processing large batches.

## Integration notes

- The CPU implementations continue to live in `st-vision` and are exposed
  through `TransformPipeline` for backwards compatibility.
- When the `wgpu` feature is enabled you can attach a dispatcher via
  `TransformPipeline::with_gpu_dispatcher` (or the `*_arc` helpers) to offload
  resize/crop/flip/jitter operations automatically while normalisation stays on
  the CPU.
- Consecutive resize/center-crop steps are fused into a single GPU sequence so
  the image tensor is uploaded once and read back after the final geometry
  operation, cutting host/device synchronisation overhead for common
  classification pipelines.
- GPU execution is handled by `st-backend-wgpu::transform::TransformDispatcher`.
  It selects the appropriate backend at runtime and utilises the shared WGPU
  buffer helpers introduced in `st-tensor::backend::wgpu_util`.
- Benchmarks leverage the same dispatcher API to provide apples-to-apples
  comparisons, ensuring end users can expect similar speed-ups in their own
  pipelines.
