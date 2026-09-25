# Host depthwise Conv2d: CPU versus WGPU

This is a bounded routing measurement, not a PyTorch comparison or a full-model training result. The benchmark runs the same `DepthwiseConv2d` weights and input through the host CPU reference and the strict native WGPU path. WGPU timings include upload, kernel execution, and readback on every call. No resident buffers are reused across calls.

## Reproduce

- Source: `spiralreality/vision-depthwise-v1`, `crates/st-vision/examples/depthwise_conv2d_bench.rs` in this change.
- Host: Apple M4 integrated GPU, Metal backend, macOS 26.4.1, `rustc 1.97.0 (2d8144b78 2026-07-07)`.
- Command: `cargo run -q --release -p st-vision --features wgpu --example depthwise_conv2d_bench`.
- Release example binary SHA-256 for this run: `95c275cb7b2f1015861147b77c1654320774b312ac26591e702debc816098085`.
- Each shape has 2 warmups and 7 measured calls per route. Route order alternates each iteration; medians are computed separately. GPU fallback is forbidden, and CPU/WGPU values are compared after every pair.

| NCHW | 7x7 work estimate | CPU median ms | WGPU median ms | WGPU / CPU | Max absolute error |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1x4x32x32 | 200,704 | 0.161 | 0.346 | 2.142 | 1.2e-7 |
| 1x8x48x48 | 903,168 | 0.730 | 0.696 | 0.953 | 1.2e-7 |
| 1x8x64x64 | 1,605,632 | 1.300 | 0.754 | 0.581 | 1.2e-7 |
| 2x16x64x64 | 6,422,528 | 5.305 | 1.283 | 0.242 | 1.2e-7 |
| 4x32x128x128 | 102,760,448 | 84.169 | 14.017 | 0.167 | 0.9e-7 |

Raw measured milliseconds from this run, sorted for compact comparison:

| NCHW | CPU samples | WGPU samples |
| --- | --- | --- |
| 1x4x32x32 | 0.157875, 0.158750, 0.161333, 0.161375, 0.163167, 0.163375, 0.166375 | 0.329291, 0.331375, 0.334208, 0.345667, 0.472583, 0.586625, 0.879542 |
| 1x8x48x48 | 0.720875, 0.726708, 0.729250, 0.730292, 0.731500, 0.737208, 0.970250 | 0.532667, 0.634625, 0.671042, 0.696333, 0.737792, 1.155250, 1.450500 |
| 1x8x64x64 | 1.295459, 1.295667, 1.296208, 1.299625, 1.299834, 1.304125, 1.327084 | 0.564792, 0.670292, 0.715750, 0.754458, 0.791834, 0.804542, 1.108042 |
| 2x16x64x64 | 5.202333, 5.218583, 5.241875, 5.304750, 5.343333, 5.347750, 5.358583 | 1.164000, 1.196000, 1.213208, 1.282833, 1.458750, 1.486791, 1.606667 |
| 4x32x128x128 | 83.898708, 84.128333, 84.145792, 84.168875, 84.192875, 84.316875, 84.719166 | 7.587917, 7.820084, 7.953834, 14.016875, 14.272000, 14.948042, 15.130250 |

The 48x48 case is near the crossover and WGPU samples vary substantially. The provisional Auto guard is 1,500,000 estimated multiply-accumulate opportunities, above that case and below the measured 64x64 case. This is a conservative route for the measured host, not a universal device-calibrated threshold. Backward remains CPU-only; neither complete ConvNeXt training throughput nor accuracy is measured here.
