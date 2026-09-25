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

## Final-tree repeat

After rebasing on merged PR #2136, separating output-value policy routing from the depthwise work estimate, and validating the low-level output geometry, the same strict benchmark was run again. Its release binary SHA-256 is `ebf4dcc96fd97fb411070258e6a6b69b9260199f1999566c42ff34a1b287fe10`. The original run above remains visible rather than being replaced by the repeat.

| NCHW | CPU median ms | WGPU median ms | WGPU / CPU | Max absolute error |
| --- | ---: | ---: | ---: | ---: |
| 1x4x32x32 | 0.250 | 0.570 | 2.279 | 1.2e-7 |
| 1x8x48x48 | 0.959 | 0.692 | 0.721 | 1.2e-7 |
| 1x8x64x64 | 1.490 | 0.621 | 0.417 | 1.2e-7 |
| 2x16x64x64 | 5.402 | 1.295 | 0.240 | 1.2e-7 |
| 4x32x128x128 | 89.836 | 11.482 | 0.128 | 0.9e-7 |

Sorted repeat samples in milliseconds:

| NCHW | CPU samples | WGPU samples |
| --- | --- | --- |
| 1x4x32x32 | 0.247042, 0.249667, 0.250000, 0.250167, 0.253334, 0.274958, 0.285209 | 0.503291, 0.505417, 0.543083, 0.570084, 0.603917, 0.686542, 0.704500 |
| 1x8x48x48 | 0.888291, 0.956875, 0.958167, 0.958500, 1.031667, 1.068875, 1.088750 | 0.546458, 0.595333, 0.635667, 0.691542, 0.696250, 0.734334, 0.933959 |
| 1x8x64x64 | 1.430250, 1.432375, 1.479625, 1.490292, 1.544833, 1.562000, 1.674542 | 0.571209, 0.594708, 0.619792, 0.620875, 0.747583, 0.767750, 0.837583 |
| 2x16x64x64 | 5.336333, 5.381583, 5.393292, 5.401958, 5.404500, 5.433375, 5.457500 | 1.168291, 1.196834, 1.233292, 1.295209, 1.339208, 1.517375, 2.036083 |
| 4x32x128x128 | 88.910958, 88.973166, 89.261708, 89.835709, 89.902125, 91.630042, 91.853125 | 6.854792, 7.579125, 7.688125, 11.482125, 11.673417, 11.766958, 12.938000 |

The two runs agree on the small-shape regression and clear 64x64-and-up improvement on this host. They do not agree on the size of the 48x48 advantage, so that shape remains below the provisional Auto guard.
