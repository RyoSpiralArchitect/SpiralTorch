# Resident ConvNeXt narrow LayerNorm diagnostic on Apple M4

This is a bounded synthetic forward result, not a PyTorch comparison, real-image
quality result, GPU backward result, or training-throughput claim. The same
model and input are compared across CPU, resident-input WGPU, and fresh-upload
WGPU within each process. The GPU routes include terminal readback; the
resident-input route excludes the initial upload. No hidden CPU fallback is
used by `forward_resident`.

## Reproduce

- Host: Apple M4 integrated GPU, Metal, macOS 26.4.1, `rustc 1.97.0`.
- Baseline source: commit `0d487381`, including the seven-case benchmark and
  an ignored per-block stage diagnostic. The optimization is commit `34ea25e1`.
  The baseline executable and full raw stdout were
  not retained; its per-process medians below were captured before the change.
- Final optimized release example SHA-256:
  `5736425d36a7600706180618d96e7db2599370e393e7d6e2c85051592ae13b85`.
- Build: `cargo build --release --locked -p st-vision --features wgpu --example convnext_resident_bench -j 1`.
- Measure: `target/release/examples/convnext_resident_bench`.
- The benchmark and manual stage diagnostic reject a CPU WGPU adapter.
- Stage diagnostic: `cargo test --release --locked -p st-vision --features wgpu --lib resident_block_stage_diagnostic -j 1 -- --ignored --nocapture`.
- Browser correctness replay: build `st-backend-wgpu`'s
  `layer_norm_resident_browser` example for `wasm32-unknown-unknown --release`,
  run `wasm-bindgen 0.2.104 --target web --out-name spiraltorch_wasm` on that
  example, then run the `layer-norm-resident` fixture with Playwright available
  through `NODE_PATH`, following the
  [browser replay protocol](../../layer-norm-resident/README.md).
  The ordinary `bindings/st-wasm` package does not export this fixture's
  `run_layer_norm_checks` function.
- Each case uses 3 warmups and 11 measured calls per route, alternating route
  order. Values below are medians in milliseconds from two independent process
  runs, written as `run 1 / run 2`. The original example also prints all raw
  samples and checks finite CPU/WGPU output parity on every iteration.

| Case: NCHW, stage dims, depths | CPU before | Resident WGPU before | Fresh-upload WGPU before |
| --- | ---: | ---: | ---: |
| 1x3x16x16, [4,8], [1,1] | 0.096 / 0.067 | 4.637 / 3.265 | 3.458 / 2.874 |
| 1x3x32x32, [8,16], [0,0] | 0.706 / 0.645 | 1.328 / 1.078 | 1.741 / 1.046 |
| 1x3x32x32, [8,16], [1,0] | 0.987 / 1.136 | 5.747 / 4.534 | 6.093 / 4.575 |
| 1x3x32x32, [8,16], [1,1] | 1.671 / 0.803 | 8.382 / 5.855 | 7.615 / 5.610 |
| 2x3x64x64, [16,32], [0,0] | 0.985 / 1.107 | 3.970 / 3.311 | 4.229 / 3.646 |
| 2x3x64x64, [16,32], [1,0] | 3.811 / 3.659 | 34.644 / 27.900 | 35.356 / 28.162 |
| 2x3x64x64, [16,32], [1,1] | 6.906 / 4.959 | 39.769 / 35.705 | 39.836 / 35.841 |

| Same cases, same order | CPU optimized | Resident WGPU optimized | Fresh-upload WGPU optimized | Max absolute CPU/WGPU error |
| --- | ---: | ---: | ---: | ---: |
| 1x3x16x16, [4,8], [1,1] | 0.076 / 0.085 | 2.301 / 5.128 | 3.317 / 5.427 | 1.93e-6 |
| 1x3x32x32, [8,16], [0,0] | 0.765 / 1.194 | 1.446 / 2.226 | 1.502 / 1.458 | 0 |
| 1x3x32x32, [8,16], [1,0] | 0.854 / 1.319 | 2.672 / 3.608 | 3.706 / 4.006 | 9.54e-7 |
| 1x3x32x32, [8,16], [1,1] | 1.412 / 0.915 | 4.306 / 4.381 | 3.300 / 3.388 | 1.07e-5 |
| 2x3x64x64, [16,32], [0,0] | 1.322 / 2.613 | 3.509 / 3.576 | 3.927 / 4.887 | 0 |
| 2x3x64x64, [16,32], [1,0] | 3.911 / 4.884 | 12.589 / 13.777 | 13.200 / 14.463 | 1.85e-6 |
| 2x3x64x64, [16,32], [1,1] | 6.896 / 7.152 | 16.744 / 16.474 | 17.055 / 17.083 | 2.96e-5 |

The ignored 2x16x32x32 block diagnostic attributed the largest interval to
LayerNorm: before, full block 25.509 / 24.869 ms and synchronized LayerNorm
26.362 / 23.972 ms; after the 32-lane path, full block 10.069 / 8.147 ms and
LayerNorm 9.357 / 7.870 ms. Each diagnostic stage forces its own readback, so
these intervals are **not additive** and are not the unsynchronized model path.
The shader retains its Wide arithmetic and pairwise reduction structure; only
widths up to 32 select a 32-thread workgroup instead of 256 threads. Width 33
and above remain on the previous pipeline. Standalone Tensor, inference graph,
and training graph use the same width rule.

## Verification and limits

- Native `st-backend-wgpu` passed 232 tests, including 28 LayerNorm tests for
  narrow and wide widths, extreme finite inputs, requested VJPs, and training
  graph paths. The new inference-graph boundary test passed at widths 16, 32,
  and 33.
- Native `st-vision/wgpu` passed 70 tests (one manual diagnostic ignored),
  including ConvNeXt block/backbone parity. CPU-only `st-vision` passed 61
  tests. `cargo check --locked -p st-vision --features wgpu --target wasm32-unknown-unknown -j 1 --quiet` passed.
- The [Chrome WebGPU boundary fixture](2026-09-28-vision-layernorm-browser-boundary.json)
  passed seven existing LayerNorm cases, three added widths (16, 32, 33),
  and 400 learning steps on the browser `BrowserWebGpu` runtime. The separate
  adapter probe reported non-fallback Apple Metal. This checks browser
  correctness, not browser performance or browser ConvNeXt execution.
- CPU times and the small cases varied substantially across processes. Thermal state and other
  system load were not controlled. Across processes, constructor-initialized
  weights need not be identical; only the three routes within one process share
  exactly the same model and input. The block-depth ablations are different
  models, not per-kernel timings. No general CPU/GPU crossover, PyTorch
  advantage, training speedup, or accuracy improvement follows from this data.
