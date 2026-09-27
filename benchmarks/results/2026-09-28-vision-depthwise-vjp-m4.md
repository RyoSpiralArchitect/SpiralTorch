# Resident depthwise VJP on Apple M4

This is a bounded synthetic backward diagnostic, not a PyTorch comparison,
ConvNeXt training result, optimizer benchmark, or GPU speedup claim. The Rust
backend now computes `(input, weight, bias)` gradients on the owning GPU queue
without intermediate readback or floating-point atomics. All three results
share one validity guard, so overflow in any requested gradient rejects the
whole VJP. The model-owned `DepthwiseConv2d::vjp_resident` exposes those
gradients without silently mutating host parameters.

## Reproduce

- Native adapter: Apple M4 integrated GPU, Metal, macOS 26.4.1.
- Build and run: `cargo run --release --locked -p st-vision --features wgpu --example depthwise_resident_vjp_bench -j 1`.
- The example rejects a CPU WGPU adapter, alternates routes, uses three
  warmups and 11 measured calls per route, prints raw samples, and compares
  all three gradients on every iteration. Inputs and weights are reused.
  The [three-process sample receipt](2026-09-28-vision-depthwise-vjp-m4-samples.json)
  records measured calls in order, rounded to six decimal places in ms.
- Browser: build `st-backend-wgpu` example `depthwise_vjp_resident_browser` for
  `wasm32-unknown-unknown --release`, run matching `wasm-bindgen 0.2.104
  --target web --out-name spiraltorch_wasm`, then use
  `tools/test_resident_browser.cjs` with fixture `depthwise-vjp-resident`.
  The [Chrome WebGPU report](2026-09-28-vision-depthwise-vjp-browser.json)
  records the generated WASM, JavaScript, and fixture hashes.

| NCHW, kernel | CPU backward median ms, three processes | Resident VJP through three terminal readbacks median ms, three processes | Max absolute gradient error |
| --- | ---: | ---: | ---: |
| 1x4x16x16, 3x3 | 0.016 / 0.016 / 0.016 | 0.748 / 0.856 / 0.862 | < 1e-6 |
| 2x16x32x32, 7x7 | 2.318 / 2.297 / 2.347 | 2.382 / 1.758 / 2.492 | 3e-6 |

The CPU route includes host parameter-gradient accumulation. The GPU route
returns resident gradients and includes three explicit terminal readbacks, but
does not apply an optimizer update. Therefore the two routes are not matched
training steps. Small workloads are clearly CPU-favored; the larger condition
changes ordering across process runs. System load and thermal state were not
controlled. The raw samples contain several multi-millisecond GPU stalls; a
single median hides that tail. No general CPU/GPU crossover or PyTorch
advantage follows.

Native `st-backend-wgpu` passed 234 real-GPU-enabled tests, including a
non-contiguous stride/padding/dilation VJP reference, empty batch, invalid
source, and cross-gradient overflow guard. The `st-nn` model-owned VJP matched
the existing CPU `DepthwiseConv2d::backward` without host accumulation. Seven
CPU-only depthwise tests passed. `st-backend-wgpu` and `st-nn/wgpu` passed
`wasm32-unknown-unknown` checks. The Chrome fixture passed three gradient
comparisons, three empty-batch checks, and three shared-guard checks on the
browser `BrowserWebGpu` runtime; a separate adapter probe reported non-fallback
Apple Metal. This does not establish browser performance or identify the Rust
adapter as the probed device.

Strict native and WASM `st-backend-wgpu` clippy passed. Repository-wide
`st-nn --features wgpu --lib` tests passed (787 tests), but its strict clippy
still fails on 24 pre-existing library warnings outside this VJP slice.

Full resident ConvNeXt learning still needs the remaining block VJPs, a
model-owned GPU update path, and end-to-end matched training evidence.
