# Resident LayerNorm: numerical foundation

The explicit Rust backend `ResidentTensor::layer_norm_affine` returns an owning
forward value and reusable `ResidentLayerNorm` tape. `backward(seed, scale,
requested)` returns selected input/gamma/beta cotangents. Statistics, centered
values, cotangents and parameter updates can remain on the same GPU queue.
Native WGPU and browser WebGPU compile the same embedded WGSL.

This is **not** a default-route switch. Ordinary `pure::Tensor` LayerNorm VJP
still uses its existing CPU-statistics/GPU-utility hybrid. NN graph integration,
Python/WASM public facade exposure, and a faster common-case kernel remain
follow-up work. The browser fixture exercises the real Rust backend, not a new
JavaScript reimplementation.

## Contract

- Normalize the last axis; gamma/beta have identical `[cols]` or `[1, cols]`
  shape. Non-contiguous N-D views are packed on GPU.
- Epsilon is finite and nonnegative. Empty leading axes are valid. A constant
  row with zero epsilon fails through the deferred whole-operation guard.
- Affine gradient scale is finite, applied after the row sum, never to dx.
  Unrequested gradients are not evaluated; their overflow cannot reject dx.
- Fresh output/tape storage survives repeated calls and dropped parents. All
  requested gradients share one guard, including inherited input failures.
- Private three-component significands and extended binary exponents preserve
  cancellation and intermediate range. This is **not IEEE f64 emulation** or
  an all-input equivalence proof. The centered tape uses 16 bytes per element,
  a deliberate initial correctness cost rather than a claimed optimal layout.
- Input VJP cancels centered variance/covariance products before dividing by
  variance. Epsilon is added after that cancellation so a small positive value
  is not lost. Each row retains inverse standard deviation and raw squared sum
  in 32 bytes. Zero-epsilon scale directions, tiny positive epsilon, and signed
  seed-scale/permutation variants are regression-tested without relaxed bounds.
- Zero components bypass unnecessary expansion work. Ordered FastTwoSum keeps
  exact residuals with three integer-rounded additions; a native 16,708-pair
  GPU test compares both sum and residual bits against unordered CPU TwoSum.
  Input VJP division is row-constant and computed once per row, not per element.
  Preflight accounts for both reductions and all four shared row scalars.
- Forward retains the CPU's rounded-f32 normalized value and affine operation
  boundaries. The affine product decodes subnormal bits before multiplication:
  otherwise `(tiny normalized value) * large gamma` can incorrectly become zero.

## Exploratory Comparisons

The native example compares unchanged CPU, unchanged hybrid WGPU, and explicit
resident WGPU. The Python harness exercises ATen on CPU and real MPS with
fallback disabled. Both measure input materialization, affine forward, all
three VJPs, affine gradient scale 0.5, and four owning CPU outputs. Pipeline
construction is warmed. Cases are 2x3, 8x257, 32x256, 64x768, 128x1025 and
256x256, with three warmups and eighteen rotating-order intervals per route.

These are shared-machine exploratory timings, **not** a matched multi-round
performance admission. Python/ATen autograd and Rust direct VJP have different
host overheads. Four terminal observations intentionally expose current API
costs; they are not the cost of a wholly resident training step. Preserve slower
cells and CPU wins, not just the hybrid-to-resident speed ratios.

Run each command through `run.py ROOT TARGET NEW_OUTPUT COMMAND...` to retain
source hashes, dirty-source patches, untracked source, logs, status, and failure
receipts. Each output directory must be new. Build jobs and Rayon are limited
to four. Large binaries stay local; public evidence should include their hashes.

```sh
cargo +1.98.0 run --locked --release -p st-tensor \
  --no-default-features --features cpu,wgpu_dense --example layer_norm_resident_bench
PYTORCH_ENABLE_MPS_FALLBACK=0 python3 -I -B benchmarks/layer-norm-resident/torch_bench.py
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo +1.98.0 test --locked -p st-backend-wgpu \
  --lib resident_tensor::normalization::tests -- --nocapture --test-threads=1
cargo +1.98.0 build --locked --release -p st-backend-wgpu \
  --target wasm32-unknown-unknown --example layer_norm_resident_browser
```

For the browser test, generate that example with wasm-bindgen 0.2.104 using
`--target web --out-name spiraltorch_wasm`, then use
`tools/test_resident_browser.cjs MODULE_DIR CHROME NEW_REPORT "" "" "" "" layer-norm-resident`.
The harness launches an isolated test browser, never the user's profile.

## Retained Negative Controls

The initial exploratory 400-step affine SGD fixture (2x3 input, learning rate
0.05) did not meet an initially assumed loss ratio of 1e-4. CPU and PyTorch also
missed it: this was an unsupported convergence assumption, not a measured VJP
regression. The same unchanged fixture is retained with CPU parameter/loss
parity and explicit non-convergence. Separately, the established
`layer_norm_autograd.rs` fixture keeps its data, rate 0.1, 400 steps and 1e-4
criterion unchanged, on both native and browser resident paths.

A subsequent subnormal-affine probe reproduced a real error: expected
`-0.09223365`, observed `0`. It is now a native/browser regression case. Neither
its input nor the 2e-5 scaled comparison tolerance was relaxed.

The installed PyTorch 2.12.1 MPS training control also failed when the input did
not require a gradient. A separate direct `aten.native_layer_norm_backward`
probe on the fixed 3x3 fixture found incorrect affine gradients for masks
`[false,true,false]`, `[false,false,true]` and `[false,true,true]`; CPU passed all
seven masks, and MPS passed the masks requesting dx. Requesting an unused dx
restored the two 400-step MPS controls. This is a **local, mask-specific negative
result**, not a general PyTorch finding or an accepted same-work workaround.
The performance harness already requests all gradients, and checks their
values. `torch_mask_probe.py` exits normally after gathering diagnostics;
`all_masks_valid=false` must never be interpreted as numerical acceptance.
No installed PyTorch code or default route was modified.

The initial two-component normalized-tape candidate also failed a zero-epsilon
scale-nullspace probe and an independent review's large-dynamic-range input
VJP. Adding division refinements alone did not fix the latter. Historical
timings do not describe the revised centered three-component implementation;
re-measure it separately. `decimal_probe.py` reproduces a 100-digit independent
oracle used alongside the existing centered-f64 reference, not in place of its
original review-case check.
