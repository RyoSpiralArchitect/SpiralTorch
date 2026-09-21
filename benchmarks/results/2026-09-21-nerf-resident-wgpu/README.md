# Resident NeRF: native WGPU / browser WebGPU / PyTorch control

Measured implementation: `42a7f2a5a61db83f164dafc0efc84be33388d43f`, clean
checkout, 2026-09-21. All 14 final stages used this identical source snapshot.
Cargo build jobs and Rayon threads were limited to four; stages ran serially.

## Result and boundary

- All 36 conditions passed: rays 1/65/256, samples 1/8/64,
  constant/varying affine NN fields, midpoint/seed-17 stratification.
- Maximum absolute error versus the independent eager PyTorch control:
  `1.1324882507324219e-6` for both native and browser.
  The unchanged pointwise acceptance bound is `4e-7 + 4e-6 * abs(reference)`.
- Native/browser emitted f32 outputs are identical on this machine.
  This does not establish cross-vendor portability or general bitwise determinism.
- Both runtimes passed five shape/zero-sample/subnormal-width/inherited-error/
  retained-version guards. Backend tests also cover thin opacity, an opaque tail,
  16,384-sample integration, device identity and legacy buffer layouts.
- The complete backend suite passed 168 unit tests and one shader integration
  test with real-GPU tests enabled. Eight comparator-admission tests, native and
  WASM strict Clippy, workspace formatting and inventory checks passed.

The computation is `sample -> resident NN graph -> composite -> snapshot`:
no intermediate CPU observation. The sampler and compositor write directly into
owning tensor buffers. Noncontiguous NN inputs can still require GPU-side packing.
Native uses Apple M4 / Metal. Chrome 153.0.8010.48 uses BrowserWebGpu; its WGPU
adapter information is masked. The separate JavaScript probe reports
`apple / metal-3 / is_fallback_adapter=false`; it does not attest the exact Rust
runtime adapter. PyTorch is 2.12.1, eager CPU, four intra-op and one inter-op thread.
There is no native CPU/software fallback accepted by the fixture.

This is a **correctness and connection fixture**, not a timing or training
benchmark. The field is affine, not the full positional-encoded
`st_vision::NerfField`. No GPU NeRF VJP, automatic trainer migration, Python
binding, general JavaScript class, scene convergence, or CUDA result is claimed.
Shader arithmetic is f32 with rounded compensated sums; the control uses
f32 affine evaluation and f64 integration. Extreme/subnormal numerical
equivalence is not established. CPU/GPU seeded jitter streams differ.

## Reproduction

At the measured implementation, follow
[the fixture instructions](../../nerf-resident-wgpu/README.md).
Build the locked native example and the locked wasm32 example, generate web
bindings with wasm-bindgen 0.2.104, run the isolated browser fixture and then
the PyTorch control. Exact commands and successful exits are in
`validation.json`; their wall times are command durations, **not performance
measurements**. Build/dependency warnings are retained locally; the selected
crate passes strict linting.

`results.json` publishes every condition, output/input hashes, errors, device
observations and all guards. No large raw arrays or executable/WASM binaries
are checked in. `local-raw-manifest.json` identifies retained exact inputs,
outputs, binaries, generated JS/WASM assets, commands and logs under the local
`Library/Logs/SpiralTorch/nerf-resident-wgpu-20260921` directory. Public artifacts
allow source rebuild; byte-identical local payload replay requires those
retained files. Numerical/GPU replay is distinct from archive verification:

```sh
python3 -I -B benchmarks/nerf-resident-wgpu/verify.py benchmarks/results/2026-09-21-nerf-resident-wgpu
python3 -I -B benchmarks/nerf-resident-wgpu/verify.py benchmarks/results/2026-09-21-nerf-resident-wgpu --raw --source-root .
python3 -I -B benchmarks/nerf-resident-wgpu/test_verify.py
```

The publisher rechecks all raw output comparisons before stripping vectors.
`verify.py` checks bytes, grid completeness, aggregate error consistency and
validation stages; it does not recompute the GPU outputs or prove the numerical
comparison without the raw files.

## Failed attempt retained

The first long-prefix GPU test failed at `0.00024989722` versus
`0.0002498750535187465`. Ordinary floating-point Kahan expressions did not retain
the required accuracy on the actual Metal compiler. Reusing the backend's
integer-defined rounded addition resolved it without loosening tolerances.
The source snapshot, dirty patch, untracked sources, stdout, stderr and failure
receipt are retained in `failed-long-prefix/`.

To reconstruct that attempt, start from the base commit in its receipt,
apply `source.patch` and overlay the `untracked/` files in an isolated checkout;
then run the recorded command. This failure precedes the owning-output
optimization and is not the final source. All development/preflight attempts,
including successful intermediate runs, remain in the local raw manifest.
