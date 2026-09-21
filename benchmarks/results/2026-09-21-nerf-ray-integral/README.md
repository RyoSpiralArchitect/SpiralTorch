# Full-interval NeRF integration: correctness and bounded comparison

Measured source: `9101f6e2410bfb5247f6a67669529a58eba9eb80`.
Runtime implementation: `fa78587c5041b02a262e3840bb1e431a61506ff7`.
The fixture is in [`../../nerf-ray-integral`](../../nerf-ray-integral/README.md).
This archive publishes every condition and negative result, not raw arrays or
worker binaries. Those remain at the local paths in `local-raw-manifest.json`.

## Correctness

- All seven new public integration tests fail against the admitted pre-repair
  libraries and pass after repair. Their frozen source, build/test output and
  library hashes are in `excluded-and-baseline/baseline-v2`.
- Those baseline libraries were built at `6899536b46bea49d70d901330451bfbe275cf446`;
  `git diff` confirms no changes to `crates`, `Cargo.toml` or `Cargo.lock` between
  that commit and baseline main `2ab49c1ffa2f05a9df7461e9981ec1ca788f5174`.
  The baseline receipt's runtime commit describes source equivalence, not its
  original build stamp.
- Actual native and Node/WASM `NerfTrainer` execution passes the analytic
  constant-density, zero/thin/opaque interval and logical-layout checks.
- An independent PyTorch 2.12.1 eager CPU implementation uses prefix optical
  depth (rather than the Rust recurrence) and autograd for one SGD step.
  Maximum rendering difference across all measured records: `2.9802322e-8`.
  Maximum difference in the one-step parameter report: `9.0949470e-13`.
- The Rust compositor also passes finite-difference VJP tests. The original
  fixed-seed 20-step learning tests remain unchanged in criterion, seeds and
  learning rate and all pass. This establishes progress, not scene convergence.
- Twelve source-bound local stages passed: formatting, strict vision clippy,
  vision regressions, no-default-feature tests, WASM checking, native/WASM
  fixture builds, bindgen and all three runtime preflights.

Input identity uses exact **f32 little-endian hashes**, shapes and config values.
Incidental f64 JSON formatting is not tensor identity. Native/WASM render-output
hashes and each runtime's repeated outputs are preserved per condition.

## Performance

18 conditions x 3 runtimes x 6 balanced worker orders = **324 measured records**;
54 preconditioning records are kept separately. Each record has five warmups,
nine intervals and four renders per interval. Workers ran serially with no
concurrent build/training work. Native requests `RAYON_NUM_THREADS=4`; PyTorch
uses four intra-op/one inter-op threads; Node runs scalar CPU WASM, not WebGPU.

Ratios are within-block `PyTorch median / runtime median`, then geometric means
over the six blocks. Values greater than one favor that Rust runtime.

| Rays | Samples/ray | Field | Native ratio | WASM ratio |
|---:|---:|:---|---:|---:|
| 1 | 1 | constant | 23.645 | 12.018 |
| 1 | 1 | varying | 24.172 | 12.057 |
| 1 | 8 | constant | 14.990 | 5.063 |
| 1 | 8 | varying | 14.966 | 4.967 |
| 1 | 64 | constant | 4.922 | 1.357 |
| 1 | 64 | varying | 4.897 | 2.360 |
| 32 | 1 | constant | 7.307 | 3.707 |
| 32 | 1 | varying | 7.361 | 3.734 |
| 32 | 8 | constant | 2.049 | 1.016 |
| 32 | 8 | varying | 2.078 | 1.023 |
| 32 | 64 | constant | 0.923 | 0.419 |
| 32 | 64 | varying | 0.940 | 0.418 |
| 256 | 1 | constant | 2.034 | 0.879 |
| 256 | 1 | varying | 2.051 | 0.892 |
| 256 | 8 | constant | 0.972 | 0.420 |
| 256 | 8 | varying | 0.984 | 0.419 |
| 256 | 64 | constant | 0.510 | 0.218 |
| 256 | 64 | varying | 0.510 | 0.219 |

The equally weighted grid geomeans are 3.084x native (12/18 favorable conditions)
and 1.340x WASM (10/18). **These are not universal speedups.** At the largest
varying condition the medians of block medians are native 2.424 ms, WASM 5.659 ms,
PyTorch 1.236 ms. PyTorch wins that condition in every block. Small-shape native
ratios have considerable block spread (e.g. 9.53-30.61x for one varying sample);
short intervals and process/JIT/host effects limit strong steady-state claims.
All interval values and ratio ranges are retained in `results.json`/`summary.json`.

The eager reference caches constant frequency tables and midpoint indices. It
does not use `torch.compile` and is not claimed to be the fastest PyTorch route.
Timing includes field-input construction, field evaluation, integration, f32
output and destruction, but excludes setup/value export/serialization. Rust
also validates rays; WASM includes JS call/free overhead. Input grids, fields
and integration are matched, but frontend/validation overhead is not identical.
No causal pre/post speedup is claimed against the old **incorrect** integral.

The implementation reduces VJP weights/transmittance scratch from `16*B*N`
bytes to `16*N` bytes and removes temporary sample-input concatenation.
This is not a measured peak-memory claim: widths now use f64, and field,
gradient, optimizer and output allocations remain.

## Audit and Replay

```sh
python3 -I -B benchmarks/results/2026-09-21-nerf-ray-integral/verify.py
python3 -I -B benchmarks/results/2026-09-21-nerf-ray-integral/verify.py --raw --source-root /absolute/path/to/SpiralTorch
python3 -I -B benchmarks/results/2026-09-21-nerf-ray-integral/test_verify.py
python3 -I -B benchmarks/results/2026-09-21-nerf-ray-integral/test_analyze.py
```

`verify.py` checks bytes, source links, complete grids, receipts and aggregation.
Even `--raw` hashes payloads/workers; it **does not rerun numerical computation**.
For numerical replay, check out the measured source, use the fixture README's
native/Node/PyTorch build/run commands, then use `analyze.py native.json native.json
wasm.json torch.json`. `check.py` and `measure.py` preserve the original full local
validation/measurement orchestration. Their absolute local CLI/environment paths
must be adapted on another host. Never reuse a shared target for this standalone
fixture; keep native/wasm-bindgen versions and locked dependencies fixed.

The failed initial test-harness constructor, strict-lint errors and two validator
startup failures are preserved. The latter ran **no measurement workers**:
one compared a f32 width to a f64 literal; one mistook JSON formatting for f32
input identity. They were corrected without relaxing numerical tolerances.
An attempted offline lockfile refresh was also rejected by the resolver; no
dependency refresh was adopted. Builds use the previously admitted lockfile
versions, with only the fixture package name changed.

Separate WGSL sampling/compositing pipelines are not called by this trainer.
Their parity, browser NeRF API exposure, GPU residency, training throughput,
real-scene image quality and physical-distance reparameterization are not
established by this archive. The next useful targets are the larger-render CPU
path and the separate WGSL sampler, not another aggregate speedup headline.
