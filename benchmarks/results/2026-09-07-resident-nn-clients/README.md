# One Rust NN Plan, Python And WASM Clients

Implementation: `8597915cefe0e6ad658068cd8b929da70969a9b3`,
tree `4909a220054134beb220d9a16258ed2f56664a84`. Later commits add
evidence only. This is public-client correctness evidence, not a new timing
study; the [earlier resident NN comparison](../2026-09-07-resident-nn/README.md)
retains both improvements over legacy execution and slower-than-PyTorch cases.

## What Ran

An existing Python `Sequential(Linear(2,3), Gelu, Linear(3,2))` with fixed,
explicit parameters lowers through the Rust `InferencePlan`. The default
source-built wheel executes it on native WGPU Metal (Apple M4). Its exact
exported plan bytes are then imported by the public WASM `InferencePlan`
and executed through the same Rust resident dense backend.

The fixtures have logical input/output shapes `[2]`, `[2,2]`, and
`[2,3,2]`. Every case includes a second, different input. Outputs agree with
the existing Python module and an independent small f64 oracle at
`atol=1e-5, rtol=1e-4`. The browser checks the Python outputs too, with an
additional register-2x2/compensated case for the 3D fixture.

Old output snapshots remain valid across input replacement, later dispatch,
and workspace destruction. Invalid uploads do not change generations.
Freeing a WASM plan during compilation, or a snapshot while its read promise
is pending, is safe because the pending work owns the Rust resources.
Four overflowing GELU intermediate cases are rejected even after a valid
re-upload. No intermediate host readbacks or host-side NN implementation are
inserted by either client.

## Results

- Rust NN CPU regression: **691 passed**, including exact f32 parameter-bit
  JSON roundtrip checks over extrema, signed zero, subnormals, and thousands
  of finite bit patterns.
- Default Python wheel: **6 resident NN tests**, **3 existing resident matmul
  tests**, and **37 existing smoke tests** passed. The NN GPU suite exercises
  all six scalar/register-2x2 and sequential/tiled/compensated combinations.
- CPU-only Python wheel: **3 plan tests passed**, **3 GPU tests explicitly
  skipped**, and **37 existing smoke tests passed**. GPU compilation raises
  `NotImplementedError`; the plans still serialize and import.
- Public browser WebGPU package: **3 cross-client cases**, **58 rejection
  checks**, and **4 intermediate overflow cases** passed. The browser adapter
  identifies as `BrowserWebGpu / Other` with an empty name: this is not an
  asserted physical M4 identity. No page errors were observed.
- Public CPU-only WASM package: **3 browser transport cases** and a separate
  Node transport/rejection regression passed.
- Generated/shipped TypeScript contracts, pinned CI rustfmt, and WASM feature
  builds passed. No benchmark thresholds were applied.

Python wheels were built separately with default features and
`--no-default-features --features python-default,cpu`, in separate fresh
venvs. Each installed extension was byte-matched against its wheel.
WASM packages were separately built with `--features webgpu` and
`--features nn`. Builds and browser assets are hash-bound in the reports.
This is source/build-log attribution, **not embedded Git attestation or
reproducible-build proof**.

## Artifacts And Limits

- [client-fixture.json](client-fixture.json) contains the actual parameters,
  inputs, independent expected values, and native outputs.
- [verification.json](verification.json) records source/product identity,
  wheel/extension hashes, served WASM/JS/fixture hashes, and browser results.
- [raw-logs.tar.xz](raw-logs.tar.xz) preserves build/test logs, development
  failures, and the native-image verification helper. It excludes binaries,
  wheels, venvs, and generated WASM packages.
- [SHA256SUMS](SHA256SUMS) covers the retained artifact files.

The development failures are retained, not counted as final successes:
fixture parameter names and Tensor shape-method usage were corrected; a
WASM type annotation and Map-versus-object adapter transport were corrected.
Final source-built products passed after those corrections.

This remains explicit Linear/GELU inference. Source-model updates require a
new plan and compilation. Ordinary Module.forward/backward, host-backed 2D
Tensor, and optimizers are unchanged. There is no general N-D broadcasting,
resident training, CUDA execution, browser timing, or new PyTorch speedup
claim here. The wire parser has a caller-overridable 64 MiB JSON byte budget,
not a total memory cap; portable element counts are u32-bounded and actual
device allocation limits are checked separately.

See the [API and reproduction guide](../../../docs/resident_nn_inference.md).
