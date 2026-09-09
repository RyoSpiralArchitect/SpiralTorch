# Public Resident Graph And N-D Tensor Clients

Python and the production WASM package now expose the same Rust-owned forward
graph and immutable GPU tensor storage. This is bounded correctness/ownership
evidence, **not a PyTorch speedup or an unconditional merge approval**.

The validated source is a working-tree patch against
`6da443ad3d4ec3e4a01a114be04bbecf40e15b8a`. `summary.json` records SHA-256 for
all 28 changed/new source, test, workflow and guide files. The final driver
checks those files and the frozen products again before accepting the run.
`manifest.json` binds each compressed artifact to both compressed and original
bytes. Native/WASM binary hashes are retained, but the binaries are not shipped
in this archive. Initial checks are separate from the final hash-bound run.

## What Ran

- Python 3.12, current-source debug native extension, Apple M4 / Metal:
  30 tests passed without skips and 12 mixed-graph capture recipes passed.
  The loader asserts the exact frozen native-library path; it never uses the
  installed wheel. No installed package or package version was changed.
- Production WASM release build, Chrome 152.0.7977.83 / BrowserWebGpu:
  12 recipes and 536 assertions passed, with no page errors. The adapter reports
  `Other` and an empty name; physical GPU identity is **UNKNOWN**.
- Torch 2.12.1 eager CPU/MPS with MPS fallback explicitly disabled:
  72 replays and 432 tensor comparisons passed. Maximum absolute error was
  `1.1920928955078125e-7`. These are 24 reused frozen Rust-fixture replays,
  24 new Python replays and 24 new WASM replays, not three new core executions.
- CPU-only Python: 30 discovered, 11 executed, 19 GPU-only tests explicitly
  skipped. Device creation and graph compilation raise `NotImplementedError`.
  CPU-only WASM plan transport passed, graph compilation rejected, and GPU
  tensor exports were absent rather than silently routed to CPU.
- Rust backend serial suite: 111 passed, zero failed/ignored, GPU tests enabled.
  Six forward-fixture admission tests and five existing client-admission tests
  also passed; these 11 are validation-only, not GPU tests.
- All 20 final checks passed, including workspace rustfmt, standalone Python
  `extension-module,wgpu` feature check, CPU/GPU builds, generated/shipped
  TypeScript contract checks and whitespace validation.

The frozen recipes cover two seeds, 1-D/2-D/3-D shapes, scalar/sequential and
register-2x2/compensated kernels, repeated dispatch and six output captures.
Inputs, plans and recipe settings must match the hash-bound core fixture before
Torch comparison. Python/JS do transport and orchestration, not the numerical
implementation. Additional public API tests cover views/broadcasting, scalar
and empty tensors, error inheritance, recovery, graph-to-graph composition,
training prediction/input-gradient captures and the older dense executor.

## Boundaries

Compiled graph activations remain on-device. Public graph boundaries use
explicit GPU-only copies, not zero-copy aliases. Captured tensors and pending
readbacks survive workspace reuse/destruction; reads consume their snapshot.
The ordinary `Module::forward`, `pure::Tensor`, autograd and `ModuleTrainer`
paths are not automatically changed. No throughput, training-quality or CUDA
comparison was performed in this run.

The [previous core archive](../2026-09-10-resident-graph-forward/README.md)
retains a one-off default-parallel backend test failure. Its cause remains
**UNKNOWN**. This run's serial pass does not resolve it, and no suppression or
tolerance adjustment was applied. No push, PR, merge or PyPI release was made
as part of this validation.

See the [public API and reproduction guide](../../../docs/resident_graph_forward.md).
The local verification/loader/packing drivers are preserved under `raw/`;
their machine-specific paths are evidence, not a portable installation script.
