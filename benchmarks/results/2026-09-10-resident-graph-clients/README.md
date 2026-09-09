# Production Resident Graph Clients

Correctness and weight-only resume evidence, **not a throughput benchmark or a
model-quality result**. Python and production browser-WASM wrappers execute the
same Rust-owned mixed graph; no loss, VJP or optimizer is reimplemented in the
production client code.

## Source And Scope

- Clean tested commit: `62c6b969aca8e7316e630a95bdacdf42765fef42`.
- Tree: `0fc9e4dff4240052459eac8492b4047d387e5a6c`.
- Base: `a87bc3cd893509b8da1a5c21cd42da237938830a`.
- Native client: current-source Python 3.12 extension, Apple M4 / Metal.
- Browser client: Chrome `152.0.7977.83`, production `spiraltorch-wasm` webgpu build.
  Rust reports `BrowserWebGpu / Other`; physical adapter and fallback status are
  not attested by this metadata. No private example-only graph wrapper is used.
- Reference: eager PyTorch `2.12.1`, independent CPU/MPS process, MPS fallback
  explicitly disabled. There is no new CUDA evidence in this slice.

## Results

| Check | Result |
| --- | --- |
| Shared Rust contracts | 19 passed, strict Clippy passed |
| WGPU backend | 110 passed, native/WASM strict Clippy passed |
| NN library and resident integration | 733 + 7 passed |
| Python WGPU-build suite | 20 passed, zero skips |
| Python CPU-only suite | 8 executed, 12 GPU tests explicitly skipped; GPU compilation unavailable |
| Production browser graph / CPU-only transport | 24 / 24 recipes passed |
| Legacy dense Python/browser clients | Retained suite and three browser recipes passed for both builds |
| Independent Torch replay | 192 replays, 11,088 tensor/scalar comparisons passed |
| Largest absolute numerical difference | `3.8743019104003906e-7` |

Recipes cover two seeds, ranks 1/2/3 (including 258 flattened rows), both explicit
gradient policies and scalar/sequential versus register-2x2/compensated kernels.
Each captures four updates, resumes Python step-2 weights for two browser updates,
and returns browser step-4 weights for one Python update. Comparisons include
mean-MSE, prediction, input VJP, raw/effective gradients and all updated parameter
roles. Recompilation resets runtime counters and requires a batch and policy.

The browser additionally passed 3,874 rejection assertions, two analytic policy
cases, two late-gain/unbroadcast rollback-and-recovery cases and a parameterless
ReLU graph. Snapshots survive graph drop and batch changes; repeat reads fail.
This does not convert generic `Tensor`, autograd, `ModuleTrainer`, or GNN into a
resident graph, and does not add a general forward-only graph compiler.

## Reproduce And Audit

See [the production-client commands](../../../docs/resident_graph_training.md#reproduce).
`summary.json` is the compact result; `manifest.json` binds 43 archived files to
both compressed and restored SHA-256 digests. `raw/receipt.json.xz` records exact
commands, clean source identity and frozen extension/WASM product hashes.
The binary products remain local; they are not shipped as wheels in this change.
`driver/` contains the host-specific isolated loader/build driver. Decompress the
three handoff JSON reports before invoking the independent Torch validator.

`development-failures/` preserves pre-commit harness failures separately from
the clean-source run: a LayerNorm test-constructor mismatch, omitted gain entries
in a test model state dict, a JS BigInt assertion serializer, and decimal f64 JSON
comparison against Rust's shortest-roundtripping f32. The latter was replaced
by **f32 bit equality**, not a looser tolerance. These are not backend performance
regressions or additional successful GPU evidence.
