# Forward-Only Resident Graph: Native And Browser Evidence

Feature correctness passed. One initial parallel backend-suite failure remains
unexplained; this archive is **not** an unconditional all-tests-green or merge
approval claim.

The source is a working-tree change against
`4cf624d533265f93c1ff5a337712bd3748f4f8e7`. `summary.json` records SHA-256 of
all 18 changed/new source, test, workflow and guide files at verification time.
These are not hashes of a published commit. `manifest.json` binds 48 compressed
artifacts to both compressed and original bytes. No native executable or WASM
binary is published here; their hashes are retained in the verification receipt.

## What Ran

- Apple M4 / Metal / IntegratedGpu: 12 forward graph recipes.
- Chrome 152.0.7977.83 / BrowserWebGpu: the same 12 recipes. Rust reports `Other`
  and an empty adapter name; physical GPU identity is not independently verified.
- Torch 2.12.1 CPU/MPS, explicit MPS fallback disabled: 48 replays and 288
  tensor comparisons; maximum absolute error `1.1920928955078125e-7`.
- Final serial tests: 111 backend tests, 733 NN library tests, eight native GPU
  integration tests, and five Python admission-only tests. No Rust test ignored.
- Native/WASM backend Clippy with `-D warnings`, workspace rustfmt, current-source
  Python binding check and WASM webgpu package check passed. There is no new
  public Python/JS mixed-inference handle in this change.

Each recipe lowers an existing Scaler/Linear/GELU/ReLU/Scaler/Linear model,
preserves 1-D/2-D/3-D logical shapes, packs strided inputs, dispatches repeatedly,
and composes resident preprocessing, postprocessing and another NN graph. All
output reads occur after the workspaces have been reused/dropped. Nine guard
groups cover masked dense/pointwise overflow, inherited errors, recovery, device
mismatch, broadcast/permuted views and dense v1/v2/specialized compatibility.

This is correctness and ownership evidence, not a throughput win, general
autograd residency, model-quality comparison or CUDA result. Graph boundaries
use explicit GPU-only copies; only internal activation edges share buffers.
See the [implementation guide](../../../docs/resident_graph_forward.md).

## Unresolved Parallel Failure

The first candidate `cargo test --release -p st-backend-wgpu --lib` with GPU
runtime tests enabled and the default parallel test scheduler failed once in
`resident_training::stage_tests::transposed_vjps_cover_rectangular_tiles_and_partial_edges_when_enabled`:

```text
prediction[0]: 0 != -0.044921875
110 passed; 1 failed
```

The isolated test passed. A rebuilt clean pre-change tree (commit
`4d0b5d74aac3eac45ec6a25651ad7d1e0129fb7c`) and the candidate then each passed
one full parallel run plus ten additional full-suite repetitions. The candidate
also passed all 111 tests with the serial scheduler used by the GPU CI step.
No core workaround, test suppression or numerical tolerance change was applied.
The cause is **UNKNOWN**: it is neither claimed fixed nor proven pre-existing.

`raw/final-backend-tests.log.xz` and `raw/final-receipt.json.xz` retain the
failure. The baseline/candidate repetition logs and receipts are separate from
the successful final verification. The local serial command driver is archived
in `raw/verify-final.py.xz`; its first-attempt version is preserved separately.
