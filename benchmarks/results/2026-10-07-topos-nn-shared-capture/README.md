# Compact Shared Gates In Native NN Learning

Implementation: `2fb849edd6143672da412f1af814992e05e9977b`, based on
`d004bd627d7121b7b7951119acaf6b4ac98e1f18`.

The native CPU `ToposResonator` now retains an F-value shared gate and returns
an F-value gate VJP directly from Rust core. It no longer expands that gate or
materializes an N-value gate VJP for a separate CPU tensor reduction. The core
streams the pre-reduction RMS, preserving the existing NN audit semantics.
The legacy host WGPU executor still expands at dispatch and uses its own
reduction. CPU -> GPU -> CPU backward can reuse the same compact CPU tape.

## Correctness Results

- Rust core: 22 tests passed; strict core Clippy passed.
- Native NN: 25 CPU-only tests and 29 WGPU-feature tests passed. The latter
  includes the CPU tests, 12 actual mixed-route combinations, and GPU then CPU
  replay on one compact capture. GPU tests executed, not skipped.
- Fresh Python extension: 123 tests passed, including real resident NN tests.
- Fresh scalar WASM: Node and Chrome each passed 27 cases, 469 checks and 240
  synthetic updates, with exact next-update replay. Browser console/page errors
  were empty. Legacy scalar-WASM capture also passed 24 cases and 240 updates.
- Native `Sequential` and `Parameter::apply_step`: two 100-update synthetic SGD
  trajectories matched an independent CPU Torch implementation. Variable row
  counts were 1/3/8/2, with 5 features and alternating logical tensor layouts;
  coupling 0.2, 5 iterations, porosity 0/0.3, learning rate 0.03.

The Torch comparator evolves its own weights without resetting to Rust weights.
It checks output, input/gate gradients, updated gate and loss at each update,
using rtol `5e-4`, atol `3e-5`. Maximum absolute errors were:

| Field | Maximum Error |
| --- | ---: |
| Output | 1.1920928955078125e-7 |
| Input gradient | 5.960464477539063e-8 |
| Gate gradient | 1.1920928955078125e-7 |
| Updated gate | 2.9802322387695312e-8 |
| Loss | 1.1688770540363436e-7 |

## Boundaries And Records

This is correctness and storage-layout evidence, **not a speed benchmark**,
pretrained FT, GPU-resident HF training, or evidence of model-quality advantage.
The core tape payload is `12*N + 4*F` bytes instead of `16*N` bytes for an
expanded f32 capture. Capacity, NN outputs, parameter snapshots and temporary
buffers are excluded; no peak-memory measurement is claimed.

Strict NN-wide Clippy was attempted and failed on unchanged files: 23 library
diagnostics and 34 with tests, none in the modified files. No lint suppression
was added. The initial CPU test compilation also exposed a WGPU-only test
helper; that test was fixed before the implementation commit. Failed logs are
retained locally alongside successful runs, with hashes in `verification.json`.

`learning.json.gz` contains all 200 native updates; `clients.json` contains
the complete Node/browser reports. `verification.json` binds source, native
and WASM artifacts, local logs, the uncompressed learning record and the Torch
comparison. `SHA256SUMS` binds the public files. Raw build logs and binaries
remain local. The independent source review found no actionable P1/P2 issues;
it was static review, not an independent rerun of these experiments.

## Reproduce

Use fresh output paths and an environment with CPU Torch. No models, datasets
or network calls are needed. See [the learning guide](../../../docs/topos_learning.md)
for Python/WASM builds and client commands.

```bash
cargo test --locked --offline --release -p st-core --lib dynamics::topos_resonator
cargo test --locked --offline --release -p st-nn --lib layers::topos_resonator
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo test --locked --offline --release -p st-nn --features wgpu --lib layers::topos_resonator -- --test-threads=1 --nocapture
cargo run --locked --offline --release -p st-nn --example topos_shared_gate_probe -- /tmp/topos-nn-compact-new.json
python tools/check_topos_shared_gate_learning.py /tmp/topos-nn-compact-new.json /tmp/topos-nn-compact-torch-new.json
python -I -m pytest --import-mode=importlib --confcutdir=tools --rootdir=tools -q tools/test_topos_shared_gate_learning.py
```
