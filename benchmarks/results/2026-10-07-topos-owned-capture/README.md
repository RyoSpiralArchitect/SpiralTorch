# Topos Owned Capture

`ToposResonatorOperator::capture_owned` transfers input and gate `Vec<f32>`
allocations into the learning tape. Python sequence/buffer capture and WASM
capture now use it after establishing Rust ownership. Borrowed Rust `capture`
remains available; both paths share the same audited transition. There is no
change to the finite map, gradients, audit, budgets or validation order.

Two N-element internal copies are removed (8*N bytes copied, 1.5 MiB at
2x128x768). Foreign Python/JavaScript safety copies remain. The tape still has
four f32 vectors and retains any supplied spare capacity. This is neither
zero-copy transport nor a measured whole-process peak-memory improvement.

## Measurements

Apple M4, CPU f32, Rust 1.97 release, Torch 2.12.1, two Torch threads, seed 239.
All routes request input and broadcast-gate gradients. Four process pairs per
condition use AB / BA / BA / AB, two warmups and twelve rotated route rounds.
The plan was written before measurement. All 32 reports and 1,536 timing
samples are included, without pilot selection or a timing threshold.

Medians of four process medians, milliseconds:

| Shape | Iterations | Captured before | Captured after | Recomputed before | Recomputed after | Torch before | Torch after |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2 x 16 x 64 | 4 | 0.067334 | 0.068396 | 0.088198 | 0.086083 | 0.213594 | 0.217094 |
| 2 x 128 x 768 | 4 | 5.184740 | 5.167407 | 6.820282 | 6.681834 | 4.216104 | 4.273542 |
| 2 x 16 x 64 | 16 | 0.122333 | 0.121167 | 0.253355 | 0.258594 | 0.792844 | 0.800604 |
| 2 x 128 x 768 | 16 | 19.702375 | 19.568427 | 31.669635 | 31.618323 | 18.467469 | 18.574136 |

Coupling is 0.25 for four iterations and 0.75 for sixteen; saturation 1.0,
porosity 0.2. Captured timings are nearly neutral: large medians decrease about
0.3-0.7%, while small four-iteration capture increases about 1.6%. Several
individual pairs regress, including two of four large four-iteration pairs.
There is no demonstrated general speedup or confidence interval. List routes
and every pair remain in the archive. Large captured inputs remain slower than
the independent Torch reference; Rust also retains its audit work. Do not
compare absolute times across earlier studies as if they were paired runs.

Every old/new Rust output, input-VJP and reduced gate-VJP hash matches exactly
across all three native routes. Torch uses rtol=5e-4 / atol=3e-5, checked outside
timing. The benchmark client is unchanged. No browser, WebGPU, accelerator,
full-model throughput or model-quality claim is made.

## Verification

- Rust: all 1,022 st-core tests and eleven Topos NN/trainer tests pass. Fourteen
  core Topos tests include pointer-and-capacity identity for owned inputs,
  borrowed/owned audit and VJP equality, repeated VJPs, surviving operator/tape
  destruction, and identical errors followed by successful operator reuse.
- Python: 1,038 geometry/study tests and 120 existing benchmark/receipt tests
  pass. Aliased list and bulk-buffer inputs remain safe after mutation, failed
  capture, resizing and object destruction. The expanded publication tests
  verify this bundle too; the combined tool suite passes 121 tests.
- Native and NN-enabled wasm32 release builds, pinned workspace formatting,
  and Rust 1.99 scoped st-core wasm32 Clippy pass. Eighteen existing Torch JIT
  deprecations and 158 existing vendored WGPU Clippy warnings remain.
- Actual Node-hosted scalar WASM preserves 24 output/VJP/audit cases, 240 SGD
  updates and saved-gate continuation. Fifty-four rejection guards and two
  alias/overlap ownership cases pass. Source and returned-output mutation,
  failed capture, kernel disposal and repeated VJPs leave the tape independent.
- Both frozen 73-file packages were rehashed. Only the native library differs.
  No previous study, weight, corpus, score or result bundle was rewritten.

The initial receipt comparison rejected an extra generated-wrapper hash change,
not a numerical mismatch. Export order changed. Both artifact identities were
checked against their bytes; all non-identity receipt fields match exactly.
Both Topos wrapper classes and their TypeScript declarations are byte-identical;
the full TypeScript line multisets match. Initial passed probe receipts remain
in `verification.json.gz`, alongside fresh ownership probes of both binaries.

## Identity And Reproduction

Candidate runtime source: `8e1071e1e1caf1bfcb4be6dd794e601ff3eadde0`.
Baseline runtime source: `e03719056adc6f34ccd7972d4819b849164adf93`.
The expanded ownership probe and publication tests are subsequent test-only
changes; their client hashes are recorded separately.

`measurements.json.gz` preserves the original plan and numeric report JSON
strings, including all timings. `verification.json.gz` binds packages, clients,
reports, WASM receipts and local logs by hash. These records do not independently
prove private execution history or pre-run chronology. `SHA256SUMS` closes this
directory. Binaries and full logs remain local; no cleanup was performed.

Build separate packages from the pinned revisions, preserving existing ones:

```bash
cargo test --locked --offline --release -p st-core --lib
cargo test --locked --offline --release -p st-nn --lib topos
cargo build --locked --offline --release -p spiraltorch-py
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
export SPIRALTON_MAGIC=0 SPIRALTON_TORCH=0 SPIRALTON_MODEL_PATCHES=0 SPIRALTON_NUMPY=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
PYTHONPATH="$PACKAGE_ROOT" "$PYTHON" -P -B tools/benchmark_topos_learning.py \
  --shape 2 128 768 --iterations 4 --coupling .25 --native-profile release \
  --output "$NEW_REPORT"
cargo build --locked --offline --release -p spiraltorch-wasm \
  --target wasm32-unknown-unknown --features nn
wasm-bindgen --target web --out-dir "$NEW_MODULE_DIR" \
  target/wasm32-unknown-unknown/release/spiraltorch_wasm.wasm
node tools/probe_topos_capture_wasm.mjs "$NEW_MODULE_DIR" "$NEW_WASM_REPORT" \
  --require-capture
cargo +1.99.0 clippy --locked --offline -p st-core \
  --target wasm32-unknown-unknown --all-targets --no-default-features \
  --features cpu,wgpu-rt -- -D warnings
python -I -B -m pytest --import-mode=importlib --confcutdir=tools --rootdir=tools \
  -q tools/test_topos_evolution_results.py
```

Use wasm-bindgen 0.2.104 and the archived plan for every shape, coupling, phase
and process position. Timings are observations, never a CI pass/fail criterion.
Torch remains a comparison client, not a second production semantic owner.
