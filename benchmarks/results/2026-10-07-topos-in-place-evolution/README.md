# Topos In-Place Evolution

The Rust Picard loop now updates independent stalks in place, retaining the
iteration-major order and all finite guards. The final-update L-infinity audit
is computed only on the final iteration; closure adjustments and rewrite counts
still include every iteration. No formula, f32 association, derivative, budget,
convergence policy or error traversal changes.

This removes one N-element f32 temporary (4*N bytes, 768 KiB at 2x128x768).
The captured tape still retains four f32 vectors, 16*N bytes. This is not a
claim about whole-process peak memory, zero-copy transport or GPU residency.

## Matched Measurements

Apple M4, CPU f32, Rust 1.97 release, Torch 2.12.1, two Torch threads, seed 239.
All routes request both input and broadcast-gate gradients. The plan was written
before measurement: four process pairs per condition, AB / BA / BA / AB, two
warmups and twelve cyclically rotated route rounds per process. All 32 processes
and 1,536 timing samples are published. No pilot selection or timing threshold.

Medians of four process medians, milliseconds:

| Shape | Iterations | Captured before | Captured after | Recomputed before | Recomputed after | Torch before | Torch after |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2 x 16 x 64 | 4 | 0.072698 | 0.070573 | 0.093073 | 0.095313 | 0.231854 | 0.229563 |
| 2 x 128 x 768 | 4 | 6.289614 | 5.596094 | 7.549313 | 7.429469 | 4.679198 | 4.678979 |
| 2 x 16 x 64 | 16 | 0.138136 | 0.129053 | 0.271854 | 0.270386 | 0.853188 | 0.849375 |
| 2 x 128 x 768 | 16 | 23.541636 | 20.817865 | 34.556063 | 34.126688 | 19.288021 | 19.247959 |

Coupling is 0.25 for four iterations and 0.75 for sixteen; porosity 0.2 and
saturation 1.0. Large captured medians fall about 11-12%; all four paired
captured comparisons improve in each large condition. The recomputed route is
nearly unchanged, and small four-iteration bulk regresses. List-route costs and
negative individual pairs are also retained in `summary.json` and the raw
reports. There is no general speedup claim or confidence interval. Large
captured inputs remain slower than the matching Torch reference, and Rust
retains its extra audit work. This is not full-model throughput or quality.

Every old/new Rust output, input-VJP and reduced gate-VJP hash is identical
across list, recomputed bulk and captured routes. The independent Torch formula
uses rtol=5e-4 / atol=3e-5, checked outside each timed interval. The benchmark
client is unchanged from the preceding inlining study.

## Verification

- All 1,020 release st-core unit tests pass, including twelve Topos tests.
  A frozen double-buffered reference checks
  526 finite configurations in both captured and noncaptured modes, including
  signed zero, subnormals, saturation/porosity boundaries and 4,096 iterations.
  Output, sensitivity and every audit field are exact; 20 additional invalid
  configurations per mode preserve error order, including opposite-sign
  overflows occurring in different iterations. Finite cases must succeed, not
  merely agree on rejection.
- Eleven NN-layer, trainer and sequencer Topos tests pass. The new native
  package passes 1,036 Python geometry/study tests and 117 benchmark/receipt
  tests; 18 existing Torch JIT deprecation warnings remain. Three additional
  tests reconstruct this publication; all 120 benchmark/receipt tests also pass
  together.
- Native and NN-enabled wasm32 release builds pass. Rust 1.99 scoped st-core
  wasm32 Clippy with cpu,wgpu-rt and denied warnings passes; 158 existing
  vendored WGPU warnings remain. This is not a workspace-wide lint claim.
- Node-hosted scalar WASM: both binaries preserve all output/VJP hashes and
  captured-audit JSON hashes across 24 cases. The same 240-update synthetic SGD
  trajectory, saved-gate next update and 50 rejection guards pass. The shared
  probe now records audit hashes and verifies snapshot independence. No
  browser, WebGPU or WASM timing claim is made.
- All 73 files in both frozen runtime packages retain their hashes. Only the
  native library differs between packages. No pretrained study, heldout score,
  private weight or previous result bundle was rerun or rewritten.

The first JS-generation invocation could not find `wasm-bindgen` on PATH; the
already-installed cached 0.2.104 CLI succeeded. An initial whole-receipt equality
check then rejected the expected changed WASM binary digest, not a numerical
difference. Both binary identities were checked against their bytes and every
other receipt field compared exactly. The initial records remain in
`verification.json.gz`; the expanded audit probe was run afresh on both binaries.

## Identity And Reproduction

Candidate runtime source: `e03719056adc6f34ccd7972d4819b849164adf93`.
Baseline runtime source: `9c37f618896ef985b4befde0ec7fb2da5366bf12`.
Final audited WASM probe source: `87e38c3d` (runtime unchanged).

`measurements.json.gz` preserves the original plan JSON and all original numeric
report JSON strings, so their byte hashes can be independently reconstructed.
`verification.json.gz` binds both packages, clients, reports, WASM receipts and
private full logs. It does not independently prove private execution history or
pre-run chronology. `SHA256SUMS` closes this public directory. Binaries and raw
logs stay local; no cleanup was performed.

Build separate packages from the pinned sources without replacing old packages:

```bash
cargo test --locked --offline --release -p st-core --lib topos_resonator
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

Use the archived plan for every shape, coupling, phase and process position;
do not substitute a one-process timing for the paired matrix. No timing is a CI
pass/fail criterion. Pure Torch remains a comparison client, not a second
production owner of the geometry.
