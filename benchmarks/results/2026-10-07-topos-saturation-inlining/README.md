# Topos Cross-Crate Inlining

Four `#[inline]` hints expose the existing canonical saturation and slope to
cross-crate callers. No formula, threshold, association, iteration order,
validation, audit or learning contract changes. The Python package is identical
except for its native library. This is a small CPU optimization, not a new
geometry or a claim of general superiority over PyTorch.

## Matched Measurements

Apple M4, CPU float32, Rust 1.97.0 release, Torch 2.12.1, two Torch threads,
seed 239, porosity 0.2, saturation 1.0. Every route requests both input and
broadcast-gate gradients, including host transport and Torch gate reduction.
Each fresh process has two warmups and twelve cyclically rotated route rounds.
Each condition has four before/after pairs; process order was frozen before
the paired run as AB, BA, BA, AB. All runs are sequential, not concurrent.

Values below are medians of four process medians, in milliseconds. The full
samples and per-pair ratios are retained, not just this aggregation.

| Shape | Iterations / coupling | Captured before | Captured after | Recomputed before | Recomputed after | Torch before | Torch after |
|---|---|---:|---:|---:|---:|---:|---:|
| 2 x 16 x 64 | 4 / 0.25 | 0.080865 | 0.070614 | 0.105145 | 0.088406 | 0.217459 | 0.218344 |
| 2 x 128 x 768 | 4 / 0.25 | 6.492563 | 6.127573 | 8.331667 | 7.339458 | 4.629094 | 4.765073 |
| 2 x 16 x 64 | 16 / 0.75 | 0.172604 | 0.142261 | 0.324271 | 0.259406 | 0.800125 | 0.809104 |
| 2 x 128 x 768 | 16 / 0.75 | 23.053761 | 22.589333 | 37.999698 | 32.501427 | 18.751615 | 18.785041 |

Recomputed bulk medians improve by roughly 12-20%; large already-captured
inputs improve only about 2-6%. The large four-iteration paired speed ratios
are 0.869, 1.024, 1.096 and 1.124: one pair regresses. The small four-iteration
condition also has a negative pair. Companion Torch timings vary despite an
unchanged implementation. These medians are not confidence intervals, and
the large captured path remains slower than the matching Torch reference.
No browser, WebGPU, full-model throughput or model-quality gain is claimed.

`measurements.json.gz` contains all 45 reports: twelve baseline pilots, one
candidate screen, and the 32 primary paired runs. Only the latter enter the
table. The prewritten pairing plan is included. No unfavorable run was removed.
Every paired old/new Rust output and both gradient hashes match exactly,
including signed zero, across list, recomputed bulk and captured routes.
Independent Torch arithmetic is checked at rtol=5e-4 and atol=3e-5; all measured
errors remain in the reports. Native retains its additional audit work.

## Correctness Scope

- Rust: 84 tensor, 10 core and 7 NN Topos tests pass (101 total).
- Python: 987 geometry regression tests and 114 benchmark/replay tests pass.
  Three frozen-receipt tests check this archive without rerunning a study;
  all 117 benchmark/replay/receipt tests also pass together in the pinned package.
- Strict Rust 1.99.0 wasm32 Clippy passes for st-core with cpu,wgpu-rt and
  `-D warnings`. Existing vendor warnings remain; this is not workspace lint.
- NN-enabled scalar WASM in Node has identical before/after receipts for
  24 cases and 240 SGD updates, with 50 rejection guards and exact saved-gate
  continuation. Both runs require the capture API. This is not browser or
  WebGPU execution/performance evidence.
- One auxiliary GPT-2 replay resumes the frozen mixed-adapter cursor 2 and
  reproduces step 3, loss 4.093510150909424. All 12 raw gradient tensors,
  adapter parameters, Adam and RNG match; the base model remains unchanged.
  This is migration parity, not a rerun of the original study or heldout scoring.

An extra combined-test invocation initially used Python isolated mode without
installing the candidate package, so three imports failed against the unrelated
installed package. The corrected invocation explicitly selected the frozen
candidate package and passed all 117 tests. Both logs are retained and hashed;
the original successful 987/114-test runs were not replaced.

## Identity And Reproduction

Candidate source: `9c37f618896ef985b4befde0ec7fb2da5366bf12`.
Its parent is `fc2fb987cac9765bdf2c5f3e398f8e09ecf4197d`.
The frozen baseline runtime source is
`bfdbbc3035a4e67f179913933f53a6eae35b60c9`; subsequent parent changes affect
tests, evidence and CI only. `verification.json.gz` binds both 73-file packages,
the clients, original validation logs, WASM probes and single-update replay.
Only the native library differs between the frozen Python packages.
Raw logs, private weights and checkpoint/optimizer/RNG states remain local.

Build separate release packages from the pinned sources with Rust 1.97.0;
never overwrite a frozen package or original study directory. Run the candidate
benchmark client against both packages in the archived pair-plan order. Use
fresh report paths, the shapes/couplings above, and the same two-thread runtime:

```bash
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export SPIRALTON_MAGIC=0 SPIRALTON_TORCH=0 SPIRALTON_MODEL_PATCHES=0 SPIRALTON_NUMPY=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
cargo build --locked --offline --release -p spiraltorch-py
PYTHONPATH="$PACKAGE_ROOT" "$PYTHON" -P -B "$CLIENT/benchmark_topos_learning.py" \
  --shape 2 128 768 --iterations 4 --coupling .25 --native-profile release \
  --output "$NEW_REPORT"
cargo build --locked --offline --release -p spiraltorch-wasm \
  --target wasm32-unknown-unknown --features nn
wasm-bindgen --target web --out-dir "$NEW_MODULE_DIR" \
  target/wasm32-unknown-unknown/release/spiraltorch_wasm.wasm
node "$CLIENT/probe_topos_capture_wasm.mjs" "$NEW_MODULE_DIR" \
  "$NEW_WASM_REPORT" --require-capture
cargo +1.99.0 clippy --locked --offline -p st-core \
  --target wasm32-unknown-unknown --all-targets --no-default-features \
  --features cpu,wgpu-rt -- -D warnings
```

The replay uses the original privately retained mixed-adapter study and frozen
client, checking model, corpus, token and checkpoint identity before one update.
Missing or changed source artifacts fail closed; there is no substitute dataset:

```bash
PYTHONPATH="$PACKAGE_ROOT:$FROZEN_STACK_CLIENT" "$PYTHON" -P -B \
  "$CLIENT/replay_geometry_stack_update.py" --topos-capture \
  --previous "$ORIGINAL_STACK_RUN" --client-root "$FROZEN_STACK_CLIENT" \
  --model-dir "$MODEL_DIR" --corpus "$CORPUS" \
  --package-root "$PACKAGE_ROOT" --runtime-manifest "$RUNTIME_MANIFEST" \
  --output "$NEW_REPLAY_DIRECTORY"
python -I -B -m pytest --import-mode=importlib --confcutdir=tools --rootdir=tools \
  -q tools/test_topos_inlining_results.py
```

`SHA256SUMS` closes this directory. No original study was rewritten, no model
or corpus was downloaded, and no cleanup was performed.
