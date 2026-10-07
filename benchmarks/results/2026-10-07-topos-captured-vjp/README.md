# Topos Captured VJP: Matched CPU And WASM Evidence

This changes transport and finite-unroll reuse, not the geometric map, its
derivative, broadcast reduction or optimizer. The Rust-owned tape retains
input, per-element gate, output and drive sensitivity. Backward uploads only
the upstream direction; inference does not allocate a learning tape.

## Matched CPU Measurements

CPU float32, Torch 2.12.1, two Torch threads, seed 239, porosity 0.2, saturation
1.0. Every route computes forward and both input and broadcast-gate gradients.
Native builds are release. Two warmups and twelve cyclically rotated rounds per
process balance route positions. Each table entry is the median of three
process medians in milliseconds; all 24 reports are in `benchmark-runs.json.gz`.

| Shape | Iterations / coupling | Old public | New list | New bulk, recomputed | New captured | Torch reference |
|---|---|---:|---:|---:|---:|---:|
| 2 x 16 x 64 | 4 / 0.25 | 0.486313 | 0.488000 | 0.107520 | 0.082584 | 0.224583 |
| 2 x 128 x 768 | 4 / 0.25 | 48.956062 | 49.199875 | 8.694188 | 6.561104 | 4.807646 |
| 2 x 16 x 64 | 16 / 0.75 | 0.723521 | 0.724188 | 0.340000 | 0.175521 | 0.848375 |
| 2 x 128 x 768 | 16 / 0.75 | 74.931271 | 75.732042 | 37.675209 | 23.168999 | 18.128750 |

The large four-iteration case is about 7.46x faster than the old public bridge;
the new captured route is about 1.33x faster than the new recomputed bulk route.
At sixteen iterations, capture improves the new bulk control by about 1.63x.
This separates Python scalar-boxing removal from recurrence reuse. It does not
attribute the entire improvement to a faster Rust kernel.

Large inputs remain about 1.28-1.36x slower than the matching Torch reference;
small inputs favor the captured route. Torch implements the same finite Picard
map using ordinary differentiable operators, not a second production backend.
Native retains its extra audit work. Old/new native output and both gradient
hashes match exactly, including signed zero. Independent Torch arithmetic uses
rtol=5e-4, atol=3e-5, with observed errors in each report.

Before and after binaries were measured sequentially, not interleaved. These
are process medians, not confidence intervals or a general-library speed claim.
No accelerator, end-to-end model throughput or language-quality timing was
measured. The captured tape retains roughly 16*N bytes plus metadata, excluding
client tensors and temporary buffers (3 MiB for the large input).

## Correctness And Real Update Replay

- Rust: 10 core Topos tests and 7 NN Topos tests pass, including exact legacy
  output/audit/VJP parity, ownership and domain guards.
- Python: 983 geometry regression tests and 114 benchmark/replay tests pass.
  Four additional frozen-receipt checks validate this directory. Native tests
  cover NumPy absence, native-endian buffers, empty/noncontiguous/negative views,
  broadcast reduction, independent copies, concurrent VJPs and no-grad inference.
- Scalar WASM in Node: 24 conditions have identical old/new output and VJP
  hashes. Paired 240-step SGD trajectories are exact; MSE goes from
  0.13497054626500438 to 9.992007221626409e-15. Saved-gate continuation is exact;
  the new capture path rejects 50 invalid requests. This is not browser/WebGPU
  execution or performance evidence.
- One auxiliary pretrained GPT-2 update resumes the frozen mixed-adapter study
  at cursor 2 and reproduces step 3, loss 4.093510150909424. All 12 raw gradient
  tensors, adapter parameters, Adam and RNG match the saved expected state.
  The base model is unchanged; all four geometry families remain connected.
  Topos contributes only two forward uploads and one upstream upload. This is
  migration parity, not a new training study, heldout evaluation or quality claim.

The initial regression command accidentally selected no explicit test files
and collected unrelated historical benchmark scripts. Its four collection
errors are retained privately, hash-bound in `validation.json`. The corrected
bounded suite passed; no original artifacts were rewritten. Strict workspace
Clippy was not rerun for this revision, so no new strict-lint pass is claimed.

## Identity And Reproduction

Baseline branch revision: `88dbb8a429ffd86fe758dbfafd559faacaba7123` (#2222).
Its frozen runtime source is `ea0b954dd8f25e8d39846a42dc29737bb91cb685`;
the later baseline commit only adds evidence.
Candidate source: `bfdbbc3035a4e67f179913933f53a6eae35b60c9`.
`runtime-before-sha256.json` and `runtime-sha256.json` bind every file in the
73-file packages, including native binaries. Only the facade exports/stubs,
Topos client and native library changed. `validation.json` binds the benchmark,
WASM probe and replay client. Public receipts contain numeric results and hashes;
raw checkpoint/optimizer/RNG states and complete logs remain local.

Use the pinned runtime versions and Rust 1.97.0. Freeze separate package copies
from baseline and candidate; never overwrite the original study environment.
Build with `cargo build --locked --offline --release -p spiraltorch-py` and copy
the resulting native extension into each package. For each package, run the
candidate benchmark client three times for each shape/iteration/coupling pair
in the table, using fresh output names:

```bash
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export SPIRALTON_MAGIC=0 SPIRALTON_TORCH=0 SPIRALTON_MODEL_PATCHES=0 SPIRALTON_NUMPY=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
PYTHONPATH="$PACKAGE_ROOT" "$PYTHON" -P -B "$CLIENT/benchmark_topos_learning.py" \
  --shape 2 128 768 --iterations 4 --coupling .25 --native-profile release \
  --output "$NEW_REPORT"
cargo build --locked --offline --release -p spiraltorch-wasm \
  --target wasm32-unknown-unknown --features nn
wasm-bindgen --target web --out-dir "$NEW_MODULE_DIR" \
  target/wasm32-unknown-unknown/release/spiraltorch_wasm.wasm
node "$CLIENT/probe_topos_capture_wasm.mjs" "$NEW_MODULE_DIR" "$NEW_WASM_REPORT"
```

For the bounded replay, use the original mixed-adapter study's privately
retained midpoint, states, report and frozen client. The command validates the
source artifacts, model/corpus/token identity and runtime before the single
update. It fails closed if the private artifacts are unavailable or differ:

```bash
PYTHONPATH="$PACKAGE_ROOT:$FROZEN_STACK_CLIENT" "$PYTHON" -P -B \
  "$CLIENT/replay_geometry_stack_update.py" --topos-capture \
  --previous "$ORIGINAL_STACK_RUN" --client-root "$FROZEN_STACK_CLIENT" \
  --model-dir "$MODEL_DIR" --corpus "$CORPUS" \
  --package-root "$PACKAGE_ROOT" --runtime-manifest "$RUNTIME_MANIFEST" \
  --output "$NEW_REPLAY_DIRECTORY"
```

`SHA256SUMS` closes this public directory. No cleanup or model/corpus download
was performed, and the two previously user-deleted archive chunks are not part
of this feature's changes.
