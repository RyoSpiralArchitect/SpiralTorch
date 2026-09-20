# Direct Owning Prediction Destinations

Measured candidate: `2a50f54aad83b5095047f65907327ec58b088b65`, tree
`0d673543ce52c771d93a510894c46ebc3a106c3c`.
Frozen baseline: `4f78325a88fffeb442e04aff00b19a2cf2227f66`, tree
`0dd74b56bd4858291f7ff8e806f2320d0eafaca4`.
See the [resident autograd contract](../../../docs/resident_graph_autograd.md).

The later `e4dc7493de5e08bf3e29f19d4fd6be826c9efb38` follow-up only restricts
a test-only capture helper to native tests. Its complete verification is
separate; timings remain pinned to the measured candidate, not relabeled as
results from that newer build. `test-cfg-followup.json` records the exact diff
and source-file hashes. The evidence commit does not change runtime code.

## Result

Direct prediction destinations preserve the recorded outputs, VJPs and updates
in this bounded suite. They are **not a consistent end-to-end speed win**.
Ratios below are geometric means of 18 case medians (three shapes/depths,
three seeds, two rounds), baseline time / candidate time. Greater than one
favors the candidate. These are correlated diagnostic samples, not confidence
intervals or GPU kernel timings.

| Optimizer | Native Immediate | Native Deferred | Browser Immediate | Browser Deferred |
| --- | ---: | ---: | ---: | ---: |
| Plain | 1.023 | 1.002 | 1.008 | 0.942 |
| Topos EMA | 0.972 | 0.974 | 1.015 | 0.996 |
| Clipped Topos EMA | 1.039 | 1.005 | 0.966 | 1.013 |

130 of 216 case/cadence ratios favor the candidate. Individual ratios range
from 0.476 (native Topos EMA, immediate, `[2,129,32]`, depth 4, seed 43,
round B) to 1.394 (native Topos EMA, deferred, `[4,32,64]`, depth 8, seed 43,
round B). Slow conditions and outliers were not discarded or selectively
rerun. This experiment does not identify their causes.

The eager Torch MPS control's time / candidate time ranges from 0.161 to
1.074, geometric mean 0.661 across the 108 native ratios. Torch has lower
latency on average here, but lacks equivalent finite guards, owning versions
and atomic rollback. This is not an equivalent-runtime or fastest-Torch
comparison.

All 54 recipes passed saved-state validation. Maximum absolute error against
the recorded independent Torch reference was `3.259629011154175e-8`.
All 108 native/browser case records have matching baseline/candidate
fingerprints. Fingerprints are consistency receipts, not independent
numerical replay or physical-device attestation.

## Design And Verification

The terminal dense/GELU or pointwise producer writes directly into an owning
prediction version. This removes the subsequent full-value identity
copy/check dispatch; a guard-only dispatch still freezes every upstream
failure. A bounded pool reuses terminal bindings and unheld versions. Retained
tensors, views, weak owners and resident inputs prevent reuse. Busy or oversized
versions spill without waiting. Existing Python/WASM `forward` methods share
this Rust implementation without client-side heuristics or new switches.

Prediction and VJP output pools each retain at most four versions / 32 MiB of
value storage, separately. Held outputs can consume additional memory, and
these budgets exclude binding/tape resources. Intermediate activations and
preactivations, input packing/copies, the old terminal scratch allocation,
guard dispatches and update receipt reads remain. Ordinary MSE/SGD and
forward-only workspaces retain their existing path.

The lanes use **different frozen, source-bound runtimes** and byte-identical
benchmark drivers/fixtures. Both use ordinary unfused quadratic/quartic seed
evaluation, two exact VJPs, weights 0.75/0.25, and SGD rate 0.01. This is not the
earlier materialized-versus-direct-pointwise-seed experiment.

- Shapes/depths: `[2,16,32]`/2, `[2,129,32]`/4, `[4,32,64]`/8; seeds 17, 29, 43.
- Plain, Topos EMA (damping 0.5), and clipped Topos EMA (max norm 1/1024).
- Each interval resets initial weights and runs eight nonzero updates.
  Immediate and deferred receipt cadence each retain eight blocks after two
  warmups. Lane order rotates; full initial/terminal captures are outside
  timing, and other intervals retain consistency fingerprints.
- Round A runs plain/EMA/clipped, native then browser. Round B reverses both
  optimizer and client order. All 5,400 intervals are published, including
  warmups; 4,320 are retained. All 2,160 browser intervals were revalidated
  against progress records.
- macOS/aarch64, native device admission Apple M4/Metal, Torch 2.12.1 MPS with
  fallback disabled; Chrome 153.0.8010.48 WebGPU. Its separate adapter probe
  reports Apple metal-3, not fallback, but does not attest the exact physical
  adapter of the Rust browser workspaces. Owned GPU jobs ran serially;
  host exclusivity is UNKNOWN.

Both clean-source verification runs passed all 28 stages: strict native and
wasm32 backend Clippy; 160 backend tests plus one WGSL syntax test; 769 NN
tests; GPU and CPU-only Python builds with 38 tests each (GPU: zero skips;
CPU: ten expected GPU skips); 21 benchmark admission tests; formatting;
native/WASM benchmark builds; and all three isolated browser client pages.

Regression tests poison the old terminal scratch, compare output/VJP against
the unchanged scratch encoder, and cover dense/GELU/pointwise/mixed graphs,
strided N-D inputs, retained views, delayed failed snapshots, recycled guards,
bounded retention, previous predictions used as inputs, parameter updates
and workspace destruction.

The measured-source local wasm Clippy check covered only `--lib`. Hosted CI
also compiled test targets and correctly caught an unused native-test helper
in wasm lib tests. The one-line test-target fix is preserved separately;
`verification-final/` checks wasm **`--all-targets`**, without suppressing
warnings. `preflight/` retains the earlier large-enum lint failure/correction
and the hosted wasm failure record. There was one complete measurement
invocation, two prescribed rounds, and no selective performance retries.
The publication verifier also initially assumed the previous experiment's
optional route-label map existed; ordinary unfused records omit it. That
schema correction is recorded in `preflight/`, and its tamper test now rejects
an injected route map. No measurements or numerical results were changed.

## Evidence And Rerun

`intervals.json` retains every condition, order, route flag, timing and case
fingerprint without raw tensor captures. `summary.json` contains all 216
ratios and group aggregation. `validation/` retains all six numerical
validation records; both verification runs include logs, browser reports and
product receipts. `manifest.json` binds the published bytes.

Raw tensors, progress records and frozen native/Python/WASM products remain
local under `~/Library/Logs/SpiralTorch/resident-direct-prediction-20260921`;
`raw-inventory.json` gives every path, size and SHA-256. The earlier baseline's
products remain under
`~/Library/Logs/SpiralTorch/resident-pointwise-cotangent-20260915/verified-b`;
`baseline-inventory.json` binds them separately. Existing published archives
are untouched. New publication contains results and validation, not omitted
raw payloads or binaries.

From the repository root, check published bytes, source/product links, both
verification receipts, the complete grid, route labels and timing aggregation:

```sh
python3 -S benchmarks/results/2026-09-21-resident-direct-prediction/verify.py
python3 -S benchmarks/results/2026-09-21-resident-direct-prediction/test_verify.py
```

On the originating machine, also check all retained raw/baseline bytes:

```sh
python3 -S benchmarks/results/2026-09-21-resident-direct-prediction/verify.py \
  --raw-root "$HOME/Library/Logs/SpiralTorch/resident-direct-prediction-20260921" \
  --baseline-root "$HOME/Library/Logs/SpiralTorch/resident-pointwise-cotangent-20260915/verified-b"
```

These commands do not replay GPU work or numerics. Public hashes cannot
reconstruct omitted tensors. Numerical replay needs those raw files and
`tools/validate_resident_training_bench.py`, or a new live run.

For a new comparison, build `resident_training_bench` with Rust 1.98.0,
`--locked --release -p st-nn --features wgpu --example resident_training_bench`
at **each** pinned clean source in separate build directories. Freeze both
executables. From the measured candidate checkout, set `BASE_WORKER` and
`CANDIDATE_WORKER` to those binaries, `TORCH_PYTHON` to a Python with working
Torch MPS, and `OUT` to a new output path outside the checkout:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 "$TORCH_PYTHON" -I \
  tools/bench_resident_training_vs_torch.py \
  --baseline "$BASE_WORKER" \
  --baseline-source 4f78325a88fffeb442e04aff00b19a2cf2227f66 \
  --candidate "$CANDIDATE_WORKER" \
  --candidate-source 2a50f54aad83b5095047f65907327ec58b088b65 \
  --device mps --graph --learner --output "$OUT"
```

Do not add `--direct-learner-seeds`: that would change the experiment. For
each source, also build `resident_training_bench_browser` for
`wasm32-unknown-unknown` and generate web bindings with wasm-bindgen 0.2.104.
Set `BASE_MODULE` and `CANDIDATE_MODULE` to those separate output directories,
`CHROME` to the browser binary, and `NODE_PATH` to Playwright. The matching
plain browser run is:

```sh
node tools/bench_resident_training_browser.cjs \
  "$BASE_MODULE" "$CANDIDATE_MODULE" "$CHROME" "$OUT" \
  learner standard none none
```

`reproduction/measure.py` records the complete two-round schedule and
cross-runtime validation commands. `verify.py` and `verify-final.py` in that
directory record the separate clean-source verification runs. These exact
runners have machine-specific paths: adapt ROOT, build/dependency paths and
output directories, and point LOADER at the included `python-client.py`.
Preserve clean-source admission, source binding, frozen products, serial GPU
work, fallback disabling and failure retention. The copied `package.py`
performs extraction, not numerical validation.

This does not migrate `pure::Tensor`, ModuleTrainer or HF training, and makes
no CUDA, uninterrupted-FT, learning-quality or general GPU-speedup claim.
