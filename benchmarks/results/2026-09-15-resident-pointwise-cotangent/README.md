# Direct Pointwise Cotangents: Matched Route Comparison

Measured source: `4f78325a88fffeb442e04aff00b19a2cf2227f66`, tree
`0dd74b56bd4858291f7ff8e806f2320d0eafaca4`. The later evidence commit does
not change this measured runtime. See the
[API contract](../../../docs/resident_pointwise_cotangent.md).

## Result

The direct route is numerically consistent with the materialized route and the
recorded independent Torch reference in this bounded suite. It is **not a
consistent speed win**. Ratios below are geometric means of 18 case medians
(three shapes/depths, three seeds, two rounds), materialized time / direct time;
greater than one favors direct. These are correlated diagnostic samples, not
confidence intervals or GPU kernel times.

| Optimizer | Native Immediate | Native Deferred | Browser Immediate | Browser Deferred |
| --- | ---: | ---: | ---: | ---: |
| Plain | 0.993 | 1.016 | 1.010 | 1.022 |
| Topos EMA | 1.024 | 1.009 | 1.008 | 0.999 |
| Clipped Topos EMA | 1.013 | 1.014 | 0.994 | 1.008 |

Only 134 of 216 case/cadence ratios favor direct. Individual ratios range from
0.682 to 1.365; both extremes are native/plain/deferred, shape `[2,16,32]`,
depth 2, round B, with seeds 29 and 17 respectively. No slow condition was
discarded or selectively rerun. The eager Torch MPS control's time / direct time
ranges from 0.232 to 1.224, geometric mean 0.650 across the 108 native ratios:
Torch has lower latency on average here, but does not implement equivalent
finite guards, owning versions or atomic rollback. This is not a fastest-Torch
or an equivalent-runtime comparison.

All 54 recipes passed saved-state validation. Maximum saved-state absolute
error against the recorded independent Torch reference was
`3.259629011154175e-8`. All 108 native/browser case records have matching
baseline/candidate fingerprints. Fingerprints are consistency receipts, not
independent numerical replay or physical-device attestation.

## Design And Verification

Both Rust lanes use the **same frozen executable or WASM module** and identical
fused quadratic/quartic pointwise programs. The baseline materializes each seed
then calls `backward`; the candidate evaluates it into the VJP tape through
`backward_pointwise`. Two independent VJPs, weights 0.75/0.25, SGD rate 0.01,
owning results and update acceptance remain unchanged. Direct routing avoids
one seed-value allocation/copy and one submission per VJP, not all graph copies.

- Shapes/depths: `[2,16,32]`/2, `[2,129,32]`/4, `[4,32,64]`/8; seeds 17, 29, 43.
- Plain, Topos EMA (damping 0.5), and clipped Topos EMA (max norm 1/1024).
- Each interval resets weights and runs eight nonzero updates. Immediate and
  deferred receipt cadence each retain eight blocks after two warmups. Lane
  order rotates within each cadence; full initial/terminal capture is outside
  timing, and other intervals retain consistency fingerprints.
- Round A runs plain/EMA/clipped, native then browser. Round B reverses both
  optimizer and client order. All 5,400 intervals are published, including
  warmups; 4,320 are retained for timing. All 2,160 browser intervals were
  revalidated against progress records.
- macOS/aarch64, native device admission Apple M4, Torch 2.12.1 MPS with fallback
  disabled; Chrome 152.0.7977.84 WebGPU. Its separate adapter probe reports Apple
  metal-3, not fallback, but the Rust browser workspaces do not expose an exact
  physical adapter identity. Owned GPU jobs ran serially; host exclusivity is
  UNKNOWN. Complete tool commands, build identity and adapter boundaries are
  retained in the receipts and validation records.

At the measured source, all 28 verification stages passed: native and wasm32
backend strict Clippy; 156 backend tests plus one WGSL syntax test; 769 NN tests;
GPU and CPU-only Python builds with 37 tests each (GPU: zero skips; CPU: nine
expected GPU skips); 21 benchmark admission tests; formatting; native and WASM
benchmark builds; and all three isolated browser client pages. Regression tests
cover strided/broadcast inputs, stale/foreign tokens, layout/device rejection,
masked overflow, immediate recovery after dropping an invalid input slot,
owning gradient lifetime and zero-weight-invalid-source rollback.

An earlier verification attempt stopped at a nonexistent Python test path in
the external runner. Its receipt and error log are preserved in
`earlier-attempt/`; the corrected complete run is `verification/`. This was a
harness path failure, not a product numerical failure. There was one complete
measurement invocation, with two prescribed rounds and no selective retries.

## Published And Local Evidence

`intervals.json` retains every condition, order, route flag, timing and case
fingerprint without raw tensor captures. `summary.json` contains all 216 ratios
and group aggregation; `validation/` retains all six numerical validation
records. Verification logs, browser reports, exact runner scripts and the
frozen-extension Python loader are included. The package is hash-bound by
`manifest.json`.

The 123 raw files (1,312,839,545 bytes), including full tensor captures,
progress records and frozen native/Python/WASM products, remain local under
`~/Library/Logs/SpiralTorch/resident-pointwise-cotangent-20260915`.
`raw-inventory.json` records every path, size and SHA-256. Existing published
archives are untouched. This follows the explicit results-and-validation-only
publication policy for new measurements.

From the repository root, verify published bytes, source/product links, the
complete grid, route labels and reaggregated timings without GPU dependencies:

```sh
python3 -S benchmarks/results/2026-09-15-resident-pointwise-cotangent/verify.py
```

To also verify the retained raw bytes on the originating machine:

```sh
python3 -S benchmarks/results/2026-09-15-resident-pointwise-cotangent/verify.py \
  --raw-root "$HOME/Library/Logs/SpiralTorch/resident-pointwise-cotangent-20260915"
```

Neither command reruns numerics or GPU work. Public hashes cannot reconstruct
the omitted raw tensors. A numerical replay needs those raw files with
`tools/validate_resident_training_bench.py`, or a new live run. Use a clean
checkout of the measured commit, source-bind all workers, and place new outputs
outside that checkout without overwriting this archive.

For a new native plain run, build with Rust 1.98.0 and use the same worker in
both lanes; `TORCH_PYTHON` must import Torch with working MPS, and `OUT` must be
a new output path:

```sh
cargo +1.98.0 build --locked --release -p st-nn --features wgpu \
  --example resident_training_bench
SOURCE=4f78325a88fffeb442e04aff00b19a2cf2227f66
WORKER="${CARGO_TARGET_DIR:-target}/release/examples/resident_training_bench"
PYTORCH_ENABLE_MPS_FALLBACK=0 "$TORCH_PYTHON" -I \
  tools/bench_resident_training_vs_torch.py \
  --baseline "$WORKER" --baseline-source "$SOURCE" \
  --candidate "$WORKER" --candidate-source "$SOURCE" \
  --device mps --graph --learner --direct-learner-seeds --output "$OUT"
```

For the browser, build the `resident_training_bench_browser` example for
`wasm32-unknown-unknown` with the same features and generate web bindings using
wasm-bindgen 0.2.104. Set `MODULE` to that output, `CHROME` to the browser binary,
and `NODE_PATH` to the installed Playwright dependency. The matching plain run:

```sh
node tools/bench_resident_training_browser.cjs \
  "$MODULE" "$MODULE" "$CHROME" "$OUT" \
  learner standard direct-learner-seeds none
```

The exact full-suite commands and serial two-round schedule are recorded in
`reproduction/verify.py` and `reproduction/measure.py`. Their top-level paths
are machine-specific: adapt ROOT, runtime/dependency paths, and the output
directory before running them elsewhere, and point LOADER at the included
`python-client.py`. Preserve clean source checks, fallback disabling, source
binding, product freezing and complete failure retention. The original
`reproduction/package.py` records extraction, not numerical validation.

This is explicit seed evaluation, not differentiation of an arbitrary objective,
automatic `pure::Tensor`/ModuleTrainer/HF migration, uninterrupted FT, or a
learning-quality/CUDA claim. Forward prediction capture, input-to-tape copies,
shared intermediate gradient scratch and acceptance receipts still exist.
