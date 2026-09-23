# Adaptive resident LayerNorm affine workgroups

Native Apple M4 / Metal exploratory comparison on macOS 26.4.1. Baseline
source is `b9b23501123abfa7179bc969cdd0291c3cddb5d0`; optimized backend
source is `10fe5b0ae4a6464684f5e3fd9dabc012327d1442`. The browser fixture
is committed at `7f567a0412bbc85b40e9e48bb9c6b80f4106713a`.
Each release binary was retained locally and hashed: baseline
`00dd24f16c8c44ed507ae4be5f6a49006f07edc21bfb4ef84e1547c546d9f555`,
candidate `6c55c0c0034335e649c08233006ed2bcaa2a82b4dabfa893c83dd074cfc5730b`.
Only 33-64 rows select 64 lanes and 65-128 rows select 128 lanes; other
shapes retain the original 256-lane affine shader. The CPU route and public
tensor/autograd contracts are unchanged.

Three sequential, counterbalanced pairs ran candidate/baseline,
baseline/candidate, candidate/baseline. Each report contains six shapes,
three warmups, nine measured intervals per route and stage, and 32 training
steps per timed interval. Each table cell is the median of the three run
medians in milliseconds. All six raw native reports remain local; the six
`validated-*.json` files publish every condition's route/stage medians,
numeric validation (including matching input/target digests), and raw-report
SHA-256 hashes. The input/target digests themselves remain in the raw reports.
The timed intervals exclude first-use shader/pipeline creation; startup
latency was not compared.

| Shape | All-VJP baseline | All-VJP candidate | Affine-only baseline | Affine-only candidate |
| --- | ---: | ---: | ---: | ---: |
| 2x3 | 27.033 | 26.895 | 25.048 | 24.954 |
| 8x257 | 42.800 | 42.471 | 26.187 | 26.235 |
| 32x256 | 50.925 | 50.877 | 29.814 | 29.855 |
| 64x768 | 207.647 | 192.023 | 116.341 | 102.107 |
| 128x1025 | 529.678 | 487.708 | 274.179 | 230.001 |
| 256x256 | 268.092 | 268.050 | 131.235 | 131.088 |

These 32-step times use the preloaded-input route and include one terminal
readback. The affine-only route omits dx and is not work-matched to the
all-VJP route. At 64x768 and 128x1025, the candidate's all-VJP runs were
about 7.5% and 7.9% shorter; affine-only runs were about 12.2% and 16.1%
shorter. Every measured pair at these two shapes moved in the same direction.
The other four shapes did not show a material end-to-end change.

| Shape | Affine stage baseline | Affine stage candidate | Input VJP baseline | Input VJP candidate |
| --- | ---: | ---: | ---: | ---: |
| 2x3 | 0.370 | 0.369 | 0.499 | 0.495 |
| 8x257 | 0.689 | 0.699 | 0.681 | 0.697 |
| 32x256 | 0.827 | 0.832 | 0.704 | 0.717 |
| 64x768 | 2.664 | 1.901 | 2.676 | 2.676 |
| 128x1025 | 4.603 | 3.243 | 8.376 | 8.208 |
| 256x256 | 1.933 | 1.894 | 3.162 | 3.110 |

The seven stage probes each include one terminal snapshot and are not
additive parts of the 32-step run. The unchanged input-VJP stage and the
256-row route are diagnostic controls. Across the six native reports,
1,296 training intervals and 2,268 stage intervals were present. Every
report passed the public Rust/PyTorch numerical gate, including matched f32
input/target digests, decreasing loss, and all final outputs. The largest
scaled error was 0.000272 of the allowed
`5e-4 * (1 + abs(reference))` bound. The three PyTorch 2.12.1 CPU/MPS
control reports from the preceding fixed-input benchmark were reused for
**numerical validation only**; no new PyTorch speed comparison is claimed.

Chrome 153 on a non-fallback Apple WebGPU adapter passed the v5 browser
fixture, including two new 64/128-row cases with three affine requested masks
each, all seven prior cases, 400 learning steps, guards and grouped readback.
`browser.json` records the adapter probe, zero page/console errors, and the
WASM/JS/page hashes. The probe is separate from the Rust runtime adapter and
is not an attestation that they are the same device.

Verify the published artifacts from this directory with
`shasum -a 256 -c SHA256SUMS`. The raw native reports and PyTorch controls
are intentionally not committed; each published validation JSON records
their SHA-256 hashes for full local replay.

Replay from separate clean checkouts at the two source commits above. In each
checkout, build the release example with the command from
`benchmarks/layer-norm-training/README.md` (using `cargo build` in place of
`cargo run`). Set `BASE` and `CAND` to those checkout roots, and `RAW` and
`TORCH` to existing local result directories. Run the two binaries
sequentially, never concurrently. Generate the numerical control once per
pair:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 python3 -I -B \
  "$BASE/benchmarks/layer-norm-training/torch_bench.py" \
  > "$TORCH/torch-b9b23501-run1.json"
```

The first native pair is:

```sh
SPIRALTORCH_STRICT_GPU=1 "$CAND/target/release/examples/layer_norm_training_residency_bench" \
  > "$RAW/fixed-candidate-run1.json"
SPIRALTORCH_STRICT_GPU=1 "$BASE/target/release/examples/layer_norm_training_residency_bench" \
  > "$RAW/fixed-baseline-run1.json"
```

Reverse that order for run 2, then use candidate/baseline again for run 3.
For each pair, validate both raw reports against the matching baseline
PyTorch report:

```sh
python3 -I -B "$CAND/benchmarks/layer-norm-training/compare.py" \
  "$RAW/fixed-baseline-run1.json" "$TORCH/torch-b9b23501-run1.json"
python3 -I -B "$CAND/benchmarks/layer-norm-training/compare.py" \
  "$RAW/fixed-candidate-run1.json" "$TORCH/torch-b9b23501-run1.json"
```

Repeat for runs 2 and 3. Rebuild the browser example for
`wasm32-unknown-unknown`, bind it with `wasm-bindgen 0.2.104 --target web`,
then use `tools/test_resident_browser.cjs` with fixture
`layer-norm-resident` and an isolated Chrome instance. This is a local
shared-machine improvement for these shapes, not a universal GPU speed claim
or a change to default Tensor dispatch.
