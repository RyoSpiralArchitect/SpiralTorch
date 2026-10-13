# Fisher-Rao byte-corpus controls

This extends the [seven-arm calibrated study](../2026-10-10-byte-corpus-calibrated-controls-v4/README.md)
with categorical Fisher-Rao squared, trained and frozen. Rust owns preparation,
resident QKV learning and checkpoint restoration; Python supplies an independent
CPU-f32 Torch reference. The nine arms per seed do not contain a geometry mixture.

## Fixed Comparison

The [pre-registered recipe](pre-registered.json) was fixed before preparation or
learning. All seven previous arms remain unchanged, including fitted flat gains.
Fisher gains alone are fitted to the **same** original Poincare causal row-centered
bias RMS, separately for every block/head, using training batches 0, 1, 2, 3.
No held-out loss is used for fitting or choosing gains.

- Three seeds: 11, 23, 37; nine arms each; 128 SGD updates at 0.05.
- Batch 2, 32 byte positions, width 16, hidden width 32, two heads, one block.
- Geometry width 4, curvature -0.75, Topos off. Fixed 256-byte alphabet.
- 8,192 training target bytes per arm; 2,048 held-out target bytes at 0/64/128.
- Geometry arms each have 11,290 parameters. Frozen arms train the 11,216
  backbone parameters but still backpropagate through geometry to embeddings.
- Ordinary has 11,216 parameters and less compute. It is a context arm, not
  a capacity- or compute-matched quality control.
- One native-prepared request is shared unchanged by native, browser and Torch:
  `95ef8c21c6c9c8221d7aced6adcda9d80ce4c0dccf6ec3819e40475c52c1db38`.
- Its frozen v3 source is
  `759ed1282bad72c2351fcc9c78fef0f661087bf57f578f96e741882763204849`.

Initial RMS must agree within `1e-5` relative error. Every CE and final parameter
uses `3e-6 + 5e-5 * abs(reference)`. Trained geometry changes must be nonzero,
have reference L2 norm greater than `1e-8` and relative change-vector error at
most `0.002`. Frozen tensors must retain their initial float32 bits. No criterion
was loosened after observing a result.

## Measured Result

All 27 native Metal cases and all 27 actual browser WebGPU cases pass the
independent Torch checks. Lower held-out BPB is better. The native differences
below have the same signs in the browser; positive values favor the comparator:

| Seed | Fisher - Poincare | Fisher - matched flat | Fisher trained - frozen |
| --- | ---: | ---: | ---: |
| 11 | +0.00123780 | +0.00137414 | -0.00007864 |
| 23 | +0.00199641 | +0.00166979 | +0.00006393 |
| 37 | -0.00010910 | -0.00012880 | -0.00002116 |

Native maximum errors versus Torch: training CE `9.54e-7`, held-out batch CE
`1.43e-6`, final parameter `5.96e-7`, geometry-change relative L2 `1.93e-5`.
Browser maxima are `1.43e-6`, `1.43e-6`, `3.58e-7` and `9.87e-6`, respectively.
The three Fisher trained/frozen contrast discrepancies versus Torch are at most
`5.38e-8` BPB. Numerical qualification is not a statistical-quality conclusion.
Fisher is worse than both geometric comparators in two seeds, better in one.
Learning Fisher geometry also has a small, mixed effect. All seeds and adverse
contrasts are retained, not filtered into a winning configuration.

This is a tiny fixed repository-prose corpus, not large-corpus LLM fine-tuning,
generation-quality evidence or a speed comparison. RMS matching does not match
score patterns, attention entropy, parameter conditioning or subsequent dynamics.

Both runtimes also preserve all 21 v4 full case reports exactly. Separate
first-update runs check all 12 geometry pairs per runtime: same initial losses,
same first backbone update bits, moving embeddings, frozen geometry and nonzero
trained geometry changes. See `native-controls.json` and `browser-controls.json`.

## Checkpoint And Resume

Both runtimes ran a separate 37-update prefix and resumed to 128 in a fresh
process or page. Full and resumed report bytes are identical for all 27 cases,
including every parameter and loss. Both resumed reports independently pass
the unchanged Torch criteria. The checkpoint at 37 matches the paused readback
and history, without inserting an extra evaluation at the pause.

A further 128-to-128 no-op resume preserves the entire report and final
checkpoint bytes in each runtime. The native/browser resume and no-op
verification files bind the inputs and saved tensors. This is same-runtime
continuation evidence, not a cross-runtime handoff claim.

## Repairs And Limits

The first native preparation attempt correctly failed because the new caller
tried to attach geometry twice. Preparation now rebuilds each candidate through
the checked plan constructor; a CPU test verifies that only metric identity
changes before fitting. The original fail-closed model builder is unchanged.

Independent review also found that the two artifact validators could accept
contradictory per-metric Poincare reference RMS matrices for one seed. Both now
require one shared matrix per seed; coordinated `2/2` versus `20/20` mutations,
including every cell of a multi-head/multi-block matrix, are rejected. This was
a verifier repair: actual measurement arithmetic, frozen inputs and gates did
not change. The fixed validators rechecked the retained native/browser packets.

The [independent native evidence audit](independent-native-review.md) recomputes
every contrast and checks the small gain updates, including no-op/ULP mutation
controls. Its scope is explicitly native-only and offline. Local path prefixes
are replaced by `$RAW`, `$REPO`, `$V3`, and `$V4`; original reports are retained.
The separate [browser artifact audit](independent-browser-review.md) recomputes
browser numerical, first-step, resume and no-op checks. Neither artifact audit
claims to have rerun the GPU or attested its execution.

[validation.json](validation.json) records the final source, raw-artifact and
public-file hashes, test counts and limits. Only scalar trajectories, errors,
verification records and reproducible source are public. Original failed
preparation artifacts remain local and their hashes are retained.

## Reproduction

Reconstruct the unchanged v3 source using the
[metric-control recipe](../2026-10-10-byte-corpus-metric-controls-v3/README.md)
and confirm its hash. Build the native example and WASM module using the
[calibration documentation](../../../docs/byte_corpus_calibration.md).
Use a new `$RAW` directory and the retained v3 directory as `$V3`:

```sh
target/release/examples/resident_byte_learning "$V3/request.json" \
  --calibrate-fisher 0,1,2,3 > "$RAW/native-prepared-packet.json"
python3 -I -S -B tools/verify_byte_corpus_preparation.py \
  "$V3/request.json" "$RAW/native-prepared-packet.json" "$RAW/input"
python3 -I -B tools/byte_corpus_study.py reference \
  "$RAW/input/request.json" "$RAW/torch.json"
target/release/examples/resident_byte_learning "$RAW/input/request.json" \
  > "$RAW/native.json"
python3 -I -S -B tools/byte_corpus_study.py compare \
  "$RAW/input/request.json" "$RAW/torch.json" "$RAW/native.json" \
  "$RAW/native-comparison.json"
```

On loopback, run the actual browser learning page with these exact request bytes.
The preparation page's `?fisher=1` path qualifies WASM preparation separately;
its per-device fitted request must not replace the common comparison input.
Run stop-1 controls and stop-37/fresh-runtime resume as documented. For inherited
reports and first-update controls, supply the retained v4 request and runtime
report to `verify_byte_corpus_calibrated_controls.py`.

Full raw weights, requests, checkpoints, logs and binaries remain local. Public
scalar projections are not complete comparator inputs; hashes identify retained
artifacts, not a cryptographic attestation that computation happened.
