# Optimizer-Aware Language Study Qualification

This qualifies the Rust-owned v9 optimizer recipe, same-device resume and
immutable inference handoff. It is **not** a language-quality, speed, tuned
optimizer, or Riemannian-Adam benchmark.

The source snapshot is pinned by the parent commit and per-file hashes in
[verification.json](verification.json). Full requests, corpus bytes, parameters,
moments, source-built wheels and raw logs remain private. This directory contains
all conditions' summaries, verification records, hashes and reproduction steps.
No PyPI upload was performed; validation wheels retain 0.4.31 metadata and must
not be published.

## Matched Small Pilot

Three fixed seeds (11, 29, 47), each with ordinary, learned Poincare geometry and
identically initialized frozen geometry. Both explicit SGD and AdamW use the
same documents, selected windows, initial parameters, rate 0.001 and 12 updates.
Model: batch 2, window 8, width 8, hidden 16, two heads, one block, four geometry
coordinates. Each case sees 192 training target bytes and 64 held-out target
bytes. This is deliberately a small qualification run, not a large-corpus study.

AdamW uses beta1 0.9, beta2 0.999, epsilon 1e-5, ordinary decay 0.01 and geometry
decay 0. Zero-rate geometry freezes weights, moments and accepted tensor clocks.
The controls are normalized to float32 and reported explicitly. Identical rates
are not optimizer-specific tuning; the SGD control also has no weight decay.
Neither the update rule nor decay is factorially isolated by this comparison.

Mean final held-out bits per byte (lower is better), native Apple M4 / Metal:

| Declared Recipe | Ordinary | Learned Geometry | Frozen Geometry |
| --- | ---: | ---: | ---: |
| SGD | 8.054104101 | 8.054994628 | 8.054995258 |
| AdamW | 7.925212564 | 7.927629092 | 7.927685617 |

AdamW moves this fixed small recipe further than SGD, but this does not establish
a tuned optimizer advantage. **Both geometric arms are worse than the ordinary
arm for every seed under both recipes.** The observed learned-minus-frozen
point estimate is negative for seeds 11 and 29 and positive for seed 47 under
both recipes. These tiny differences are not established quality gains.
Retain all results; do not promote geometry or tune on this held-out set.

All individual seeds and contrasts are retained in
[native-sgd.json](native-sgd.json), [native-adamw.json](native-adamw.json) and
[browser-adamw.json](browser-adamw.json). The v2 preparation identity is in
[preparation.json](preparation.json); v9 wrapper hashes differ by optimizer.

## What Passed

- Independent CPU PyTorch 2.12.1 versus native WGPU: all nine cases for each
  optimizer. Parameters, CE, geometry deltas and AdamW moments/clocks qualified.
- Actual BrowserWebGpu versus the same PyTorch AdamW reference: all nine cases.
  Maximum parameter error was 2.517e-6 (native: 2.581e-6); maximum per-tensor
  moment relative L2 errors stayed below 4.5e-6 on both engines.
- Pause at update 5, then resume to 12: exact uninterrupted/resumed reports on
  native and on a fresh browser page. See [native-resume.json](native-resume.json)
  and [browser-resume.json](browser-resume.json).
- Browser artifact handoff retained opaque model JSON, rejected foreign source
  bytes and altered artifacts, and reproduced the final scheduled loss exactly
  as float32. See [browser-handoff.json](browser-handoff.json).
- CPU-only installed wheel read all nine trained AdamW models, round-tripped
  exact Rust JSON, and rejected GPU execution explicitly. See
  [cpu-transport.json](cpu-transport.json).
- Rust CPU study tests: 67 passed. Native byte-language tests with real WGPU:
  116 passed. Independent Python study/verifier tests: 72 + 22 passed.
- Installed WGPU wheel: 34 passed, 2 CPU-only tests skipped. Installed CPU-only
  wheel: 21 passed, 15 GPU tests skipped. Both release-smoke suites: 9 passed.
  Python also ran prepared 25-arm contextual mixtures with AdamW, frozen-history
  checks, exact resume and fresh-process training/inference restoration. That
  contextual test is not the v2 corpus/PyTorch comparison above.

Two independent read-only local reviews covered the Rust integration and Python
verifiers. The verifier review found erased tiny moments could hide within a
weight tolerance, and terminal paused history could escape no-op-resume binding.
Both were fixed and re-reviewed. Negative tests now cover those corruptions.
Moment qualification requires per-tensor relative L2 error <= 0.002 with no
absolute floor, in addition to the existing elementwise gate; terminal paused
and resumed reports must be exactly equal.

## Limits And Reproduction

Pinned rustfmt passed. A separate strict st-nn Clippy run on Rust 1.99 failed with
24 diagnostics in files unchanged from parent `d487e367`; none targeted this
patch's files. This is not a clean whole-crate lint result. The parent PR's last
CI run did not start because of an account billing lock. Local validation does
not replace latest-head green CI, and no merge bypass was used.

Reconstruct the source corpus at its declared revision, not the latest docs:

```bash
python -I -B tools/byte_corpus_study.py prepare "$RAW/paired" \
  --revision d487e367c1cc97112564a56cea2d9fccf6bce527 \
  --train-doc docs/byte_corpus_learning.md \
  --train-doc docs/byte_corpus_calibration.md \
  --validation-doc docs/resident_adamw.md \
  --seed 11 --seed 29 --seed 47 --batch 2 --steps 8 --width 8 \
  --hidden 16 --heads 2 --blocks 1 --updates 12 --checkpoint-every 4 \
  --validation-batches 4 --rate .001 --frozen-geometry-control
python -I -B tools/byte_corpus_study.py bind-optimizer "$RAW/paired/request.json" \
  examples/configs/byte_corpus/adamw.json "$RAW/adamw-request.json"
python -I -B tools/byte_corpus_study.py reference \
  "$RAW/adamw-request.json" "$RAW/adamw-torch.json"
cargo build --locked --release -p st-nn --no-default-features --features wgpu \
  --example resident_byte_learning
target/release/examples/resident_byte_learning "$RAW/adamw-request.json" \
  --checkpoint-out "$RAW/adamw-checkpoint.json" > "$RAW/adamw-native.json"
python -I -B tools/byte_corpus_study.py compare \
  "$RAW/adamw-request.json" "$RAW/adamw-torch.json" \
  "$RAW/adamw-native.json" "$RAW/adamw-comparison.json"
```

Use the checked-in SGD recipe for the other control. Outputs are exclusive-create;
choose a new private directory rather than overwriting previous evidence.

For resume, run the same native example with `--stop-after 5 --checkpoint-out`,
then with `--resume` and the saved checkpoint. Pass request, uninterrupted,
partial, resumed and partial-checkpoint files to
`tools/verify_byte_corpus_resume.py`, followed by a new output path.

Build `resident_byte_learning_browser` for `wasm32-unknown-unknown` with WGPU,
then use wasm-bindgen 0.2.129 to emit the web package. Serve the existing learning
and resume pages on loopback with the same exact request bytes. Download opaque
results, place the partial checkpoint at the documented path and open
`?resume=1` on a fresh page. Apply the same independent compare/resume verifiers.
Build/test commands and raw receipt hashes are recorded in verification.json.

These checks do not establish cross-device bitwise equality, elapsed-time
performance, broad language quality, model promotion, or computation attestation.
The next quality experiment needs larger independent data and a predeclared,
optimizer-tuned comparison, not a favorable reading of this tiny cohort.
