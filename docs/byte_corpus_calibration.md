# Training-Only Bias Calibration

The five-arm [byte corpus study](byte_corpus_learning.md) holds all geometric
initial weights equal, but Poincare and flat distances produce different bias
strengths. v4 adds two arms per seed: flat with calibrated initial gains,
geometry-trained and geometry-frozen. This separates a one-time strength
treatment from the distance and trainability treatments. It does not match
attention patterns, probabilities, optimizer conditioning or later dynamics.

## Two Phases, One Authority

`st_nn::resident::ByteCorpusBiasCalibration::from_json(v3_bytes, indices)`
validates the unchanged five-arm source and 1..16 distinct training-batch
indices. It preflights the expanded request and worst-case resumable checkpoint
before requesting a GPU. v4 admits at most 21 cases: three complete seven-arm
seed groups. Old versions retain their 16-case limit.

`prepare(runtime).await` uses the same Rust learner's revision-zero forward
on those training windows. Per block/head, it measures the causal row-centered
bias RMS, fits flat raw gains, and remeasures their realized device RMS against
Poincare with a fixed `1e-5` relative gate. See the
[calibration mathematics](causal_bias_calibration.md). No held-out evaluation,
SGD update or differentiable normalization is performed.

`ByteCorpusPreparedBiasStudy` contains `request_json` and a preparation `report`.
The request explicitly records `bias_initialization` on every case and
`bias_calibration` with the exact v3 source SHA, training indices and fixed gate.
The original five arms per seed are preserved. Only raw gains may differ in
the new pair, and that pair shares every initial float32 bit. All seven share
the same non-geometric parameters, data selections and SGD schedule.

Freeze **one** prepared request for native, WASM and the independent Torch
reference. A separately browser-prepared request may qualify preparation
mechanics, but must not silently replace the common input in a comparison.
Different devices may fit slightly different floating-point weights.

## Prepare And Consume

Start with a retained v3 request from `--metric-geometry-control`. Choose
calibration indices before observing held-out results, and use new output paths.
The following fits on the first four training batches:

```sh
target/release/examples/resident_byte_learning "$V3/request.json" \
  --calibrate-flat 0,1,2,3 > "$RAW/native-prepared-packet.json"
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

The packet carries opaque `request_json` and `report_json` strings. The verifier
extracts the exact request bytes, checks source identity, unchanged source arms,
gain-only transformation, model-checkpoint parameter bits, coverage and reported
RMS arithmetic. It does not independently attest device execution. Raw requests,
checkpoints and weights should remain local; publish scalar results, source,
hashes and reproduction instructions.

The independent CPU-f32 Torch reference consumes the fitted gains without
fitting. Before learning it remeasures the same statistic using only the declared
training windows. Failure is a qualification failure, not permission to refit
or relax the gate. Standard loss/parameter and geometry change-vector criteria
remain unchanged.

Build the existing `resident_byte_learning_browser` example as documented in
the corpus recipe. All three browser pages share that one Rust module:

- `byte_learning_browser.html` runs the frozen request at
  `target/resident-byte-learning-web/request.json`.
- `byte_learning_resume_browser.html` runs/resumes versioned checkpoints.
- `byte_learning_prepare_browser.html` qualifies the preparation export
  `prepare_resident_byte_learning(input, indices)` on `source-v3.json`, with
  the page's predeclared indices `[0,1,2,3]`. Download its opaque packet.

## Resume And Interpretation

Every v4 execution, including uninterrupted runs and no-op resumes, remeasures
**temporary initial models** against the fixed RMS gate on the selected device.
It never fits again, changes resumed weights, adds evaluations, or advances a
learner revision. These startup forwards/snapshots are explicit extra work,
not a resident-performance optimization or part of a speed comparison.

Study request/result/partial/checkpoint schemas are v4; model checkpoint schemas
are unchanged. Existing revision-zero initial checks and frozen-geometry checks
now bind the actual fitted request values. There is no hidden mutable calibration
state in a checkpoint. Keep exact request bytes and the source v3 artifact.

Run the normal stop-37/fresh-process-or-page resume test and
`tools/verify_byte_corpus_resume.py`. Also run a separate stop-1 segment, then:

```sh
python3 -I -S -B tools/verify_byte_corpus_calibrated_controls.py \
  "$RAW/input/request.json" "$V3/request.json" "$RAW/native.json" \
  "$V3_NATIVE_RESULT" "$RAW/native-first-step.json" "$RAW/native-controls.json"
```

Repeat with browser results. This checks exact retained v3 full case reports,
same-initialization first losses and first backbone update bits, moving
embeddings, frozen geometry and nonzero trained-geometry changes. The comparator
identifies arms by metric, update policy **and initialization treatment**, never
by their order or an assumed flat-arm name. All old contrasts and seven additional
calibration contrasts remain visible for every seed, including unfavorable ones.

RMS matching is an initialization control, not proof of equivalent computation
or superior geometry. The fixed small repository corpus is still not a large
language benchmark, pretrained LLM fine-tuning or a generation-quality study.
