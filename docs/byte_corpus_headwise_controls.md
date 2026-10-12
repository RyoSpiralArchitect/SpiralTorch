# Fixed Headwise Corpus Controls

The v6 study combines the existing Poincare, Euclidean chord and categorical
Fisher-Rao pair biases *inside* the QKV attention training path. This is fixed
per-head routing, not a learned geometry selector or an additive mixture.
The Rust [headwise geometry plan](headwise_geometry_attention.md) owns forward,
backward and checkpoint semantics. Python is an independent Torch reference;
WASM is another client of the same Rust learner.

## Design

Start with an unchanged [v5 calibrated study](byte_corpus_calibration.md#fisher-rao-control-v5).
The runner currently requires exactly one block and two heads; the underlying
model API supports more general block/head grids.

Every seed keeps all nine v5 arms. Six ordered heterogeneous assignments
(P/E, P/F, E/P, E/F, F/P, F/E) each get trained and geometry-frozen twins.
That is 21 arms per seed, at most three seeds and 63 arms per request.
An ordinary model remains a lower-parameter context arm, not a matched
compute baseline.

Each routed head takes its *same head index's* raw gain from P-original,
E-RMS-matched or F-RMS-matched. No gains are fitted again. Every other initial
parameter, document, batch selection, update count and evaluation stays fixed.
Both runtimes check actual revision-zero causal row-centered bias RMS on the
declared training batches against the original P target (relative error <= 1e-5).
The independent Torch learner measures the same qualification without fitting.
RMS matching does not match the bias pattern, derivative or subsequent dynamics.

## Interfaces

Rust exposes `ByteCorpusHeadwiseControls::from_json(v5_bytes)` and
`.prepare(runtime).await`. The result retains opaque `request_json` and a
preparation report, as with bias calibration. After independent preparation
verification, use `ByteCorpusStudy` for learning, pausing and resuming.

```sh
cargo build --locked --release -p st-nn --no-default-features --features wgpu \
  --example resident_byte_learning
target/release/examples/resident_byte_learning "$V5/request.json" \
  --prepare-headwise > "$RAW/native-prepared-packet.json"
python3 -I -S -B tools/verify_byte_corpus_headwise_preparation.py \
  "$V5/request.json" "$RAW/native-prepared-packet.json" "$RAW/native-preparation.json"
```

Extract the verified packet's `request_json` verbatim into one new
`request.json`, without parsing and reserializing its contents. Every runtime
must consume these exact bytes, not its own freshly prepared alternative.

```sh
python3 -I -B tools/byte_corpus_study.py reference \
  "$RAW/request.json" "$RAW/torch.json"
target/release/examples/resident_byte_learning "$RAW/request.json" > "$RAW/native.json"
python3 -I -S -B tools/byte_corpus_study.py compare \
  "$RAW/request.json" "$RAW/torch.json" "$RAW/native.json" "$RAW/native-comparison.json"
```

Build `resident_byte_learning_browser` with the existing
[WASM recipe](byte_corpus_calibration.md). For preparation, serve the unchanged
v5 source at `target/resident-byte-learning-web/source-v5.json` and open
`crates/st-nn/tests/byte_learning_prepare_browser.html?headwise=1`.
The page calls `prepare_resident_byte_learning_headwise(input)`; JavaScript
does not select gains or implement model mathematics. Verify its downloaded
packet too, but run the browser learner on the same common request chosen above.
The existing learning and resume pages accept v6.

## Verification

v6 requests carry a `headwise_controls.source_request_sha256` lineage field.
Heterogeneous cases use `head_pair_metrics: [[metric0, metric1]]`, with no
`pair_metric`. Existing cases remain unchanged. Requests, partial/full reports
and study checkpoints use v6; heterogeneous *model* checkpoints use v3.
Resume validates every route, revision, parameter, frozen value and data identity.
Hashes bind artifacts; they are not signatures or proof of past execution.

The comparator preserves all old numerical gates and adds a per-head gain
learning-delta gate for heterogeneous arms. A large Poincare update cannot hide
a disconnected smaller Fisher head. Unqualified tiny reference deltas fail;
they are not permission to relax tolerance.

Use `verify_byte_corpus_calibrated_controls.py` with the v6 request and v5
source to check every inherited full report plus separately run first-update
trained/frozen controls. Use `verify_byte_corpus_resume.py` on an uninterrupted
run, a pause at 37, a fresh-runtime continuation to 128, and a completed no-op.
All these are within-runtime exact checks, separate from Torch tolerances.

The scalar comparison publishes all six routes for each seed, trained versus
frozen, each homogeneous comparator, reversed-head assignment and Torch
discrepancies. This bounded corpus is a mechanism experiment, not evidence of
pretrained LLM quality or throughput superiority. Full weights/checkpoints stay
local; scalar evidence, hashes, fixed gates and failures belong in the results.
