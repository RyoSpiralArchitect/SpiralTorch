# Byte-Corpus Learned-Mixture Controls

For position-conditioned mixtures with one shared intercept, see the
[v8 contextual study](byte_corpus_query_controls.md); v7 below remains unchanged.

Study v7 connects the [resident learned-mixture model](learned_geometry_mixture.md)
to document-held-out byte-language learning. Rust owns preparation, resident
forward/backward, atomic SGD and checkpoint restoration. Python provides an
independent Torch reference; JavaScript transports opaque requests and reports.
There is no Python/JavaScript production geometry fitter.

## Comparison Design

Start from a complete, unchanged [v5 study](byte_corpus_calibration.md#fisher-rao-control-v5).
Keep its nine arms per seed. Add every distinct two/three-component subset of
Poincare squared (P), origin-scaled Euclidean chord squared (E), and categorical
Fisher-Rao squared (F): P/E, P/F, E/F, and P/E/F. Each subset has two arms:

- Learned router: train the coordinate encoder, causal wave, shared head gains,
  router logits, embeddings and ordinary Transformer weights.
- Fixed router: run the same forward and full pullback, but set only the router
  parameter rates to zero. Geometry is not frozen or detached.

This is 17 arms per seed, with at most three seeds. The bounded v7 recipe uses
one block and two heads. It does not redefine v1-v6 or their old measurements.
The router is shared over sequence positions, not a token-conditioned selector.
Both arms start with canonical +0 logits, hence uniform softmax weights.

The positive-gain ratio of each calibrated v5 metric to v5 P defines a fixed
component unit scale per head. Then Rust fits only the initial shared gain of
each mixture to the P row-centered causal-bias RMS on the existing predeclared
training windows. Realized relative error must be at most `1e-5`. Component
units and routers do not fit; there is no held-out calibration. Every learning
or resume entry rechecks the initial RMS without refitting.

## Run And Verify

Use a new local artifact directory for each attempt. Keep requests, raw weights,
checkpoints, executables and full logs local. `SOURCE` is the exact prepared v5
request; `RAW` is a new local directory; `BIN` is the built native example.

```sh
cargo build --locked --release -p st-nn --no-default-features --features wgpu \
  --example resident_byte_learning
"$BIN" "$SOURCE" --prepare-mixtures > "$RAW/native-prepared-packet.json"
python3 -I -S -B tools/verify_byte_corpus_mixture_preparation.py \
  "$SOURCE" "$RAW/native-prepared-packet.json" "$RAW/input"
"$BIN" "$RAW/input/request.json" > "$RAW/native.json"
python3 -I -B tools/byte_corpus_study.py reference \
  "$RAW/input/request.json" "$RAW/torch.json"
python3 -I -S -B tools/byte_corpus_study.py compare \
  "$RAW/input/request.json" "$RAW/torch.json" \
  "$RAW/native.json" "$RAW/native-comparison.json"
```

Build `resident_byte_learning_browser` with the existing
[WASM recipe](byte_corpus_calibration.md). To exercise browser preparation,
serve the exact v5 source as `target/resident-byte-learning-web/source-v5.json`
and open `crates/st-nn/tests/byte_learning_prepare_browser.html?mixtures=1` on
loopback. The page calls `prepare_resident_byte_learning_mixtures(input)`.
Independently verify its downloaded packet, but select **one** prepared request
for all learning engines. Browser/native calibration may differ by rounding;
substituting per-engine fitted requests would invalidate the matched comparison.

Serve the selected opaque request as
`target/resident-byte-learning-web/request.json`. Run `byte_learning_browser.html`
for uninterrupted learning and `byte_learning_resume_browser.html?stop=37`
for a checkpoint. Resume in a fresh page with `?resume=1`, then repeat from the
completed checkpoint to test a no-op resume. Use the same comparator and
`verify_byte_corpus_resume.py` on downloaded reports. Native equivalents are
`--stop-after 37 --checkpoint-out NEW.json` and
`--resume OLD.json --checkpoint-out NEW.json`. Run GPU jobs serially.

Also execute a separate stop-1 segment. Pass the new full report, retained v5
report and first-update report to `verify_byte_corpus_calibrated_controls.py`.
It checks exact inherited reports, original geometry-frozen controls, and each
new router pair's identical initial evaluation, first loss and **every
non-router parameter update**. Fixed-router bits must remain unchanged while
the encoder, shared gain and embeddings still learn.

## Evidence Boundaries

`spiraltorch.byte_corpus.request.v7` carries
`mixture_controls.source_request_sha256`. Each added case has `mixture` with
ordered `metrics`, per-block `[head,component]` `scales`, and `router_update`.
It has neither `pair_metric` nor `head_pair_metrics`. Initial tensors add
`geometry.mixture_logits.0` immediately after the head gain. Saved mixtures use
model checkpoint v4 inside corpus checkpoint v7; fixed routers are checked
against the original bits before resuming.

The comparator checks all losses and final parameters against Torch, the
learning delta of every trained geometry family, and every individual router
component. A component below the predeclared sensitivity floor is a failed
qualification, not permission to omit that component. Calibration checks all
four subsets against the same P target for each seed.

The descriptive contrast is learned-router minus fixed-router held-out BPB,
alongside each arm's differences from trained P, calibrated E and calibrated F.
Equal data and update counts do not imply equal compute across metric subsets.
Completion, numerical equivalence and a corpus improvement are separate claims.
This small pilot does not establish general LLM efficacy, a tokenizer-free
advantage, manifold QKV projections, or a preferred geometry.
