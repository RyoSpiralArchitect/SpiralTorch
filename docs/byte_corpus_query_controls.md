# Contextual Geometry Mixtures In Byte-LM Learning

Corpus study v8 extends [v7](byte_corpus_mixture_controls.md) without changing
its seventeen arms per seed. Each P/E, P/F, E/F and P/E/F subset gains two arms:
a trainable contextual projection and an identical projection whose weight
rate is zero. Both retain a trainable shared router and the full coordinate
pullback. No gradient is detached.

The projection reads causal-wave coordinates, not future tokens:

```text
logits[b,h,q,k] = shared[h,k] + (coordinates[b,q] @ W)[h,k]
weights = softmax(logits, component_axis)
attention_bias[b,h,q,j] = sum_k weights[b,h,q,k] * scale[h,k] * metric_k[b,h,q,j]
```

The model's existing affine graph has a bias tensor, but this study requires
canonical zero initialization and a zero update rate for that bias throughout
learning and resume. The shared table is the only trainable intercept.
Both W and bias initially contain canonical +0. The frozen-W arm is a
structural no-context control, not frozen geometry. Coordinate projection,
causal-wave parameters, gains, shared logits and ordinary QKV still learn.
Extra parameter storage and execution cost are reported, not called a matched
parameter/compute budget or a speed comparison.

## Preparation And Learning

Rust's `ByteCorpusQueryMixtureControls::from_json(source)?.prepare()` extends
a complete v7 request on CPU. It draws no randomness, fits nothing, and
consumes no updates. All source data, order, seeds, units and gains survive
unchanged. Learning rechecks the inherited training-only initial RMS on the
actual GPU; it never refits on validation documents.

```sh
"$BIN" "$SOURCE_V7" --prepare-query-mixtures > "$RAW/prepared.json"
python3 -I -S -B tools/byte_corpus_query.py \
  "$SOURCE_V7" "$RAW/prepared.json" "$RAW/input"
"$BIN" "$RAW/input/request.json" > "$RAW/native.json"
python3 -I -B tools/byte_corpus_study.py reference \
  "$RAW/input/request.json" "$RAW/torch.json"
python3 -I -S -B tools/byte_corpus_study.py compare \
  "$RAW/input/request.json" "$RAW/torch.json" \
  "$RAW/native.json" "$RAW/comparison.json"
```

The same `resident_byte_learning_browser` build and learning/resume pages
accept v8. Browser preparation is exposed as
`prepare_resident_byte_learning_query_mixtures(input)`, calling the same Rust
constructor without a GPU. Use **one** selected opaque request across all
engines. Keep raw requests, weights, checkpoints and executables local.

The protocol permits one block, two heads, all four subsets, and one to three
seeds: 25 arms per seed. The request records `query_controls.source_request_sha256`.
Contextual cases add `mixture.query_projection` with `weight_update` and
`bias: "frozen_zero"`. Model checkpoints use v5; the enclosing corpus
checkpoint uses v8 and verifies every frozen query parameter before resume.

## Checkpoint And Resume

Keep the request and executable fixed, and use fresh output paths. For a
pause between scheduled held-out evaluations:

```sh
"$BIN" "$RAW/input/request.json" --stop-after 37 \
  --checkpoint-out "$RAW/checkpoint-37.json" > "$RAW/partial-37.json"
"$BIN" "$RAW/input/request.json" --resume "$RAW/checkpoint-37.json" \
  --checkpoint-out "$RAW/checkpoint-128.json" > "$RAW/resumed.json"
python3 -I -S -B tools/verify_byte_corpus_resume.py \
  "$RAW/input/request.json" "$RAW/native.json" "$RAW/partial-37.json" \
  "$RAW/resumed.json" "$RAW/checkpoint-37.json" "$RAW/resume-check.json"
```

The verifier consumes the checkpoint at the **pause**, not the final checkpoint:
it binds saved tensor bits and histories to the paused report, then requires
the full resumed report to match uninterrupted learning exactly. A completed
checkpoint can be reloaded separately to test a no-op, without consuming updates.
Browser pause uses `byte_learning_resume_browser.html?stop=37`; resume requires
a fresh page with `?resume=1` and the opaque downloaded checkpoint served as
`target/resident-byte-learning-web/checkpoint.json`. Native and browser each
compare against their own uninterrupted result; cross-runtime bitwise equality
is not required. Keep checkpoints and full parameter reports local.

For the browser completed-state no-op, preserve the pause checkpoint and
serve the final downloaded checkpoint under that same endpoint instead.
Open another fresh `?resume=1` page, save both returned artifacts to new paths,
and compare the report against the original browser result:

```sh
python3 -I -S -B tools/verify_byte_corpus_resume.py \
  "$RAW/input/request.json" "$RAW/browser.json" "$RAW/browser-resumed.json" \
  "$RAW/browser-noop.json" "$RAW/browser-checkpoint-128.json" \
  "$RAW/browser-noop-verification.json"
cmp "$RAW/browser-checkpoint-128.json" "$RAW/browser-noop-checkpoint.json"
```

## Interpretation

Compare contextual training against both frozen-W and shared-only arms.
The frozen-W/shared-only difference exposes implementation-path rounding;
initial function equality is not a claim of bitwise-equal optimizer trajectories.
All training losses, held-out losses and final parameters still use the
existing independent Torch gates, including every trained geometry family
and individual shared-router component. Tiny unresolved deltas fail rather
than disappear from the comparison.

The separate contextual-model v1 browser qualification remains **failed** at
its frozen per-step rounded-delta gate. Its recorded one-ULP diagnostic is not
a waiver and this corpus study does not retroactively change that result.
Corpus completion, numerical qualification and improved held-out performance
are distinct. This small byte-model study is not pretrained-LLM FT.
