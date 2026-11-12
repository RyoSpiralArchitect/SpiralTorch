# Native QKV Corpus Runs

This requires a matching SpiralTorch **source build**, not the currently
published wheel. It trains one fused QKV matrix through the Rust CPU decoder;
other base weights and attached adapters stay frozen. It does not download a
model, import Torch/NumPy/Transformers, or claim improved language quality.

## Inputs

Use a local supported GPT-2-family `config.json` and `model.safetensors`.
Encoded data is a JSON object with `token_documents` and optional matching,
unshifted `labels` arrays. `-100` excludes loss, not context. Tokenization and
terminators remain explicit caller choices; existing `CausalCompletion` helpers
can produce answer-only labels without a foreign training backend.

The [example recipe](../../bindings/st-py/examples/native_qkv_corpus_recipe.example.json)
declares the selected layer, every AdamW/clipping option and the full native
training schedule. Its values illustrate the interface, not tuned settings.
`context_length` includes the target token. The shared covering-target planner
retains short tails and overlaps neighboring windows by one token, without
cross-document attention or full-document history. All-masked windows are
omitted, but ignored input content still enters the corpus identity.

## Run And Resume

From the repository root, choose an absolute stop boundary within the native
plan (`--stop-after` can be omitted to finish that fixed plan):

```sh
python bindings/st-py/examples/native_qkv_corpus_training.py \
  --model-dir "$MODEL" --data-json "$DATA" \
  --recipe-json bindings/st-py/examples/native_qkv_corpus_recipe.example.json \
  --save-every 10 --stop-after "$STOP" --output "$NEW_RUN"
```

Optional frozen geometry uses `--artifact` and `--artifact-id`. A frozen
LoRA/geometry source uses all of `--session-checkpoint`,
`--session-checkpoint-id` and `--session-boundary`, as in the existing generation
example. These source options are mutually exclusive. Select the same original
`--readout-accumulator` (`f32`, `f32_wide_backward` or `f64`) as the source.
Model config may resolve through a local cache symlink; recovery inputs must be
regular non-symlink files. All reads are bounded.

The run creates a new private directory with `checkpoint-NNNNNNNNN.bin`,
`checkpoints.jsonl`, `steps.jsonl`, a final `qkv.bin`, and `result.json`. Retain
the printed identities independently. To resume an explicitly selected session
checkpoint, supply the same model, frozen-source options, readout and corpus:

```sh
python bindings/st-py/examples/native_qkv_corpus_training.py \
  --model-dir "$MODEL" --data-json "$DATA" \
  --resume "$SESSION_CHECKPOINT" --checkpoint-id "$SESSION_PIN" \
  --save-every 10 --output "$NEW_CONTINUATION"
```

Do not provide a replacement recipe on resume. Native state owns configuration,
weights, moments and cursor, and verifies the supplied corpus. Checkpoints do
not embed the frozen model or corpus. Their hashes are integrity pins, not
authentication. No automatic latest-checkpoint choice or retry occurs.

For an already constructed trainer, the same orchestration is importable:

```python
from spiraltorch.native_qkv_runs import run_qkv_corpus
result = run_qkv_corpus(session, new_directory, save_every=10, stop_after=stop)
```

## Held-Out Scoring

The same Rust corpus evaluator can score a detached QKV snapshot without an
optimizer update or Torch/NumPy. Use data not selected for training, keep the
same documents and labels for the baseline and trained comparison, and choose
the context/window policy before reading results. `frozen_source` below is the
exact `NativeCausalLm` used to create the trainer, or its pinned reload:

```python
snapshot = session.optimizer_snapshot()
pair = snapshot.evaluate_against_source(
    frozen_source,
    heldout_token_documents, context_length=128,
    microbatch_windows=2, labels=heldout_labels,
)
assert pair.serving_id == snapshot.serving_id
baseline, trained = pair.evaluations
assert baseline.data_id == trained.data_id
print(baseline.nll, trained.nll, trained.target_tokens)
```

The ordered pair records the QKV layer, accepted update count and exact serving
ID; each evaluation retains the corpus/plan identities and target-weighted
unsmoothed NLL. `pair.to_json()` preserves the complete receipt. The source
must have the same base, frozen LoRA, geometry and readout identities as the
snapshot. Rust shares only blocks strictly before both models' first adapter
site and the trained QKV layer. `snapshot.evaluate_documents(...)` remains
available for a single trained-model score. This API makes a matched
comparison possible; it does not itself show that training improved language
quality or that the paired path is fast enough for a large real model.

For answer-only labels, request the Rust-owned answer breakdown on the same
frozen corpus. It keeps the source/serving identity and safe prefix cap, and
reports each contiguous run of selected targets, even across windows:

```python
detail = snapshot.evaluate_against_source_by_answer(
    frozen_source,
    heldout_token_documents, context_length=128,
    microbatch_windows=2, labels=heldout_labels,
)
assert detail.serving_id == snapshot.serving_id
assert detail.evaluations[0].data_id == detail.evaluations[1].data_id
print(detail.contrast(0, 1))  # trained minus source; negative NLL is better
print(detail.answer(0))
```

`contrast(0, 1)` separates token-weighted and answer-weighted NLL changes,
top-1 counts and improved/tied/regressed answer counts. These are descriptive
scores on the chosen corpus, not a significance test or a new held-out split.
Answer detail is opt-in, bounded by 65,536 answer-by-variant cells, and may
cost more than aggregate NLL because it scores selected logits by answer.

On a source build with the CPU autotune fast path, set
`SPIRALTORCH_AUTOTUNE=0` before starting Python to skip first-use
microbenchmarking and use the deterministic default matmul kernel. This is
useful for a bounded diagnostic, not a steady-state performance comparison.

## Inference And Failure Boundaries

The final `qkv.bin` is the existing optimizer-snapshot format, separate from
the corpus checkpoint. Its ID is `result["qkv"]["checkpoint_id"]`. Pass it to
`native_geometry_generation.py` with `--qkv-checkpoint` and
`--qkv-checkpoint-id`, plus the same model/frozen-source/readout arguments and an
explicit prompt/decoding recipe. That route supports the existing native cache
and Z-Space sampler. Generation does not verify training corpus content.

`save_checkpoint_new` and `advance_to_checkpoint_new` share Rust's existing
no-clobber publisher with adapter training. Advancement allows 1..=1024 updates
and replaces the live session only after publishing the complete checkpoint.
Existing paths, including links, are never overwritten. On Unix, publication
requires hard-link support; use a private directory with a single writer.

After a run, `spiraltorch.native_qkv_runs.verify_qkv_run(directory)` checks the
completion marker, ordered index and step log, every trainer/optimizer SHA-256
pin, consistent optimizer owners and training-plan length, exact equality of
the final optimizer payload with `qkv.bin`, and the Rust serving projection ID
recomputed from its frozen owner and QKV weights. It
returns the parsed result or raises on a missing, truncated, inconsistent or
oversize file. This is a local integrity check, not authentication, a model
reload, or proof of learning quality. Keep the original files for recovery.
Step log continuity and finite JSON are checked, but individual report fields
are not hash-pinned by this v1 output format; do not treat them as authenticated.

On **any** error, stop. A native publication error may leave a complete file
while the live session remains unchanged. A later Python index/report error
occurs after native publication and can therefore leave both a durable new
checkpoint and an advanced live session, but incomplete logs. Neither case is
a successful run or permission to retry. Preserve the directory and verify an
explicitly chosen checkpoint against its retained pin before recovery. Missing,
partial or invalid `result.json` is not completion evidence; even a parseable
result must be checked against its pinned files. Caller-visible I/O errors are
not swallowed. This is not a multi-file transaction or multi-writer protocol.

[Verification](../../benchmarks/paired-prompt-contrast/QKV-FILE-RUN.md) covers
synthetic CPU execution, resume, failure boundaries and generation handoff.
