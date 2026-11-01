# Native Document Training Without Torch

Available in [SpiralTorch 0.4.33](https://pypi.org/project/spiraltorch/0.4.33/).
`NativeCausalLmTrainer` moves the ordered
corpus/optimizer loop into Rust; Python supplies encoded documents and performs
file I/O. It does not subclass HF Trainer or call Torch, Transformers, NumPy or
Accelerate. Supported model architecture remains the frozen native GPT-2 decoder;
this learns attached geometry, not base weights or arbitrary HF architectures.

For a real pretrained-model comparison using this public wheel, see the
[three-seed matched-start book study](../../benchmarks/results/2026-10-31-native-book-ablation/README.md).
On the sampled excerpts from a different novel, learned-radius WaveGate improved
over fixed radius and the frozen base, but ordinary affine had the lowest NLL.
This is held-out prediction evidence, not a generation-quality or speed claim.

```python
import spiraltorch as st

# model = native base.with_geometry(trainable_artifact)
optimizer = st.NativeGeometryAdamW(model.geometry_artifact(), max_gradient_norm=0.1)
trainer = st.NativeCausalLmTrainer(
    model, optimizer, token_documents, 128,
    learning_rate=0.001, weight_decay=0.01,
    microbatch_windows=2, accumulation_steps=4, epochs=2,
    label_smoothing=0.1,
)
while not trainer.exhausted:
    step = trainer.step()
    print(step.update, step.loss, step.target_tokens)

checkpoint, pin = trainer.to_json(), trainer.checkpoint_id
restored = st.NativeCausalLmTrainer.from_json(base, token_documents, checkpoint, pin)
adapted_model = restored.model
```

Choose context/batch sizes that fit the model and attached geometry budgets.
The returned `.model` and `.optimizer` are immutable snapshots; keeping one does
not let later trainer steps change it. `.step()` returns `None` when exhausted.
There is no automatic retry, best-checkpoint selection, deployment or Hub upload.

## One Shared Training Path

The existing Rust `CausalDocumentWindows` determines every row. Documents are
not concatenated, padded or assigned invented EOS tokens. Full fixed-width
windows are disjoint within each document, incomplete tails are dropped, and
the first token of each window is context rather than a target. `plan_json()`
reports those omissions. This is the same boundary policy as `HfGeometryDataset`.

Windows run in document order, repeated in that same order each epoch. There is
no RNG, shuffle, dropout or learning-rate schedule in this entry. The fixed rate,
weight decay, smoothing, batch/accumulation sizes and epoch count are sealed in
the checkpoint. Short final microbatches and accumulation groups are retained;
a group never crosses an epoch boundary.

`labels=` optionally supplies one unshifted integer sequence per document.
Lengths must exactly match; `-100` excludes a target but not its context.
Each planned microbatch must contain at least one selected shifted target; the
constructor rejects zero-target batches instead of inventing a skip policy.
All tokens and labels, including dropped tails and unscored first labels, are
validated. Target counts and token-weighted means come from the existing Rust
accumulator; clipping and one AdamW update happen only after the full group.

Each step stages all computation before replacing model, optimizer and cursor.
A failed gradient or numerical update leaves all three unchanged. Zero learning
rate still consumes windows and reports their losses, but retains model values,
moments and optimizer clocks under the existing native AdamW policy.

`.epoch` is zero-based for the next update and equals the configured epoch count
on completion; `.next_window` is the next index in the native window plan, or its
length when complete. `step.epoch` describes the update just performed.

## Resume And Ownership

### Observe an update (source after 0.4.33)

Use `trainer.step_observed()` **instead of** `trainer.step()` on an update you
want to measure. It advances the same optimizer/cursor exactly once and returns
`None` when exhausted. `observed.step` is the ordinary training report;
`json.loads(observed.to_json())` adds model/data identity and a `sites` map.
In Rust, `NativeCausalLmTrainer::step_observed` returns the typed
`CausalLmObservedTrainingStep`. Neither path stores observations in checkpoints.

Each site accumulates the actual pre-insertion activation `x` and the effective
residual `y - x` on the **primary forward**, after strength and f32 output
rounding. RMS uses every contextual activation element across all microbatches,
including rows whose labels are ignored. Backward block replays and auxiliary
loss pullbacks do not add samples. `relative_residual_rms` is
`sqrt(sum((y-x)^2) / sum(x^2))`, or null for an all-zero input.

`trainable_gradient_l2` and the signed `log_radius_gradient` use the same
reduced f32 objective gradient handed to AdamW, **before** global clipping,
moments or weight decay. This is not the mean of microbatch gradient norms.
For a radius buffer, `pre_log_radius` is present and its gradient is null;
both fields are null for non-radius recipes. A trainable radius with zero
gradient reports zero, not null. Exact per-site radii and gradients are
private learned-state diagnostics; choose a publication projection explicitly.
Ordinary training does not accumulate these sums or create a report.

### Restore a checkpoint

`to_json()` includes geometry, optimizer, the complete fixed recipe and accepted
update cursor. The base checkpoint and encoded documents are deliberately not
duplicated. Keep them plus the checkpoint's separately retained `checkpoint_id`.
Restoration verifies the outer bytes, both nested owners, exact native base ID,
full document/label content ID and cursor/optimizer-clock consistency.

`data_id` includes document order, every token and label (including dropped
tails), explicit-versus-implicit label mode and the window plan. It is not a
tokenizer identity, rights grant, authorship proof or training provenance.
Hashes do not authenticate a caller who supplies both modified bytes and a new
pin. Exact continuation is tested within one runtime, not promised across
software versions, devices or architectures. Optimizer clocks may already be
nonzero when a fresh run is constructed; that starting revision and every slot's
starting clock are retained. The shared optimizer verifies that each trainable
placement advanced uniformly by the accepted nonzero-rate update count.

Input budgets are 1,048,576 documents, 16,777,216 total token IDs and the existing
native plan limit of 1,048,576 windows. Context/model/geometry work is admitted
before Python token extraction. Labels use the same lengths; this is an in-memory
trainer, not a streaming or distributed corpus service. Checkpoint JSON is capped
at 128 MiB, with existing nested artifact/optimizer caps still enforced. These are
array/work bounds, not total-memory guarantees.

## Local Files

The [offline example](../../bindings/st-py/examples/native_geometry_corpus_training.py)
accepts `{"token_documents": [[...], ...], "labels": [[...], ...]}`; omit labels
for ordinary next-token supervision. Tokenization is external and explicit.
It uses local `config.json` / `model.safetensors`, a pinned initial artifact and
an explicit JSON recipe containing context/batch/accumulation/epoch/rate/decay/
smoothing fields plus `max_gradient_norm` (a positive value or `null`).
Source builds after 0.4.33 additionally accept optional `max_log_radius_drift`
in the **fresh-run** recipe to enable [per-site projected AdamW](native-geometry-adamw.md#bound-wavegate-radius-drift-source-after-0433).
On resume the optimizer restores that policy from its pinned bytes, not a new recipe.

```sh
python -I bindings/st-py/examples/native_geometry_corpus_training.py \
  --model-dir "$MODEL" --data-json "$DATA" --recipe-json "$RECIPE" \
  --artifact "$GEOMETRY" --artifact-id "$ARTIFACT_ID" \
  --output "$NEW_PRIVATE_RUN" --save-every 10
```

Resume uses `--resume "$CHECKPOINT" --checkpoint-id "$PIN"` instead of the
recipe/artifact arguments. `--stop-after` is an absolute accepted-update boundary;
it does not shorten or rewrite the sealed run recipe. Always use a new output
directory. Checkpoints are saved at complete optimizer boundaries, not during
pending accumulation. The example stops on I/O failure; it is not a durable
transactional store or a guarantee against power loss during file writes.
On POSIX the new run directory uses mode `0700`, independent of a permissive
process umask. This is local access restriction, not encryption or a Windows ACL
guarantee; retain inputs and backups with equivalent care.

### Keep an observed recovery point (source after 0.4.33)

Add `--observe-every 32` to the command above to observe updates 32, 64, ...
in the **original** run, even across resume. The initial/resumed cursor is not
observed again. Each selected update is trained **once** and saves its full
native post-update checkpoint, even between ordinary saves. Overlapping
save/observe/stop boundaries write one checkpoint, not two. Budget disk space
accordingly. Without the flag there is no observation file or change to training.

The same operation is importable. For a private directory created by the caller:

```python
from spiraltorch.native_corpus_runs import step_and_save_observation, restore_observed_checkpoint

# Use this INSTEAD of trainer.step() at a chosen update.
step, entry = step_and_save_observation(trainer, private_run_dir)
# Record entry["observation_id"] separately if independently pinned evidence is needed.
restored = restore_observed_checkpoint(
    private_run_dir, step.update, expected_observation_id=entry["observation_id"],
    base=base, token_documents=token_documents,  # pass the original labels if used
)
```

The client closes the checkpoint before publishing the observation receipt;
the CLI then appends its pin to `observed-checkpoints.jsonl`. A complete index
line identifies a pair even when `result.json` was never reached. Pick an
explicit update and pin; never infer a recovery point from the largest
checkpoint filename or a torn index line. The reader checks the receipt hash
and Rust checks the original base, ordered data and checkpoint bytes. It does
not load an optimizer recipe from the observation. A lone checkpoint or
`.pending-*` file is not a completed pair. On failure stop and use a **new**
directory for continuation, not a retry against those filenames.

These local files need a private, single-writer directory and a filesystem
supporting hard links. Receipts contain per-site measurements; keep them
private along with the optimizer/geometry state. File pins identify bytes,
not authors or past execution; the example does not guarantee recovery after
power loss, and does not upload, prune or auto-select checkpoints.

For [inference from the same saved pair](native-generation.md#generate-from-a-specific-observed-update-source-after-0433),
`load_observed_model` verifies the recorded boundary without requiring or
validating training documents. That is intentionally distinct from a data-verified resume.

The [offline process-cut probe](../../bindings/st-py/tests/probe_observed_checkpoint_recovery.py)
uses a deterministic two-block test model, not the book-study weights. Given
an interpreter with the reference wheel and one with this source build, it
compares all four ordinary step reports and checkpoints, cuts the observed
process at update two, and restores into a third process. No GPU, network
fetch or model-quality benchmark is involved:

```sh
python3 -I -B bindings/st-py/tests/probe_observed_checkpoint_recovery.py \
  --reference-python "$REFERENCE_PYTHON" --candidate-python "$CANDIDATE_PYTHON" \
  --output "$NEW_PRIVATE_DIR"
```

The optional HF Trainer remains an explicit interoperability path for its
supported models and policies. Neither client silently falls back to the other.

## Native Corpus Evaluation

Source builds after 0.4.32 also expose read-only evaluation on either the frozen
base or a trained geometry snapshot, with no HF Trainer or Torch dependency:

```python
score = trainer.model.evaluate_documents(
    heldout_token_documents, 128, microbatch_windows=2,
    # labels=heldout_unshifted_labels,  # Optional; -100 excludes a loss term.
)
print(score.nll, score.perplexity, score.target_tokens)
```

Rust computes **unsmoothed** next-token NLL and reduces batch losses by their
actual selected-target counts, never an unweighted average of batch means.
Perplexity is `exp(nll)`, or `None` if exponentiation would overflow f64; a
non-finite NLL is an error. With custom labels this scores those labels, not an
assertion about the original text's likelihood. Evaluation neither updates
parameters/optimizer clocks nor advances the training cursor. Inserting it
between native training steps does not change the checkpoint recipe.

Evaluation deliberately differs from the disjoint, drop-tail training plan:
each window has at most `context_length` IDs, including its last target, and
adjacent windows share exactly one context token. Every within-document target
at positions `1..len` is covered once, including short tails. Empty/single-token
documents have no targets and are never joined to another document. Positions
and geometric history reset at each window; this is **not** full-history or
maximal-overlap sliding-window perplexity. Compare models using the same policy
and context length. All IDs/labels are validated, even first labels or lone IDs
that are not scored. All-masked windows are counted in `skipped_windows`; an
entire corpus without selected targets is rejected before model execution.

Only equal-width consecutive windows share a microbatch; tails are not padded.
The result's `to_json()` retains the config, base/geometry identities, plan/data
IDs, active `readout_accumulator` (`f32` or opt-in `f64`), window counts,
selected/ignored target counts and executed microbatches. The score also exposes
`.readout_accumulator`; scoring follows the model's selected policy without
changing it. Evaluation always excludes the trainer's label smoothing and
optional repetition-unlikelihood auxiliary objective. It remains plain NLL
when inserted between repetition-training steps or after checkpoint restoration.
The data ID binds exact documents, labels and window policy, including unscored
content, but not batch size. Different batch sizes can change floating-point
rounding; no across-engine bitwise score guarantee is made. These identities do
not certify that supplied documents are independent of training data.

Admission checks at most 1,048,576 documents/windows and 16,777,216 transported
IDs, plus the model's context/geometry and exact selected-logit budgets.
The corpus is copied at the Python boundary; native execution releases the GIL.
This is bounded in-memory evaluation, not an unbounded streaming loader or an
RSS guarantee. Execution uses the existing `rust_f32_cpu_prepacked` decoder;
there is no GPU fallback or speed claim. The Rust API uses
`CausalLmEvaluationConfig` and `NativeCausalLm::evaluate_documents`.

To evaluate a separate encoded corpus during the packaged training example,
add `--eval-data-json "$HELDOUT" --eval-every 10` to the command above. It uses
the training context/batch size but always unsmoothed evaluation NLL. Evaluation
runs before the first update, at the requested interval (default `save-every`)
and at the stopping boundary, and writes `evaluation.jsonl` under the private
run directory. The supplied evaluation data is validated/scored before creating
that directory. The holdout is separate caller state, not embedded into a
trainer checkpoint; supply it explicitly again on resume. There is no automatic
best-checkpoint selection, early stopping, data-split inference or claim that
lower training/evaluation loss proves improved generation quality.

[Local evaluation qualification](native-corpus-evaluation-validation.md) records
the complete four-condition GPT-2 comparison and the optional 30-case geometry
reference suite, separately from training quality and speed claims.
The later [stack integration qualification](../../benchmarks/results/2026-10-19-native-language-integration/README.md)
adds active readout-policy, repetition-training and saved-generation combinations.

## Qualification Boundary

The [one-chapter GPT-2 qualification](../../benchmarks/results/2026-10-18-native-corpus-training/README.md)
completed 20 native updates and reproduced the remaining 18 updates and final
checkpoint bytes in a new process. The original strict f32 Torch parameter
comparison failed despite matching losses within its gate. Same-input f64
diagnostics localize a substantial loss-gradient precision contribution; this
is not a claim of across-engine bitwise training parity or language quality.
A separately sealed post-failure control, using only float64 reference CE with
the same f32 decoder/geometry/AdamW, passed all 20 updates under the unchanged
loss and parameter gates. The original f32 failure remains published alongside it.
