# One native adaptation contract in Rust and Python

This source API builds on the native GPT-2 LoRA path. It is not a statement about
APIs present in an already published wheel. Rust owns the decoder, loss,
parameter pullback, SGD update, document window plan, cursor and checkpoint.
Python only validates/transports inputs and calls those same owners. No PyTorch,
NumPy, Transformers or Python optimizer is needed.

## Composition and update modes

One LoRA adapter may occupy `transformer.h.N.attn.c_proj`, followed by one
geometry placement at `transformer.h.N.attn` in that same block:

`frozen projection + LoRA -> geometry -> residual addition`

`lora`, `geometry` and `joint` select which parameter group is updated. A frozen
group still participates in the forward computation and, where required, the
pullback. Joint gradients are computed from one immutable pre-update model;
both new parameter groups are validated before the new model is returned.
The frozen base and all previous model snapshots remain unchanged.

Low-level `adaptation_loss_gradients` and `step_adaptation_sgd` use the same
contract in both languages. Sealed gradients include both adaptation owners,
the frozen base identity and readout policy; stale gradients cannot be applied
after changing either group. The older single-group gradient methods remain
available for single-group models. Use the adaptation methods for composition.

By default the low-level call still returns gradients for **every** attached
group. To train LoRA through an unchanged geometry, select the update mode:

```python
g = model.adaptation_loss_gradients(window_ids, [1, len(window_ids)], mode="lora")
next_model = model.step_adaptation_sgd(g, "lora", 0.01)
assert next_model.geometry_artifact_id == model.geometry_artifact_id
```

Rust exposes `adaptation_loss_gradients_for(windows, shape, mode)`. Sessions
select from this same operation automatically. A partial gradient cannot be
used for a group it did not compute. Geometry-only stops at the first geometry
site without differentiating LoRA; LoRA-only still applies the geometry's input
adjoint. For standalone geometry, `batch.vjp_input(upstream)` (or Python's
`vjp_input_buffer`) returns that adjoint alone. Native affine, WaveGate,
Topos, elliptic and fractional recipes skip the unrequested parameter reductions:
an overflow confined to a discarded gate gradient does not invalidate a finite
input result. This is not a throughput benchmark.

## Shared document session

`NativeCausalLmTrainingSession` is a fixed-order, document-safe SGD session.
Each update consumes one planned window by default. To compute **one** mean-loss
SGD update over independent rows, pass `batch_windows=N`, or call Rust
`NativeCausalLmTrainingSession::new_batched(model, documents, config, n)`.
Each row keeps its own context; the last batch of each epoch may be smaller.
The session neither adds EOS nor shuffles, accumulates across updates or invents
a scheduler. SGD is stateless: there are no hidden momentum buffers.

Batching is opt-in and changes the learning trajectory: two serial SGD updates
are not one update over two windows. The default retains v1 checkpoint bytes;
unlabeled disjoint batched sessions use v2, seal their batch size, and resume from the same
epoch-local batch boundary. `window_start` and `window_end` report the actual
half-open window range; `target_tokens` counts the scored shifted targets in it.

Given an existing native base model and compatible geometry artifact:

```python
import spiraltorch as st

model = base.with_lora(0, 2, 4.0).with_geometry(geometry)
session = st.NativeCausalLmTrainingSession(
    model, token_documents, 16,
    learning_rate=0.01, mode="joint", epochs=2,
)
report = session.step()
checkpoint, checkpoint_id = session.checkpoint_with_id()
resumed = st.NativeCausalLmTrainingSession.from_json(
    base, token_documents, checkpoint, checkpoint_id,
)
while (report := resumed.step()) is not None:
    print(report.update, report.loss, report.mode)
trained_model = resumed.model
```

### Train on answers, not prompts

Keep the prompt and answer in **one tokenized document**. Supply one unshifted
label per input ID; `-100` excludes a target from the mean cross-entropy, not
from causal attention. The first label of each window is never scored. Given
a local, lossless tokenizer with character offsets, let Rust align the targets:

```python
from tokenizers import Tokenizer  # optional local text codec, not needed for pre-encoded IDs

tokenizer = Tokenizer.from_file("model/tokenizer.json")
prompt, answer = "Question: What is two plus two?\nAnswer:", " four."
text = prompt + answer
ids, row_labels = st.encode_supervised_document(tokenizer, text, [(len(prompt), len(text))])
docs, labels = [ids], [row_labels]
session = st.NativeCausalLmTrainingSession(
    model, docs, 32, mode="joint", learning_rate=0.01,
    labels=labels, window_policy="covering_targets",
)
steps, pin = session.advance_to_checkpoint_new(1, new_checkpoint_path)
resumed = st.NativeCausalLmTrainingSession.from_json(
    base, docs, new_checkpoint_path.read_bytes(), pin, labels=labels,
)
```

`covering_targets` retains short documents and tails, sharing exactly one
context ID across adjacent windows; it does **not** carry the whole previous
prompt across a window boundary. It batches adjacent equal widths only, without
padding, reordering or crossing documents within a row. An all-ignored batch
does not issue an optimizer update; an entirely unsupervised plan is rejected.
With no option the existing disjoint full-window policy is unchanged.

The mask and window policy are independent: explicit labels or covering windows
use v3. Its data ID binds all IDs, labels (even ignored/dropped positions) and
the window plan; no documents or labels are embedded. Resuming requires the
original label presence and content. Explicit labels equal to the IDs give the
same updates as implicit labels under the same plan, but a different data ID.
Use `labels=` on `adaptation_loss_gradients` or standalone `lora_loss_gradients`
for the same loss on flattened rows. In Rust use `new_with_policy` and
`prepare_restore(...)?.resume_with_labels(...)`.
Both CLI examples also accept a document JSON object with exactly
`token_documents` and `labels`; the Python example accepts `--window-policy`
for a fresh session. Legacy unlabeled input remains a bare JSON array.

**Starting with text.** Pass any ordered, nonoverlapping list of half-open
target spans (e.g. all assistant turns) in Python Unicode character positions.
`encode_supervised_document` encodes the whole string once without adding special
tokens. Use a deterministic tokenizer: configured padding, truncation and BPE
dropout are rejected before encoding. Each contiguous group of context or
target IDs must decode to the exact original substring. Tokens split across one Unicode character are fine;
a token shared between prompt and answer is **not** relabeled or split. A gap,
zero-width token, truncated input, a scored first token without a predecessor,
or a lossy decode fails before learning. Choose a separator that makes the
boundary unambiguous, then re-encode the whole example. If the model requires
special IDs with no source offset, provide explicitly prepared IDs and labels.

The Python training example can read
`{"examples":[{"prompt":"Q: What is two plus two?\nA:","completion":" four."}]}`.
Alternatively, each record may contain `text` and `target_spans`. Supply
`--tokenizer-json FILE --tokenizer-sha256 HEX` alongside the usual options;
only this text path needs the optional `tokenizers` package. It uses
`covering_targets`, including on resume. Keep the tokenizer pin with the run:
the session binds the resulting IDs and labels, not the source text, chat
template or tokenizer file. Select a known tokenizer for the base model; a
file hash identifies bytes, not vocabulary compatibility. Training and
evaluation on encoded IDs never need this dependency.

For a custom encoder, Rust's `st_nn::language_supervision::CausalTargetAlignment`
and Python's `st.CausalTargetAlignment` take the source length, IDs, full
monotone source offsets and target spans. They trust the client's offsets; only
the text helper checks the decoded text. WASM (`nn` feature) exposes the same
operation as `alignCausalTargetsJson`, with JSON fields `source_length`,
`token_ids`, `token_offsets` and `target_spans`. Use one coordinate system:
JavaScript's UTF-16 `text.length` is *not* the number of Python characters.
The output includes ordinary numeric IDs, unshifted labels and runs ordered as
`[token_start, token_end, source_start, source_end]`. The native per-document
limits are 1,048,576 tokens/source units and 65,536 target spans; WASM's JSON input
is also limited to 8 MiB. An alignment is not a checkpoint or a tokenizer identity.

The corresponding Rust calls use the same serialized contract:

```rust
use st_nn::causal_lm::{
    CausalLmAdaptationMode, CausalLmTrainingSessionConfig,
    NativeCausalLmTrainingSession,
};

let model = base.with_lora(0, 2, 4.0)?.with_geometry(geometry)?;
let config = CausalLmTrainingSessionConfig {
    context_length: 16,
    epochs: 2,
    learning_rate: 0.01,
    mode: CausalLmAdaptationMode::Joint,
};
let mut session = NativeCausalLmTrainingSession::new(model, documents.clone(), config)?;
let report = session.step()?;
let (checkpoint, checkpoint_id) = session.checkpoint_with_id()?;
let mut resumed = NativeCausalLmTrainingSession::from_json(
    base, documents, checkpoint.as_bytes(), &checkpoint_id,
)?;
while let Some(report) = resumed.step()? {
    println!("{} {}", report.update, report.loss);
}
let trained_model = resumed.model().clone();
```

Each checkpoint includes both attached adaptation states, their identities, the
frozen base identity, document-content identity, cursor, config and readout
policy. Supply the separately retained original base weights and documents when
restoring. The saved readout policy is restored automatically. The caller must
retain the checkpoint ID separately: a checksum recomputed from untrusted bytes
is not proof that those bytes are the intended checkpoint.

Rust writes the JSON consumed by Python, and Python returns the same Rust-written
JSON consumed by Rust; there is no Python-specific checkpoint schema. Checkpoint
validation is bounded, rejects mismatched documents/base and commits no partial
session state on failure. The format is versioned independently of the older
geometry/AdamW trainer checkpoint. It is not an import path for arbitrary PEFT
adapters or a cross-version compatibility promise.

### One completion, three uses

In a development wheel with `CausalCompletion`, the prompt, answer and optional
literal terminator have distinct roles. Train on answer + terminator, measure
answer content only, and generate from the independently encoded prompt alone:

```python
import spiraltorch as st

fit = st.encode_supervised_completion(tokenizer, "Question: 2+2?\nAnswer:", " four", terminator="\n")
ids, labels = fit.training_document()
session = st.NativeCausalLmTrainingSession(model, [ids], 64, labels=[labels], mode="joint",
                                          window_policy="covering_targets", learning_rate=0.01)
session.step()  # model already has trainable geometry and LoRA

held = st.encode_supervised_completion(tokenizer, "Question: 3+3?\nAnswer:", " six", terminator="\n")
content, targets = held.evaluation_document()
score = session.model.evaluate_documents([content], 64, labels=[targets])
gen = st.NativeCausalLmGenerator(session.model, held.prompt_ids, '{"max_new_tokens":8}')
generated_ids = gen.advance_ids(8)
```

No separator, BOS, EOS or stop rule is inferred. Choose a generation budget and
stop policy independently of the reference answer. Without a reference at serving
time, encode only the incoming prompt and give those IDs to the same generator.
The text helper checks full-source offsets, lossless decoding and exact prefix-ID
agreement for separately encoded prompt and prompt + answer. A merge across
either boundary is rejected, not truncated. For multi-turn or arbitrary spans
keep using `encode_supervised_document`.

Already have IDs? `st.CausalCompletion(prompt_ids, prompt_answer_ids, training_ids)`
is tokenizer-free. The three sequences include the prompt; the second ends at
the answer, the third may add a terminator. This proves the prefix relationship,
not the provenance of those IDs or their vocabulary compatibility. The local
training CLI opts into the text checks with `{"prompt":"...", "completion":"...", "terminator":"\n"}`;
the older two-field training-only format is unchanged.

Rust exposes `st_nn::language_supervision::CausalCompletion::from_tokenizations`
and `from_alignment`. WASM with the `nn` feature exposes `prepareCausalCompletionJson`
with the same three ID fields and optional `source: {length, offsets, answer_span}`.
Its compact response contains `training_ids`, `training_labels`, `prompt_end`
and `answer_end`: slice IDs before `prompt_end` for generation, and both arrays
before `answer_end` for evaluation. These are token indices, not string offsets.

### Train matched variants without repeating the frozen prefix

To separate the effect of *applying* geometry from the effect of *learning* it,
start three arms from the same LoRA parameters. Give the last two the exact same
geometry artifact (for example a WaveGate on the final attention block):

```python
Session = st.NativeCausalLmTrainingSession
lora = base.with_lora(layer, 2, 4.0)
both = lora.with_geometry(geometry)
control = Session(lora, docs, 32, mode="lora", learning_rate=0.01,
                  labels=labels, window_policy="covering_targets")
arms = [control, control.fork_variant(both, "lora"), control.fork_variant(both, "joint")]
while (update := Session.step_matched(arms)) is not None:
    shared_layers, reports = update  # reports follow input order

scores = st.NativeCausalLm.evaluate_variants_by_answer(
    [arm.model for arm in arms], eval_docs, 32, labels=eval_labels,
)
wave_learning = scores.contrast(reference=1, candidate=2)
```

See [paired answer scores](native-corpus-training.md#compare-the-same-answers)
for target- versus answer-weighted deltas and scoring pinned arms from disk.

`fork_variant` is only for an unadvanced session. It shares immutable corpus
storage and the window schedule, while checking the new model's own shape and
selected-logit limits. It does not copy trained weights from the session to
the supplied model. Independent sessions (including separately pinned resumes)
can also use `step_matched`. They must agree on the frozen base, readout,
data ID including labels, window policy, batch size, epochs, learning rate and
cursor. The API does **not** assert identical initial predictions or parameter
counts across arms; choose explicit starting controls for a scientific comparison.

One call embeds one batch and executes blocks strictly before the earliest
**attached** adapter once, even if that adapter is frozen in an arm. Each arm
then computes its own suffix, loss, backward replay and SGD update. It returns
the actual shared depth and ordinary step reports; at exhaustion it returns
`None`. This is not a persistent KV cache. A scheduling or numerical failure
leaves every arm's weights and cursor unchanged. Save each arm using its
ordinary v1/v2/v3 checkpoint and externally retained pin. The in-memory
transaction is **not** a multi-file checkpoint transaction.

Rust exposes `fork_variant(model, mode)` and
`NativeCausalLmTrainingSession::step_matched(&mut [&mut control, &mut joint])`.
For the fixed synthetic CPU benchmark, run
`python -I -B tools/bench_native_matched_sgd.py --output NEW_DIRECTORY`
from a checkout with its development wheel installed. It compares complete
three-arm updates with independent steps on the **same** wheel, and saves
every timed pair as well as its report/checkpoint identities. It does not
benchmark PyTorch, GPU training or language quality.

The existing `NativeCausalLmTrainer` continues to provide its geometry-only
AdamW/accumulation/objective path. This SGD session does not replace its optimizer
or checkpoint schema. Both use the same no-overwrite segment commit; see
[geometry AdamW and observed boundaries](native-corpus-training.md#keep-an-observed-recovery-point-source-after-0433)
for persisting moments and the final update's site measurements together.

### Score or generate without the training documents

Source builds after 0.4.33 can export one serialization and its exact digest:
`payload, pin = session.checkpoint_with_id()`. Retain the pin, completed update,
recorded data ID and both post-update adapter IDs separately. A scorer with the
bare matching base can then use the same snapshot, without shipping the corpus:

```python
model = st.NativeCausalLm.from_session_checkpoint_at(
    base, payload, pin, update=update, recorded_data_id=data_id,
    post_lora_id=lora_id, post_geometry_id=geometry_id,
)
score = model.evaluate_documents(held_out, 64, microbatch_windows=2)
```

`None` means that group **must be absent**. Both IDs, the cursor and the
recorded data ID must match; the saved readout policy is restored. This is
not a claim that the scorer verified the actual training data or that the
session exhausted its planned windows. Use `from_json` with the documents
to continue training. Rust exposes the same `from_session_checkpoint_at`.
The binding accepts immutable `bytes` directly or strictly encoded UTF-8
`str` up to the native session limit (128 MiB). The saved geometry must admit
the minimum training shape implied by the recorded cursor, even without access
to the original corpus.

For local files, pass `--session-checkpoint CHECKPOINT --session-checkpoint-id PIN
--session-boundary RESULT.json` to `bindings/st-py/examples/native_geometry_generation.py`
alongside its ordinary model, prompt, recipe and output arguments. The training
example's JSON result contains the required five boundary fields; a compact
copy of those fields without step reports also works. The checkpoint pin must
come from the caller's retained selection, not be recomputed from the file
being opened. No training document path is used by generation.

## Local-file example and verification boundary

Rust and Python expose `session.save_checkpoint_new(path)`. It stages the exact
session payload in the destination directory, syncs it, publishes it **without
overwriting an existing name**, and returns the same pin as `checkpoint_with_id`.
Keep that pin separately. Use an existing directory under your control; on
Unix the filesystem must support hard links and directory sync. A write error
before publication leaves the final name absent and earlier checkpoints intact.
An error after publication (for example, directory sync failure) can leave a
complete file: compare its bytes to the intended pin rather than retrying or
deleting it. Hard process termination can leave a temporary file, not a new
guarantee about unsaved updates. A step log is not a resumable checkpoint.

For long LoRA, geometry or joint SGD runs, Rust and Python also expose
`advance_to_checkpoint_new(stop_after, path)`. The cursor is absolute: choose
between 1 and `MAX_CHECKPOINT_INTERVAL` (1,024) remaining updates per call.
It performs the same ordinary steps on a private session snapshot, publishes
one checkpoint in the session's own format at the **new** path, then advances the live
session. It returns the ordered step reports and the pin for the written bytes:

```python
from pathlib import Path

reports, pin = session.advance_to_checkpoint_new(16, "u16.json")
# Record the pin independently before starting the next segment.
resumed = st.NativeCausalLmTrainingSession.from_json(
    base, token_documents, Path("u16.json").read_bytes(), pin,
)
```

An invalid cursor, failed gradient or failed native publication leaves the
live cursor and adaptation weights unchanged. A late filesystem error may
leave a complete orphan at the new name; do not overwrite or automatically
adopt it. The previous independently pinned checkpoint remains usable. There
is no hidden retry or extra per-step file transfer.

`bindings/st-py/examples/native_training_session.py` accepts a local GPT-2
configuration, local safetensors archive, encoded token documents and an optional
geometry artifact. Its checkpoint can be resumed with `--resume` and a separately
retained `--expected-id`. It does not download a model or tokenizer.

Add `--checkpoint-every 16` to publish an initial anchor (or use the supplied
resume pin), every absolute multiple of 16 and the final requested cursor.
Intermediate files are named `OUTPUT.u000000016.json`; the final checkpoint
is `OUTPUT`. The new, private `OUTPUT.progress.jsonl` is flushed and synced
after each saved boundary and before the next segment. Each complete line
contains its checkpoint pin, cursor, adapter IDs and that segment's ordinary
reports. The final stdout summary points to the journal rather than keeping
every past report in memory. Without this flag, the original single-output
CLI behavior is unchanged. No checkpoint is pruned.

After interruption, use a **known complete** boundary record and its pin with
`--resume FILE --expected-id PIN --output NEW_FILE`. Do not infer state from
a partial journal line, the largest filename or an unindexed checkpoint. If
writing the journal fails *after* a checkpoint was published, the example
stops without starting another segment. Keep pins separately if the directory
itself is not trusted.

The Rust `native_training_session` example accepts six positional paths/values:
`CONFIG WEIGHTS DOCUMENTS CHECKPOINT EXPECTED_ID OUTPUT`. It restores one checkpoint,
performs one update, writes a new checkpoint file, and prints the update and
new ID as JSON. Neither example silently overwrites an existing checkpoint.
An optional seventh argument, `STOP_AFTER`, commits a bounded segment to that
absolute update and prints its ordered `steps`.
To exercise the Python → Rust → Python handoff explicitly after building the
local extension:

```bash
cargo build -p st-nn --example native_training_session
ST_NATIVE_SESSION_EXAMPLE="$PWD/target/debug/examples/native_training_session" \
  python bindings/st-py/tests/test_native_causal_lm_session.py
```

Without that environment variable only the cross-language executable test is
skipped with an explicit reason; the fresh-Python-process restore test still runs.

Tests use small deterministic fixtures to check group freezing, ownership,
continuous-versus-restored training, f32/f64 policy and fresh-process generation
with optional ML imports forbidden. These establish mechanics, not language
quality or speed. This slice is CPU, one LoRA projection and one same-layer
attention geometry; no GPU residency, full-model updates, arbitrary placement
composition, joint AdamW, label smoothing or repetition objective is promised by this session.
