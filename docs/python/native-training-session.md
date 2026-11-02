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
group. To train LoRA through an unchanged geometry, use the optional mask:

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
batched sessions use v2, seal their batch size, and resume from the same
epoch-local batch boundary. `window_start` and `window_end` report the actual
half-open window range; `target_tokens` includes all shifted targets in it.

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

The existing `NativeCausalLmTrainer` continues to provide its geometry-only
AdamW/accumulation/objective path. This SGD session does not replace its optimizer
or duplicate the observation/receipt recovery work in PR #2301.

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
v1/v2 payload in the destination directory, syncs it, publishes it **without
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
one unchanged v1/v2 checkpoint at the **new** path, then advances the live
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
retained `--expected-id`. It does not download a model or tokenize text.

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
composition, joint AdamW, masked labels or repetition objective is promised.
