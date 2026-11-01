# Native GPT-2 attention-output LoRA

This source-tree feature connects a low-rank adapter to one GPT-2 attention output
projection, `transformer.h.N.attn.c_proj`. It is newer than the v0.4.33 release.
Rust owns forward execution, language-loss gradients and updates; Python only
transports inputs and holds returned owners. The base GPT-2 weights remain frozen.
This is CPU execution, not a resident WGPU path or a PEFT checkpoint importer.

## One update and reload

Given a `NativeCausalLm` loaded from a supported local GPT-2 checkpoint:

```python
model = base.with_lora(layer=0, rank=2, alpha=4.0)
window = [1, 2, 3, 4]  # Exact IDs from your tokenizer; arbitrary IDs only demonstrate mechanics.
gradients = model.lora_loss_gradients(window, [1, len(window)])
updated = model.step_lora_sgd(gradients, learning_rate=0.01)

payload = updated.lora_to_json()
restored = base.with_lora_json(payload)
assert restored.lora_artifact_id == updated.lora_artifact_id
```

Initialization uses a fixed seeded ChaCha8 stream for small A values and zero B, so the initial adapter has zero
delta. `alpha / rank` scales the low-rank branch. Layers and ranks are validated;
only one attention output projection is admitted. The gradient object exposes
`loss`, `target_tokens`, `window_shape`, `base_model_identity`,
`lora_artifact_id`, and `parameter_gradients()` (`a` and `b`). Gradients are sealed
to the exact base, adapter and readout policy; stale gradients are rejected.
Updates return new immutable models, leaving the original base and adapter intact.

The objective is mean next-token cross entropy over all shifted targets of full
windows. This initial API does not accept masked labels, label smoothing,
repetition objectives, or geometry co-training. Existing geometry-only APIs keep
their separate behavior; attempting to attach geometry and LoRA together fails.

## Continuing learning

The JSON contains adapter parameters, shape, placement, scale and base identity.
Reload the same frozen base in another process, attach the JSON, then supply the
next window and learning rate to `lora_loss_gradients` and `step_lora_sgd`.
For the same native execution conditions, this reproduces continuing without a
save/reload boundary. The caller owns the data cursor, learning-rate schedule and
readout policy. The adapter JSON does not store `readout_accumulator`; restore the
same `f32` or `f64` policy with `base.with_readout_accumulator(...)` before learning.
Stateless SGD has no hidden moments; this is **not** a persistent document trainer
or an AdamW optimizer checkpoint. Retain your next-window index and schedule
alongside the adapter if your application needs automatic continuation.

`lora_artifact_id` identifies canonical adapter content, not its authenticity.
Retain and compare an expected identity when transferring an adapter. Base identity,
configuration, tensor lengths, finite values and artifact size are checked on load.
No base tensors or tokenizer are embedded in the adapter JSON.

## Generation and scoring

Pass the restored model to the existing `NativeCausalLmGenerator` or
`NativeCausalLmContext`. Both full-prefix and cached execution apply the adapter.
Contexts own an immutable model snapshot. A later training update does not alter
an already-created context.

Generation checkpoints pin the adapter identity as well as base, geometry and
readout policy. Restore with the exact adapter snapshot; a newer adapter cannot
silently continue an older generation checkpoint. No-LoRA checkpoints omit the
optional identity field and retain their previous serialization. Evaluation
reports include `lora_artifact_id` when present.

The [local-file example](../../bindings/st-py/examples/native_lora.py) performs one
update, writes/reloads the adapter, and generates one token:

```bash
python bindings/st-py/examples/native_lora.py \
  /path/to/config.json /path/to/model.safetensors /path/to/window.json adapter.json
```

Use `--resume-adapter adapter.json` to start from a saved adapter and `--readout`
to retain the same policy. Neither flag recovers a dataset cursor. No remote files
or optional ML packages are needed. Tiny fixture tests cover gradients, frozen
base, adapter ownership, fresh-process SGD continuation and generation; these
mechanical checks do not establish language quality, speed, or superiority over
geometry learning.
