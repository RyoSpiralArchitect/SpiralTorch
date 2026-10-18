# Geometric Fine-Tuning Through HF Trainer

[Language MLOps](../language_mlops.md) | [Portable state](geometry-artifacts.md)

**Source-candidate API after 0.4.31, not a released wheel.** Build/install this
source wheel, then install its `hf-geometry` extra. The Trainer integration is
qualified against Transformers **4.57.6**, Torch >= 2.6 and Accelerate; other
Transformers versions fail explicitly because checkpoint hooks are private APIs.

This is actual geometry-only fine-tuning of an existing frozen causal LM, not
telemetry injection. `st.hf_geometry_trainer_class()` reuses the HF training loop:
optimizer discovery, accumulation, clipping, scheduler, evaluation, and resume.
Rust still owns geometric operators/VJPs and portable state validation. Python
does not implement a second optimizer loop or geometry formula.

## Train

The caller supplies a verified float32 base model, its runtime identity, and
train/evaluation datasets. Each item must contain `input_ids` and `labels` for
one complete, unpadded, single-document prefix. Use equal-length batches and
all-one attention masks. Do not pack unrelated documents into a sequence.
Route to explicitly selected tensor-returning modules; the GPT-2 path below is
an example, not an architecture heuristic.

Use [HfGeometryDataset](hf-geometry-data.md) for Rust-planned, document-safe
contexts instead of packing text rows together. It plugs directly into the
default collator below and reports dropped tails and actual target coverage.

```python
import spiraltorch as st
from transformers import TrainingArguments, default_data_collator

# model, train_dataset, eval_dataset and verified_base_identity are supplied
# by the experiment executor. Identity is a verified sha256 runtime ID.
model.requires_grad_(False)
model.zero_grad(set_to_none=True)
model.config.use_cache = False
geometry = st.GeometryAdapterStack({
    "transformer.h.0.mlp": st.WaveGateAdapter(
        model.config.n_embd, log_radius=0., learnable_radius=True),
    "transformer.h.1.mlp": st.FractionalAngleGainHistoryAdapter(
        model.config.n_embd, kernel_len=32, initial_angle=.2),
})
Trainer = st.hf_geometry_trainer_class()
trainer = Trainer(
    model=model, geometry=geometry,
    base_model_identity=verified_base_identity,
    args=TrainingArguments(
        output_dir="private-geometry-run", report_to=[],
        learning_rate=1e-3, max_steps=100, optim="adamw_torch",
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        max_grad_norm=.1, save_steps=20,
    ),
    train_dataset=train_dataset, eval_dataset=eval_dataset,
    data_collator=default_data_collator,
)
trainer.train()
metrics = trainer.evaluate()
trainer.save_model("private-geometry-export")
artifact_id = trainer.last_geometry_artifact_id
```

The wrapping model owns the frozen base and geometry as siblings. This makes
HF's standard optimizer **and gradient clipping** see exactly the trainable
geometry. The original base has no adapters registered inside it. Each training
step keeps attachment active through backward, then removes only its own hooks;
prediction/evaluation also attach automatically. `trainer.base_model` and
`trainer.geometry` expose the two owners. Direct wrapper calls outside these
scopes fail rather than silently omitting geometry.
Trainer uses `geometry.attach(base, require_all=True)`: every configured site
must execute in each step. Unused branches fail before the optimizer step or
evaluation result. Standalone attachment retains its optional-site behavior.

For the ordinary causal loss, the HF item counter sees `labels[..., 1:]` so it
does not include the unpredicted first label in the denominator. HF label
smoothing uses HF's causal smoother, weighted by each microbatch's valid shifted
target count so unequal label masks accumulate correctly. At least one shifted
target must remain in each labeled batch. Custom `compute_loss_func` and
pre-shifted targets are not supported in this slice.

## Save, Resume, Infer

`checkpoint-N/spiraltorch-geometry.json` is the Rust-validated portable geometry
artifact. `last_geometry_artifact_id` is its exact-byte hash; retain that hash in
an independently pinned run record at the corresponding save boundary. HF saves
`optimizer.pt`, `scheduler.pt`, `rng_state.pth`, `trainer_state.json`, and training
arguments alongside it. The base weights are **not** copied into the checkpoint.
On MPS, `spiraltorch-mps-rng.pt` also preserves the Torch device generator omitted
by HF 4.57.6; the extended HF RNG hooks save/restore it and require it on resume.
Calling `save_model()` alone is an inference export, not a resumable checkpoint.

To resume, reconstruct the same base and the same initial geometry recipe, make
a new Trainer with the same data, schedule and training configuration, then:

```python
trainer.train(
    resume_from_checkpoint="private-geometry-run/checkpoint-80",
    geometry_checkpoint_id=independently_retained_checkpoint_80_id,
)
```

The required geometry ID, base identity and Rust-owned `layout_id` must match
before live geometry changes. The layout binds ordered paths, recipes, shapes,
parameter/buffer roles and trainability, but excludes learned values and
train/eval mode. It is available from Rust, Python and WASM (`layoutId`). Missing
optimizer/scheduler/RNG/trainer-state companions fail instead of silently
restarting part of the run. HF then restores its ordinary optimizer, cursor and
RNG state. `ignore_data_skip=True` is rejected for resume.

Training uses a finite map-style dataset with epoch-seeded random permutations:
`data_seed` (or `seed`) plus the epoch restored in HF's Trainer cursor. This
preserves the epoch even when Accelerate rebuilds a dataloader to skip batches
inside a later epoch. `group_by_length` and streaming training are rejected;
data/collators must be deterministic for exact replay, including worker output.
Callbacks may stop the whole run at an update boundary; skipping the rest of an
epoch while continuing training is rejected before checkpoint save because it
would invalidate HF's step-derived resume cursor. Arbitrary cursor mutations
and stochastic worker/collator state are outside this replay contract.

For inference, use the [portable artifact loading recipe](geometry-artifacts.md),
reload/verify the same base, put it and the restored geometry into evaluation
mode, then attach for complete-prefix calls with `use_cache=False`. A training
checkpoint and its companion state need not be retained in the serving process.

## Boundaries

- Single-process float32, frozen-base training only. No AMP, quantization,
  device-map sharding, DDP/FSDP, tensor parallelism, compilation, activation
  checkpointing, NEFTune, in-process HP search, or KV caching.
- No automatic `load_best_model_at_end` or Hub upload. Evaluate saved candidates
  explicitly and retain selected artifact IDs. A lower loss is not an automatic
  promotion decision; keep ordinary/frozen controls and generation checks.
- The base ID is a caller-supplied assertion, not a hash recomputed from the live
  model. The surrounding executor must verify base, tokenizer, data/cursor,
  runtime, training recipe and all HF companion files. Geometry's ID does not
  authenticate those files or establish their provenance.
- Only the geometry JSON replacement is file-atomic. The multi-file HF
  checkpoint is not a crash-atomic sealed bundle. Resume only from complete,
  trusted checkpoints; do not load arbitrary untrusted HF pickle companions.
- Rust geometry currently executes on CPU with Torch buffer transfers. Workflow
  tests do not qualify MPS/CUDA performance, full GPU residency,
  distributed recovery, long-run convergence or language-quality gains.

The [qualification](../../benchmarks/results/2026-10-18-hf-geometry-trainer/README.md)
uses local tiny GPT-2/Llama models, dropout, accumulation, active clipping and
fresh-process continuation. No model or corpus download is needed.
