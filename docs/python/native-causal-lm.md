# Native Causal Language Models

Source-only extension after the published 0.4.32 release. The immutable release
wheels do not include this decoder or change the optional HF Trainer implementation.

`NativeCausalLm` executes a complete supported decoder from token IDs to
vocabulary logits in Rust. Its first explicit checkpoint importer supports
standard GPT-2: learned absolute embeddings, pre-LayerNorm residual blocks,
fused QKV, structural causal multi-head attention, tanh-GELU MLP, final
LayerNorm, and tied or untied output embeddings. Python is transport and
orchestration, not a second implementation of the transformer.

## Local Checkpoint to Logits

```python
from pathlib import Path
import spiraltorch as st

directory = Path("/your/local/gpt2/snapshot")
weights = st.SafeTensorArchive(directory / "model.safetensors")
model = st.NativeCausalLm.from_gpt2(
    (directory / "config.json").read_text(), weights
)
del weights  # The model owns its frozen native parameter storage.

ids = [464, 3616, 286, 1204, 318]  # GPT-2 IDs for "The meaning of life is".
logits = model.forward(ids, [1, len(ids)])  # NdTensor [B,T,V]
last = model.forward(ids, [1, len(ids)], last_only=True)  # [B,1,V]
next_ids = model.next_token_ids(ids, [1, len(ids)])  # one greedy ID per row
nll = model.next_token_loss(ids, [1, len(ids)])  # T-1 next-token targets
```

The core needs no Torch, Transformers, NumPy or Python safetensors. Tokenization
is external: the input is exact integer IDs, never text implicitly normalized
or mapped through a different vocabulary. `next_token_loss` shifts independently
within each window, never between batch rows. Greedy ties choose the lowest ID.

To compare two possible continuations without copying the vocabulary into Python,
use `model.next_token_contrast(ids, target_id, alternative_id)`. Pass one
unpadded prompt and two distinct vocabulary IDs. Only the prompt is forwarded;
the report gives both raw logits, their signed difference
(`target_minus_alternative`), and the unrestricted `greedy_id`. A positive
margin does not mean either candidate won. See the [paired answer/LF
example](../../benchmarks/paired-prompt-contrast/README.md) for safe tokenization
at a prompt boundary. This API is available in source builds containing it.

When you don't know which other token is most competitive, use
`model.next_token_score(ids, target_id)`. It forwards the prompt once, does not
append the target, and returns a small dictionary instead of a list of logits:

```python
score = model.next_token_score(ids, target_id=262)
print(score["target_rank"], score["strongest_rival_id"], score["target_minus_best_rival"])
```

`target_rank` is one-based over the full vocabulary; exact ties (including
signed zeros) favor the smaller ID. The signed margin is the target logit
minus the best **other** logit, even when the target wins. `nll` is ordinary
full-vocabulary negative log likelihood, not two-class loss. Both are computed
in f64 from the raw f32 logits, with the same normalization used by Rust
classification. There is still one vocabulary projection, not a sampling
shortcut or a change in model precision.

For a retained row of little-endian f32 bytes, `st.score_token_logits_f32le(raw,
target_id)` returns the same fields **without running a model**. WASM exposes
`JSON.parse(scoreTokenLogitsJsonF32(float32Array, targetId))` for a client-owned
row, not a browser GPT-2 decoder. Both transports admit 2 to 1,048,576 finite
logits and reject invalid IDs. Use `forward(..., last_only=True)` and keep the
raw bytes when checking exact replay: equal summaries do not prove equal logits.

Positions restart at zero in each row. No padding, token-type IDs, attention
mask, custom position IDs or dropout are implemented here. Base-weight updates
are available only through the separate, opt-in QKV optimizer described below.
The separate [native context](native-generation.md#cached-execution) adds owned
single-sequence K/V reuse without changing the full-prefix `forward` contract.
Frozen inference and the geometry-only learning below are
not full-model native fine-tuning.

The initial execution backend is `rust_f32_cpu_prepacked`. Layer projections
are packed once; a tied output projection shares the token table storage.
Last-only output skips vocabulary projection for earlier positions, not the
causal prefix computation. F16/BF16 checkpoint values explicitly become f32.
This is not a resident-GPU or speedup claim. The Rust core compiles for WASM;
a JavaScript client and browser model execution require separate work.

## Answer-QKV Learning

Source builds containing `NativeCausalLmQkvAdamW` can update one decoder layer's
fused Q/K/V weight matrix, with the rest of the model and attached LoRA/geometry
frozen. Forward, backward, loss, AdamW, admission and checkpoint ownership stay
in Rust. Python passes token IDs, controls the schedule and stores bytes; no
Torch, NumPy, Transformers or Python safetensors are required.

```python
# Continue with the local NativeCausalLm loaded above.
trainer = st.NativeCausalLmQkvAdamW(model, layer=0)
prompts = [ids, ids + [262]]  # Unpadded prefixes, independently positioned.
answers = [262, 995]         # One final-position target per prefix.
report = trainer.step_answers(
    prompts, answers, learning_rate=1e-4, max_rows=2,
    label_smoothing=0.1,     # Opt-in, not a recommended universal setting.
)
print(report["objective_loss"], report["gradient_l2"])  # Q, K, V norms.
payload, checkpoint_id = trainer.checkpoint_with_id()
restored = st.NativeCausalLmQkvAdamW.from_bytes(model, payload, checkpoint_id)
assert restored.last_logits_f32le(ids) == trainer.last_logits_f32le(ids)
```

`step_answers` mutates the trainer, not `model`, and commits one atomic optimizer
step for the whole logical batch. Each sequence gets one equal vote, regardless
of its length. `max_rows` (default 1) groups adjacent equal-length prefixes; it
does not pad, reorder or split one prefix. There are at most 4,096 sequences and
1,024 actual groups, with additional native work/activation limits checked
before copying prefix token values. This is answer supervision, not all-token
SFT or full-model fine-tuning. The demonstrated execution is CPU, not GPU.

Omit `label_smoothing` for ordinary CE and a `gold_nll` report. An explicit
coefficient, including zero, reports `objective_loss` and `label_smoothing`
instead: the target is `(1-epsilon) * one_hot(answer) + epsilon / vocabulary`.
Smoothing does not change inference logits. The constructor's `epsilon` is
AdamW's numerical stabilizer, **not** the smoothing coefficient.

`answer_gradients` accepts the same batch and optional smoothing without an
update. It returns the corresponding loss, `gradient_shape=(3*width, width)`
and output-major `gradient_f32le` bytes, not HF Conv1D's transposed layout.
`last_logits_f32le` returns one full-vocabulary row of little-endian f32 bytes;
it can be passed directly to `st.score_token_logits_f32le(raw, target_id)`.

The checkpoint binds the frozen source, attached adapters, arithmetic policy,
selected layer, updated weights, AdamW moments and step counter. Restore with
the matching original model. Preserve the next inputs, learning rate, decay,
smoothing and batch order separately to reproduce continuation. `serving_id`
identifies the updated inference state, not the full optimizer or data history.
Invalid inputs do not change optimizer state; zero learning rate freezes it.
Checkpoint file storage and exporting a new full-model archive are not provided
by this wrapper. Source builds also connect QKV snapshots to the existing native
generator for sampling and EOS handling, as described below.

The [binding verification](../../benchmarks/paired-prompt-contrast/PYTHON-QKV.md)
separates local API checks from wheel publication and language-quality evidence.

### Optional QKV Gradient Clipping

Choose a positive, finite f32 `max_gradient_norm` when creating the optimizer
to use the same Rust `GlobalNormClip` contract as the geometry optimizer:

```python
trainer = st.NativeCausalLmQkvAdamW(model, 0, max_gradient_norm=1.0)
report = trainer.step_answers(prompts, answers, learning_rate=1e-4, max_rows=2)
print(report["gradient_clip"])  # max_norm, before_l2, after_l2, ordered factors
```

The value above illustrates the API, not a recommended universal threshold.
The norm covers the entire Q/K/V weight derivative **after** the logical batch
mean, not each microbatch, prefix, head, or Q/K/V partition separately. Rust
scales that effective gradient before AdamW updates its moments. Weight decay
and AdamW's adaptive normalization are not clipped: this is not a direct bound
on parameter displacement. `answer_gradients` still returns the raw derivative,
and the existing `gradient_l2` report stays the raw Q/K/V norms.

The shared rule intentionally leaves norms at or below `f32::EPSILON` unchanged,
and skips factors within `f32::EPSILON` of one. Thus a very small configured
limit is not an unconditional exact projection bound, and this is not a claim
of PyTorch clipping equivalence. Ordered f32 scale factors follow the common
contract rather than a separate Python normalization. Nonfinite gradients are
not repaired by clipping.

`max_gradient_norm` defaults to `None`; that path keeps the previous serialized
reports and native checkpoint bytes. With clipping enabled, the optimizer
checkpoint retains the limit's f32 bits and restore inherits it without an
override. The getter `trainer.max_gradient_norm` exposes that policy. The
inference `serving_id` still identifies serving weights/adapters/arithmetic,
not clipping policy or moments. Zero learning rate leaves optimizer state
unchanged while still computing gradient diagnostics.

[Verification](../../benchmarks/paired-prompt-contrast/QKV-GRADIENT-CLIP.md):
218 native CPU tests and 70 isolated Python tests pass. This verifies the
update/restart contract, not improved convergence or language quality.

### Cached Inference After QKV Updates

`trainer.prefill(ids, max_cache_values=...)` creates an immutable
`NativeCausalLmProjectionPrefix` bound to the trainer's **current** `serving_id`.
It retains the native model snapshot, K/V state and any causal-geometry history.
Calling `branch(token_id)` returns a new prefix with one token appended, without
changing the parent. Python does not implement attention or mutate cache buffers.

```python
prefix = trainer.prefill(ids, max_cache_values=1_048_576)
left = prefix.branch(262)
right = prefix.branch(257)
assert prefix.token_ids == ids
assert left.logits_f32le() == trainer.last_logits_f32le(ids + [262])
print(left.serving_id, left.cache_values, left.max_cache_values)
```

If training subsequently changes the weights, these prefixes still serve their
old snapshot. Start a **new** prefill to serve the new weights; do not treat old
cache logits as outputs of the latest trainer. Restoring a matching optimizer
checkpoint and prefilling again reconstructs serving state; this acceleration
cache itself is not a serialized checkpoint.

The default limit is 16,777,216 retained f32 cache values **per branch**, including
K/V and required geometry history. It excludes frozen weights, logits and
temporary transaction buffers; keeping many branches multiplies memory usage.
Lengths and cache capacity are admitted before Python prefix tokens are copied.
Invalid tokens, capacity exhaustion and position overflow leave the parent
prefix unchanged. Causal geometry may replay its retained input history; this
API is not a zero-copy, GPU or speedup claim.

The [cache verification](../../benchmarks/paired-prompt-contrast/PYTHON-QKV-CACHE.md)
covers ordinary/key-lag attention with frozen LoRA and geometry, all three
readout policies, exact full-prefix agreement and snapshot isolation.
The [saved-model comparison](../../benchmarks/paired-prompt-contrast/PYTHON-QKV-RESTORE.md)
also restores four historical, 256-update synthetic-model endpoints and matches
all 6,144 previously recorded rows. This is artifact/runtime compatibility,
not a new language-quality result or a published-wheel guarantee.

### Generation After QKV Updates

The source API `trainer.generator(ids, config_json)` returns the existing
`NativeCausalLmGenerator`, bound to the trainer's current `serving_id`. It uses
the same Rust-owned greedy/sampling, Z-Space decoding, EOS, repetition rules and
full-prefix/KV-cache execution as ordinary native generation. Python does not
implement a second decoding loop or call Torch/NumPy for the arithmetic.

```python
import json

weights, weights_id = trainer.checkpoint_with_id()
stream = trainer.generator(ids, json.dumps({
    "max_new_tokens": 16,
    "seed": 81,
    "execution": {"kind": "kv_cache", "max_values": 1_048_576},
}))
assert stream.projection_id == trainer.serving_id
first_token = stream.step_token_id()
state, state_id = stream.to_json(), stream.checkpoint_id

# `model` is the same original base with the same frozen adapters and readout.
restored = st.NativeCausalLmQkvAdamW.from_bytes(model, weights, weights_id)
continued = restored.restore_generator(state, state_id)
remaining_ids = continued.advance_ids(15)
```

Choose a generation length and cache capacity that fit the model. Admission
covers the largest evaluated prefix before Python token values are copied.
The final emitted token does not itself require another model evaluation.

The generation JSON retains the token history, decoding policy and exact
`projection_id`; **it does not contain QKV weights or AdamW state**. Preserve
both checkpoints and their expected hashes. Resume through the matching QKV
trainer, not `NativeCausalLmGenerator.from_json(model, ...)`. The ordinary-model
route rejects QKV generation state, and the QKV route rejects ordinary state
or state from a different serving projection. A trainer with different moments
but identical serving weights can resume inference; learning continuation still
requires the optimizer checkpoint and data schedule.

A generator already created from a snapshot stays on those weights after later
training updates. Create a new generator to use new weights. The acceleration
cache is reconstructed on restore, not serialized. Existing ordinary generator
JSON omits `projection_id`, preserving its previous serialization.

[Validation status](../../benchmarks/paired-prompt-contrast/QKV-GENERATION.md):
215 Rust CPU tests, Python-binding compilation and 66 isolated Python runtime
tests passed. The new methods were exercised in a freshly built CPU extension,
including sampled Z-Space continuation. This is local source-build verification,
not a published-wheel, GPU or pretrained-model quality guarantee.

### File-Based QKV Generation

`bindings/st-py/examples/native_geometry_generation.py` can load a saved QKV
checkpoint over the selected frozen model and adapters. The example only
orchestrates file input/output; native Rust still owns checkpoint validation,
sampling, cache execution and random continuation.

```sh
python bindings/st-py/examples/native_geometry_generation.py \
  --model-dir "$MODEL_DIR" \
  --artifact "$GEOMETRY_JSON" --artifact-id "$GEOMETRY_ID" \
  --qkv-checkpoint "$QKV_CHECKPOINT" --qkv-checkpoint-id "$QKV_ID" \
  --readout-accumulator f32_wide_backward \
  --prompt-json prompt.json --recipe-json generation-recipe.json \
  --stop-after 2 --output generation-part
```

Use the checkpoint's original arithmetic policy and frozen adapter source.
Omit the geometry arguments if there was no geometry. For a model with trained
LoRA/geometry, use the existing `--session-checkpoint`, `--session-checkpoint-id`
and `--session-boundary` source instead of `--artifact`; then overlay the QKV
checkpoint. Do not select two different frozen adapter sources at once.

To resume, keep those same model/QKV/source arguments, replace the prompt and
recipe arguments with `--resume generation-part/generation.json --resume-id
"$GENERATION_ID"`, and select a new output directory. `GENERATION_ID` is the
separately retained `checkpoint_id` in the partial run's `result.json`. Omit
`--stop-after` to continue to the recipe's original end. A resumed run cannot
replace its prompt or decoding recipe.

The QKV input needs both a bounded binary file and its expected ID. Its digest
is checked before model loading, then Rust verifies the frozen model/adapters,
readout and serving ownership. Missing flags, wrong owners and mismatched
generation state refuse before creating the output directory. Input files are
never rewritten. `result.json` records `qkv_snapshot` (checkpoint/serving IDs,
layer and optimizer counter), actual readout policy and frozen adapter IDs.
It explicitly does **not** claim to verify the training documents.

The [69-test handoff verification](../../benchmarks/paired-prompt-contrast/QKV-GENERATION.md#file-based-handoff)
includes the earlier 66 tests, not 69 additional independent tests. It uses
synthetic fixtures and the same preserved CPU extension, without a rebuild.

## Explicit Readout Arithmetic

The default is still f32, with unchanged checkpoint serialization. The
vocabulary head has three explicit accumulation policies:

| `readout_accumulator` | Forward logits | Backward head transpose product |
|---|---|---|
| `f32` (default) | f32 | f32 |
| `f32_wide_backward` | f32 | f64 |
| `f64` | f64 | f64 |

```python
# model has an attached trainable geometry; window_ids includes its next-token targets.
wide = model.with_readout_accumulator("f32_wide_backward")
assert model.readout_accumulator == "f32"  # Original handle is unchanged.
assert (wide.readout_forward_precision, wide.readout_backward_precision) == ("f32", "f64")
grad = wide.adaptation_loss_gradients(window_ids, [1, len(window_ids)], mode="geometry")
```

The opt-in wide direction(s) use four fixed f64 partial sums per dot product.
The split policy leaves full, last-only and cached logits and the losses on a
given snapshot identical to `f32`; it changes the head pullback during
geometry, LoRA and joint learning, including the repetition auxiliary.
Operands, weights, returned logits and gradients are still f32. Attention,
MLP and geometry keep their existing arithmetic.
This is neither a full-f64 model nor a Torch/GPU fallback. Rust owns the kernel
through `Tensor::matmul_prepacked_wide`; the same core compiles for WASM CPU.

The policy follows full/last-only forward, selected-target loss, geometry
learning and K/V contexts. Gradients and microbatch accumulators expose
`readout_accumulator` and reject batches or model handles with another policy.
Training and generation checkpoints bind it separately from the unchanged
weight/geometry identities. A geometry artifact alone does **not** record this
choice: preserve the runtime policy when using an artifact independently.

Generation and geometry-trainer checkpoints require a matching handle at
restore; their legacy missing policy field still means `f32`. Training
**session** checkpoints instead restore the recorded policy on a matching
unadapted base. Neither scheme treats `f32_wide_backward` as plain `f32`.
To deliberately start a different trajectory from old weights, restore them
under their recorded contract first, then select the new policy. Older
readers unaware of the new token reject it. No speed or model-quality gain
is implied by selecting a wider pullback.

The [pinned projection study](../../benchmarks/results/2026-10-19-native-lm-projections/README.md)
checks the old f32 trajectory, every compared prefix, native learning/generation
resume and the remaining CPU performance gap separately.

## Checkpoint Admission

The importer accepts bare GPT-2 names (`h.0...`, `wte.weight`) or the HF LM
prefix (`transformer.h.0...`). It rejects mixed naming, missing/extra tensors,
shape/dtype mismatch, non-finite values, and conflicting tied output weights.
Legacy persistent attention masks are accepted only after checking their exact
causal triangle. It does not silently discard LoRA tensors or unknown weights.

Supported config variants include `n_inner`, tied/untied head, unscaled
attention and inverse-layer scaling. Other activations, cross-attention,
pruned heads and reordered/upcast attention are rejected. Unrelated config
metadata such as generation defaults or training dropout probabilities does
not change this explicitly frozen execution policy. No remote code
or arbitrary HF architecture is loaded.

The normalized semantic config and exact checkpoint byte identity form
`model.base_model_identity`. This identifies a source/policy, not a trust
certificate or training provenance. It is intentionally not interchangeable
with earlier caller-defined HF runtime IDs.

Before Python ID extraction, forward admission checks nonempty B/T shape,
position capacity, QKV/MLP activation sizes, total attention pairs across all
layers and the requested logits. Constructor limits (keyword overrides):

| Budget | Default |
| --- | --- |
| `max_parameter_elements` | 200,000,000 imported f32 parameter values |
| `max_activation_elements` | 16,777,216 per checked activation |
| `max_attention_pairs` | 4,194,304 across layers, batches and heads |
| `max_logit_elements` | 16,777,216 for full or last-only output |
| `max_training_checkpoint_elements` | 16,777,216 saved block-boundary values during geometry learning |

The separate [weight-loader limits](native-weights.md) still apply. These are
explicit work/array bounds, not an RSS promise; live temporaries and checkpoint
bytes coexist. Context capacity alone does not promise every B/T fits a budget.

## Geometry Reaches Model Outputs

```python
# Export/create the artifact using this exact native model identity.
adapted = model.with_geometry(artifact)
changed_logits = adapted.forward(ids, [1, len(ids)])
assert model.geometry_artifact_id is None  # Base handle remains unchanged.
```

Supported placements are `transformer.h.N.attn` (projected attention branch
before residual addition), `transformer.h.N.mlp` (MLP branch before residual
addition), and `transformer.h.N` (the completed block). Every placement must
match an existing layer and feature width. A foreign model identity or unused
placement is rejected, never silently ignored. Existing native affine, Wave,
elliptic, Topos and fractional recipes retain their own sequence budgets.

Model handles and their attached geometry are immutable snapshots. Attach a
new artifact or apply a sealed gradient batch to obtain a new handle; existing
handles remain unchanged. Nested placements (a block and one of its own
children) remain rejected by the shared artifact contract.

## Language-Loss Geometry Learning Without Torch

`geometry_loss_gradients` now propagates mean next-token NLL through the
**required frozen decoder path** into every geometry placement. This is not a local
activation-MSE surrogate: vocabulary projection, final LayerNorm, residual
branches, MLP/GELU, causal Attention and preceding blocks all contribute to
the gradients. The existing Rust kernels own both primal and pullback math.
Embeddings and base projection/norm parameters stay frozen; their parameter
gradients are not computed. All 14 currently supported native recipe
configurations use the same path, including learned radius, elliptic, Topos
and fractional/history controls.

```python
# artifact must use model.base_model_identity and contain parameters with
# requires_grad=True. Module training/eval metadata is not an update permit.
base = model
model = base.with_geometry(artifact)
windows = [464, 3616, 286, 1204, 318, 407]  # T+1 IDs, no padding
gradients = model.geometry_loss_gradients(windows, [1, len(windows)])
print(gradients.loss)  # pre-update mean NLL across the T shifted targets
updated = model.step_geometry_sgd(gradients, 0.02)
assert model.geometry_artifact_id == gradients.geometry_artifact_id

saved = updated.geometry_artifact()
loaded = st.GeometryAdapterArtifact.from_json(
    saved.to_json(), saved.artifact_id, base.base_model_identity
)
restored = base.with_geometry(loaded)
```

`gradients.parameter_gradients()` returns detached copies keyed by placement
and parameter name, including frozen parameters for inspection but excluding
buffers. The sealed batch is bound to the exact base and geometry identities;
it cannot be constructed from caller-supplied gradients. Applying it against
a different/updated artifact fails. SGD updates all trainable placements from
one loss snapshot, validates the whole candidate, and returns a new model.
Frozen parameters and fixed buffers never change; invalid rates or updates
leave the source untouched. There is no hidden optimizer state. An artifact
alone is not a data-cursor/RNG checkpoint or a provenance certificate.

For adaptive updates, the separate [native geometry AdamW API](native-geometry-adamw.md)
uses shared Rust clipping, moments and accepted-update clocks. It returns an
atomic `(model, optimizer)` pair and supports exact pair checkpoint continuation;
it does not delegate to `torch.optim` or alter the stateless SGD contract above.

For larger effective batches, use the
[native microbatch accumulator](native-geometry-adamw.md#token-weighted-microbatches).
It accepts different unpadded window lengths, weights each loss/gradient by its
number of shifted targets, and applies one update after the group. Token
extraction, forward/backward admission and snapshot checks remain native.

For an ordered document corpus, the [native corpus trainer](native-corpus-training.md)
reuses this accumulator and Rust's document boundaries. It owns the model/optimizer
pair, fixed recipe and data cursor, so whole runs can continue without HF Trainer
or a Python reimplementation of the loss/optimizer loop.

The learned model also feeds [native generation](native-generation.md): Rust owns
full-prefix or explicitly cached execution, greedy/sampled decoding, EOS,
Z-Space control and pinned random continuation.
Python no longer needs a Torch generation loop for this supported model family.

For read-only validation, [native corpus evaluation](native-corpus-training.md#native-corpus-evaluation)
scores complete document target coverage, including tails, and computes
target-weighted NLL/perplexity without importing the HF training stack.

### Competing for the first answer token

A change in the greedy ID can exchange two incorrect tokens without raising
the reference token's rank. To train the attached geometry on that decision,
use the opt-in, CPU-native `geometry_competition_gradients`. It retains a
frozen LoRA in the forward if present. It does not change the sampler, the
ordinary CE path, or a training session's checkpoint/resume policy.

```python
# On a training example, not the development example used to notice a flip:
window = [*prompt_ids, first_answer_id]
grad = model.geometry_competition_gradients(
    window, [1, len(window)], labels=[-100] * len(prompt_ids) + [first_answer_id],
    rivals=4, mix=0.5,
)
report = grad.competition_report()  # pre-update full CE, restricted CE, target rank and margin
next_model = model.step_adaptation_sgd(grad, "geometry", learning_rate=0.01)
assert next_model.lora_artifact_id == model.lora_artifact_id
```

For each non-ignored shifted label, Rust selects the highest-scoring `rivals`
incorrect IDs from that row (lower ID first at a tie, no fixed token list).
The objective is `(1-mix)*full_CE + mix*restricted_CE`, averaged over active
labels, where restricted CE normalizes only the target and those rivals.
The selected set is piecewise constant; at a tie its VJP uses the indicated
branch. `rivals` must be 1 through 32 and less than the vocabulary size;
`mix` must be finite in [0, 1]. Zero mix preserves the ordinary loss and logit
gradient exactly. All inputs are still admitted and decoded as causal context;
the mask above scores one answer token, not a full generated answer. Compare
full CE and `target_logit - best_non_target_logit` on examples not updated in
the comparison; a lower restricted training loss alone is not a quality gain.

To train against a **named** wrong class rather than the current top mistakes,
use a training label and a declared contrast. For example, the heads of
two field answers such as `large square` and `square` can compete in both
directions on their respective training examples:

```python
ids = prompt + [gold_head]
g = model.geometry_declared_rival_gradients(
    ids, [1, len(ids)], labels=[-100] * len(prompt) + [gold_head],
    contrasts=[(len(prompt), gold_head, [other_head])], mix=0.5,
)
next_model = model.step_adaptation_sgd(g, "geometry", learning_rate=0.01)
report = g.declared_rival_report()
```

Positions are **flat unshifted label indices** in `[batch, time]`. List them
in increasing order; each expected target must match an active shifted label,
not the first position of a window. Each list contains 1-32 sorted distinct
rival IDs excluding gold (65,536 IDs total). Omit an entry to keep ordinary
CE at that position. The blended loss averages over **all** active targets;
`declared_ce` averages over declared ones only (None if none). Zero mix or an
empty plan preserves ordinary CE. This opt-in path neither selects an
inference vocabulary nor alters LoRA/base weights or existing training sessions.

To check whether two training examples give conflicting directions, keep the
model fixed, compute one adaptation gradient per example and call
`left.alignment_with(right)`. Rust rejects different base/LoRA/geometry
owners, readout precision, or differentiated tensors. The report contains
global and per-tensor L2 norms, dot, cosine, and the first derivative of each
loss along **minus the arithmetic mean** of the gradients. A negative
`left_loss_slope` means local descent on the left objective; a negative
cosine alone need not imply either objective increases. `retained_fraction`
is the norm of the sum divided by the sum of norms. Cosine is absent for a
zero vector. Objectives receive equal weight, not weight proportional to the
number of scored tokens. No model or optimizer step is performed.

For a discrepancy in the mixed gradient on one prompt,
`model.last_attention_mixture_probe(prompt, target_id, rival_id, mix=0.5)`
keeps a **single f32 forward** fixed. It returns the vocabulary logits, three
unmodified logit-gradient seeds (`ce`, `declared`, `mixed`), and two
parameter pullbacks per seed. Only the transpose product through the
vocabulary head changes accumulation precision between `paths["f32"]` and
`paths["f64"]`; this is **not** a forward on a wide-readout model. Both paths
store f32. Every vector in `d_logits`, `boundaries` and `parameters` is
little-endian f32 bytes. Boundaries are the active token row in backward order
from head input through final normalization and the MLP to the arriving
attention-output adjoint. Parameter vectors are the full returned gradients.
The diagnostic requires the plain `f32` policy with one geometry at the final attention
output, no update, and no more than 1,048,576 result elements. A named rival
on a training input is not a decoder or an evaluation candidate policy.

### Separating update direction and distance

Equal learning rates do not guarantee equal parameter motion. To compare a
direction at a fixed Euclidean gate/bias (or general trainable parameter)
distance, use the source artifact as the origin:

```python
origin = model.geometry_artifact()
distance = origin.trainable_delta_l2(reference_geometry)
matched, receipt = origin.match_trainable_delta(proposed_geometry, distance)
same_lora_new_geometry = model.with_geometry(matched)
```

The global norm covers all tensors marked trainable parameters, not one norm
per tensor or placement. Base identity, policy, placement order, recipe, training
flags and tensor layout must agree. Frozen parameters and buffers must be
bit-identical. Rust checks both the realized radius and the relative L2 error
of the entire displacement against the ideal vector (maximum `1e-4`), including
after f32 storage rounding. An unrepresentable request fails without changing
either endpoint. Zero radius returns the exact origin; requesting the original
direction length returns the exact endpoint. Positive radius needs a nonzero
direction. Extrapolating may also fail the artifact's recipe constraints.

Applying this **once** to final weights is an endpoint counterfactual, not
matched training. For a per-update control, recompute the gradient on the
current matched model, take an ordinary geometry-only SGD proposal, then use
the **current** artifact as the origin and the length of the corresponding
reference step as the radius. Equal per-step lengths match path length, not
necessarily final distance from the training start. Neither procedure matches
operator output RMS, intrinsic hyperbolic distance, or language quality.

For a **live CE reference**, the paired Rust training step returns both new model
handles. Start both arms at the same checkpoint. On each call both see the
same windows, labels and rate, but differentiate their own current weights.
Only decoder blocks before the first adapter are shared. The mixed arm takes
the direction of its ordinary rounded SGD proposal and matches the distance
actually moved by the CE arm on that batch.

```python
ce_model = mixed_model = model  # trainable geometry attached; any LoRA stays frozen
for windows, shape, labels in batches:
    ce_model, mixed_model, step = ce_model.geometry_competition_pair_step(
        mixed_model, windows, shape, labels=labels,
        learning_rate=0.01, rivals=4, mix=0.5,
    )
    movement = step["mixed_step"]
    print(step["reference_loss"], step["competition"]["loss"],
          movement["requested_l2"], movement["realized_l2"])
```

The report contains pre-update losses, the actual CE geometry IDs and the
mixed arm's original, proposed and applied geometry IDs. Both input handles
are immutable. A mismatch in base, readout policy, LoRA, geometry layout or
frozen values rejects the pair; different **trainable** values are expected
after the first step. If either update fails, neither is returned.

If the CE step lengths are already recorded, use
`mixed_model, receipt = mixed_model.step_geometry_sgd_at_distance(grad, rate, ce_l2)`
with fresh sealed adaptation gradients from that mixed model. Even a requested
length of zero validates the original SGD proposal and its owners; a zero
direction cannot supply a positive length. These are explicit, stateless SGD
calls. They do not change session checkpoint schemas, store the data cursor
or implicitly switch a resumed training session to the competition objective.

**Competing complete answers.** To train the geometry on an actual closed-menu
decision, supply alternatives from training, not winners or labels obtained by
inspecting an evaluation set. They may start with the **same token**. Pass content
only, in distinct lexicographic token-ID order; the separate stop ID is scored
exactly once for each answer. A candidate may be a proper prefix of another.

```python
menu = [(17,), (17, 19), (23, 7)]
options = dict(stop_id=0, gold_index=menu.index((17, 19)), mix=0.5)
before = model.line_menu_loss(prompt_ids, menu, **options)  # also works on a frozen base
grad = model.geometry_line_menu_gradients(prompt_ids, menu, **options)
trained = model.step_adaptation_sgd(grad, "geometry", 0.01)
after = trained.line_menu_loss(prompt_ids, menu, **options)
```

For each line, sum full-vocabulary token NLLs through its stop. The menu
distribution is the softmax of **negative summed costs**, not averaged costs
or head logits. The loss mixes mean gold token CE (weight `1-mix`) with
negative log gold menu probability (weight `mix`). A shared content prefix
cancels from menu normalization and its VJP, but still contributes to gold CE.
The report includes full and suffix costs, menu probabilities, total rival
probability (`rival_mass`), and `winner_index` (first index on ties).
Ranking now projects one vocabulary row per distinct prefix (`prefix_nodes`)
and only decodes maximal answer windows (`decoder_lanes`). At a fork the
ranking signal is the posterior mass of the alternatives that diverge there;
mass that diverged earlier is carried explicitly, not recovered as one minus
a nearly-one probability. Shared content still receives the gold CE component.
`grad.line_menu_report()` describes the pre-update snapshot. Earlier
declared-rival and default training APIs are unchanged.

A lower `menu_nll` need not mean the model is more likely to produce the
gold line in free decoding. Compare `line_nlls[gold_index]` as well: the
gold line can lose absolute probability while the other lines lose even more.
For distinct, stop-terminated lines their total unconditional probability is
`sum(exp(-nll) for nll in line_nlls)`; use a log-sum-exp for long answers.

One causal batch holds 2 to 32 candidates. Only content and its first stop
are scored; later right padding has ignored labels. `grad.target_tokens`
counts selected targets **across the menu**, including at `mix=0`: use ordinary
masked CE when you do not want to evaluate the alternatives. Total selected
targets are capped at 4096, padded IDs at 65,536; model-specific limits may
be smaller. Base and LoRA stay frozen. This is supervised relative likelihood
on a supplied menu, not reference-model preference optimization or free generation.

**Scoring without labels.** For evaluation or inference, there is no need to
invent a gold index or an accepted subset:

```python
r = model.score_lines(prompt_ids, [(17,), (17, 19)], stop_id=0)
absolute = r["line_nlls"]          # full-vocabulary, including one stop
relative = r["conditional_nlls"]  # given that one of these lines is emitted
per_token = r["token_nlls"]        # last entry of each line scores its stop
```

The bounded, forward-only Rust readout works on the frozen base or an adapted
model; no gradients or target labels are used. Supply 1 to 32 distinct,
lexicographically sorted, nonempty content sequences excluding the stop.
An interior stop is scored even when a line prefixes another line.
`menu_set_nll` is the absolute loss for the union. Conditional losses are
computed in log space, not by taking logs of possibly underflowed probabilities.
Use `winner_index`: two rounded absolute totals can tie even though their
suffixes distinguish the winner. For a true tie the first candidate wins.
The [matched-context transfer experiment](../../benchmarks/paired-prompt-contrast/alias_fresh_transfer.py)
uses this readout on frozen models before looking up any answer labels. It
compares the change in a line's log probability on the correct context
against its change on otherwise matched incorrect contexts; a uniform
increase cancels in this contrast.

**Localize a signed logit signal before changing trainable weights.** For one
unpadded prompt, `model.projection_contrast_pullback(ids, target, rival,
layer=0, site="query", basis="dense")` differentiates the last target-minus-rival
logit. Select `query`, `key`, `value` or `output`. All use virtual
`z + bias + source @ weight` at the selected projection, rounded once to f32.
Q/K/V use the normalized block input; output uses the merged attention values
and includes the frozen LoRA branch before existing geometry. The default basis
is diagonal; dense weights are row-major [input feature, output feature].
The receipt includes both owners (if present), the base identity, readout
policy, factual token contrast, and compensated `bias_f64le` / `weight_f64le`
buffers. A float derivative is not an observed finite change.

Use `model.projection_perturbed_contrast(ids, target, rival, layer=0,
site="query", basis="dense", bias_f32le=b, weight_f32le=w)` to run a real
forward with the supplied signed deltas (already scaled, little-endian f32).
This is a full-prefix, read-only intervention, not a new checkpoint or a KV
cache edit. Zero buffers preserve the ordinary contrast. A shared key bias is
a softmax null control, up to rounding. The result/transport limit is
1,048,576 coordinates including bias, with up to 67,108,864 per-prompt
weight-coordinate reduction terms and the model's usual workspace limits.

**Prefer one context for the same next token.** Supply two distinct unpadded
prefixes (they may differ in length) and one target vocabulary ID. To isolate
word choice, include any common answer prefix in *both* contexts:

```python
a = [*positive_prompt, status_id]
b = [*negative_prompt, status_id]
g = model.geometry_context_token_preference_gradients(a, b, standby_id, margin=0.0)
before = g.context_token_preference_report()
updated = model.step_adaptation_sgd(g, "geometry", rate)
after = updated.context_token_preference(a, b, standby_id)
```

Rust computes `gap = log P(target | a) - log P(target | b)` using the full
vocabulary of each row, and minimizes `softplus(margin - gap)`. The gradient
is `sigmoid(margin - gap) * (grad CE(a) - grad CE(b))`: one loss per pair,
not a two-token mean. `margin=None` means zero; a supplied margin is a
finite nonnegative log-probability gap, not a probability threshold.
Two independent causal lanes share one forward/backward; only the last
real predictor in each lane enters the vocabulary head. No implicit prefix
or stop is added. Base/LoRA remain frozen and ordinary geometry handles,
snapshot checks and step-size guards apply. An improved gap can still mean
that both probabilities rose or both fell: inspect the two absolute NLLs
and the full stopped answer separately.

The [matched-context training comparison](../../benchmarks/paired-prompt-contrast/context_token_training.py)
uses the same training words on every arm and rotates only the negative-context
pairings in its control. Its endpoint checks distinguish lexical selectivity,
absolute wrong-context gains, and selectivity of the complete terminated line.

**Learning any acceptable answer.** If several answers are genuinely acceptable
for a training prompt, there need not be a gold index. Use an absolute loss on
the event that the model emits **any** member, including its stop token:

```python
acceptable = [(17,), (17, 19)]
report = model.line_set_loss(prompt_ids, acceptable, stop_id=0)
g = model.geometry_line_set_gradients(prompt_ids, acceptable, stop_id=0)
trained = model.step_adaptation_sgd(g, "geometry", 0.01)
```

`report["set_nll"]` is the negative log of the sum of full-line
probabilities, computed in Rust over distinct prefixes. It is not a mean
token loss, nor a softmax restricted to the supplied choices. For a singleton
it is summed token CE including the stop. Unlike a pure menu-ranking loss, it
trains shared content. A trainable geometry adapter is needed only for the
gradient call; `g.line_set_report()` records the objective before the step.
The same budgets and content/stop rules apply as above, with 1 to 32 lines.

Only the longest lines along each branch need decoder windows: an answer that
is a strict prefix still has its own scored stop, at the corresponding interior
position. The vocabulary head and its backward pass use **one row per distinct
prefix**, including the empty answer prefix. For `[(2,), (2,4), (2,4,7), (2,5)]`
the report records `decoder_lanes=2`, `prefix_nodes=5`; the logical target
count is still 12. `grad.window_shape` describes the supplied candidates, not
physical lanes. Length-only checks bound the input before copying it; exact
model limits apply to the resulting lanes and vocabulary rows before decoding.
There is no cross-branch KV cache: branches that diverge still run their common
prompt in each maximal window. Both objectives use this physical layout,
but their losses and pullbacks are different.

For a reproducible CPU comparison, use
[`bench_native_line_set.py`](../../tools/bench_native_line_set.py) with the
reference and candidate Python executables and their installed extension
SHA-256 pins. It uses artificial weights, alternates the two executables,
checks losses and gradients, and times the loss-only call separately from the
full forward-plus-backward gradient call for prefix chains, forks, and a
singleton. Use `--objective ranking` for gold CE alone, half CE plus ranking,
and ranking alone on chains, shared-prefix forks and a minimal two-choice menu.
No pretrained data or updates.

For the **same** model and set, full NLL of a designated line equals
`set_nll + menu_nll` for that gold index. More total mass does not promise that
each answer, the gold rank, or free-generation quality improves. Keep incorrect
distractors out of an acceptable-set training label.

**Accepted aliases versus near misses.** When several answers count as
correct but you also have known distractors, pass one menu and a nonempty
proper subset of its indices. Both lists must be sorted and distinct.

```python
menu = [(17, 2), (17, 3), (17, 14)]  # e.g. "status on", "status off", "status enabled"
options = dict(accepted_indices=[0, 2], stop_id=0)
g = model.geometry_line_subset_gradients(prompt_ids, menu, **options)
before = g.line_subset_report()
trained = model.step_adaptation_sgd(g, "geometry", 0.01)
after = trained.line_subset_loss(prompt_ids, menu, **options)
```

`conditional_nll` is `-log(P(accepted) / P(menu))`, with each stopped line
scored under the full vocabulary. The sealed gradient contrasts only pairs
on opposite sides of the supplied partition; there is no designated gold
alias or uniformity penalty among valid aliases. A singleton accepted index
recovers pure menu ranking. If every line is acceptable use `line_set_loss`
instead. Inspect `accepted_set_nll` for absolute mass and `line_nlls` for
individual aliases: conditional gain alone promises neither more absolute
correct probability nor improvement of the least likely acceptable answer.

The offline [alias study](../../benchmarks/paired-prompt-contrast/alias_subset_study.py)
compares this with alternating single golds and absolute accepted-set training
on a small symbolic fixture. Run `prepare NEW`, then `train NEW FIXTURE_PIN TRAIN`,
`score NEW FIXTURE_PIN TRAIN TRAIN_PIN SCORED`, and
`audit NEW FIXTURE_PIN TRAIN TRAIN_PIN SCORED SCORE_PIN AUDIT` with the
installed Python interpreter. Each command prints the pin of its manifest and
refuses to overwrite its output. Only the last, model-free command reads the
score labels; it reports absolute and conditional mass and the weaker alias.

**One step from several objectives.** Compute geometry-only gradient handles on
the same immutable model and call
`next_model, receipt = model.step_geometry_cohort_sgd(handles, 0.035)`.
This takes **one equal vote per handle**, regardless of its logical target count
or window shape. A repeated handle counts again. This is not token-weighted
microbatch accumulation and it does not run the decoder. Frozen LoRA, buffers
and frozen geometry tensors are not updated. Mixed owners, mixed readout
policies and non-geometry handles are rejected before a proposal is returned.

`target_l2=0.001` optionally matches the proposal's actual trainable displacement
to that length, without changing the origin. `receipt["member_loss_dots"]`
contains each input gradient dotted with the rounded displacement, in input
order: these are first-order predictions, **not** measured losses. Use a fresh
forward to check them. The cohort is capped at 256 handles and 16,777,216
member-by-geometry-parameter elements, including frozen parameter entries.

The [alias interference experiment](../../benchmarks/paired-prompt-contrast/alias_interference.py)
reopens only the initial and subset-trained artifact from the alias study. Its
`plan`, `measure` and model-free `audit` phases compare three fixed class
means on the same 18 training prompts with realized finite steps at two radii.
It never continues the training trajectory or scores the evaluation split.

**Protecting several group objectives.** Use
`next_model, receipt = model.step_common_geometry_sgd([on, off, idle], 0.035, target_l2=0.001)`,
where each list contains sealed geometry gradients from that model. Within
each group, examples have equal votes. Among groups, Rust finds the closest
point to zero in the convex hull of the unit mean gradients and checks its
worst-group cosine against its norm. Its negative direction maximizes the
worst normalized first-order decrease within the stated numerical tolerance.
`unit_group_weights` apply to normalized means; `raw_group_weights` undo
that normalization and sum to one, setting the ordinary proposal scale.

The method accepts 2 to 8 nonempty groups (the total cohort bounds still
apply). It refuses zero means, a zero or numerically unresolved margin,
and rounded displacements that do not strictly descend for every **group
mean**. It does not claim every example improves, or that the finite-step
loss decreased: re-evaluate the losses. `group_loss_dots` are predictions
on the actual displacement, in input group order. No fallback policy or
optimizer state is implicit.

The [finite common-step comparison](../../benchmarks/paired-prompt-contrast/alias_common_step.py)
has a model-free `plan` that freezes an independent three-class prediction
from the interference study. Its `measure` phase reopens the two synthetic
training snapshots and makes equal and common steps at the same radii.
`audit` rebuilds the measured changes from the saved artifacts and losses.
This is a local step test, not a held-out evaluation or continued training.

To **train for several steps**, rebuild all groups from the returned model on
every iteration. Handles from the previous geometry are stale. Compare losses
at the returned weights, not just the first-order predictions in the receipt.
The [matched trajectory study](../../benchmarks/paired-prompt-contrast/alias_common_trajectory.py)
does this for 32 steps from two starting geometries, with equal-vote SGD and
a straight continuation of the first common direction as separate controls.
All three paths use the same training prompts and measured step lengths. Its
[offline audit](../../benchmarks/paired-prompt-contrast/alias_common_trajectory_audit.py)
separates group loss, absolute accepted-answer mass and the probability of
the initially less likely alias. No pretrained or held-out inputs are used.

**Protecting a named answer.** A conditional class objective and an absolute
line objective can be separate groups in the *same* native update. For
example, if both `status idle` and `status standby` are acceptable, but it
also matters that the model can emit the latter, add one singleton-set
gradient per idle prompt:

```python
# on, off and idle contain the current model's sealed subset-loss gradients.
standby = [
    model.geometry_line_set_gradients(prompt, [standby_ids], stop_id=stop_id)
    for prompt in idle_prompts
]
next_model, receipt = model.step_common_geometry_sgd(
    [on, off, idle, standby], rate, target_l2=radius,
)
```

`standby_ids` contains the full answer *without* the stop; the objective
includes the stop once. This protects a first-order change in the **mean
full-line** loss, not every prompt, not the word in isolation, and not its
share among accepted answers. The [matched protection study](../../benchmarks/paired-prompt-contrast/alias_protection.py)
uses a three-group baseline and an otherwise identical fourth group protecting
the other correct answer. It measures all three quantities at the returned
weights; an infeasible step is a refusal, not a silent fallback.

**Guarding individual prompts without a larger hull solve.** A group mean
can descend while one member gets worse. Pass the proposal and gradients
from its *starting* model to a separate Rust guard:

```python
protected, check = model.guard_geometry_step(
    next_model, [on, off, idle, *([g] for g in standby)], min_cosine=1e-5,
)
```

This protects three means and each standby prompt separately. If every
normalized loss slope on the proposed displacement clears the floor, the
candidate geometry is returned byte-for-byte. Otherwise Rust turns toward
the negative unit gradient of the worst group (first in input order on a
tie). It intersects the cosine constraints on the unnormalized segment,
uses the midpoint of its feasible interval, and normalizes to the original
step length. `applied_loss_dots` and `applied_min_cosine` are checked after
f32 rounding. They are predictions: always measure the actual finite losses.

The guard accepts 1 to 256 groups within the existing total member and
member-by-parameter limits. It does not run a new decoder pass, claim a
maximin direction, or search outside this one segment. An unrepresentable
move, stale handle, incompatible candidate or unavailable repair is an
error. The 2-to-8-group optimizer and its contract are unchanged.
The [finite individual-step study](../../benchmarks/paired-prompt-contrast/alias_individual_step.py)
tests this distinction by replaying 64 archived proposals and scoring one
independent guarded endpoint from each original starting point.

**Making a preference the primary objective.** Adding a preference to the
common-descent groups can give it *zero* weight when the existing direction
already satisfies it. To steer instead, start from a strictly safe anchor:

```python
anchor, _ = model.guard_geometry_step(proposal, protected_groups, min_cosine=1e-5)
next_model, proof = model.steer_geometry_step(
    anchor, pair_gradients, protected_groups, min_cosine=1e-5,
)
```

All gradient handles are geometry-only and belong to `model`, not to the
anchor. Each group weights its members equally; the primary and all protected
groups share the 256-handle limit. Rust tries the negative primary mean at the
anchor's displacement length. If it would miss a protected cosine floor,
Rust goes halfway to the first such boundary on the segment from the anchor
direction, then normalizes. It checks the **actual rounded displacement**:
the protected cosines must clear the floor, the primary loss dot must be
negative and lower than at the anchor, and its cosine gain must exceed
`1e-12`. Otherwise the step refuses, including when no improvement over an
already aligned anchor is resolved. No line search or fallback is implicit.
In the receipt's per-group arrays the primary is first;
`limiting_protected_group` indexes the protected groups only. These are
first-order statements, not measured finite losses.

For **continuous training**, use the selected model as the next model and recompute
all gradients from it. Do not feed the unguarded candidate into the next
iteration or reuse gradient handles across updates. The
[guarded training example](../../benchmarks/paired-prompt-contrast/alias_guarded_training.py)
compares these policies at matched step lengths. It persists the geometry
and boundary at step 64, then a fresh CPU process verifies its first update
against a prospective, uncommitted step from the live process before
continuing to step 128. This geometry-only SGD recipe needs the exact
frozen base, LoRA, readout policy and training inputs on restore; a geometry
artifact alone is not a general trainer checkpoint.

**Small projection updates.** For a fixed-radius `wave_stage_adapter.v1` projection,
open an explicit local coordinate chart *before* computing new gradients:

```python
centered = model.with_geometry(model.geometry_artifact().anchor_wave_projection("transformer.h.11.attn"))
```

The new artifact keeps `gate_anchor` and `bias_anchor` as immutable f32 buffers.
Its `gate` and `bias` parameters are f32 **offsets**, initially zero, with their
original training flags. The baseline affine calculation retains its original
f32 rounding; the input times the gate offset and the bias offset are added in
f64 before the radial map. The projected output and residual branch still use
f32. An offset can therefore affect inference without first being rounded back
into the full gate. Not every small movement is visible at the final logits.

The source artifact, existing checkpoints, and other placements do not change.
The new layout has its own identity; start a new trajectory, not a resume with
old gradients or optimizer state. Within a chart, distance and common-descent
checks concern the actual stored offsets, with the *same* vector and length
tolerances. Offsets can still run out of precision as they grow. This is a new
representation, not an automatic fallback for a refused step, error feedback,
or a guarantee of better loss. The four-way wave factorial observer does not
admit this chart. WASM artifact clients built with `nn` use `anchorWaveProjection`.

**Pairing two examples on one snapshot.** This is distinct from matching two
models. Compute two `mode="geometry"` gradients on the *same* model, then call:

```python
left = model.adaptation_loss_gradients(left_ids, [1, len(left_ids)], mode="geometry")
right = model.adaptation_loss_gradients(right_ids, [1, len(right_ids)], mode="geometry")
next_model, step = model.step_paired_geometry_sgd(left, right, 0.01, policy="common_descent")
```

The default `equal` policy averages the two **example means**, not individual
tokens across both windows. `common_descent` keeps that exact step if both
gradients have a negative dot product with the actual f32 displacement (relative
tolerance `1e-12`). Otherwise it weights each gradient inversely to its norm,
using only trainable geometry coordinates. It refuses a step if the resulting
displacement still fails either test. Base, LoRA, frozen tensors and the source
model are unchanged. Pass `target_l2` to match the step length before testing
the dots. The receipt's `equal` and `applied` entries report `loss_dots` in
left/right order. These are **first-order predictions**: measure post-update
loss separately. No extra forward pass or implicit session update is performed.

**Optional candidate-set coverage.** Pass `coverage_guard=0.5` to
`geometry_competition_pair_step` to discourage losing probability mass from
the CE reference's target and top rivals. On each supervised row, let `S` be
those IDs from the **pre-update** CE readout. The extra per-row loss is
`strength * max(0, log P_ce(S) - log P_mixed(S))`, averaged over **all**
supervised rows. The reference is detached; the original mixed objective
still chooses its own strongest rivals. No additional decoder forward is
needed. Equality has zero subgradient, and omission or zero strength
retains the original algorithm and report. Finite strengths in [0, 1] are
accepted. For enabled guards, `step["competition"]["coverage_guard"]`
reports activation count, the two mean coverage NLLs, and the mean positive
violation *before* weighting. These are pre-update measurements: the guard
is not a line search or a guarantee of post-update probabilities.

### Selected Targets and Label Smoothing

Source builds also accept HF-style **unshifted** `labels` with the same flat
`[B,T]` layout as the input windows. This enables completion-only supervision:
prompt tokens remain context, but their loss terms can be excluded. Python
only passes integer labels and configuration to Rust; `st-tensor` owns mean
cross entropy, uniform smoothing and the matching VJP.

```python
ids = [464, 3616, 286, 1204, 318, 407]
labels = [-100, -100, -100, 1204, 318, 407]
gradients = model.geometry_loss_gradients(
    ids, [1, len(ids)], labels=labels, label_smoothing=0.1
)
assert gradients.target_tokens == 3
score = model.next_token_loss(
    ids, [1, len(ids)], labels=labels, label_smoothing=0.1
)

group = model.geometry_accumulator(label_smoothing=0.1)
group.accumulate_loss(model, ids, [1, len(ids)], labels=labels)
model, optimizer = group.step_adamw(model, optimizer, 0.001)
```

Each row's first label is never predicted; every other `-100` label excludes
one loss term. All labels, including the first, must be valid vocabulary IDs or
`-100`. Non-ignored labels need not equal input IDs. With `labels=None`, input
IDs supply all shifted targets as before. Smoothing is finite in `[0,1]`, with
default zero, and is **not** a generation temperature or a change to geometry.
Mean reduction divides by the non-ignored shifted target count. A row can be
fully ignored, but an entire microbatch with no targets rejects before decoder
execution; it does not silently trigger decay or advance optimizer clocks.

Label exclusion is not an attention/padding mask: the complete unpadded prefix
still runs, masked prompt positions can receive gradients through later
predictions, and no EOS or document boundary is invented. Input and label
context budgets are checked before foreign array extraction. Invalid inputs do not
change an existing accumulation. Its `label_smoothing` policy is sealed at
creation, and adding a sealed gradient batch with another policy rejects.
Different label selections under the same policy are expected and combine by
their actual target counts, not by context lengths.

For Rust callers, `CausalLmObjective` and the `*_with_objective` methods expose
the same contract; default method signatures are unchanged. Preserve the
objective, labels/data cursor and schedule alongside the artifact/optimizer
pair for continuation. They are not inferred from optimizer moments or saved
inside a pending accumulation. These entries do not silently replace
`HfGeometryTrainer` or enable arbitrary HF architecture execution.

Selected-label execution compacts only the **terminal readout**: final LayerNorm,
vocabulary projection, CE and their pullbacks run on selected rows. The shared
row-gather/scatter kernels restore terminal gradients to the full context before
decoder reverse execution. This is not attention pruning or zeroing all prompt
gradients. When every shifted target is selected, no gather/scatter is added.

With explicit labels, `max_logit_elements` bounds
`non_ignored_shifted_targets * vocabulary`, not every contextual position.
Admission first bounds input/label transport, QKV/MLP, attention, geometry and
learning checkpoints; after bounded label extraction it validates the exact
logit count **before decoding**. At least one vocabulary row must fit. Without
labels the full output count is already known and still checked before token
extraction. Ordinary `forward`/last-only budgets are unchanged. Unrequested
vocabulary outputs need not be materialized or finite; invalid source weights,
IDs, labels and required intermediates are not silently accepted.

This reduces head/adjoint work and logit storage for sparse supervision, not
decoder context work, packed-weight storage or full-backward checkpoint sizes.
Numerical comparisons use tolerances; different selected row counts may choose
different native matrix kernels. No cross-version bitwise-continuation promise
is implied by exact resume within one tested runtime.

The [complete local readout comparison](../../benchmarks/results/2026-10-18-native-lm-selected-readout/README.md)
retains the initial 4-row regression and the CPU autotune fix, alongside both
full-head and selected-head Torch references. It measures frozen loss evaluation,
not geometry learning speed or quality; Torch is still substantially faster.

Learning saves block-boundary activations from the earliest participating
geometry layer and replays one block at a time, preserving the full `[B,T]`
partition. Reverse execution stops immediately after collecting that earliest
site, including frozen sites whose parameter gradients are exposed. Unused
frozen-prefix derivatives are neither computed nor allowed to cause failures.
That earliest site uses a parameter-only geometry pullback, so an unrequested
input adjoint cannot overflow and suppress otherwise valid parameter gradients.
Earlier-site propagation still requests the complete input derivative.
Forward and replay use the same code,
not separate Python or Rust decoder implementations. The checkpoint budget
is checked before Python ID extraction; ordinary activation/attention/geometry
budgets apply to each replay. Backward and recomputation do additional work.
Frozen projection transposes are packed lazily once, with construction
synchronized across concurrent first calls, and shared across model
variants; learning can retain another weight-sized orientation per projection.
Input-only LayerNorm VJPs skip unused affine gradients. These are explicit
CPU memory/work tradeoffs, not a resident-GPU or wall-clock speedup claim.

See [selected-site geometry learning](native-geometry-runtime.md) for the
standalone VJP/MSE entry, which remains available without loading a decoder.

## Reproduce Checks

The [integration follow-up](native-runtime-integration-validation.md) records
parameter-only pullbacks, independent review and the repeated real-model
comparison after integration with the optional HF observation/Trainer client.

```sh
cargo test -p st-nn --lib causal_lm:: --locked
python -I -B bindings/st-py/tests/test_native_causal_lm.py
python -I -B bindings/st-py/tests/test_native_causal_lm_accumulation.py
python -I -B bindings/st-py/tests/test_native_causal_lm_objective.py
# Optional comparison environment only:
python -I -m pytest --import-mode=importlib -q bindings/st-py/tests/test_native_causal_lm_parity.py
python -I -m pytest --import-mode=importlib -q bindings/st-py/tests/test_native_causal_lm_learning_parity.py
python -I -m pytest --import-mode=importlib -q bindings/st-py/tests/test_native_causal_lm_accumulation_parity.py
python -I -m pytest --import-mode=importlib -q bindings/st-py/tests/test_native_causal_lm_objective_parity.py
```

The no-framework test exercises a complete small synthetic decoder, batch
isolation, causality, last-only projection, NLL, geometry placement and
pre-extraction limits. The optional test compares full logits/NLL to a
Transformers eager GPT-2 in f32 using F32/F16/BF16 source weights, explicit
architecture variants and all three geometry output-hook positions.
Learning checks additionally cover all-recipe LM-loss gradient/SGD parity,
joint placements, finite differences, batch-independent mean reduction, frozen
base/parameters/buffers, stale updates and exact artifact continuation.

An [offline pretrained GPT-2 probe](../../benchmarks/results/2026-10-18-native-gpt2/README.md)
compares all vocabulary logits and NLL for three prompts, with/without a
fixed Wave intervention, plus four greedy steps. All six short continuations
matched the reference token IDs; numerical errors and all raw-result hashes
are published. This is a forward/intervention compatibility observation,
not evidence of training quality or a speed advantage.

The [three-step geometry learning probe](../../benchmarks/results/2026-10-18-native-gpt2-learning/README.md)
then propagates language loss through the same pretrained GPT-2 into simultaneous
Wave, fractional and Topos placements without Torch. It compares every parameter
gradient/update against the optional HF reference and verifies exact native
artifact continuation. These short-window checks establish mechanism agreement,
not held-out quality or whole-model fine-tuning.

The [selected-target follow-up](../../benchmarks/results/2026-10-18-native-gpt2-objective/README.md)
uses explicit suffix labels, smoothing and native target-weighted accumulation
on the same real GPT-2. It records every window, update and next-group
continuation check; it is not a new matched real-model Torch or quality study.
## Learning one attention-output coefficient

On a local native decoder, `model.scalar_output_adapter(layer, left_f64le, right_f64le)`
fixes two unit vectors and starts one trainable coefficient at zero. The
immutable handle shares the frozen decoder, LoRA and geometry. Its ordinary
readout uses the selected output matrix packed from the original weights
plus the scaled outer product. Each new coefficient is packed from the
source, never by repeatedly adding to rounded weights.

```python
import hashlib
import struct

# u and v are fixed unit sequences of length model width; no NumPy needed.
axis = lambda xs: struct.pack(f"<{len(xs)}d", *xs)
state = model.scalar_output_adapter(0, axis(u), axis(v))
state, step = state.step_pairs(positive_prompts, negative_prompts, target_id, alternative_id,
                               learning_rate=1.0, max_step=0.05, max_abs=2.0)
logits = state.last_logits_f32le(prompt)
record = state.checkpoint_json()
restored = model.restore_scalar_output_adapter(record, hashlib.sha256(record.encode()).hexdigest())
```

A pair teaches that the target-minus-alternative logit gap should be
larger on the positive prompt than on the negative one. The native loss is
the mean pairwise logistic loss, not full-vocabulary cross-entropy. The
fixed-axis backward reduction is linear in width per token. Repacking one
square output matrix per changed coefficient is quadratic in width. Readout
uses ordinary dense inference, not a new factorized throughput route.

To avoid accepting an overshoot, use the same pair batch to measure the
ordinary forward loss before and after the proposed update:

```python
state, search = state.step_pairs_backtracking(
    positive_prompts, negative_prompts, target_id, alternative_id,
    learning_rate=1.0, max_step=0.05, max_abs=2.0,
    max_trials=8, min_decrease=1e-7,
)
if search["status"] == "accepted":
    accepted_loss = search["trials"][-1]["objective"]["loss"]
```

The gradient is taken once. The first candidate is the feasible clipped
SGD step, then the displacement is halved on each rejection (1 to 16
candidates). A strict measured drop must exceed or equal both the absolute
floor in nats and 0.0001 times the predicted linear decrease. A refusal
returns the same checkpoint and step count; errors are not refusals.
This only guards the supplied training pairs, never other data. Use
`state.pair_objective(...)` for a forward-only measurement with no update.

## Reusing prompts across counterfactual pairs

If different questions or assignments reuse the same prompts, capture each
distinct prompt once and describe the comparisons with row indices:

```python
bank = model.output_contrast_bank(prompts, target_id, alternative_id,
                                  layer=0, left_f64le=axis(u))
d = bank.direction([[0, 1], [2, 3], [0, 3], [2, 1]], [0, 1, 0, 1])
if d.certificate()["refusal"] is None:
    state = d.scalar_adapter(model)
```

Each pair means contrast(positive row) minus contrast(negative row).
Re-pairing and regrouping do not call the decoder. Repeated pairs keep
their statistical weight. Use 1 to 64 **distinct** prompts, at most 32
pairs and 2 to 8 contiguous nonempty groups. The default fits a common
direction for the normalized group means; `rule="equal_mean"` is a control.
A fixed unit left factor uses width-sized adjoints and produces a right
factor; omitting it uses square weight coordinates and does not create a
rank-one scalar learner. A refusal is not a zero-filled update. Model,
LoRA, geometry and readout identities must match to open a scalar learner.
Both `step_pairs` and the guarded step now reuse repeated prompts *within*
their own forward/gradient call. Receipts distinguish decoder evaluations
from the number of pair terms; no observations carry into another trial.

The separate Rust `NativeCausalLmRankOneProjection` trains the right vector,
not just a scalar coefficient. Its [answer guard](../../benchmarks/paired-prompt-contrast/ANSWER-GUARD.md)
can require a measured decrease in gold-token loss and protect a second
training batch before committing a binding correction. This is a Rust API
and local example; it is not a method on the Python scalar learner above.
