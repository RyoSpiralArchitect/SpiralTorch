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
Positions restart at zero in each row. No padding, token-type IDs, attention
mask, custom position IDs, KV cache, dropout or base-weight updates are
implemented here. Frozen inference and the geometry-only learning below are
not full-model native fine-tuning.

The initial execution backend is `rust_f32_cpu_prepacked`. Layer projections
are packed once; a tied output projection shares the token table storage.
Last-only output skips vocabulary projection for earlier positions, not the
causal prefix computation. F16/BF16 checkpoint values explicitly become f32.
This is not a resident-GPU or speedup claim. The Rust core compiles for WASM;
a JavaScript client and browser model execution require separate work.

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
not change this explicitly frozen, no-cache execution policy. No remote code
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
budgets are checked before foreign array extraction. Invalid inputs do not
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
