# Native Causal Language Models

Source-only extension after the fixed 0.4.32 candidate. This does not establish
PyPI publication or change the existing optional HF Trainer implementation.

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
mask, custom position IDs, KV cache, dropout or whole-model backward/update is
implemented here. Frozen inference is not full-model native fine-tuning.

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

The model and attached geometry are frozen snapshots. To use updated geometry,
attach the newly saved artifact to a new handle; this method does not claim a
language-loss pullback through the entire decoder. See
[native geometry learning](native-geometry-runtime.md) for existing selected-site
VJP/SGD, which remains distinct from full-model training.

## Reproduce Checks

```sh
cargo test -p st-nn --lib causal_lm:: --locked
python -I -B bindings/st-py/tests/test_native_causal_lm.py
# Optional comparison environment only:
python -I -m pytest --import-mode=importlib -q bindings/st-py/tests/test_native_causal_lm_parity.py
```

The no-framework test exercises a complete small synthetic decoder, batch
isolation, causality, last-only projection, NLL, geometry placement and
pre-extraction limits. The optional test compares full logits/NLL to a
Transformers eager GPT-2 in f32 using F32/F16/BF16 source weights, explicit
architecture variants and all three geometry output-hook positions.

An [offline pretrained GPT-2 probe](../../benchmarks/results/2026-10-18-native-gpt2/README.md)
compares all vocabulary logits and NLL for three prompts, with/without a
fixed Wave intervention, plus four greedy steps. All six short continuations
matched the reference token IDs; numerical errors and all raw-result hashes
are published. This is a forward/intervention compatibility observation,
not evidence of training quality or a speed advantage.
