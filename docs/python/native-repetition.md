# Native Repetition-Aware Language Learning

Source-tree API after PyPI 0.4.32; not included in those immutable wheels.
See [native versus HF ownership](native-hf-boundary.md) and
[native corpus training](native-corpus-training.md) for loading a supported
frozen GPT-2 and attaching trainable geometry.

Rust owns the candidate selector, loss, vocabulary/decoder pullbacks and
geometry update. Python only transports configuration, IDs and results.
Torch, Transformers and NumPy are unnecessary for this path. The existing
HF Trainer helper is still a separate optional foreign-model client.

## Use the Native Objective

`model` below is a `NativeCausalLm` with trainable geometry attached;
`documents` contains integer token-ID lists for that model's vocabulary.
No tokenizer, padding or attention mask is inferred.

```python
import spiraltorch as st

repetition = {
    "kind": "native_v1",
    "config": {
        "strength": 0.3,
        "candidate_source": {"kind": "prior_continuation", "ngram_order": 2},
        "context_window": 64,
        "max_candidates_per_position": 8,
    },
}
optimizer = st.NativeGeometryAdamW(model.geometry_artifact())
trainer = st.NativeCausalLmTrainer(
    model, optimizer, documents, 128,
    learning_rate=0.001, microbatch_windows=2, accumulation_steps=4,
    epochs=2, repetition=repetition,
)
step = trainer.step()
if step is not None:
    print(step.loss, step.repetition_loss, step.repetition_positions)

# For diagnostics or custom update groups, use the same Rust objective.
batch = model.geometry_loss_gradients(
    [1, 2, 1, 2, 1, 3], [1, 6], repetition=repetition,
)
print(batch.causal_loss, batch.repetition_loss, batch.loss)
group = model.geometry_accumulator(repetition=repetition)
group.add(batch)
updated = group.step_sgd(model, 0.001)
```

The same optional JSON-compatible `repetition` dictionary is accepted by
`next_token_loss`. General mapping objects such as `MappingProxyType` are not
accepted; use a plain `dict` at this bounded JSON transport boundary.
**Omit it when reporting ordinary held-out CE/perplexity**: explicitly passing
it requests the combined objective, not ordinary evaluation loss.
`repetition_json()` on gradient batches/accumulators exposes the owned recipe.
Recipe mutation in Python does not change an existing native owner.
The offline `native_geometry_corpus_training.py` example also accepts this
optional `repetition` entry in its training recipe JSON; resumed runs use the
recipe already pinned in their checkpoint.

## Three Candidate Sources

- `prior_continuation` with `ngram_order`: historical alternative continuations
  for the current preceding prefix; ranked by count, recency, then token ID.
- `model_topk_history` with `proposal_top_k`: native model proposals that occur
  in valid bounded history, excluding the supervised target.
- `model_topk_periodic` with `proposal_top_k`: the same proposals, restricted
  to those completing a periodic suffix (period at most 16, at least 3 repeats).

Native top-k is stop-gradient, descending finite logit then ascending token ID
on ties. It does not claim Torch top-k tie equivalence. The shared selector is
used without relabeling the legacy HF v3 report as a native result.
Candidate history restarts at each document window and never crosses batch rows.
With positive strength, supplied labels must equal the input IDs or `-100`.
Ignored labels remove objective terms, not their tokens from attention/history.
No selected targets is an error; no negative candidates gives exact zero UL.

## Reduction and Continuation

For each active position, take the candidate mean of
`-log(1 - min(softmax(logits)[candidate], 1 - 1e-6))`.
UL is the mean across active positions. The objective is
`CE + strength * UL`; smoothing affects CE only.

CE and its gradients accumulate by selected shifted target count, while UL and
its gradients accumulate by active-position count. They combine once before the
optimizer's clipping/update. Thus a microbatch with no candidates contributes CE
but does not dilute UL. A simple average of combined microbatch losses is wrong.
The separate pullbacks cost extra work; this is not a speed optimization claim.

The versioned recipe, arithmetic policy, complete data/labels, optimizer and
cursor are bound by the native training checkpoint. Resume at accepted update
boundaries; pending accumulators are not checkpointed. Recipe-mismatched batches
are rejected before accumulation. Zero strength is an exact numerical CE no-op,
but the explicit recipe still has a distinct checkpoint identity.
Absent repetition preserves the original default serialization.

Admission rejects invalid configurations and excessive sequence/proposal/plan
work before decoder execution. The trainer preflights every microbatch's mask
before its first update, including batches later in the corpus.
Storage remains CPU f32 with stable f64 loss
reductions, not resident WGPU execution. Correct VJPs and exact resume establish
mechanical correctness, not better language, fewer loops or whole-model FT.
