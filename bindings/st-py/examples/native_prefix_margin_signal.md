# Trace a prefix gate to two candidate tokens

Given a native GPT-2 decoder already carrying a prefix-transport adapter on
the **last MLP output**, ask how its gate could change the next-token logit
difference. This is a local, CPU-only development API, not part of a newly
published wheel. It does not use Torch, NumPy, generation, an optimizer, or
an English parser.

```python
# model: spiraltorch.NativeCausalLm; prompt_ids: a single nonempty prefix
r = model.last_mlp_prefix_margin_signal(prompt_ids, target_id, rival_id)
steps, width = r["shape"]
axis = 0
per_position = [
    r["source_features"][t * width + axis] * r["bias_gradient"][axis]
    for t in range(steps)
]
print("target minus rival:", r["contrast"]["target_minus_alternative"])
print("gate slope, reconstructed / native:", sum(per_position), r["gate_gradient"][axis])
```

`site_input` contains the factual pre-adapter activations X[t]. In row-major
`source_features`, each row is R^(steps-1-t) X[t] / steps. The `upstream`
vector is the unit logit difference's adjoint at the last site's output,
including the actual final layer norm and vocabulary projection. The raw
bias gradient includes residual strength and f32 rounding. Multiplying
these factors gives contributions to a **parameter gradient**, not a
derivative with respect to discrete token IDs or an attribution of meaning.
The reconstructed sum can differ slightly from the native gradient because
the runtime rounds the prefix mean to f32.

Both candidate IDs must be distinct and in range. The model's own bounds
and a separate cap of 262,144 returned vector elements are enforced before
Python copies IDs. The report identifies its base, active/frozen LoRA,
geometry and readout precision. Existing weights, caches and trainable
flags are unchanged; a frozen gate or bias still has an observable raw
derivative. A downstream block-output adapter is not silently ignored.
The corresponding Rust method is
`st_nn::causal_lm::NativeCausalLm::last_mlp_prefix_margin_signal`.
