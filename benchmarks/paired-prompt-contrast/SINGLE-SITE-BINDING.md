# Learning K or Q from counterfactual quartets

**Result:** K-only training on the ordinary pairwise loss did not beat the matched Q route
on enough new pairs. Changing *where the logistic is applied* did improve the measured
binding interaction across partners, both input directions and both frozen geometry states.
It still did not make the model choose the requested answer.

## Native mechanism

[NativeCausalLmRankOneProjection](../../crates/st-nn/src/causal_lm/quartet_projection.rs)
learns one width-vector at a selected layer and Q, K, V or O site. With fixed unit left factor
`u`, its right factor `v` merges as `f32(f64(W) + u * v^T)` from the frozen origin each
time. At Q/K/V only the appropriate columns of the fused matrix change; updating clears
the affected transpose cache. The private model is not returned under the frozen model's
identity. The ordinary decoder supplies both inference and VJP. The compact pullback
accumulates `(X u)^T dZ` without a width-squared gradient; physical deployment still
repacks a matrix. No speed claim follows.

For four indexed prompts with gaps `a,b,c,d` in assignment-by-query order, pairwise training
averages four losses on `a-c, d-b, a-b, d-c`. Interaction training instead applies
`softplus(-s)` to `s = ((a-b) + (d-c))/2` separately for each edge and presentation.
Adding arbitrary row-only or column-only offsets to a quartet leaves this latter loss
unchanged. This property belongs to the objective: it does not constrain main effects
in the *deployed logits*. The symmetric control reflects the query in the second copy of
each quartet, without changing which prompts occur. The Rust learner evaluates each
unique prompt once per step even when its quartet is presented twice.

## Prospective comparison

The same eight identity roles from the [partner trial](PARTNER-ROUTING.md) have 28
unordered pairs. Of the 16 unused pairs, eight offset-3 edges form the two training
batches and eight offset-2 edges form the two check batches. No previously evaluated
edge was reused. **All 28 pairs in this population are now consumed.** These are new
partner combinations of known entities, not new vocabulary or independent model seeds.

Each of 32 arms starts from zero: Q or K x either of the two fixed Walsh directions x
pairwise or interaction x aligned or symmetric supervision x two dependent geometry
states. Only the right factor trains (32 coordinates); the base, other projections,
LoRA and geometry are frozen. All arms took eight nonzero native steps of length 0.08
on the same batch schedule. First losses and all 32 components of the first gradient
in every arm agreed with an independently contracted dense pullback from the existing
wheel. Unrestricted endpoint logits agreed **bit-for-bit** on all 4,352 rows
(34 models including holds x 128 prompts) between the Rust driver and ordinary
inference on the separately edited and reimported safetensors archive. This is a
synthetic 2-block, 4-head decoder with CPU f32 activations, not pretrained GPT-2.

| Key-only at the original K direction | Initial geometry | Subset-rank geometry |
| --- | ---: | ---: |
| Pairwise: mean-margin wins vs hold / same-direction Q | 7/8 / 4/8 | 7/8 / 4/8 |
| Interaction: wins vs hold / same-direction Q | 7/8 / 7/8 | 7/8 / 7/8 |
| Interaction: wins vs symmetric K / aligned pairwise K | 7/8 / 7/8 | 8/8 / 7/8 |
| Interaction: mean margin change vs hold | +0.000332 | +0.000351 |
| Interaction: all four margins improve vs hold | 0/8 | 0/8 |

The prespecified pairwise lead therefore **failed** (needed at least 6/8 wins against
each control in each state). The interaction alternative **passed the mean-margin
gate at both fixed directions**. No arm improved two-way choices beyond 32/64
or ranked both correct cells above both incorrect cells within even one of the
16 edge/order quartets. Full-vocabulary top-1 stayed 6/64. These are the same
eight check edges assessed multiple ways, not independent replications.

Why the mean is not enough: at the original K direction, even after interaction
training, all 64 held prompts still prefer token 2 over 16 at both starts.
As a descriptive readback, write the four order-averaged margin changes as
`S+x, S-x, S+y, S-y`. On all eight edges at both starts, `max(|x|,|y|)>|S|`.
The median absolute interaction change is 0.000197 / 0.000232, whereas the median
of that larger nuisance change is 0.00436 / 0.00412. A scalar answer bias alone
cannot fix the poor within-quartet ranking.

A next test is **where** keys are built. In the first block, key projection sees
a normalized per-position embedding, before context mixing; a status cue's input
at that position cannot distinguish who precedes it. The second block can see
contextualized cues. This is a hypothesis, not a localization from attention maps.
Test layer 0 vs layer 1 at the same budget on new pairs, and retain both the strict
within-quartet and actual-answer decisions.

## Replay

Local evidence directory: `2026-11-04-native-single-site-binding-v1`. Training
records hold the exact right-factor bits after every update; the plan binds these
to the source identities, layer, site and fixed direction. Never treat a bare
right vector as a portable model. The [Rust example](../../crates/st-nn/examples/native_quartet_projection.rs)
accepts separately held config, weights, LoRA and geometry. The [orchestrator](single_site_binding.py)
uses the previously sealed wheel only for source export, dense reference checks
and independent ordinary replay, not for optimization. With Python 3.12, that
wheel installed, a fresh output parent and `SPIRALTORCH_BACKEND=cpu`:

```text
cargo build --locked --offline --release -p st-nn --example native_quartet_projection
single_site_binding.py freeze PROTOCOL PRIOR_DIR PRIOR_PLAN_SHA PRIOR_AUDIT_SHA DRIVER
single_site_binding.py train  PROTOCOL PLAN_SHA DRIVER
single_site_binding.py score  PROTOCOL PLAN_SHA DRIVER TRAIN_SHA
single_site_binding.py audit  PROTOCOL PLAN_SHA TRAIN_SHA SCORE_SHA
```

Invoke Python with `-I -B`, use a 240-second external wall limit for each physical
phase, and read the next hash from the successful stage's stdout. No retries, model
download, GPU, wheel rebuild, push or release was part of this comparison.

| Artifact | SHA-256 |
| --- | --- |
| Prior plan | `68d373c90c2e26f20fff218ef647a3343a9bf177123816c558f995083eed5f97` |
| Prior audit | `3149399d253c5d59340594585d727b76ff5cb4b8b8926448304cf54a68c218e4` |
| Native driver in this run | `ff5c567b48e305b90e47c05ee77bd25b885801d51b8ee0799c3cc67e72d68237` |
| New plan | `113a0efec6e84807acdad593bf265004267577dfadc1a54bb8005b4f563b0f8b` |
| Training records | `b3150939b7b5e91daa0b714890199b856d3f15e27df1d57c894b0e86be70d2c1` |
| Scored raw row inventory | `5456805cb6ac3a4ede4b6e960d7f8b9953b0532c78fea8fd73dc9db29dcae9d8` |
| Model-free readback | `6cde9819797df12802bb7b3fb6919635247eb2abe6f19609a9cc249e13b016a8` |

194 Rust causal-LM tests and 143 Python procedure tests passed. The new Rust API
is usable directly; it is not a new class in the previously built Python wheel.
