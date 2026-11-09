# Causal Key Lag: A Native QKV Capacity Test

This is a **synthetic training-set capacity result**, not pretrained GPT-2
fine-tuning, held-out generalization, a geometry ablation, or a PyTorch speed
comparison. All model forward/backward/optimizer work is native Rust.
Python freezes inputs, invokes the executable, and independently reads saved
logits and checkpoint bytes; it does not train a second model.

## Why Change The Operator?

The preceding [finite answer guard](ANSWER-GUARD.md) could protect answer CE
without fitting the entity-to-answer assignment. This follow-up changes a
specific information path: a fact's entity key can address its successor value.

At visible value slot `j`, lag `d` scores the query against raw key
`K[max(0, j-d)]`, but reads `V[j]`. The mask, query position, value position,
and any score-bias position remain unchanged. The source cannot cross a head
or sequence boundary. Backward scatter-adds the key gradient to the original
source, including the duplicated clamped first key. Incremental inference
retains the ordinary raw-key cache.

- Lag 0 uses the unchanged ordinary attention entry point.
- Nonzero lag requires the explicit `spiraltorch_gpt2_key_lag_v1` architecture
  and `spiraltorch_key_lag: {layer, lag}` in the config.
- Layer and lag participate in the base identity. Existing adapters cannot
  silently migrate to a changed operator; this study explicitly rebinds only
  their base-identity field, keeping every numerical parameter fixed.
- The old Python wheel rejects both tagged lagged architectures. It remains
  an independent ordinary-route replay control, not a lagged-model executor.

The core is in [attention.rs](../../crates/st-kernel-contracts/src/attention.rs)
and [its VJP](../../crates/st-kernel-contracts/src/attention/backward.rs).
The decoder, training replay and KV path use that same contract.

## Fixed Comparison

Use only the preceding study's 32 already-consumed `train0` prompts: four
unordered identity pairs, both fact orders, both value assignments, and both
queries. These are **not 32 independent samples**. There is one initialization,
one frozen geometry start, two blocks, width 32, four heads and 64 vocabulary IDs.

Train all 3,072 Q/K/V weight entries in block 0. Bias, embeddings, output head,
block 1, rank-2 LoRA and geometry remain frozen. The ordinary, lag-1 and lag-2
variants start with identical weight bytes.

The 12 lanes are three lags x two layouts x two label conditions:

- Primary layout ends in the query entity followed by the answer marker.
- Secondary layout swaps only those final two IDs.
- Aligned training duplicates the same 32 prompts/labels.
- Reflected training uses the same duplicated prompts, with opposite labels
  in the second copy, cancelling the answer assignment while retaining the
  vocabulary and prompt exposure.

Each lane uses AdamW (0.9/0.999, epsilon 1e-8, learning rate .01, no decay) for
256 logical updates. Checkpoints are fixed at 0/16/64/128/256. Total: 3,072
updates, plus a predetermined 128-step resume witness, not another selected run.

The final-step capacity gate requires **32/32 unrestricted vocabulary top-1**
and **8/8 strictly ranked assignment/query quartets**. NLL <= .1 is a separate
threshold. A lag-1-specific endpoint additionally requires ordinary aligned,
lag-2 aligned and lag-1 reflected controls all to miss the capacity gate on
the same layout. No best-checkpoint selection rescues the final result.

## Microbatch Amendment And First Failures

The first protocol stopped in preflight, before any update: a 32x9x32 tensor
exceeded the frozen geometry adapter's 4,096-value limit. That rejection and
its executable remain saved. Version 2 does **not** raise the limit or change
weights. It keeps complete sequences in eight-row microbatches (2,304 values).

`NativeCausalLmQkvAdamW::answer_gradients_microbatched` weights each microbatch
mean by its actual sample count, accumulates in compensated f64, and converts
once to the logical f32 mean. `step_answers_microbatched` changes parameters
and the AdamW clock only after every microbatch succeeds. This is one optimizer
update per 64 examples, not eight sequential optimizer updates. Tests include
unequal chunk sizes, unchanged full-batch fast path, invalid tails, zero rate,
unchanged weights on failure and the single clock increment.

The interrupted November 4 CPU-test attempt has no completion receipt and is
not counted as passing. On November 10 the first resumed build stopped at its
4 GiB disk floor. That first stop is retained. Only after explicit approval,
three named unused incremental caches were removed; their measured free-space
increase was 1.279 GiB. The subsequent CPU tests/build completed without a
numerical retry. No wheel was rebuilt for this study.

## Observed Endpoints, 2026-11-10

All values below are from the predetermined final step 256.

| Layout | Lag | Labels | Full-vocab correct / 32 | Ranked quartets / 8 | Gold NLL |
| --- | ---: | --- | ---: | ---: | ---: |
| Marker-last | 0 | Aligned | 16 | 0 | 0.809043 |
| Marker-last | 0 | Reflected | 16 | 0 | 0.809080 |
| Marker-last | 1 | Aligned | **32** | **8** | **0.046369** |
| Marker-last | 1 | Reflected | 16 | 0 | 0.810364 |
| Marker-last | 2 | Aligned | 29 | 7 | 0.537161 |
| Marker-last | 2 | Reflected | 16 | 0 | 0.809246 |
| Query-last | 0 | Aligned | 18 | 0 | 0.809921 |
| Query-last | 0 | Reflected | 16 | 1 | 0.813142 |
| Query-last | 1 | Aligned | **32** | **8** | **0.042135** |
| Query-last | 1 | Reflected | 16 | 1 | 0.820259 |
| Query-last | 2 | Aligned | **32** | **8** | **0.052752** |
| Query-last | 2 | Reflected | 16 | 0 | 0.815999 |

The primary lag-1 capacity and specificity gates pass **within this fixed
experiment**. The secondary layout passes capacity but **fails lag-1
specificity**, because lag 2 also fits. The latter is important: the outcome
depends on layout and does not establish a universally special one-token lag.

Lag 1 first reaches the primary choice/ranking gate at the fixed step-64
checkpoint, but its NLL then is .113545. The final comparison is still step
256. No claim about ordinary Attention's general capacity follows from its
failure under this one initialization, trainable subset and update budget.

## Verification

- 204 native decoder CPU tests and 11 attention-contract tests pass.
- 163 model-free Python procedure tests pass.
- All 18 physical Q/K/V directional finite differences pass before training
  (all three operators and both layouts).
- Changing only gradient microbatch size from eight to four gives equal loss
  and maximum gradient difference 5.960464477539063e-8.
- All 1,920 saved checkpoint/question rows have exactly equal ordinary and
  prefix-cache logits, stronger than the preregistered tolerance.
- All 24 start/final lane snapshots are physically reimported from freshly
  spliced safetensors; ordinary and cached logits agree bitwise with the
  optimizer's serving state. Ordinary variants also agree with the old wheel.
- The predefined lag-1/marker/aligned split 128+128 reproduces all remaining
  update reports and the final checkpoint, including optimizer moments, exactly.
- Geometry's contribution is not isolated: it is held constant in every lane.

## Reproduce And Preserve The Boundary

Use [causal_key_lag_fit.py](causal_key_lag_fit.py) and the native
[example](../../crates/st-nn/examples/native_key_lag_fit.rs), with the pinned
parent inputs and reference wheel. The ordered phases are `freeze`, `check`,
`fit`, `score`, and `audit`; each prints its sealed output SHA-256.
Every stage takes the preceding pins, uses exclusive output directories and
checks source/binary/input identities. CPU, one attempt, stop on error, each
physical phase <=240 seconds. The local supervisor also enforces a 4 GiB disk
floor. Do not reuse output directories or treat missing receipts as passes.

Parent plan: `1ed621e25b71bbf6c58b8f408bb433d826ab6055259750d53e4e1a414b28ab53`.
Parent training: `3debcb9ae09b4d6d5a980084edf75398912285b5c7334f03cec764fba9489744`.

| Artifact | SHA-256 |
| --- | --- |
| v1 rejected plan | `9ea159b5229e3871d4f5e81fa7e279cb633419c110b0f4d0ba0f2c481b61eb95` |
| v1 first rejection log | `bc21e49b1d64eb50d9cc52cf5cc303e21336bcfc49d9fee9574ab14d65eaa26b` |
| v2 native executable | `ea8a131a8e2b5af6dee6cae41b9a3ad57552793d58680e086d2fff417f00262e` |
| v2 plan | `641689cc0a960fe058ab3541a91f1e557033d22c32edd7f602083afe5a035bf5` |
| v2 preflight | `07a79068e0fff9273f5ee20125da690599cf062270a0b9f1254ce421f519ddad` |
| v2 training | `25efbffe525edeaec2df2e24aa80f94fe3d400e75dd084a3d3d8a7a0d089cee3` |
| v2 serving | `6de01ac5df4130e385df6458e0c7b0543b9bf1af6981f7f37f0c4a660d819d55` |
| v2 model-free readback | `d3b7e59d6b8d45c007c5969c1d3c3bb92704ddf3bb8db8d5fc9d8dcde50e7142` |

Originals remain under the local validation folder
`2026-11-10-native-causal-key-lag-microbatch-v2`.
The original Git common directory was missing on resumption. The surviving
1,744 source files were archived and verified byte-identical during recovery;
207 differing/new files were rescued on current main as local snapshot
`be3fec89551af970071407e43c0de8ad0e194dd3`. This does not reconstruct the
missing local commit history. A separate Git bundle and the source archive
preserve the rescue. Nothing here implies push, merge, GPU validation or release.

Next scientific boundary: freeze unseen associations and varied entity/value
separations before inspecting their outputs, using these already-selected
endpoints. Distinguish memorized template capacity from transferable binding.
No such holdout was opened in this run.

