# Finite CE protection for native answer learning

**The guard prevented an increase in either training batch's gold loss on every
guarded update. It did not teach this model to answer the binding question.**
This tests a 32-coordinate first-block V projection of a synthetic decoder, not
pretrained GPT-2, general language quality, or speed.

## Rust API

[The guarded step](../../crates/st-nn/src/causal_lm/quartet_projection/answer/guard.rs)
uses ordinary f32 decoder logits, with unsmoothed full-vocabulary NLL reduced in
f64. First it tries a normalized CE step, halving the displacement as needed.
An anchor must strictly lower active CE and not increase optional protected CE.
Next it tests a correction toward the interaction-only (or fixed mixed) proposal.
Its segment starts at the safe anchor, or at the original state if no anchor was
found. It must preserve the configured fraction of the anchor's **measured**
answer improvement, satisfy the CE floor/Armijo test, preserve protected CE,
and beat both the original and anchor interaction loss by the strict gain floor.
Otherwise it falls back to the anchor or holds. Numerical errors propagate;
the original is not mutated partway through a failing search.

```rust
let guard = CausalLmAnswerGuard {
    length: 0.08, max_trials: 5, retain_answer: 0.75,
    min_answer_drop: 1e-8, min_interaction_gain: 1e-8,
    correction: CausalLmAnswerCorrection::Interaction,
};
let other = CausalLmAnswerProtection { prompts: &other_fit, gold: &other_gold };
let step = projection.step_answers_guarded(&fit, &gold, binding, Some(other), guard)?;
```

Only the active training roster contributes gradients. Protecting the other
training roster evaluates it **without** a backward pass. A local 75% guarantee
does not promise final loss within 25% of a separately trained CE-only model.
Reports expose baseline, eligible/rejected trials, radius/fraction, snapshot IDs,
measured objectives, forward counts, and the committed disposition.

## Matched comparison

Two related starting geometries x two batch orders x six arms: unguarded CE,
unguarded train-scaled mixture, CE anchor alone, guarded fixed-mix correction,
guarded interaction correction (primary), and its query-reflected control. The
latter changes only quartet pairing, never gold labels. All use the same V
coordinates and at most 64 opportunities. Forward counts are **not** matched.

The remaining 16 unused cross-pool A/B pairs supplied eight fit and eight new
within-fit-ID pairs. Those two original pools have no unused cross-pool pairs
left. Eight pairs from two different identity pools supplied an out-of-fit-ID
transfer check. The token IDs are in the same synthetic vocabulary, not novel
natural-language words.

All 1,536 opportunities produced updates. Of the 1,024 guarded updates, 559
committed a correction and 465 committed the CE anchor. No lane stalled. Of
3,662 measured trial candidates, 2,079 were rejected. The primary correction was
accepted on 256/256 opportunities; the reflected correction on 52/256. Guard
objective checks cost 299,904 additional native prompt forwards, not including
the gradient forwards. The unprotected mixture increased a training batch's CE
by as much as 0.0698 nats in one step; the maximum increase across all guarded
steps and both fit batches was zero.

| Within-fit-ID check | Gold NLL range (both starts/orders) | Two-way correct / 64 |
| --- | ---: | ---: |
| Plain CE and guarded anchor | 1.727-1.755 | 32 |
| Unguarded calibrated mixture | 4.388-4.618 | 33 |
| Guarded, same mixed proposal | 1.839-1.875 | 32 |
| Guarded, interaction-directed correction | 1.874-1.922 | 32 |

The primary improved the order-averaged binding margin on 5/8 familiar-ID pairs
and 6/8 out-of-fit pairs versus each of CE-only, anchor-only and reflected, at
both starts and both orders. Only the transfer **mean-margin** gate passed.
Binary accuracy stayed 32/64 on both splits. No endpoint (including unguarded
arms) ranked both correct cells above both incorrect cells on even one of the
16 quartets on **any** split, including fit. Do not promote the guarded correction
as a quality default on this evidence.

**Fit-only follow-up:** a [linear probe](fit_logit_probe.py) read the first 64
saved fit rows, checked against their independent wheel hash, from hold, CE-only
and guarded-primary endpoints. It fit shift-invariant vocabulary-logit features
to answer polarity with two fixed dual ridge strengths (1e-4 and 1e-10 times
mean Gram diagonal). We kept all weights, same-fit and leave-one-fit-edge-out
predictions' scorecards, plus a coordinatewise 1e-4 jitter bound. None of the
20 probes ranked a complete quartet even in-sample; none did on omitted fit
edges. The least-regularized fit gave 34/64 nominal, zero robust binary decisions.
This diagnoses two specified linear fits, **not** impossibility of a separator
or absence of information in earlier states. No decoder call, held/transfer
feature, update to a model, or new quality result was involved.

A next capacity test is to learn an explicitly causal association between an
entity and the following cue (e.g. a previous-position key paired with the
current value), first on one fit batch with ordinary/lagged and shuffled controls.
Require the training quartets to be solved before spending another fresh check
set. Do not keep selecting scalar-loss coefficients on these consumed pairs.

## Verification

Independent ordinary wheel replay reconstructed all trial owners and evaluated
3,850 distinct physical weight states on both fit batches: **246,400 rows**.
It verified measured CE and interaction, every finite acceptance predicate, and
which state was actually committed. The prespecified lane resumed at step 32;
the remaining 32 decisions, trial reports, factors and final owner matched
exactly. Rust ordinary, shared-prefix inference, and independent wheel output
were bit-identical on all 4,992 endpoint rows. 199 Rust causal-LM tests and
158 procedure tests passed. No GPU, wheel rebuild, cache cleanup or publication.

Local raw evidence: `2026-11-04-native-answer-guard-v1`.

| Artifact | SHA-256 |
| --- | --- |
| Prior plan | `d5e38904611f8cf2e07e17dfae30dec6d8410bbf8115483cbb3dc2cc9a09d133` |
| Prior repaired readback | `06820bf153a58efbf812f0d5578462baf06c6e96c34503f1932c90112dc061ac` |
| Native executable | `d439a9fff298958547e31e5b8244b1ee08afc3a5bef929f2718c2e05b3d204ea` |
| Protocol | `1ed621e25b71bbf6c58b8f408bb433d826ab6055259750d53e4e1a414b28ab53` |
| Origin calibration | `04303d18e86e2922cf713e6c3a7203fa7794d3665ab0c0bdbf65cb9d3cd7ab2e` |
| Training | `3debcb9ae09b4d6d5a980084edf75398912285b5c7334f03cec764fba9489744` |
| Independent trial replay | `249f75445130cafda31721c826baf84e9cfc08c3074a3e44283f09cb683ae521` |
| Endpoint serving | `75a1352773f7d0071237e5b356996aed4ec26295346ccbff295e0a629b1e4759` |
| Model-free readback | `12527f079bb216703239a1b39e64139477a1c7973adb213876dec7b8675a3697` |
| Post-hoc fit-only probes | `40fc2ee77df387571af0f64104e739c42f870e0847d5866a0d89458ba01e4b09` |

Build `cargo build --locked --offline --release -p st-nn --example native_quartet_projection`.
Use Python 3.12, `-I -B`, the pinned preceding CPU wheel, `SPIRALTORCH_BACKEND=cpu`
and a fresh private output parent. [Study](guarded_answer_learning.py) phases,
using the preceding printed hashes; external 240-second bound per physical phase:

```text
freeze    PROTOCOL PRIOR_DIR PRIOR_PLAN_SHA PRIOR_READBACK_SHA DRIVER
calibrate PROTOCOL PLAN_SHA DRIVER
train     PROTOCOL PLAN_SHA DRIVER CAL_SHA
witness   PROTOCOL PLAN_SHA DRIVER CAL_SHA TRAIN_SHA
serve     PROTOCOL PLAN_SHA DRIVER TRAIN_SHA
audit     PROTOCOL PLAN_SHA TRAIN_SHA WITNESS_SHA SERVE_SHA
```
