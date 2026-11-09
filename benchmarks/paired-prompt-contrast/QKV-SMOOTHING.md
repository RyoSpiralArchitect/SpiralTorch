# Native Answer-QKV Label Smoothing

Status: implemented and CPU-tested. A subsequent
[fixed-coefficient comparison](QKV-SMOOTHING-COMPARISON.md) now records relative
improvement and remaining competence failures; the evidence below concerns
the original implementation and diagnostic readback.
This does not change the previously sealed [layout experiment](LENGTH-POSITION.md)
or turn its failed competence gate into a pass.

## What The Saved Outputs Show

Post-hoc decomposition of the same primary filler-58 / untrained-pair logits,
with no new model calls, selection, temperature tuning or quality gate:

| Endpoint / outcome | Count | Mean gold NLL | Mean predicted top probability |
| --- | ---: | ---: | ---: |
| Ordinary-varied / correct | 289 | 0.757466 | 0.469260 |
| Ordinary-varied / wrong answer | 287 | 0.864657 | 0.469486 |
| Lag1-fixed / correct | 347 | 0.135770 | 0.887016 |
| Lag1-fixed / wrong answer | 229 | 5.914106 | 0.825388 |
| Lag1-varied / correct | 460 | 0.125763 | 0.892078 |
| Lag1-varied / wrong answer | 116 | 5.213732 | 0.849487 |

All incorrect final predictions select the other answer ID; none select an ID
outside the two answers. The lag1-varied model's 116 errors contribute about
91% of its summed NLL. This is a descriptive error concentration and confidence
observation, not a calibration study or proof of its internal cause.
Original logits, the pinned readback procedure and `error-profile.json` remain
in local validation folder `2026-11-10-native-length-position-v1`.

Label smoothing is one testable training intervention, not an established cure
for mistaken entity binding. It may trade accuracy for probability quality;
it must be compared prospectively without optimizing its strength on these
already-consumed evaluation outputs.

## Rust API

`NativeCausalLmQkvAdamW` now exposes:

- `answer_sequence_gradients_with_label_smoothing(prompts, gold, max_rows, epsilon)`.
- `step_answer_sequences_with_label_smoothing(prompts, gold, max_rows, epsilon, step)`.

The target distribution is `(1-epsilon) * one_hot(gold) + epsilon / vocabulary`.
The existing `CrossEntropyConfig` / tensor CE and VJP own that calculation;
there is no second Python objective. Epsilon must be finite and within `[0, 1]`.
Admission occurs before forwarding any batch. Ragged sequences still receive
one answer vote each, not one vote per prefix token or per length group.

The gradient method returns **smoothed objective loss** and its derivative.
The step returns `CausalLmQkvSmoothedStep` containing `objective_loss`,
`label_smoothing`, `steps`, and Q/K/V `gradient_l2`. It deliberately has no
`gold_nll` field: smoothed training CE must not be mistaken for unsmoothed
evaluation NLL. The legacy methods and `CausalLmQkvStep` retain plain CE and
their original report fields.

The checked optimizer kernel is shared and remains atomic. Zero rate freezes
the state; invalid epsilon cannot alter weights, moments or clock. Checkpoint
schema and serving identities are unchanged. Like the next batch and learning
rate, the next epsilon belongs to the caller's recorded schedule, not to the
optimizer checkpoint. Exact continuation requires preserving all of them.
Smoothing is not applied to inference logits or ordinary NLL evaluation.

## Verification

212 native decoder CPU tests pass. New checks cover ordinary/lag-1 attention,
frozen LoRA/geometry, epsilon 0/.1/1, uneven length groups, independent full-vocab
loss calculation and singleton-gradient means. Physical edits of all 48 QKV
weights under each of two attention routes (96 finite differences, epsilon .2)
agree with backpropagation within absolute `1e-4`.

Epsilon zero yields exactly the legacy loss and optimizer checkpoint. Invalid
values and zero rate preserve state. Continuing with the same smoothing after
restore reproduces the full checkpoint exactly. Serialization checks ensure
the new report cannot silently label the objective as gold NLL.

Evidence: `2026-11-10-native-qkv-smoothing-v1/decoder-tests-receipt.json`.
This implementation-only stage did not retrain the saved model or perform GPU,
wheel, Python/WASM binding, push or release work. The linked follow-up froze
its coefficient, matched CE control and evaluation boundary before execution.
