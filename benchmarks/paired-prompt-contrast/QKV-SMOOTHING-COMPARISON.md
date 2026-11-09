# Fixed-Coefficient QKV Smoothing Comparison

CPU comparison completed on 2026-11-10. **Relative efficacy passed; absolute
competence failed.** This is an exploratory synthetic-model result on already
consumed associations and layouts, not a fresh holdout or pretrained-LM claim.

## Frozen Comparison

Use the saved [varied-layout training schedule](LENGTH-POSITION.md), identical
initial numerical weights, block-0 QKV trainable, and frozen geometry/LoRA and
remaining weights. Cross ordinary/lag-1 Attention with plain CE/label smoothing
**0.1**, fixed before model calls. No coefficient search or selected checkpoint.

Each of four lanes receives the same 256 updates, 96 answers and 1,056 input
tokens per update, cap four, learning rate .01 and zero decay. Save every state
for batch transport; score only final step 256. Replay the predetermined
smoothed lag-1 step 128 through 256, adding 128 updates but no selected run.

The plain-CE controls must reproduce every previous update report, optimizer
checkpoint digest and final output bit. A mismatch aborts rather than being
interpreted as a smoothing effect. Evaluation uses ordinary **unsmoothed** gold
NLL and strict full-vocabulary top-1. Smoothed training loss is never substituted
for NLL. Primary efficacy requires lower NLL without losing correct answers on
the six filler-58 untrained-pair cells. Competence still requires >=87/96 in
every cell; filler-59 results cannot rescue either primary gate.

## Observed Endpoints

| Attention | Objective | Correct / 576 | Gold NLL |
| --- | --- | ---: | ---: |
| Ordinary | CE | 289 | 0.810875 |
| Ordinary | Smoothed .1 | 285 | 0.846612 |
| Lag 1 | CE | 460 | 1.150424 |
| Lag 1 | Smoothed .1 | **480** | **0.876930** |

Lag-1 smoothing gains 20 answers (3.47 percentage points) and lowers NLL by
0.273493. The primary relative gate passes. However, its NLL is still worse
than ordinary CE, and the ordinary route worsens on both measures when smoothed.
This is not a reason to change the library's default objective.

All six primary cells, lag 1:

| Added fillers | Placement | CE correct / 96 | Smooth correct / 96 | CE NLL | Smooth NLL |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | After BOS | 76 | 74 | 1.115142 | 0.939160 |
| 1 | Between facts | 62 | 73 | 1.445278 | 0.959452 |
| 1 | Before query | 82 | 86 | 0.567016 | 0.524310 |
| 3 | After BOS | 72 | 76 | 1.942508 | 1.366078 |
| 3 | Between facts | 72 | 76 | 1.726144 | 1.203226 |
| 3 | Before query | 96 | 95 | 0.106454 | 0.269356 |

Five cells still miss competence. Two lose accuracy, and the previously perfect
cell also worsens in NLL. The aggregate gain does not mean every case improves.

Additional predefined views, lag 1:

| Filler / pair group | CE correct | Smooth correct | CE NLL | Smooth NLL |
| --- | ---: | ---: | ---: | ---: |
| 58 / trained pairs, out of 192 | 155 | 167 | 1.081459 | 0.773564 |
| 59 / trained pairs, out of 192 | 149 | 165 | 1.185695 | 0.810235 |
| 59 / untrained pairs, out of 576 | 450 | 483 | 1.237688 | 0.893866 |

All ordinary-route and per-cell outcomes remain in the audit, including its
secondary untrained-pair decrease from 291 to 284. One initialization and one
coefficient do not establish a general advantage or the internal cause of the
earlier confident errors. Frozen common geometry is not an isolated ablation.

## Implementation And Verification

The native example now accepts optional `label_smoothing` for `fit` and
`gradient`. Omission preserves legacy CE and its report schema. Explicit
smoothing uses the [shared Rust objective](QKV-SMOOTHING.md), returning
`objective_loss` and `label_smoothing`, never a mislabeled `gold_nll`. Gradient
exports use the same distinction. Equal-length admission still applies unless
`batch_kind` explicitly selects unpadded sequences. Inference is unchanged.

Three example tests and 12 Python procedure tests pass. The backend's preceding
212 tests cover the native objective and finite differences. All 512 CE control
updates reproduce prior checkpoints/reports, and all 3,072 CE evaluation rows
reproduce prior logits exactly. The smoothed 128-step continuation is exact at
every update. All 6,144 scored rows have identical ordinary/cache outputs.

There was one successful local executable build, one training phase (49.49s)
and one scoring phase (2.63s), with no retries or altered gates. These times
include process/file overhead, not a throughput benchmark. Inputs, saved
executable and source inventories stayed unchanged during physical phases.
Previous raw experiments were not overwritten; their source-bound readers
require their recorded source revision, not an updated helper's hash.

Runner: [qkv_smoothing_study.py](qkv_smoothing_study.py). Execution source:
`96e9927`. Local originals: `2026-11-10-native-qkv-smoothing-comparison-v1`.
Phases: `freeze PARENT DRIVER NEW_PLAN`, `train PLAN PLAN_SHA`,
`score PLAN PLAN_SHA TRAIN_SHA`, `audit PLAN PLAN_SHA TRAIN_SHA SCORE_SHA`.
Every output is exclusive; all first failures must be retained.

| Artifact | SHA-256 |
| --- | --- |
| Executable | `e6b33580584113f8b2fe36f2a19edd6a280b36e1b2935347c99c701372bc7802` |
| Plan | `72361929d059757822c9ce43b25ef10f953945d9279f69c5b3447696da5f8648` |
| Training | `4b739c0c9dbea1890d6c7f1c93306b8e7921f9a148ae2cbe0456a13a468fb06f` |
| Scoring | `e9067ee01c9ff6e15f8e502e2a802f7935741d2839717178d7864c5cf6e5cc38` |
| Audit | `119489d6836040d0aaa9b3fdac5ec49570a31b18cf5af9d8aae0bdc914647a2b` |

No GPU, wheel rebuild, Python/WASM binding exposure, push or release occurred.
Smoothing stays opt-in. A further experiment needs a newly fixed question and
evaluation boundary, rather than selecting another epsilon on these scores.
