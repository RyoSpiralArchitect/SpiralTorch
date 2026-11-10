# Matched Length/Position Training

Status: the frozen comparison completed on CPU, 2026-11-10.
**Relative efficacy passed; absolute competence failed.**
This follows the failed layout-robustness gates in [KEY-LAG-TRANSFER.md](KEY-LAG-TRANSFER.md)
and uses the [native unpadded QKV API](RAGGED-QKV.md).

Four lanes cross ordinary/lag-1 attention with fixed/varied filler placement.
All start from the same saved synthetic numerical weights, with explicit
architecture ownership. Only block-0 QKV learns. Geometry and LoRA remain fixed,
so any result cannot isolate a geometry contribution.

Every update has the same 96 labels, order and lengths: 32 examples each of
length 9, 11 and 13, totaling 1,056 input tokens. Four-row groups retain the same
shapes and computational work across lanes. Fixed placement adds fillers after
BOS; varied placement rotates between after BOS, after the first fact separator
and after the second separator. Entity/cue adjacency never changes. Token
multisets, query positions and labels match between paired conditions.
This is a layout comparison, not an isolated absolute-position intervention.

The plan materializes all 256 batches for each placement policy before running.
It uses one fixed input permutation, one initialization and no model-seed search.
The planned budget is 1,024 main updates plus one predefined 128-step exact
continuation check. Evaluation is only at origin and final step 256, never a
selected intermediate checkpoint.

Evaluation uses lengths 10 and 12, three placements and fillers 58/59. Each
cell contains 32 trained-pair views and 96 untrained-pair views. These token
strings differ in length from training, but the associations are already-used
research material. **They are not a globally fresh confirmatory holdout.**

The primary comparison uses filler 58: lag1-varied must beat lag1-fixed in
aggregate full-vocabulary accuracy without worsening mean gold NLL. Separately,
competence requires at least 87/96 correct in every one of six untrained-pair
cells. Report failed gates, all four conditions and all per-cell results.
Filler 59 is secondary and cannot rescue the primary. The factorial interaction
is descriptive; one initialization does not establish general superiority.

Preparation code: [length_position_protocol.py](length_position_protocol.py).
Four model-free tests cover every update's matched work/labels, entity/cue
adjacency, complete evaluation rosters, exact factual-prefix boundaries and
invalid inputs. The sealed local plan is in validation folder
`2026-11-10-native-length-position-v1`, SHA-256:
`d87cf993b08086a2d9f0f322bdd8a870fbbaaa8c533a15fb09e09a7954876a46`.

The [phase runner](length_position_run.py) carries optimizer state between
single-step calls so the batch actually changes each update. It retains all
raw states/responses for transport and verification, not checkpoint selection.
The existing example's single `fit` call repeats one batch; using it unchanged
for 256 steps would **not** execute this varied-placement protocol. The sealed
plan retains its original preparation status; separate completion receipts
record actual execution.

No GPU, rebuild, wheel, push or publication is included. Execution must preserve
the fixed 600-second training / 240-second scoring phase bounds, 4 GiB disk
floor and first-error stop. No numerical retries or post-result gate changes.

## Observed Results

Primary filler-58 evaluation, untrained pair combinations over six cells:

| Attention | Training layout | Correct / 576 | Mean gold NLL |
| --- | --- | ---: | ---: |
| Ordinary | Fixed | 285 | 0.810955 |
| Ordinary | Varied | 289 | 0.810875 |
| Lag 1 | Fixed | 347 | 2.433060 |
| Lag 1 | Varied | **460** | **1.150424** |

Lag-1 varied training improves by 113/576 answers (19.62 percentage points)
and lowers NLL by 1.282636 versus lag-1 fixed. This passes the predefined
relative efficacy gate. The descriptive accuracy interaction is +109 answers
after subtracting the ordinary route's +4 gain. This is one initialization,
not a statistical estimate across seeds or a pretrained-LM result.

**NLL remains worse than ordinary Attention**, despite higher accuracy. The
result does not establish a uniformly better predictive distribution.

All primary untrained-pair cells, correct / 96:

| Added fillers | Placement | Ordinary fixed | Ordinary varied | Lag-1 fixed | Lag-1 varied |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | After BOS | 46 | 47 | 60 | 76 |
| 1 | Between facts | 46 | 51 | 51 | 62 |
| 1 | Before query | 49 | 45 | 55 | 82 |
| 3 | After BOS | 48 | 48 | 66 | 72 |
| 3 | Between facts | 47 | 48 | 48 | 72 |
| 3 | Before query | 49 | 50 | 67 | 96 |

The competence threshold is 87/96 in **every** cell. Five cells fail; the
96/96 cell cannot rescue them. Every cell improves versus lag-1 fixed in both
accuracy and NLL, but substantial layout sensitivity remains.

Additional predefined views, without replacing the primary:

| Filler / pair group | Ordinary fixed | Ordinary varied | Lag-1 fixed | Lag-1 varied |
| --- | ---: | ---: | ---: | ---: |
| 58 / trained pairs, correct / 192 | 99 | 97 | 118 | 155 |
| 59 / trained pairs, correct / 192 | 98 | 92 | 121 | 149 |
| 59 / untrained pairs, correct / 576 | 290 | 291 | 348 | 450 |
| 59 / untrained pairs, mean NLL | 0.812379 | 0.812087 | 2.291738 | 1.237688 |

Even trained pairs remain imperfect under the new layouts. This is not solely
an unseen-pair problem. Filler identity changes behavior; neither filler is
assumed semantically neutral. Further diagnosis should use the saved logits
before another intervention, preserving this consumed evaluation boundary.

## Execution Evidence

- Nine model-free protocol/runner tests pass, including state carry, changing
  batches, first-error stop, no overwrite, checkpoint owners/clocks, full-vocab
  scoring, ties, nonfinite outputs and cache mismatches.
- All 1,024 main updates completed once. The prescribed 128-step continuation
  matches every saved optimizer-state digest, serving ID and update report.
- All 9,216 origin/final question rows have bitwise equal ordinary/cache logits.
- Original inputs, executable and source inventories remained unchanged through
  training, scoring and model-free readback. No rebuild or wheel import.
- CPU phase times were 48.65s training including resume and 3.26s scoring.
  These include process/file overhead and are not throughput benchmarks.

Source at execution: local commit `3ce5ad7`. The saved executable is the same
one checked in [RAGGED-QKV.md](RAGGED-QKV.md). Exact raw calls, all checkpoints,
per-cell metrics, source inventory and receipts remain in the local validation
folder named above. No push, CI, GPU or publication occurred.

| Artifact | SHA-256 |
| --- | --- |
| Runner seal | `00fed18b9de4af52e584b8782f99070a7814cc756cec7a700c0a3be779b2450a` |
| Training | `26de6093e228893680f816e07e7fc4630d3dbf133bd33bffc76f30d78e79f9ad` |
| Scoring | `c3230e455c4e31e6d255545410c0dd912df32d827deecb5eb0b10ece5b5b5b10` |
| Model-free audit | `5fa1be8365b3bbd32a3f2ea053050b592b39768b8dde604cb38a567d462d050b` |

The runner phases are `seal PLAN PLAN_SHA`, `train PLAN SEAL_SHA`,
`score PLAN SEAL_SHA TRAIN_SHA`, and
`audit PLAN SEAL_SHA TRAIN_SHA SCORE_SHA`. Each output is created exclusively;
do not rerun a consumed phase into the same directory or select a successful
retry. Python orchestrates calls and reads outputs; Rust owns all model,
gradient and optimizer semantics.
