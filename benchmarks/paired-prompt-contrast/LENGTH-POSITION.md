# Matched Length/Position Training: Prepared Protocol

Status: model-free preparation complete; **no training or scoring performed**.
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

Next implementation work is the phase runner: carry optimizer state between
the saved executable's calls so batches can change each update, retain raw
responses and first failures, and verify the prescribed resume before scoring.
The existing example's single `fit` call repeats one batch; using it unchanged
for 256 steps would **not** execute this varied-placement protocol.

No GPU, rebuild, wheel, push or publication is included. Execution must preserve
the fixed 600-second training / 240-second scoring phase bounds, 4 GiB disk
floor and first-error stop. No numerical retries or post-result gate changes.
