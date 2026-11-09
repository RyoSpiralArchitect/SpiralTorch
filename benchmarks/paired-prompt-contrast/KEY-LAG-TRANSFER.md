# Key-Lag Transfer And Matched Distance

A read-only follow-up to [native key-lag capacity](CAUSAL-KEY-LAG.md).
**Both preregistered success gates failed.** There is a relative recombination
gain, but the earlier 32/32 fit result is not robust generalization.

## Frozen Scope

Use nine fixed marker-last models: origin, aligned final-256, and reflected
final-256 for lag 0, 1 and 2. Reuse the saved Rust executable and immutable
checkpoints. No learning, interpolation, route selection or new wheel.

Four fit pairs yield 32 views. All twelve other pairings of the same four
left/four right identifiers yield 96 views, including both presentation
orders, assignments and queries. These are untrained by the current QKV
endpoints, **not globally fresh research holdouts**: earlier studies already
consumed some pairs. Their old scores and other held decks are not read.

Seven input variants:

- Original length-nine template.
- For each filler ID 58 or 59, length thirteen with entity/value distance
  one, two or three. Insert two copies of the same filler per fact.

For a given filler, all three distances have identical token multisets,
value positions, query/answer-marker positions and lengths. Only entity/filler
positions change. Fillers are not assumed semantically neutral. Even the
distance-one padded condition is a distribution shift from training.

The primary competence gate is >=87/96 strict unrestricted vocabulary top-1
and >=22/24 ranked assignment/query quartets on original-length untrained
pairs, with strictly more correct answers than ordinary aligned, lag-2 aligned,
own origin and own reflected controls. The distance double-match gate also
requires each matching padded condition to pass competence before attributing
its difference from nonmatching distances to distance sensitivity.

Run: **9 models x 7 variants x 128 questions = 8,064 scored pairs**. Each has
ordinary and prefix-cache outputs. Training updates: zero. Endpoints and input
roster were sealed before the first new model call.

## Original-Length Untrained Pairs

| Lag | Endpoint | Full-vocab correct / 96 | Ranked quartets / 24 | Mean gold NLL |
| --- | --- | ---: | ---: | ---: |
| 0 | Origin | 3 | 0 | 4.357002 |
| 0 | Aligned | 47 | 0 | 0.809586 |
| 0 | Reflected | 48 | 0 | 0.809593 |
| 1 | Origin | 3 | 0 | 4.367193 |
| 1 | Aligned | **81** | **18** | 1.053386 |
| 1 | Reflected | 46 | 0 | 0.817310 |
| 2 | Origin | 3 | 0 | 4.368982 |
| 2 | Aligned | 53 | 9 | 1.067767 |
| 2 | Reflected | 48 | 0 | 0.813549 |

Lag 1 gains 34 correct answers over ordinary training and 28 over lag 2,
but misses both absolute competence thresholds. Its mean NLL is **worse than
ordinary**, so the accuracy gain is not a general improvement in the
probability distribution.

A post-hoc decomposition of the same stored logits, with no new model calls:
lag 1's 81 correct cases have mean gold NLL .222162; its 15 errors have mean
5.541993 and maximum 8.912325. Five errors choose vocabulary item 9, outside the
two answers. This is descriptive error analysis, not an additional passed gate
or a tuning set for a claimed confirmatory result.

## Padded Untrained Pairs

All numbers are full-vocabulary correct / 96; columns change only entity/filler
placement within the same length and fixed value/query positions.

| Filler | Learned lag | Distance 1 | Distance 2 | Distance 3 |
| --- | ---: | ---: | ---: | ---: |
| 58 | 0 | 46 | 45 | 47 |
| 58 | 1 | **38** | 31 | 0 |
| 58 | 2 | 0 | **31** | 0 |
| 59 | 0 | 53 | 50 | 51 |
| 59 | 1 | **57** | 45 | 49 |
| 59 | 2 | 49 | **50** | 56 |

Matching conditions are bolded, not declared successful. All four matching
lag/filler conditions miss competence. Therefore the distance double-match
gate fails and each distance attribution is marked **inconclusive because its
matching condition already fails**. Even lag 1 at distance one scores only
38/96 or 57/96. Training-pair padded-adjacent scores also fall from 32/32 to
12/32 or 24/32. A novel-pair-only explanation cannot account for all the damage.

The inference from these observations is limited: fixed-location fitting did
not provide robustness to padding/position changes. This run does not identify
whether absolute positions, the chosen filler embeddings, their interactions
or other internal paths cause the drop. It does not show that a different
fixed lag or more lag branches would fix it.

## Implementation And Checks

[causal_key_lag_transfer.py](causal_key_lag_transfer.py) freezes the roster,
calls only the existing Rust scoring operation, and reads back saved logits
without a model. Answers are not included in model requests.

The new measurement path derives factual-prefix length from the actual input
(`len(prompt)-2`); the earlier capacity helper is deliberately left unchanged
at its original seven-token prefix. Reusing that helper for padded inputs
would merge different factual prefixes. A targeted regression covers this.

170 Python procedure tests pass. All 8,064 ordinary/cache output pairs agree
bitwise. The nine original-fit baselines also reproduce their saved bytes.
Each scoring reply retains the same serving identity and step count; optimizer
checkpoints are hash-checked before and after scoring. The frozen executable
is unchanged. No Rust recompilation, GPU, new pretrained evaluation or
publication occurred.

The first guarded procedure-test launch stopped **before starting a process** because
free disk space was below the unchanged 4 GiB floor. After explicit approval,
only two idle incremental cache directories were removed, yielding .954460
GiB of actual free space. The stop and deletion receipts are retained. No
model failure was retried and no metric or gate was loosened.

## Reproduction And Evidence

Run the phases `freeze`, `run`, `audit` in order with
`python3 -I -B -S benchmarks/paired-prompt-contrast/causal_key_lag_transfer.py`.
Arguments after the phase are:

- `freeze OUTPUT_PROTOCOL PARENT PARENT_PLAN_SHA PARENT_TRAIN_SHA PARENT_SCORE_SHA DRIVER`
- `run OUTPUT_PROTOCOL PLAN_SHA DRIVER`
- `audit OUTPUT_PROTOCOL PLAN_SHA MEASUREMENT_SHA`

Use the pinned parent from the capacity report, the retained CPU executable,
exclusive output folders, `SPIRALTORCH_BACKEND=cpu` and a <=240-second
one-attempt physical window. Stop on first execution error. All these
question views are now consumed, not untouched future holdouts.

| Artifact | SHA-256 |
| --- | --- |
| Frozen plan | `8d5d864af5822cfae83e17ca810627a99216d45880db653018ecb029c0d7a45e` |
| Measurements | `a2e41b1b86c4644d7383b04f93fd0d448aba6478c75c6641177ff9eea5e40686` |
| Model-free audit | `32c5c412aaf38847b3f75c87b51a8f1aa70b076f0018cb0f90af158416cd3f23` |

Local originals: `2026-11-10-native-key-lag-transfer-v1` under the validation
folder. Source was fixed at local commit `4608e28` before measurement.
This result suggests training-only position/layout variation and protection
against severe wrong-answer losses as next development directions. Future
success must be checked on separately frozen data; do not relabel this
consumed deck as a fresh test after tuning.
