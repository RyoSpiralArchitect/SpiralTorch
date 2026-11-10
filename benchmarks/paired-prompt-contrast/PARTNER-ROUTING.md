# Does learning Q/K help with a new partner?

This is a **negative transfer result**, not a throughput or pretrained-language benchmark.
The first attention block really changed: four native gradient checks agreed with
finite differences on physically edited, re-imported weights. But neither changing
partner coverage nor training Q/K instead of the output projection made the decoder
choose the correct answer more often on unused entity pairs.

## What was held fixed

The base is a 2-block, width-32, 4-head **synthetic** decoder. The two starting
geometries share the same frozen base and LoRA; they are not independent seeds.
Eight token IDs take new entity roles in this task, not new vocabulary entries.
Each update contains four edges, both fact orders, both assignments and both questions
(32 prompts). A matching gives each identity one partner; alternating the two colours
of a cycle gives it two. A third, disjoint matching supplies four check edges.
Those check prompts are now consumed. The previous task's reserved pairs were not opened.

A and Q comparisons change only cue placement and only queried identity respectively.
They share the same positive/negative multisets; they are not two independent datasets.
Aligned supervision repeats each pair. The broken control toggles the queried identity
in both members on the second presentation: exactly the same input frequencies, no
consistent query-to-gold association. The optimizer uses the mean paired logistic loss,
without the answer labels at deployment.

Both routes train 64 free coordinates. Output learns two right factors against fixed
orthogonal, zero-mean left factors; routing uses one of those factors in Q and the
other in K. Their lift into physical weight space is an isometry. No V, bias,
embedding, LoRA or geometry parameters are updated. All arms take the same three
native dense pullbacks per prompt (including the common compute control).
The 16 arms each took eight normalized steps of length 0.08; **128/128** were
nonzero. This matches path lengths, not endpoint distances, relative parameter
changes or the number of independent observations. This restriction on the learned
subspace matters; a negative result does not rule out learning attention.

## Check results

For each of the four partners we first average the two orders, then form two
correct-versus-incorrect margins for each of A and Q. Their mean, S, measures
the component shared across both ways to swap the answer. **An increase in S
alone is not enough:** the four margins on an edge must *each* improve by
more than 5e-6.

| Starting geometry | Check two-way correct, hold / cycle O / cycle QK | S, hold / cycle O / cycle QK | Cycle QK edges with all 4 gains vs hold |
| --- | --- | --- | --- |
| Initial | 16 / 16 / 16 of 32 | -0.000354 / -0.000907 / -0.000455 | 0/4 |
| Subset-rank | 16 / 16 / 16 of 32 | -0.000358 / -0.000878 / -0.000481 | 0/4 |

Cycle QK also improved all four margins on **0/4** edges against matching QK,
cycle output, or broken cycle QK at either start. The QK-minus-output difference in S
is positive, but both are below hold. The cycle-minus-matching difference of that
difference is only +0.0000208 / +0.0000153. Neither preregistered gate passed.
The output route's full-vocabulary top-1 increased from 1/32 to 11/32 or 12/32;
its broken control also reached 11/32 or 12/32. That is not evidence of better binding.

One lead survives the endpoint ablation. Installing **only the learned K part** from
the aligned cycle improved S versus hold on 4/4 edges in each state (mean gains
+0.000121 and +0.000133). It still improved all four margins on 0/4 and
remained at 16/32 choices. This is not a separately trained K-only arm, a direct
attention-probability measurement, or a justification to select this checkpoint.
A prospective K-only vs Q-only comparison with a fresh partner roster would
test that lead; do not call these same four pairs unseen again.

## Reproduction and records

Use Python 3.12 and the sealed prior query-swap protocol plus its already installed
CPU development wheel. The [study](partner_routing_study.py) pins the predecessor,
source, wheel inventory, data and gate **before** fitting. In a fresh, private output
parent, run its phases as `plan PROTOCOL PRIOR_PROTOCOL PRIOR_SHA`,
`train PROTOCOL PLAN_SHA`, and `score PROTOCOL PLAN_SHA TRAIN_SHA`; each prints the
next SHA. Use `-I -B`, `SPIRALTORCH_BACKEND=cpu` and a 240-second external wall limit
per physical phase. The native need not be rebuilt. Run the [offline audit](partner_routing_audit.py)
as `PROTOCOL PLAN_SHA TRAIN_SHA SCORE_SHA AUDIT_JSON` using standard Python only.
It reconstructs all 1,984 full-vocabulary rows' scores and all 128 step lineages. No GPU,
GPT-2, Torch/NumPy, network measurement or public release was part of this run.

| Artifact | SHA-256 |
| --- | --- |
| Prior protocol | `980cf3f219b4834e40c1c3e224600471a5714f11a5df2d1db6ce005b32d9e538` |
| CPU wheel | `cc28c0711c4c0d0c500a62939e16acc51e67f3397c4d36ffd270f321e4054d09` |
| New frozen protocol | `68d373c90c2e26f20fff218ef647a3343a9bf177123816c558f995083eed5f97` |
| Training endpoints and journal inventory | `94e4b1189747e9f098ab1b3a6fe8f7c04781a5edea99629d3b37d3789a0d901c` |
| Ordinary scoring and raw row inventory | `fae7a9705dd5b394483d66387be8844af851fd90cdbd08423d7a6af6b2969572` |
| Offline reconstruction | `3149399d253c5d59340594585d727b76ff5cb4b8b8926448304cf54a68c218e4` |

Local research record: `2026-11-04-native-partner-qk-factorial-v1`. The immutable
checkpoints, first attempts, complete raw logits, and command receipts are kept
outside Git. The procedure suite passed 139 tests; that validates the harness, not transfer.
