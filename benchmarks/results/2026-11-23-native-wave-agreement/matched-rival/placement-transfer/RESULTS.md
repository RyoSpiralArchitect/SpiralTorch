# Which site learned the head cue?

Only **WaveGate on the second block's MLP output** (`transformer.h.1.mlp`)
cleared the [frozen placement and matched-affine criteria](PROTOCOL.md).
All three orders improved the complete-answer head margin on all four unseen
noun pairs. **No choice flipped:** every order still got the same 8/16
two-choice answers right. This is a conditional probability change, not an
accuracy gain or evidence about English or pretrained GPT-2.

## What changed

For that site on the held-out split, the head-trained margin gain was
0.001052..0.001182 nats; its excess over attractor-trained was
0.001750..0.002033. The two-role interaction was 0.005533..0.006551.
Full-vocabulary head-gold NLL fell 0.001068..0.001226 nats/token. The weakest
head-margin pass was only 0.000052 nats above the declared 0.001 floor.
Relative to affine at the same site, both the excess in head-margin gain and
the excess in interaction exceeded 0.001 in *each* order.

- The **answer token** contributed only 0.000078..0.000088 nats to the signed
  head-margin gain. **Stop after the particular answer** contributed
  0.000973..0.001094 nats, or 92.55..92.57% of that gain. An unconditional
  preference for stopping would cancel in this balanced contrast.
- A different decomposition: 81.7..82.4% of the full-vocabulary gold NLL
  gain came from assigning more probability to the two complete lines
  *together*, rather than improving their within-menu odds.
- The head-trained paths all travelled 0.25 in parameter L2 (rounded), but
  the net displacements at the winning site were 0.0425..0.0464. The
  constraint matches each **parameter step**, not endpoint distance, gradient
  angle, forward KL or initial predictions.
- First-block attention illustrates the value of the control: its median
  Wave interaction was bigger than the qualifying site's, but its head
  margin, gold NLL gain and across-pair transfer missed the joint gate.
  An interaction by itself does not mean the requested rule was learned.

### All six insertion sites

Medians across the three training orders on the unseen lexical pairs.
For legibility values here are **1,000 times** the changes in nats
(gold gain: nats/token). H is the head-trained shift toward the correct
complete line; S is the difference in which cue the two training rules
favoured. Gate decisions use unrounded per-order values, not these medians.

| Block / output | Wave H | Wave S | Wave gold gain | Affine H | Affine S | Qualified |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| First / attention | 0.113 | 8.715 | 0.155 | -0.0004 | 0.0030 | Neither |
| First / MLP | -0.0019 | 0.0044 | -0.0051 | 0.0167 | 0.0005 | Neither |
| First / residual block | -0.0005 | 0.0044 | 0.0004 | -0.0080 | 0.0046 | Neither |
| Second / attention | -0.00003 | 0.0024 | -0.0031 | 0.0619 | 0.0031 | Neither |
| Second / MLP | **1.052** | **5.835** | **1.068** | -0.0052 | 0.0051 | Wave only |
| Second / residual block | -0.264 | 0.0003 | -0.914 | -0.0037 | 0.0008 | Neither |

The moving LoRA-only reference (12 scalars, no geometry) had median H 0.076,
S 4.399 and gold gain -0.116 in these units; this is not a six-scalar control.
The [per-order table](per-order.tsv) includes all 39 comparisons, the matched
teacher contrast B, content/stop split, NLL decomposition, pair counts and
realized distances. A single tiny generated base, four held-out noun pairs
and three orders are not independent model replications. The head-trained
gain on the *training* split at the qualifying site was also much smaller
(0.000098..0.000105 nats). Do not select this as a default site from this run.

## Execution and readback

Protocol commit `193997c`; runner and analysis frozen at `39dcf30`. One CPU
attempt, no retry: 4,992 Rust updates, 4,368 label-free complete-menu scores,
273 uncached full-vocabulary sentinel forwards, 78 final owner checkpoints.
Max relative measured step-length error 4.121e-5 (limit 1e-4); max direct
token-cost mismatch 4.45e-16 nats (limit 3e-6). No external ML stack, real
weights, old English evaluation, GPU, generation or publication. A disjoint
rehearsal (24 updates, 72 scores) is not in these totals. The first unit-test
expectation for a manufactured coverage gain mistakenly counted per-line
instead of per-token nats; fixed before rehearsal, no model outcome observed.

Local evidence set: `native-role-placement.pOPLCS` under the SpiralTorch
validation folder. `formal/spec.json` binds source and fixture bytes;
`formal/completed.json` binds the raw numeric scores, step-by-step owner
journal and final checkpoints. Independent regrouping by head or attractor
number reproduces all 39 test-contrast rows within 1e-12 and the same gates.
To inspect archived results without executing the model, use new output paths:

```sh
python3 -ISB test_study.py
python3 -ISB analyze.py "$EVIDENCE/formal" "$OUT/readback.json"
python3 -ISB pair_readback.py "$EVIDENCE/formal" "$OUT/readback.json" "$OUT/per-order.tsv"
cmp "$OUT/per-order.tsv" per-order.tsv
```

| Artifact | SHA-256 |
| --- | --- |
| Formal specification | `d7625e0a722d9500a9ff830f3728b536b87b19c65032c8c1f430025bb88cc8ef` |
| Formal raw manifest | `22f9187158cf2d1e16b5edfd6a75e66002d3800e604e488108db8a2c87c9a390` |
| Full readback, both splits | `69559801b01f2b2c9368e04894f63331e11d0901349da53af1de6819f1d37eb7` |
| Per-order table | `42bd7f369279e158465e1b714c5ee01217ad3658f164994678dcff717172abe8` |
| Local first-attempt archive (rehearsal and formal, including receipts) | `b0a259d62fd56efdd0c5014077ca34009b6ad32602b78771b8d729e342eb69a2` |

The natural next test is whether training that scores **only the answer
token**, against a matched complete-line objective, can move the first-token
decision on fresh synthetic bases. See the [separate three-base
answer-scope comparison](answer-scope/RESULTS.md); its runs and decision
criteria are not part of this placement study.
