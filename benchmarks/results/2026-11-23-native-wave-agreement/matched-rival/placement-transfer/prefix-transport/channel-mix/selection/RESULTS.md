# Train-only choice moved decisions, but did not teach the number contrast

The selector chose its routes before seeing the two-modifier prompts. It passed neither
[preregistered](PROTOCOL.md) transfer criterion: **0/3 bases**. Raw head-teacher correct
decisions fell from 141 to 123 out of 288 repeated judgments (32 prompts x 3 bases x
3 training orders). There were 9 improvements and 27 regressions, all on base 563.
They were the **same three improvements and nine regressions** in each order, not 36
independent prompts. No raw head ties. Attractor-teacher correct decisions went from
141 to 144 (three improvements). The fitted discrete choice is not a new default.

| Policy | Head gain h | Teacher selectivity S | Raw head correct | Wrong to right | Right to wrong |
| --- | ---: | ---: | ---: | ---: | ---: |
| Origin | 0 | 0 | 141 | - | - |
| Selected using training only | -0.001794 | +0.002224 | 123 | 9 | 27 |
| Blind, data-independent choice | -0.001564 | +0.001721 | 129 | 5 | 17 |
| Fixed identity | -0.001380 | +0.001208 | 132 | 3 | 12 |
| Fixed Cayley | -0.001940 | +0.002286 | 123 | 9 | 27 |
| Fixed inverse | +0.000885 | +0.000007 | 144 | 3 | 0 |

Averages weight all 18 order/template/base cells equally. The inverse's three
improvements do not replace the primary result. The trained choice was stable across
orders but differed by base and teacher: 557 = Cayley/identity, 563 = Cayley/inverse,
569 = inverse/identity (head/attractor). All three candidates were trained in the earlier
study; this study fitted only which candidate to use. No gradient or weight update.

## Why flips are not agreement

In each of the 18 cells we compared pairs differing only in head number while holding
the nearby noun's number fixed. With the selected route, **none of the 144 orderings
changed**: 48 were correct before and after (no ties). Bases 557 and 563 had the wrong
ordering on every pair; base 569 had the right ordering on every pair. Attractor-teacher
orderings likewise stayed at 48/144. Subtracting the preregistered training-origin
verb preference gave 138 to 141 centered head decisions; that is a different endpoint
from correct raw decisions. [All per-order measures and controls](orders.tsv).

For example, in the first head-modified frame on base 563, order 17, only the head
number differs (fourth token); the alternative noun and remaining context are identical.
D = logit[7] - logit[8]. The two prompts moved across zero **together**:

| Prompt | Correct verb | D at origin | D after head training | Change in correctness |
| --- | ---: | ---: | ---: | --- |
| 3 4 6 **1** 9 5 1 10 | 7 | +0.005615 | -0.069453 | correct to wrong |
| 3 4 6 **2** 9 5 1 10 | 8 | +0.030346 | -0.053930 | wrong to correct |

**Post hoc, no new calls.** For each endpoint we estimated its mean change in D on
the 32 *training* inputs. Removing that constant reproduced all 36 full head flips
unchanged. Applying just the training shift reproduced only 3/36. On base 563 the
training shift ranged from -0.00396 to -0.00372 across orders, while the mean transfer
shifts ranged from -0.03663 to -0.03630 (head-modified) and -0.02224 to -0.02208
(attractor-modified). A single training-set intercept was not the explanation.

As a retrospective decomposition, compute the mean shift across the four number
assignments **within each transfer frame**. The mean-only ablation reproduced 21 of
36 full head flips and introduced 3 others. Removing those frame means left 9 flips,
all regressions. Counts do not add: crossing the zero threshold is nonlinear.
The transfer-frame means account for roughly 82% and 50% of squared margin change
on base 563 on the two templates respectively. They use transfer siblings and are
**not** an available inference-time correction or proof of a unique causal mechanism.

**Next test:** train the change in verb preference *between paired head-number forms*,
with a matched objective penalizing drift shared by both forms. Balance the head and
attractor assignments, keep the delivered RMS fixed, and compare to an equally costly
ordinary paired update on newly frozen prompts. First require correct within-pair
ordering; report raw decisions separately. Do not choose the shift or route from
these transfer results. No such fit or real GPT-2 run is claimed here.

## Run and replay

The formal run used the previously pinned CPU wheel. At nine zero-gate origins
and 54 archived endpoints it made 2,016 training-input and 2,016 transfer forwards.
All 18 Rust selections matched an independent exact-rational calculation before
transfer was opened. All 288 origin checks matched the earlier native training
verb logits byte for byte. Separate rehearsal: 36 + 72 forwards. Both completed
on the first attempt. The new 32 prompts are consumed. No GPU, torch, NumPy or publication.

The Rust selector module passed four unit tests and `cargo check --offline --locked
--release -p st-nn --lib`. A small binary compiled from that same Rust module handled
route selection; the existing Python wheel handled native model forwards. The selector
is not claimed as a built-in of that wheel. Both CPython 3.8 and 3.12 reproduced the
primary JSON and TSV, and the separate shift JSON, byte for byte. The source executing
the primary comparison was `4b8c947b5826feebb3bd20fc060097bd941a6e2f`.

| SHA-256 identity | Value |
| --- | --- |
| Input manifest | `c177b60f7f173faf0ef4b517199ebabd06afa5de983d33ef956eea50ebba7c2a` |
| Training-only selector seal | `c00b5cb7d2a69efe2cfeb27d167e6b673a93804a7e8b44aaa6337646d38c8f69` |
| Training logits | `b11da78e9e0e358747f69438dca83f8d5f271ff1e0adb77468ffb11368be6363` |
| Transfer logits | `6015cb49ff422870920bc48b03f0c52604e370a434894aa1ccc2ece09970640b` |
| Primary result | `f24568e5b9735cd518dd2c5ff6aa4a5fee3b7cfa9609e6b046440c74736bef49` |
| Full-order table | `f22db7bdf6caf25222c71541c4ceaad8d8cba120da6a630150e74d70390552b8` |
| Exploratory decomposition | `045b226dbf3cd67edcd9abd691120b7a789bd6beb7fe88d8f9a1f6b57a96cca8` |
