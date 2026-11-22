# Protecting the most changed prompt

**Adding a per-prompt bound changed which owners could train at all.** On the
tiny ordinary/WaveGate decoder, only LoRA completed six updates with both
the original mean KL requirement and a maximum on each prompt. Geometry
and joint stopped on the **first** training example with unchanged weights.
The primary experiment therefore cannot rank the three owners by final loss.

[Protocol fixed before implementation or execution](LINE_ANCHOR_ENVELOPE_PROTOCOL.md)
at `75f2ddc8c57f`. Same generated decoder, nonzero LoRA, three training
examples/anchors and two epochs as the [original comparison](LINE_ANCHOR_KL.md).
T = 1/65536 nat; C = 2T. All measurements use the full next-token
distribution of each anchor. No pretrained model, evaluation text, GPU,
PyTorch or NumPy was used.

| Rule | LoRA | Geometry | Joint | What is matched on every accepted step? |
| --- | ---: | ---: | ---: | --- |
| Original mean | 6/6 | 6/6 | 6/6 | Mean KL in [0.98T,T] |
| Mean with individual cap | 6/6 | **0/6** | **0/6** | Same mean band, and each KL at most C |
| Peak only | 6/6 | 6/6 | 6/6 | Maximum KL in [0.98C,C], no mean floor |

## Why the first update stopped

The loss gradient was computed on anchor 1 (the set example, prompt
`[1,2]`); the most changed prompt was *not* that anchor. Below are the
independently measured KLs divided by T, at the **highest mean actually
observed under the cap** among each rejected search's 20 candidates:

| Owner | Anchor 1: set | Anchor 2: menu | Anchor 3: subset | Mean / T | Minimum required |
| --- | ---: | ---: | ---: | ---: | ---: |
| Geometry | 0.000008 | 1.999964 | 0.127083 | **0.709018** | 0.98 |
| Joint | 0.019318 | 0.559691 | 1.999849 | **0.859620** | 0.98 |

The corresponding uncapped first steps did enter the mean band, but reached
2.777942T on anchor 2 (geometry) and 2.304455T on anchor 3 (joint). By contrast
the original LoRA step never breached C across its six updates; the capped
LoRA run reproduced its movements and final scores. These 20-point searches
are **not** proofs about every distance on a nonlinear ray. They *do* show
why merely returning the largest tested cap-safe step would not be a matched
mean comparison. For this fixture, reshaping the gradient to distribute change
away from the limiting anchor is a more informative next control than shrinking
the scalar step again.

## A different comparison that did finish

| Peak-only owner | Mean KL / T across steps | Parameter path | Final set NLL | Final menu loss | Final subset NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| Before training | n/a | 0 | 4.568401 | 2.570731 | 0.114927 |
| LoRA | 1.134-1.275 | 0.027618 | 4.567602 | 2.559744 | **0.114061** |
| Geometry | 0.696-0.707 | 0.079956 | 4.568369 | **2.554151** | 0.114658 |
| Joint | 0.847-1.250 | 0.033051 | **4.567389** | 2.555187 | 0.114092 |

The owner ordering on each of the three training losses is the same as in
the mean-only control. This does not rescue the missing capped-mean ranking
or demonstrate generalization. A different budget is matched; all scores
are in-sample and each loss represents a different example/objective.

The [follow-up direction comparison](LINE_SECANT.md) holds the capped-mean
criterion fixed and changes the ray instead; all stopped trajectories are retained.

Rust owns the bounded search, weights and cursor. The six new trajectories
made **24 updates and 2 bounded refusals**; another 18 mean-only updates
were reference checks. The native client recomputed KL from actual pre/post
logits, and rebuilt all 40 rejected candidates from the unchanged model and
one sealed gradient per refusal. Node and Chrome independently checked these
same candidates. Their restore checks applied 16 suffix updates and reproduced
two refusals each (Node from native cuts; Chrome from its own). Fresh Python
also reproduced both refusals and the four completed final checkpoint pins.
No refusal is retried as part of the primary comparison.

The [compact result](line_anchor_envelope_results.json) records every step's
mean/T, peak/T and zero-based peak index, stopped endpoints, candidate counts
and separate rankings only where all owners completed. The [reducer](line_anchor_envelope_summary.py)
validates it from native results, the Node report and a rendered Chrome DOM.
To reproduce locally from the checkout and a development CPU wheel use **fresh**
paths; the reference can be rebuilt without access to a private archive:

```sh
mkdir prior
python -IB bindings/st-wasm/tests/native_line_kl_fixture.py run prior/raw.json prior/bridge.json
python -IB bindings/st-wasm/tests/native_line_envelope_fixture.py run prior raw.json cuts.json
python -IB bindings/st-wasm/tests/native_line_envelope_fixture.py resume cuts.json
ST_SYNTHETIC_ENVELOPE_CUTS="$PWD/cuts.json" ST_SYNTHETIC_LINE_KL_BRIDGE="$PWD/prior/bridge.json" node bindings/st-wasm/tests/native_causal_lm_node.mjs /path/to/pkg/spiraltorch_wasm.js > node.json
```

The study checks both prior input hashes before beginning. Both files were
regenerated byte-for-byte on the reference macOS CPU build; a build producing
different numbers must not silently replace these controls. See the
[API and browser demo](../../../../../bindings/st-wasm/tests/native_causal_lm.md)
for the bounded-step contract and browser setup. The receipt proves behavior
on this synthetic fixture, not a performance or release guarantee.
