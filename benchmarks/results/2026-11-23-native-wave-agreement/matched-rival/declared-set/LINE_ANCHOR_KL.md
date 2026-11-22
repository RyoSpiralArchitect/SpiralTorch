# Equal movement in predictions, not in adapter weights

What happens when LoRA and WaveGate must change the next-token distributions
by the same amount? On the same tiny artificial decoder and three *training*
prompts as the [parameter-distance comparison](LINE_OWNER_DISTANCE.md), all 18
updates reached the prescribed mean KL. Geometry alone had the lowest final
menu loss here; joint had the lowest set loss, and LoRA the lowest subset loss.
These are three different examples/objectives, not accuracy on a test set.

| Matched owner | Steps | Mean KL / target (range) | Parameter path | Origin-to-end mean KL | Set NLL | Menu loss | Subset NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Before learning | 0 | n/a | 0 | 0 | 4.568401 | 2.570731 | 0.114927 |
| LoRA | 6 | 98.01-99.94% | 0.024780 | 0.00006017 | 4.567679 | 2.560755 | **0.114157** |
| Geometry | 6 | 98.65-99.04% | 0.112061 | 0.00006020 | 4.568359 | **2.551310** | 0.114696 |
| Joint | 6 | 98.28-99.78% | 0.032471 | 0.00006495 | **4.567310** | 2.556326 | 0.114238 |

At fixed raw distance (1/256 per step) the ordering on the menu example had
been joint, LoRA, geometry. The replay reproduced all 18 control losses and
the three endpoints. On that control, the observed per-step mean KL was
only 5.7-13.2% of the new target for geometry, versus 34.3-216.8% for LoRA.
At matched mean KL geometry traversed **4.52 times** the raw parameter path
of LoRA. Raw LoRA factor distance depends on the parameterization; this is not
a 4.52x learning-efficiency or speed claim.

## What is matched

The native line learner computes one full-answer loss gradient per scheduled
example. It holds that direction and origin fixed while varying its length.
The acceptance test is the arithmetic mean across three distinct fixed prompts
of forward KL, `KL(p_before || p_proposal)`, in nats over the full vocabulary at
each prompt's last position. Each of six updates must spend 98% to 100% of
**1/65536 nat**. The ceiling (1/8 parameter L2) is tried first; a bracketed
search tries at most 19 further points. No too-small fallback, retry of a
failed trajectory, or checkpoint advance on refusal. Same initial nonzero LoRA,
WaveGate, training examples, order, two epochs and tile size (11) as before.

This is **not** a per-prompt upper bound. The largest single-prompt KL reached
2.90 times the mean target for geometry (2.30 for joint, 1.69 for LoRA).
It also does not bound how all answer continuations or unseen inputs change:
only the three next-token anchor rows are constrained. KL from the common
origin to the endpoint is measured separately, not summed from update KLs.
A next control would cap individual-anchor KL alongside the mean and check
whether the menu ordering survives.

## Cost and evidence

The Rust searches measured 73, 57 and 72 trial models for LoRA, geometry
and joint respectively. Including the origins that is **660 single-prompt
forward readouts for 18 accepted updates**, in addition to the underlying
pullbacks and the fixture's independent observations. This is a calibration
tool, not a throughput benchmark. Full pretraining, GPT-2 prompts, held-out
scoring, Torch, NumPy and GPU are not involved.

The [machine result](line_anchor_kl_results.json) includes every per-step
mean and individual KL, pre-update loss, movement totals and the digest of the
native raw trial log. Python independently recomputes KL from the actual
pre/post logits. Node and real Chrome do likewise for all 18 matched updates,
and both continue 12 suffix updates: Node from native-Python checkpoints,
Chrome from its own. Fresh Python reaches the same final pins from its three cuts.
The browser table displays the live results, not this report.

With an installed development CPU wheel and **new** output paths, from the
worktree root:

```sh
python -IB bindings/st-wasm/tests/native_line_kl_fixture.py run raw.json cuts.json
python -IB bindings/st-wasm/tests/native_line_kl_fixture.py resume cuts.json
ST_SYNTHETIC_LINE_KL_BRIDGE="$PWD/cuts.json" node bindings/st-wasm/tests/native_causal_lm_node.mjs /path/to/pkg/spiraltorch_wasm.js > node.json
```

Use the [browser demo](../../../../../bindings/st-wasm/tests/native_causal_lm.md)
and capture its rendered DOM after the page passes. The [stdlib-only reducer](line_anchor_kl_summary.py)
takes `raw.json node.json rendered.html NEW_OUTPUT` and does not run a model.
