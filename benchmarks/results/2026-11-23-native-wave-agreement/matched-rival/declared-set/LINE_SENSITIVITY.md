# WaveGate's strongest coordinate spends nearly all of its KL on one prompt

The [preregistered comparison](LINE_SENSITIVITY_PROTOCOL.md) probed the loss
direction restricted to each trainable tensor or to each scalar. On these
generated weights, changing the choice of **individual** parameter rescued a
joint LoRA/WaveGate update, but none of the four policies rescued WaveGate by
itself. No KL threshold or parameter ceiling changed.

| Trainable | Block, gradient size | Block, predicted envelope | Scalar, gradient size | Scalar, predicted envelope |
| --- | ---: | ---: | ---: | ---: |
| LoRA | 6 raw | 6 raw | 6 raw | 6 raw |
| WaveGate | 0 | 0 | 0 | 0 |
| Joint | 1 rescued + 5 raw | 1 rescued + 5 raw | 0 | 1 rescued + 5 raw |

The joint scalar control chose WaveGate `bias[0]` and stalled. Looking at
the **same 18 probes**, the envelope selector chose LoRA `b[4]` and the
actual first update reached 0.99704T. At block granularity both rules
completed, choosing LoRA B and A respectively. Notably the quadratic
estimate predicted only 0.82214T for B, but its actual accepted step reached
0.99308T. A prediction below the floor is not a reason to reject a measured
step. A prediction above the floor is never sufficient to accept one.

All eight alternate first-task geometry attempts refused. The selected block
was always bias, the selected scalar always its first coordinate. This is one
initial decoder and three training prompts, not twelve independent failures.
The earlier [prompt-identity rescue](LINE_PROMPT.md) was a different, joint
descent direction; this comparison does not show that geometry was necessary
for any of these rescued **first** updates.

## Geometry sensitivity

These are **measured** full-vocabulary KLs divided by target T=1/65536,
at the original parameter probe distance h=1/256. Rows of the
[full table](line_sensitivity_results.json) include all six coordinates.

| Original loss on prompt 0 | Prompt 0 | Prompt 1 | Prompt 2 |
| --- | ---: | ---: | ---: |
| Bias 0 | 0.00000121 | 0.37680 | 0.01888 |
| Bias 1 | 0.00000036 | 0.00000518 | 0.00001085 |
| Bias 2 | <0.00000001 | 0.00000378 | 0.00000277 |
| Largest across the 3 gate coordinates | <0.00000001 | <0.00000006 | <0.00000005 |

Among the *actually tested* cap-safe terminal trials on the chosen geometry
block and coordinate for task 0, the best mean was 0.70902T and 0.70896T.
The required floor is 0.98T.

A separate **post-hoc linear-model bound** asks if arbitrary mixtures of
these coordinates could work *if* their measured finite responses were
constant local Jacobian column lengths. Let b[p,j]=sqrt(KL[p,j]/T) and
express parameter displacements x in units of h, so ||x|| <= 32. For each
prompt consider a hypothetical linear map B[p] with those column lengths
and KL/T equal to ||B[p]x||^2. No assumption about angles between columns
is needed. Set j to the strongest coordinate, p to its limiting prompt,
and R[q]=32*sqrt(sum over k!=j of b[q,k]^2). Reverse triangle at the cap
gives |x[j]| <= min(32,(sqrt(2)+R[p])/b[p,j]). Hence for each q the
cap-safe KL/T is at most min(2,(b[q,j]*|x[j]|+R[q])^2), using that upper
bound for |x[j]|. For this chart the three upper bounds are **0.00049,
2, 0.20832** (rounded up); their unrounded mean is at most **0.73627T**
in KL units. The other two first tasks yield almost the same estimate.

This bounds the *surrogate*, **not the nonlinear decoder**. Finite probes,
f32 rounding, and direction-dependent curvature can change its columns.
A useful next control is a signed, two-scale probe of the dominant bias and
its nearest competitors. If the response stays concentrated, compare where
the adapter is placed rather than relaxing the learning criterion.

## Evidence and replay

There were 20 cells, 55 decisions, 39 ordinary and 3 rescued updates, and 13
unchanged-state refusals. The primary searches used 1,075 scalar trials, 92
additional parameter probes, and 3,666 anchor readouts (including reference
banks); they did not require 92 extra backwards. Python independently rebuilt
all 92 initial probes from named f32 weights and gradients, measured logits
for the three anchors, and checked the complete parameter state after each
rescue. Those checks are additional work, not part of optimizer cost. At 40
origins the raw stage also matched the saved same-state reference exactly.
Node/WASM checked the same 55 decisions, selected axes, trial distances,
accepted KL from pre/post logits and endpoint logits. The largest cross-client
probe KL difference was 3.31e-10 nats; this is not a speed comparison.

The Rust library passed 1,345 tests, the isolated Python binding passed
110 session/decoder/LoRA tests, and two explicit linear-map tests exercised
the independent bound. A fresh zero-initialized LoRA regression separates an
unprobed zero gradient from an actual zero-KL probe and checks nonadvancement.
The old midpoint WASM report, including 12 restored updates and three
restored refusals, matched its archived bytes; the native checkpoint replay
matched the rescued prefix and all suffix receipts as well.

Only generated in-sample examples were used. No pretrained decoder, held-out
selection, GPU, remote run, or publication. All four settings share the
same unchanged raw-first admission, finite-cap requirement and 20+20 bound;
block and coordinate bases intentionally have different probe counts.

With a CPU wheel and web-target WASM glue, use new output paths and the
[previous recipe](LINE_PROMPT.md) to produce the pinned journal:

```sh
python -IB bindings/st-wasm/tests/native_line_sensitivity_fixture.py prompt/results.jsonl results.json
node bindings/st-wasm/tests/native_line_sensitivity_node.mjs /path/to/pkg/spiraltorch_wasm.js results.json results.jsonl > wasm.json
python -ISB benchmarks/results/2026-11-23-native-wave-agreement/matched-rival/declared-set/line_sensitivity_summary.py results.json results.jsonl wasm.json
```

[Plan syntax, constraints and receipt fields](../../../../../bindings/st-wasm/tests/native_causal_lm.md).
