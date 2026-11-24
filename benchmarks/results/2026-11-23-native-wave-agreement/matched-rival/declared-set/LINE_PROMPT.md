# The helpful prompt was not a helpful bank position

On the same three generated complete-answer tasks, the joint LoRA/WaveGate
update advanced **only** when its first midpoint used the normal from prompt
0. That was the source of the loss, not the probe's largest KL (prompt 2).
Making the selection explicit removed the accidental dependence on bank order.
It did **not** make a geometry-only step feasible in this bounded search.

[Protocol](LINE_PROMPT_PROTOCOL.md) fixed at `ce174e4`. Six permutations of
the same three constraints; same two-epoch loss order, initial weights, target
and cap as the [prior run](LINE_MIDPOINT.md). Counts below are how many bank
orders finished all six steps, **not** independent seeds or success rates.

| Trainable parameters | Largest change | Cyclic successor | Source of loss |
| --- | ---: | ---: | ---: |
| LoRA | 6/6 | 6/6 | 6/6 |
| WaveGate only | 0/6 | 0/6 | 0/6 |
| Both | 0/6 | 3/6 | **6/6** |

Every successful joint trajectory was one corrected update followed by five
unchanged raw updates, with identical final weights, logits and individual
losses. The three successful successor orders (012, 120, 201) chose prompt 0;
the reverse orientations chose 1 and refused. There were no tied probe peaks.
LoRA alone never needed the correction. WaveGate alone stopped atomically
on the first update even when all three semantic normals were tried in
separate cells. Among its raw and midpoint trials that respected the peak cap,
the highest measured mean was **0.71028T**, short of the required 0.98T.
This is not a proof that no acceptable direction exists.

Starting instead with loss task 1 or 2 gave a different answer: all twelve
joint first attempts accepted their raw ray, so none tests a preference for
the current prompt. The twelve geometry-only first attempts refused, covering
all three normals for each task. LoRA accepted all twelve raw attempts.
Changing the first task changes its objective too. There is **one** demonstrated
joint rescue origin, not evidence that the loss prompt is universally special.
For the remaining geometry failure, a more informative follow-up is to
compare the three prompts' directional sensitivities on the same weights
before trying larger steps or changing the KL envelope.

The [directional sensitivity comparison](LINE_SENSITIVITY.md) takes that
next step with matched tensor and coordinate probes, without relaxing the cap.

## Implementation and verification

The opt-in `loss_prompt` selector resolves a sealed complete-line prefix,
not a supplied index. Rust, Python and WASM expose the same fingerprint.
The raw-first policy still accepts a feasible raw step unchanged. Probe
measurement still visits every anchor, but no longer retains the full bank
of candidate vocabulary rows: only the selected row, plus a wraparound row
for the successor rule. Origin reference distributions are unchanged. This
is a storage bound on probe rows, not a speed benchmark.

All 54 main trajectories and 36 alternate-first-task cells were run once
through the CPU Python binding: **225 primary decisions, 186 updates and
39 refusals**, with 48 measured directions. Python checked KL from actual
before/after logits, the probe peak and selected prompt, training cursor and
owner IDs; an unmatched update had to leave the checkpoint byte-identical.
All 21 canonical legacy decisions matched their archived receipts exactly.
Node/WASM repeated all 90 cells, checked the same decisions, semantic
choices, trial distances and endpoint logits (float tolerances, not
cross-platform bitwise claims), and independently recomputed each accepted
KL. No pretrained model, held-out data, GPU, remote job or publication.

The [compact readback](line_prompt_results.json) contains per-policy outcomes,
distinct trajectories and both raw-file hashes. To reproduce with the generated
[prior journal](LINE_MIDPOINT.md), a local CPU wheel and web-target WASM glue,
use fresh output paths:

```sh
python -IB bindings/st-wasm/tests/native_line_prompt_fixture.py midpoint/raw.jsonl results.json
node bindings/st-wasm/tests/native_line_prompt_node.mjs /path/to/pkg/spiraltorch_wasm.js results.json
python -ISB benchmarks/results/2026-11-23-native-wave-agreement/matched-rival/declared-set/line_prompt_summary.py results.json results.jsonl midpoint/raw.jsonl
```

[Binding contract and plan syntax](../../../../../bindings/st-wasm/tests/native_causal_lm.md).
