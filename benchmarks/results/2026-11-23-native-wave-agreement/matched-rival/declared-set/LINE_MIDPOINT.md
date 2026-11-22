# One corrected joint step unlocked five ordinary steps

On the same tiny generated language decoder and unchanged per-prompt KL
limits, the cyclic-successor control completed all six joint LoRA/WaveGate
updates. The first was rescued; the remaining five were **plain loss-gradient
steps**. Protecting the most changed prompt itself did not rescue joint.
Neither policy unlocked geometry alone. LoRA no longer regressed: both
policies preserved all six original updates and did not probe a second direction.

[Protocol fixed at `716b2aa`](LINE_MIDPOINT_PROTOCOL.md) before implementation
and new measurements. The three distinct training examples are also the anchor
bank, always in the same order. T = 1/65536 nat; each accepted step must have
mean next-token forward KL in [0.98T,T] and all three values <=2T.

| Trainable parameters | Raw reference | Raw first, then protect peak | Raw first, then protect its successor |
| --- | ---: | ---: | ---: |
| LoRA | 6 raw | 6 raw | 6 raw |
| WaveGate | stopped at 1 | stopped at 1 | stopped at 1 |
| Joint | stopped at 1 | stopped at 1 | **1 corrected + 5 raw** |

## What changed at the first step?

The unmodified joint ray spent at most **0.85962T** in mean KL among its 20
tested cap-safe proposals. The probe changed the third prompt the most.
Bisecting the angle toward the gradient projected off that prompt's normal
spent at most 0.70820T under the cap. The successor in this bank was the
**first prompt, also the current loss example**. Its midpoint had cosine
0.85165 to the raw gradient and reached an admissible point in 9 more
scalar trials. At that point, the actual KLs divided by T were:

| First prompt | Second | Third | Mean |
| ---: | ---: | ---: | ---: |
| 0.06451 | 1.01328 | 1.91839 | **0.99872** |

The measured loss on the training example went from **4.568401 to 4.566932**
on that update. The final three in-sample losses were
`[4.566806, 2.548806, 0.114161]` vs the initial
`[4.568401, 2.570731, 0.114927]`. LoRA's six raw updates ended at
`[4.567679, 2.560755, 0.114157]`; the joint endpoint is not better on
every example. These are three different complete-answer objectives on
**training prompts**, not held-out language accuracy. The matched quantity
is predictive displacement, not parameter path length or compute.

The [previous full correction](LINE_SECANT.md) on the same joint successor
reached only 0.69090T under the cap and did not advance. For geometry the
current midpoint reached 0.71028T (peak) or 0.70784T (successor), both still
below the 0.98T floor. These describe finite searches, not impossibility
proofs. **This does not establish a general successor advantage.** With
a three-prompt bank, a cyclic position can coincide with a meaningful
prompt identity by accident. A useful next control is to permute the bank
without changing the loss schedule, and distinguish choosing the loss
example by identity from choosing an adjacent index.

## Evidence and reproduction

The [compact results](line_midpoint_results.json) preserve all four triggered
corrections, cap-safe maxima, first refusal, costs, final losses and input
hashes. The primary attempt counts are **17 raw updates, one corrected update,
three terminal refusals**. Before those decisions: 350 scalar trials, 1,125
anchor readouts (including origin and probe banks) and four extra checkpointed
anchor forwards/backwards. A success via rescue can cost more than an ordinary
step; the winning attempt used 93 readouts. No loss gradients or origin
references are recomputed *by the rescue*.

A separate raw-only shadow was run from each of the 21 primary origins with
the same loss example and bank order. These shadows used 281 scalar
candidates and 906 anchor readouts; the independent reconstruction of the
four probes also recomputed four loss gradients. None is in the primary
work count. Shadow receipts matched raw accepts or the entire recorded
first refusal. Python recomputed KL from actual logits,
checked model states on raw accepts, negative source gradient dot realized
movement on the rescue, and checkpoint immutability on refusal. The old
nine-arm secant study's native result, journal and cuts also reproduced
byte-for-byte as a separate compatibility test. No pretrained model, held-out
text, GPU, Torch, NumPy, remote CI or publication was involved.

Node/WASM repeated all six trajectories from origin and verified 12 suffix
updates and three terminal refusals from native checkpoints. The browser's
opt-in **Keep the step that works** comparison ran in headless Chrome; the
six rows fit 390px and 1280px views. An older WASM package was deliberately
served with the new page: its unsupported policy surfaced as an error,
without a result table, and the current package returned to ready on reload.

Final verification after moving rare refusal data off the common return path:
1,336 Rust unit tests and 107 isolated native Python tests (32 session, 71
decoder, four LoRA) passed. The final wheel reproduced the rescued prefix
step, all 12 recorded suffix updates and three unchanged-state refusals with
identical receipts; the rebuilt WASM report was byte-identical. These
compatibility checks are not additional primary trajectories.

From a local CPU wheel, the [previous synthetic recipe](LINE_SECANT.md) yields
the required pinned `secant/raw.jsonl`. Use fresh output paths and web-target
WASM glue built with `--no-default-features --features nn`:

```sh
python -IB bindings/st-wasm/tests/native_line_midpoint_fixture.py secant/raw.jsonl raw.json cuts.json
python -IB bindings/st-wasm/tests/native_line_midpoint_fixture.py verify raw.json cuts.json raw.jsonl
node bindings/st-wasm/tests/native_line_midpoint_node.mjs /path/to/pkg/spiraltorch_wasm.js raw.json cuts.json > wasm.json
python -ISB benchmarks/results/2026-11-23-native-wave-agreement/matched-rival/declared-set/line_midpoint_summary.py raw.json raw.jsonl cuts.json wasm.json summary.json
```

For the interactive version, see the [browser setup and API](../../../../../bindings/st-wasm/tests/native_causal_lm.md).
