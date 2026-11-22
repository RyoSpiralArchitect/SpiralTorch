# Changing the direction did not rescue the capped mean

On the **same generated language decoder, three training prompts and two
epochs**, fully removing a single anchor's estimated normal did not unlock the
previously stopped WaveGate or joint runs. It also stopped a previously viable
LoRA update. Keep the raw ray as the default. This negative result is about
the specified one-probe projection, not about every direction or about GPT-2.

[Protocol frozen in `7676b8c`](LINE_SECANT_PROTOCOL.md), before implementation
or measurement. Six opportunities per trajectory. For each new proposal the mean
forward KL on the three prompts must be between 0.98T and T; each prompt must
stay at or below 2T, where T = 1/65536 nat. Same parameter ceiling (1/8), 20
scalar trials maximum. The one raw probe at distance 1/256 is never committed.

| Selected parameters | Raw loss gradient | Project out largest probe anchor | Project out its cyclic successor |
| --- | ---: | ---: | ---: |
| LoRA | **6/6** | 0/6 | 0/6 |
| WaveGate | 0/6 | 0/6 | 0/6 |
| Joint | 0/6 | 0/6 | 0/6 |

For WaveGate with the largest-anchor policy, the normal for the second
prompt was nearly parallel or antiparallel to the loss gradient: only **1.73%**
of its norm remained after projection. The optimizer still normalized this
residual to the requested step length. Even at the parameter ceiling it spent
only **0.00311T** in mean KL (the successor reached 0.06093T). That distinction
matters: the search failed on measured predictive movement, not because it
silently took a smaller parameter step.

For joint, **93.11%** of the norm remained after removing the third prompt's
normal. But on the best tested cap-safe candidate the limiting prompt shifted
from the third to the **second**; spending **0.6994T** still missed the 0.98T
floor. Raw joint had reached 0.8596T; the successor reached 0.6909T.
For LoRA, protecting the third prompt made the **first** prompt the limiting
one and the best cap-safe mean fell to 0.8892T. None of these bounded searches
proves that no admissible update exists. There is no nine-way comparison of
final loss and there are no held-out measurements.

## What was checked

The [machine-readable result](line_secant_results.json) includes the best
observed cap-safe trial of every refusal, selected bank indices (zero based),
work and file hashes. Six applied steps, eight unchanged-state refusals,
195 scalar candidates, 645 last-position anchor readouts across all nine arms.
The shaped arms also needed six checkpointed anchor forwards with seeded
backwards. These counts exclude independent audit work: Python reconstructed
the raw probe for each shaped attempt, recomputed its KLs and selection,
checked accepted movement from logits and weights, and compared all three raw
controls to the [previous run](LINE_ANCHOR_ENVELOPE.md) including checkpoint
pins. Rust finite differences on a **different** generated fixture verify
that the projected direction can be admitted, that frozen coordinates stay
fixed and that the reported source-gradient dot displacement is correct.

Node/WASM imported all nine Python cuts and verified four suffix updates and
eight refusals. The final Rust suite passed 1,332 tests; the final isolated
CPU wheel passed 31 session, 71 decoder and four LoRA tests. The session tests
also exercised a successful shaped update on another artificial fixture and
refused to resume its checkpoint with a different anchor rule.
The primary comparison used a fresh CPU development wheel;
no PyTorch, NumPy, GPU, pretrained input, online service or restart of a
refused trajectory. A post-study parser check additionally found and fixed
acceptance of stray keys on explicit raw directions; no valid study plan
uses that syntax. The final WASM build rejected that malformed plan and
reproduced the cross-runtime report byte-for-byte. The primary runs were not
repeated to get a better result.

## Reproduce with generated inputs

Use an installed local CPU extension, generated WASM web glue, and fresh
output paths. Build the previous envelope inputs using the
[preceding recipe](LINE_ANCHOR_ENVELOPE.md); here `envelope/` contains its
`raw.json` and `cuts.json`. Both must match their pinned hashes.

```sh
python -IB bindings/st-wasm/tests/native_line_secant_fixture.py envelope raw.json cuts.json
node bindings/st-wasm/tests/native_line_secant_node.mjs /path/to/pkg/spiraltorch_wasm.js raw.json cuts.json > wasm.json
python -ISB benchmarks/results/2026-11-23-native-wave-agreement/matched-rival/declared-set/line_secant_summary.py raw.json raw.jsonl cuts.json wasm.json summary.json
```

A useful next control, **not run**, is to leave any already feasible raw step
alone and test a predeclared partial rather than full correction on the blocked
step. Evaluate it against the same unchanged all-prompt acceptance rule; do not
interpret easier acceptance as better language learning.
