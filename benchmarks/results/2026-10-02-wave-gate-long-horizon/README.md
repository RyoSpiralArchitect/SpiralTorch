# Long-Horizon Radius Study: Completed

All nine fixed-endpoint runs completed, including exact next-update continuation
from disk for adapter parameters and Adam state. The actual training/evaluation
process exited successfully. A subsequent completed-study resume verified the
saved artifacts without rerunning endpoints; the result hash remained unchanged.
The executable source/protocol was committed before training at
`82d5ed3d4815ddee32e434941f553709e2e7319a` and run from a frozen local copy.

See [the protocol and restart contract](../../../docs/wave_gate_long_horizon.md).
The run used the already validated native radius binary, without rebuilding
Rust or downloading models/corpora. Mathematical ownership remains Rust; Python
adds experimental orchestration rather than a second geometric implementation.

## Locked Design

- Cached, frozen float32 CPU GPT-2; gate/bias adapter at `transformer.h.0.mlp`.
- Three seeds, three active arms: tangent, fixed radius 4, learned-from-radius 4.
- 512 updates per run, batch two, 128-token blocks, Adam lr 0.001, strength 0.1.
- 130048 causal targets per run; **4608 completed primary updates** in total,
  plus 18 validation-only updates excluded from the fixed endpoints.
- Development: the same 16 previously inspected Pride tail blocks every 128 steps.
- Final endpoints: the other 120 Pride tail blocks and 32 fixed Alice blocks,
  evaluated only after all nine runs and their continuation checks complete.
- Shared zero-update baseline; no best-checkpoint selection, early stopping,
  outcome-driven schedule changes or speed comparison.

The learned-radius arm has one additional parameter. Alice appears in older
unrelated repository studies; neither book is claimed absent from GPT-2
pretraining. Evaluation novelty is limited to this WaveGate comparison.

## Result

Mean causal cross-entropy across the three paired data-order seeds, lower is
better. Baseline is the same frozen zero-update model, evaluated once per set.

| Arm | Unused Pride tail, 120 blocks | Alice transfer, 32 blocks |
| --- | ---: | ---: |
| Frozen baseline | 4.073713 | 4.011807 |
| Tangent control | **4.016924** | **3.976307** |
| Fixed radius 4 | 4.049989 | 3.997139 |
| Learned from radius 4 | 4.034821 | 3.987644 |

Every active arm improves versus baseline in every seed on both sets. However,
tangent wins every paired comparison. Learned radius improves its fixed-radius
control in every seed but remains worse than tangent by mean CE **0.017896** on
Pride and **0.011336** on Alice. This is not a demonstrated geometric advantage.

The learned endpoint radii are **9.321881, 9.197344 and 9.297201** for seeds
41/43/47, up from 4. In the final *pre-update training forwards*, relative radial
gain averages are about 0.235-0.244 for fixed radius versus 0.589-0.601 for learned
radius. These local map diagnostics support reduced projection contraction;
they are not the full loss Jacobian, nor post-update evaluation-set diagnostics.
Some affine values now enter elementwise saturation, unlike the short pilot,
so the earlier no-saturation observation must not be generalized to this run.

Do not extend a radius grid based on these final scores and call it independent
validation. The next hypothesis is a matched ordinary nonlinear control and
geometry that learns directions/relationships rather than only shrinking norms.
The present result establishes working, stable gradient plumbing, not a reason
to prefer this geometry over the simpler adapter for this language-model task.

## Evidence

`plan.json` fixes corpus/token hashes, model/runtime/source identities, actual
data partitions and every batch schedule. `validation.json` records **62 passing
Python tests**: 50 covering geometric clients and restart/evaluation boundaries,
plus 12 for completed-result aggregation without Torch or another evaluation.
Tests include a real tiny HF interruption/resume with exact parameter, Adam and
history equality; changed-protocol/corrupt-checkpoint rejection; single-writer
exclusion; rejecting mutated-base checkpoints; and preventing early endpoint
evaluation. CI now includes these geometry tests with explicit native-export
checks and offline tiny HF models; local tests do not claim remote CI completion.

Post-run review found that resume compared a saved plan's claimed ID without
rehashing its contents. A scratch-fixture regression reproduced the issue, and
the current driver now rejects edits to saved config, data/source hashes or batch
schedules. The executed study's original plan binding was independently rehashed
and matched its frozen public copy. Its result and executed source identity are
unchanged; this later guard is not presented as part of the measured old source.

The summary tool preserves paired per-seed cross-entropy differences for each
evaluation set, including losing seeds. It verifies the sealed result hash and
refuses incomplete evidence, incorrect means, changed schedules and overwrite of
an existing record. See the protocol document for its reproduction command.

`results.json.gz` retains every condition's numeric training history and final
per-block losses, compressed from 4.56 MB to 0.58 MB. `journal.json` preserves
completed cursors, checkpoint receipts and the uncompressed result hash.
`summary.json` contains paired seed comparisons derived from those records.
Original terminal logs, model/cache data and optimizer checkpoints remain local;
the public receipt includes their verification results and relevant hashes, not
corpus text or model weights. Checkpoints were flushed before their hashes entered
the journal. A surviving journal alone was not treated as process completion.

## Reproduce Aggregation

From the repository root, with a new output filename:

```bash
RESULTS=benchmarks/results/2026-10-02-wave-gate-long-horizon
python tools/summarize_wave_gate_long_horizon.py --plan "$RESULTS/plan.json" --results "$RESULTS/results.json.gz" --journal "$RESULTS/journal.json" --output /tmp/wave-gate-long-summary.json
cmp "$RESULTS/summary.json" /tmp/wave-gate-long-summary.json
```

This was reproduced byte-for-byte from the public compressed inputs. Summary
input hashes refer to decompressed JSON bytes; `SHA256SUMS` covers the exact
published files. Aggregation does not rerun a model or independently verify local
checkpoint contents. Use the protocol document to reproduce training itself.

This is one frozen GPT-2, one adapter placement, float32 CPU, three sample-order
seeds and a small adapter (1536 or 1537 parameters), not full-model FT. Evaluation
blocks are shared across seeds and are not independent experimental replicas.
No significance, text-generation-quality, pristine-data generalization or speed
claim follows. The learned arm's extra scalar and earlier radius selection remain
part of the comparison's limitations.
