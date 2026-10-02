# Directional Elliptic Comparison: Completed

All nine fixed-endpoint runs and their exact next-update continuation checks
completed. The actual training/evaluation process exited successfully. A second
completed-study resume verified checkpoint/result hashes without scoring again;
the result hash stayed unchanged. Source and protocol were committed before
launch at `4227af8c36476eb4c85272dec0d5f379af1b0ec2` and executed from frozen copies.

See [the protocol](../../../docs/elliptic_nonlinear_study.md) for the exact arms,
reproduction command and limitations. All three arms share the learned `768 -> 2`
projection and zero-start `9 -> 768` readout: **8450 trainable parameters each**.
The controls are the Rust-derived tangent feature map, componentwise tanh of its
displacement, and the full Rust elliptic/Lie feature map at `(1,u,v)`.

## Fixed Before Outcomes

- Cached frozen float32 CPU GPT-2, explicit `transformer.h.0.mlp` placement.
- Seeds 41/43/47, paired projection initialization and paired minibatch order.
- 512 Adam updates per run, batch two, block size 128, lr 0.001, strength 0.1.
- 130048 causal targets per run; nine runs and **4608 primary updates completed**.
- 18 additional continuation-check updates, excluded from the fixed endpoints.
- Final scoring only after all runs and exact next-update continuation checks.
- No checkpoint selection, early stopping or endpoint-driven configuration change.

The Pride/Alice evaluation partitions were inspected by the preceding WaveGate
study. This is a reused exploratory benchmark, not a pristine held-out test.
Within this comparison, parameter counts and initialization match; feature rank,
expressivity, typical feature scale, gradient conditioning and compute need not.
The tanh arm is not an unconstrained learned MLP. The 8450-parameter arms must not
be presented as a parameter-matched victory over WaveGate's 1536/1537 parameters.
No speed, text-generation-quality or broad generalization claim follows.

## Result

Mean causal cross-entropy across three paired seeds, lower is better. Baseline is
the same frozen zero-update GPT-2, evaluated once per set after training finishes.

| Arm | Pride tail, 120 blocks | Alice transfer, 32 blocks |
| --- | ---: | ---: |
| Frozen baseline | 4.073713 | 4.011807 |
| Tangent control | **3.566816** | **3.673728** |
| Ordinary tanh control | 3.754525 | 3.771032 |
| Rust elliptic/Lie map | 3.717387 | 3.759478 |

Every active arm improves its baseline in every seed on both sets. The geometric
arm beats this particular tanh control in all paired comparisons: mean CE
differences **-0.037138** on Pride and **-0.011554** on Alice. However, tangent
remains best in every seed. Elliptic is worse than tangent by **+0.150571** and
**+0.085750**, respectively. This is not evidence to prefer the geometric adapter
over the simpler tangent adapter for this task.

Both learned projections receive real finite loss gradients in every run. The
zero-start readout makes the first orientation gradient zero as designed; later
updates propagate through Rust's VJP. All adapter/Adam next-update comparisons
are exactly equal after loading disk checkpoints, and the base remains frozen.
These demonstrate working learning and continuation, not geometric superiority.

The tanh comparison only distinguishes this fixed ordinary nonlinear control;
it does not isolate topology from feature scale, effective rank, conditioning or
global expressivity. Do not tune using these endpoint scores and call the next
test independent confirmation. The next mechanism question is whether geometry
helps token-to-token relationships, rather than compressing each token separately.
That remains untested here; this adapter is strictly tokenwise.

## Verification

`plan.json` records exact hashes, partitions, batch schedules and both adapter
source identities. `validation.json` records **68 passing local tests** with
zero skips. Tests include actual tiny HF loss gradients through both projections,
paired initial parameters, RNG preservation, local differential agreement,
wrong-control checkpoint rejection and interruption/resume in the second arm.
The shared runner reuses the earlier checkpoint and evaluation boundary instead
of duplicating those mechanisms. Previous published WaveGate summaries remain
byte-for-byte reproducible with the extended summary tool.

The existing native Rust binary and local model/corpora were reused unchanged.
No native build or model download was performed. All nine initial-parameter hashes
match within seed, and all nine endpoint checkpoint hashes were checked again.
The final local suite again passed 68 tests, with no skips; that is separate from
remote CI completion. The later lazy-telemetry optimization was not part of this
frozen experiment and is not retroactively included in its source identity.

`results.json.gz` retains every arm/seed's numerical training history and final
per-block losses, compressed from 2.73 MB to 0.40 MB. `journal.json` preserves
completed cursors, checkpoint receipts and the uncompressed result hash.
`summary.json` retains all paired seed comparisons, including losing conditions.
Raw terminal logs, model/data files and optimizer checkpoints remain local; no
corpus text or model weights are published here.

## Reproduce Aggregation

From the repository root, with a new output filename:

```bash
RESULTS=benchmarks/results/2026-10-02-elliptic-nonlinear-learning
python -S tools/summarize_wave_gate_long_horizon.py --plan "$RESULTS/plan.json" --results "$RESULTS/results.json.gz" --journal "$RESULTS/journal.json" --output /tmp/elliptic-nonlinear-summary.json
cmp "$RESULTS/summary.json" /tmp/elliptic-nonlinear-summary.json
```

The public compressed inputs reproduce the summary byte-for-byte, using only the
standard library. Hashes in the summary refer to decompressed JSON bytes;
`SHA256SUMS` covers the exact public files. Aggregation verifies published receipts,
not private checkpoint contents or terminal exit status. Those separate checks
are recorded in `validation.json`; use the protocol to reproduce training.

This is one frozen GPT-2, one placement, float32 CPU, three seeds and a small
adapter, not full-model FT. Seeds share evaluation blocks and are not independent
corpus replicas. Neither novel evaluation data nor absence from GPT-2 pretraining
is claimed. The result supports no significance claim or general geometry ranking.
