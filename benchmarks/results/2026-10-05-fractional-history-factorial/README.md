# History Length x Coefficient Energy

Completed offline CPU float32 learning study, not a speed benchmark.
Protocol: [history factorial](../../../docs/fractional_history_factorial.md).
Primary revision: `d9d995911217233705fbd80b4d33253a25c7ebff`.
Saved-state verifier revision: `974a6c938eea5cd4fbf03fd8c63fc42c2881f5bc`.

## Fixed Comparison

Local pretrained GPT-2, frozen base, first-MLP insertion, independent local
and strictly-past history gates, learned order initially 2. All four adapters
have 1537 trainable parameters and start as identity with the same `[-2, 1]`
history filter. Kernels/differentials run in Rust. Short means K=3 (two past
taps); full means K=32 (31 past taps). L2 fixes coefficient norm to float32
sqrt(5), not hidden-state variance or learned residual amplitude.

Seeds 41/43/47, 512 updates each, Adam 0.001, strength 0.1, batch 2, context
128, two CPU threads. All 6144 primary updates and 24 continuation-only
checks finished before fixed endpoint scoring. There were also 12 auxiliary
preflight updates. Model/data/update budgets match; operator FLOPs do not.
No intermediate development result changed the duration or selected an arm.

Training uses 1204 Pride blocks; the 16 development blocks are diagnostic.
The fixed endpoints are 120 unused Pride-tail blocks and 32 Alice blocks.
These are reused exploratory books and may occur in pretraining, not a
pristine confirmatory test set. Token/block/model hashes are published.

## All Outcomes

Mean next-token cross-entropy over the three seeds; lower is better.

| Arm | Pride | Alice |
| --- | ---: | ---: |
| Frozen base | 4.073713269 | 4.011807486 |
| Raw short | 4.055112133 | 4.000905382 |
| Raw full | 4.047328409 | 3.996844461 |
| L2 short | 4.050132384 | 3.997908764 |
| L2 full | 4.049777804 | 3.997691326 |

Every predeclared contrast below has the same sign in all three seeds on
both endpoint sets. Complete per-seed differences and descriptive sample
SDs are in `summary.json`; this is not a significance claim.

| Predeclared Contrast | Pride CE Difference | Alice CE Difference |
| --- | ---: | ---: |
| Raw full - raw short | -0.007783725 | -0.004060922 |
| L2 full - L2 short | -0.000354580 | -0.000217438 |
| L2 short - raw short | -0.004979750 | -0.002996619 |
| L2 full - raw full | +0.002449395 | +0.000846865 |
| Length-by-normalization interaction | +0.007429145 | +0.003843484 |

Longer available history helps both raw and fixed-energy variants here,
but its measured benefit becomes much smaller at fixed energy.
Normalization improves the short arm and worsens the full arm relative to
their raw counterparts. Raw full remains best under this fixed budget.
The data do not support attributing its advantage uniquely to long memory,
nor do they support adopting normalization as a universal improvement.

## Native Filter Description

The saved-state verifier also sends a unit impulse through each final
Rust recipe. This is a **post-hoc auxiliary description**, not a selection
criterion or a fresh quality test. Python never reconstructs GL coefficients.

| Arm | Final Alpha Range | Coefficient L2 Range | Energy At Lags >= 3 |
| --- | ---: | ---: | ---: |
| Raw short | 2.508-2.578 | 3.140-3.283 | 0% |
| Raw full | 3.757-4.055 | 7.106-8.606 | 18.9-26.0% |
| L2 short | 1.040-1.055 | approximately 2.236068 | 0% |
| L2 full | 0.960-0.975 | approximately 2.236068 | 0.0026-0.0070% |

The order takes very different trajectories after fixing coefficient
energy. Final orders remain moving at the fixed endpoint; none is called
optimal or converged. The impulse description does not measure hidden
variance, effective gated residuals, long-range retrieval, or behavior
across blocks. A future test should separate learnable gain from filter
shape and include a matched short ordinary filter, rather than simply
increasing the available history length.

## Verification And Reproduction

Primary process and completed resume both exited zero. Completed resume
did not retrain or rescore; all 100 sealed output files, including 96
checkpoints, remained unchanged. All 12 endpoint adapters and named Adam
states were restored and checked for recipe, dtype, shape, finiteness,
step, schedule and scalar-order consistency. The five frozen client files,
71 runtime files and seven model assets matched their hashes.

Raw full exactly replays the preceding learned-from-two arm for each seed:
all training records, development scores, final parameter bits, named Adam
moments, baseline and endpoint block losses agree. This repeats existing
evidence; it does **not** add independent seeds. The older study's 100
output files, six client files and 70 runtime files remain unchanged.

Preflight records are preserved as preflight, including their historical
`not_yet_launched` status. `validation.json` records final completion.
The validation test counts describe the verifier revision; publication
consistency tests are additional and do not rerun private saved states.

Rebuild the public summary without Torch or model weights, writing a new
file rather than overwriting this record:

```sh
python summarize_wave_gate_long_horizon.py \
  --plan plan.json.gz --results results.json.gz --journal journal.json \
  --output /tmp/history-factorial-summary.json
cmp summary.json /tmp/history-factorial-summary.json
shasum -a 256 -c SHA256SUMS
```

Actual private checkpoint verification additionally needs the hash-matching
frozen runtime/client, this study's local `study` directory, and the prior
two-lag study. See the protocol's verifier command. The published verifier
checks actual reconstructed summary **bytes**, not just matching hash
claims; source and output hashes are separate. A historical replay mismatch
is retained as a failed criterion rather than hidden. It reads continuation
receipts and saved states, not new model updates.

Weights, corpus text, runtime packages and raw logs stay local. Public
numeric checks do not themselves reproduce private checkpoint verification.
No claim is made about speed, statistical significance, general LLM quality,
GPU execution, pristine held-out generalization or converged training.
