# Completed Two-Lag And Learned-Order Comparison

All twelve frozen-GPT-2 adapter runs completed and were verified: four arms,
seeds 41/43/47, 512 updates each (6,144 primary updates), plus 24
continuation-only updates. Endpoints were evaluated only after all runs and
continuation checks finished. The process exited with code 0. This is an
exploratory reused-book study, not a speed benchmark or general LLM claim.

The [fixed protocol](../../../docs/fractional_two_lag_study.md) inserts two
zero-initialized feature gates after `transformer.h.0.mlp`. The pretrained
base stays frozen. Adam 0.001, strength 0.1, B2/T128/F768, K32/h1, CPU float32
interfaces and two threads are shared. Development does not select duration,
hyperparameters, checkpoints or the reported condition. Ordinary lag2 and
fixed-two GL have 1,536 trainable parameters; learned-order arms have 1,537.
The fixed GL adapter also stores its frozen order scalar.

## Results

Mean next-token cross-entropy in nats, lower is better. All three paired seeds
are included; the no-adapter baseline receives no updates.

| Arm | Pride Unused Tail (120 Blocks) | Alice Transfer (32 Blocks) |
| --- | ---: | ---: |
| Frozen baseline | 4.073713269 | 4.011807486 |
| Ordinary `lag2` | 4.055657884 | 4.001160219 |
| Rust GL, fixed alpha=2 | 4.055657884 | 4.001160219 |
| Rust GL, learned from alpha=2 | 4.047328409 | 3.996844461 |
| Rust GL, learned from alpha=1 | 4.054341469 | 4.000331007 |

| Predeclared Contrast | Pride Mean CE Difference | Alice Mean CE Difference | Negative Seeds (Pride / Alice) |
| --- | ---: | ---: | ---: |
| Fixed-two minus ordinary lag2 | 0 | 0 | 0/3 / 0/3 |
| Learned-two minus fixed-two | -0.008329476 | -0.004315759 | 3/3 / 3/3 |
| Learned-one minus learned-two | +0.007013061 | +0.003486546 | 0/3 / 0/3 |
| Learned-one minus ordinary lag2 | -0.001316415 | -0.000829212 | 3/3 / 3/3 |

Ordinary `-2*x[t-1]+x[t-2]` and fixed-two Rust GL match exactly: all common
training receipts, development values, final gate tensors, named Adam state
and every endpoint block loss. These are implementations of the same map,
not independent quality treatments. The ordinary reference accumulates in
float64 to match Rust's float32 output boundary; it is not a timing competitor.

Learning from two improves over the fixed-two map in every paired seed on
both sets. Learning from one improves less and loses to the two-initialized
arm in every pair. The learnable-order versus fixed-order comparison adds one
trainable scalar; it is not an equal-capacity comparison.

| Seed | Final Alpha, Initialized At 2 | Final Alpha, Initialized At 1 |
| --- | ---: | ---: |
| 41 | 4.055050373 | 2.097662687 |
| 43 | 3.757410049 | 1.949106097 |
| 47 | 3.951384306 | 2.002378225 |

Each learned order receives 511 nonzero gradients after the initial zero-gate
update, and every learned endpoint is its observed maximum. **This is not
convergence to four or evidence that four is optimal.** Changing order also
changes short-filter shape, coefficient magnitudes and conditioning. At an
integer, the strictly-past GL filter has finite short support. A bounded-tap
and gain-controlled comparison is still needed before attributing the effect
to longer fractional memory. No such ablation was run in this record.

The repeated one-initialized condition reproduces the earlier one-lag study's
gate/order tensors, Adam state, all training/development receipts and endpoint
scores exactly across all three seeds, despite the distinct native binaries.
This checks the optimized backend against previous training behavior; it is
not three additional independent seeds or a second quality confirmation.

## Verification

Training clients/configuration were frozen at
`92d6d7d5ef36d25413132d192a853e6983041382`. The 70-file native/Python package
is the buffer-transport runtime bound in the preceding publication; its native
hash is `5caece088b275f15f83a1bfb8e7216613c9470faf3f01f46ddf4b0f95f05f8f4`.
All six frozen client files and 70 package files remained unchanged.

Review identified that scalar norms alone cannot verify final-state parity.
The corrected postprocessor at `a25aa224c472be1851f17e4a956e8c846039e3ef`
is bound separately in `analysis-manifest.json`; it was not hot-swapped into
training. It hashes the exact loaded checkpoint bytes, verifies identity,
cursor and receipts, and compares gate/Adam tensor contents by parameter name.
Independent saved-state verification also restores all twelve adapters and
optimizers and checks finite values, scalar trajectories and frozen modes.

A completed resume exited without rescoring. All 100 sealed study files,
including 96 intermediate/final checkpoints, remained byte-identical. Both
summary variants are byte-identical under Python hash seeds 1 and 42. The
preflight retains its original not-yet-launched status; `validation.json`
records completion separately rather than rewriting history.

Local regression checks passed: 453 geometry/study/publication tests and 34
benchmark/publication tests, with no skips. The 18 JIT deprecation warnings
are retained. The isolated receipt-only CLI rebuild is byte-identical.

`summary.json` includes locally verified checkpoint-state hashes.
`summary-receipts.json` is independently rebuildable from public numeric data
and intentionally labels full same-math parity `unverified`: it has not read
the private checkpoint tensors. Public verification receipts and hashes do
not substitute for third-party access to those tensors.

The artifact-binding review correction separates `summary_source_sha256`
(the frozen postprocessor) from `summary_artifact_sha256` (the actual full-state
JSON) and `receipt_summary_artifact_sha256`. The verifier independently rebuilds
both outputs and requires byte equality, rather than trusting matching claimed
hashes. A changed CE with a correspondingly updated manifest hash is rejected.
Both frozen Python sources are included for inspection. The superseded derived
records remain available in commit `3f08cb4a`; their hashes are recorded in
`supersedes`. Training inputs, checkpoints and both summary outputs are unchanged.
The correction reran 456 geometry/study/publication tests and 34 benchmark
tests, with no skips. Executing the published verifier reproduces the corrected
receipt byte-for-byte; optimized Python is also checked to fail closed.

## Reproduce

Rebuild the receipt-only report without Torch, models, scoring or network:

```sh
RESULT=benchmarks/results/2026-10-05-fractional-two-lag-study
python tools/summarize_wave_gate_long_horizon.py \
  --plan "$RESULT/plan.json.gz" --results "$RESULT/results.json.gz" \
  --journal "$RESULT/journal.json" --output "$NEW_RECEIPT_SUMMARY_JSON"
cmp "$NEW_RECEIPT_SUMMARY_JSON" "$RESULT/summary-receipts.json"
```

To reproduce local full-state verification, add `--checkpoint-dir` pointing
to the original hash-matching study checkpoints and compare with `summary.json`.
The included `verify_completed.py` additionally restores every adapter/Adam
state and binds the published output bytes; with the original frozen client
and package on `PYTHONPATH`, supply `--study`, `--manifest` (the frozen-runtime
manifest), `--analysis-manifest` and a fresh `--output`. The local launch/client
manifest records must remain beside the frozen-runtime manifest. Assertions
must be enabled; the verifier refuses optimized Python execution.
For fresh training, follow the linked protocol with existing hash-matching
assets and an isolated frozen runtime. Do not resume with modified clients.
CI checks numeric reconstruction, completion receipts and publication hashes;
it does not retrain GPT-2 or access private checkpoints.

The gzip files preserve original plan/result bytes and every numeric training,
development and endpoint record. `SHA256SUMS` covers the full publication.
Weights, corpus text, checkpoints, native packages and raw logs remain local.
Books are reused, possibly seen in pretraining, and seeds share evaluation
blocks. There is no pristine-data, statistical-significance, unique geometric
causation, generation-quality, GPU or full-model throughput claim.
