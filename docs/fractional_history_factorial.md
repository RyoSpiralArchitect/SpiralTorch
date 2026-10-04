# History Length And Coefficient Energy

The [two-lag study](fractional_two_lag_study.md) found a small improvement
when learning away from alpha=2, with final orders still moving above three.
Changing order also changes the filter's gain and short-lag shape. That is
not enough to attribute the effect to longer fractional history.

This fixed four-arm follow-up crosses **available history length** with
**raw versus constant coefficient energy**. All maps and differentials remain
in Rust; Python only selects recipes and runs the shared HF trainer.

## Matched Arms

| Arm | Kernel | Coefficient Norm | Trainable Parameters |
| --- | --- | --- | ---: |
| `history_raw_short` | K=3, two past taps | Changes with alpha | 1537 |
| `history_raw_full` | K=32, 31 past taps | Changes with alpha | 1537 |
| `history_l2_short` | K=3, two past taps | Fixed to sqrt(5) | 1537 |
| `history_l2_full` | K=32, 31 past taps | Fixed to sqrt(5) | 1537 |

Every arm starts with log-alpha=log(2), independent zero local/history gates
and the same `[-2, 1]` past filter. The normalization constant is rounded to
float32 by the binding. The initial adapter is exactly identity; nonzero-gate
output, input VJP and gate gradients are checked against the ordinary two-tap
Torch reference. At integer order two, the full filter retains nonzero order
derivatives in older taps whose forward coefficients are zero. The short
filter cannot learn those taps.

The production adapters are `FractionalHistoryAdapter` and
`FractionalL2HistoryAdapter`. Study wrappers reject loading another arm's
recipe. All four have the same parameter names, registration order and Adam
setup. The raw-full arm repeats the preceding learned-from-two recipe, not a
new independent seed. Exact historical replay is checked separately rather
than counting that repeat twice.

Fixing coefficient energy does **not** fix hidden-state variance, residual
amplitude after learned gates, or the optimizer's geometry. Longer kernels
also change normalization and the learned short taps. The factorial separates
these declared interventions; it cannot uniquely identify a memory mechanism.

## Frozen Protocol

`hf_fractional_pride_history_factorial.json` preserves the preceding local
GPT-2 snapshot, corpus hashes, first-MLP insertion point, seeds 41/43/47,
512 updates per arm, Adam 0.001, strength 0.1, batch 2, context 128 and two CPU
threads. The frozen base and 1537 trainable adapter parameters are identical
across arms. Model/data/update budgets match; operator arithmetic differs.
This is not a speed benchmark.

Use 1204 training blocks, 16 diagnostic development blocks, and delayed
120 Pride / 32 Alice endpoint blocks. Complete all 6144 primary updates and
24 continuation-only checks before endpoint scoring. Development cannot
select duration, checkpoints, hyperparameters or which arm is reported.
These reused exploratory books may appear in pretraining; they are not
pristine confirmation. Three minibatch-order seeds do not establish
statistical significance or general LLM superiority.

The five predeclared CE contrasts are raw-full minus raw-short, L2-full minus
L2-short, L2-short minus raw-short, L2-full minus raw-full, and the interaction
`(L2-full - L2-short) - (raw-full - raw-short)`. Negative differences favor
the positive-weight arm. A negative interaction means the length contrast
is more favorable with normalization, not that normalization wins every arm.

The summary validates shared initialization/capacity, complete schedules and
order trajectories, and preserves a failed first-update receipt match. A
receipt match is not full tensor-state verification. Do not compare final
states for equality across these four different learned maps. Separately
verify checkpoint hashes, reconstructed recipes, continuation and the
historical raw-full repeat. Report all outcomes, including losses and orders
that continue moving at step 512; do not call the endpoint convergence.

## Run And Resume

Freeze the source client/helpers and isolated native package. Use existing
hash-matching local assets, no downloads, and a fresh study directory:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 \
  python bindings/st-py/examples/hf_fractional_history_factorial.py \
  --config bindings/st-py/examples/hf_fractional_pride_history_factorial.json \
  --model-dir "$LOCAL_GPT2" --corpus "$LOCAL_PRIDE" \
  --transfer-corpus "$LOCAL_ALICE" --output-dir "$NEW_STUDY_DIRECTORY"
```

The shared driver owns the writer lock, immutable checkpoints, atomic
journal, runtime/data identity and exact cursor. Repeat with identical
arguments plus `--resume`; completed resume verifies saved results and does
not rescore endpoints. No KV cache, mixed precision, padding, packed documents
or persistent history across blocks is supported by this protocol.

Summarize read-only with `tools/summarize_wave_gate_long_horizon.py` and
`--plan`, `--results`, `--journal`, and a fresh `--output`. The summary is
Torch-free and does not load model weights. The optional `--checkpoint-dir`
is specific to the earlier two-lag parity study; do not use it here or imply
that scalar receipts establish full saved-state verification.

Publish complete numerical outcomes, verification records, hashes and
reproduction instructions. Keep weights, corpus text, runtime packages and
raw logs local. Passing preflight or a running job is not a completed study.
