# One-Lag Controls For Fractional LM Learning

The [completed independent-history comparison](fractional_history_study.md)
ended with learned orders near one. At exactly alpha=step=1, strictly-past
Grunwald-Letnikov (GL) history is simply `-x[t-1]`. This study tests whether
ordinary short mixing explains the result before attributing it to long
fractional memory. It is an exploratory follow-up, not a new default.

## Four Controls

Every arm inserts after frozen GPT-2's `transformer.h.0.mlp` and uses
`local = x + 0.1*tanh(local_gate)*x` and
`output = local + 0.1*tanh(gate)*history(x)`. Both feature gates start at zero.

| Arm | History | Parameters (Total / Trainable) |
| --- | --- | ---: |
| `lag1` | Ordinary Torch `-x[t-1]`, zero at the first position | 1536 / 1536 |
| `history_fixed_one` | Rust GL, alpha fixed at 1 | 1537 / 1536 |
| `history_learned_one` | Rust GL, alpha initialized at 1 and learned | 1537 / 1537 |
| `history_learned_half` | Rust GL, alpha initialized at 0.5 and learned | 1537 / 1537 |

The first two are the **same mathematical map and trainable capacity**. Their
comparison is an implementation check, not two independent quality treatments.
The negative sign keeps gate initialization and optimizer trajectories aligned;
an unconstrained positive lag could represent the same function but would not
have the same parameter signs. The fixed scalar is real, not padding.

Learning alpha adds one scalar. At alpha=1, zero older-lag coefficients do
**not** imply zero derivatives with respect to alpha. The Rust derivative
retains them; for example, `d(alpha*(alpha-1)/2)/dalpha = 1/2` at one.
No Python fractional formula or truncated surrogate gradient is introduced.
Starting at one versus one-half changes filter initialization and conditioning.

The primary contrasts are fixed-one minus lag1 (parity), learned-one minus
fixed-one (learnable departure from one lag), learned-half minus learned-one
(initialization), and learned-half minus lag1 (the preceding recipe against
ordinary lag mixing). The half-order arm is rerun, not copied from a historical
endpoint or pooled as an additional independent seed.

## Fixed Protocol And Acceptance

`hf_fractional_pride_lag.json` retains the preceding model/corpus hashes, three
seeds (41, 43, 47), 512 updates per arm, Adam at 0.001, batch 2, context 128,
CPU float32 and two threads. Rust GL uses K=32, step=1. The ordinary control
retains the same input budget but actually uses only one past tap. No speed
claim follows from comparing different maps or timing these training runs.

Use 1204 train blocks, 16 diagnostic development blocks and delayed 120 Pride /
32 Alice blocks. Finish all twelve runs and 24 continuation-only updates before
scoring endpoints. Development cannot select the checkpoint, run duration or
hyperparameters. These books are reused exploratory endpoints, potentially in
pretraining, not pristine confirmation or independent block-level replicas.

Before launch, require exact ordinary-versus-fixed-GL outputs and gate gradients
at nonzero gates, including the real 2x128x768 shape, zero-length-history T=1
and noncontiguous tensors. Input VJPs use rtol=atol=3e-6 to allow different
floating-point accumulation paths. A tiny HF loop must have exact final gate
and named Adam-state equality. All four arms must pass exact interrupted
resume and next-update continuation while preserving the frozen base.

For the pretrained CPU-f32 run, require exact common per-step loss/gate
receipts, development scores, saved gate/named Adam states and per-block
endpoints for lag1 versus fixed-one. A mismatch is a failed parity criterion,
not evidence of a quality advantage. The read-only summary keeps receipt
mismatches visible as false parity fields; it does not certify checkpoint
contents. Verify those independently from the local saved states.

The shared driver binds source/runtime/data hashes, schedules, initialization,
recipe, optimizer state and cursor. Learned-one and learned-half checkpoints
carry distinct recipes even though their module types and parameter shapes
match. No padding, packed documents, persistent history, KV cache, mixed
precision or resident GPU execution is included.

## Reproduce

Use the built bindings and existing local hash-matching assets, without download:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 \
  python bindings/st-py/examples/hf_fractional_lag_study.py \
  --config bindings/st-py/examples/hf_fractional_pride_lag.json \
  --model-dir "$LOCAL_GPT2" --corpus "$LOCAL_PRIDE" \
  --transfer-corpus "$LOCAL_ALICE" --output-dir "$NEW_STUDY_DIRECTORY"
```

Freeze the client/helpers, package and native module throughout the run.
Use the same inputs and `--resume` for continuation. Completed resume checks
sealed outputs instead of rescoring. Generate the read-only report with:

```sh
python tools/summarize_wave_gate_long_horizon.py \
  --plan "$STUDY/plan.json" --results "$STUDY/results.json" \
  --journal "$STUDY/journal.json" --output "$NEW_SUMMARY_JSON"
```

Publish all numeric outcomes, input/source hashes and verification receipts,
including failed criteria and losing results. Keep corpora, weights,
checkpoints, native packages and raw logs local. Implementation tests and a
running process are not completed pretrained quality results.

## Completed Result

The [twelve-run result](../benchmarks/results/2026-10-03-fractional-lag-study/README.md)
completed all 6,144 primary updates and 24 continuation checks. Ordinary lag1
and fixed-one Rust GL matched exactly through saved gates, named Adam states
and per-block endpoints. Learning from one improved CE in all three seeds on
both reused sets (mean differences -0.002840618 / -0.001799345 versus lag1)
and outperformed initialization at one-half. The repeated half-order arms
exactly reproduce the preceding study; they are not additional independent seeds.

The one-initialized orders finish around 1.95-2.10, at their observed maxima,
not at a demonstrated optimum. At exactly two, strictly-past GL becomes the
ordinary two-tap map `-2*x[t-1] + x[t-2]`. Fixed-two controls and truncated-history
comparisons are the next separation to test before attributing the result to
long fractional memory. No such follow-up or speed claim is included here.
