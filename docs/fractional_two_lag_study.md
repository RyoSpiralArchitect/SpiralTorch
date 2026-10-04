# Two-Lag Controls For Fractional LM Learning

The preceding [one-lag study](fractional_lag_study.md) left learned-from-one
orders near two and still moving. That does not establish convergence or a
long-memory advantage. At alpha=2, h=1 and K>=3, strictly-past GL is just
`-2*x[t-1] + x[t-2]`. This fresh exploratory comparison asks whether an
ordinary short filter explains the effect before attributing it to geometry.

## Four Arms

All arms insert after frozen GPT-2's `transformer.h.0.mlp` and use independent
zero-initialized local/history gates:
`x + 0.1*tanh(local_gate)*x + 0.1*tanh(gate)*history(x)`.

| Arm | History | Total / Trainable Parameters |
| --- | --- | ---: |
| `lag2` | Ordinary Torch two-tap filter, missing taps zero | 1536 / 1536 |
| `history_fixed_two` | Rust GL, alpha fixed at 2 | 1537 / 1536 |
| `history_learned_two` | Rust GL, alpha starts at 2 and learns | 1537 / 1537 |
| `history_learned_one` | Rust GL, alpha starts at 1 and learns | 1537 / 1537 |

The first two are the same map/capacity, not independent quality treatments.
The ordinary reference accumulates the two taps in float64 before its float32
output boundary, matching Rust even near float32 cancellation limits. It is
an independent study control, not a production fractional backend or a timing
competitor. All GL coefficients and derivatives remain in Rust.

The learned scalar is real additional capacity. At the integer, older forward
coefficients vanish but their order derivatives need not: for the third past
tap, `d[-alpha*(alpha-1)*(alpha-2)/6]/dalpha = -1/3` at two.
The learned arm retains that derivative; it is not a two-tap surrogate.

Primary contrasts are fixed-two minus lag2 (parity), learned-two minus fixed-two
(learnable departure from the short filter), learned-one minus learned-two
(initialization), and learned-one minus lag2 (the preceding recipe versus the
ordinary short filter). Learned-one is rerun from scratch; old endpoints are
historical context, not extra independent seeds.

## Fixed Protocol

`hf_fractional_pride_two_lag.json` keeps the earlier model/corpus hashes and
batch schedules: seeds 41/43/47, 512 updates per arm, Adam 0.001, batch 2,
context 128, two CPU threads, float32 interfaces and K=32/h=1. The base remains
frozen. Training uses 1204 blocks, with 16 diagnostic development blocks and
delayed 120 Pride / 32 Alice endpoint blocks. These reused books may be in
pretraining and are exploratory, not untouched confirmation.

Complete all twelve primary runs and 24 continuation-only updates before
scoring endpoints. Development cannot select duration, checkpoints or
hyperparameters. Report losing arms, moving orders and failed criteria.
No full-model throughput, statistical significance, convergence or general
LLM claim follows from this small study.

Before launch, require exact ordinary/fixed-GL output and gate-gradient parity
at nonzero gates, including real BTF dimensions, noncontiguous inputs and T=1/2
boundaries. Input VJPs use rtol=atol=3e-6. A tiny HF test must preserve exact
gate/named Adam-state equality and interrupted-resume/next-update parity for
all four arms. In the real CPU-f32 study, require exact same-math loss and
common gate receipts, development/endpoint scores and final gate/Adam states.
A mismatch fails the parity criterion; it is not a quality advantage.

The shared driver seals source/runtime/data identities, adapter recipes,
optimizer state and cursor. The inherited one-lag validation helper is also
source-bound. Full unpadded contexts only: no KV cache, persistent history,
packed-document boundaries, mixed precision or resident GPU execution.

## Reproduce

Use existing hash-matching local assets with no downloads and a fresh directory:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 \
  python bindings/st-py/examples/hf_fractional_two_lag_study.py \
  --config bindings/st-py/examples/hf_fractional_pride_two_lag.json \
  --model-dir "$LOCAL_GPT2" --corpus "$LOCAL_PRIDE" \
  --transfer-corpus "$LOCAL_ALICE" --output-dir "$NEW_STUDY_DIRECTORY"
```

Freeze the client/helpers and runtime package throughout the run; use the
same inputs and `--resume` to continue. Completed resume checks sealed
results instead of rescoring. Summarize read-only with
`tools/summarize_wave_gate_long_horizon.py` and its `--plan`, `--results`,
`--journal` and new `--output` paths.

Publish all numeric outcomes, verification receipts, hashes and reproduction
instructions. Keep weights, corpus text, checkpoints, runtime packages and raw
logs local. A protocol, passing preflight or running process is not a completed
pretrained learning result.
