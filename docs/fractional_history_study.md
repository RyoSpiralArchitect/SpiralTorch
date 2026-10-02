# Independent Fractional History In LM Learning

The [completed full-GL comparison](fractional_memory_study.md) left the ordinary
pointwise gate ahead in all seeds on both reused evaluation sets. This follow-up
tests a structural hypothesis: do not tie current-feature gain to the fractional
history gate. It uses the public Rust-backed `FractionalHistoryAdapter`, not a
Python reimplementation of fractional coefficients or derivatives.

## Controls

All arms insert after frozen GPT-2's `transformer.h.0.mlp`, start at identity,
and use ordinary Adam with the same loss and learning rate. History arms share
the residual structure `local + 0.1*tanh(gate)*history(x)`, where
`local = x + 0.1*tanh(local_gate)*x`.

| Arm | History | Parameters (Total / Trainable) |
| --- | --- | ---: |
| Frozen baseline | No adapter, no updates | 0 / 0 |
| `pointwise` | None, one ordinary feature gate | 768 / 768 |
| `ema_learned` | Ordinary finite EMA, learned decay | 1537 / 1537 |
| `history_fixed` | Strictly-past GL, alpha fixed at 0.5 | 1537 / 1536 |
| `history_learned` | Strictly-past GL, learned alpha | 1537 / 1537 |

The ordinary Torch control uses
`(1-d)*sum(d^(k-1)*x[t-k], k=1..31)`, `d=sigmoid(logit_decay)`, initially 0.5.
It is a finite zero-padded EMA, not an infinite recurrence. GL uses the same
31 past positions from a 32-tap Rust kernel, step 1, initial alpha 0.5. Neither
includes the current sample or renormalizes incomplete prefixes. GL's zero-lag
tap and its order derivative are removed in Rust before convolution.

Both gates begin at zero. The local and history gates receive ordinary loss
gradients; the order/decay gradient is initially zero behind the history gate
and becomes active only through that gate. No surrogate updates or dummy
parameters are used. The two learned-history arms match parameter count, **not
filter initialization, mathematical prior, numerical conditioning or compute**.
History-versus-pointwise comparisons also change capacity. This is not a speed
comparison; performance claims require identical mathematical computations.

Primary contrasts are learned GL minus EMA, learned GL minus fixed GL, fixed
GL minus pointwise, and EMA minus pointwise. The first asks whether fractional
history helps beyond ordinary causal mixing; the second isolates the extra
learned order scalar within the same GL adapter. A difference does not by
itself prove a unique geometric advantage.

## Fixed Protocol

`bindings/st-py/examples/hf_fractional_pride_history.json` fixes 3 seeds
(41, 43, 47), 4 active arms, 512 updates per run, batch size 2, context 128,
learning rate 0.001 and CPU float32 with two threads. No dropout, base-weight
updates, auxiliary loss, optimizer geometry or telemetry intervention is added.
Each seed supplies the same minibatches to all arms; zero initialization uses
no adapter RNG. Different seeds vary order and finite-budget training coverage.

The existing model and corpus hashes, 1204 training blocks, 16 diagnostic
development blocks and delayed 120 Pride / 32 Alice evaluation blocks match
the previous protocol. All twelve runs and all 24 continuation-only updates
must finish before endpoint scoring. Development is diagnostic only; it does
not select a checkpoint, hyperparameter or run duration. These books are reused
exploratory data, not pristine confirmation, and may occur in pretraining.
Historical full-GL results are not additional independently rerun arms here.

History resets per unpadded block. There is no incremental KV cache, packed
document boundary handling, mixed precision or resident GPU claim. The native
GL path performs explicit CPU transport; the EMA control uses ordinary Torch.

## Reproduce

Build/install the bindings, then use local, hash-matching assets without download:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 \
  python bindings/st-py/examples/hf_fractional_history_study.py \
  --config bindings/st-py/examples/hf_fractional_pride_history.json \
  --model-dir "$LOCAL_GPT2" --corpus "$LOCAL_PRIDE" \
  --transfer-corpus "$LOCAL_ALICE" --output-dir "$NEW_STUDY_DIRECTORY"
```

Freeze the client/helper files, Python package and native module while running.
Resume with the same inputs and `--resume`; a completed resume verifies sealed
outputs rather than rescoring. The shared driver binds source/runtime/data
hashes, batch schedules, adapter recipe, frozen/trainable mode, Adam state and
cursor. The read-only summary reports per-seed losses and scalar trajectories:

```sh
python tools/summarize_wave_gate_long_horizon.py \
  --plan "$STUDY/plan.json" --results "$STUDY/results.json" \
  --journal "$STUDY/journal.json" --output "$NEW_SUMMARY_JSON"
```

Tests verify the EMA against an independent float64 time-domain sum, GL history
against a dense Torch polynomial at the actual training shape, strict causality,
lane separation, real parameter counts, fixed-order invariance, and all four
tiny-HF training paths with interrupted-versus-uninterrupted exact Adam equality.
Scalar receipts must be finite and continuous and agree with final checkpoints.
These establish wiring and experimental controls, not pretrained quality.

Publish all numeric outcomes, hashes and verification records, including losses.
Keep raw corpora, model weights, checkpoints and raw logs local. A live training
process or preflight receipt is not a completed result.
