# Fractional Memory In Language-Model Learning

This experiment uses the public [Rust-backed fractional adapter](fractional_learning.md)
to separate a pointwise feature gate, a fixed finite-history difference and a
learned fractional order. It is an explicit client of the same operator tested
through Python and WASM, not another Python implementation of GL mathematics.

## Controls

All active arms insert after `transformer.h.0.mlp`, start at identity with a zero
per-feature gate and keep the GPT-2 base frozen. The ordinary Adam loss gradient
and learning rate are unchanged; there is no chart preconditioner, auxiliary
loss, telemetry feedback or additional geometric mechanism in this comparison.

| Condition | Residual | Adapter Parameters (Total / Trainable) |
| --- | --- | ---: |
| Frozen baseline | None | 0 / 0 |
| Pointwise | `0.1 * tanh(gate) * x` | 768 / 768 |
| Fixed order | `0.1 * tanh(gate) * GL(x, alpha=0.5)` | 769 / 768 |
| Learned order | `0.1 * tanh(gate) * GL(x, alpha=exp(log_alpha))` | 769 / 769 |

The two GL arms use the **same** imported `FractionalMemoryAdapter`, `step=1`
and 32 causal coefficients. Only `log_alpha.requires_grad` differs. A study
recipe wrapper prevents a saved fixed-order state from being silently loaded
as a learned-order run or vice versa. The pointwise control has no unused alpha.

Primary contrasts are fixed GL minus pointwise (history) and learned GL minus
fixed GL (order adaptation), plus learned GL minus pointwise. Learning alpha
adds one trainable scalar. The finite-history map can change activation gain
and correlations as well as temporal response, so a better score would not
prove a unique geometric advantage. This is not parameter- or compute-matched
in every respect; report the budgets rather than padding controls with dummy
parameters.

## Frozen Protocol

`bindings/st-py/examples/hf_fractional_pride_memory.json` fixes three minibatch
seeds (41, 43, 47), 512 updates per active arm, batch size 2, context length 128,
learning rate 0.001 and initial alpha 0.5. The zero gates use no random adapter
initialization. Full unpadded blocks reset GL history at their first token;
there is no KV cache or state carried between blocks. CPU float32 host execution
and transfer costs are explicit, not a throughput comparison.

The shared long-horizon driver uses the same local model/corpus hashes and
paired minibatches as the earlier studies. Pride's first 90% supplies training,
16 selected tail blocks supply development diagnostics, and the remaining 120
tail blocks plus 32 Alice blocks are evaluated only after all nine runs and
exact next-update checks complete. Development does not select a checkpoint,
alpha, kernel length or run duration. These books have been inspected in prior
experiments and may be in model pretraining; they are reused exploratory data,
not untouched confirmation or a significance claim.

Checkpoints preserve the adapter's recipe, learned/frozen mode, optimizer,
update cursor, paired batch history and scalar trajectory. Fixed alpha must
remain unchanged with no gradient. Learned alpha starts with zero gradient
behind the zero gate; subsequent actual gradients and updates are recorded,
not artificially injected. The read-only summary checks mode, parameter counts,
trajectory continuity, positive finite alpha/log-alpha consistency and endpoint
agreement, and reports every per-seed contrast including losing outcomes.

## Run And Verify

Build/install the native Python bindings and provide the existing local assets:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 \
  python bindings/st-py/examples/hf_fractional_memory_study.py \
  --config bindings/st-py/examples/hf_fractional_pride_memory.json \
  --model-dir "$LOCAL_GPT2" --corpus "$LOCAL_PRIDE" \
  --transfer-corpus "$LOCAL_ALICE" --output-dir "$NEW_STUDY_DIRECTORY"
```

Keep the client files, Python package and native module frozen while running.
Use `--resume` with exactly those same inputs after an interruption. Completed
resume verifies sealed outputs rather than rescoring endpoints. A successful
process and saved-checkpoint verification are separate from summary validation:

```sh
python tools/summarize_wave_gate_long_horizon.py \
  --plan "$STUDY/plan.json" --results "$STUDY/results.json" \
  --journal "$STUDY/journal.json" --output "$NEW_SUMMARY_JSON"
```

Tiny-HF tests cover all three controls, actual order gradients, fixed-order
invariance, interrupted-versus-uninterrupted adapter/Adam equality, held-back
endpoint scoring and frozen-base preservation. Those tests establish wiring
and restart correctness; pretrained quality remains an experimental result,
not a premise of this protocol. Publish numeric results and verification hashes,
not model weights, book text, checkpoints or raw logs.
