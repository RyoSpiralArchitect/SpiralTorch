# Fractional Memory Study

**Status: running.** This is the frozen launch/preflight record, not evidence
that all runs completed or that fractional geometry improves language quality.
The [protocol](../../../docs/fractional_memory_study.md) compares pointwise,
fixed-order GL and learned-order GL against a shared no-adapter frozen baseline.

Three seeds, three active arms, 512 updates per run: 4608 primary updates and
18 additional continuation-only updates are planned. The Rust-backed adapter
uses 32 causal GL coefficients, step 1 and initial alpha 0.5. CPU-f32 GPT-2
stays frozen; only the adapter's feature gates and, in the learned-order arm,
its one order scalar are trained. No second geometric mechanism is combined.

The controls have explicit total/trainable adapter parameter counts:
pointwise 768/768, fixed GL 769/768 and learned GL 769/769. Each seed uses
identical active-arm minibatches. Seeds vary shuffle order and therefore the
finite-budget subset of the 1204 training blocks, not random adapter weights.
Each sequence is a complete unpadded 128-token block with reset GL history.

The study must finish all runs and exact adapter/Adam next-update checks before
scoring its 120 reserved-within-study Pride tail blocks and 32 Alice blocks.
These are previously inspected exploratory books, not untouched confirmation.
Development does not select alpha, kernel length, checkpoint or run duration.
No parameter-matched, compute-matched, speed or significance claim is made.

`plan.json.gz` preserves the original plan bytes (gzip encoded); `preflight.json`
records 218 passed Python tests, the 67-file frozen package/native manifest hash,
the four frozen client/config files and source hashes. Source revision:
`21121092b6e5ecbe7ba4464046f1a02d7021d3a4`.
Tiny-HF tests establish actual gradients, fixed-order invariance, three-arm
interruption/restart equality and endpoint gating, not pretrained quality.

After completion, append numeric outcomes and separate saved-state/process
verification. Do not rewrite this launch plan or reinterpret the historical
preflight's `running` status as a completion record. Models, book text,
checkpoints, packages and raw logs remain local.
