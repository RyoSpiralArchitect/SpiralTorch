# Anchored Elliptic Training: Launch Record

**Training has been launched; there are no completed study outcomes here yet.**
Do not treat preflight success, a live process, or partial development losses as
language-quality evidence.

The [fixed protocol](../../../docs/elliptic_anchored_study.md) compares ordinary
tangent and Rust nonlinear elliptic features, each with a learned fixed-anchor
or causal-context correction. All four arms have 8451 parameters. Seeds 41/43/47
each receive 512 updates on local frozen float32 GPT-2, batch 2, context 128.
All projection/readout/gate tensors and minibatches are paired within each seed.
The planned total is 6144 primary updates plus 24 continuation-only checks.

Source was frozen at `af1c10c39a97ca862568dd3e3fb2259efa0e5ced` before launch.
`plan.json.gz` contains the original, unmodified hash-sealed plan. The study ID
is `229ccd7afba7893e7e18c7f354249957e977381afac80938c4c36cdcfd4b3ca8`.
Models, corpus text, packages, checkpoints and raw logs remain local.

## Preflight

165 tests passed, zero skipped. Checks include training-shape forward/VJP parity,
all-four-arm initialization, actual tiny-HF loss/interrupt/resume, frozen base,
full Adam continuation, schema isolation, invalid-evidence rejection and portable
old-summary reproduction. The native and Python bridge hashes match the previous
anchored operator validation. This is wiring verification, not the full study.

Token data/splits, all batch schedules, base-model/configuration hashes, runtime
versions and shared training-loop/helper sources match the earlier gated study.
The gated arms are retrained under the current package; any exact historical
replays are reproducibility evidence rather than independent confirmations.

`preflight.json` records validation and hashes. `SHA256SUMS` covers these public
launch artifacts. Future completion must verify every final checkpoint and
continuation, process termination and frozen endpoint results before appending
the complete numeric outcomes. This launch record does not assert those gates.

## Interpretation

The primary contrast is anchored elliptic minus anchored tangent. Also report
geometry within gated context, anchor minus context for each map, and the paired
interaction. Keep every losing seed and the frozen baseline. Pride and Alice have
already informed this design, so the results will be exploratory, not pristine
confirmation. No speed claim is made: mathematically equivalent PyTorch/native
performance comparisons are a separate experiment.

Use the offline command in the protocol. Resume only with unchanged frozen
source/package/data/recipe. Do not change knobs from interim development losses;
all endpoints remain locked until the twelve runs and continuation checks pass.
