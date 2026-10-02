# Causal Elliptic Factorial Study: Preflight

This directory initially records **preflight verification, not completed
pretrained-model results**. The fixed offline protocol is documented in
[elliptic_causal_study.md](../../../docs/elliptic_causal_study.md).

The four arms cross tangent/elliptic features with pointwise/causal mixing.
The planned 12 runs share seeds 41/43/47, initial parameters and minibatch order;
each trains 8450 adapter parameters for 512 updates on a frozen local GPT-2.
Pride/Alice evaluation sets are reused exploratory benchmarks. No speed,
significance, resident-GPU or quality-advantage claim is made.

Preflight: 90 Python tests passed with zero skips/failures, including ordinary
Torch attention versus native Rust forward/VJP, actual tiny-HF loss gradients,
paired initialization, RNG isolation and exact interrupted Adam continuation.
Five added summary cases cover the factorial contrasts and rejection of unpaired
initializations, parameter counts and incomplete designs. The previous nonlinear
study's summary reproduces byte for byte under the extended summarizer.

`preflight.json` binds sources, the native extension and the retained local test
log by SHA-256. `SHA256SUMS` checks the public files. Runtime, raw logs, source
corpora, model and checkpoints remain local. A running job or this protocol file
must not be interpreted as twelve completed runs; completion evidence will be
added only after all final checkpoints and sealed endpoints exist.
