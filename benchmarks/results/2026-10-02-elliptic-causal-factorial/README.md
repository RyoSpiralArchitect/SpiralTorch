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

After the parent CI optional-PyTorch import fix was merged, 91 tests pass with no
skips. The running study still uses its original frozen client/native package;
the fix only adds the missing no-Torch placeholder and does not modify that
runtime. The initial preflight hashes remain the executed-source evidence.

The study has been launched from `03cee5fb8c65cffefaf4d60da4c5691a32dc6175`.
`plan.json.gz` is a lossless copy of its fixed plan, including every paired batch
schedule. Study ID: `2d5860c91905b26b6b29e729b94cfa9c710bb11ade2e4125f827c778a89c3954`.
This is launch evidence only, not completion or endpoint results.

`preflight.json` binds sources, the native extension and the retained local test
log by SHA-256. `SHA256SUMS` checks the public files. Runtime, raw logs, source
corpora, model and checkpoints remain local. A running job or this protocol file
must not be interpreted as twelve completed runs; completion evidence will be
added only after all final checkpoints and sealed endpoints exist.
