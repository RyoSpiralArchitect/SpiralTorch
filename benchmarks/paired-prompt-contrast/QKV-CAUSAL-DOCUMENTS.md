# Ragged Full-Answer QKV Updates

Status: Rust implementation, CPU binding check, one CPU extension build and
isolated Python execution verified. No pretrained-model study, GPU run, cache
deletion, push or publication. This extends the source-build
[causal-window API](QKV-CAUSAL-WINDOWS.md), not a published-wheel claim.

## Contract

`NativeCausalLmQkvAdamW::document_loss_gradients` and `step_documents` preserve
document order and batch only adjacent equal-length windows. They reuse native
`equal_width_groups`, the compensated weighted-gradient reducer, shifted causal
objective, decoder backward and atomic QKV AdamW application. The private
answer reducer was renamed `WeightedGradientMean`; its arithmetic and the
answer-only API did not change. Python reuses bounded document transport and
supplies native admission, without implementing reduction or optimizer logic.

Every group contributes its **selected-token count**, not its sequence count.
The resulting whole-batch mean is clipped once and AdamW advances once. Only
one selected QKV weight matrix learns; other weights, LoRA and geometry stay
frozen. Raw gradient queries remain unclipped and non-mutating. Zero-rate calls
preserve weights, moments and the optimizer clock. Checkpoint and inference
snapshot ownership use the existing native contracts.

The full roster is admitted structurally before Python copies token/label
values; all IDs, labels and actual sparse group logit demands are validated
before decoding. Limits are 4,096 documents, 1,024 groups, native context and
checkpoint bounds, and the existing logical QKV work bound. Each document must
have at least two IDs and one selected shifted target. All-masked documents are
rejected independently of grouping. No implicit padding, EOS, truncation,
cross-document context, reordering, scheduling or persistent data cursor exists.
Keep order, max_rows, masks and objective policy for exact continuation;
different groupings may round differently.

## Verification

All four bounded invocations succeeded on their first execution, without guard
triggers or source changes during execution:

| Check | Result | Wall time |
| --- | --- | --- |
| Offline native decoder tests | 226 passed; 0 failed/ignored | 18.34 s; test bodies 2.09 s |
| CPU Python-binding check | Passed | 15.81 s |
| Offline CPU release extension build | Passed | 113.16 s |
| Isolated Python runtime | 74 passed; no errors/failures/skips | 1.53 s; test bodies 0.305 s |

The build used the existing scoped target, two jobs, incremental output disabled,
a 4 GiB free-space floor and 600-second cap. Other caps were 180 seconds for
tests/check and 120 seconds for Python. The extension was copied into a private
83-file source-package overlay, not installed. The prior saved extension stayed
byte-identical. Import attempts and loaded modules for Torch, NumPy,
Transformers, Python safetensors and Accelerate were all zero.

Four new Rust tests check token-weighted gradients against individual windows
under ordinary/key-lag attention and smoothing off/on, including an unequal
target-count fixture that rejects document averaging. They cover one shared
clip and exact expected AdamW weights, a single update clock, zero-rate freeze,
exact restored continuation, unchanged single-group window behavior, sparse
group budgets, invalid tails and no mutation on rejection. Two new Python tests
cover completion labels, ragged gradients, reports, restore/cache integration,
pre-copy admission, strict labels/scalars and rectangular-route equality.

This verifies numerical wiring and state transitions on synthetic fixtures,
not improved training quality, real-model throughput or GPU behavior.

## Evidence

Local evidence directory: `2026-11-10-qkv-causal-documents-v1` in the SpiralTorch
validation store. Only documentation was added after runtime verification.

| Artifact | SHA-256 |
| --- | --- |
| Tested source inventory | `867d9aceff3fd34cd9114da091dfda0939c36663a197469ab71589d162406c8b` |
| Native log | `8d2879cea6a0f2879fcd8e7f8b7b0dfdc7ac3c100ca096a4603dc75ca1f2356b` |
| Binding check log | `7e2452aa38b0b935c7aa7fc296b1cf91c95556e8c0cc4da6596d8eb2f29b9e7a` |
| Extension build log | `f94e1c4089d128619a012a3bc99458eb065e66c09a475a56b7f5b41fa311723b` |
| Extension | `99f5d9456449017b166b929849083dcf43387e4846feca8306de8b56705d23ff` |
| Package manifest | `ce903a26ebb2164540dbefea3c3190cad7db79926ef37da177343eb53cbb9d9a` |
| Python runtime log | `ecfa8d1fba684329bd1bfd6f1f8fad85bec89ca9edfdcd32b0b5997abd419d3c` |
