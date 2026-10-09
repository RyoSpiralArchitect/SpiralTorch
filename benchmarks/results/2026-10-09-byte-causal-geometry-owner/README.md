# Causal geometry inside one byte-model owner

The Rust byte decoder now owns the embedding tables, causal geometry encoder,
residual blocks and output head in one revision/SGD transaction. This record is
about **full-model numerical correctness and tiny synthetic learning**, not
language quality, fine-tuning superiority or speed.

## What is connected

The summed byte/position embedding feeds both the ordinary residual path and a
tokenwise learned projection. A causal complex-state wave maps its drive to an
interior ball chart. Each block/head has a learned softplus gain for the genuine
Poincare squared-distance bias. Both coordinate roles and all consuming blocks
contribute before wave BPTT and projection backward. That embedding cotangent
is added to the ordinary path before either embedding scatter and the final
whole-model guard. No second optimizer or host activation readback is introduced.

Each selected document window resets the recurrent state to zero. Curvature is
fixed; head gains are shared across positions, not a context-conditioned router.
This is not full-model streaming or a KV-cache contract. Raw gain zero is not an
off switch; the ordinary plan omits geometry entirely.

## Frozen numerical reference

The existing independent CPU-f32 PyTorch fixture is retained unchanged:
`ef9bc46892fce5591dc4b3f0ebd527a76b493b160faeb05887ff61daec389874`.
It contains the 18-parameter plain and 31-parameter Topos/external-bias models.

The new isolated-Python generator imports only the independent Torch reference
functions, never SpiralTorch. Its two additional models contain 23 and 37
parameter tensors, including projection weight/bias, decay, phase and one gain
vector per block. Fixture SHA-256:
`631abe16ab140c6fcfee65d072963ba3ed3532a2701606c770b50241f1b5af4b`.

Every parameter VJP, logits, summed-embedding VJP and supplied external-bias VJP
uses the unchanged elementwise gate `3e-6 + 5e-5 * abs(reference)`. Each geometry
parameter VJP additionally must be nonzero and have relative L2 error <=0.002.
The reference norm must exceed 1e-8, so this check cannot pass on a powerless
near-zero control. The same gates apply to geometry CE gradients at all 16
updates, while every full-model parameter is compared after every update.
Losses, receipts and parameter/gradient snapshots are observed only after all
updates have been queued. Fixtures and tolerances were not regenerated/relaxed
to make the implementation pass.

## Controls against a false connection

- A one-block fixture has explicitly zero Q/K weights and biases, verified from
  the actual parameter arrays, so its initial score structure comes from the
  metric rather than a hidden Q/K contribution.
- A geometry-off model retains the same ordinary parameters and must produce
  a different prediction. The measured prediction difference is also compared
  relatively with the independent reference, not just checked for nonzero size.
- A detached control receives the same numerical geometric pair biases as
  caller-owned inputs. Its logits match the coupled model, but the embedding
  pullback differs. That difference is compared relatively, using actual VJPs
  from both models, to detect a severed metric-to-embedding path.
- Three negative controls plus a positive control test the relative checker.
- Prefix extension, suffix edits, zero future-byte gradients and other-document
  isolation are checked for every model, including geometry-enabled cases.
- Foreign/stale tapes, invalid final bias preflight, good/bad/good backwards,
  retained gradients, atomic rejection and recovery use the complete model.
- Native unit fault injection visits every candidate in 12-slot plain and
  18-slot geometry-enabled owners. Late byte/position scatter overflow is
  rejected at zero and nonzero rates, including all geometry gradients.

These tests use four configurations, not a matched language-quality ablation.
The loss reductions concern only the fixed synthetic byte windows. Geometry-off
and detached comparisons are fixed-weight correctness controls, not claims
about which trained architecture is better.

## Records

- [Native result](native.json)
- [Actual browser WASM/WebGPU result](browser.json)
- [Validation commands, source/artifact hashes and scope](validation.json)
- [Architecture and runnable commands](../../../docs/resident_byte_decoder.md)

Raw logs and tested native/WASM artifacts remain local. The repository contains
results, verification records and reproducibility hashes. No private corpus,
pretrained weights, credentials, release bump or full-model Python facade is included.

An additional strict `st-nn` all-target Clippy run did not pass: it reported 40
library/library-test findings across 22 files that are byte-identical to the
base revision. No lint allowance or unrelated repair was added. These existing
findings prevent a complete all-target lint result; passing learning checks do
not imply that lint passed. The manifest records the failed command and the
unchanged-file hashes separately from numerical verification.
