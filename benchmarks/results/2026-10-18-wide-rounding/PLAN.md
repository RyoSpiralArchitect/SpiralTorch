# Shared Wide Conversion Follow-Up

Baseline: `a3d1164c79b19da396ad00395698cbf8222ac3ce`. Production shader SHA-256:
`b8b008913952edc7c57c7c1cd0e4915b98d83d660bfad0d117feb8bf36cfb280`.

The preceding language VJP review retained an open midpoint witness: the last
causal row of values `[4, 2^-22, 2^-47, 0]` should round upward, but returns one.
The conversion currently ignores retained lower components for normal outputs.
This follow-up repairs conversion of the retained normalized representation,
not all arithmetic that produced it and not general IEEE-f64 computation.

The integer-oracle fixture and shared native/browser probe were added before
changing the production shader. Baseline Metal execution fails 582 of 7,042
synthetic boundary/random inputs and four of twelve actual Attention midpoint
conditions. The fixture includes sign, even/odd ties, powers of two, subnormal
and overflow boundaries, tiny retained terms and saturated extended exponents.
Its generator uses integers, not a double-rounded float64 oracle.

Before running the long learning follow-up, freeze these requirements:

- Repair only shared `wide_float` conversion. Do not mask Key gradients,
  change accumulation defaults, optimizer controls, corpus, seeds or tolerances.
- Require exact output bits for every fixture and every Attention midpoint
  check on native Metal and browser WebGPU. Keep before/after reports, including
  failures. Do not infer physical browser hardware from a WebGPU success.
- Run the existing affected backend regression suites with GPU opt-in enabled.
- Repeat the same seed11/width16 ordinary, learned-Poincare and frozen-Poincare
  1,024-update request from the prior study. Keep the exact request hash
  `b0a612eec9aeb2f890f1d37c7f4edc50acedd5d898c456cadc3c1c717849d29c`
  and original Torch reference hash
  `96d37dd50825679d52687bb699ab8bcbeb4424af881d75ecabd566bcccc0090b`.
- Use the existing strict numerical gate unchanged. Compare both to that fixed
  reference and to the immediately preceding range-repair result, not just a
  more favorable older baseline. Report remaining failures even if loss agrees.
- Treat hashes as byte identities, not attestation. Raw corpus, parameters,
  optimizer checkpoints, binaries and full model outputs stay private; publish
  selected metrics, verification records, source hashes and reproduction steps.

No quality win, speed win, universal correctly-rounded arithmetic or promotion
is implied by passing these checks. A long gate failure remains a failure.
