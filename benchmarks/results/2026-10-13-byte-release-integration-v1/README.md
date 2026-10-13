# Byte Client Release Integration: Not Yet Release-Ready

The 0.4.30 Python candidate exposes nine Rust-owned byte-LM classes. This
increment integrates main, repairs CI lint/rustdoc issues, and qualifies the
same six fixed learning cases through native Metal, browser WASM/WebGPU and an
installed Python wheel. It **does not publish or approve a release**.

## Qualified Sources

- Core: `7fb47f267e21497bdecd4e01e7423be12192da53`.
- Python candidate: `8b102c58f352beb6da2183bd7cbd97bc9e0fc9fd`.
- Wheel SHA-256: `e60a26822601d333d53ddc117243bb480349e37e13229ac303eebda0d8fe3107`.
- Fixture SHA-256: `6416cac7c885ae75cd831c5cbed0701595d1ffeaf685e01f6be44a6de93391a7`.

Each runtime qualifies 6/6, with 96 learning updates and 720 geometry-step
pullbacks. Python additionally checks 54 resumed updates against uninterrupted
execution, starting from revision seven. All numerical gates remain unchanged.
The browser run is fresh, not a relabelled earlier report.

The source-level Python parity runner used the frozen earlier native/browser
bundle as its reference. The separately executed fresh native/browser reports
were independently revalidated against the same CPU fixture.

Local validation: 107 kernel tests; 328 WGPU tests with one explicitly ignored
timestamp experiment; eight Python byte-client tests with one CPU-only artifact
gate skipped. Native/WASM Clippy and strict st-nn rustdoc pass. The corpus-tool
merge preserves all versioned controls: 64 tests pass with CPU Torch available,
plus 21 verifier tests. These checks are not a blanket CI-success claim.

## Review Still Gates Publication

The first kernel review found four recoverable numerical failures. The core
commit above addresses primitive projective-product cancellation, shared router
reduction, Fisher root underflow/projection, and wide-gap wave batch reduction.

A follow-up found two further counterexamples, reproduced on the candidate:

1. Nested router cancellation: components `[0,2^120]` repeated five times,
   equal logits, cotangents `[2^124,1,2^-124,-1,-2^124]`. The expected router
   gradient is `[-0.015625,+0.015625]`, but CPU/GPU return zero.
2. Nearby Fisher rows: logits `[0,0,e,-e]`, `e=2^-30`, gain `2^60`,
   causal cotangent `e`. The expected score is approximately -1; GPU returns
   zero. Final packing cannot recover a difference erased by f32 exponentiation.

Local, uncommitted follow-up work passes 108 CPU tests with an adaptive partial
expansion and six Fisher GPU tests with a wide exponential series. It is **not**
part of the candidate above or its three-runtime qualification. GPU nested
reduction remains open; a fixed extra compensation level is not a general fix.
The smaller-deviation limit of Wide exponent alignment also remains relevant.
Do not merge, tag or upload based only on the successful ordinary fixtures.

## Evidence And Reproduction

[Validation](validation.json) binds sources, candidate hashes, retained failed
attempts and review limits. [Matched results](matched-results.json) contain the
fresh native/browser scalar outcomes; [Python parity](python-parity.json)
contains all six installed-wheel outcomes. Raw vectors, checkpoints, binaries,
logs and screenshots remain local. The original v1 failures and earlier v2
records are unchanged.

Use the commands in the [frozen v2 protocol](../2026-10-13-matched-geometry-learning-v2/README.md)
at the qualified source above, with fresh output paths. Build/install the Python
wheel, then run `tools/validate_python_byte_decoder.py` with `--wheel` so native
extension identity is verified. The [Python guide](../../../docs/python/byte-language-models.md)
documents the public API, explicit readbacks and checkpoint ownership.

Rust 1.99.0 was used for tests/Clippy/rustdoc; Rust 1.97.0 for optimized native,
WASM and Python builds; Python 3.12.6 and Apple M4/Metal for local execution;
wasm-bindgen 0.2.129 for browser glue. No fresh Linux/Windows/universal2 wheel
qualification, CPU-only wheel execution, language-quality gain or speed claim
is made. Official signed wheels from the final merged/tagged source are still
required before PyPI publication.
