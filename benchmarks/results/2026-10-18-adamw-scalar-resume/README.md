# Scalar AdamW State Recovery

Follow-up to the read-only review of PR #2270 at `d487e367c`. The candidate here
is based on `0e8dba67a`; it does not by itself update or approve that older PR.

The shared Rust `AdamWState::validate` contract now accepts `shape=[]` as one
scalar, matching `NdLayout`. It still rejects zero-length dimensions, wrong
rank, missing/extra moments, invalid clocks and an empty parameter collection.
This closes snapshot/readback and restore for resident scalar parameters without
adding a Python/WASM special case or changing optimizer arithmetic.

The shared native/browser test now restores both fresh and nonzero-clock scalar
state, then checks three updates against the Rust candidate and bitwise against
continuous execution. Shape, moments and clocks are checked along the way; a
retained pre-update scalar snapshot must remain unchanged. This is same-device
optimizer-state recovery, not a cross-device or full model checkpoint claim.

The independent local reviewer confirmed that both P2 findings were closed at
source level, and suggested checking restored state before its first update as
well. That P3 coverage gap was addressed with explicit pre-update comparisons of
shape, moment bits, clock and parameter bits. Native and browser checks were
rerun after adding those assertions. The reviewer did not execute GPU tests.

A separate test repair replaces `f32::MAX * 2` poisoning of a frozen gradient
with the existing explicitly guarded divide-by-zero operation. Rejection no
longer depends on the driver's result for an out-of-range multiplication.

## Validation

- Shared CPU AdamW suite: five tests passed, including scalar state negatives.
- Native Metal, runtime opt-in enabled: the shared transactional suite passed,
  including scalar recovery, twelve Torch reference updates, retained snapshots,
  overflow/domain checks, and frozen-gradient atomic rejection.
- Rebuilt browser WebGPU: the same shared suite passed; complete bounded result
  is in `browser.json`. Physical browser hardware execution is unverified.
- Scoped rustfmt and diff whitespace checks passed. The initial build exposed
  a test import-path error; it was corrected before successful native and WASM
  reruns. Initial and successful logs are retained locally.

```sh
cargo +1.99.0 test --locked -p st-kernel-contracts --lib adamw
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo +1.99.0 test --locked \
  -p st-backend-wgpu --lib adamw_native_matches_torch_and_preserves_transactional_history \
  -- --test-threads=1
cargo +1.99.0 build --locked -p st-backend-wgpu --release \
  --target wasm32-unknown-unknown --example resident_parameters_browser
```

Generate web bindings with wasm-bindgen 0.2.129, place `adamw_browser.html`
beside them, and serve only that isolated output directory. The page verifies
the scalar and deterministic-invalid-gradient fields before offering a download.

The older near-one AdamW bias-correction review issue was already corrected in
`f27ca0aae`, an ancestor of this candidate but not of the unchanged PR head.
The original long-language strict gate remains failed; see the separate
[rounding follow-up](../2026-10-18-wide-rounding/README.md). These local fixes do
not retroactively qualify that study, authorize model promotion or substitute
for CI and review of the exact eventual merge head.
