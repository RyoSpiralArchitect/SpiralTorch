# Wave Time/Batch Reduction: Three-Client Follow-Up

This follow-up carries the Wave repair into the 0.4.30 Python candidate and
the standalone Rust/WASM byte decoder. It does not publish a package or
relabel the [earlier Fisher results](../2026-10-13-fisher-close-chart-v1/README.md).

## What Changed

Reducing time steps into a rounded batch total can lose a contribution before
another batch cancels the large terms. A fixed two-level correction also
loses nested small signals. CPU now retains an adaptive f64 expansion of
per-time contractions; GPU retains all three stored Wide components as
scaled f32 terms, across both time and batch. Direct final rounding keeps
tiny midpoint tails, including signed subnormal and overflow boundaries.

The guarantee concerns the sum of stored contractions, not exact primitive
products, nonlinear charts, rotations, or the recurrent adjoint. The GPU
scratch grows to `10N+8P` 32-bit words, where `N=B*T*2P`. Adaptive reduction
can be quadratic in adversarial retained-part counts. This is a numerical
repair, not a speed improvement.

The shared `CausalWaveSpec` rejects time axes above 1,048,576 before allocation.
This closes an independently identified i32 exponent-range failure for a
20-million-step reverse scan. The [bounded read-only review](review.md)
records a conservative range argument for the current arithmetic. Changing
Wide or the recurrence requires rechecking that bound; it is not a claim of
exact BPTT. Longer documents require explicit stateful chunks.

## Evidence

[Source pins, hashes, gates and retained failures](validation.json) bind:

- 123 CPU kernel tests and 340 actual native WGPU tests, with one preexisting
  explicit timestamp experiment ignored. Native/WASM Clippy, strict kernel
  rustdoc and workspace formatting pass.
- 1,058 native shader cases, including 1,024 independent integer-oracle
  random cases and signed rounding boundaries. These are native tests, not
  a claim of running all 1,058 cases in the browser.
- [Native](native-wave.json) and actual [browser](browser-wave.json) primitive
  probes: ten PyTorch cases, nine chart controls, 36 batch-cancellation
  cases, three new time/batch cases, range/overflow guards and 16 updates.
- Fresh [native](native-results.json), [browser](browser-results.json) and
  installed [Python](python-results.json) matched learning: six frozen v2
  cases, 96 updates and 720 geometry-step VJPs per runtime. The relative gate
  remains 0.002. Python also checks 54 exact resumed updates, including
  parameters, derivatives, embeddings and router weights, not just loss.
- Eight installed-wheel API tests pass. The CPU-only artifact test is
  skipped for this WGPU wheel and requires a separate artifact.
- The [wheel-packaged example](packaged-example.json) completes 16 updates
  and exact prediction resume using the documented installed-module entry.

The three repaired primitive gradients have final f32 bits `519045`,
`3261257`, and `1061490577`, identical on CPU, native GPU and browser GPU.
The last corresponds to approximately 0.7697688, previously zero.
Failed implementations and the automatic-layout probe mismatch remain in
the local receipts. The final probe uses the production explicit layout;
this does not qualify general automatic-layout behavior in the backend.

## Reproduce And Release Boundary

Use the exact core/client commits in validation.json. Reproduce native
kernel and primitive checks with Rust 1.99.0:

```sh
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo +1.99.0 test --locked \
  -p st-kernel-contracts -p st-backend-wgpu --lib -- --test-threads=1
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo +1.99.0 test --locked \
  -p st-backend-wgpu --test causal_zspace_wave -- --test-threads=1 --nocapture
```

The [Wave guide](../../../docs/causal_zspace_wave.md) gives the actual browser
probe. For full-model runs use the unchanged
[v2 fixture protocol](../2026-10-13-matched-geometry-learning-v2/README.md).
Build optimized native/WASM clients with Rust 1.97.0. Build the candidate
wheel using `MACOSX_DEPLOYMENT_TARGET=14.0`, maturin `--release --locked`,
and `--features logic,kdsl`; install it into an isolated environment.
Run `tools/validate_python_byte_decoder.py` against the fresh raw native and
browser directory with `--wheel` pointing to that exact installed artifact.
Only the `tools` sibling path is needed, never the binding-source directory.
The verifier checks distribution/extension identity against the wheel hash.

Raw tensors, binaries and failed attempts stay local; all scalar conditions,
validation records and hashes are public. This is macOS arm64 WGPU candidate
coverage, not fresh Linux/Windows/universal2 or CPU-only wheel coverage.
It demonstrates mechanics, not language quality, a winning geometry or speed.
The scoped Wave review is closed; full stacked-core review, current-head CI
and official platform release gates remain separate. No main merge, tag or
PyPI upload is asserted by this record.
