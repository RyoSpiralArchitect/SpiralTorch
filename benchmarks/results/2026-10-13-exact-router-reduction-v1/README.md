# Exact Router Reduction: Candidate Follow-Up

This increment closes the nested mixture-router cancellation counterexample
recorded in the [release-candidate review](../2026-10-13-byte-release-integration-v1/README.md).
It does **not** approve a release or relabel the earlier wheel/browser results.

## Change And Scope

Shared and querywise router pullbacks now sum the primitive polynomial
`seed * weight_k * weight_j * (component_k - component_j)` without rounding
the difference, intermediate products, per-position sums, or final weight.
The two signed products enter a bounded integer accumulator separately; only
the final output is rounded to f32, nearest/ties-even.

For stored finite f32 operands, the smallest four-factor quantum is `2^-596`.
Both weights are in `[0,1]`, and checked u32 input lengths bound the absolute
sum below `2^289`. The signed `[-608,320)` bit window covers both tails and
intermediate sums. CPU uses u128 product decomposition; WGSL uses eight
base-4096 digits and u32 carry/borrow. Neither needs shader-f64.

This is exact accumulation of **stored operands**, not exact real-valued
softmax, universal transcendental accuracy, a speed claim, or language-model
quality evidence. Other geometric reductions do not inherit this guarantee.

## Checks

- The five-position reproducer returns `[-0.015625,+0.015625]`, not zero.
- All 120 position permutations pass in shared and querywise GPU routes.
- 33 direct GPU polynomial cases cover nested tails, signed underflow,
  normal/subnormal/overflow rounding midpoints, and 96-bit primitive products.
- 112 CPU kernel tests and 331 native WGPU tests pass. One existing explicit
  timestamp experiment remains ignored; no correctness test was skipped.
- Native and WASM Clippy, workspace formatting, and strict kernel rustdoc pass.
- The frozen six-case native learning suite passes all 96 updates, with the
  existing `0.002` relative gate unchanged. Full scalar outcomes are retained.
- Independent read-only source review found no actionable issue in the exact
  router implementation. Device checks were run separately by the parent.

[Validation](validation.json) pins the code commit, source hashes, retained
attempts and test scopes. [Native results](native-results.json) contain all
six cases. Raw tensors, binaries and logs remain local; checksums are public.

## Reproduce

At the source commit in `validation.json`, using fresh output paths:

```sh
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo +1.99.0 test --locked \
  -p st-kernel-contracts -p st-backend-wgpu --lib -- --test-threads=1
cargo +1.99.0 clippy --locked -p st-kernel-contracts -p st-backend-wgpu \
  --all-targets -- -D warnings
cargo +1.99.0 clippy --locked -p st-backend-wgpu \
  --target wasm32-unknown-unknown --lib -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo +1.99.0 doc --locked -p st-kernel-contracts --no-deps
```

The native learning run uses `resident_byte_flat_metric --module-projective`
and the unchanged [frozen v2 fixture/protocol](../2026-10-13-matched-geometry-learning-v2/README.md).
Its output is independently checked by `tools/validate_module_projective_results.py`.
Only a native learning run and WASM compile were repeated in this increment,
not an installed-wheel or browser qualification.

Nearby Fisher/Wide numerical-boundary work remains deferred. No merge, tag,
official release wheels or PyPI upload is authorized by this result alone.
