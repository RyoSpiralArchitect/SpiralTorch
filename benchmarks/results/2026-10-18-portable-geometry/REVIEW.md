# Independent Local Review

Read-only independent review covered the Rust artifact, Python/WASM bindings,
Python export/reconstruction, registrations and regression tests. Final source
hashes are in `validation.json`. No reviewer edits were applied automatically.

Two P2 findings were reproduced and fixed:

1. Combined tensor-name checks allowed a parameter to become a buffer during
   export and then a parameter during reconstruction. The artifact now records
   each tensor role; Rust checks it against the concrete built-in recipe.
2. Positive finite fractional order did not prove scale/coefficient
   representability. `st-frac::learning::FractionalGlKernel::validate_recipe`
   uses the actual native coefficient builders with bounded artifact tap counts.
   Normalized and unconditional zero-map cases retain their existing semantics.

Final independent disposition: **no new findings; both previous findings closed**.
The reviewer independently ran all 27 artifact tests against the rebuilt wheel,
matched both fixture IDs and roundtrip identity, and probed the aggregate tap
boundary (1,048,576 accepted, 1,048,577 rejected) and angular-overflow rejection.

The reviewer did not execute WASM. Parent verification separately built and ran
the WASM/Node fixture and corruption checks, ran all 58 Python artifact/stack
tests, five Rust artifact tests and 23 fractional-learning regressions. This is
bounded numerical/lifecycle verification, not a language-quality endorsement.
