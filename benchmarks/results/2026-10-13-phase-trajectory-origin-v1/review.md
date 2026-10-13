# Independent Read-Only Review

The local reviewer inspected the optional Torch association, the two diagnostic
scripts, shared native/WASM boundary harness, Rust documentation and browser
checks. It did not edit files, execute tests/builds, run a GPU/browser, inspect
the broader deletion set, or independently reproduce numerical artifacts.

The first pass found two P2 issues:

1. Unchecked `zip` and positional case matching could report success for a
   truncated trace or incomplete parameter family.
2. Python `assert` could disappear under optimized execution while the artifact
   still claimed exact legacy replay.

The parent added shared input validation for case identity/order, parameter
names/shapes/counts, exactly 16 ordered revisions and finite complete tensors;
strict pairing; explicit replay errors; and hostile-input tests. The reviewer
then reinspected those changes and the new analytic/negative tests and reported
both findings closed with no new findings in that bounded source scope.

The parent ran the ten tests normally and with `-O`, reran both complete CPU
diagnostics into new files, and verified that the diagnostic numerical values
were unchanged. The parent also ran the native and actual browser residual
harnesses. These runtime/artifact checks are parent evidence, not independent
reviewer verification. Source hashes, raw hashes and exit codes are in
`validation.json`.

The review explicitly noted that forward outputs distinguish association
(`+1`/`-1` versus zero), while the equal input and projection-bias VJPs of 2
verify preserved gradient routes rather than association discrimination.

No statement here promotes the original projective-v1 result, removes its
mixed-runtime query failure, establishes broad model accuracy, or substitutes
for CI. The public evidence documents the alternate reference as a diagnosis.
