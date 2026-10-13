# Independent Read-Only Review

A local independent reviewer inspected the versioned generator, shared Rust
native/WASM qualification, query pullbacks, publisher, browser page and tests.
It did not edit files, execute tests/builds, run a GPU/browser or independently
reproduce artifact numbers. Unrelated tracked deletions were outside scope.

The review found and the parent fixed these issues:

1. A finite stopped-parameter oracle mismatch could abort and discard the
   already-collected six-case report. V2 now retains the mismatch and fails the
   qualification, while continuing the other paired case.
2. The follow-up found the same retention gap in an unrelated parameter route
   that should be unchanged by StopCoordinates. The actual caller now uses a
   tested retaining route checker for v2; legacy v1 behavior stays unchanged.
3. Arbitrary objects in imported adapter metadata could reach public output.
   The v2 publisher explicitly omits that field instead of copying it.
4. Checking causality summary errors only for finiteness lacked an acceptance
   bound. V2 now records private raw witnesses; the publisher independently
   applies the original per-element absolute/relative tolerance, checks the
   reported maxima and omits witness arrays from public output.

The final source-only recheck reported all findings closed with no new findings
in that bounded scope. Regression tests were inspected, not run by the reviewer.

The parent ran the native/WASM cases and Python/Rust tests. A deliberate finite
stopped-oracle corruption exits with failure while retaining all six cases,
all 96 updates and both paired observations. An earlier negative fixture
serialized through JavaScript normalized floating JSON numbers and stopped at
the existing checkpoint-identity check before reaching the intended control;
that failed attempt is retained separately, not counted as the negative-control
success. The corrected corruption preserves the Python fixture's number types.

Parent runtime checks, source hashes, all relevant attempts and limitations
are recorded separately in `validation.json`. Review closure is not a claim of
language quality, speed, CI success or general numerical equivalence.
