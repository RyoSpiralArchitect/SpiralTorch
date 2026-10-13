# Independent Review

Baseline: `830494be48041a3a5368718e86ac273b2a14f0e3`.

A read-only local reviewer first examined the five changed core files: the pure
wave contract and tests, resident wave allocation/preflight and tests, and WGSL
backward shader. It found no actionable issues in compensated accumulation,
scratch indexing, public offsets or propagation of shared invalidity flags.

A follow-up examined the new integration harness, browser page, scalar publisher,
tests and implementation documentation. It found four P2 validation gaps:

1. An inherited query-error field could copy a dict/list into public results.
2. Numeric measurement fields allowed strings or booleans through generic
   scalar checks.
3. Different query failures shared one generic inherited-failure signature.
4. Browser PASS checked case counts but could accept duplicated identities.

All four were addressed before final publication. Query errors now require the
known initial embedding-contrast diagnostic and finite scalar measurements;
other error kinds/locations are rejected. Numeric fields require finite numeric
values, with nonnegative errors/losses. Explicit projections exclude extra raw
fields. Browser validation requires all 36 distinct analytic identities and
both overflow channels, and computes expectations independently of reported
agreement. Added Python/Node tests cover those failure modes.

The reviewer re-read the fixes and reported no remaining actionable findings.
This was source-only review: the reviewer did not run tests, compile, operate a
GPU/browser, edit files or restore deletions. The parent ran the recorded tests
and reran the final browser page; its downloaded result was byte-identical to
the prior runtime result. See [validation](validation.json) for scope and hashes.

Review and analytic correctness do not establish general numerical robustness,
speed or language-model quality. The original QKV qualification is still 4/6;
the phase trajectory failure and legacy query failure remain published. The
slightly worse native phase comparison is explicitly retained.
