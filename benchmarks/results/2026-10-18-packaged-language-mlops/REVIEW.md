# Independent Review And Dispositions

Two read-only passes covered the new Rust/WASM adapters, generated-boundary
risks, qualification scripts, declarations and claims. The reviewer did not run
GPU work or author fixes. Parent execution supplies the recorded validation.
No Rust numerical core, threshold, geometry rule or model default changed.

## Adapter Review

The packaged path addresses the standalone adapters' pre-copy ingress limits,
JS numeric/byte coercions, retained async borrows and fresh-runtime-per-call
behavior. Immutable inputs use owned references; mutable owners use synchronous
exclusive leases returned on ordinary success/error. Freeing a JS handle is
not cancellation. Busy activation does not consume a candidate; an attempted
core admission does. All acceptance remains in `st-nn`.

An initial compile mismatch passed owned tensors instead of references to the
readback API; it was fixed before browser qualification. Forward loss is a
scalar, and revision/generation counters are decimal strings. The final source
review found no outstanding adapter or declaration defect within its scope.

## Qualification Review

Four P2 findings were fixed, then checked again:

1. **Vacuous empty results.** Generated row counts, byte counts, prompt/hash
   dimensions and logit dimensions now come from the fixture. Training must
   contain every requested case in order, matching seed, every accepted update,
   and checkpoints at the declared partial/full cursors. Empty arrays cannot
   pass `.every()` alone.
2. **Wrong-reason negative passes.** Oversized/invalid-Unicode ingress probes
   match the actual boundary errors. Busy-operation probes require `operation
   in progress`, rather than passing because rollback history is unavailable.
   The asynchronous candidate error must be `candidate is already active`.
3. **Live assets versus manifest.** The loopback server reads each static asset
   once, validates those exact bytes against the input manifest and serves its
   in-memory snapshot. A later filesystem change cannot change the served run.
   Accepted training POST bytes are likewise retained for the resume page.
4. **Private GET Host checks.** Both GET and POST require the literal loopback
   host/port. POST also requires the exact Origin. Routes are explicitly
   allowlisted; result creation is exclusive. COOP/COEP are not treated as
   substitutes for the Host check.

The new synthetic HTTP suite initially used `HTTPConnection` as a context
manager. It was reproduced as three test errors, corrected to
`contextlib.closing`, and all three tests passed twice. The focused follow-up
review confirmed the fix and reported no outstanding source findings.

After those changes, the parent reran the manifest-bound package qualification:
101 browser checks, all nine actual AdamW cases, fresh-page exact resume,
unchanged Torch numerical gates, both feature-specific Node suites and the
synthetic HTTP suite passed. [Verification](results/verification.json) binds
the final source and package; pre-review private runs were retained rather than
overwritten. No production source changed during qualification fixes.

## Remaining Boundaries

This does not establish cross-device bitwise replay, physical browser hardware
execution, language-quality uplift, concurrency-safe cold device acquisition,
GPU cancellation or a durable deployment service. Serving thresholds here are
permissive mechanics limits; the deliberately failed cohort remains recorded.
The 1,024-update source study's strict failure remains unresolved and is not
overridden by successful artifact restoration.

The latest inspected parent PR run, `38739550458`, still records an account
billing lock before execution, including its WASM build job. This historical
failure does not prove the account's current billing state. No CI/merge bypass
or package upload was performed for this milestone.
