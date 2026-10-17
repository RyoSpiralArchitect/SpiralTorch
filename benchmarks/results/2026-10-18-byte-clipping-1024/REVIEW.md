# Experiment Supervisor Review Corrections

Read-only independent review found four issues in the original harness retained
at `5cbcfbe900a453526e7408204548dcb5ec86db4a`: parent cancellation could leave
workers alive, an inner timeout could leave descendants alive, a truncated
report could prevent failure retention and the next condition, and one CLI test
launched non-isolated Python. The full long study finished with frozen sources;
its results are not retroactively attributed to repaired supervision.

The first repair added owned process groups, exception-safe cleanup, immediate
execution receipts before report parsing, sanitized failure dispositions and
isolated test launches. Its 20 tests and real two-update recovery passed, but
independent adversarial follow-up reproduced three additional issues:

- Equal cleanup deadlines let an enclosing supervisor kill a nested supervisor
  before its TERM-resistant worker had been killed.
- A signal after actual spawn but before `Popen` assignment escaped ownership.
- Historical revalidation accepted a source inventory with entries removed.

## Final Correction

The supported two-level supervisor/worker contract now uses five seconds of
cooperative grace only for the enclosing supervisor. Direct worker groups use
zero-grace SIGKILL and reaping; they do not race an identical five-second timer.
Signals received during spawn are deferred until ownership is assigned, then
re-raised through cleanup. Original handlers are restored.

During development, TERM followed immediately by KILL produced macOS `EPERM`
in cleanup tests. The zero-grace branch now issues KILL directly once instead;
permission errors are not suppressed or treated as successful cleanup.

Historical export is deliberately restricted to this experiment's exact source
commit and full original manifest digest
`8ad54e792b6704210bfce0a25af8c7d8ca119ec7e6d184b780396d215c73628e`.
Commit existence and complete manifest identity are verified before checking
the 719 source blobs or allowing the two changed harness files. Current
numerical checkers and model sources still must match the original pins.
The manifest is read once; its decoded inventory and digest use the same
captured bytes, preventing a file swap between decoding and hash verification.

## Verification

- Seven owned-process tests cover success/failure descendants, timeout,
  KeyboardInterrupt, cooperative nested shutdown, a TERM-resistant leaf and
  SIGINT/SIGTERM at the spawn-registration boundary.
- Seven qualifier tests cover bounded/default timeouts, the real timeout path,
  isolated CLI rejection and preserved public metadata.
- Eleven study tests cover exact request wrapping, private-payload exclusion,
  malformed reports, continued condition processing, failed recovery receipts,
  complete historical inventory, commit existence, current-checker drift and
  manifest decode/hash read consistency.
- The final installed-wheel two-update recovery again passes; complete report
  and checkpoint bytes equal the retained earlier AdamW fixture. The final
  receipt is [supervisor-final.json](results/supervisor-final.json).
- The final publisher verifies all original sources and reproduces the original
  `results.json`, `inputs.json` and `artifacts.json` byte-for-byte, without new
  training. See [postprocessing-final.json](results/postprocessing-final.json).

The earlier `supervisor-recheck.json` and `postprocessing.json` identify the
first repair; the `*-reviewed.json` pair predates the final single-read hardening.
None are silently overwritten with later source hashes.
Supervision does not claim power-loss safety, cleanup after external SIGKILL of
the supervisor itself, or ownership of deliberately detached/uncooperative
daemons. These repairs do not change model math or reinterpret failed gates.

## Independent Closure

The read-only reviewer confirmed the three follow-up findings closed within the
declared two-level process contract, independently ran the 24 then-current
tests, and checked the manifest pin, commit, all 719 historical blobs and 717
non-exempt current sources. A final narrow review confirmed that only the
single-read change/test followed; its 11 study tests pass with no new findings.
The combined final scoped suite has 25 tests. GPU recovery and complete export
revalidation remain implementing-agent checks, not reviewer GPU observations.

Final reviewed implementation SHA-256 identities:

| Source | SHA-256 |
| --- | --- |
| `tools/owned_process.py` | `a5dc7ccf742576b6969ee73f76a6d858246f9726a66c7517001c3c65f8f77b60` |
| `tools/test_owned_process.py` | `dfd2cd88d98086acb13f9005a84ce8e8a8cf0eea05f28ac012eb0f7c977f91a2` |
| `tools/qualify_byte_corpus_runs.py` | `75bd5dd2f0bbe02b7215207da0fd1782a3285ccf11f0890ef8f5033b50f64540` |
| `tools/test_qualify_byte_corpus_runs.py` | `0463e2d5296757270bb00e26fe36c3ead56e50363de2d619f1b5f0f7927185cd` |
| `study.py` | `30e97018e3cd9239fa6295ec958fd152f3fb2396193477cfcdf954ba4b09460f` |
| `test_study.py` | `2b7b994f4068a5de1dedff91b77037d8a633738b144bc897e4af494a010856e8` |
| `.github/workflows/ci.yml` | `b0ba8d941bf698f138d3629a9693379bd410e31530629c905f9efe59c17821bd` |
