# Resident Geometry Guard Batching

Status: **rejected by the frozen combined performance screen; not adopted**.
The candidate remains in `0f09558b9aa70b893ebfbfa0b4765634db24d24b` and captured
local artifacts. Its five Rust files have been restored byte-for-byte to
`edf5e2008e810fca3e498d50bf80e11be714da36`; no unrelated files were restored.
No formal PR, merge, CI or general speed claim is made by this record.

Pair-bias (Poincare, Euclidean and Fisher-Rao), causal-wave, and headwise-mixture
forward/backward in the candidate encoded their existing pre/post validity captures in the
operation's command encoder. Querywise mixtures reuse the headwise path.
Each of these six candidate methods made one explicit queue submission instead of
three. Shader arithmetic, reductions, captures and guard propagation remain.
This is not a whole-model submission count or a timing result.

## Executed Checks

- 27 backend tests passed with GPU execution enabled: validity 4, pair metrics
  11, causal wave 4, and shared/querywise mixture 8. Existing integration tests
  exercise strided views, retained tapes and late/nonfinite failures.
- 27 byte-decoder tests passed, including a serial rerun with GPU execution
  enabled. The added guard test records captures before and after a producer
  failure in one encoder and checks immutable earlier aliases and original data.
- Native and browser each reran the same four contextual full-LM fixtures,
  sixteen updates per case, with VJPs, checkpoint/resume and guard checks.
  Each complete raw report is byte-identical to its retained same-runtime
  baseline; [validation.json](validation.json) binds their hashes.

The original contextual-model browser qualification remains **failed**:
the zero-initialized two-block case exceeds the unchanged rounded router-delta
gate at update 14. Its reference delta is `6.812899059e-6`, relative error
`0.004374396586`, limit `0.002`. Batching preserves that failure, not fixes it.
Native still passes. See the [original result](../2026-10-13-contextual-mixture-v1/README.md).

## Review And Performance Decision

Independent read-only local review found no actionable P1/P2 issues in the
five changed Rust files against `edf5e2008e810fca3e498d50bf80e11be714da36`.
That reviewer did not execute builds/tests or inspect helpers outside those
files. Parent-executed tests are not an independent execution replication.
Abandoning an unsubmitted encoder is not directly covered by the new test.

The [predeclared correctness protocol](pre-registered.md) remains fixed.
The separate [performance protocol](performance-protocol.md) was frozen before
timing: three warmup-plus-ABBA blocks per runtime, at least 5% geometric-mean
improvement and no slower block. All 36 invocations finished and their full
same-runtime reports matched byte-for-byte (browser: the existing page's
JS-normalized JSON), including the original browser failure.
[performance-results.json](performance-results.json) retains every scalar
timing, including warmups, and the unchanged decision rule gives:

| Runtime | Block 1 B/A | Block 2 B/A | Block 3 B/A | Geometric Mean | Screen |
| --- | ---: | ---: | ---: | ---: | --- |
| Native | 0.916752 | 0.927348 | 0.890217 | 0.911304 | pass |
| Browser | 0.969415 | 0.913823 | 1.041959 | 0.973660 | fail |

The native artifact pair shows 8.87% lower geometric-mean block-median time.
The browser pair shows only 2.63%, with a 4.20% regression in block 3; it fails
both the minimum-improvement and no-regression requirements. The combined
screen therefore fails. No trial was replaced and no threshold was relaxed.
The candidate was not promoted, and its additional full v8 corpus adoption
gate was not run. Completed v8 learning/resume uses the original baseline
executables, not this candidate.

These are fixed full-qualification artifact-pair measurements, not isolated
kernel timings, statistical significance, general FT throughput or a Torch
speed comparison. Native includes process startup/output writes; browser times
the awaited Rust export. Their absolute times are not comparable. Variation
between blocks remains visible and is not attributed solely to batching.

The [timing-harness review](timing-review.md) records three initial P2 findings,
their repairs and the reviewed seven CPU-only tests. All three browser blocks
executed verified-byte blob modules and completed the required IndexedDB
transactions. Fresh-page recovery and crash durability were not exercised.

After the rejection, 53 scoped Rust tests passed on the restored source with
GPU runtime tests enabled and one test thread: validity 3, pair metrics 11,
causal wave 4, mixture 8, byte decoder 27. The restored WASM release example
also built successfully; that fresh artifact was not rerun in the browser.
[performance-evidence.json](performance-evidence.json) binds raw trial packets,
unchanged timing-harness files, rollback logs and exact five-file restoration.
Candidate checks above are historical; the encoded-group test was removed
with the rejected private encoder API, while all original checks remain.

Reproduce the four-case check with the existing
[contextual geometry recipe](../../../docs/contextual_geometry_mixture.md)
and the same frozen fixture, then compare complete raw bytes with the baseline.
Run scoped Rust tests with `SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1` and
`--test-threads=1`. Raw tensors, checkpoints, binaries and logs stay local;
published hashes identify retained bytes, not execution attestation.

To reproduce this artifact-pair workload, use fresh local output paths and
keep this completed screen intact; a new run is not a replacement trial:

```sh
python3 -I -S -B bench_native.py performance-protocol.json \
  "$BASELINE_BIN" "$CANDIDATE_BIN" "$FIXTURE" "$RAW/performance/native"
```

On loopback, serve `bench_browser.html`, `/performance-protocol.json`,
`/reference.json`, and the pinned JS/WASM pair under each of `/A/` and `/B/`.
Run fresh pages with `?block=1`, `?block=2`, then `?block=3`, one at a time;
download each private packet as `browser-block-N.json` under `$RAW/performance`.
Reopening an existing block recovers its saved record without rerunning it.
Then publish scalar results only:

```sh
python3 -I -S -B summarize_performance.py performance-protocol.json \
  "$RAW/performance" "$NEW_SCALAR_RESULT"
```
