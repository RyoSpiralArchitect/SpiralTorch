# Sustained Browser Calls With A Completed Read Per Forward

Measurement source: `ef607370681b9ff68e28e1b7082dc42af05f4394`.
Frozen runtime baseline: `04fec016ad1ee8871d2a786974e94caeb556b84b`.
Frozen runtime candidate: `6703df35e8389c3ebb9ddf5eb099f440995a930d`.
No runtime code was changed or rebuilt for this record.

## Question And Protocol

The [earlier in-pass guard record](../2026-09-12-module-inline-guard/README.md)
had coarse 0.2/0.3 ms browser single-call readings: its small-case paired-median
ratio was 1.250, while the pooled elapsed ratio was 0.935. Both remain preserved.
This follow-up measures a different, predeclared sustained workload instead of
replacing that endpoint or selecting another favorable statistic.

Each measured interval contains **256 independent fixed-input forwards, each
followed by its own completed host read and output release**. Only the outer
interval is timed. All typed output arrays remain alive until the interval ends;
every returned value is then checked against the frozen native reference outside
the timer. This includes per-call host output allocation/retention and release,
not just GPU compute. It is not a final-read-only burst or recurrent decoding.

The same page loads the two frozen WASM packages into separate device contexts.
Four matrices cover the same nine shape/seed cases (seeds 17, 29, 43), with two
warmup and nine retained intervals per route. Original Module calls and the
untargeted explicit graph dispatch schedule are both measured. Route order
rotates; context creation and case order alternate across matrices. Setup,
upload, cold compilation and numerical comparison are outside the intervals.
There were no retries, trimmed samples, adaptive sample counts or discarded runs.

## Results

Primary endpoint: median of 12 case-median candidate/baseline ratios per shape.
Lower is faster. The control is not used to normalize away drift.

| Shape | Module ratio | Module range | Explicit graph ratio | Module pooled elapsed ratio |
| --- | ---: | ---: | ---: | ---: |
| `[2,3,7]` | 0.9984 | 0.9905-1.0047 | 1.0078 | 0.9905 |
| `[2,8,64]` | 0.9990 | 0.9932-1.0078 | 1.0034 | 1.0049 |
| `[4,8,128]` | 0.9985 | 0.9866-1.0050 | 1.0015 | 1.0061 |

Module case-median ratios regress in 3/12, 4/12 and 5/12 pairs respectively;
the explicit graph control regresses in 12/12, 9/12 and 7/12. All intervals and
slow tails remain in [the raw archive](manifest.json). The primary medians and
pooled totals are separate statistics, not interchangeable summaries.

The shortest interval was 59 ms. The clock probe observed a smallest positive
delta of approximately 0.1 ms, so even the shortest interval covered about 590
observed clock increments. This is a granularity diagnostic, not a timing-error
bound or evidence of exclusive GPU access. Chrome was `152.0.7977.84`, with
`crossOriginIsolated=false`. Reported backend was `BrowserWebGpu`; physical
browser GPU identity and host contention remain UNKNOWN.

**Decision:** keep the in-pass guard grouping without browser-specific policy.
There is no material consistent speed change in this sustained completed-read
fixture, and no 25% slowdown here. This neither establishes a browser speedup nor
disproves the earlier isolated-call result. The prior native improvement remains
separate evidence; this record did not rerun PyTorch or native performance tests.

## Verification And Evidence

- All ten verification/measurement steps passed in 203.7 seconds. The new
  interval/cadence tests (14) and existing admission tests (nine) passed.
- 405,504 forwards with completed reads, including warmup, checked 697,737,216
  returned values. Maximum absolute reference error was `2.384185791015625e-7`.
  Every model context compiled once and submitted 2,817 forwards including its
  cold call. There were no browser page or console errors.
- The existing nine-case short browser protocol was rerun through the modified
  driver and admitted separately, using the prior frozen Python controls. It is
  a driver regression check, not a new cross-platform performance comparison.
- All 34 frozen baseline/candidate products were rehashed before and after the
  run and during archival. No Rust rebuild, CPU fallback, CUDA/Furnace run,
  package release or push is part of this record.

[summary.json](summary.json) records the identities, counts and aggregate ratios.
[manifest.json](manifest.json) binds 37 compressed source/input/output records:
30,843,283 raw bytes and 514,556 compressed bytes, round-trip checked. The archive
includes the fixed harness, validator, tests, runtime receipts, exact served
fixture, all intervals and streamed cases. The collector streams each completed
case before proceeding; the validator requires those rows to equal the final
report and checks their hashes, order, read/submission counts and all cached
model counters. Frozen WASM binaries remain in the local verification directories.

The next implementation candidate is bounded staging reuse for
`ResidentTensor::snapshot`, which still allocates a new readback buffer while
explicit graph snapshots reuse a pool. That difference is verified in the
source, but is **not yet a measured bottleneck or an implemented optimization**.
Any reuse must preserve outstanding snapshots, cancellation, shape boundaries
and invalid-value guards before being measured with this same protocol.
