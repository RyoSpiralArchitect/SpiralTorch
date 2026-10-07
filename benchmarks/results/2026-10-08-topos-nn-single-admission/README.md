# Direct CPU Topos Admission: Avoid One Repeated Scan

The native NN wrapper previously checked input/gate finiteness and their
finite product, then core capture performed those same checks again. Direct
CPU and legacy Auto now rely on core's complete admission. Accelerator
requests retain the early check, even when their size threshold selects CPU,
so invalid inputs still fail before route metadata or availability probes.
There is no unchecked core API, changed recurrence, weaker tolerance, or
Python/WASM math change. The host WGPU executor is unchanged.

Baseline source: `5eefa75ba1627d72e44e7ac122ab91484d577f24`.
Candidate source: `66ad6eff886d6823fad206ee0b8b9b3a4ce245df`.
The only measured production-file change is the native NN wrapper; the
exact NN/Torch harness from the preceding study is unchanged.

## Measurements

Apple M4, macOS 26.4.1, Rust 1.97.0 release, CPU Torch 2.12.1 eager with
one thread. The shape/iteration matrix, ABBA order, two warmups, 20 samples
per route and median-of-process-medians aggregation were all frozen before
measurement. There are 36 native and 36 Torch reports, **2,880 timings**,
with forward/reverse/forward/reverse case order. Both excluded pilots are
retained. No samples or conditions were removed.

`F+B` includes forward, both VJPs and shared gate-gradient accumulation;
only Rust includes its semantic audits. Setup, reset, comparisons and file
transport are outside timing. F records a training tape, not inference.
All values below are milliseconds; Torch uses the two candidate-phase runs.

| Rows x Features | Iterations | Native F Before | Native F After | Native F+B Before | Native F+B After | Torch F+B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 x 3 | 1 | 0.000458 | 0.000417 | 0.000802 | 0.000739 | 0.056406 |
| 8 x 3 | 5 | 0.000521 | 0.000521 | 0.000834 | 0.000854 | 0.231396 |
| 8 x 3 | 16 | 0.000834 | 0.000875 | 0.001188 | 0.001208 | 0.689032 |
| 64 x 128 | 1 | 0.047636 | 0.044188 | 0.067344 | 0.064188 | 0.089635 |
| 64 x 128 | 5 | 0.092875 | 0.089062 | 0.112135 | 0.109167 | 0.381230 |
| 64 x 128 | 16 | 0.218229 | 0.211125 | 0.236396 | 0.230469 | 1.225469 |
| 256 x 768 | 1 | 1.141198 | 1.027271 | 1.631864 | 1.507969 | 1.405594 |
| 256 x 768 | 5 | 2.136021 | 2.061042 | 2.623177 | 2.545271 | 6.744354 |
| 256 x 768 | 16 | 4.918396 | 4.786208 | 5.380240 | 5.262136 | 22.057833 |

Large native F+B cases are about **7.6%, 3.0%, 2.2% shorter** in this study.
At 256 x 768 / one iteration, native still takes about 1.073 times Torch's
time. Tiny 5/16-iteration F+B cases regress by about **2.5% / 1.7%**; their
per-process medians and every raw sample remain in the bundle. Normal desktop
noise was not controlled, two processes per arm do not establish a confidence
interval, and tiny timings are particularly noisy. No simultaneous builds
or training ran during measurement. No compiled-Torch, accelerator, optimizer,
end-to-end FT, peak-memory, general speed or model-quality claim is made.

Before editing production, 18 diagnostic processes measured four overlapping
public scopes. At 256 x 768 / one iteration, their median times were 0.110 ms
for state validation, 0.997 ms for core capture, 1.127 ms for NN forward and
0.413 ms for an audited captured core VJP. These are **not additive phases**:
capture already includes validation and NN forward includes capture. They
motivated checking duplication; the matched full-work matrix above evaluates
the actual change. These diagnostic timings are not pooled with that matrix.
There are 31 zero-duration validation-only samples at shape 8 x 3, below the
clock's effective resolution. They are retained, not replaced or removed;
no speed ratios are calculated from these diagnostic scopes. All 2,880
full-work native/Torch timings in the comparison matrix are positive.

## Correctness And Records

All six native vector digests and both audits are bit-identical across all
four runs of each condition. Every Torch output and both gradients passed
rtol `5e-4`, atol `3e-5`; maximum absolute gate error was
`2.6226043701171875e-6`, maximum normalized error `0.023776` (limit `1`).

Focused Rust tests: **27 CPU, 31 WGPU-feature**, including executed GPU
mixed-route and compact-tape replays. New tests cover 96 combinations of
policy, layout, gate type and invalid values, matching the previous core
error and rejecting before route/forward events. Failed CPU forwards retain
the previous valid tape, audit and gradient. A fresh native 200-update SGD
trajectory is byte-identical to the preceding public learning record and
again passes the independent Torch comparison. It is referenced by hash,
not duplicated here. Python/WASM artifacts were not rebuilt for this
host-wrapper-only change; no new browser or accelerator speed result is claimed.

`measurements.json.gz` retains the complete fixed plan, build identities,
unchanged harness sources, all exact JSON reports, pilots and summaries.
`diagnostics.json.gz` retains all diagnostic reports and their probe source.
`verification.json` binds local original files, binaries, vectors, logs and
the fresh learning comparison. `SHA256SUMS` binds this public package.
The full float32 vectors and logs remain local as requested. Saved-record
checks reconstruct results and reject inconsistencies; they are not fresh
execution or cryptographic proof of execution. Independent source review
found no actionable P1/P2; it was static, not a benchmark rerun.

Reproduction uses the same instructions and unchanged harness as the
[preceding native comparison](../2026-10-07-topos-nn-shared-benchmark/README.md),
substituting the two revisions above and the frozen plan in this bundle.
Use fresh paths; no models, datasets or API calls are needed.

```bash
cargo run --locked --offline --release -p st-nn --example topos_shared_phase_probe -- 256 768 1 /tmp/topos-phase-new.json
python -I -S -B tools/test_topos_single_admission_results.py
```
