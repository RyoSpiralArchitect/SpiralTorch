# Row-Major CPU Tiles Without Packing

Follow-up to [panel reuse](../2026-09-21-cpu-panel-reuse/README.md). The baseline is
`64e500768abe4cdf7e78e0674e1c07cc588ab163`; selected source is
`634222f991502f1a84b518180e240f4f976c4bf7`. Results below are relative to that
already-optimized baseline, not the original scalar implementation.

## Implementation And Ownership

Unpacked matrix products now read ordinary row-major A/B directly through 8x12
or 4x16 tiles. Independent column accumulators preserve each output's sequential
float32 K reduction, without FMA or reassociation. No A/B packing buffers are
needed. Existing column-packed RHS data and its public API are unchanged.

Partial rows/columns use fixed 8/4-column accumulators, including nonzero-output
accumulation in internal tests. A final partial row group alone no longer causes
a parallel launch. Both kernels, padded strides, zero K, surrounding output guards,
and all 160 tile-tail combinations are checked.

Autotuning now distinguishes row-major from prepacked-column execution in both
the cache key/context and the timed function. Revision 5 also identifies the
optional portable-SIMD feature in its key. Default Tensor Auto/GPU routing is not
changed; Rust owns the implementation used by its existing Python/WASM clients.

## Measured Results

Native measurements use Rust 1.98.0 release on Apple M4. Host exclusivity and
thermal state are unknown. The unchanged extended harness has ten fixed shapes,
packed/unpacked, serial/configured-four-thread modes and two AB/BA rounds: **160
measured Rust condition-runs**, 80 recorded full-grid preconditioning conditions,
and 40 PyTorch conditions. All Rust cases are bitwise-equal to sequential f32 and
pass an independent f64 reference at atol=1e-3, rtol=1e-4. PyTorch 2.12.1 CPU
passes the same tolerance with matching float32 input formula, reusable output,
RHS strides and configured intra-op threads (one inter-op thread).

Inputs, output allocation and prepacking are outside native timing. Each case
has three warmups, nine untrimmed intervals, and four calls per interval (two for
the largest volume). HOME and the autotune-store path are removed for fixed-kernel
comparisons. Configured threads do not prove physical utilization; small products
may be serial. Validation separately exercises locally isolated tuning stores.

Geometric means of baseline/candidate median-time ratios; greater than one
favors the candidate:

| Grid | Ordinary Row-Major | Prepacked Control |
| --- | ---: | ---: |
| Native extended, serial | 2.173 | 1.005 |
| Native extended, four-thread configuration | 3.458 | 1.006 |
| WASM/Node client, six shapes | 1.166 | 0.982 |

All 40 native extended row-major shape/mode/round comparisons favor the candidate.
Their range is 1.236-8.703. Its warmed Rust allocation samples are zero throughout
that row-major grid; this excludes cold initialization, tuning, Tensor wrappers,
and any output allocation outside the timer. It is not a process-wide zero-memory
claim. At 32x768x3072/four-thread, requests drop 517 -> 0 and time ratios are
8.703/7.841.

**PyTorch remains faster:** the extended ordinary-layout geometric-mean gap is
about 18.41x serial / 11.72x with four configured threads. This is a tolerance-
matched eager CPU comparison, not identical reduction implementation, fastest
PyTorch, GPU speed or learning-quality evidence.

`dense/` retains the full original 15-shape grid, including empty and tiny cases:
240 bitwise-valid condition-runs. Its all-shape ratios are 1.004/1.416 unpacked
and 0.748/1.016 prepacked in serial/four-thread modes. The first serial candidate
process is much slower on early tiny cases, even for the unchanged packed control;
the cause is not established. These values are not trimmed or hidden. In the
four-thread rounds, the formerly regressed 17x37x29 and 32x64x32 cases now have
ratios 1.700-1.732 and 2.235-2.244 respectively. No universal speedup is claimed.

`wasm/` uses the real CPU-only WASM build through `AutogradTensor` with gradients
disabled. Six shapes and both layouts give **48 measured condition-runs** plus
24 recorded preconditioning conditions. Every result equals an independent,
exact dyadic reference; output hashes also match between all module/round pairs.
Nine intervals contain two complete forward-and-free calls after three warmups.
Input construction, one-time prepacking and JS value export are excluded, but
output construction and destruction are included. All 12 ordinary-layout ratios
are favorable (1.036-1.491); packed results are mixed and mildly negative overall.
**Node execution is not a browser or WebGPU performance result.**

`crosscut/` reuses the 36-condition InfoNCE/fractal API harness. All 72 candidate
condition-runs, all baseline cases and all six matched Torch conditions pass their
independent-f64 objective/bitwise-fractal checks. Normalized 96x128 row-major
Tensor InfoNCE allocation requests fall 15 -> 13 and bytes 177,904 -> 124,656;
time ratios are 1.180/1.168. This is forward API evidence, not an FT quality claim.

## Verification And Provenance

The selected source passes 481 Tensor CPU tests; 13 release serial kernel tests;
13 optional portable-SIMD tests on nightly-2026-04-15; 32 self-supervision CPU tests
(three existing ignored cases); 26 CPU-only dispatch/autograd tests; scoped strict
native/WASM Clippy; three real native Python-extension tests; seven WGPU-feature
tests with a required live GPU probe; and **20 WASM forward/backward cases** plus
eight numerical-boundary tests. This is scoped validation, not all-workspace
tests or every browser/device combination.

`preflight/` retains the first prototype, its exact patch on the baseline, all
measurements and numerical checks. That version launched parallel work for an
otherwise serial tile plus one partial row, and had slow scalar column tails.
The selected implementation fixes those causes; the rejected results remain.
Probe-b was measured before its clean commit: source hashes match the committed
candidate exactly, and subsequent clean-source builds produce identical dense
and crosscut executable hashes. Matching-root compilation is required and
identical old/new executables are rejected by the extended measurement driver.

```bash
python3 -B -I verify.py
python3 -B -I test_verify.py
```

These verify hash-bound files, complete conditions and recorded gates, **not a
fresh numerical replay**. Native, Python-extension and WASM products remain local;
paths/hashes are in `provenance.json`. Raw compiler/test logs preserve their bytes.

For replay, use clean checkouts at the two pinned revisions and the recorded
Rust/bindgen versions. Build the unchanged examples at both revisions, freezing
each executable before switching checkouts. Force the changed crate to rebuild
if sharing a Cargo target; the preceding archive documents a real stale-artifact
failure. Run the archived drivers into fresh directories:

```bash
cargo +1.98.0 build --locked --release -p st-bench \
  --example cpu_dense_workspace --example cpu_dense_extended --example source_crosscut
python3 -B -I -S measure_extended.py measure --directory /absolute/new-native \
  --baseline /absolute/baseline-extended --candidate /absolute/candidate-extended \
  --torch-python /absolute/python --torch-site /absolute/site-packages
python3 -B -I -S verification_driver.py /absolute/candidate \
  /absolute/cargo-target /absolute/new-verification
python3 -B -I -S measure_wasm.py measure --directory /absolute/new-wasm \
  --harness /absolute/cpu_dense_bench.cjs \
  --baseline /absolute/baseline/spiraltorch_wasm.js \
  --candidate /absolute/candidate/spiraltorch_wasm.js
```

Adapt machine-local paths in the validation driver. `measure_dense.py` and
`measure_crosscut.py` preserve the earlier protocols. Next work should profile
actual NN-layer routing and remaining kernel throughput; do not silently change
the existing prepacked layout or infer training improvement from these timings.
