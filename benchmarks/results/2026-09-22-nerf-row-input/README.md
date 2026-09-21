# Affine row input: frozen M4 native/WebGPU comparison

Measured implementation: `2a17ac01a7cc3f1e11e4cb7c6f9bf36a66f32bf1`.
Three rotated serial rounds, 12 ray/sample/field combinations, bursts 1/4,
9 paired blocks per cell per round: **3,888 retained intervals**, including
eager PyTorch CPU/MPS. No condition or unsuccessful exploratory attempt omitted.
Both graph routes use the same binary and three render submissions; only the
input-addressing policy differs. This is not the previous submission comparison.

## Results And Limits

The geometric mean of each cell's median paired **packed / rows** time ratio
is **1.089923 native**, **1.085365 browser** (above one favors row addressing).
Native: 21/24 cells above one, range 0.946600-1.306471. Browser: 20/24 above one,
range 0.930233-1.250000. These are descriptive shared-desktop ratios, not
statistical significance, isolated GPU timestamps or a universal speed claim.
Near-one browser ratios are especially sensitive to clock quantization.

The 1x1 cases already use identical contiguous addressing in both routes and
save no packing. Their paired browser ratio still reaches 1.25, demonstrating
the measurement noise; do not attribute that difference to row addressing.
Over the 20 cells where packing is actually avoided, the corresponding
geometric means are 1.107039 native and 1.077429 browser. This eligibility
subset follows the layout contract, not selection of favorable timings.
All cells, including regressions, remain in the primary aggregate and table.

All routes passed the unchanged independent-oracle tolerance
`4e-7 + 4e-6 * abs(reference)`. Maximum absolute deviation is
`1.2516975402832031e-6`; maximum scaled deviation is 0.493533 (acceptance <= 1).
Packing/direct-row pairs agree; the oracle uses f64 geometry/integration and
f32 NN, while timed eager Torch CPU/MPS uses f32 and stable thin-opacity math.

Native adapter: Apple M4 / Metal. Browser: Chrome 153.0.8010.48 / BrowserWebGpu;
its separate Apple/metal-3 nonfallback probe is not Rust-device attestation.
Torch 2.12.1 used actual CPU/MPS, 4 intra-op / 1 inter-op threads, no compile.
This benchmark does not establish general PyTorch superiority, training
improvement, scene quality, cross-vendor performance or CUDA readiness.

### All Conditions

Times are median **whole interval milliseconds**, not per-render kernel times.
Burst4 observes only its last output. Paired ratios are medians of paired
block ratios, so they need not equal the ratio of displayed marginal medians.
N/B mean native/browser. Hidden=0 is affine; hidden=32 is Linear/ReLU/Linear.

| Rays | Samples | Hidden | Burst | N packed | N rows | B packed | B rows | CPU | MPS | N paired | B paired |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0 | 1 | 0.440 | 0.437 | 0.400 | 0.400 | 0.128 | 1.173 | 1.018 | 1.000 |
| 1 | 1 | 0 | 4 | 1.069 | 1.048 | 0.900 | 0.800 | 0.387 | 3.051 | 1.057 | 1.125 |
| 1 | 1 | 32 | 1 | 0.425 | 0.434 | 0.400 | 0.400 | 0.109 | 1.073 | 0.985 | 1.250 |
| 1 | 1 | 32 | 4 | 1.050 | 1.052 | 0.900 | 0.800 | 0.392 | 3.148 | 0.975 | 1.143 |
| 1 | 64 | 0 | 1 | 0.636 | 0.616 | 0.600 | 0.600 | 0.118 | 1.286 | 1.049 | 1.000 |
| 1 | 64 | 0 | 4 | 1.579 | 1.220 | 1.200 | 1.300 | 0.405 | 3.094 | 1.295 | 1.083 |
| 1 | 64 | 32 | 1 | 0.651 | 0.620 | 0.600 | 0.600 | 0.123 | 1.189 | 1.091 | 1.000 |
| 1 | 64 | 32 | 4 | 1.539 | 1.341 | 1.400 | 1.200 | 0.428 | 3.713 | 1.237 | 1.091 |
| 65 | 64 | 0 | 1 | 0.644 | 0.617 | 0.700 | 0.600 | 0.310 | 1.317 | 1.065 | 1.000 |
| 65 | 64 | 0 | 4 | 1.554 | 1.348 | 1.400 | 1.200 | 1.033 | 3.585 | 1.154 | 1.182 |
| 65 | 64 | 32 | 1 | 0.760 | 0.646 | 0.800 | 0.600 | 0.488 | 1.605 | 1.179 | 1.143 |
| 65 | 64 | 32 | 4 | 1.716 | 1.578 | 1.500 | 1.400 | 2.093 | 4.668 | 1.081 | 1.077 |
| 256 | 64 | 0 | 1 | 0.662 | 0.631 | 0.700 | 0.600 | 0.673 | 1.290 | 1.046 | 1.167 |
| 256 | 64 | 0 | 4 | 1.705 | 1.344 | 1.500 | 1.400 | 2.445 | 3.326 | 1.306 | 1.000 |
| 256 | 64 | 32 | 1 | 1.072 | 0.927 | 1.000 | 1.000 | 1.919 | 1.530 | 1.048 | 1.111 |
| 256 | 64 | 32 | 4 | 2.786 | 2.834 | 2.700 | 2.400 | 8.246 | 5.535 | 0.947 | 1.136 |
| 256 | 256 | 0 | 1 | 1.364 | 1.289 | 1.500 | 1.300 | 2.667 | 1.343 | 1.088 | 1.154 |
| 256 | 256 | 0 | 4 | 4.841 | 4.646 | 4.400 | 4.400 | 10.570 | 5.084 | 1.058 | 0.930 |
| 256 | 256 | 32 | 1 | 3.079 | 2.537 | 2.500 | 2.500 | 7.733 | 4.555 | 1.094 | 1.000 |
| 256 | 256 | 32 | 4 | 10.570 | 10.399 | 8.500 | 8.400 | 26.654 | 12.641 | 1.020 | 1.012 |
| 1024 | 64 | 0 | 1 | 0.872 | 0.738 | 0.900 | 0.800 | 2.452 | 1.327 | 1.156 | 1.250 |
| 1024 | 64 | 0 | 4 | 2.198 | 1.985 | 2.000 | 1.800 | 8.361 | 4.587 | 1.110 | 1.125 |
| 1024 | 64 | 32 | 1 | 2.285 | 2.150 | 2.200 | 1.900 | 6.611 | 3.086 | 1.134 | 1.118 |
| 1024 | 64 | 32 | 4 | 8.833 | 8.390 | 7.500 | 7.300 | 31.974 | 12.460 | 1.050 | 1.028 |

## Verification And Replay

29 clean-source stages passed: 180 backend tests with real GPU enabled,
30 shared-contract tests, 9 upper-NN integration tests, 17 current/legacy
protocol/archive mutation tests, strict native/WASM Clippy, formatting,
builds, and both previous native/browser benchmark modes. New tests cover
same-storage/different-layout caching, aborted compositions, singleton
columns/rows, multiple tiles, all dense kernel/accumulation choices, fallback,
direct-to-stable dispatch, inherited errors and held outputs after drop.

`exploration.json` retains the earlier 3-round screen separately (1.067874
native, 1.076084 browser), all recorded attempts, their source identities and
the failed archiver-import attempt. That collision between historical modules
was fixed by loading the shared archiver by exact path. No earlier timings
were pooled into the clean-source result. Historical archives are unchanged.

`results.json` contains every interval, numerical bound and input/oracle hash.
`source.json` binds measured source bytes; `validation.json` contains exact
commands/times and clean-source checks. `local-raw-manifest.json` binds 397
local files / 50,446,292 bytes, including raw arrays, frozen binaries, source
snapshots/patches and receipts. Raw files remain local under
`~/Library/Logs/SpiralTorch/graph-row-input-20260922`.

```sh
python3 -I -B benchmarks/nerf-row-input/evidence.py verify benchmarks/results/2026-09-22-nerf-row-input
python3 -I -B benchmarks/nerf-row-input/evidence.py verify benchmarks/results/2026-09-22-nerf-row-input --raw-root /PATH/TO/graph-row-input-20260922 --source-root /CHECKOUT/OF/2a17ac01
```

Fixity/aggregation verification is not numerical reexecution. For fresh
numbers, use the frozen commit and [replay protocol](../../nerf-row-input/README.md)
with new output paths and the recorded serial order. Setup and validation are
outside the timed boundary; allocation, encoding, submission, terminal owning
copy/map/completion are inside. No performance-based default for submission
coalescing is changed by this work.
