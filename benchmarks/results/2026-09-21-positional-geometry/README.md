# Positional Geometry: Native and Actual WASM

## Scope

Baseline `a98d81c1513b42ea02b57691d02a3bfceb400b8f` retains old numerical
behavior but includes the native-only Faer portability repair and regression
test sources. Candidate `6899536b46bea49d70d901330451bfbe275cf446` fixes
logical layouts, phase/size checks and exact RoPE cache identity; encoding writes
directly into aligned Tensor storage and evaluates paired sin/cos.

The NeRF constructor now uses reproducible Xavier layers and an active constant
density head (0.1 in inverse ray-distance units). Legacy `Linear::new` retains
its ramp parameters; the new common `Linear::new_xavier` is opt-in. Field
parameter names/shapes remain unchanged. Input-position derivatives remain
intentionally zero, as in the existing field Module contract.

## Results

Median-time baseline/candidate ratios, geometric mean across both measured rounds
(above 1 favors the candidate):

| Scope | Native | Node/WASM |
| --- | ---: | ---: |
| All 36 conditions | 0.805x | 0.899x |
| Nonzero frequency bands, all row counts | 1.061x | 1.100x |
| Nonzero bands, multiple rows | 1.031x | 1.146x |
| Zero bands, multiple rows | 0.316x | 0.478x |

All 32 multi-row/nonzero-band Node comparisons improved (range 1.048-1.196x);
native was mixed (23/32 improved, range 0.942-1.118x). The all-condition aggregate
regresses and must not be represented as a universal optimization win.

The worst native ratio is an empty-output case that previously did no input
validation: at (1024,6,0,false), round A changes from 46.875 ns to 1,822.875 ns.
The same Node case changes from 166.75 ns to 1,974 ns. A nonempty zero-band
identity case also regresses: native 2,687.5 to 4,385.5 ns and Node 4,302.125 to
6,583.375 ns. Those costs buy explicit finite-input admission; no case was dropped.

All 72 paired output hashes match within each native/WASM cohort on this host,
in addition to the independent tolerance checks. This observation does not
promise platform-independent libm bits.

Torch 2.12.1 / native candidate ratios for nonzero bands are 26.294x for one row,
2.819x for 32 rows, and 0.816x for 1024 rows. The tiny ratios are dominated by
different Python/Rust entry costs; for 1024 rows Torch is faster overall, with
mixed individual conditions (ratio range 0.487-1.702x). Do not call this a
general victory over PyTorch. All 360 measured and 180 preconditioning
condition records are included.

Fixed-evaluation MSE after 20 training steps:

| Initialization seed | Before | After |
| --- | ---: | ---: |
| 0 | 0.175710159 | 0.174930237 |
| 1 | 0.179923527 | 0.179631091 |
| 7 | 0.171884669 | 0.170912373 |
| 13 | 0.182344755 | 0.181931201 |

This is small but nonzero progress, not convergence. The positive density
initialization prevents an initially dead field; it does not guarantee that
density units cannot become inactive later.

## Measurement Boundary

- 36 fixed conditions: rows 1/32/1024, coordinates 3/6, frequency bands 0/4/10,
  original-coordinate residual off/on. Row-major only for performance because
  the old other-layout implementation was incorrect.
- One full baseline/candidate preconditioning cohort, then AB/BA; each case
  has three warmups and 15 intervals. Native uses eight calls per interval;
  Node uses 512 for one-row cases and eight otherwise.
- Frequency is exactly `2^band`, without a pi factor; row/band/dimension/sin-cos
  order. Independent double-precision expected values use absolute tolerance
  2e-6. Setup, export, validation and field contracts are outside timed regions.
- Output allocation/free is timed. Native Rust, Node-to-WASM and Python-to-Torch
  have different entry costs. Torch is a vectorized eager CPU no-grad control
  at four intra-op / one inter-op threads, not torch.compile or a fastest-Torch
  claim. No GPU or scene-render/training throughput is measured here.
- Extra finite checks and their cost are part of the new contract, including
  zero-feature cases. Buffer-copy removal is a code-structure observation;
  allocation counts were not instrumented for this experiment.
- Builds and timings launched here are serial. Host exclusivity and thermal
  state are unknown. Medians/geometric means describe this host, not a confidence
  interval or platform-wide speed guarantee. All slow conditions remain visible.

## Correctness and Learning

The same actual full-crate fixture is compiled natively and for WASM, with a
private Cargo target separate from workspace validation. It is not a mocked or
path-imported encoder. Both runtimes exercise nine field input/seed layout pairs,
parameter gradients and a single optimizer update, plus a nearby-angle cache
negative control. Node additionally exercises 36 encoder layout conditions,
three invalid-input guards, empty outputs and signed zero.

Old-runtime failures are retained as negative controls, not silently excluded
from correctness reporting. Their row-major encoding timings remain valid.
Standalone fixture parameters are overwritten by deterministic values so changed
model initialization cannot confound the layout checks.

Native contract tests independently check parameter VJPs with finite differences,
failure-before-gradient-mutation for malformed seeds, dimension overflow, key
identity, eviction and seeded initialization. RoPE trigonometry itself is
unchanged and still has no production attention caller.

Training is a small constant-color regression, not novel-scene quality evidence.
Four seeds (0,1,7,13), 20 steps, LR 1e-3, 32 rays and eight samples/ray retain the
original workload. A separate midpoint evaluator does not advance the training
RNG. The stronger exploratory convergence criterion failed; final engineering
tests require an MSE decrease greater than 1e-5. See `PREFLIGHT.md` and all failed
receipts. This revised criterion is not independently held-out/preregistered.

## Replay and Integrity

Fourteen local validation stages passed at the candidate source: format,
strict feature-enabled vision Clippy, 1,790 core/NN/vision tests, existing-warning
Clippy for core/NN, real vision WASM checking, public WASM build/bindgen, four
GELU and twenty matmul forward/backward WASM cases, Python build and nine
Python tests, eleven WGPU-enabled Linear tests and two strict resident-graph
runtime tests. The latter explicitly requires the GPU runtime and does not
silently skip on adapter failure. This is regression coverage, not GPU NeRF.
Core/NN Clippy still reports existing warnings; it is not a zero-warning claim.

`verify.py` checks exact file inventory/bytes, recorded sources/builds/workers,
complete cohorts, negative controls and recomputed timing summaries.
`test_verify.py` deliberately mutates evidence, including rehashed semantic
corruptions. Neither script reruns floating-point kernels or verifies quality
from hashes.

Raw output arrays and compiled native/WASM workers remain at the local root in
`provenance.json`; all conditions, summaries, source/worker/raw hashes and build
and validation logs are published. No raw arrays or binaries are embedded.

For numerical/performance replay:

1. Create separate clean checkouts at the two commits above. Preserve the pinned
   repository lockfile and vendored patches; do not upgrade dependencies.
2. Copy `reproduction/fixture` into a local scratch directory and rebase only its
   recorded absolute dependency/patch paths in Cargo.toml to the selected checkout.
   Use a dedicated fixture Cargo target, never a shared workspace validation cache.
   Record those path-rebasing changes and source hashes.
3. Build with Rust 1.98.0, `cargo build --release --locked --manifest-path
   fixture/Cargo.toml`, then the same command with `--target wasm32-unknown-unknown
   --lib`. Set CARGO_BUILD_JOBS=4 and RAYON_NUM_THREADS=4.
4. Freeze the executable as `baseline-build/positional-geometry-fixture` or
   `candidate-build/positional-geometry-fixture`. Run wasm-bindgen 0.2.104 on
   `positional_geometry_fixture.wasm` with `--target nodejs --out-dir
   <lane>-build/wasm`. Freeze both workers only after successful builds.
5. Copy `measure.py` and `wasm.cjs` beside those directories. Run Python 3.12
   `measure.py <scratch> native`, then `measure.py <scratch> wasm`. Run
   `torch_encoding.py` separately with the recorded Torch version. Do not overlap
   builds or timing. Every output directory must be new; do not overwrite old runs.
6. `reproduction/check.py` records the validation commands. For client checks,
   rebase its recorded python-client.py location to `reproduction/python-client.py`
   and supply a workspace-only Cargo target. Hosted CI remains a separate result.

The Node fixture is a test adapter, not a newly deployed browser API.
No LLM/FT improvement, camera-gradient support or GPU NeRF execution is claimed.
