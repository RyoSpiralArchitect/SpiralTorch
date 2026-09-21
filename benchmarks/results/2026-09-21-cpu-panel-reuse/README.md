# CPU Panel Reuse

This follows the [CPU workspace comparison](../2026-09-21-cpu-dense-workspace/README.md).
It measures explicit CPU forward kernels on Apple M4 / macOS, Rust 1.98.0 release,
with PyTorch 2.12.1 CPU as a separate comparator. Host exclusivity and thermal
state are unknown. There is no training, CUDA, browser-speed, or resident-GPU
performance claim.

## Implementation

Packed A rows are now reused across multiple column blocks. A prepacked RHS is
consumed directly without another copy, so its parallel A scratch is allocated
per row task rather than per row task and column block. For an unpacked RHS,
column groups target 64 KiB of RHS scratch, with at least one microkernel panel.
A single minimum panel can exceed that target at large inner dimensions. This
is not a 64 KiB limit on total scratch, allocation requests, or peak memory.

The inner float32 reduction order is unchanged. Row/column tails are handled
once, including accumulation into a nonzero destination in internal tests.
Both microkernels are covered. The autotune revision changes from 3 to 4;
high-level Tensor Auto routing, public APIs and GPU routing are unchanged.

The selected source is `967ae20cc00a451865efdc9d55cf083b09806756`; baseline is
`574750bb708c50b7ea30f638d28f7651bd876538`. The identical new extended harness
is overlaid on the baseline's sparse source checkout. Production-source and
harness hashes, build logs and executable hashes are recorded. Legacy baseline
workers are the prior archive's selected workers, byte-for-byte; production
sources did not change in the intervening evidence-only baseline commit.

## Fixed Comparisons

`extended/` contains ten prespecified wide/tall shapes, packed/unpacked, serial
and configured-four-thread modes. Each mode records full-grid preconditioning
for both binaries followed by two AB/BA rounds: **160 measured Rust condition-runs,
80 preconditioning condition-runs and 40 PyTorch conditions**. All numerical gates
passed. Each Rust result is bitwise-equal to sequential float32 accumulation and
also checked against an independent float64 reference at atol=1e-3, rtol=1e-4.
Torch uses the same float32 input formula and float64 tolerance.

Inputs, reusable output buffers and prepacking are outside timing. Each condition
has three warmups and nine untrimmed intervals, with four calls per interval
(two for the largest volume). Torch uses `mm(..., out=...)`, verifies that the
output pointer stays fixed, and matches ordinary versus column-major RHS strides.
Torch intra-op threads match the configured Rust count, with one inter-op thread.
This matches configuration, not measured physical thread utilization. Small Rust
products may stay serial. HOME and the autotune-store variable are removed from
worker environments to fix the default kernel, not change global settings.

Geometric means of baseline/candidate median-time ratios across all ten shapes
and both rounds (greater than one favors the candidate):

| Mode | Unpacked | Prepacked |
| --- | ---: | ---: |
| Serial | 1.121 | 1.618 |
| Four-thread configuration | 1.150 | 2.141 |

All 40 prepacked shape/mode/round comparisons favored the candidate; individual
ratios span 1.233-3.761. Unpacked results are mixed, with minima of 0.900/0.894.
At large inner dimensions a group contains only one microkernel panel, so the
unpacked path retains repeated work. At 64x256x1024 serial, grouped RHS scratch
raises allocation-request bytes from 20,480 to 69,632: packing reuse trades some
bounded temporary storage for less work. There is no universal memory reduction.

At 32x768x3072 prepacked/four-thread, warmed Rust allocation requests fall from
516 to 2 and requested bytes from 12,588,992 to 49,152; time ratios are
2.956 and 2.973. These are allocation requests, not peak resident memory.

**PyTorch is still faster.** Across the extended prepacked grid, its geometric
mean speed advantage is about 10.39x serial and 6.67x at four threads. The unpacked
gap is about 40.29x/39.03x. Layout-specific results remain separate, and neither
this CPU comparison nor a reduction in allocations establishes learning quality.

`dense-initial/` preserves the complete earlier 15-shape grid (240 bitwise-valid
condition-runs), without full-grid preconditioning. Its serial aggregate ratios
are 0.887 unpacked and 0.864 packed; four-thread ratios are 1.154/1.148. The first
candidate serial process was much slower on the first eight shapes than the
second process; the cause was not established. Those results are not discarded
or replaced by a favorable subset. The extended protocol and grid were declared
before the replacement runs, and its preconditioning output is also retained.

`crosscut/` retains the unchanged 36-condition InfoNCE/fractal API comparison,
two AB/BA rounds and six matched PyTorch conditions. This is a separate high-level
regression check. Its complete per-condition timings, allocations and float64
objective/bitwise fractal checks are in `comparison.json`; no training or autograd
performance claim is inferred from this forward-only harness.

## Invalid Attempt And Validation

`excluded/shared-cache/` retains an invalid first extended comparison in full.
Both purported revisions were the same executable: the shared Cargo cache did
not rebuild st-tensor after switching worktrees. Numerical success did not make
that a valid source comparison. No timing there supports an optimization claim.
The corrected driver uses mtime-only touches on the changed source and harness,
checks source bytes against Git, requires matching-root compilation logs, and
rejects identical executable hashes before running the benchmark. No cache files
were deleted and no source content was changed by these touches.

The selected source passes 479 Tensor CPU tests, 11 serial kernel tests, 32
self-supervision CPU tests (three existing ignored cases), 26 additional CPU-only
dispatch/autograd tests, scoped strict native/WASM Clippy, three real native Python
extension tests, seven WGPU-feature tests with a required live GPU probe, and
18 WASM forward/backward cases plus eight numerical-boundary smoke tests.
WASM tests execute the CPU-only build in Node, not a browser speed benchmark.
High-level Auto may use another CPU implementation for smaller WASM shapes.

## Replay

```bash
python3 -B -I verify.py
python3 -B -I test_verify.py
```

These verify archive integrity, source/result bindings and recorded gates, not
numerical replay. To rerun, use separate checkouts at the pinned revisions and
place the exact extended harness on the baseline. Then, from this directory:

```bash
python3 -B -I -S build_extended.py /absolute/baseline /absolute/candidate \
  /absolute/cargo-target /absolute/new-build-output
python3 -B -I -S measure_extended.py measure --directory /absolute/new-results \
  --baseline /absolute/new-build-output/baseline-extended \
  --candidate /absolute/new-build-output/candidate-extended \
  --torch-python /absolute/python --torch-site /absolute/site-packages
python3 -B -I -S verification_driver.py /absolute/candidate \
  /absolute/cargo-target /absolute/new-verification
```

Use the recorded Rust/bindgen versions and adapt machine-local paths in the
validation driver. It requires a clean candidate checkout. `measure_dense.py`
and `measure_crosscut.py` preserve the earlier protocols. Native, extension and
WASM binaries remain local under paths/hashes in `provenance.json`. Results,
validation logs, failed attempts and replay drivers are published.

Next: profile the remaining one-panel/deep-inner unpacked path, SIMD/kernel
throughput and actual high-level routing separately. Fewer A packs do not remove
the measured GEMM gap or establish a fastest-PyTorch comparison.
