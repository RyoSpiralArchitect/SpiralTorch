# Matmul Accumulation: Precision Without A Global Slow Fallback

Captured 2026-09-04. Rust owns three explicit modes shared by resident Python
and WASM: `sequential` (unchanged default), `tiled` (two-level K-tile partial
sums), and `compensated` (Neumaier correction with integer-defined rounding
boundaries). Both scalar and register-2x2 kernels expose the same contract.
No automatic dispatch, host Tensor precision default, or training policy changes.

## Correctness First

The gate remains `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`, against
float64 reference math. Every failed variant is retained, receives **no timing**,
and makes the overall run exit nonzero. Consequently all four reports here are
intentionally non-green: the old sequential mode still fails long-K fixtures.

| Run | Sequential | Tiled | Compensated |
| --- | ---: | ---: | ---: |
| Native, 4 shapes x 3 seeds x 2 kernels | 20/24 | 24/24 | 24/24 |
| Native follow-up, 2 shapes x 10 seeds x 2 kernels | 28/40 | 40/40 | not requested |
| Browser, 3 shapes x 3 seeds x 2 kernels | 14/18 | 18/18 | 18/18 |
| Browser repeat, 4 shapes x 3 seeds x 2 kernels | 20/24 | 24/24 | not requested |

The ten-seed native follow-up tests `128x3072x768` and `512x512x512` with seeds
17, 29, 43, 59, 71, 89, 101, 127, 149, 173. On the long-K shape alone,
sequential passes 4/10 seeds per kernel; tiled passes 10/10. All PyTorch CUDA
controls pass. Native inputs use Python `random.Random`; browser inputs use
xorshift32, **not the same bytes across platforms**. Each within-platform
comparison uses identical inputs for all variants (and for native PyTorch).

On native `128x3072x768`, the three-seed maximum absolute errors are about
`4.49e-5` sequential, `1.16e-5` tiled, and `1.09e-6` compensated. The relative
term explains why a maximum error above `1e-5` need not fail the gate.
These are fixture measurements, not universal error bounds.

## Speed And Cost

Milliseconds below cover **16 separate resident calls plus synchronization**,
not one kernel and not GPU timestamps. Values are medians of seed medians;
allocation, uploads and readback are excluded. Every seed has 5 warmups and
20 randomized paired timing samples. Native PyTorch uses float32, TF32 off,
preallocated output, one CPU thread and the same call count.

| Workload / mode | Tiled ms | Comparator ms | Qualified seeds |
| --- | ---: | ---: | --- |
| RTX 5090, 128x3072x768, scalar | 3.431 | PyTorch CUDA 0.457 | 10/10 each |
| RTX 5090, 512 cubed, register-2x2 | 1.124 | PyTorch CUDA 0.239 | 10/10 each |
| Browser, 512 cubed, scalar | 32.3 | sequential 32.8 | 3/3 each |
| Browser, 512 cubed, register-2x2 | 14.3 | sequential 14.3 | 3/3 each |

For long-K native inputs where **both** tiled and sequential qualify (4 seeds),
the median paired tiled/sequential ratio is 1.004 for scalar and 0.996 for
register-2x2. This supports roughly unchanged throughput, not a speed win.
PyTorch CUDA remains substantially faster: about 7.5x on long-K scalar and
4.7x on blocked 512-cubed, at this API/queue boundary.

Compensation is deliberately opt-in. In the three-seed long-K native run,
scalar compensation takes 36.29 ms versus tiled 3.49 ms; register-2x2 takes
79.53 ms versus 5.33 ms. Browser compensation takes approximately 543-560 ms
versus tiled 34-75 ms. Do not install it as a global fallback.

## Rejected Optimizations And Regression Tests

A naive floating-point Neumaier implementation returned fifteen zeros for
`[1e8, 1, -1e8]` repeated 1025 times, rather than fifteen 1025s, on NVIDIA.
The original failure excerpt and prototype patch are retained. WGSL permits
[reassociation](https://www.w3.org/TR/WGSL/#floating-point-evaluation), so the
algebraic expression alone does not establish a portable compensated sum.
We did not inspect generated machine code and do not claim a proven compiler cause.

The accepted shader defines f32 round-to-nearest-even addition through integer
bits at three boundaries: sum, dominant-minus-sum, and residual-plus-smaller.
An attempted reduction to two boundaries failed the browser cancellation
preflight; `rejected-two-rounding-boundaries.json` preserves that failure and
its module hash. It is not a benchmark result and its candidate was rejected.
All three boundaries were restored before the four accepted reports above.

Real-device tests cover 16,784 rounded-add bit pairs, both-kernel cancellation,
tile/layout variants, ownership and chaining. Two browser xorshift cells
(seed 29, row 124, column 630; seed 43, row 26, column 183) have frozen float64
references in the Rust test. The native MT fixtures remain in the full benchmark.
The integer helper test handles subnormal/non-finite **bits**; it does not
promise that complete floating-point dot products avoid overflow, flushing,
product error, correction error or non-finite propagation.

## Provenance And Reproduction

- Native: Furnace RTX 5090, NVIDIA Vulkan through WGPU versus PyTorch 2.13.0+cu132
  CUDA on the single visible GPU. This is **not a native SpiralTorch CUDA GEMM
  optimization**. Existing CUDA/SpiralK/Golden/Black Cat work is retained unchanged.
- Browser: isolated headless Chrome, Apple `metal-3` non-fallback adapter probe.
  The probe is separate from Rust; WGPU 0.20 cannot attest exact browser adapter
  identity. A four-byte completion fence is included in browser synchronization.
- The private 0.4.27 Linux wheel is not a PyPI artifact. Raw reports retain the
  loaded extension/WASM hashes, harness hashes, adapter metadata and raw samples.
- `build-source-files.sha256` identifies the mirrored source used for the measured
  binaries. `reviewed-source-files.sha256` differs only in a source comment and
  the corrected browser-fixture regression test after measurement; runtime code
  is unchanged. The latter manifest was checked against both worktrees.
- Both builds use the source based on PR2062 head
  `687b3ca7dae91b7d3c945c97122634e7e74e9213` plus this accumulation change.
  The isolated Furnace worktree has an older detached Git HEAD plus mirrored
  files; its HEAD alone must not be used as binary provenance.

## Validation

- Backend: 9 real-device resident tests passed; native and wasm32 backend
  Clippy passed with `--all-targets -- -D warnings` (pre-existing vendor warnings remain).
- Tensor: 472 unit tests plus 69 integration tests passed, 1 existing ignored.
  The broad run found an old test still expecting automatic int8 above a size
  threshold. Only that expectation was corrected to cover explicit opt-in/off;
  runtime policy was not changed. The default-f32 integration test also passed.
- Python: all 3 resident tests passed with the private GPU wheel; CPU-only
  `cargo check --no-default-features --features python-default,cpu` passed.
- WASM: all 7 Node test scripts and the generated TypeScript contract passed.
  Browser preflights include actual compensation, mixed-policy GPU chaining,
  invalid JS types and overlapping snapshots; matrix results are tabulated above.
- Benchmark preflight: 6 tests passed. Pinned nightly rustfmt, workspace
  inventory and `git diff --check` passed. Raw benchmark gates were not relaxed.

Recompute all aggregates and check that failed variants were never timed:

```bash
node benchmarks/results/2026-09-04-matmul-accumulation/summarize.cjs
```

The checked-in `summary.json` is that command's output. Runtime commands are in
the [resident API guide](../../../docs/resident_webgpu_matmul.md). Use explicit
`--tiles-mnk '16,16,16'`, both kernels and all three accumulations. Native first-run
shapes are `1,768x768,768;32,768x768,768;128,768x768,3072;128,3072x3072,768`.
Browser first-run M/K/N shapes are `7,3073,65;64,64,64;128,3072,768`;
the tiled repeat adds `512,512,512`. Do not interpret the retained nonzero exits
as setup failures or remove failing sequential cells to make the reports green.
