# Owning VJP capture reuse: verified, throughput still mixed

Baseline: `5fa0859e004f85b2b5e2d78f549af9b033130837`, tree
`0914d08850ba7c7f2461ff2efbd3f6b76ceeacb7`.
Candidate: `5fddb415bfbb0f64366db6d45b7675db17a16f61`, tree
`1bcb92ea916ee0d63ef2b10732e8b4d771bcab6b`.

The candidate reuses complete, unobserved VJP output versions, up to four
batches and 32 MiB of values/guards per autograd workspace. A retained member,
view or weak owner pins the entire batch. Reuse clears its guard before the
unchanged capture pass. Busy/oversized batches allocate separately: no waiting,
aliasing or CPU fallback. Gradient copies and numerical checks remain.
See the [implementation contract](../../../docs/resident_vjp_capture_pool.md).

This reduces resource creation by construction, not necessarily execution time.
For 25 parameters, each reused VJP avoids 27 new buffers and 26 output bind
groups. The ordinary two-VJP loop can reuse two batches after its warmup.
All three optimizer modes are affected; plain SGD is not an unchanged control.

## Verification

- All 69 source-bound regression stages passed, including backend 151 tests,
  NN 760 library tests, six resident integration tests, Python GPU/CPU-only
  clients, actual browser execution, and frozen forward controls.
- The new shared native/browser scenario runs nine VJPs, retains seven versions,
  checks pending reads and masked failures, and observes after workspace drop.
  Native pool tests additionally check weak owners, budget accounting, spill,
  signed zero, scalar/empty/tail shapes and no-retention behavior.
- Independent eager PyTorch CPU/MPS replay passed 432 cases and 197,976 recorded
  tensor/scalar comparisons; maximum absolute error `1.811981201171875e-5`.
  Four existing tiny-tail reference gaps remain separately labelled, not matched.
  The tensor-WGPU fractional-GL test remains ignored, not passed.
- Both ordinary performance runs passed complete state/receipt validation;
  maximum absolute error versus the recorded Torch oracle `3.259629011154175e-8`.
- Separate host diagnosis validated 1,080 instrumented native intervals and
  1,080 ordinary controls, plus 54 browser diagnostic/control pairs. All 324
  captured states matched the ordinary Torch oracle at the same maximum error.
  These clocks measure host enqueue/wait time, not GPU phase cost. Before/after
  host diagnoses are separate runs, not paired throughput measurements.

## Ordinary Performance

Three optimizer modes, three seeds, three graph shapes/depths, two receipt
cadences, two warmups and eight retained intervals per cadence. Each interval
resets outside timing and performs eight updates. Native baseline/candidate/
Torch order rotates; browser baseline/candidate alternate. Run B repeats the
entire matrix with reversed mode order, without retrying selected cells.
There are 4,320 retained intervals across both runs and both execution routes.

Ranges below span the nine recipe-specific median **baseline/candidate elapsed
ratios**, not confidence intervals. Above 1 favors reuse; below 1 is slower.

| Mode / Receipt Cadence | Native A | Native B | Browser A | Browser B |
| --- | --- | --- | --- | --- |
| SGD / immediate | 0.851-1.272 | 0.981-1.118 | 0.726-2.188 | 0.975-1.281 |
| SGD / deferred | 0.777-1.287 | 0.704-1.053 | 0.733-1.251 | 0.903-1.026 |
| EMA / immediate | 0.919-1.227 | 0.926-1.120 | 0.927-1.146 | 0.954-1.100 |
| EMA / deferred | 0.714-1.255 | 0.975-1.136 | 0.890-1.320 | 0.930-1.189 |
| Clipped EMA / immediate | 0.925-1.099 | 0.999-1.086 | 0.989-1.250 | 0.982-1.073 |
| Clipped EMA / deferred | 0.964-1.082 | 0.938-1.055 | 0.928-1.066 | 0.912-1.025 |

Do not interpret the largest ratio as a representative speedup. For example,
run B's native SGD/deferred `[2,129,32]`, depth 4, seed 29 slowed from a median
32.89 to 46.73 ms. Retained ranges were 25.12-95.55 and 25.45-114.50 ms,
respectively. Negative results and substantial variance are retained, not
filtered. No stable overall throughput improvement or routing threshold follows.

Eager Torch MPS was faster in every native comparison in run B; run A was mixed.
Its checks/rollback behavior differ from SpiralTorch's guarded updates. These
are bounded matched-math workloads, not the fastest possible Torch baseline.
Native execution used Apple M4/Metal; browser execution is BrowserWebGpu, with
physical GPU identity UNKNOWN. Host exclusivity is UNKNOWN. A load/thermal
snapshot is retained, but cannot establish the cause of timing variation.

## Evidence And Next Boundary

`summary.json` contains every recipe, test count, source/product binding and
host diagnosis. `manifest.json` indexes compressed raw records; `verification.json`
records their independent decompression/original/Git/product integrity check.
Frozen binaries stay outside Git with their hashes; full captures, stderr,
timings, helpers, generated client text and relevant source pairs are archived.

The pool is an ownership-safe foundation for directing backward writes into
owning outputs. It does not yet eliminate scratch-to-output gradient copies.
No claim of GPU-copy bottleneck attribution, universal speedup, CUDA performance,
FT quality improvement, optimizer-history resume, release, push or merge is made.
