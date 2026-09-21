# Resident NeRF direct graph: M4 application-path comparison

Measured source: `bea3cdef` (full commit and file hashes in `source.json`).
Native Apple M4/Metal; Chrome 153.0.8010.48 WebGPU; PyTorch 2.12.1 CPU/MPS.
This was a shared desktop, not a dedicated idle benchmark host.

## Result

Connect sampling -> `ResidentGraph::forward_tensor` -> compositing through
`ResidentNerf::render_graph`, avoiding the old stable-workspace input/output
value copies. It remains forward-only and position-only.

Three full rounds, twelve shape/field conditions, bursts 1 and 4, nine paired
measured blocks after three warmups: **3,888 measured intervals**, all retained
in `results.json`. Ratios below are medians of paired staged/direct times per
cell, followed by a descriptive geometric mean across the 24 cells.

| Runtime | Geomean staged/direct | Range | Faster cells |
|---|---:|---:|---:|
| Native WGPU | 1.1298 | 0.9641-1.3653 | 23/24 |
| Browser WebGPU | 1.1307 | 1.0000-1.4000 | 20/24 (4 ties) |

The native 65 rays x 64 samples, hidden-32, burst-1 cell regressed about 4%.
The browser clock is coarse at these durations; ties and small differences
are not evidence of equivalence. These are descriptive measurements, not
confidence bounds, universal speedups or isolated GPU timings.

Selected interval medians in milliseconds, with all other cells in the JSON:

| Rays x samples / hidden / burst | Native direct | Browser direct | Torch CPU | Torch MPS |
|---|---:|---:|---:|---:|
| 1x1 / 0 / 1 | 0.501 | 0.500 | 0.084 | 1.229 |
| 65x64 / 32 / 1 | 0.742 | 0.700 | 0.374 | 1.429 |
| 1024x64 / 32 / 4 | 7.073 | 5.600 | 17.400 | 9.875 |
| 256x256 / 0 / 4 | 3.839 | 3.500 | 7.441 | 3.763 |

Small jobs favor CPU strongly. The affine 256x256 burst-4 case also shows eager
MPS slightly faster than native direct. Do not turn these data into a blanket
"faster than PyTorch" claim. The controls use eager stabilized f32 prefix
integration, while WGPU uses checked compensated shaders; neither identical
kernels nor `torch.compile` were compared.

## Correctness And Limits

All accepted outputs pass `4e-7 + 4e-6 * abs(reference)` against independent CPU
f64 geometry/integration with f32 NN. Maximum absolute final-output error:
WGPU (both routes and both runtimes) and MPS `1.25169754e-6`; CPU `1.19209290e-7`.
Every timed output is checked against its route reference; final outputs and
the WGPU reference additionally pass the independent oracle.

Shape/device/queue admission, inherited invalid positions, retained output
ownership and repeated rendering are covered by real GPU checks. Full backend
validation: 171 unit tests plus one shader integration test; 11 Python admission
tests; strict native and wasm32 Clippy; native and actual Chrome execution.

Timings include sampling/RNG, NN, compositing, per-render allocation/submission,
and a final owning RGBA/guard map/copy with queue completion. They exclude setup,
initial uploads, pipeline creation, validation and serialization. Burst 4 reads
only its final result, not every temporary guard. No NeRF training, GPU VJP,
image quality, cross-vendor result or Python NeRF wrapper is claimed.

Browser Rust WGPU reports `Other/BrowserWebGpu` without a device name; a separate
probe reports apple/metal-3/non-fallback. This probe does not attest the exact
Rust runtime adapter. Do not relabel the runtime report as named M4 telemetry.

## Retained Attempts

`negative-attempt.json` preserves the initial eager MPS `expm1` failure (up to
`3.93390656e-6`) and stage diagnostics. RNG/positions agreed; thin-opacity error
accumulated. A fourth-order thin-opacity expansion fixed the timed CPU/MPS
controls without widening tolerance or changing the f64 oracle.

The first complete run at `edd0b86d` was excluded because the publication gate
incorrectly demanded a named native GPU class from browser WGPU. The adapter
reporting rule was corrected and regression-tested, then all validation and
the full three-round protocol were rerun at `bea3cdef`. The excluded run's
complete compact results are in `excluded-metadata-gate.json`, not the headline
ratios. All original preflights, arrays, binaries and logs remain local and are
hash-listed, not overwritten.

## Verification

- `results.json`: complete accepted timings, errors, input/oracle hashes.
- `validation.json`, test logs: source-stable commands and outcomes.
- `source.json`: clean measured commit and source-file hashes.
- `local-raw-manifest.json`: original payload hashes/lengths, relative to local
  `Library/Logs/SpiralTorch/nerf-direct-graph-20260921`.
- `manifest.json`: public archive fixity.

```sh
python3 -I -B benchmarks/nerf-direct-graph/archive.py verify benchmarks/results/2026-09-21-nerf-direct-graph
```

Add `--source-root` for the measured source checkout and `--raw-root` for local
payload verification and exact summary recomputation. These check fixity, not
numerical reexecution. For actual reruns use the
[benchmark protocol](../../nerf-direct-graph/README.md) and the commands in
`validation.json`; preserve new output paths and the three rotating runtime
orders. Raw arrays and executable artifacts are intentionally not committed.
