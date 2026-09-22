# Pointwise-first resident graph input: M4 / browser / Torch

Measured source: `7775f6e0de13ac072267db20cd99a5b9f2a8d4d6`.
This is a fixed-submission, same-binary comparison, not a before/after claim.
Both paths apply the same **input ReLU**, field parameters, sample/NN/composite
chain and terminal owning RGBA/guard read. Direct pointwise input avoids a
temporary value allocation and packing dispatch for noncontiguous views.
The original shader pipeline is shared; only immutable layout metadata differs.

## Results

Three rotated serial rounds, 12 conditions, bursts 1/4, nine paired blocks:
**3,888 retained intervals**, no conditions removed. `results.json` contains
every interval, per-condition timing/error, hashes and runtime identity.
Ratios below are packed / direct-view; above one favors direct views.

| Runtime | Geometric mean | Cell range | Cells above 1 |
| --- | ---: | ---: | ---: |
| Native M4 / Metal | 1.051432 | 0.879049-1.371614 | 15/24 |
| Browser WebGPU | 1.058481 | 0.722222-1.400000 | 17/24 |

The independent CPU f64 geometry/integration + f32 NN oracle accepted every
route at `4e-7 + 4e-6 * abs(reference)`; maximum normalized error was
0.526675 (limit 1). Eager Torch 2.12.1 CPU/MPS executes the same input ReLU
and remains a full application-path control, not matched shaders or
`torch.compile`. No general Torch superiority or training-quality claim.

The 1x1 conditions are already contiguous and save no pack. Their native paired
ratios span 0.930269-0.997354 and browser ratios 0.846154-1.181818, illustrating
substantial shared-desktop/coarse-clock variability. Noncontiguous conditions
also regress: 65x64 / affine / burst1 browser ratio 0.722222; 256x64 / affine /
burst4 native ratio 0.879049. Do not dismiss these or derive a dispatch threshold
from the modest aggregate improvement. No isolated GPU timestamps, dedicated
hardware, scene-quality, cross-vendor or automatic training-path claim.

Native is Apple M4 / Metal. Chrome is 153.0.8010.53. Its separate nonfallback
adapter probe does not attest the Rust runtime's BrowserWebGpu device.
Both routes keep three submissions per render. Setup, compilation, validation
and serialization are excluded; allocation, encoding, submission and terminal
owning copy/map/completion are included. Burst4 observes only its last output.

## Validation And Earlier Attempts

All **34 clean-source stages** succeeded on the measured commit: formatting,
native/WASM strict clippy, 183 backend tests with real GPU checks enabled,
30 layout/kernel contract tests, 9 upper-NN inference/training tests and
26 protocol/archive tests. Native/browser probes exercise six view layouts,
parameter broadcasts/residual inputs, stable dispatch transitions, held outputs
and masked inherited errors outside timing. Existing staged/direct,
submission-count and row-input modes and prior published archives still pass.
The default no-prelude Torch path was additionally replayed across all 12
conditions (`legacy-torch-default` in exploratory receipts).

`exploration.json` preserves all earlier attempts and a separate three-round
screen: native 1.071354, browser 1.051941. They are not pooled into final results.
One initial oracle mutation test failed because its synthetic weights were all
zero, so changing input could not affect output. The fixture now has an active
weight and verifies both the ReLU effect and nonfinite guard preservation.
The failure and corrected rerun are retained, not overwritten.

## Replay And Fixity

Follow [the protocol](../../nerf-pointwise-input/README.md) from the measured
commit, serially, with four Cargo/Rayon workers and Torch threads 4/1.
Exact executed commands and times are in `validation.json`. Reuse one fixed
native input fixture for all Torch rounds.

Raw arrays, native/WASM binaries, receipts, source snapshots and all attempts
remain local: **484 files / 56,920,596 bytes**, indexed by
`local-raw-manifest.json`, under
`/Users/ryospiralarchitect/Library/Logs/SpiralTorch/graph-pointwise-input-20260922`.
No raw arrays or binaries are published.

```sh
python3 -I -B benchmarks/nerf-pointwise-input/evidence_pointwise.py verify benchmarks/results/2026-09-22-nerf-pointwise-input
# Optionally add --raw-root LOCAL_RAW and --source-root FROZEN_CHECKOUT.
```

Verification checks manifest bytes, complete grids/receipts and recomputed
statistics; optional arguments check raw/source hashes. It does not rerun GPU
or numerical calculations. Preserve this archive rather than updating its
historical measurements after later backend changes.
