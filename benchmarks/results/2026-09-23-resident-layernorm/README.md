# Resident LayerNorm foundation, 2026-09-23

Measured implementation: `a37acdcfaf5aa54a89695723288e9730b295ec68`.
The new explicit backend keeps statistics, normalized tape, VJPs and updates on
GPU. Ordinary Tensor routing is unchanged. See the [protocol and replay guide](../../layer-norm-resident/README.md).

## Correctness

- 198 WGPU backend tests, 31 pure-contract tests, 11 existing LayerNorm autograd
  tests and 4 existing numerical tests passed, with real native WGPU enabled.
- Native and WASM strict clippy, formatting and 12 source-stable validation
  stages passed from the same clean commit.
- Chrome 153.0.8010.53 / BrowserWebGpu passed seven boundary fixtures with all
  seven nonempty VJP masks, guarded failures and a 400-step resident learning
  loop. Loss went from 1.98799634 to 7.81713472e-10 without intermediate reads.
- A subnormal affine-product bug was caught (`0` versus `-0.09223365`), fixed,
  and retained as both a native and browser regression. Comparison tolerance
  stayed `2e-5 * (1 + abs(reference))`.

## Exploratory Timings

Apple M4 native Metal; PyTorch 2.12.1 CPU/MPS; four host threads. Each cell is
the median of eighteen intervals after three warmups. All 540 measured
intervals are in the JSON files, including slower routes. Milliseconds include
uploads, forward, all VJPs and four owning CPU observations.

| Shape | Rust CPU | Old Hybrid WGPU | Resident WGPU | PyTorch CPU | PyTorch MPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2x3 | 0.002 | 4.135 | 1.562 | 0.077 | 1.792 |
| 8x257 | 0.020 | 4.295 | 2.283 | 0.078 | 1.576 |
| 32x256 | 0.067 | 4.495 | 2.529 | 0.083 | 1.599 |
| 64x768 | 0.387 | 5.883 | 2.322 | 0.166 | 1.676 |
| 128x1025 | 1.016 | 8.700 | 5.033 | 0.262 | 2.741 |
| 256x256 | 0.506 | 7.035 | 3.948 | 0.179 | 2.074 |

Resident WGPU improves on the old hybrid in these samples, but CPU is still
faster, and MPS is faster in five of six cells. The extended-range math and
16-byte normalized tape are correctness-first costs, not an optimal kernel.
One shared-machine round, distinct host overheads and four terminal reads do
not establish general framework speed, wholly resident training speed, or
model-quality improvements. Native values matched the CPU reference within the
unchanged tolerance; no all-input exactness claim is made.

## Negative Controls

The exploratory rate-0.05 learning fixture misses the assumed 1e-4 loss-ratio
goal after 400 steps on Rust CPU, resident WGPU and PyTorch CPU. Its unchanged
data and failure remain recorded, alongside the established rate-0.1 fixture
that meets the original goal.

PyTorch MPS's affine-only training control diverged on this installation. The
direct operator probe reproduced wrong affine outputs only for the three masks
omitting dx; CPU passed all seven, and MPS passed masks requesting dx. Computing
an extra unused input gradient restored the 400-step controls. That workaround
is explicitly labeled **different work**, not used to replace the failed
masked result. The all-gradient timing route is separate and passed its value
checks. This is a fixed-fixture local observation, not a universal PyTorch bug
claim. No third-party package was patched.

## Integrity

`validation.json` records every stage and its command, including failed attempts.
`measured-source.json` identifies the clean measured sources. `raw-manifest.json`
covers 260 local files, including dirty prototypes, full logs and generated
binaries. `public-manifest.json` covers the published data payload (this prose
is separately versioned by Git). Full raw artifacts remain local.

```sh
python3 -I -B benchmarks/layer-norm-resident/evidence.py verify \
  /path/to/resident-layernorm-20260923 benchmarks/results/2026-09-23-resident-layernorm
```
