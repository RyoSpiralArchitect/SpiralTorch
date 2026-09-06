# MidK Candidate-Tile Selection

## Shared Kernel Change

Exact MidK with at most 32 tiles now assigns one selection workgroup to each
candidate tile, instead of scanning every tile inside one workgroup per row.
Each candidate binary-searches other sorted tiles for its global rank. The same
total (value, source-index) order makes valid output destinations unique.
Workgroups initialize only the disjoint missing-value tail, never the valid
prefix. Host and resident Rust dispatch share the grid; Python/WASM use it
without new API or policy knobs. Planner tile choices and the >32-tile merge
remain unchanged. This is still two ordered compute stages, not shader fusion.

Baseline: `2635b9675a3ee0057be3ae3db2027c564bc88156` (new isolation harness,
unchanged pre-optimization kernel).
Candidate: `eaa0ffb8d0d560d3778cc9d92934c10c2a947833`.

## Native Comparison

Furnace reports RTX 5090 for both WGPU Vulkan and PyTorch `2.13.0+cu132` CUDA.
Eight runs, 18 cases each, all pass exact values/canonical indices, clean
source/build binding, stable image and pre/post foreign-compute-PID gates.
These gates are not an exclusive GPU reservation. Both protocols were run in
baseline/candidate/candidate/baseline order; frameworks use separate blocks.
All runs disable readback diagnostics.

The shared request hash is
`f987634dda04fa6dc7ab232441c62cdb184d02771f18afaf184b015188b600e7`.
Seeds 17/29/43 cross shapes `(2,32,257)` and `(3,128,1025)`, three rank kinds,
and k=7. Bounded integer operands keep fp32 projection and cutoff ties exact.
Native matmul is scalar/sequential, rank tile_cols=256; CUDA TF32 is disabled.
This is not a general floating-point, configuration-search or full-model result.

### Resident-Only Protocol

`--resident-only` measures sixteen composed calls plus a completion fence per
interval, divided by sixteen. No maps/uploads occur between intervals. Inputs
are fixed; correctness is checked before and after all samples, not after each
sample. The report records this boundary. Optional probes still run later in a
separate process, never interleaved into these samples.

Microseconds per chain below are medians of three seed medians, each from twelve
samples after two warmups. CUDA uses preallocated matmul/stable-sort outputs and
the same repetition count. These are host-API timings, not GPU-event times.

| Kind | Columns | Baseline 1 | Candidate 1 | Candidate 2 | Baseline 2 | CUDA 2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TopK | 257 | 56.45 | 57.12 | 56.46 | 56.53 | 16.23 |
| TopK | 1025 | 64.73 | 65.06 | 65.25 | 65.45 | 16.75 |
| MidK | 257 | 49.48 | 49.16 | 48.76 | 48.85 | 16.22 |
| MidK | 1025 | 75.12 | 62.64 | 62.11 | 75.17 | 16.75 |
| BottomK | 257 | 56.45 | 56.65 | 58.96 | 56.72 | 16.19 |
| BottomK | 1025 | 65.23 | 65.24 | 65.27 | 64.84 | 16.74 |

MidK/1025 improves about **16-17%** on this isolated boundary. MidK/257 changes
little, and controls are mixed: retain the slower BottomK/257 second run instead
of claiming universal throughput improvement. CUDA remains about 3.7x faster
for the target MidK/1025 chain. No confidence interval or allocator-level cause
of the earlier readback-pool regression has been established by these runs.

### Full Protocol

The original six-mode rotated protocol is preserved. It validates each timed
mode outside its resident interval and also measures one complete composed
call including final readback. The following are **MidK/1025 only**, in us.

| Boundary | Baseline 1 | Candidate 1 | Candidate 2 | Baseline 2 |
| --- | ---: | ---: | ---: | ---: |
| Composed call, final readback included | 91.69 | 81.40 | 81.50 | 91.95 |
| Resident composed call, per chain | 76.87 | 58.08 | 58.33 | 73.79 |

The readback-inclusive improvement is approximately **11%**. Do not pool the
rotated resident result with the isolated protocol: workload history and
completion boundaries differ, and their absolute resident times differ too.

## Browser Comparison

Four real Chrome runs use the same current fixture and baseline/candidate/
candidate/baseline assets, with 54 cases and 1,180 assertions each; all pass.
Fifty-one cases include timings; three additional cases are validation-only.
The targeted rank-only shape has two rows, 1025 columns and k=7. It is **not**
the native projection-chain workload. Ten intervals each dispatch eight repeats
and await a four-byte asynchronous completion fence; results below are median
us per repeat. No timed upload/full readback occurs. The browser clock is coarse,
and these timings include event-loop/fence overhead rather than GPU-event time.

| MidK Tile Columns | Baseline 1 | Candidate 1 | Candidate 2 | Baseline 2 |
| --- | ---: | ---: | ---: | ---: |
| 32 (33 tiles, unchanged fallback) | 1150.00 | 1150.00 | 1150.00 | 1150.00 |
| 256 (target) | 112.50 | 75.00 | 75.00 | 100.00 |
| 1024 | 150.00 | 162.50 | 150.00 | 156.25 |
| 1025 | 193.75 | 187.50 | 200.00 | 200.00 |

The target improves in both candidate runs, but larger tiles are mixed and the
33-tile fallback remains slow. This is not an all-browser or all-shape speedup.
Browser reports expose `BrowserWebGpu` without a physical adapter name. They bind
assets and the fixture by hash, not by native source/build attestation:

- Fixture: `20fb86b3b84af4180a8a94bbb0b8ff5da7d7575f65dfdaf9908638a3634570f7`.
- Baseline WASM: `be8994cd4a7c24cf743d0ca3855613cd6fc84c483902cc54e216cd055e322b23`,
  the same asset hash retained with PR #2074, rerun against the new fixture.
- Candidate WASM: `573f30b2d237b8e498ffdd2e9b5fa66f7e19d7ec8f8b78ab7a3fb66e1d2230be`.

## Correctness And Scope

- Apple M4 and Furnace RTX 5090: 69 backend unit tests plus shader validation
  pass. Tests cover the 32/33-tile grid, ties, signed zeros, non-finite candidates,
  k=1/7/full-width, repeated finite/all-NaN transitions, owned snapshots and the
  storage-sort path for larger tile strides. They do not exhaust all fp32 inputs.
- Real Chrome composed matmul/rank fixture: 27 cases / 4,887 assertions pass,
  including scalar, register and compensated projection configurations.
- Rebuilt private Python wheel: 97 passed, one optional skip. TypeScript contract,
  native/backend and wasm all-target strict clippy, pinned fmt, and six harness
  unit tests pass. No package version or public release changed.

The [resident guide](../../../docs/performance/resident_rank.md) documents APIs
and the new isolation option. CUDA is a comparator, not a silently selected
SpiralTorch execution fallback. This change preserves exact rank semantics; it
does not establish language-model quality gains or finish backend optimization.

## Artifact Hashes

| Artifact | SHA-256 |
| --- | --- |
| midk-resident-baseline-2635b967.json | 31afe1176c7dde1f1a5e3fbec84ce046a0fd9a299f19ba3d7fd5aabb947abe51 |
| midk-resident-baseline-repeat-2635b967.json | 6d42cd50a66e6ab63fa6e6807c112b1290a70b7a2894845311923ca973235873 |
| midk-resident-candidate-eaa0ffb8.json | 84caa3b5a9556d2319bcf99a558b8d912dfc208c97bbf2302d546ac8c16ed4b9 |
| midk-resident-candidate-repeat-eaa0ffb8.json | 5283d00800ff06d31d1253e244818fc4c02ac0063c7e500ccb11a2e853702bb8 |
| midk-full-baseline-2635b967.json | 961f13b666226123c5f8c245f5ba220c037481979c04b78dd916ef3b3df99ec1 |
| midk-full-baseline-repeat-2635b967.json | be533aede07e48ac6c8b499699e1f92a7d6c41fa643ea395d31c20c4a43b9f4c |
| midk-full-candidate-eaa0ffb8.json | 54418c342126e75765b36fcdf6bd938b38856939e9340df77efaa1586ca2b0f5 |
| midk-full-candidate-repeat-eaa0ffb8.json | e33c2a21d220ec2f79b0fd59f403e395bd1cc868a1e04b64264529dd474f3a0f |
| spiraltorch-midk-browser-baseline.json | 2ed2c88a408ffab380334d4a8099705c99b9aad557675e80799317ccae944b79 |
| spiraltorch-midk-browser-baseline-repeat.json | 14e8f37d9aa36e9390167611c5afa9ecbd51afe762a359581c56dd934bd70fda |
| spiraltorch-midk-browser-candidate.json | a1834e06c2d76caf9eb986f61777290c9822a9c0b29e1531a3c69f9dc9e2a394 |
| spiraltorch-midk-browser-candidate-repeat.json | 42e534c6cfe40fa42a898c70642987eae8f88f5cd26005d71953275e707a1494 |
| spiraltorch-midk-tile-browser-chain.json | 386cbcc25284e7f12edfe62ce75e47a6b13b58975ffd8a2a4935688d79ff5573 |
| spiraltorch-midk-tile-furnace-tests.log | 0aa256fef226e6daf95e0d849b2ed95bd3a59e517e8196a965cd26ad119552d7 |
