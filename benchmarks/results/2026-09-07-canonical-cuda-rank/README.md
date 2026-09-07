# Canonical CUDA Rank Reference

Compiled native and benchmark source: `dc840c9725d11a445d5df109da62e9e7c2479b2d`.
Later commits are documentation/evidence only. Rust/backend/WASM source equals
`8a3624f89b60b1586167e5c2978b2d19848dadeb`; this study changes the comparison
reference, not the WGPU implementation. It measures no new browser speedup.

## Contract

The old CUDA selector used stable sort whenever any row contained duplicates,
even outside the retained set. The v2 rank-only harnesses now compare eligible
plain topk, index-repaired topk, stable sort and exact packed-integer topk.
Admission uses f32-rounded values and distinguishes retained ties from cutoff
ties. Every control checks exact output bits and canonical source indices.
Packed keys also handle mixed signed zeros and MidK. Non-finite inputs remain
outside this finite-fixture CUDA comparison and are rejected, not silently
ranked under another contract.

Output/explicit scratch allocation and geometry-only index words are outside
timing. Input-dependent key encoding, index repair, gather, and any internal
Torch workspace costs occur inside every operation. Nothing caches the answer
or value-dependent keys between operations. See the
[control contract](../../../docs/development/cuda_rank_reference.md).

All eligible CUDA controls rotate in one process, with two warmup batches and
twelve retained batches of sixteen operations plus completion. Exact readbacks
run outside timing after every batch. The primary CUDA reference is the
hindsight lowest sample mean among those controls. This is not online policy,
a confidence interval, GPU-event timing, or the fastest possible PyTorch claim.
Native WGPU and CUDA remain separate process blocks on the same named RTX 5090.

## Results

PyTorch **2.13.0+cu132**, native WGPU/Vulkan, Rust 1.98.1. Source-bound mode-555
native images retain matching before/after hashes. All **98 matched cases**
pass, retaining 318 WGPU fixed-control cells, 256 CUDA control cells and 704
real correctness-gated Rust UCB observations. Fixed WGPU/CUDA controls never
seed Black Cat. All cells and losing controls remain in [summary.json](summary.json).

The first two runs reproduce the preceding finite-bound study's fixture:
three seeds, TopK/MidK/BottomK, rows 2/64, cols=8193, k=65, and six WGPU tiles.
The generator is byte-identical to that study. Each run has 18 requests.

| Run | Best CUDA: Topk / Repair / Stable / Packed | Best-Fixed WGPU / Best-Fixed CUDA Range |
| --- | --- | --- |
| First | 4 / 3 / 11 / 0 | 0.721-1.724 |
| Repeat | 4 / 2 / 12 / 0 | 0.731-1.748 |

Eight cases per run favor the measured fixed WGPU control, while ten favor
CUDA. Four small-row untied TopK/BottomK cases still favor plain CUDA topk;
MidK also remains slower in WGPU. The four internal-tie repair candidates range
0.984-1.002x stable sort initially and 0.995-1.012x on repeat. These small
differences and winner changes do not establish a meaningful repair speedup.
Packed-key selection is correct but never the fastest control in these timed
fixtures, especially for MidK. It is retained as a negative performance result,
not substituted for the faster stable-sort reference.

The separate **54-case standard suite** also passes through the other rank-only
harness. It retains 36 topk and 18 stable-sort winners, with WGPU/CUDA ratios
0.953-3.948. These smaller widths/tiles use one fixed WGPU control per request;
do not pool them with the six-candidate best-fixed comparisons.

### Targeted Tail Ties

The previous 90 cases do not isolate duplicates only outside the retained set.
Two additional four-case diagnostic runs therefore use shuffled exact f32
values with repeated interior zeros but unique retained values and cutoff.
They are synthetic admission tests, not representative workload speedups and
not a revision of old published results.

Ratios below are CUDA topk / legacy CUDA stable-sort sample-mean latency:

| Kind | Rows | First | Repeat |
| --- | ---: | ---: | ---: |
| TopK | 2 | 0.489 | 0.488 |
| TopK | 64 | 0.585 | 0.586 |
| BottomK | 2 | 0.510 | 0.509 |
| BottomK | 64 | 0.589 | 0.585 |

The old rule would compare WGPU with stable sort, yielding apparent WGPU/CUDA
ratios 0.778-0.824. Admitting the equally correct topk control changes them to
**1.328-1.659**, now favoring CUDA in every targeted case. No WGPU kernel changed.
This demonstrates why out-of-band ties must not force an unnecessarily weak
reference and why WGPU wins must remain qualified by the actual CUDA control.

## Validation And Replay

Live Torch CPU and CUDA each pass 19 tests. Coverage includes partial/full k,
cutoff/internal/outside ties, f32 rounding collisions, signed zeros, extreme
finite/subnormal values, buffer reuse, packed-key recomputation after upload,
input/layout admission, and all timing-control samples. Dependency-free tests
also run without importing Torch. Six existing rank-adaptation receipt tests
and six existing matmul-rank validator tests pass. The matmul-rank harness,
public API, Rust feedback semantics, dependencies and CI jobs are unchanged.

The preceding finite-bound summary also replays byte-for-byte against the new
repository validators. Owned GPU jobs never overlap, including across hosts;
no same-host build overlaps timing. Availability checks are point observations,
not continuous/exclusive reservations. Existing CPU workloads remain untouched.

[raw-logs.tar.xz](raw-logs.tar.xz) contains the full reports, generators, source
bundle, native identities, tests/build logs and GPU-only availability records.
It excludes large executables and private session listings. Extract and replay
using the unchanged measured benchmark files, without Torch or GPU access:

```sh
python -I analyze_references.py --repo /path/to/SpiralTorch --output recomputed.json
```

The summary records hashes for all 35 raw files. A separate extraction and
offline replay reproduced the published summary byte-for-byte.

SHA-256:

```text
b02681da478523162c55963fc5e78263771fe871663718b62e6b0ab40551204c  raw-logs.tar.xz
d425d838053e6783bb936a1f650ca6f53383244103941bbc60a704722edcae1e  summary.json
```
