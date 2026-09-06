# Resident Matmul To Exact Rank

## Execution Evidence

Two clean-source runs of `49603681647e0dc4c7c48d8f457a52ccc0bce29d`
passed all 18 projection/rank cases on Furnace: WGPU Vulkan and PyTorch
`2.13.0+cu132` CUDA, both reporting the NVIDIA GeForce RTX 5090. Each run
passed source/build binding, execution-image stability and preflight/postflight
foreign-compute-PID gates. The process gate is not an exclusive reservation.
The request SHA-256 is identical in both runs:
`f987634dda04fa6dc7ab232441c62cdb184d02771f18afaf184b015188b600e7`.

The fixture crosses seeds 17, 29 and 43, projection shapes `(2, 32, 257)` and
`(3, 128, 1025)`, and TopK/MidK/BottomK with k=7. Bounded integer operands
make fp32 projection values exact, including ties. CUDA TF32 is disabled.
All CUDA modes now use stable sorting and check selected values **and indices**
against the canonical reference. These fixtures do not prove arbitrary
floating-point rank stability or model quality.

## Bounded Timing Result

Each cell is the median of three seed medians; each seed has 12 samples after
two warmups. Native bridge modes rotate within a measurement block. Frameworks
are measured in separate blocks, not interleaved.

Host and device bridge times below are milliseconds. Both start with resident
operands and include the same final rank readback. Only the intermediate logits
path changes: a full host map/upload versus a device-to-device copy.

| Kind | Columns | Host Run 1 | Device Run 1 | Host Run 2 | Device Run 2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TopK | 257 | 1.666 | 0.785 | 1.650 | 0.769 |
| TopK | 1025 | 1.694 | 0.766 | 1.665 | 0.760 |
| MidK | 257 | 1.669 | 0.773 | 1.667 | 0.765 |
| MidK | 1025 | 1.700 | 0.768 | 1.670 | 0.762 |
| BottomK | 257 | 1.676 | 0.774 | 1.638 | 0.764 |
| BottomK | 1025 | 1.710 | 0.772 | 1.674 | 0.768 |

Removing the intermediate host round trip reduces these small-head group
medians by about 53-55%. This is not zero-copy aliasing or a fused kernel:
matmul, the owned GPU copy, and rank still use separate queue submissions.

Resident times below are microseconds per operation, with 16 chains plus a
completion fence divided by 16, no host maps, and preallocated outputs.

| Kind | Columns | WGPU Run 1 | CUDA Run 1 | WGPU Run 2 | CUDA Run 2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TopK | 257 | 67.97 | 16.38 | 68.83 | 16.26 |
| TopK | 1025 | 77.06 | 16.78 | 77.90 | 16.76 |
| MidK | 257 | 60.86 | 16.35 | 61.10 | 16.06 |
| MidK | 1025 | 87.30 | 16.80 | 87.76 | 16.74 |
| BottomK | 257 | 68.60 | 16.49 | 68.58 | 16.09 |
| BottomK | 1025 | 77.74 | 16.78 | 78.10 | 16.74 |

**PyTorch remains faster.** These are host-API throughput observations, not
GPU-event timings, confidence intervals, or full-model speedups. Native matmul
uses the scalar sequential kernel and rank uses tile_cols=256; this is not a
claim that either framework uses its globally optimal configuration.

## Review Correction And Retained Negative Evidence

The first two raw reports from `3546ecd2` used CUDA `torch.topk` for TopK and
BottomK without checking canonical cutoff tie indices. Their successful value
checks were insufficient for a matched exact-rank comparison.
The [review finding](https://github.com/RyoSpiralArchitect/SpiralTorch/pull/2072#discussion_r3942906426)
was fixed in `49603681`, with three regression tests for canonical index
validation. Only the stable reports above are admitted for CUDA comparison.

The original bytes are retained in `rejected-unstable-cuda/`, with `audit.json`
explicitly denying matched-CUDA admission even though the raw reports say
`passed`. Their native host/device bridge observations remain independently
usable; do not mix their CUDA times into the stable tables.

## Cross-Client Validation

- Native rank suite: 10 passed locally and on Furnace, including current-source
  checks, source overwrite/drop ownership, wrong-shape and different-device
  rejection, and transactional generation-overflow failure.
- Existing native resident matmul suite: 9 passed locally.
- Real Chrome WebGPU: 9 cases and 882 assertions, no page errors; generated
  TypeScript contract passed. Browser adapter identity is `BrowserWebGpu` with
  no physical GPU name, so no browser hardware-name claim is made.
- Rebuilt private Python wheel: 76 passed, one optional skip across rank,
  resident matmul, native ownership and GPU-probe tests.
- Native backend/example and wasm st-core all-target strict clippy passed;
  pinned rustfmt passed.

Browser evidence binds generated assets and the harness by hash, not by native
source/build attestation. Rust, Python and WASM API examples and reproduction
commands are in the [resident rank guide](../../../docs/performance/resident_rank.md).

## Artifact Hashes

| Artifact | SHA-256 |
| --- | --- |
| matmul-rank-stable-49603681.json | 9af5a8ebd9b1f825b4bdd162b011e23d143093893d2d0207bc57ea8b26a8807b |
| matmul-rank-stable-repeat-49603681.json | 00799f6f160282d7de7e378677bdb8fb0c64a3efe15c217e59390d6fe402acbb |
| spiraltorch-matmul-rank-browser.json | 07d0a2e61c1535878b070037d52917f4471dedf7be54c110a98495ba5af9096d |
| rejected-unstable-cuda/audit.json | 175d7a2aa3aa67dd01fbf708035510b94c721f6c60f62aba24332c6db4ba7a2e |
| rejected-unstable-cuda/matmul-rank-3546ecd2.json | 76ef5887bef76ad55e530b093307bc82782615943e3ae467433b2fc9fd8e0b62 |
| rejected-unstable-cuda/matmul-rank-repeat-3546ecd2.json | a94407af94f9fa2ae192ab6d372feffc2cc34ebe4bc262c66285c7e3a6cc9688 |
