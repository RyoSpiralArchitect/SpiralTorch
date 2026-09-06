# Matmul/Rank Submission Composition

## Execution Evidence

Two clean-source runs of `c080e916e5080df58b7bcec889f8e0ec89ac3827`
passed 18 cases each on Furnace. WGPU Vulkan and PyTorch `2.13.0+cu132`
CUDA reported the NVIDIA GeForce RTX 5090. Source/build binding, execution-image
stability and preflight/postflight foreign-compute-PID gates passed. The process
gate is not an exclusive GPU reservation.

The shared request SHA-256 is
`f987634dda04fa6dc7ab232441c62cdb184d02771f18afaf184b015188b600e7`.
Seeds 17, 29 and 43 cross projection shapes `(2, 32, 257)` and `(3, 128, 1025)`,
with TopK/MidK/BottomK and k=7. Bounded integer operands make fp32 projection and
cutoff tie checks exact. CUDA TF32 is disabled; its stable sort checks both
values and canonical indices. Every native timing mode validates its output.

## Resident Result

Units are microseconds per chain. Each cell is the median of three seed medians,
with 12 samples after two warmups. Each interval contains 16 complete chains plus
a completion fence, divided by 16, and no host maps.

- Split: 16 calls each submitting matmul, copy and rank separately (48 submits).
- Composed: 16 calls each submitting one complete chain (16 submits).
- Batch: one call submitting 16 complete chains (one submit).

The completion fence is additional and uses the same backend implementation
for all native resident modes. All six native timing modes rotate within blocks;
frameworks are measured in separate blocks, not interleaved.

| Kind | Columns | Split 1 | Composed 1 | Split 2 | Composed 2 | CUDA 2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TopK | 257 | 73.10 | 60.87 | 72.77 | 60.90 | 16.17 |
| TopK | 1025 | 81.54 | 69.46 | 79.95 | 69.63 | 16.74 |
| MidK | 257 | 64.11 | 52.37 | 64.13 | 52.39 | 16.22 |
| MidK | 1025 | 85.41 | 70.09 | 83.65 | 70.04 | 16.74 |
| BottomK | 257 | 72.56 | 60.63 | 73.45 | 60.80 | 16.10 |
| BottomK | 1025 | 79.52 | 69.51 | 78.43 | 69.96 | 16.73 |

Composed group medians improve about 11-18% over split in these two runs.
**PyTorch remains faster.** This is host-API throughput, not GPU-event timing,
a confidence interval, or a full-model result. Native matmul uses scalar
sequential accumulation and rank tile_cols=256; this is not an exhaustive
configuration search.

## Retained Negative Result

Packing all 16 chains into one command buffer was slower than 16 composed calls
in every group in both runs:

| Kind | Columns | Batch 1 | Batch 2 |
| --- | ---: | ---: | ---: |
| TopK | 257 | 68.16 | 68.43 |
| TopK | 1025 | 77.61 | 77.76 |
| MidK | 257 | 59.41 | 59.39 |
| MidK | 1025 | 79.15 | 78.48 |
| BottomK | 257 | 68.26 | 68.72 |
| BottomK | 1025 | 78.26 | 77.78 |

Do not assume fewer submissions always means lower latency. These measurements
do not isolate the cause of that batch penalty. Repetitions remain an explicit
opt-in replay of fixed operands, not an automatic scheduling policy.

The separate-copy and composed single-call paths including final rank readback
both remain around 0.75-0.78 ms, with mixed directions across groups/runs.
There is **no demonstrated final-readback latency improvement** from submission
composition. The earlier host-round-trip removal is a separate result.

## Cross-Client Validation

- Native rank suite: 11 passed on Apple M4 and Furnace RTX 5090, including
  missing/partial inputs, shape/device mismatch, invalid repetitions, generation
  overflow, source overwrite/drop, and owned snapshots across later chains.
- Existing resident matmul suite: 9 passed locally after factoring out encoding.
- Real Chrome WebGPU: 27 cases and 4,482 assertions, no page errors. The matrix
  crosses three shapes, three rank kinds, and scalar/sequential,
  register/tiled, and register/compensated configurations.
- Generated TypeScript contract passed; rebuilt private Python wheel tests:
  85 passed, one optional skip.
- Native backend/example and wasm st-core all-target strict clippy passed;
  pinned rustfmt passed.

The browser reports `BrowserWebGpu` without a physical adapter name. Its
artifact binds generated assets and the harness by hash, not by native
source/build attestation. These fixtures establish bounded integer correctness,
not arbitrary floating-point rank stability or model quality.

The [resident rank guide](../../../docs/performance/resident_rank.md) documents
the Rust/Python/WASM API, ownership and reproduction commands. This is command
submission composition with an owned GPU copy, not shader fusion or zero-copy
aliasing. Freshness is recorded after submission, not GPU completion.

## Artifact Hashes

| Artifact | SHA-256 |
| --- | --- |
| matmul-rank-submit-c080e916.json | 0c23c5e5393948ec92edbe1059ad919b8c6c5fca27a00e0704ca33ef2a479fd1 |
| matmul-rank-submit-repeat-c080e916.json | 766472f028ddc05b4ce03cb4b0753db4b4b640f2f2c627814bf83ee7a6c7ae84 |
| spiraltorch-single-submit-browser.json | e508260e392a1d0e67fa06d51d005b667f9e1f4d40116dc7b30956d91b8e1fbd |
