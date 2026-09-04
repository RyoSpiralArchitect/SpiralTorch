# Rank Adaptation on SpiralReality Furnace

This is a correctness-gated optimization probe, not a claim that SpiralTorch is
faster than PyTorch. The retained artifact is
[`rank-comparison-cuda-workgroup-attempts.json`](rank-comparison-cuda-workgroup-attempts.json).

## Boundary

- Host: SpiralReality Furnace, NVIDIA GeForce RTX 5090, Vulkan driver 595.84.
- Oracle: PyTorch 2.13.0+cu132 on CUDA.
- Native path: Rust host buffers to strict CUDA or WGPU execution and back.
- Torch path: Python host tensors to CUDA and back.
- The wrappers are not paired, so cross-runtime values are diagnostics only and
  no speed ratio is reported.
- Fixtures cover TopK, BottomK, and MidK; 256-column random rows and
  2048-column concentrated-winner rows; seeds 17, 29, and 43.

The v2 run used 36 cases, 972 equal-count control timings, and 432 adaptive
observations. All 36 cases passed, every observation passed the value/index
oracle before reward credit, and every session closed without a pending
selection. UCB produced 81 forced-exploration decisions: one for each available
arm in every case. No candidate was quarantined in this successful run.

## What Changed

CUDA candidates are real SpiralK workgroup plans (`wg: 32`, `wg: 128`, and
`wg: 256`). The CUDA executor now consumes that workgroup for heap, exact-rescan,
and MidK launches, derives dynamic shared memory from it, and validates the plan
limit before launch. Admission is deliberately narrower than a device's raw
limit: workgroups above 256 and dynamic shared memory above the portable 48 KiB
no-opt-in envelope fail before CUDA initialization. WGPU retains direct/two-stage
candidates where both are exact.

Black Cat tracks selection attempts separately from credited observations.
Incorrect or abandoned work therefore cannot train the posterior. A
correctness-failed arm is quarantined and cannot be selected again; abandonment
closes only that attempt. Every decision witness records `selection_attempts`,
`observations`, and quarantine state. Candidate admission also deduplicates the
effective execution signature, rather than accepting distinct scripts whose
knobs reach the same kernel path.

## Measured Signals

Control medians below use 36 timings per candidate, gathered in a rotating
round-robin order without updating Black Cat. The selection counts come from the
separate adaptive stream. They are controller diagnostics, not confidence
intervals or a cross-runtime speed comparison.

| Native case | Candidate medians (ms) | Black Cat selections |
| --- | --- | --- |
| CUDA TopK, 256 cols | wg32 0.0928; wg128 0.1789; wg256 0.2934 | 30 / 3 / 3 |
| CUDA BottomK, 256 cols | wg32 0.0930; wg128 0.1789; wg256 0.2935 | 30 / 3 / 3 |
| CUDA TopK, 2048 cols | wg32 0.1308; wg128 0.1205; wg256 1.0525 | 15 / 18 / 3 |
| CUDA BottomK, 2048 cols | wg32 0.1306; wg128 0.1204; wg256 1.0524 | 15 / 18 / 3 |
| CUDA MidK, 256 cols | wg32 0.0928; wg128 0.0477; wg256 0.0478 | 3 / 14 / 19 |
| CUDA MidK, 2048 cols | wg32 3.6590; wg128 1.0184; wg256 0.6347 | 3 / 9 / 24 |
| WGPU TopK, 256 cols | direct 0.8450; two-stage 0.8603 | 17 / 19 |
| WGPU BottomK, 256 cols | direct 0.8308; two-stage 0.8561 | 18 / 18 |
| WGPU MidK, 256 cols | direct 1.5182; two-stage 0.8491 | 9 / 27 |

The sole exact WGPU arm at 2048 columns measured 0.8547 ms for TopK, 0.8561 ms
for BottomK, and 2.0190 ms for MidK. PyTorch CUDA host-to-host medians range from
0.0304 to 0.0594 ms. Most
SpiralTorch cases, especially WGPU and wide MidK, remain slower at this boundary.
The useful win here is narrower: the controller now finds materially different
native optima by shape instead of "tuning" knobs that CUDA ignored.

## Receipts

- Artifact SHA-256: `33e06b3b9d13e190b5c4b3b86f2644369ead45e5cdf67816f68b9a30a7380e9b`
- Native executable SHA-256: `541b3d2816a50eff2302adec34128b5b7db843d9662fbcf15b9c77be2ecb74ff`
- Request SHA-256: `f38179b7b3bad24ebd983bbadf6b0faaec68b5c98054a4ba3b3da43a749c7ffe`
- Source commit: `b1581bf7e032e2e517cdbe86dfaaf699c9c533d8`
- Source tree: `400c1f23c0e9cc83425460a9125f85f4a29d33ad`
- Tracked status/diff SHA-256: both the empty-input digest
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
  with `tracked_dirty: false` before and after execution.
- Contract: `spiraltorch.rank_adaptation.v1`, owned by
  `st-core::runtime::rank_adaptation`.

The retained binary was built from a detached clean worktree into a fresh
commit-specific target directory. The harness executed a private hardlink of
that binary and verified its SHA-256, device, inode, size, timestamp, and source
identity before and after the run. The native process returned zero with empty
stderr; the top-level provenance gate is `valid: true`.

## Reproduction

```bash
CARGO_TARGET_DIR=/tmp/spiraltorch-rank-bench-target \
cargo build --release --locked -p st-core \
  --features cuda,wgpu-rt,kdsl --example backend_rank_bench
python tools/bench_rank_vs_torch.py \
  --executable /tmp/spiraltorch-rank-bench-target/release/examples/backend_rank_bench \
  --output rank-comparison.json
```

The remaining optimization targets are the WGPU host boundary and CUDA wide
MidK. The fixed single-warp CUDA `k=1` path is now represented by one effective
execution signature, so workgroup-only script variants are rejected as duplicate
arms rather than advertised as tuning opportunities.
