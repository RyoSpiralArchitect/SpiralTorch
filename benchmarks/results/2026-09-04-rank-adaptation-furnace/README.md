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
| CUDA TopK, 256 cols | wg32 0.0931; wg128 0.1789; wg256 0.2919 | 30 / 3 / 3 |
| CUDA BottomK, 256 cols | wg32 0.0928; wg128 0.1790; wg256 0.2922 | 30 / 3 / 3 |
| CUDA TopK, 2048 cols | wg32 0.1308; wg128 0.1207; wg256 1.0524 | 15 / 18 / 3 |
| CUDA BottomK, 2048 cols | wg32 0.1308; wg128 0.1207; wg256 1.0527 | 15 / 18 / 3 |
| CUDA MidK, 256 cols | wg32 0.0927; wg128 0.0477; wg256 0.0479 | 3 / 9 / 24 |
| CUDA MidK, 2048 cols | wg32 3.6575; wg128 1.0157; wg256 0.6345 | 3 / 9 / 24 |
| WGPU TopK, 256 cols | direct 0.8376; two-stage 0.8560 | 17 / 19 |
| WGPU BottomK, 256 cols | direct 0.8360; two-stage 0.8495 | 18 / 18 |
| WGPU MidK, 256 cols | direct 1.5182; two-stage 0.8529 | 9 / 27 |

The sole exact WGPU arm at 2048 columns measured 0.8533 ms for TopK, 0.8836 ms
for BottomK, and 2.0132 ms for MidK. PyTorch CUDA host-to-host medians range from
0.0303 to 0.0590 ms. Most
SpiralTorch cases, especially WGPU and wide MidK, remain slower at this boundary.
The useful win here is narrower: the controller now finds materially different
native optima by shape instead of "tuning" knobs that CUDA ignored.

## Receipts

- Artifact SHA-256: `932e8a47d97b43c1d670af98bfcbf60e81be30ca13f976a0994464c1e2cf71e1`
- Native executable SHA-256: `02592b704490a494e610410ad49c5ed8cc2f1b06fb5b7c168d838b0bacb0eb54`
- Request SHA-256: `f38179b7b3bad24ebd983bbadf6b0faaec68b5c98054a4ba3b3da43a749c7ffe`
- Source commit: `307733f8af125668bbd7ae23bb9ed90a48492133`
- Source tree: `b816e51f4ce0c2b89642d05e7af6cf7ef81a5f30`
- Tracked status/diff SHA-256: both the empty-input digest
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
  with `tracked_dirty: false` before and after execution.
- Contract: `spiraltorch.rank_adaptation.v1`, owned by
  `st-core::runtime::rank_adaptation`.

The retained binary was built from a detached clean worktree into a fresh
commit-specific target directory. The harness executed a private hardlink of
that binary and verified its SHA-256, device, inode, size, timestamp, and source
identity before and after the run. The refreshed receipt also requires the
binary's embedded build commit/tree and clean-build bit to match that checkout.
The native process returned zero with empty stderr; the top-level provenance
gate is `valid: true`.

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
