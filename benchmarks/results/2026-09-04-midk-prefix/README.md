# MidK / BottomK Prefix Repair

The expanded real-GPU backend run exposed a pre-existing compaction failure on
Furnace's RTX 5090 (NVIDIA Vulkan ICD, driver 595.84). An isolated checkout of
main `3ae0749ad2baa0085eeb51c3797c78d642265310`, using unpatched crates.io WGPU,
reproduced output counts `[0, 0]` instead of `[5, 3]`. The exact baseline log is
retained as `main-baseline-negative.log`. This is not a resident-matmul regression.

## Repair

The Blelloch downsweep needs one active lane at stride 128, then twice as many
lanes as stride halves. Both portable downsweeps instead started with 128 active
lanes and halved their count, accessing outside their 256-element scratch array
and leaving incorrect prefixes. The repair restores that tree traversal.

All lanes now snapshot the row block's total before the root is cleared. A
second barrier finishes scratch reads before the next 256-tile block reuses it.
No CPU substitute, approximate result, changed ranking policy, or global slow
path was introduced. The optional subgroup apply implementations are unchanged.

## Validation

On the same device, the fixed backend passed all 51 unit tests and the complete
WGSL syntax test. The original multi-tile case now passes. A new real-GPU test
checks 18 fixtures: 1, 255, 256, 257, 300, and 65,793 columns, with sparse, dense /
empty-row, and last-column-only masks. It compares exact counts and stable output
order, and verifies untouched tails and row-stride padding. The largest case
requires 258 tiles and exercises repeated row-prefix scratch blocks.

Run:

```bash
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 \
  cargo test --locked --release -p st-backend-wgpu midk_bottomk
```

The existing macOS WGPU job now runs this family with real-GPU tests enabled.
This receipt is NVIDIA/Vulkan correctness evidence, not a throughput win,
subgroup-path qualification, or a browser execution result for this family.
Full local build/test logs are in
`~/.local/state/spiraltorch/benchmarks/resident-matmul-v1/midk-*.log`.
