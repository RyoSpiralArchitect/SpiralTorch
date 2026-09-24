# Prepared resident LayerNorm SGD updates

Native Apple M4 / Metal exploratory comparison on a shared macOS host. The
three-route source is `4836538965222cd0f1e0cf9c8b94fb2e23bb0b58` and its
retained release binary has SHA-256
`5643ee4e3fbf1cedb5b5d63b2e537343ae1f6d8cc726a4d078df5b67f23c8b60`.
`sequential` is the unchanged default (four update submissions per step);
prepared `batched` and `fused` use two submissions per step. Preparation is
outside the timed intervals. No ordinary `Tensor`, trainer, or browser default
is changed.

Three counterbalanced cycles ran `fused/batched/sequential`,
`sequential/fused/batched`, and `batched/sequential/fused`. Each report has six
shapes, three warmups and nine measured intervals per route and stage, with 32
training steps per route interval. Each table cell is the median of three run
medians in milliseconds. The preloaded routes include one terminal readback;
the affine-only route omits dx and is not work-matched to the all-VJP route.

| Shape | All-VJP sequential | Batched | Fused | Affine-only sequential | Batched | Fused |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2x3 | 27.670 | 24.240 | 22.966 | 25.916 | 20.477 | 19.722 |
| 8x257 | 43.256 | 42.260 | 42.320 | 26.046 | 22.850 | 22.828 |
| 32x256 | 51.335 | 51.040 | 50.626 | 30.775 | 29.360 | 29.420 |
| 64x768 | 192.392 | 192.677 | 194.155 | 102.008 | 102.055 | 101.559 |
| 128x1025 | 489.559 | 488.287 | 489.070 | 231.259 | 231.112 | 230.238 |
| 256x256 | 268.806 | 268.525 | 269.253 | 131.122 | 131.240 | 131.428 |

| Shape | Update-stage sequential | Batched | Fused |
| --- | ---: | ---: | ---: |
| 2x3 | 0.475 | 0.408 | 0.314 |
| 8x257 | 0.455 | 0.434 | 0.358 |
| 32x256 | 0.467 | 0.431 | 0.377 |
| 64x768 | 0.539 | 0.474 | 0.398 |
| 128x1025 | 0.633 | 0.576 | 0.498 |
| 256x256 | 0.545 | 0.497 | 0.433 |

The fused update-stage median was shorter than sequential in all 18 paired
shape/run comparisons. The 2x3 all-VJP end-to-end improvement was about 17%
and its affine-only improvement about 24%; both held in all three cycles.
The 8x257 all-VJP route improved only about 2%, while its affine-only route
improved about 12%. Larger-shape end-to-end times were essentially flat.
The stage probes include their own snapshots and are not additive parts of
the 32-step intervals. These are steady-state observations, not a cold-start
or universal WGPU speed claim.

All nine reports passed `benchmarks/layer-norm-training/compare.py`: matched
f32 input/target digests, decreasing loss, 32-step outputs, and the strict
numerical bound. Across the three modes and three runs, serialized final
outputs were identical for every shape. The nine per-mode validation JSON
files publish the validated update mode, all route/stage medians, numerical
checks, and SHA-256 hashes of the retained raw reports. The three
fixed-input PyTorch CPU/MPS controls from the earlier comparison were reused
for **numerical validation only**, not a new PyTorch speed comparison. The raw
1.4 MB reports and the binary remain local under
`~/Library/Logs/SpiralTorch/layernorm-update-dispatch-20260924/`.

## Rejected grouped candidate

A separate source-pinned experiment at
`0496d4cd95542ffd917e689ad1a0417166a87308` added a generic API to
submit independent prepared fused chains in one queue submission. Its retained
binary has SHA-256
`f2cf9fa83a2834126772b6c7edfc0d1c89c75c6fcce438bc73f4fcdecf4bdc55`.
Three balanced fused/grouped pairs used the same six shapes and strict Metal
runtime. All six reports passed numerical validation and their final outputs
were identical. Median-of-run-median all-VJP times (two submissions / one
submission) were `22.612/22.886` ms at 2x3, `42.413/42.853` at 8x257,
`50.508/50.610` at 32x256, `192.468/192.920` at 64x768,
`488.534/491.079` at 128x1025, and `268.067/267.951` at 256x256. The
isolated update stage moved only slightly, and the end-to-end improvement was
not reliable. The candidate API is **not merged**. Its source is retained on
the `spiralreality/layernorm-grouped-candidate-v1` experimental branch and its
six validated reports are published here so the negative result is auditable.

## Replay

Verify published files here with `shasum -a 256 -c SHA256SUMS`. Build the
release example from the three-route source above with the command in
`benchmarks/layer-norm-training/README.md` (use `cargo build` instead of
`cargo run`). Pin and hash the resulting binary before measuring. For each
mode, run the example separately with strict GPU routing, for example:

```sh
SPIRALTORCH_STRICT_GPU=1 SPIRALTORCH_LAYER_NORM_UPDATE_EXECUTION=fused \
  target/release/examples/layer_norm_training_residency_bench \
  > "$RAW/fused-run1.json"
python3 -I -B benchmarks/layer-norm-training/compare.py \
  --expected-update-execution fused \
  "$RAW/fused-run1.json" "$TORCH/torch-b9b23501-run1.json" \
  > "$RAW/validated-fused-run1.json"
```

Repeat in the counterbalanced order above for runs 2 and 3, passing each
run's expected update mode and matching fixed-input PyTorch control. For the
grouped experiment, build from the separate source commit, compare `fused`
against `grouped_fused` (pass that name as the expected mode), and use pair
orders `fused/grouped`, `grouped/fused`, `fused/grouped`. Do not run GPU modes
concurrently. These records do not measure browser WebGPU performance or
establish an NN training advantage.
