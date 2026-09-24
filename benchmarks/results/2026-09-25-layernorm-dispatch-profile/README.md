# Resident graph LayerNorm backward dispatch profile

Diagnostic Apple M4 / Metal measurements on macOS 26.4.1 with rustc 1.97.0.
The code under test is commit `a68070e184c8edb4a5a3d5b78c6102d87ffebd18`;
the release example binary's SHA-256 is
`36dd22a082f48d486e4671b0d0677a9279bfe5932f6650ad83911a77d009405d`.
The normal `step()` and `step_profiled()` schedules remain unchanged. The new
`step_profiled_layer_norm_split()` is diagnostic only: it executes the same
LayerNorm input and affine dispatches in two passes instead of one. Both modes
use a private timestamp-capable device, and every profile read validates GPU
error scopes and the attempted training step.

Five original-first and five split-first paired runs followed two warmup
steps per shape. Each pair used the same resident graph and a zero learning
rate for the two measured steps; the warmups used a nonzero learning rate.
The example checked loss, prediction, input gradient, raw/effective parameter
gradients, and parameter values bit-for-bit between the paired steps. All 30
shape/run pairs passed that check; all 60 reports were accepted with complete
timing. Shapes were 2x3, 32x256, and 128x1025. Every run's results, including
all phase totals, losses, GPU spans, and validation fields, are in
[`results.json`](results.json). The ten full raw reports are retained locally
and not committed. The published JSON's SHA-256 is
`ae80e0920145946f9d9bdc134dc87930236244668a9ed97c0179f1dbb1f54ab0`.

| 128x1025 LayerNorm backward | Minimum | Median | Maximum |
| --- | ---: | ---: | ---: |
| Original combined pass (ms) | 8.910 | 8.948 | 9.068 |
| Diagnostic input pass (ms) | 6.982 | 7.062 | 7.110 |
| Diagnostic affine pass (ms) | 1.884 | 1.898 | 1.920 |

The input VJP dominates the split diagnostic at this shape, making it the
next kernel to investigate. These figures are **not** a speedup or a PyTorch
comparison. The extra pass boundary changes scheduling, so summing split
durations does not reconstruct normal training time. GPU timestamps exclude
CPU encoding, uploads, query readback, and untimed copies; the GPU span may
include inter-pass gaps. Small-shape timings are especially sensitive to
fixed costs and are published as observations rather than a performance claim.

## Raw report hashes

The names below refer to files under the local
`~/Library/Logs/SpiralTorch/layernorm-dispatch-profile-v1/pinned-a68070e1/`
directory. They correspond to `id` in `results.json`.

| Raw report | SHA-256 |
| --- | --- |
| paired-1.json | `e20bb92d8efa095ed660330d27e650d01b753e9aedf8efd92fb82c639dc313d7` |
| paired-2.json | `ab661b274521f19687db391f39aaa397e97056de251ebea3e52e2efcd6957dd8` |
| paired-3.json | `69e351220b65618564b3bb2d9b9b800a96987415936d8d802ab39f87921131a0` |
| paired-4.json | `c3614c9610c1018fc77b62adc553e0b5404680d1e72043c79eb85d29758af3aa` |
| paired-5.json | `7400186b756334b8158c163938af3657b7b047b47e4a1f9962f2a5a8a25610ec` |
| reverse-1.json | `3be242be0a6c63b958f6aeb6b6e3a0d1b772c8f6436b3a0dd60bbc8ade177f3c` |
| reverse-2.json | `0ebe4b1e5678d7e047fb878eebe4ac06ffcffca58d307fb6aa650243557e93d2` |
| reverse-3.json | `ac0bacdd8d88b845dbbb5c77a76c4b8a316586a2d2e1295a1095e081efff1296` |
| reverse-4.json | `295f20b54c416aa5f38248c0137940d4c7b74f2e7dd45d5162f79874957135a4` |
| reverse-5.json | `518025058736608470161efbb0a26690cb2d114f17161cf0e8fa7fa7035ba8be` |

## Replay

From a checkout at the code commit above, build the example in release mode:

```sh
cargo build --release -p st-backend-wgpu --example resident_graph_layer_norm_profile
```

Run `target/release/examples/resident_graph_layer_norm_profile --paired`
five times, then `--paired-reverse` five times, writing each stdout JSON to
a separate file. Runs should be sequential, not concurrent with other GPU
measurements. The no-argument form remains the original profile example and
was smoke-tested separately. To check the split profile's state and pass
labels on a real GPU:

```sh
SPIRALTORCH_RUN_WGPU_TIMESTAMP_TESTS=1 cargo test -p st-backend-wgpu split_layer_norm_profile_preserves_state_and_labels
```

The source also passed `cargo check --workspace`, `cargo test -p st-nn
--features wgpu`, the WGPU backend runtime suite, strict backend Clippy, and
the `spiraltorch-wasm` WebGPU release build. This is a measurement tool and
a prioritization result, not evidence of an end-to-end training optimization.
