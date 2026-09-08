# Resident N-D Tensor: Correctness And First Performance Baseline

Measured source: `6d9392be7fb355c277e8322d039aa2f1b481790c` (clean tree
`82b1b14a8f5354911bcdf824dd794f9204cca2e6`). Native release and browser WASM
products were built and copied before execution. Subsequent publication changes
do not change the measured Rust core or fixtures. A later review fix adds bounded
response/exit deadlines to the benchmark controller; the retained timings remain
bound to the original harness at this source, not relabeled as a new measurement.

## What Passed

- Native WGPU and Chrome 152 WebGPU each execute the same three Rust fixtures:
  `[2,3,4]/seed17/0 iterations`, `[3,5,7]/seed29/1`, and
  `[2,4,11]/seed43/20`. The zero-iteration case enters NN as a strided view.
- Each fixture covers N-D preprocessing, existing `Sequential(Linear, Gelu,
  Linear)` inference, N-D postprocessing, and eight resident mean-MSE/SGD steps.
  The 20-iteration preprocessing case chains 60 GPU elementwise operations.
- Four bad-input/bad-target training attempts per runtime retain all weights
  bitwise. Masked overflow, empty-result guard propagation, foreign devices,
  shape/generation errors, host-batch recovery and snapshot lifetime are checked.
- Independent PyTorch 2.12.1 CPU/MPS replay checks preprocessing, inference,
  final loss, input VJP, parameter gradients and post-update weights: 12 checks,
  maximum absolute error `1.4901161193847656e-8`. Replay was repeated against
  the published fixtures with the same outcome.
- Local checks: kernel contracts 14, host tensor 436, WGPU backend 108 and
  selected NN tests 741 passed, including explicitly enabled GPU tests.
  Native/WASM backend and tensor strict Clippy, strict tensor rustdoc and both
  production binding compilation checks passed. Ten benchmark admission,
  reaggregation and worker-supervision tests also pass.

The two attempted strict **whole st-nn** Clippy checks remain red on 22 existing
library warnings in unchanged source. Ordinary native/browser example Clippy
passes after fixing the new fixture's two redundant allocations. The initial
check report is retained as failed; it is not relabeled as an all-green strict run.

Native timing admitted one Apple M4, matching WGPU Metal and Torch MPS, with
MPS fallback disabled. Browser execution reports `BrowserWebGpu`; its separate
adapter probe reports Apple/metal-3 and non-fallback, but is not an attestation
of the exact Rust-selected adapter. Browser timing is **not** measured here.

Review found the original benchmark controller could wait indefinitely for a
live worker that never completed its response. The follow-up controller uses
120-second response and 10-second exit/cleanup deadlines, retains partial error
reports, and never restarts a worker automatically. Tests cover silence, partial
lines, oversized responses, EOF, an exit hang and ignored termination. This does
not change the Rust timed workload or retroactively alter the archived samples.

## Performance: Still Slower Than Eager MPS

Both sides start with resident inputs, run 20 repetitions of broadcast add,
scalar multiply and tanh-GELU, and include views and the final contiguous host
read. Upload, pipeline creation and JSON are excluded. Two warmups and all eight
retained intervals per recipe are present; execution order alternates. Rust
validates intermediate finiteness and the terminal read, whereas eager Torch
does not have the same intermediate guard cost. macOS foreign GPU load is unknown.

| Root Shape | Seed | Rust Median ms | Torch MPS Median ms | Rust / Torch |
| --- | ---: | ---: | ---: | ---: |
| `[8,16,64]` | 17 | 5.902250 | 1.260292 | 4.683 |
| `[8,16,64]` | 29 | 4.750166 | 0.992292 | 4.787 |
| `[8,16,64]` | 43 | 4.860146 | 0.916500 | 5.303 |
| `[16,32,128]` | 17 | 5.660688 | 1.519167 | 3.726 |
| `[16,32,128]` | 29 | 6.046646 | 1.770833 | 3.415 |
| `[16,32,128]` | 43 | 6.718021 | 1.974646 | 3.402 |

These negative results are the baseline, not a reason to claim a residency
speedup. The new path removes intermediate data readback, but still allocates
outputs/metadata/bindings and submits each elementwise operation separately.
Batching submissions and reusing/fusing compatible work are the next candidates;
their causal contribution has **not** been profiled or measured yet.

The repeated transform is contractive, so tiny final benchmark errors
(`3.3306690738754696e-16`) are not strong standalone numerical evidence.
The zero-/one-iteration fixtures and full NN VJP/SGD replay supply that coverage.
This is not arbitrary N-D autograd, model-training quality evidence, a CUDA
result, fastest-PyTorch comparison, or automatic acceleration of ordinary forward.

## Revalidate The Published Data

The full benchmark record, including all inputs, captured outputs and intervals,
is preserved in `mps-bench.json.gz` (not only a timing summary). `manifest.json`
binds compressed and decoded bytes. Native/browser fixture arrays are plain JSON.
Binary/module bytes and full development/build logs remain local; their hashes
and source identities are recorded, not substituted for independent execution.

From the repository root:

```bash
python3 -I tests/test_nd_tensor_bench.py
python3 -I tools/validate_nd_tensor_bench.py benchmarks/results/2026-09-09-resident-nd-tensor/mps-bench.json.gz
python tools/validate_nd_tensor_vs_torch.py --reports benchmarks/results/2026-09-09-resident-nd-tensor/native-nd.json benchmarks/results/2026-09-09-resident-nd-tensor/browser-nd.json --devices cpu mps --source 6d9392be7fb355c277e8322d039aa2f1b481790c --output /tmp/nd-replay-new.json
```

The first revalidator recomputes all medians and capture agreement, not GPU
execution. The second executes Torch and compares the published Rust outputs.
Use a fresh output path and choose `--devices cpu` on non-MPS hosts. See
[the Rust API and execution guide](../../../docs/resident_nd_tensor.md) to build
and run the actual native and browser products.
