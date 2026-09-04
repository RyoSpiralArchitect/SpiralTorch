# WASM-First Backend Pilot, 2026-09-04

Furnace: RTX 5090, NVIDIA driver 595.84, native WGPU Vulkan adapter verified by
the strict rank runner. PyTorch 2.13.0+cu132, CUDA capability 12.0, system CUDA
toolkit/NVRTC 13.1. The process-specific Vulkan ICD was pinned to NVIDIA.
The GPU was idle before the pilot; an existing CPU llama-server was left alone.
The local WASM CPU comparison ran separately on macOS; do not compare its
absolute times to the Furnace timings.

## Results

`wgpu-vs-cuda-final.json`: 36/36 numerical gates passed, three seeds, 20 warmed
samples per implementation, five warmups. Median of seed medians, milliseconds:

| Operation / shape | ST before | ST after | PyTorch CUDA host-to-host |
| --- | ---: | ---: | ---: |
| gather 64x64 | 2.052 | 0.922 | 0.048 |
| gather 256x256 | 2.088 | 0.998 | 0.090 |
| gather 512x512 | 2.318 | 1.265 | 0.214 |
| scatter 64x64 | 2.052 | 0.955 | 0.049 |
| scatter 256x256 | 2.157 | 1.054 | 0.089 |
| scatter 512x512 | 2.475 | 1.369 | 0.214 |
| matmul 64x64 | invalid precision | 0.932 | 0.056 |
| matmul 256x256 | invalid precision | 1.018 | 0.110 |
| matmul 512x512 | invalid precision | 1.335 | 0.298 |

The shared bounded-poll change reduced gather/scatter latency about 1.8-2.2x in
this pilot. ST remains slower than PyTorch CUDA at this boundary. This is not a
cross-machine claim, a confidence interval, or a kernel-only speedup. The first
exploratory rerun (`wgpu-vs-cuda-fixed2.json`, 30 samples) is also retained;
`final` used the same sample count as baseline without a concurrent build.

- `wgpu-vs-cuda-baseline.json`: all nine matmul cases failed precision checks
  because large float32 RHS values were silently quantized. No valid before/after
  matmul latency claim can be made from those cases.
- `cuda-probe-before.json`: strict CUDA failed NVRTC compilation on missing
  `cuda_runtime.h`; non-strict execution silently succeeded through fallback.
- `cuda-probe-fixed2.json`: the same request executed strictly on CUDA.
- `rank-baseline.json`: 18/18 WGPU cases passed; CUDA passed 12/18, losing
  concentrated TopK/BottomK winners in six cases.
- `rank-fixed.json`: 36/36 WGPU/CUDA cases passed, including concentrated inputs.
  Black Cat observations follow executed, verified variants. The small trial
  does not establish bandit convergence or controller speedup.
- `faer-vs-torch-cpu.json`: 36/36 separate CPU controls passed with the same fixture generator.
  This original report only records the requested Faer backend: its matmul
  uses Faer, but gather/scatter use ST's CPU indexing implementation, not Faer.
  It is retained unchanged as the pre-review provenance fixture.
- `faer-vs-torch-cpu-review.json`: 36/36 rerun gates passed, with requested and
  effective backends recorded in every case: matmul=faer, gather/scatter=cpu.
  This reruns the corrected harness against the same private native wheel;
  it is not a new wheel build or evidence for a backend speed change.
- `macos-wasm-vs-torch.json`: actual WASM CPU, 27/27 cases passed. At size 128,
  WASM matmul/gather/scatter medians were 0.474/0.022/0.042 ms, versus PyTorch
  CPU 0.004/0.002/0.017 ms. Browser WebGPU throughput was not measured.

## Provenance And Limits

The native wheels are private development candidates using the existing 0.4.27
package metadata, not uploads or replacements of public PyPI artifacts.
Native paths/hashes and benchmark-script hashes are in the tensor reports. The
rank reports retain executable and complete-request hashes; source scripts
reconstruct every request. The WASM binary SHA-256 is in its report; it is the
earlier typed-indexing validation build, before the cross-realm ingress fix.
The CPU math tested here is unchanged by that ingress-only fix.

Furnace checkout started at b89190f696f332c89ea063113eacd093857c55ca and received
the exact edited files through `spiral push`. Thus a report's Git HEAD alone is
not the source identity of its development wheel. Native/executable hashes and
the change-specific build logs are required to identify artifacts.

Furnace durable artifacts and logs live at
`~/.local/state/spiraltorch/benchmarks/wasm-first-v1/`.
Local/remote source hashes were checked. A clock skew caused the first attempted
rebuild (`build-fixed1.log`) to reuse the old library; that wheel was not used.
Only transferred changed sources were timestamp-refreshed, then `build-fixed2`
actually compiled them. No cache deletion or system-clock change was used.
Timing uses monotonic clocks, not those file timestamps.

Golden coverage is a functional parity check, not a PyTorch speed comparison:
two workers x three epochs x 32 batches match sequential weights exactly.
The first Golden test run exposed an existing rank-hint panic in barycenter
synchronization; the fixed suite passed all eleven tests. CUDA runtime's six
tests passed with actual device execution, including heap-boundary ties.

The initial isolated Tensor test build required system Fontconfig solely for
the old benchmark PNG renderer. Histogram output is now SVG, and the Plotters
development dependency is headless. Cargo regenerated the lockfile offline;
unused font dependencies were removed, with no dependency-version upgrades or
system package installation.

## Review Follow-Up

The quantization cache now uses the original RHS precision decision even when
63x65 and 64x64 map to the same bucket. Synthetic autotune samples use the same
decision; cache revision 3 prevents reuse of the earlier ambiguous entries.

Golden now accepts worker-ID-keyed epoch input. With dropout allowed and quorum
maintained, surviving workers can continue without replacement models or
positional reassignment. Regression tests drop ID 0, reorder IDs 1/2, and compare
all weights with sequential training using distinct per-worker learning rates.

Furnace follow-up validation passed: 13 Golden tests, 44 matmul-filtered unit
tests, and 8 row-indexing/precision integration tests with real WGPU enabled,
strict GPU execution, and the NVIDIA Vulkan ICD pinned. The SVG random-init
benchmark also executed successfully (`cargo bench -- --test`). The local
standard-library benchmark harness suite passed all 5 tests. Logs are retained
under the same durable audit directory with the `review-` prefix.

Next priorities are asynchronous browser WebGPU resident workloads, upload and
readback reuse with explicit lifetimes, and larger model-shaped matmul/rank
fixtures. Preserve numerical gates before optimizing the control heuristics.
