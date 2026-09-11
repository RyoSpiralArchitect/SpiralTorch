# Original Module Resident Forwarding

Adopted code: `c5349790398d406d1a5a39566cf5d3d22db20181`;
tree `d94114158a89609975886d2197aa570749d91549`.
The complete `verified-c` run passed **44 sequential steps**, binding source,
16 frozen products, commands and outputs. Earlier attempts are not substituted
for the adopted run.

## What Is Connected

The original Rust `Module` now owns an exact-match, bounded resident graph cache.
Python `model(WgpuTensor)` and browser `Sequential.forward(WgpuTensor)` use that
same core. Supported Linear/GELU/Scaler/ReLU compositions preserve leading N-D
axes, avoid per-layer CPU readback, and follow parameter updates. Returned GPU
captures survive subsequent calls, cache clearing and model destruction.
Browser training can hand weights back to its original model through the same
checked Rust policy.

See [API and boundaries](../../../docs/module_resident_forward.md),
[summary](summary.json), and [archive manifest](manifest.json).

## Verification

- Rust: contracts 23; host tensor 440; WGPU tensor 488; NN 750; backend 117;
  integration 6; CPU handoff 7. Counts are separate feature configurations,
  not unique tests. One existing fractional-GL live-adapter test remains ignored.
- Python: 53 tests in the GPU-enabled build, including host/surface tests;
  6 selected tests in a separately built CPU-only extension. No skips in these
  selected suites. The rectangular Torch reference also has its own explicit
  Apple-MPS/CPU regression check.
- Browser: original-model forwarding, cache reuse and training handoff pass
  49 assertions across widths 3/7/16. Existing graph, pointwise, learner,
  autograd, handoff and generated/shipped type checks also pass.
- Independent Torch CPU/MPS replay: 936 new browser output comparisons, maximum
  absolute difference `1.1920928955078125e-7`. Existing training/VJP replay:
  120 cases, 17,136 comparisons, maximum `5.7220458984375e-6`.
- Native GPU: Apple M4, WGPU Metal, Torch 2.12.1 MPS with fallback disabled.
  Chrome reports BrowserWebGpu; its exact physical adapter is **UNKNOWN**.

## Matched Timings

Each block is Scaler -> Linear -> GELU -> ReLU. There are three parameter seeds,
three warmup rounds, and nine retained rotated rounds per condition. Each
resident burst performs eight independent forwards of one fixed resident input
and includes the final owning host read. This is not twenty recurrent forwards,
and not just asynchronous submission latency.

The table shows medians across the three per-seed medians, in ms/forward.
Ratio ranges use each seed's paired medians; below 1 means less elapsed time.

| Shape | Blocks | Original Module | Torch MPS eager | Module / Torch range |
| --- | ---: | ---: | ---: | ---: |
| [2,3,7] | 2 | 0.1809 | 0.1235 | 1.341-1.465 |
| [2,8,64] | 8 | 0.2121 | 0.3331 | 0.624-0.653 |
| [4,8,128] | 16 | 0.5362 | 0.6698 | 0.795-0.823 |

All nine module cases compile once and record 108 cache hits / 109 submitted
forwards including setup and warmup. Host parameter-bit comparison and GPU
input/output capture copies are included; cold setup is recorded separately.

The small original-model route is **slower** than Torch and is 1.70-2.07 times
the precompiled scalar-graph burst time. The middle/large ranges against that
graph are 1.12-1.19 and 1.00-1.12. The connection is useful, not free; these
measurements do not isolate CPU comparison from GPU-copy/dispatch overhead.
Host-to-host and device-to-host single-call timings are retained separately,
not treated as identical boundaries.

These are bounded eager controls, not peak throughput, fastest PyTorch,
`torch.compile`, training-quality or universal speedup evidence. Host GPU
contention is UNKNOWN. Owned GPU jobs were serial.

## Negatives And Limits

- The first browser feedback fixture used unscaled initial weights and produced
  non-finite output at the second width. This is retained, including the exact
  page matching its recorded hash. The adopted recurrent fixture uses fixed
  bounded diagonal weights, not a relaxed validity threshold.
- `verified-a` was deliberately interrupted to fix a guard that only inspected
  the upper NN policy. Direct tensor bindings and nested policies now reject
  correctly before resident submission; the interrupted run is not a pass.
- `verified-b` completed the timing matrix but failed independent browser replay:
  the old Torch helper incorrectly restored input width after rectangular
  matmul. Only the reference was corrected; kernel math and tolerances were not
  changed. Earlier timings and the traceback are retained.
- No implicit CPU fallback, ordinary host-Tensor storage migration, generic
  autograd/ModuleTrainer GPU conversion, zero-copy, CUDA/Furnace run, wheel
  release, push or merge is claimed here. Foreign producers must synchronize
  writes with forwards; bitwise cache matching is not a concurrency protocol.

The manifest contains 147 compressed records: 66,235,962 uncompressed bytes,
1,424,248 compressed bytes. Hashes cover both forms. Full native binaries remain
in the local frozen-product directory; their hashes are in the run receipt.
The compressed `run.py` and pre-fix harness retain exact commands and local
runtime paths. The public benchmark entry is
`tools/bench_graph_forward_paths.py --include-module`; independent aggregation
and browser replay use `tools/validate_module_forward_paths.py`.
