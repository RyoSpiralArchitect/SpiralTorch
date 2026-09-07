# Existing NN To Resident WGPU

Compiled implementation: `37e726f6a7bc0acf2c3983243faa7dc7cdb67b00`,
tree `d49ecac0723b2dc5f3f211bf4e53ae72e2a0ba51`.
Subsequent commits add evidence, not changes to the measured Rust implementation.
Native and browser artifacts embed this clean source identity. Native timing
uses mode-555 copied images with matching before/after hashes, not a shared
Cargo target executable. Browser compiler output and served WASM have distinct
recorded hashes; this is not a reproducible-build claim.

## Implementation

The existing Rust `Linear`, `Gelu`, and nested `Sequential` lower through
`InferencePlan` into the common `ResidentDense` backend. Weights/bias and
activation buffers persist on the device. Producer output is consumer input,
without intermediate readbacks or device-to-device copies. A chain submission
contains one pass per dense stage. A separate snapshot submission copies the
final output plus every stage's finite-value flags.

GELU fusion preserves square/cubic/inner/output finite checks on the GPU;
snapshot reads reject earlier-stage failures even when later operations mask
them. The host also checks final output finiteness. Invalid uploads do not
change generations or output validity; snapshots survive reuse and workspace
drop. Parameters are immutable compilation snapshots, not silently refreshed
copies of a training model.

`NdLayout` provides checked shape/element-stride/offset views and storage
indexing. The first executor accepts contiguous, nonempty, zero-offset input
and applies Linear along the last axis. Existing `pure::Tensor`, autograd,
optimizers, ordinary `Module::forward`, and Python public APIs are not migrated
to general N-D/resident training here. Unsupported modules/layouts are rejected.
See [the Rust API and browser fixture](../../../docs/resident_nn_inference.md).

## Measurements

Apple M4 / native WGPU Metal versus PyTorch **2.12.1 MPS**. Three complete matrix
runs, three seeds per shape, twelve retained samples after two warmups:
**27 cells and 324 samples per implementation**. The two repeats were added
after observing initial variability; the first run and every slower sample
remain included. No performance thresholds or kernel knobs were selected from
these results.

The entries below are medians of nine **per-cell sample-mean ratios**, not
ratios of pooled timing medians or confidence intervals.

| Logical Input | Linear Stages | Legacy / Resident (Larger Is Better) | Resident / MPS (Smaller Is Better) |
| --- | ---: | ---: | ---: |
| `[2,3,7]` | 2 | 1.651 | 0.979 |
| `[2,8,64]` | 16 | 4.843 | 1.263 |
| `[4,8,128]` | 16 | 5.401 | 1.456 |

All 18 deep-chain cells are faster than legacy NN, with ratios spanning
3.524-11.352. The small case is not a reliable win: one resident cell is
2.386 times slower than legacy. Resident beats MPS in only 11/27 cells;
the deep-chain medians still favor MPS. Maximum absolute native/reference
error is `4.470348358154297e-8`; PyTorch output parity also passes at
`atol=1e-5, rtol=1e-4` for every sample.

These are host-input-to-owned-host-output inference timings, with new input
upload and final readback inside each sample and setup outside. Legacy model
prepacking/caching behavior is unchanged. Resident and PyTorch weights/bias
stay on device. Native arms alternate; PyTorch runs in a later process block.
TF32, MPS fast math, and MPS CPU fallback are disabled. Rust maintains its
finite guards, while the eager PyTorch baseline has no added per-stage guards
for these frozen finite fixtures. Parameter checks moved to plan compilation,
activation fusion, buffer reuse, launch count and transfer boundaries change
together; this does not isolate the causal contribution of any one change.

MPS device admission verifies exactly one Apple GPU and a matching WGPU Metal
identity, but contention is **unknown** on an active macOS desktop. Owned GPU
jobs and builds do not overlap timing. This is not evidence of a universal
speedup, fastest-PyTorch performance, GPU-event latency, or training throughput.
Furnace CUDA comparison is **not run**: an existing foreign GPU training process
was active at the admission check. It was not interrupted.

## Correctness And Replay

- Rust NN CPU regression: **686 passed**; kernel contracts: **12 passed**.
- Native NN GPU tests: **3 passed**, including 32 source operations, N-D/vector
  inputs, width-changing `4 -> 7 -> 3`, all six kernel/accumulation combinations,
  invalid uploads, snapshot ownership, and overflowing GELU intermediates.
- Shared matmul live regression: **9 passed**. Checked dense shader/decode:
  **2 passed**, including Naga validation of six shared shader variants.
- Browser Chrome `152.0.7977.77`: **6 real Rust Sequential cases** and
  **4 intermediate overflow cases** pass on `BrowserWebGpu`, versus an explicit
  WASM CPU oracle. Its adapter name is anonymous, not an asserted M4 identity.
  No browser timings are reported. No page errors or generated-module failures
  were observed; one non-module HTTP 404 console message is retained in the raw
  report rather than relabeled as an entirely silent browser run.
- Python fixture/admission validator: **4 passed**. Native strict backend /
  kernel-contract Clippy and targeted rustfmt pass. Full strict NN Clippy is
  still red on pre-existing NN diagnostics; ordinary diagnostic collection
  succeeds with no warnings in the new resident modules/examples. Unchanged
  warning sites and early import/borrow/WASM dependency failures are retained.

The WASM dependency repair keeps native Faer enabled but excludes its native
threadpool from WASM via target-specific dependencies in `st-nn`, `st-text`,
and `st-logic`. The actual browser fixture builds the high-level Rust NN crate,
not a JS reimplementation or a report-only facade.

[summary.json](summary.json) retains all per-cell timings and ratios.
[raw-logs.tar.xz](raw-logs.tar.xz) includes complete reports, inputs/parameters,
build/test/failure logs, product hashes, and the offline validator. Executables
and generated browser modules are excluded. The summary hashes each raw file.

After extracting the archive, replay without Torch, CUDA, or a browser:

```bash
python -I analyze_nn.py --repo /path/to/SpiralTorch --output recomputed.json
```

The source/fixture/sample validator checks all 27 cells and six browser cases.
Independent extraction and replay reproduced `summary.json` byte-for-byte.
See [SHA256SUMS](SHA256SUMS) for the archive and summary digests.
