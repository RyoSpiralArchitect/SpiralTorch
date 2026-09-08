# Resident Pointwise: Full Four-Lane Record

Measured Rust/worker source: `78af0d8c7b36761611f851338339febe7dd98e38`,
clean tree `9b4a10703624f672edc924155f610d72bb9b6e98`.
No edits, compilation, or other owned GPU jobs overlapped the timed run.
Frozen binary/module hashes and executed commands are in `frozen-run.json`.
Uncontrolled macOS/third-party GPU load remains unknown.

## Native M4 Result

Existing immutable N-D storage, a narrowed/permuted input, and twenty
broadcast-add / scalar-multiply / tanh-GELU iterations (60 operations).
One prepared Rust chain runs through sequential, batched and fused scheduling.
The fourth lane is eager PyTorch 2.12.1 MPS, with fallback disabled and the same
single Apple M4 admitted for native Metal and MPS.

Two warmups and eight retained samples per lane, rotating/reversed ordering.
Each interval includes views, execution, output allocation and final contiguous
host readback. Initial upload, plan/pipeline construction and JSON transport are
outside timing. Rust checks all intermediates; Torch does not have equivalent
finite guards.

| Root Shape | Seed | Sequential ms | Batched ms | Fused ms | Torch ms | Torch/Fused |
|---|---:|---:|---:|---:|---:|---:|
| 8 x 16 x 64 | 17 | 7.013375 | 5.312062 | 0.403458 | 1.160188 | 2.8756 |
| 8 x 16 x 64 | 29 | 6.395896 | 5.576729 | 0.364020 | 1.042229 | 2.8631 |
| 8 x 16 x 64 | 43 | 5.435374 | 5.370938 | 0.433000 | 0.992979 | 2.2933 |
| 16 x 32 x 128 | 17 | 6.480625 | 5.673958 | 0.529354 | 1.626979 | 3.0735 |
| 16 x 32 x 128 | 29 | 6.751479 | 6.624521 | 0.555646 | 1.233480 | 2.2199 |
| 16 x 32 x 128 | 43 | 6.788521 | 6.051501 | 0.571375 | 1.217458 | 2.1308 |

Fusion was 11.88..17.57x faster than this run's sequential Rust lane and
2.13..3.07x faster than eager MPS. Batching alone remained slower than MPS.
The [earlier negative baseline](../2026-09-09-resident-nd-tensor/README.md) is
preserved; cross-run timing differences are not attributed to a particular
source change. These medians are a bounded diagnostic, not a universal speedup,
best available PyTorch implementation, browser/CUDA timing, `torch.compile`
comparison, or training-throughput result.

`pointwise-bench.json.gz` is the full **35,038,752-byte** raw record compressed
to 315,212 bytes, not a selected summary. It retains all six recipes, ten
intervals per recipe with all four lanes, and full input/output captures.
The repeated transform is contractive: the very small final error
(3.3306691e-16) alone is weak correctness evidence.

## Numerical And Ownership Checks

- Native and actual browser WebGPU fixtures each retain the original three
  cases plus nine pointwise cases (three modes x zero/one/twenty iterations).
- Each feeds the existing Linear/GELU graph and eight resident SGD steps.
  PyTorch CPU/MPS independently replay all 48 platform/mode/recipe combinations;
  maximum absolute error across outputs, loss, VJP and updated weights:
  **1.4901161193847656e-8**.
- Native/browser guards reject masked Add/Multiply/GELU overflow, failed RHS
  and empty-output lineage, mixed host and foreign queues. They check output
  reuse, same actual queue through another wrapper, and signed-zero Identity.
- CPU tensor tests: 437; backend tests: 108 with runtime opt-in; kernel
  contracts: 15. NN resident fixture/training tests: 3. Python admission,
  complete-record reaggregation and worker supervision: 11.
- Strict backend/tensor checks pass on native and wasm32. The first ordinary
  NN-example check found one new derivable-Default warning; it was fixed before
  the frozen source. The final frozen check retains only 22 pre-existing NN
  library warnings (unchanged `crates/st-nn/src`) and vendored WGPU warnings.

The browser is Chrome 152.0.7977.83. Rust reports `BrowserWebGpu`; a separate
browser adapter observation reports Apple/metal-3/nonfallback. That separate
probe is not exact attestation of the Rust runtime adapter. Page errors and
console messages were empty.

General N-D autograd, automatic arbitrary Module graph lowering, public Python
or JS pointwise classes, and fusion across dense matmul boundaries are not
implemented here. The new Rust result uses the existing NN bridges, which
still copy on GPU to/from mutable NN workspace storage.

## Reproduction

```bash
shasum -a 256 -c benchmarks/results/2026-09-09-resident-pointwise/SHA256SUMS
python -I tools/validate_nd_tensor_bench.py \
  benchmarks/results/2026-09-09-resident-pointwise/pointwise-bench.json.gz
python -I tools/validate_nd_tensor_vs_torch.py \
  --reports benchmarks/results/2026-09-09-resident-pointwise/native-nd.json \
            benchmarks/results/2026-09-09-resident-pointwise/browser-nd.json \
  --devices cpu mps --source 78af0d8c7b36761611f851338339febe7dd98e38 \
  --output /absolute/new-pointwise-replay.json
```

The second command reaggregates recorded intervals, not GPU execution.
The third executes independent Torch models against captured Rust values.
The publication-time replay labels each mode explicitly; this is a reporting
addition after the measured commit, not a new Rust/GPU timing run.
To rebuild and rerun both GPU fixtures, use the recorded commands and frozen
source in `frozen-run.json`. Binaries remain local, not in this repository;
recorded hashes do not attest binary bytes absent from this checkout.
