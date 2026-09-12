# Fixed-Storage Resident Microbatch Learning

Verified source: `ecfe51001bca4606a2ab547b65833d5c52965bd1`.
The resident learner now accumulates exact parameter VJPs from changing input
batches at one unchanged parameter state, then performs one transactional SGD
update. Rust, Python and WASM share this implementation. See the
[usage guide](../../../docs/module_resident_microbatch.md).

The accumulator reuses one buffer per parameter plus a whole-window guard.
It does not retain every source VJP. Diagnostic snapshots explicitly allocate
owning outputs; retaining those still consumes memory. The existing checked
composition, source-guard and parameter-update shaders are byte-identical to
the feature base. Normal same-forward SGD retains its original contract.
Parameter revisions use a learner identity and checked counter, without a new
revision-object allocation on each update.

## Verified Learning

All 69 source-bound stages passed in 614.10 seconds, covering Rust CPU/GPU,
Python GPU/CPU-only, browser WebGPU, generated/shipped TypeScript and independent
PyTorch CPU/MPS replay with MPS fallback disabled. The unchanged Python guide
separately ran 96 microbatches and 32 updates on the frozen GPU library; returning
weights to the original Module produced zero maximum output difference.

Each native/browser learning case uses seven deterministic source batches,
fixed N-D shape `[2,3,4]`, with 6, 4, 2, 5, 3, 1 and 6 valid labels. The cases
replay 95 microbatches through 32 windows of 2/3/4 contributions. Sample weights
account for ignored labels; no GPU observation occurs before all 32 updates are
submitted. Every retained prediction, loss, input/parameter VJP, accumulated
parameter gradient and updated parameter is checked against ordinary Rust
Modules. All seven parameter tensors are explicitly returned to the Module.

The displayed native/browser initial and final losses agree. Loss is evaluated
over the same seven source batches, weighted by valid-label count:

| Seed | Reduction | Update Policy | Initial CE | Final CE |
| --- | --- | --- | --- | --- |
| 17 | Mean | Exact | 1.10717690 | 0.69343251 |
| 17 | Mean | ModuleCompatible | 1.10717690 | 0.77892089 |
| 29 | Sum | Exact | 1.12106454 | 1.04203045 |
| 29 | Sum | ModuleCompatible | 1.12106454 | 1.04495609 |
| 43 | Mean | Exact | 1.13250434 | 1.04473209 |
| 43 | Mean | ModuleCompatible | 1.13250434 | 1.04542899 |

All twelve route/case trajectories reduced loss. These are small synthetic
training fixtures, not held-out accuracy, generalization or LLM fine-tuning.
Reduction varies with seed for coverage; the table is not a controlled ablation
of reduction or update-policy quality. ModuleCompatible preserves the compiled
microbatch gain row average, not an extra virtual-batch average.

Independent PyTorch 2.12.1 replay adds 24 cases and 34,104 tensor/scalar
comparisons, with maximum absolute error `9.5367431640625e-7`. Including all
existing fixtures: 384 cases and 125,400 comparisons, maximum error
`1.811981201171875e-5`. Tolerance is `atol=2e-5`, `rtol=2e-4`; counts are not
individual element counts. Four pre-existing tiny-tail reference differences
remain separate from matched comparisons.

Python and WASM public clients each run four mean/sum and Exact/ModuleCompatible
cases: 96 microbatches, 32 updates, invalid-label rejection at update 33 and
recovery at 34. Each update receipt is checked. Retained accumulator snapshots
survive later resets and dropped producers. The browser client's `gain` field
contains its independent scalar reference, not a raw GPU parameter dump; its
GPU output is checked against that reference. The shared native/browser fixture
contains the observed GPU losses, gradients and parameters.

Backend tests additionally cover 300 contributions, finite negative weights,
empty windows, source/parameter-generation mismatch, foreign learners, counter
overflow, ordinary-SGD invalidation, zero-weight invalid sources, multiplication
overflow followed by cancellation, atomic rejection and reset/recovery.

## Evidence And Limits

[summary.json](summary.json) records all trajectories, test counts, client
results and 39 checked frozen products, including 22 candidate products.
[manifest.json](manifest.json) binds 210 compressed raw/source records,
114,735,880 uncompressed bytes, to original and packed hashes. Binaries stay in
their recorded local directories rather than Git.
[verification.json](verification.json) records the separate decompression,
original-file/Git-blob comparison and product-hash verification.
Preliminary helper/import compilation failures remain archived and labeled;
they are not counted as frozen-source verification.

Native: Apple M4/Metal. Browser: WebGPU, physical GPU UNKNOWN. Host exclusivity
UNKNOWN; owned GPU work was serial. Existing forward controls were retained,
but this run is not a sustained-interval speed comparison. The existing ignored
`wgpu_frac` live-adapter test remains ignored; 141 backend tests, including real
resident-GPU tests, ran successfully. No implicit normalization, variable-shape
recompilation, generic ModuleTrainer migration, hypergrad/band substitution,
optimizer-state transfer, clipping, momentum, CUDA/Furnace, release, push or
merge is included.
