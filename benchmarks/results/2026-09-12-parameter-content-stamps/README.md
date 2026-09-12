# Revocable Parameter Content Stamps

Runtime source: `04fec016ad1ee8871d2a786974e94caeb556b84b`.
Baseline: `91b8b4c796ff306abf6b78d9e3381414d1466ebe`, the independently
verified assembly-only runtime, not the earlier regressing bulk-comparison patch.
Both are frozen local products. No push, release, CUDA or Furnace run is included.

## What Changed

`st-tensor::TensorContentStamp` uses weak storage ownership plus shape/layout.
It does not retain tensor values or force a value-sized COW copy. Rust mutation
dissociates the weak owner; successful shared writable DLPack export revokes
existing stamps through all aliases. Failed/copy-only exports do not revoke the
source. Foreign storage, including read-only imports, never takes this shortcut.
TensorBuffer COW cloning isolates writable values so independent tracking cannot
miss an exported alias. External writes must occur between Rust calls, not race
with Rust reads.

Parameter finite validation reuses only a matching positive result. The resident
Module cache uses matching stamps, otherwise retains the previous exact-bit
comparison against its frozen values. An equal-value replacement can refresh its
stamp without recompiling. New/changed invalid weights still fail. The witnesses
are not hashes, portable revisions, or a finite-value guarantee on their own.

This also fixes a CPU correctness bug: a writable export made after prepacking
could leave normal matmul using stale packed weights. The red regression records
the old packed final value `14` versus the live value `99`; both pack orientations
now use the same revocable contract, and untracked sources are repacked.

The GPU shaders, arithmetic, graph I/O, guards, output slots and queue schedule
are unchanged. Python and WASM inherit this Rust behavior, not separate caches
that reconstruct its semantics. Generic Tensor/autograd/ModuleTrainer execution
is not silently migrated to resident GPU execution.

## Verification

All 50 full verification steps and 12 predeclared replication steps passed.
The CPU-only and WGPU products were built from the same frozen source.

- Tensor CPU: 449 tests. Tensor WGPU-enabled: 497 passed, one existing ignored
  fractional-backend test retained; not every test in that binary is a GPU test.
- NN CPU: 706 unit tests plus the stale-pack regression. NN WGPU-enabled: 758
  unit tests plus the regression. WGPU backend: 130; contracts: 23; resident
  integration: six. GPU work was serial with the real-runtime opt-in enabled.
- Descriptor allocation checks: four each on CPU/WGPU builds. The existing flat
  control remains 69 allocations versus one; supplied nested capacity needs zero
  descriptor allocations. This is not global allocation-free execution.
- Python: 58 tests, including late export after CPU and GPU caches already exist,
  plus six selected CPU-only tests and the CPU capability/handoff probes.
- Browser original Module: 111 assertions. Explicit graph forward: 536 assertions.
  Pointwise, learner, autograd, fusion, parameter handoff, type declarations and
  CPU-only WASM compilation also passed.
- Torch CPU/MPS training and VJP replay: 120 cases, 17,136 comparisons, maximum
  absolute error `5.7220458984375e-6`. Original browser Module replay: 936
  comparisons, maximum `1.1920928955078125e-7`.

Focused tests verify zero exact comparisons across 20 unchanged native cache
checks, one comparison for an equal-value replacement, and a comparison on every
reuse after a later mutable export. Weak stamps do not keep values alive or copy
single-owner mutations. NaN, last-word changes, signed zero, metadata, retained
outputs, device changes and checked training handoff remain covered.

## Matched Timing

Each matrix uses seeds 17/29/43, three shapes, and Scaler/Linear/GELU/ReLU blocks.
There are three warmups and nine retained rotated timing blocks per case.
Burst is eight independent forwards of the same resident input followed by one
completed terminal host read, not recurrent decoding. D2H reads every forward.
Input upload and cold compilation are excluded from these two routes.

All four matrices are retained: initial baseline/candidate, then candidate/baseline,
baseline/candidate, candidate/baseline native process order. Browser A/B loads both
WASM packages in one page and rotates module and explicit-graph routes. Native
Apple M4/Metal is recorded in the fixtures; physical browser GPU and exclusive
host access remain UNKNOWN. No trials were retried or removed.

Median of 12 paired seed-run ratios per shape; lower is faster:

| Shape; blocks | Python burst | Python D2H | Browser burst | Browser D2H |
| --- | ---: | ---: | ---: | ---: |
| `[2,3,7]`; 2 | 1.012 | 0.995 | 0.937 | 1.000 |
| `[2,8,64]`; 8 | 1.008 | 0.956 | 1.000 | 1.000 |
| `[4,8,128]`; 16 | 0.973 | 0.851 | 1.000 | 0.900 |

The largest native D2H case improved in all 12 pairs (ratios 0.835-0.867);
its burst ratios were 0.965-0.980. This supports a bounded benefit, particularly
for completed individual calls, not a universal throughput improvement.
Middle native D2H improved in all pairs (0.942-0.972), while its burst median
regressed slightly. Browser burst medians are largely unchanged and quantized;
even the small browser's baseline/candidate pooled medians are both 0.075 ms
despite its lower median *paired ratio*. Those statistics are not interchangeable.

The record retains 17/36 strict native burst regressions and 6/36 browser burst
regressions, plus 5/36 native and 8/36 browser D2H regressions. Small native burst
ratios span 0.970-1.511; unchanged explicit-graph controls span 0.975-1.622 and
Torch controls 0.880-1.437 there. Small browser burst spans 0.833-1.400 and D2H
0.667-1.500. These are not discarded or normalized away as noise.

Candidate native burst/eager Torch MPS medians are 0.730/0.538/0.771 for these
three fixtures. This is not fastest-PyTorch, compiled/fused Torch, peak-GPU,
generic training superiority, or a causal GPU-phase timing claim.

## Evidence

[summary.json](summary.json) includes every matrix, ratio range, regression,
verification count and source/product identity. [manifest.json](manifest.json)
binds 111 compressed records to original bytes: 73,617,003 raw bytes,
1,723,956 compressed. Every compressed member was round-trip checked.
The frozen runtime products are retained locally under
`Library/Logs/SpiralTorch/parameter-content-stamps-20260912/verified-a`; their
hashes are in the archived receipt. Binaries are not added to Git.

`raw/` includes the exact verification/replication/archive scripts, commands,
receipts, correctness outputs and timing samples. Development import/API-name
mistakes and the expected stale-pack failure are preserved and labeled separately
from adopted-source validation; they are not missing benchmark trials.
