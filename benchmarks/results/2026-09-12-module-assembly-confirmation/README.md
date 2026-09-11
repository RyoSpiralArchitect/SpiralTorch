# Main-Worktree Confirmation Of Checked Assembly

Measured source: `91b8b4c796ff306abf6b78d9e3381414d1466ebe`;
tree `041c9e48690177011a5b670e787a4e8517fe88a0`.
Baseline products remain the original shared-stage-guard source
`542c744593dedf86e949aa82772e435acf35db04`, not the regressing combined candidate.

**Outcome: retain checked single-vector assembly, restore per-element parameter
comparison.** The flat 64-descriptor allocation test retains 69 -> 1 allocations.
Group-level burst timing is approximately baseline in this confirmation; the
previous combined candidate's adverse group medians do not remain. This is
regression avoidance with an allocation reduction, not a universal speedup or
a proof that every individual call is no slower.

See [API](../../../docs/module_resident_forward.md), [four-way isolation](../2026-09-12-module-preparation-factorial/README.md),
[summary](summary.json), [manifest](manifest.json), and the
[retained combined-candidate result](../2026-09-12-module-descriptor-assembly/README.md).

## Confirmed Source

The selected production code matches the assembly-only isolation. It was
rebuilt in the main worktree, rather than substituting the isolation binaries.
The exact-bit/foreign-write regression tests remain, with names no longer tied
to the discarded bulk implementation. The extra direct bytemuck dependency in
st-nn is removed; other crates' dependencies are not changed.

Built-in modules still append checked descriptors to one vector. The custom
Module compatibility bridge and failure-prefix restoration remain. Per-element
comparison still distinguishes signed zero and layout changes, observes foreign
DLPack writes, and does not treat mutable pointer identity as proof of equality.
Linear finite-value validation still runs. Per-call parameter scans remain
O(parameter values); no mutation/versioning shortcut is introduced.

GPU arithmetic, direct I/O, error guards, output reuse, supported models and
host Tensor semantics are unchanged. Python and WASM share the same Rust
implementation. This does not migrate generic autograd or ModuleTrainer.

## Verification

- **50 full-pipeline steps plus 12 predetermined replication steps pass.** All
  17 products per version are hash-checked before and after measurement. Source
  is committed and clean throughout the runs; there are no retries or discarded
  matrices. Combined with the preceding isolation, this work has 122 serial
  verification/build/measurement steps, not 122 unique unit tests.
- Rust: contracts 23; CPU tensor 440; WGPU-enabled tensor 488; NN 752; backend
  130; integration 6; CPU handoff 7. One existing tensor test remains ignored.
- Four allocation/compatibility tests pass in each of CPU-only and WGPU-enabled
  builds. Flat 64-descriptor assembly uses one allocation versus 69 for the
  former assembly algorithm with the same checked leaf descriptors. Nested
  assembly with caller capacity allocates zero times in each of 20 calls.
  This is test-only instrumentation, not old-binary instrumentation or a claim
  that all nested/default model calls are allocation-free.
- Python: 57 tests in the GPU-enabled build and six selected CPU-only tests;
  three admission tests and the independent rectangular Torch reference test.
- Browser: 111 original-module assertions, 536 explicit graph-client assertions,
  plus learner, training, autograd, pointwise, handoff and TypeScript checks.
  CPU-only Python/WASM builds also pass.
- Independent Torch replay: 936 original-model browser comparisons, maximum
  absolute error `1.1920928955078125e-7`; existing training/VJP replay has
  120 cases / 17,136 comparisons, maximum `5.7220458984375e-6`.
- Four paired browser matrices contain 36 shape/seed cases and 371,664 output
  comparisons, maximum absolute error `2.384185791015625e-7` against the admitted
  CPU reference. All original-Module cache counters satisfy the existing contract.

## Four Confirmation Matrices

The fixture is unchanged: seeds 17/29/43, shapes [2,3,7], [2,8,64], [4,8,128],
and 2/8/16 Scaler -> Linear -> GELU -> ReLU blocks. Three warmups and nine
retained rotated blocks per route. Each burst performs eight independent
forwards of fixed resident input and one completed terminal host read; it is
not recurrent decoding or enqueue-only timing. Compilation is separate, while
per-call assembly, parameter checks and output handling are included.

Native order alternates baseline/candidate, candidate/baseline,
baseline/candidate, candidate/baseline. Each matrix also has same-page rotated
browser baseline/candidate routes. Each row below contains **12 paired seed-run
ratios**, not twelve devices. Lower ratios are faster. These are medians of
paired ratios, not ratios of separately pooled absolute time medians.

| Shape | Python Ratio Median | Python Range | Browser Ratio Median | Browser Range |
| --- | ---: | ---: | ---: | ---: |
| [2,3,7] | 0.999 | 0.802-1.058 | 1.000 | 0.833-1.200 |
| [2,8,64] | 0.994 | 0.975-1.022 | 1.000 | 0.923-1.083 |
| [4,8,128] | 0.998 | 0.938-1.083 | 1.000 | 0.972-1.029 |

The original combined-candidate record had middle browser and large native
burst paired-ratio medians of 1.083 and 1.017. The factorial repeated the
combined version's middle browser regression and supported testing assembly
alone. This fresh confirmation is against the original baseline. It is not
a direct simultaneous final-versus-combined comparison, and does not establish
an isolated CPU/GPU mechanism for the old regression.

## Residual Limits

- Individual burst regressions remain: 13/36 native pairs and 14/36 browser
  pairs are strictly above one. Native counts by size are 6/2/5; browser counts
  are 4/4/6. All raw values remain, including sub-resolution differences.
- Small native explicit-graph controls span 0.688-1.106 and Torch controls
  0.870-1.142. Middle/large explicit-graph medians are 1.004/0.999 and Torch
  medians 0.996/1.006. Browser control medians are approximately one. No ratios
  are normalized to erase host or device drift.
- Native single-call d2h ratio medians are 0.997/1.002/1.002; browser medians
  are approximately one. The prior bulk comparison's large single-call benefit
  is not retained. Burst behavior is the reason for the selected implementation.
- Against eager Torch MPS in the candidate process, module burst ratio medians
  are 0.727/0.531/0.785. This is a bounded fixture comparison, not fastest Torch,
  torch.compile, a training-quality result or peak GPU performance.
- Native versions run in separate processes; browser pairs use separate
  WASM/device instances in one page. Exact physical browser GPU and host
  exclusivity remain UNKNOWN. Browser timing is visibly quantized.
- No CUDA/Furnace, push, merge or release. The next optimization should reduce
  redundant parameter work only with a sound ownership/mutation contract, not
  by omitting finite checks or ignoring external writes.

The archive retains **105 records**, 73,391,798 raw bytes compressed to
1,698,352 bytes. The manifest checks raw/compressed SHA-256 values and round-trip
identity. Production binaries are identified by hashes, not committed.
