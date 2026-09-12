# Checked Pointwise Fusion In Resident NN Graphs

## Change And Boundary

`InferencePlan::fuse_pointwise()` connects high-level NN lowering to the existing
checked fused forward/VJP primitives. The Rust transform is also exposed through
Python `fuse_pointwise()` and WASM `fusePointwise()`. It creates a new portable
plan; parameter IDs, roles, shapes and values stay fixed. Stage IDs refer to the
new plan. `Scaler -> ReLU -> Scaler` can become one pointwise stage without
dropping intermediate finite checks, deterministic gain reductions, or rollback.

The transform keeps residual-reference and resource boundaries. Dense-only
legacy plans keep their specialized path. This is **opt-in**, not a migration of
`pure::Tensor`, ordinary `Module::forward`, generic autograd, or `ModuleTrainer`.
No default selection, wheel release, push, merge, CUDA or remote GPU run is claimed.
See [the API contract](../../../docs/resident_graph_training.md#opt-in-pointwise-fusion).

Frozen GPU implementation: `f3a4b271bdf5929088eda74f3f9d7e2f42634bdc`.
Baseline: `04a014f077b2eab53e2efdfb581f7a5ca947177a`, from the prior adopted profile
verification, running the ordinary uninstrumented path.
Final benchmark page/types: `f8b98be7083000a40dfa5564c7c788247e06c4f2`.
The latter changes only client declarations/tests and browser recipe admission;
the frozen native/WASM products still identify `f3a4b271`.

## Paired Measurements

Two complete rounds, each with 9 standard and 9 wide configurations. Each
configuration uses seeds 17/29/43, eight nonzero-rate SGD updates, two warmups
and eight retained timing blocks per observation cadence. The reported ratios
are geometric means of **baseline median / fused median** across nine cases:
above 1 means faster. Entries show round B / round C, not selected best runs.

| Runtime | Matrix | Immediate Loss Reads | Deferred Loss Reads |
| --- | --- | --- | --- |
| Native Metal | Standard | 1.047 / 1.129 | 1.161 / 1.093 |
| Native Metal | Wide | 1.124 / 1.073 | 1.088 / 1.070 |
| Chrome WebGPU | Standard | 1.169 / 1.171 | 1.174 / 1.167 |
| Chrome WebGPU | Wide | 1.110 / 1.103 | 1.112 / 1.119 |

Standard shapes/depths are `[2,16,32]/2`, `[2,129,32]/4`, `[4,32,64]/8`;
wide shapes/depths are `[4,64,64]/8`, `[2,128,128]/8`, `[2,64,256]/4`.
The seeded Linear/GELU/gain/ReLU graphs shrink from 6/12/24 stages to 5/9/17
at depths 2/4/8 without changing the forward operation sequence or parameters.

There are **individual regressions**: the worst standard native immediate ratio
is 0.887; the worst standard browser deferred ratio is 0.865. Timings also vary
between rounds. These results do not justify a universal/default speed claim.
For wide native workloads, eager Torch MPS / candidate time geomeans are
0.465 to 0.514: eager Torch is still about 1.94 to 2.15 times faster.
This is not a comparison against `torch.compile` or the fastest possible Torch.

The two complete rounds retain all 3,600 raw timing intervals, 2,880 non-warmup
intervals and 28,800 nonzero-rate updates across the measured lanes. Reset,
compilation and initial/final zero-rate VJP probes are outside timing. Each step
loss is read; Rust additionally checks intermediates and commits all parameters
transactionally, unlike the eager Torch finite-fixture reference. Browser timing
is only paired Rust A/B, not a cross-framework browser speed ratio or exact
physical-adapter attestation. Native identity is Apple M4 Metal/MPS.

## Correctness And Provenance

- Fused/unfused training: six trajectories per native/browser fixture, rank
  1/2/3 and both gradient policies, with numerical max absolute difference 0.
- Sixteen fused/unfused guard cases per runtime retain forward/adjoint,
  unbroadcast and candidate overflow rejection, bit-identical parameter rollback,
  owning snapshot failures after reuse/drop, and successful workspace recovery.
- Independent eager Torch CPU/MPS replay: 48 trajectories, 10,368 comparisons,
  maximum absolute error `1.1920928955078125e-7`. Full captured benchmark states
  differ from their Torch reference by at most `1.4901161193847656e-8`.
- Contracts 22, NN 734, WGPU backend 114 and integration 6 tests passed.
  Python public clients: 32 tests, zero skips. The production WASM client checks
  the same fused forward and four-step learning for both policies, including
  analytic gain gradients. Generated and shipped TypeScript declarations passed.
- Benchmark admission 11 and profile admission 4 tests passed, as did formatting,
  strict contracts Clippy and the CPU-only NN check. This is not a whole-repo/CI
  claim. The older one-off parallel GPU VJP failure remains **UNKNOWN**;
  current owned GPU tests and measurements were serial.

`timing-a` is retained as a failed attempt: native standard measurements completed,
but the browser stopped before its first timing sample because admission compared
JSON key insertion order. Structural field comparison fixed the harness. No shader
or GPU result was changed to resolve it. That incomplete paired round is not
folded into the table; its native result, browser failure and receipt are archived.

[`summary.json`](summary.json) contains every per-round geomean, worst case and
test count. [`manifest.json`](manifest.json) binds all raw logs, fixtures,
receipts and reproduction drivers to both compressed and decompressed SHA-256
hashes. Every compressed stream was decompressed and checked before admission.
Original external logs and immutable GPU products were also retained.
The archive has 81 artifacts, 183,508,356 compressed bytes and 3,580,350,872
decompressed bytes; the largest individual compressed artifact is 46,364,356 bytes.

Reproduction uses the existing training harness with `--graph --fuse-pointwise`
and `--matrix standard` / `wide`; browser arguments end with
`graph standard fuse-pointwise` / `graph wide fuse-pointwise`. The validator
checks original and executed plans as ordered operations with explicit operand
identities, as well as all captured gradients, parameters and losses. Product
hashes and source/harness revisions are in the archived receipts.
