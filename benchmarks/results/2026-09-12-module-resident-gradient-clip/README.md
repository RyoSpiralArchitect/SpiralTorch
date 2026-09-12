# Resident Global Gradient Clipping

Optional global L2 clipping now sits between weighted gradient composition /
gain normalization and the existing transactional SGD commit. Rust owns the
norm contract; Python and WASM expose configuration, not optimizer arithmetic.
The disabled resident update shaders are byte-identical to the feature base.

## Source And Checks

- Runtime commit: `32414050677e97a3e365f7d0f5e971eedb4e8fa5`.
- Runtime tree: `00babb974c34afed756ecdb9f8928c99001ca66e`.
- All 69 source-bound verification stages passed in 640.32 seconds. Native
  backend 144, WGPU-enabled NN 760, and integration 6 tests passed; additional
  CPU, tensor, Python, generated/shipped TypeScript and forward controls passed.
- Two published Python code blocks ran unchanged, with a fresh accumulation
  window prepared between them: 99 microbatches, 33 updates, then explicit
  weight handoff with maximum output difference 0.
- Independent integrity verification checked 216 compressed source/log/data
  records and 39 frozen candidate/control products. Binaries remain outside
  Git; hashes and source identities are in [summary.json](summary.json).

## Numerical Results

The shared native/browser fixture uses Scaler, Linear, GELU, ReLU and CE, seven
parameter tensors, three seeds and both gradient policies. Each case runs 95
microbatches over 32 updates before observing GPU values. Clip limits cycle
through `0.05`, disabled, `0.1`, `2.0`, `0.001`; this tests enable/disable and
normalization order, not a recommended training schedule.

PyTorch 2.12.1 CPU/MPS independently replayed the six clipped cases on each
route: **24 cases, 34,104 tensor/scalar comparisons**, maximum absolute error
`1.430511474609375e-6`. Full replay including existing cases passed 408 cases /
159,504 comparisons. Four existing tiny-tail reference gaps remain separately
labelled, not counted as matches. MPS fallback was disabled.

All six clipped cases lowered their training-fixture loss, but **every clipped
case finished worse than its unclipped control at the same 32 updates**:

| Seed / Policy | Initial | Clipped Final | Unclipped Final |
| --- | ---: | ---: | ---: |
| 17 / Exact | 1.107177 | 0.929803 | 0.693433 |
| 17 / ModuleCompatible | 1.107177 | 0.940835 | 0.778921 |
| 29 / Exact | 1.121065 | 1.072396 | 1.042030 |
| 29 / ModuleCompatible | 1.121065 | 1.072544 | 1.044956 |
| 43 / Exact | 1.132504 | 1.077451 | 1.044732 |
| 43 / ModuleCompatible | 1.132504 | 1.077181 | 1.045429 |

These native values agree with the browser fixture. This demonstrates bounded
updates, not a quality advantage: the deliberately restrictive schedule slows
progress on these small synthetic training examples. There is no held-out,
LLM/FT or generalization claim.

## Boundaries And Failures

Public Python and WASM each check eight cases (four clipped), with 96
microbatches / 32 updates, rejected update 33, and recovery 34. They verify raw
snapshots, source guards, stale generations, invalid configuration preservation
and explicit handoff. Browser `reference_gain` is the scalar oracle, not a GPU
parameter dump; the shared fixture records actual GPU parameters and losses.

The initial native implementation failed on 513 finite gradients of magnitude
`1e38`: the wide-norm probe accepted an unclipped update. The failed logs remain
under `raw/runs/backend-a.log.xz` and `backend-b.log.xz`. Exponent-aware norm
merging and normal-f32 scale factors fixed the case. Final native/browser wide
probes pass at norm limits `1` and `1e-20`, with per-coordinate updates about
`0.04415108` and `4.415108e-22`. These use an analytic reference, not a Torch
wide-f32 norm match. The ordinary trainer also retains nonzero finite gradients
when the total norm exceeds f32 range.

Norms at most `f32::EPSILON` and near-one scales retain the shared historical
no-op rule. Genuine subnormal outputs remain subject to backend f32 handling.
Clipping cannot sanitize an invalid source, sum or VJP, even at zero weight/rate.
Native adapter: Apple M4 / Metal. Browser: BrowserWebGpu, physical GPU UNKNOWN.
Owned GPU work was serial; host exclusivity UNKNOWN. No CUDA, sustained
performance, generic ModuleTrainer migration, release, push or merge is implied.

See the [usage guide](../../../docs/module_resident_gradient_clip.md),
[full results](summary.json), [archive manifest](manifest.json), and
[independent integrity check](verification.json). The evidence/documentation
commit adds no runtime changes beyond the verified source above.
