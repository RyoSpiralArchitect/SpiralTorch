# Resident Topos EMA Momentum

Topos' gradient-history primitive now runs inside the resident graph learner,
after weighted gradient composition, gain normalization and optional global
clipping. Rust owns its transition and validation; Python and WASM expose the
same settings and owning GPU history snapshots. No tensor or history readback
is needed inside the update loop.

This is a zero-initialized EMA, **not PyTorch heavy-ball/Nesterov momentum**:

`m_t = damping * m_(t-1) + (1 - damping) * effective_gradient`

## Source And Verification

- Runtime commit: `5229ba90f4f049068b11bfe702fa339ab7b81843`.
- Runtime tree: `769b83bb2e5b43153a216fb9177407db6ae4e2cf`.
- All 69 source-bound verification stages passed in 656.43 seconds. Shared
  contracts 28, native backend 147, WGPU-enabled NN 760 plus one doc test, and
  integration 6 tests passed. CPU, Python, generated/shipped TypeScript,
  browser WebGPU and existing forward controls also passed. The tensor-WGPU
  suite retained one ignored fractional-GL adapter test; it is not a pass.
- The published Python example ran unchanged against the frozen native
  binding: 32 updates / 32 batches, then weight-only Module handoff with
  maximum output difference 0. Both history snapshots remained readable and
  nonzero after reset, disable and learner destruction.
- Independent archive verification checked 213 compressed source/log/data
  records and 39 frozen candidate/control products. Binaries remain outside
  Git. See [verification.json](verification.json) and [manifest.json](manifest.json).

## Numerical And Learning Results

The shared native/browser fixture uses Scaler, Linear, GELU, ReLU and CE, seven
parameter tensors, three seeds and both gradient policies. Each case runs 95
microbatches across 32 updates before observing actual GPU weights, losses and
history. The ordinary Rust reference calls `ToposOptimizerStateControl`.

PyTorch 2.12.1 independently differentiates the same graphs and replays the EMA
with tensor arithmetic, not `torch.optim.SGD`. CPU and MPS replay of both routes
passed **24 momentum cases / 38,472 tensor or scalar comparisons**, maximum
absolute error `9.5367431640625e-7`. Full replay passed 432 cases / 197,976
comparisons, maximum absolute error `1.811981201171875e-5`. Four existing
tiny-tail reference gaps remain separately labelled, not counted as matches.
MPS fallback was disabled.

Clip limits cycle through `0.05`, disabled, `0.1`, `2.0`, `0.001`; damping cycles
through `0.6`, `0.85`, `0`, disabled, `0.3`, with explicit reset and zero-rate
probes. These exercise lifecycle transitions, not a recommended schedule.
Every case lowered its training-fixture loss, but **every EMA case finished
worse than its matched clip-only control at the same 32 updates**:

| Seed / Policy | Initial | EMA Final | Clip-Only Final |
| --- | ---: | ---: | ---: |
| 17 / Exact | 1.107177 | 0.997179 | 0.929803 |
| 17 / ModuleCompatible | 1.107177 | 1.000533 | 0.940835 |
| 29 / Exact | 1.121065 | 1.088526 | 1.072396 |
| 29 / ModuleCompatible | 1.121065 | 1.088772 | 1.072544 |
| 43 / Exact | 1.132504 | 1.096821 | 1.077451 |
| 43 / ModuleCompatible | 1.132504 | 1.096949 | 1.077181 |

The table shows native values; browser results differ by at most
`1.1920928955078125e-7` for these final losses. This demonstrates executable
stateful updates and numerical agreement, not a quality advantage. There is no
held-out, LLM/FT, generalization or speed claim.

## State Safety And Boundaries

Native probes verify that invalid gradients or overflowing candidates reject
all weights and history together, including invalid zero-weight sources.
Zero-rate attempts validate but preserve both states. An unused plain-SGD
candidate cannot reject an otherwise finite EMA update. Changing enabled
damping preserves history; disabling and re-enabling starts from zero. Invalid
configuration preserves the existing settings and state.

Public Python and WASM each exercise 12 cases, including four with clipping and
EMA, plus rejected/recovery probes and history lifetime checks. Browser client
`reference_gain` / `reference_momentum` fields describe the scalar reference
after the 32 learning updates, before subsequent probes; they are not GPU dumps.
The shared fixture separately records actual GPU history and parameters.

Momentum is opt-in. Its two history-sized buffers per parameter are allocated
once and reused. Disabled updates add no EMA dispatch. The original gradient
composition, source-guard and parameter-update shaders are byte-identical to
the feature base; the clip shader deliberately shares a gradient-only helper
with the new path. Forward controls are not sustained performance evidence.

The clip limit bounds the new effective gradient, not retained history. The
full Topos RMS-control pipeline, generic ModuleTrainer migration and optimizer
checkpoint/resume are not implemented here. History snapshots are observations;
Module handoff remains weight-only. Native adapter: Apple M4 / Metal. Browser:
BrowserWebGpu, physical GPU UNKNOWN. Owned GPU work was serial; host exclusivity
UNKNOWN. No CUDA/Furnace, release, push or merge is implied.

See the [usage guide](../../../docs/module_resident_momentum.md) and
[full results](summary.json). Pre-freeze native checks are preserved separately
under `raw/runs/`, not counted as frozen verification. The evidence/documentation
commit adds no runtime changes beyond the verified source above.
