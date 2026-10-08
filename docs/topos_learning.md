# Captured Topos Learning

Topos retains the finite Picard map, not an implicit fixed-point derivative:
`state[0] = 0; state[n+1] = saturate(input * gate + coupling * state[n])`.
`ToposResonatorOperator::capture` stores the audited output and the exact
finite-unroll drive sensitivity in Rust. Repeated VJPs do not rerun the
recurrence or consult mutable caller inputs. Forward audit calculations,
shape/finite/budget checks and derivative multiplication order are unchanged.

```python
from array import array
from spiraltorch import ToposResonatorKernel

kernel = ToposResonatorKernel(coupling=0.25, iterations=4, porosity=0.2)
batch = kernel.capture_buffer(array('f', [0.2, -0.3]), array('f', [0.5, 0.8]), 1, 2)
output = memoryview(batch.output_buffer()).cast('f')
dx, dg = batch.vjp_buffer(array('f', [1.0, 1.0]))
```

Sequence clients can use `capture`, `output`, and `vjp`; legacy stateless
`forward`/`backward` remain available, with corresponding bulk-buffer methods.
Bulk inputs must expose a C-contiguous native-endian float32 buffer. Outputs
are independent bytearrays. Inputs are copied into an owned immutable tape;
this is not zero-copy. A tape retains four float32 vectors of element count N
(input, per-element gate, sensitivity, output), about **16 N bytes** plus
metadata, excluding temporary transport buffers and the client's saved tensors.

`topos_resonator_autograd` and `ToposResonatorAdapter` use the same core tape.
Feature gates `[F]` (or `[1, ..., 1, F]`) now use shared-row capture: the client
transports F gate values, not N, and Rust returns N input gradients and F gate
gradients. Other broadcast shapes retain the elementwise route and Torch's
reduction. Only the upstream gradient is uploaded during backward.
Optional NumPy enables bulk transport; the sequence fallback has identical
semantics. Saved Torch tensors still enforce in-place version checks. These
are first derivatives only, and the execution backend remains **Rust f32 CPU**
with explicit host transfers for accelerator tensors, not GPU residency.
Inference (`no_grad`, `inference_mode`, or neither input requiring gradients)
uses the stateless forward and does not allocate the learning tape.

WASM exposes `kernel.capture(Float32Array, Float32Array, rows, features)`,
`batch.output`, `batch.audit_json()`, and `batch.vjp(upstream)`. Pullbacks have
typed `grad_input` and `grad_gate` arrays. Copies outlive their Rust handles;
call `.free()` on batches and pullbacks when finished. A batch can outlive its
kernel. Direct Rust consumers use the same `ToposResonatorLearningBatch` core.

### Unexpanded Shared-Row Transport

Rust offers `forward_shared_rows`, `capture_shared_rows`, and
`capture_shared_rows_owned`. Python exposes the first two, also with `_buffer`
suffixes; WASM exposes `forwardSharedRows` and `captureSharedRows`. Forward
does not allocate a tape. A shared tape stores input, output and sensitivity
of length N plus a gate of length F: **12 N + 4 F bytes** of f32 storage,
excluding allocation capacity, metadata, temporary and client buffers.
`gate_layout` / `gateLayout` and `gate_values` / `gateValues` describe the tape
and its gate-VJP shape. Existing elementwise methods are unchanged.

The same finite recurrence, derivative multiplication order and forward audit
are used. Row contributions to the shared gate VJP are individually checked in
f32, accumulated in row order in f64, then rounded once to f32 and checked again.
This is a **sum, not an average**, matching CPU `Tensor::try_sum_axis0`. It can
retain cancellation when an intermediate f32 sum would overflow. It is not
promised bit-identical to Torch's f32 reduction, and historical optimizer
trajectories need not remain bit-identical after migration. Repeating a saved
state with the same implementation still must reproduce the next update.

Both N and F must fit the kernel's value budget, even when rows is zero; an
empty input returns F zero gate gradients. Shared `vjp_audited` reports RMS of
the reduced F gradients. The external per-element audit API is unchanged.
`vjp_audited_elementwise` returns the same reduced gradient while streaming
the RMS of its pre-reduction contributions, for compatibility with NN audits.
This is CPU/scalar-WASM transport optimization, **not GPU-resident HF training**
or evidence of better language-model quality. The native `st-nn` CPU shared
layer below also uses this compact tape and core-owned row sum.

Reproduce the scalar-WASM contract in Node or Chrome and compare equivalent
finite-unroll CPU routes (including the old expanded captured control):

```sh
node tools/probe_topos_shared_transport.mjs /tmp/shared-node /tmp/shared-node-new.json
node tools/test_resident_browser.cjs /tmp/shared-web "$CHROME_EXECUTABLE" /tmp/shared-browser-new.json "" "" "" "" topos-shared-transport
python tools/benchmark_topos_learning.py --shape 2 128 768 --iterations 5 --include-expanded-capture --rounds 15 --native-profile release --output /tmp/shared-benchmark-new.json
```

The benchmark keeps bitwise checks for Rust output and input VJPs; shared gate
VJPs are checked against an independent wide sum of the legacy elementwise
VJPs. It also records numerical differences from the legacy Torch-f32 sum.
Timings cover forward plus both VJPs and host transport, not optimizer updates.

## Rust NN Learning

The CPU route of `st_nn::ToposResonator` now captures that same core tape during
forward. Backward calls `batch.vjp_audited_elementwise(upstream)`, which checks finite
gradients and the sensitivity bound and computes the same backward audit from
the saved transition. It does not run the finite recurrence twice again.
This is an audit of Rust-owned results, not independent verification of an
external executor; WGPU results still use the full formula-comparison audit.
Direct CPU and legacy Auto routes rely on the core capture's complete input,
gate and finite-drive checks instead of repeating the same scan in the NN
wrapper. Accelerator requests retain early admission before route metadata,
availability checks and dispatch, including requests that select CPU via a
size threshold. Invalid forward calls do not replace an existing valid tape
or gradient. This does not expose an unchecked core entry point.

The NN cache shares its immutable tape across repeated band pullbacks, checks
input and gate consistency bit-for-bit (including signed zero), and invalidates
it on mutable parameter access or topos/config changes. CPU/WGPU routing can
change between forward and backward;
only an actual CPU captured pullback reports `capture_reused: true`.

Capture trades extra forward work and retained storage for cheaper backward.
The elementwise core tape owns four N-element vectors; a shared-row tape owns
three N-element vectors and one F-element gate. The returned NN output is a
separate Tensor; the older NN cache shared the caller's Tensor storage. This
is not an inference-speed or peak-memory improvement. Stateless
`ToposResonatorOperator::forward` and Python's no-grad path remain available
without capture. Tests compare saturated and unsaturated gradients/audits and
100 synthetic SGD updates against recomputation, not language-model quality.

### Shared Gates For Variable Batches

`ToposResonator::with_shared_gate` owns a `(1, features)` trainable gate instead
of one parameter per input element. Each forward accepts a different row count;
the supplied topos must admit the full `rows * features` volume. Existing
constructors keep their elementwise gate and exact-shape behavior.

```rust
use st_nn::{Module, OpenCartesianTopos, Tensor, ToposResonator, ToposResonatorConfig};

let features = 8;
let max_rows = 32;
let topos = OpenCartesianTopos::new(-1.0, 1e-6, 1.0, 16, max_rows * features)?;
let mut layer = ToposResonator::with_shared_gate(
    "topos.gate", features, ToposResonatorConfig::new(0.2, 5)?, topos,
)?;
let input = Tensor::from_fn(3, features, |r, c| (r + c) as f32 * 0.1)?;
let output = layer.forward(&input)?;
let upstream = Tensor::from_fn(3, features, |_, _| 1.0)?;
let grad_input = layer.backward(&input, &upstream)?;
assert_eq!(grad_input.shape(), input.shape());
assert_eq!(layer.parameter().gradient().unwrap().shape(), (1, features));
```

The CPU layer retains an unexpanded F-element gate and sums each checked f32
VJP contribution in row-order f64 inside the core pullback. It does not allocate
an N-element gate gradient or invoke a second tensor reduction. There is **no
additional row average**: a mean-loss factor belongs in the upstream gradient.
The row sum and final finite f32 checks match CPU `sum_axis0`. The core streams
pre-reduction statistics in that same pass, so the NN backward audit continues
to describe expanded elementwise VJPs, not the reduced parameter gradient.

The legacy host WGPU route expands the gate only when dispatching to that
executor and still uses its WGPU `sum_axis0` reduction and execution receipt.
A GPU forward followed by CPU backward retains the existing recomputation and
CPU tensor reduction. A CPU capture can serve GPU backward and later CPU
backward without replacing its compact tape. Nonfinite sums are rejected
before adding a gate gradient or committing a backward audit. Empty inputs
still validate the feature gate before dispatch.

Input, gate and upstream tensors are normalized by logical row/column layout,
including the existing elementwise mode. Cache checks compare canonical input
bits and an isolated canonical parameter snapshot, not the differently shaped
expanded shared gate. Synchronized writes through a writable DLPack alias are
detected before backward; WGPU cached operands are isolated as well. On
successful accumulation, parameter values and Tensor-backed gradients are
normalized together to row-major optimizer storage. Tape gradients already use
that logical coordinate order. Metadata distinguishes `gate_rows` / `trainable_parameters` from
the logical `expanded_gate_rows` / `expanded_gate_values`. Actual executor gate
and gate-VJP sizes are `materialized_gate_values` and
`materialized_gate_gradient_values`; `gate_gradient_reduction_kernel` separates
`st_core.f64_row_sum` from `tensor_util.sum_axis0`. No tensor-reduction receipt
is fabricated for the fused core sum. `ToposGateLayout` re-exports the core
layout enum under the existing NN API name.

This is a host-Tensor Rust NN layer, usable in ordinary `Sequential`; it does
not make that graph GPU-resident. Host `Sequential` retains the inputs from its
actual forward and reuses them for backward, including repeated band pullbacks.
It no longer reruns the layers to reconstruct activations: Dropout masks,
recurrent state and the Topos tape remain those used by the original prediction.
This removes container-driven replay, not any internal recomputation a child
module chooses to perform; it is not a claim that every graph is replay-free.
The latest successful forward is authoritative; a new or failed forward,
structural/mutable-parameter access, state load, text infusion or mode change
invalidates it. Input/parameter guards reject mismatches before child pullbacks.
Shape/input validation failures permit a valid retry; a child backward failure
invalidates the tape, but does not roll back already accumulated gradients.
Clear accumulators before restarting such a failed step.

`zero_accumulators()` clears gradients without discarding the captured forward,
including nested sequences and Topos. `Module::scale_learning_rates` likewise
keeps the capture while prevalidating all optimizer rates before mutation.
Trainer coherence-driven LR adjustments use this path at their original point
before backward; moving them after backward is not required. Arbitrary mutable
parameter visits still invalidate the container capture. Containers and band schedules call
`Module::backward_retained`; the built-in LSTM, SpiralRnn, WaveRnn, WaveScan,
coherence scan and coherence-wave layers support this without consuming their
forward caches. Their ordinary one-shot `backward` remains available. Custom
modules with consuming backward caches must implement `backward_retained` too;
its default only delegates to `backward`. Direct retained-pullback callers must
keep the input, parameters and module state unchanged. Sequential checks that
boundary for its owned children.

`eval()` changes layer behavior, not gradient recording. Explicit
`Sequential::forward_untracked` / `forward_untracked_owned` (Python:
`model.forward_untracked(x)`) bypass container activation retention, propagate
through nested sequences and invalidate any old host tape. Individual layers
may still keep their own internal caches. The original owned-forward path is
used there; this is not an implicit CPU fallback for resident inputs.

Saved activation snapshots increase retained host memory and can prevent
in-place storage reuse. Protected row-major parameters use cheap content stamps;
foreign or non-row-major parameters require isolated comparison values. This
is a correctness repair and removal of redundant forward executions, not a
measured wall-clock or peak-memory improvement. The existing WASM WebGPU graph
already uses its separate opaque forward tokens and is unchanged.
The `sequential_forward_tape_contract` Rust example executes Dropout, seven
consuming layer/stack types, nested Topos, repeated pullbacks and gradient clears
when compiled to scalar WASM and loaded in Node. This is a host-Tensor contract
check, not a browser WebGPU benchmark. LSTM's native CPU scan clock is unavailable
on wasm32; its optional `bptt_scan_elapsed_us` is `null` there rather than calling
unsupported `std::time::Instant::now()` or fabricating a zero-duration result.
Isolated parameter and WGPU operand snapshots can add copies;
previous capture timings are not measurements of this revised path.
Python Torch feature-gate adapters and scalar WASM shared-row methods use the
compact core path described above; generic Torch broadcasts still use the
legacy expanded route. For the distinct resident NN path, use the shared-gate
graph below.

### Resident Kernel Building Block

`st_backend_wgpu::resident_tensor::pointwise::PointwisePlan::topos_resonator`
prepares an N-D forward operation from the validated
`st_kernel_contracts::topos_resonator::ToposResonatorKernel`. Its gate is either
elementwise or shared across leading axes (`[F]` or `[1, F]`). Wrapping the plan
in `PointwiseVjpPlan` reuses the existing finite guards and deterministic
broadcast adjoint: gate gradients are summed, never implicitly averaged.
Topos is one indivisible operation, so every pointwise execution policy uses
one forward dispatch. VJP recomputes the finite unroll; it does not yet store
the drive sensitivity in a resident capture.

The porous rewrite and slope are shared with `OpenCartesianTopos`; `st-core`
retains depth/volume admission, geometry and semantic audits. The low-level
resident kernel is not permission to bypass those policies. Its integration
with `Sequential` and transactional graph SGD is explicit, as described below;
the standalone low-level plan is not that integration. In particular, do not
route this gate through Scaler's module-compatible row averaging.

```sh
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo test --locked --release -p st-backend-wgpu --lib resident_tensor::pointwise -- --nocapture
cargo run --locked --release -p st-backend-wgpu --example topos_resident_learning -- /tmp/topos-resident-new.json
python tools/check_topos_shared_gate_learning.py /tmp/topos-resident-new.json /tmp/topos-resident-check-new.json
```

The probe keeps forward, mean MSE, both VJPs, gate reduction and immutable
`gate - learning_rate * gradient` updates on GPU. It submits 100 updates before
the first explicit readback, for each of two synthetic trajectories. Inputs
and targets are still uploaded, plans cover four batch layouts, and per-step
allocation remains. Results are correctness evidence, not a throughput,
pretrained-model, browser-execution or transactional-SGD claim. The CPU Torch
checker keeps its own gate across all updates; a distinct resident receipt
schema prevents relabeling historical CPU runs. Use an isolated Python runtime
without site-startup monkey patches when reproducing the comparison.

### Shared Topos Gates In Resident NN Graphs

`ToposResonator::from_shared_gate(name, gate, config, topos)` takes an isolated
snapshot of a `(1, F)` Tensor. Both shared-gate constructors now lower through
`InferencePlan::from_module` to a dedicated `ToposResonator` graph stage and
`gate` parameter role. A leading batch/sequence dimension can vary between
plans; an already compiled plan retains its fixed N-D shape. Elementwise
Topos layers remain host-only and fail explicitly when lowered to a graph.

The portable graph uses schema **v5** only when Topos is present. v2/v3/v4
plans cannot admit Topos stages or gate parameters. Rust validates coupling,
iteration count, saturation, porosity, exclusive parameter ownership and the
expanded activation's volume before allocating a GPU graph. The module's
depth admission is checked during lowering; the portable stage records the
validated finite-unroll parameters and volume bound, not a complete topos or
an optimizer geometry. Program/cache identity preserves signed-zero bits.

The graph supports resident inference, arbitrary-seed VJP, the explicit graph
learner and finite-checked transactional SGD. `exact` and `module_compatible`
both use a **sum** for Topos gates, with no additional mean. Failure guards also
check the next residual drive used by host Topos validation. They do not
materialize the core semantic audit. Readback remains explicit, and owning
outputs can add GPU copies. No CPU fallback or
implicit hypergradient/ModuleTrainer policy is introduced.

Training/autograd/learner graph preparation now retains the drive sensitivity
already evaluated by each Topos forward. This costs one private f32 per output
element per Topos stage (`4 * output_len` bytes), written in the same forward
dispatch. Backward reads it instead of unrolling the recurrence again; shared
gate reduction, checked arithmetic and dispatch count are unchanged. Inference
does not allocate this tape, and public low-level `PointwiseVjpPlan::new` keeps
its recomputing behavior. No new Python/JavaScript mathematics is introduced.

Only the latest opaque graph-forward token can reuse the tape, including
multiple independent cotangents. Another forward or an accepted input/parameter
replacement invalidates it. Older owning predictions and gradients remain
valid. Forward failures survive a zero cotangent, and a failed cotangent does
not poison a later VJP of a valid forward.

`cargo run --locked --release -p st-backend-wgpu --example topos_graph_profile`
emits nine synthetic shape/unroll cases, each with three warmups and nine GPU
pass-timestamp samples. `--reverse` reverses case order. It uses zero-rate SGD
to keep values fixed, includes full final state bits for matched verification,
and checks that gates did not change. Compile the identical probe on both
revisions and alternate the saved binaries; do not compare against a different
model or algorithm. Timestamp readback is outside timed passes: this does not
measure Python/browser end-to-end throughput or establish LLM quality.

Python and WASM reuse an existing kernel object's Rust configuration; that
object's scalar `forward`/`backward` retain their original CPU/WASM semantics.
Only the module's resident entry selects WebGPU:

```python
import spiraltorch as st

kernel = st.ToposResonatorKernel(coupling=.2, iterations=5,
                               porosity=.3, max_values=24)
model = st.nn.Sequential()
model.add_topos_resonator("topos", st.Tensor(1, 3, [.8, -.4, 1.1]), kernel)
base = model.inference_plan([2, 4, 3])
training = base.compile_graph_training_wgpu(gradient_policy="exact")
training.upload_batch_values([.25] * 24, [0.] * 24)
training.step(.03)
state = training.state_snapshot().read_state()  # Explicit checked readback.
base.apply_parameters_to(model, state.to_plan())
device = st.WgpuTensorDevice.create()
output = model(device.upload([2, 4, 3], [.25] * 24))  # Still resident.
```

```javascript
// After initializing the WebGPU-enabled WASM module as `st`:
const kernel = new st.ToposResonatorKernel(.2, 5, 1, .3, 24);
const model = new st.Sequential();
model.addToposResonator("topos", new Float32Array([.8, -.4, 1.1]), kernel);
kernel.free(); // The Module owns its configuration and gate.
const base = model.inferencePlan([2, 4, 3]);
const graph = await base.compileGraphTrainingWebGpu("exact");
graph.uploadBatch(new Float32Array(24).fill(.25), new Float32Array(24));
graph.step(.03);
const snapshot = graph.stateSnapshot();
const state = await snapshot.readState();
const updated = state.toPlan();
base.applyParametersTo(model, updated);
for (const handle of [updated, state, snapshot, graph, base, model]) handle.free();
```

Handoff remains transactional and rejects stale host parameters or existing
optimizer state unless the caller explicitly selects `reset`. A successful
handoff invalidates the original Module's resident cache, not older owning
outputs. Geometry and pretrained-model quality remain separate questions.

### Controls Before Language-Model Claims

The Torch residual adapter starts with a zero shared gate. At that point the
finite recurrence never leaves the unsaturated branch: all porosities have the
same output and gate gradient. A nonzero gate gradient proves connectivity,
not that the porous part has contributed to the language-model objective.

Let `A_n = sum(coupling**i for i in 0..iterations)`. Within the unsaturated
region, the mathematical response is just `A_n * input * gate`, for every
porosity. With zero porosity and nonnegative coupling, the entire mathematical
response reduces to `clip(A_n * input * gate, -saturation, saturation)`.
These identities are control designs, not permission to replace the checked
finite-f32 implementation: rounding, kink conventions and overflow/admission
checks still matter. Above the kernel's `f32::EPSILON` cutoff, positive porosity
has a decreasing tail beyond saturation; its finite-unroll sensitivity can
differ from both ordinary gain and hard clip.

Matched ordinary controls should therefore use the same feature count,
placement, residual strength, initial gate and local gain `A_n`. Otherwise a
comparison can measure extra capacity or an initial gradient-scale difference
rather than the porous rewrite. Matching the local gain does not make later
optimizer trajectories identical or match wall-clock compute.

`test_topos_matched_controls.py` exercises these controls through the public
`GeometryAdapterStack` in randomly initialized, frozen GPT-2 and Llama models.
Synthetic token IDs, a fixed nonzero gate and a deliberately small saturation
expose the nonlinear branch in forward and loss pullback. There is no model or
corpus download, optimizer step, pretrained/heldout score or speed claim.
Ordinary Torch formulas stay in test references; production mathematics remains
Rust-owned. Real FT must still establish whether the tail is encountered under
its actual activation scale and whether it helps beyond these ordinary controls.
The focused Rust tests exercise 300 synthetic updates and mixed
Linear/Topos/Linear VJPs. Python compares 100 updates with independent Torch;
`bindings/st-wasm/tests/resident_topos_graph.html` compares 100 actual browser
WebGPU updates with the scalar Rust/WASM kernel. Neither is a speed claim.

```sh
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo test --locked --release -p st-nn --features wgpu --lib resident::graph::topos_tests -- --nocapture
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 python -P bindings/st-py/tests/test_nn_resident_topos.py
node tools/test_resident_browser.cjs "$WEBGPU_MODULE_DIR" "$CHROME_EXECUTABLE" /tmp/topos-graph-new.json "" "" "" "" topos-resident-graph
```

### Host Shared-Gate Reproduction

The correctness-only example and independent Torch checker run two 100-update
synthetic SGD trajectories using `Parameter::apply_step`, variable row counts
and both tensor layouts:

```sh
cargo run --locked --release -p st-nn --example topos_shared_gate_probe -- /tmp/topos-shared-new.json
python tools/check_topos_shared_gate_learning.py /tmp/topos-shared-new.json /tmp/topos-shared-check-new.json
```

Both commands require fresh output paths. The checker uses a true `(1, F)`
Torch leaf gate, native broadcasting, autograd and mean loss, keeping its own
weights across all updates rather than resetting to Rust weights. This is not
a speed comparison or evidence about pretrained-model quality.
Reference values, observed values and comparison errors must remain finite;
the saved Rust f64 loss is not downcast to f32 for comparison. A historically
passing fixture does not cover every guard: the separately appended
[review corrections](../benchmarks/results/2026-10-07-topos-shared-gate-review.json)
retain the new adversarial failures and corrected validation. The original
bundle's `parent_revision` means comparison base, not immediate source parent;
its measured source's immediate parent is `7c9d4f33161685d589af18ae6f3710b817c4dd43`.

Rust callers with owned `Vec<f32>` inputs can use
`ToposResonatorOperator::capture_owned(input, gate, rows, features)` to transfer
both allocations into the tape without cloning them. The vectors are consumed
even on error, and any spare capacity is retained; borrowed `capture` remains
available with its existing behavior.
Python sequence/buffer capture and WASM capture use this owned path after
establishing Rust ownership. Foreign inputs are still copied and remain safe
to modify or discard after capture. This removes two internal N-element copies,
not the foreign-memory safety copy, and does not change the four-vector tape
or imply whole-process memory or throughput gains. See the
[owned-capture measurements](../benchmarks/results/2026-10-07-topos-owned-capture/README.md).

For matched performance comparisons, `tools/benchmark_topos_learning.py`
requests both input and broadcast-gate gradients on every route: legacy list,
bulk with recurrence recomputation, public captured bulk, and an independent
Torch finite unroll. Native routes must match bitwise; Torch comparisons allow
the documented f32 tolerance. `tools/probe_topos_capture_wasm.mjs` checks scalar
WASM parity, learning and saved-gate continuation in Node, not WebGPU speed.

The browser page `bindings/st-wasm/tests/topos_resonator_learning.html` uses
captured VJPs for its gate-learning loop. Its shared `.mjs` contract also runs
in Node CI through `tools/probe_topos_browser_learning.mjs`, keeping browser
and CLI checks aligned. The page reports import/fixture failures explicitly
and bypasses fixture HTTP caching. Native reference values must be finite
numbers; the real-WASM CI guard suite rejects 27 malformed output/input-VJP/
gate-VJP references, including nonfinite numbers and coercible strings. See the
[browser execution and reproduction record](../benchmarks/results/2026-10-07-topos-browser-captured-learning/README.md).

The [matched measurements and single-update migration replay](../benchmarks/results/2026-10-07-topos-captured-vjp/README.md)
publish all conditions, hashes and numerical receipts, not model weights or text.
