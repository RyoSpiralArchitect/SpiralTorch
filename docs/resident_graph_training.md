# Graph-Owned Resident Training

`InferencePlan` lowers existing `Scaler` and `Relu` modules alongside `Linear`,
`Gelu`, and nested `Sequential`. The opt-in general training compiler keeps
forward values, gradients, weights, biases, gains and SGD candidates on the same
WGPU device. It reuses matrix kernels and [pointwise VJP](resident_pointwise.md),
not a second host model or a diagonal-matrix approximation of Scaler.

```rust
use st_nn::{layers::{Scaler, Relu, Gelu}, Linear, Sequential};
use st_nn::resident::{InferencePlan, GraphGradientPolicy};
use st_tensor::NdLayout;

# fn example(runtime: st_backend_wgpu::runtime::WgpuRuntime) -> Result<(), Box<dyn std::error::Error>> {
let mut model = Sequential::new();
model.push(Scaler::new("input_gain", 4)?);
model.push(Linear::new("projection", 4, 3)?);
model.push(Gelu::new());
model.push(Relu::new());
model.push(Scaler::new("output_gain", 3)?);
let plan = InferencePlan::from_module(&model, NdLayout::contiguous(&[2, 5, 4])?)?;
let mut graph = plan.compile_graph_training_wgpu(runtime, GraphGradientPolicy::Exact)?;
graph.upload_batch(&vec![0.1; 40], &vec![0.2; 30])?;
for _ in 0..8 {
    graph.step(0.01)?; // enqueue only, no host readback
}
let state = graph.state_snapshot()?.read()?;
let exported = InferencePlan::from_graph_definition(state.graph)?;
let restored = InferencePlan::from_json(&exported.to_json()?)?;
# Ok(()) }
```

WASM snapshots use `read_async().await`. `upload_batch_tensors` accepts same-device
resident N-D inputs/targets, GPU-packs views, and includes upstream failure flags
on every step. `prediction_tensor` and `input_gradient_tensor` freeze terminal
tensors with the **whole transaction's guard**, even if a later gain fails.

## Contract

- `Exact` computes mathematical VJPs of mean-MSE and plain SGD.
- `ModuleCompatible` additionally divides **only gain** optimizer gradients by
  flattened leading-row count, matching the existing CPU Scaler backward policy.
  Linear gradients and input VJPs are unchanged. `GraphState` includes raw and
  effective gradients, the policy, pre-update output and post-update parameters.
- Every candidate is checked before one graph-wide decision. Nonfinite forward,
  loss, derivative, unbroadcast, update product or candidate rejects all updates,
  even at zero learning rate. A step number means submitted, not accepted: read
  a guarded snapshot to verify acceptance. Snapshots survive more steps and drop.
- Parameters can be exported before training or after rejected steps. Stable
  IDs, roles and topology are preserved, but this is **weight-only** transport:
  batches, counters and gradient policy must be supplied explicitly on resume.

## Compatibility

The specialized `compile_wgpu` / `compile_training_wgpu` Linear/GELU fast paths
are unchanged. They reject rich graphs instead of silently dropping Scaler/ReLU.
Legacy `parameter_snapshots()` exposes Linear weight/bias pairs only; use
`graph_definition().parameters()` and `with_graph_values()` for every role.

Dense JSON remains v1. V2 includes every parameter role, shape, ID and pointwise
stage, with strict fields, shape validation and the existing explicit byte budget.
Graphs require nonempty contiguous rank >= 1 input, u32-addressable buffers,
1..=4096 stages and <=8192 parameter slots. Each parameter has exactly one owner;
tying is rejected rather than guessed. Device limits may be tighter. Scaler's
Module lowering keeps its exact feature-width check, not extra broadcasting.

Current-source Python/WASM bindings expose the same graph training compiler.
Plan transport accepts v1 and v2; `is_dense` / `isDense` distinguishes the legacy
dense representation. The specialized dense compilers still reject v2 before
requesting a device. CPU-only builds transport plans but explicitly reject GPU
execution. Older published wheels may not have the graph client APIs.

This does not change `pure::Tensor` storage, generic autograd, `ModuleTrainer` or
GNN execution. [Forward-only graphs and resident N-D tensor handles](resident_graph_forward.md)
are also available in Rust, Python and WASM. Python's `upload_batch_tensors` and
JavaScript's `uploadBatchTensors` admit those same-device handles without host
readback. The clients can also upload a host batch once; intermediate values and
subsequent training steps stay on the GPU. Frozen plans do not follow live host
model updates, and the resident optimizer does not modify the source Module.

## Python And Browser Clients

```python
import spiraltorch as st

model = st.nn.Sequential()
model.add(st.nn.Scaler("input_gain", 4))
model.add(st.nn.Linear(4, 3, name="projection"))
model.add(st.nn.Gelu())
model.add(st.nn.Relu())
plan = model.inference_plan([2, 5, 4])
graph = plan.compile_graph_training_wgpu(
    gradient_policy="exact", kernel="register_2x2", accumulation="compensated",
)
graph.upload_batch_values([0.1] * 40, [0.2] * 30)
for _ in range(8):
    graph.step(0.01)  # enqueue, not a host synchronization or acceptance receipt
state = graph.state_snapshot().read_state()
for parameter in range(state.parameter_count):
    print(state.parameter_role(parameter), state.parameter_shape(parameter),
          state.parameter_gradient_values(parameter), state.effective_gradient_values(parameter))
plan_json = state.to_plan().to_json()  # post-update parameters; pre-update prediction/VJP
```

`upload_batch(Tensor, Tensor)` also accepts the exact flattened leading-axes
matrices, e.g. `(10, 4)` input and `(10, 3)` target above. Both shapes and values
are checked before either batch buffer changes. `gradient_policy` is required:
`"exact"` and `"module_compatible"` are canonical Rust-parsed names, not client
heuristics. Parameter accessors use stable **parameter IDs**, not layer indices;
they return host copies only after explicit readback.

```javascript
import init, {InferencePlan} from "./spiraltorch_wasm.js";
await init();
const plan = InferencePlan.fromJson(planJson); // Python's plan_json
const graph = await plan.compileGraphTrainingWebGpu("exact", undefined, "register_2x2", "compensated");
plan.free();
graph.uploadBatch(new Float32Array(40).fill(0.1), new Float32Array(30).fill(0.2));
graph.step(0.01);
const snapshot = graph.stateSnapshot();
graph.free();
const pending = snapshot.readState();
snapshot.free(); // pending read owns its buffers, no borrowed JS handle
const state = await pending;
const updated = state.toPlan();
const nextPlanJson = updated.toJson();
updated.free();
state.free();
```

`loss_snapshot` / `lossSnapshot` reuses `TrainingLossSnapshot`; full-state and
parameter-only reads use `GraphTrainingSnapshot` and
`GraphTrainingParametersSnapshot`. Snapshots are single-consume, including
failed reads, and survive graph drop or later batch changes. JavaScript batch
values must be `Float32Array`; counters are `bigint`. On numerical rejection,
loss/state reads expose `code="training_step_rejected"`, `stage`, and `flags`.
Parameter-only snapshots remain readable after rollback and before the first
step. Recompiling saved weights starts counters at zero and requires a new batch
and explicit policy. It does not update the source Python model in place.

## Reusable Workspace

Graph construction prepares pointwise forward/VJP bind groups, contribution buffers,
and two-pass unbroadcast partials once. `step()` reuses them on the owning queue;
it creates no pointwise buffers or bind groups. Kernels, reduction order, validation
and transactional commit are unchanged. This is per-graph ownership, not a global
pool or shared mutable scratch on `PointwiseVjpPlan`: standalone `run()` calls still
produce independent results. The graph retains the extra scratch until dropped.
Command encoders, queue staging and requested snapshots can still allocate.

On Metal, all mixed-graph forward dispatches share one compute pass. Loss,
VJP/unbroadcast, validation-copy and prepare/vote/commit boundaries remain
unchanged, as does specialized dense training. BrowserWebGpu retains its
one-forward-dispatch-per-pass schedule: paired trials were essentially flat for
immediate loss reads and could regress for deferred reads. Other backends also
keep their old schedule until measured. This changes encoding only, not
shader math, parameter ownership, intermediate checks or optimizer semantics.
Validation includes masked overflow at each of eight alternating dense/gain
positions, both gradient policies and zero/nonzero learning rates. Rejected
captures retain their guards after recovery and graph drop; no parameter may
partially commit.

The shared `resident_training_bench` native/browser worker accepts `graph: true`
in its configuration. `tools/bench_resident_training_vs_torch.py --graph` and the
optional final `graph` argument of `tools/bench_resident_training_browser.cjs` run
the same fixed nine mixed-graph workloads, including more than 256 reduction rows.
Use `--matrix wide` and the browser runner's final `wide` argument for an additional
fixed nine-case matrix: `[4,64,64]/8`, `[2,128,128]/8`, `[2,64,256]/4` (shape/depth),
again seeds 17/29/43 and eight updates. The standard matrix is unchanged; neither
matrix is an application-quality test or automatically labels a case compute-bound.
Freeze a clean harness-only baseline and a clean optimized revision before building
both products. The harness validates source identity and numerical trajectories,
rotates lane order, and retains eight measured blocks after two warmups per cadence.
Each interval includes eight real SGD updates and every requested loss readback;
setup, resets and initial/final probes are excluded. Eager PyTorch does not perform
the equivalent per-stage finite checks or atomic rollback, so this comparison is
not a claim against the fastest available PyTorch configuration.

See the [source-bound workspace comparison](../benchmarks/results/2026-09-10-resident-graph-workspace/README.md)
for all measured cases, including regressions and the remaining PyTorch gap.
The [forward-pass study](../benchmarks/results/2026-09-10-graph-training-forward-pass/README.md)
retains both standard/wide matrices and the non-adopted browser trial.

## Diagnostic Pass Profiling

Rust's `InferencePlan::profile_graph_training_wgpu(policy, tile, kernel,
accumulation).await` creates a **private timestamp-capable device** and returns
`ProfiledGraphTraining`. This is an opt-in diagnostic workspace, not a replacement
for the default runtime. Upload a host batch, call `step_profiled(rate)`, then
`read()` (native) or `read_async().await` (WASM) on its owning receipt. The result's
`report()` exposes the Rust-owned `spiraltorch.graph_training_gpu_profile.v1`
schema. This API is not yet exposed by the production Python/WASM binding classes;
the shared browser benchmark example calls the Rust API directly.

Profiling calls the **same step encoder** as ordinary training. It neither splits
nor fuses passes: Metal's forward is `forward_mixed`; `dense_backward` and
`update` may contain multiple dispatches and are not isolated GEMM timings.
Pointwise contribution and each unbroadcast pass remain distinct. Copies are
untimed; phase sums exclude them, CPU encoding, uploads, query resolution and
readback. GPU span includes inter-pass/copy gaps. Instrumentation can perturb
execution, so use the separate end-to-end benchmark for performance decisions.
Unsupported timestamp features are errors, never replaced with a CPU clock.
Absolute ticks and counters are decimal strings; zero/quantized browser intervals
are retained rather than discarded.

Until the profile is read and validated, new steps, uploads and snapshots fail
with `PendingProfile`. A decoded numerical rejection proves rollback and permits
an explicit new attempt. An abandoned/cancelled receipt or GPU/readback failure
quarantines that workspace: construct a new one rather than silently retrying.
An owning receipt remains readable after the profiler is dropped. No device or
resident tensor handle escapes the private workspace.

The shared native/browser benchmark's profiling fixture compares twelve
sequential updates from identical weights on the profiled workspace, an
uninstrumented timestamp-capable control, and an ordinary device. All losses and
final predictions, gradients and parameters are compared; three samples are
warmups and nine retained. It also checks pending/cancellation, rollback and
recovery. These are diagnostic synthetic trajectories, not long-run training.

After freezing a clean source and building the existing benchmark examples:

```bash
python -I tools/profile_resident_graph_training.py \
  --binary /path/to/resident_training_bench --source SOURCE_SHA --output /tmp/new-profile.json
node tools/test_resident_browser.cjs /path/to/bench-module /path/to/chromium \
  /tmp/new-browser-profile.json '' '' '' '' nn-graph-training-profile
```

The matrix covers both standard/wide recipes at seeds 17/29/43 with exact VJPs,
plus the smallest recipe with module-compatible gain scaling at all three seeds.
The fixture rejects reported CPU adapters and checks adapter metadata equality;
browser physical GPU identity and background contention still require independent
evidence.

## Reproduce

Production client validation (use fresh output paths and a current-source WGPU
wheel; the Torch validator runs in a separate environment without SpiralTorch):

```bash
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 python -I bindings/st-py/tests/test_nn_resident_graph_training.py -v
python -I bindings/st-py/examples/resident_graph_training.py --output /tmp/new-graph-fixture.json
cargo build --locked --release -p spiraltorch-wasm --target wasm32-unknown-unknown --features webgpu
wasm-bindgen --target web --out-dir /tmp/new-graph-client --out-name spiraltorch_wasm \
  target/wasm32-unknown-unknown/release/spiraltorch_wasm.wasm
cp /tmp/new-graph-fixture.json /tmp/new-graph-client/graph-fixture.json
node tools/test_resident_browser.cjs /tmp/new-graph-client /path/to/chromium \
  /tmp/new-graph-browser.json '' '' '' '' nn-graph-clients
python -I bindings/st-py/examples/resident_graph_training.py \
  --fixture /tmp/new-graph-fixture.json --browser-report /tmp/new-graph-browser.json \
  --output /tmp/new-graph-return.json
PYTORCH_ENABLE_MPS_FALLBACK=0 /path/to/torch-python -I tools/verify_resident_graph_clients_torch.py \
  --python-report /tmp/new-graph-fixture.json --browser-report /tmp/new-graph-browser.json \
  --return-report /tmp/new-graph-return.json --output /tmp/new-graph-torch.json
```

This freezes 24 recipes: rank 1/2/3 (including 258 rows), two seeds, both gradient
policies, scalar/sequential and register-2x2/compensated kernels. Each recipe
captures four SGD states, resumes Python's step-2 weights for two browser steps,
then returns the browser's step-4 weights for one more Python step. The independent
Torch oracle compares prediction, loss, input VJP, every raw/effective parameter
gradient and every updated parameter on CPU/MPS. The test also exercises delayed
readback after graph drop, invalid JS types, atomic uploads and gain rollback.
To verify CPU-only WASM behavior, build a separate product with `--features nn`
(without `webgpu`) and run `nn-graph-clients-cpu` against the same fixture. A CPU-only
Python wheel uses `--no-default-features --features python-default,cpu`.

The original Rust-core/browser-example fixture remains independently runnable:

The [source-bound client results](../benchmarks/results/2026-09-10-resident-graph-clients/README.md)
retain the complete Python/browser/return trajectories, independent Torch checks,
CPU-only behavior, product digests and pre-commit harness failures.

```sh
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo test --locked -p st-nn --features wgpu --test resident_graph_training
cargo run --locked -p st-nn --features wgpu --example resident_graph_training > native.json
cargo build --locked --release -p st-nn --no-default-features --features wgpu \
  --target wasm32-unknown-unknown --example resident_graph_training_browser
wasm-bindgen target/wasm32-unknown-unknown/release/examples/resident_graph_training_browser.wasm \
  --target web --out-dir browser-module --out-name spiraltorch_wasm
node tools/test_resident_browser.cjs /absolute/browser-module /absolute/chrome /absolute/browser.json \
  '' '' '' '' nn-graph-training
PYTORCH_ENABLE_MPS_FALLBACK=0 python -I tools/validate_resident_graph_training_vs_torch.py \
  native.json browser.json --devices cpu mps --output torch-replay.json
```

The same native/browser fixture checks input/middle/output gains, rank 1/2/3,
more than 256 reduction rows, both policies, eight updates after a zero-rate probe,
immutable snapshots, weight-only resume and late-failure all-parameter rejection.
It also interleaves two independent graphs while changing batches, holds 24
snapshots until after both graphs are dropped, and recovers failed workspaces.
Torch independently replays every prediction, loss, input VJP, raw/effective
parameter gradient and update. This is correctness evidence, not a benchmark.
