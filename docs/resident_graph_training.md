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

This addition is a Rust API and real browser-WASM fixture. General graph training
is **not yet exposed by production Python/JavaScript wrapper classes**; their
dense-only plan entrypoints explicitly reject rich/v2 graphs before requesting a
device. It does not change `pure::Tensor` storage,
generic autograd, `ModuleTrainer` or GNN execution. General forward-only compilation
and production client exposure remain follow-up work.

## Reusable Workspace

Graph construction prepares pointwise forward/VJP bind groups, contribution buffers,
and two-pass unbroadcast partials once. `step()` reuses them on the owning queue;
it creates no pointwise buffers or bind groups. Kernels, reduction order, validation
and transactional commit are unchanged. This is per-graph ownership, not a global
pool or shared mutable scratch on `PointwiseVjpPlan`: standalone `run()` calls still
produce independent results. The graph retains the extra scratch until dropped.
Command encoders, queue staging and requested snapshots can still allocate.

The shared `resident_training_bench` native/browser worker accepts `graph: true`
in its configuration. `tools/bench_resident_training_vs_torch.py --graph` and the
optional final `graph` argument of `tools/bench_resident_training_browser.cjs` run
the same fixed nine mixed-graph workloads, including more than 256 reduction rows.
Freeze a clean harness-only baseline and a clean optimized revision before building
both products. The harness validates source identity and numerical trajectories,
rotates lane order, and retains eight measured blocks after two warmups per cadence.
Each interval includes eight real SGD updates and every requested loss readback;
setup, resets and initial/final probes are excluded. Eager PyTorch does not perform
the equivalent per-stage finite checks or atomic rollback, so this comparison is
not a claim against the fastest available PyTorch configuration.

See the [source-bound workspace comparison](../benchmarks/results/2026-09-10-resident-graph-workspace/README.md)
for all measured cases, including regressions and the remaining PyTorch gap.

## Reproduce

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
