# Python recipes

[Documentation index](../README.md) | [Project entry](../../README.md)

This is the detailed source-tree reference, moved from the README.
Examples retain their individual feature, device, and optional-dependency requirements.
A source-tree API is not a claim that an older PyPI wheel exposes it.
Run repository commands from the repository root.

## Contents

- [Minimal usage](#minimal-usage)
- [Model Zoo discovery + launch](#model-zoo-discovery--launch)
- [Rank-K execution](#rank-k-execution)
- [Hello SpiralSession](#hello-spiralsession)
- [Canvas projector quickstart](#canvas-projector-quickstart)
- [Atlas telemetry quickstart](#atlas-telemetry-quickstart)
- [Z-space + optim quickstart](#z-space--optim-quickstart)
- [AmegagradSession quickstart](#amegagradsession-quickstart)
- [Canvas → Atlas → Session quickstart](#canvas--atlas--session-quickstart)
- [Text → optim → zspace quickstart](#text--optim--zspace-quickstart)
- [Z-space inference quickstart](#z-space-inference-quickstart)
- [SoT-3Dφ → TensorBiome quickstart](#sot-3dφ--tensorbiome-quickstart)
- [SpiralK KDSl plan rewrite quickstart](#spiralk-kdsl-plan-rewrite-quickstart)
- [Maxwell-coded envelopes → SpiralK hints quickstart](#maxwell-coded-envelopes--spiralk-hints-quickstart)
- [Streaming Z-space trainer quickstart](#streaming-z-space-trainer-quickstart)
- [Ecosystem bridges](#ecosystem-bridges)
- [Desire pipeline orchestration](#desire-pipeline-orchestration)
- [Native trainer harness](#native-trainer-harness)
- [SpiralTorchRL quickstart](#spiraltorchrl-quickstart)
- [Legacy `rl` imports](#legacy-rl-imports)
- [Open-topos learning and inference hints](#open-topos-learning-and-inference-hints)
- [SpiralTorchRec quickstart](#spiraltorchrec-quickstart)
- [SpiralSession backend planning](#spiralsession-backend-planning)

## Minimal usage

### Model Zoo discovery + launch

```python
import spiraltorch as st

entries = st.model_zoo.list_models(task="classification")
print("classification recipes:", [entry.key for entry in entries[:5]])

suggested = st.model_zoo.suggest_models(
    "llm_char",
    task="language-modeling",
    prefer_tags=["coherence"],
)
print("suggested:", [entry.key for entry in suggested[:3]])

zspace_stream = st.model_zoo.suggest_models(
    focus="zspace_stream",
    available_only=True,
    limit=3,
)
print("zspace stream track:", [entry.key for entry in zspace_stream])

cmd = st.model_zoo.build_model_command("mlp_regression", "--help")
print("command:", " ".join(cmd))
```

```bash
spiral-model-zoo focuses
spiral-model-zoo list --task language-modeling
spiral-model-zoo suggest llm_char --task language-modeling --prefer-tag coherence
spiral-model-zoo suggest --focus zspace_stream --available-only
spiral-model-zoo run zspace_stream_online_vision -- --steps 16 --flush-every 2
spiral-model-zoo run zspace_stream_frame_aggregator -- --steps 12 --native-frames
spiral-model-zoo run mlp_regression -- --help
```

### Rank-K execution

```python
import spiraltorch as st

plan = st.plan_topk(rows=2, cols=4, k=2, backend="auto")
print(plan.kind, plan.effective_backend)
print("tile/workgroup:", plan.tile, plan.workgroup)
print(plan.to_unison_script().splitlines()[0])
print(plan.fft_spiralk_hint().splitlines()[0])
```

### Hello SpiralSession

```bash
python bindings/st-py/examples/hello_session.py
```

Aligns a barycenter with a hypergrad tape, prepares a Sequential module, and
finishes a roundtable epoch entirely from Python.

```python
from spiraltorch import Tensor, Hypergrad, LanguageWaveEncoder

encoder = LanguageWaveEncoder(-1.0, 0.5)
wave = encoder.encode_z_space("SpiralTorch in Rust")
rows, cols = wave.shape()
z = Tensor(rows, cols, [0.0] * (rows * cols))

tape = Hypergrad(-1.0, 0.05, *z.shape())
tape.accumulate_pair(z, wave)
tape.apply(z)
print(z.tolist())
```

### Canvas projector quickstart

```bash
python bindings/st-py/examples/canvas_projector_quickstart.py
```

Renders a radial energy field into an RGBA surface (written as
`spiraltorch_canvas.ppm`) and prints the row-wise FFT power spectrum tensor
shape.

### Atlas telemetry quickstart

```bash
python bindings/st-py/examples/atlas_quickstart.py
```

Builds an `AtlasFrame` from a Python dict and prints the district aggregation.

```python
import spiraltorch as st

route = st.telemetry.AtlasRoute()
route.push_bounded(
    st.telemetry.AtlasFrame.from_metrics({"psi.total": 1.0}, timestamp=0.0),
    bound=32,
)
route.push_bounded(
    st.telemetry.AtlasFrame.from_metrics({"psi.total": 1.5}, timestamp=1.0),
    bound=32,
)
print("summary:", route.summary()["frames"], "frames")
print("psi perspective:", route.perspective_for("Concourse", focus_prefixes=["psi."])["guidance"])
```

### Z-space + optim quickstart

```bash
python bindings/st-py/examples/zspace_optim_quickstart.py
```

```python
import spiraltorch as st

opt = st.optim.Amegagrad((1, 3), curvature=-0.9, hyper_learning_rate=0.03, real_learning_rate=0.02)
weights = st.Tensor(1, 3, [0.2, -0.1, 0.05])

opt.accumulate_wave(st.Tensor(1, 3, [0.4, -0.6, 0.2]))
opt.step(weights)  # tunes rates via DesireGradientControl + applies both tapes
print("weights:", weights.tolist())
topos_audit = opt.topos_telemetry_contract()
print(topos_audit["semantic_owner"], topos_audit["summary"])

trainer = st.ZSpaceTrainer(z_dim=4)
loss = trainer.step({"speed": 0.2, "memory": 0.1, "stability": 0.9, "gradient": opt.real.gradient()})
print("z:", trainer.state, "loss:", loss)
print("optimizer audit:", trainer.last_optimizer_report["adam"])
```

The class contains no Python Adam or fractional-FFT fallback. Initialization,
checkpoint restore, observation normalisation, the FFT-derived periodic
fractional Sobolev gradient, bounded Topos controls, and the complete Adam
transition are owned by `st-core::runtime::zspace_optimizer`. Z-space partial
and telemetry fusion, including Amegagrad's `topos.*` flattening and audit
summary, is owned by `st-core::telemetry::zspace_fusion`; Python only collects
sources, caches inference metadata, serializes transition-plus-commit per
trainer, and commits a successful Rust report. Native FFT work releases the GIL. The low-level
`zspace_meta_optimizer_init`, `zspace_meta_optimizer_restore`, and
`zspace_meta_optimizer_step` functions expose the same versioned contract for
custom orchestrators.

The same report can control native model-parameter learning rates without
reimplementing its Topos rules in Python:

```python
model = st.nn.Sequential()
model.add(st.nn.Linear("controlled", 3, 2))
module_trainer = st.nn.ModuleTrainer(
    backend="cpu",
    curvature=-0.9,
    hyper_learning_rate=0.03,
    fallback_learning_rate=0.02,
)
module_trainer.prepare(model)
receipt = module_trainer.apply_zspace_meta_optimizer_report(
    model,
    trainer.last_optimizer_report,
)
print(receipt["absolute_learning_rate_scale"], receipt["changed"])
```

Rust validates the complete report, re-derives its Topos learning-rate scale,
rejects stale or conflicting steps, and converts the absolute scale to an
idempotent relative update. Latent clipping, fractional regularization, and
gradient bias remain inside the latent optimizer.

### AmegagradSession quickstart

```bash
python bindings/st-py/examples/amegagrad_session_quickstart.py
```

### Canvas → Atlas → Session quickstart

```bash
python bindings/st-py/examples/canvas_atlas_session_quickstart.py
```

Runs a small closed-loop demo:

- `AmegagradSession` updates weights
- `CanvasProjector` renders + emits a loopback patch
- `CanvasProjector.emit_atlas_frame(...)` streams metrics into `AtlasRoute`

### Text → optim → zspace quickstart

```bash
python bindings/st-py/examples/text_optim_zspace_quickstart.py
```

```python
import spiraltorch as st

encoder = st.LanguageWaveEncoder(-1.0, 0.5)
rows, cols = encoder.encode_z_space("SpiralTorch").shape()
opt = st.optim.Amegagrad((rows, cols), curvature=encoder.curvature())
weights = st.Tensor(rows, cols, [0.0] * (rows * cols))

opt.absorb_text(encoder, "Z-space wants to be steerable.")  # pads/truncates to (rows, cols)
opt.step(weights)
print("weights (head):", [v for row in weights.tolist() for v in row][:5])

trainer = st.ZSpaceTrainer(z_dim=4)
trainer.step({"speed": 0.2, "memory": 0.1, "stability": 0.9, "gradient": opt.real.gradient()})
```

### Z-space inference quickstart

```bash
python bindings/st-py/examples/zspace_inference_quickstart.py
```

```python
import spiraltorch as st

trainer = st.ZSpaceTrainer(z_dim=4)
checkpoint = trainer.state_dict()
checkpoint["z"] = [0.12, -0.03, 0.48, -0.2]
trainer.load_state_dict(checkpoint)
control = st.ZSpacePartialBundle(
    {"speed": 0.2, "memory": 0.1, "stability": 0.9, "gradient": [0.1, 0.0, 0.0, 0.0]},
    gradient_basis="example.observed.control.v1",
)
loss = trainer.step_partial(control)
print("z:", trainer.state, "loss:", loss)
print("last inference:", trainer.last_inference.residual, trainer.last_inference.confidence)
```

`ZSpacePosterior`, `decode_zspace_embedding(...)`, and `infer_from_partial(...)`
are Python orchestration surfaces over one Rust-owned posterior contract. The
spectral decode, fractional energy, gradient normalization, metric aliases,
barycentric projection, residual/confidence update, and telemetry adjustment
live in `st-core::inference::zspace_posterior`; Python does not carry fallback
formulas. Use `zspace_posterior_decode(...)` or
`zspace_posterior_project(...)` when a versioned audit payload is preferable to
the high-level dataclasses. Contract v2 reports Parseval-normalized
`spectral_energy`, `parseval_relative_error`, `fractional_energy_ratio`,
`spectral_centroid`, and `spectral_bins`. Its `gradient` is always the latent
finite-difference gradient in
`ZSPACE_POSTERIOR_LATENT_GRADIENT_BASIS`; basis-tagged external gradients are
preserved separately as `control_gradient` and are never padded, normalized, or
used to replace the latent gradient. `ZSpaceTrainer.step_partial(...)` therefore
trains from the Rust latent gradient; it retains the external control for audit
but does not apply it without a future explicit control-to-latent projection.
Confidence is `exp(-observed_metric_rms)`
times a conservative telemetry reliability no greater than one:

```python
contract = st.zspace_posterior_project(
    [0.12, -0.03, 0.48, -0.2],
    {"speed": 0.3, "mem": -0.2, "gradient": [0.2, -0.1]},
    gradient_basis="example.optimizer.control.v1",
    telemetry={"psi": {"energy": 2.0, "focus": 0.4}},
)
assert contract["semantic_owner"] == "st-core::inference::zspace_posterior"
assert contract["gradient_basis"] == st.ZSPACE_POSTERIOR_LATENT_GRADIENT_BASIS
assert contract["control_gradient"]["basis"] == "example.optimizer.control.v1"
```

Coherence-to-Z-space projection is also Rust-owned. Native
`CoherenceDiagnostics` exposes its measured weights, entropy, energy ratio,
fractional order, Z-bias, repair counts, and pre-discard counts directly.
`zspace_coherence_project(...)` returns the complete versioned contract;
`coherence_partial_from_diagnostics(...)` returns only its `partial` map for
existing inference pipelines:

```python
import spiraltorch as st

topos = st.OpenCartesianTopos(-0.9, 1e-5, 10.0, 32, 1024)
sequencer = st.ZSpaceCoherenceSequencer(16, 2, -0.9, topos=topos)
x = st.Tensor((1, 16), data=[0.1] * 8 + [0.8] * 8)
_, coherence, diagnostics = sequencer.forward_with_diagnostics(x)
contour = sequencer.emit_linguistic_contour(x)
control = diagnostics.control
contract = st.zspace_coherence_project(
    diagnostics,
    coherence=coherence,
    contour=contour,
)
partial = st.coherence_partial_from_diagnostics(diagnostics, contour=contour)

assert contract["contract_version"] == "spiraltorch.zspace_coherence_projection.v2"
assert contract["semantic_owner"] == "st-core::inference::zspace_coherence"
assert contract["derived"]["distribution_source"] == "normalized_weights"
assert contract["classification"]["label"] == diagnostics.observation.label
assert control["contract_version"] == "spiraltorch.zspace_coherence_control.v1"
assert contract["control"]["spectral_pressure"] == control["spectral_pressure"]
assert partial["speed"] == contract["partial"]["speed"]
```

Rust recomputes normalized HHI concentration and `H / ln(N)` from the measured
probability simplex, so `speed` and `stability` do not drift merely because a
model uses more Maxwell channels. Projection v2 also verifies that entropy,
support counts, and dominant channel agree with that simplex, and that raw mean
coherence agrees with the supplied response. Missing or contradictory evidence
and invalid gains fail at the contract boundary instead of being silently
replaced or clamped by Python.
`ModuleTrainer.push_coherence_diagnostics(diagnostics)` feeds that same Rust
control payload into learning; raw mean coherence and raw Shannon entropy remain
audit metrics, while normalized radius, entropy, and pressure drive the policy.
Trace replay is guarded and rejects missing, legacy, or tampered control fields.
Current trace schema v2 includes a complete Rust-owned simplex witness rather
than asking replay code to trust independently stored entropy and concentration
scalars. Build and validate the same portable evidence directly from Python:

```python
witness = st.zspace_coherence_distribution_witness([0.5, 0.3, 0.2])
summary = st.validate_zspace_coherence_distribution_witness(witness)

assert witness["contract_version"] == (
    "spiraltorch.zspace_coherence_distribution_witness.v1"
)
assert witness["semantic_backend"] == "rust"
assert summary["weight_mass"] == 1.0
```

Both calls require the compiled Rust core. Python performs input shaping and
orchestration only; it does not recompute entropy, concentration, or effective
channel count.
`diagnostics.observation.signature` exposes the same Rust-owned normalized
entropy, concentration, effective channel count, label, reason, formula,
contract version, and policy thresholds. Python never reclassifies the trace.
Use `diagnostics.classify(...)` or pass `background_energy_ratio_max` and
`cascade_energy_ratio_min` to `zspace_coherence_project(...)` to run a custom
policy in Rust.

### SoT-3Dφ → TensorBiome quickstart

```bash
python bindings/st-py/examples/sot_biome_quickstart.py
```

### SpiralK KDSl plan rewrite quickstart

```bash
python bindings/st-py/examples/spiralk_plan_rewrite_quickstart.py
```

### Maxwell-coded envelopes → SpiralK hints quickstart

```bash
python bindings/st-py/examples/maxwell_spiralk_bridge_quickstart.py
```

### Streaming Z-space trainer quickstart

```bash
python bindings/st-py/examples/zspace_stream_training_quickstart.py
```

### Ecosystem bridges

Native tensors support legacy and DLPack 1.0 versioned exchange for contiguous
2D CPU `float32` buffers. NumPy can consume them directly: use
`np.from_dlpack(tensor, copy=False)` to share or `copy=True` for an independent
writable array (NumPy 2.x). Read-only imports are preserved until Rust mutation
materializes owned storage. See the [DLPack contract and examples](../dlpack_interop.md)
for version negotiation, legacy compatibility, and lifetime/gradient boundaries.

SpiralTorch tensors can flow into PyTorch or JAX without copies thanks to the
`spiraltorch.ecosystem` helpers. CuPy round-trips also accept optional CUDA
streams so you can coordinate asynchronous pipelines, and the helpers can
resolve friendly stream aliases on demand:

```python
import spiraltorch as st
from spiraltorch.ecosystem import (
    tensor_to_cupy,
    tensor_to_jax,
    tensor_to_tensorflow,
    tensor_to_torch,
    cupy_to_tensor,
    jax_to_tensor,
    tensorflow_to_tensor,
    torch_to_tensor,
)

spiral = st.Tensor(2, 2, [1.0, 2.0, 3.0, 4.0])

try:
    import torch

    torch_tensor = tensor_to_torch(spiral, dtype=torch.float32)
    roundtrip = torch_to_tensor(torch_tensor)
    print("torch:", roundtrip.shape())
except Exception as exc:
    print("torch bridge skipped:", type(exc).__name__)

try:
    jax_array = tensor_to_jax(spiral)
    spiral_again = jax_to_tensor(jax_array)
    print("jax:", spiral_again.shape())
except Exception as exc:
    print("jax bridge skipped:", type(exc).__name__)

try:
    # stream can be an explicit cupy.cuda.Stream, or a lazy alias such as
    # "current" (resolve the active stream) or "null" (select the default stream).
    cupy_array = tensor_to_cupy(spiral, stream="current")
    spiral_from_cupy = cupy_to_tensor(cupy_array, stream="current")
    print("cupy:", spiral_from_cupy.shape())
except Exception as exc:
    print("cupy bridge skipped:", type(exc).__name__)

try:
    tf_tensor = tensor_to_tensorflow(spiral)
    spiral_from_tf = tensorflow_to_tensor(tf_tensor)
    print("tensorflow:", spiral_from_tf.shape())
except Exception as exc:
    print("tensorflow bridge skipped:", type(exc).__name__)
```

```python
import spiraltorch as st
from spiraltorch.nn import Linear, MeanSquaredError, Sequential

trainer = st.nn.ModuleTrainer(
    backend="cpu",
    curvature=-1.0,
    hyper_learning_rate=1e-2,
    fallback_learning_rate=1e-2,
)
schedule = trainer.roundtable(
    2,
    1,
    st.nn.RoundtableConfig(top_k=1, mid_k=1, bottom_k=1, here_tolerance=1e-5),
)
model = Sequential()
model.add(Linear(2, 1, name="layer"))
model.attach_hypergrad(curvature=-1.0, learning_rate=1e-2)

loss = MeanSquaredError()
x = st.Tensor.rand(2, 2, seed=3)
y = st.Tensor.rand(2, 1, seed=4)
stats = trainer.train_epoch(model, loss, [(x, y)], schedule)

print(f"roundtable avg loss {stats.average_loss:.6f} over {stats.batches} batch")
```

### Desire pipeline orchestration

```python
import spiraltorch as st

pipeline = st.nn.DesirePipeline(vocab_size=2, concepts=2)
step = pipeline.step([1.2, -0.4], previous_token=0, concept=[0.6, 0.4])
print("phase", step["phase"], "entropy", step["entropy"])

adapter = st.build_desire_adapter_from_downstream_hook(
    {
        "geometry_bias_coherence": {"score": 0.7},
        "top_probability": 0.8,
    }
)
pipeline.ingest_geometry_bias(adapter["geometry_bias_signal"], source="zspace")
print("geometry bias:", pipeline.geometry_bias_metrics())
```

### Native trainer harness

Python callers can keep the training loop small while still using the Rust
roundtable trainer. For heavier HPO/serving flows, use this loop as the inner
objective and wrap it with your Optuna/Ray/BentoML/TorchServe tool of choice.

Optimizer ingress is also Rust-owned. `ModuleTrainer` validates negative finite
curvature, positive finite learning rates, optional realgrad, and gradient
clipping through `st-core::runtime::trainer_optimizer`; Python does not carry a
parallel validator. Invalid control updates raise `ValueError` without changing
an already active realgrad rate or clipping guard. The versioned receipt is
available for run cards and preflight logs:

```python
trainer.enable_realgrad(5e-3)
trainer.set_grad_clip_max_norm(1.0)
optimizer_contract = trainer.optimizer_config_contract()
assert optimizer_contract["semantic_backend"] == "rust"
```

The same Rust owner now defines optimizer resume state. A checkpoint includes
hypergrad and realgrad accumulators, Topos momentum and custom guards, local
spectral state, curvature/spectral/SoftLogic policy state, cumulative backend
counters, phase-event thresholds and edge-detector history, and the execution
topology. Model values stay in the module
`state_dict`; per-parameter fingerprints reject a mismatched model before any
state is mutated:

```python
model_state = model.state_dict()
runtime_bundle = trainer.runtime_checkpoint_bundle(model)

# Attach the same shared Desire bridges or distributed provider first.
# Rust-owned PSI/coherence runtimes can bootstrap from the bundle.
resumed_model.load_state_dict(model_state)
resumed_trainer.prepare(resumed_model)
receipt = resumed_trainer.restore_runtime_checkpoint_bundle(
    resumed_model,
    runtime_bundle,
)
assert receipt["semantic_backend"] == "rust"
assert receipt["deterministic_resume_ready"] is True
```

The bundle is versioned by `st-core::runtime::trainer_checkpoint`. SHA-256
digests bind its independently versioned optimizer and external child payloads,
so accidentally mixing or modifying either child fails before trainer,
parameter, Desire, PSI, or coherence state is changed. Model values remain
external and fingerprint-guarded. Concrete resources must be attached before
the single restore call, and unsupported external components make the bundle
fail closed.

The lower-level `optimizer_checkpoint()` and `external_state_checkpoint()`
methods remain available for component audit and staged orchestration.
`external_state_required` names
enabled bridges or distributed runtimes whose own state must be checkpointed
alongside this optimizer contract; Python neither suppresses nor reinterprets
that limitation. The optimizer receipt's `deterministic_resume_ready` describes
the optimizer payload in isolation, so it remains `False` whenever that list is
non-empty even when the combined bundle is complete. Only the bundle receipt
composes child coverage and native resource reattachment into one readiness
claim.

Supported external runtime state has a separate Rust-owned checkpoint. Python
only transports this payload and orchestrates native restore; it does not
rebuild component accounting or readiness rules. The v4 contract captures the
full FIFO consumed by `DesireTrainerBridge`, Desire roundtable controls/latest
impulse/pending trainer summary, PSI configuration/EMA/sample clock, and known
accumulator-provider descriptors. It also records the ZSpaceTrace subscription
topology plus the trainer-pending and bridge-latest coherence signals. Those
signals retain only the distribution witness, raw observations, repair counts,
and classification policy; control metrics and labels are re-derived by the
canonical Rust coherence contract during validation and restore. Desire
timestamps use exact `unix_seconds + subsec_nanos` fields, so browser transport
does not round Rust state to milliseconds. Other unsupported controllers remain
explicit in `unresolved_components`, while an accumulator resource must already be
reattached and verified before the receipt can report deterministic resume:

```python
external_state = runtime_bundle["external"]
assert external_state["semantic_backend"] == "rust"
assert receipt["payload_complete"] is True
assert receipt["deterministic_resume_ready"] is True
```

The same v4 envelope captures a preattached `RoundtableGnnBridge` as one locked
history/latest snapshot, including its history limit and the trainer's last
published signal. Checkpoints retain raw band energy, schedule occupancy,
spectral observations, and exact issuance timestamps. Rust rejects malformed
evidence and re-derives all message-passing multipliers; Python neither computes
nor repairs GNN influence values.

Enable Rust PSI metering through the roundtable config when its EMA should be
part of continuation state. A resumed trainer reconstructs this Rust-owned
meter directly from the checkpoint; unlike bridges and distributed providers,
it does not require a placeholder meter or a dummy training step first:

```python
config = st.nn.RoundtableConfig(psi_enabled=True)
schedule = trainer.roundtable(rows, cols, config)
```

```python
import spiraltorch as st
from spiraltorch.nn import Linear, MeanSquaredError, ModuleTrainer, RoundtableConfig, Sequential

dataset = [
    (st.Tensor(1, 2, [0.0, 1.0]), st.Tensor(1, 1, [1.0])),
    (st.Tensor(1, 2, [1.0, 0.0]), st.Tensor(1, 1, [0.0])),
]

for label, lr in [("warmup", 1e-2), ("refine", 5e-3)]:
    trainer = ModuleTrainer(
        backend="cpu",
        curvature=-1.0,
        hyper_learning_rate=lr,
        fallback_learning_rate=lr,
    )
    schedule = trainer.roundtable(
        1,
        1,
        RoundtableConfig(top_k=1, mid_k=1, bottom_k=1, here_tolerance=1e-5),
    )
    model = Sequential()
    model.add(Linear(2, 1, name="layer"))
    model.attach_hypergrad(curvature=-1.0, learning_rate=lr)
    stats = trainer.train_epoch(model, MeanSquaredError(), dataset, schedule)
    print(label, f"avg_loss={stats.average_loss:.6f}")
```

## SpiralTorchRL quickstart

`spiraltorch.spiral_rl` packages the Rust-side reinforcement-learning harness so
Python notebooks can select actions, update native agents, and inspect compact
state dictionaries without reimplementing the loop in Python.

### Legacy `rl` imports

Older notebooks sometimes `import rl` directly. The Python binding now
discovers whether the native wheel exposes `spiraltorch.rl` before wiring a
lazy import hook. If another library has already populated `sys.modules["rl"]`
we leave it untouched; otherwise importing `rl` defers to the SpiralTorch
module on demand. Wheels built without SpiralTorchRL skip the hook entirely so
third-party modules remain unaffected.

```python
from spiraltorch.spiral_rl import stAgent

agent = stAgent(state_dim=4, action_dim=2, discount=0.97, learning_rate=0.02)

state = 0
next_state = 1
trace = agent.select_action_trace(state)
action = int(trace["action"])
agent.update(state, int(action), reward=1.0, next_state=next_state)

print("action:", action)
print("policy:", agent.policy_report(state))
print("trace:", trace)
print("epsilon:", agent.state_dict()["epsilon"])
```

The generic `spiraltorch.rl.Agent` wrapper exposes the same loop with an
explicit config object and exploration schedule:

```python
from spiraltorch.rl import Agent, AgentConfig, EpsilonGreedy

config = AgentConfig(
    "dqn",
    state_dim=4,
    action_dim=2,
    gamma=0.97,
    lr=0.02,
    exploration=EpsilonGreedy(0.2, 0.05, 100),
    seed=7,
)
agent = Agent(config)
trace = agent.select_action_trace(0)
action = int(trace["action"])
agent.update(0, int(action), 1.0, 1)
print(agent.algo, agent.policy_report(0), agent.state_dict()["epsilon"])
```

The RL surface is intentionally compact today: keep state/action loops native,
use `policy_report(state)` or `select_action_trace(state)` to audit Q-values,
epsilon, and greedy-vs-exploratory choices, then export `state_dict()` for
handoff.

For API-model topological routing, use
`api_llm_topos_sweep_route_rewards(report, profile="grounded")` to convert a
`run_api_llm_topos_sweep(...)` report into bounded route rewards, then call
`train_stagent_topos_route_policy(report, agent, profile="grounded")` to update
an stAgent-shaped policy and capture the selected route trace. Use
`api_llm_topos_route_policy_selection(policy, report=..., topos_profiles=...)`
to recover the selected route record and, when raw profiles are available,
rebuild the request/runtime-route payload for the next hosted-model call. The
returned `resolution_semantics` preserves the Rust resolver witness; Python
does not recalculate the selected reward or fall back from a stale label to a
different positional route.

## Open-topos learning and inference hints

For shared placement across geometric families, import `GeometryAdapterStack`.
It owns adapters separately from the base model and installs explicit, scoped
output hooks at named modules. `with geometry.attach(model):` supports ordinary
loss/backward/optimizer steps without renaming model weights; checkpoint state
binds the adapter types and placement order. See the
[multi-site learning guide](../geometry_adapter_stack.md) and offline
`examples/hf_geometry_adapter_stack.py` for mixed WaveGate/Topos/elliptic/fractional
learning. This is orchestration over existing Rust operators, not a GPU-resident
backend, automatic model patcher or language-quality claim.

For a differentiable hidden-state intervention rather than a control hint,
`from spiraltorch import ToposResonatorAdapter, topos_resonator_autograd` now
connects the Rust finite-unroll recurrence and VJP to Torch/HF training.
The gate starts at zero for an identity residual and is stored with its Rust
recipe in `state_dict`. This initial bridge is **f32 CPU execution with explicit
host transfers**, not resident GPU training. See the
[placement, checkpoint and WASM guide](../geometric_learning_bridge.md)
and the offline `examples/topos_resonator_learning.py` control experiment.

`topos_control_signal()` turns an open-cartesian guard into one compact pressure
signal, while `topos_training_hints()` and `topos_inference_hints()` split the
same signal into named controls for local learning loops and hosted-model
runtime requests.

```python
import spiraltorch as st

topos = st.hypergrad_topos(max_depth=10, max_volume=100)
signal = st.topos_control_signal(topos, observed_depth=4, visited_volume=25)
training = st.topos_training_hints(signal)
snapshot = st.topos_optimizer_snapshot(
    topos,
    sequence=1,
    hyper_learning_rate=0.04,
    real_learning_rate=0.02,
    gain=0.75,
    observed_depth=4,
    visited_volume=25,
)
adapter = st.topos_runtime_adapter(signal, request_options={"base_temperature": 0.8})

trainer = st.ZSpaceTrainer(z_dim=4, topos_control_gain=0.5)
trainer.step(st.z.metrics(speed=0.0, memory=0.0, stability=0.0, telemetry={"topos": signal}))

print("gradient bias:", training["gradient_bias_scale"])
print("snapshot hyper rate:", snapshot["optimizer_application"]["hyper_learning_rate"])
print("runtime temperature:", adapter["request"]["temperature"])
```

`topos_optimizer_snapshot()` is the step boundary for optimizer integration. Its v3 contract is
owned by Rust and binds one sequence-checked control bundle to learning rates plus the gradient
state configured on both Amega tapes. Bias is scale-relative,
`g_biased[i] = g[i] + rms(g) * bias_scale * basis[i % 10]`, so its bias term is exactly zero for a
zero raw gradient. Clipping is also scale-relative: `clip_scale=1` is an exact no-op; otherwise
Rust clamps `g_biased` at `rms(g_biased) / (1 - clip_scale)`. Momentum then follows
`m_t = damping * m_(t-1) + (1 - damping) * g_clipped`, so hidden state cannot retain an unclipped
outlier. `damping=0` exactly preserves the stateless guarded update. Python transports and audits
these fields but never reconstructs any rule.

`Amegagrad.tune()` enters one native configuration boundary that commits rates and both tape
controls only after Rust validation and momentum allocation succeed.
`Amegagrad.step()` then enters one native combined-step boundary: hyper tape, real tape, weights,
raw gradients, and momentum are committed together only after both Rust updates succeed.

## SpiralTorchRec quickstart

`spiraltorch.rec` brings the SpiralTorchRec factorisation stack to notebooks and
production jobs alike. Embeddings stay guarded by the open-cartesian topos so
psychoid limits never drift while running alternating updates in pure Rust.

```python
from spiraltorch.rec import Recommender

rec = Recommender(users=8, items=12, factors=4, learning_rate=0.05, regularization=0.002)

ratings = [
    (0, 0, 5.0),
    (0, 1, 3.0),
    (1, 0, 4.0),
    (1, 2, 4.5),
]

report = rec.train_epoch(ratings)
print(report.rmse, report.samples)
print("score:", rec.predict(0, 0))
print("top-k:", rec.recommend_top_k(0, 3))
```

### SpiralSession backend planning

`SpiralSession` is intentionally small: Python chooses orchestration order while
Rust captures one executable runtime plan and owns its semantics. Rank planning,
native trainers, schedules, checkpoints, and replay all inherit that exact plan;
they do not independently re-read backend heuristics or environment overrides.
The session plan uses Rust's `deferred` component resolution: unobserved dynamic
shapes are not called native, but the committed policy is enforced when each
operation runs. Standalone workload preflight remains `concrete` and fail-closed.

```python
from spiraltorch import SpiralSession

session = SpiralSession(backend="wgpu", tensor_util_wgpu_min_values=37)
print(session.requested_backend, "->", session.effective_backend)
print("runtime:", session.device_preflight["runtime_status"])

rank = session.plan_topk(rows=8, cols=64, k=4)
trainer = session.trainer()
print(rank.kind, rank.effective_backend, rank.tile)
assert rank.runtime_execution_plan_output_sha256 == session.runtime_execution_plan_output_sha256
assert trainer.runtime_execution_plan_output_sha256 == session.runtime_execution_plan_output_sha256

replayed = SpiralSession.from_runtime_execution_plan(session.runtime_execution_plan)
assert replayed.runtime_execution_plan_output_sha256 == session.runtime_execution_plan_output_sha256
```

`runtime_execution_plan` returns a defensive copy. Replay validates commitments,
the receiving Rust build, and current backend readiness. A supplied plan cannot be
combined with backend/capability/config overrides. For `backend="auto"`, Python
asks Rust for WGPU readiness first, materializes that plan when ready, and falls
back to CPU only when Rust returns its explicit unavailable signal and the captured
policy permits fallback. Plan validation, transport, configuration, and unrelated
Python errors propagate. Rust captures
`SPIRALTORCH_STRICT_GPU` and `SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES` once.
Under strict policy, `auto` does not retry CPU and small tensor utilities stay on
WGPU rather than crossing the threshold route. Inspect the captured values with
`spiraltorch.resolve_runtime_execution_config()`.
