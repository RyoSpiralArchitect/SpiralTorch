# SpiralTorch Python bindings

This package exposes a thin, dependency-light bridge to SpiralTorch's
Rust-first training stack. The wheel ships the same Z-space tensors,
hypergrad tapes, and unified rank planners that power the Rust API—no
NumPy, no PyTorch, and no shim layers.

## What's included

- `Tensor`, `ComplexTensor`, and `OpenTopos` for dependency-free
  geometry experiments.
- `LanguageWaveEncoder` + `Hypergrad` so Python callers can stream Z-space
  text, accumulate gradients, and project back into the Poincaré ball.
- `TensorBiome` to cultivate open-topos rewrites, weight shoots, stack the
  harvest, and guard tensors that can be re-imported into Z-space.
- Unified planning helpers (`plan`, `plan_topk`, `describe_device`) that
  reuse the same heuristics as the Rust executors.
- ROCm probing (`hip_probe`) so Python callers can reflect the stubbed
  device hints shared with the Rust runtime.
- Z-space barycentre solver (`z_space_barycenter`) to mix colour-field
  priors and chart couplings directly from Python.
- Loss-monotone barycenter intermediates (`BarycenterIntermediate`) that plug
  into `Hypergrad.accumulate_barycenter_path` so tapes converge along the
  same Z-space corridor as the solver.
- High-level orchestration via `SpiralSession` / `SpiralSessionBuilder` so
  callers can select devices, spawn hypergrad tapes, plan kernels, and solve
  barycentres with a few intuitive method calls. Structured results are
  returned through the new `ZSpaceBarycenter` class.
- `SpiralLightning` harness for quick notebook experiments—prepare modules,
  run epochs, and stream results without manually juggling trainers or
  schedules.
- Streaming dataset helpers via `spiraltorch.dataset`—build a
  shuffle/batch/prefetch pipeline entirely in Rust using the native
  `DataLoader`.
- Non-commutative differential traces via `SpiralSession.trace(...)` which emit
  `SpiralDifferentialTrace` builders and `DifferentialResonance` snapshots to
  blend homotopy flows, functor derivatives, recursive barycenter energies, and
  \(\infty\)-tower projections—optionally wiring the result straight into a
  `Hypergrad` tape.
- SoT-3Dφ spiral planners (`spiraltorch.sot`) that collapse to Z-space tensors,
  grow full TensorBiomes via `SoT3DPlan.grow_biome(...)`, and stitch directly
  into `SpiralSession.trace(...)` for geometry-aware exploration loops.
- Z-space projector bindings (`spiraltorch.nn.ZSpaceProjector`) so spiral
  trajectories can be rendered onto the canvas or reused inside sequential
  transformer stacks.
- Deployment and optimisation bridges via `spiraltorch.integrations`: archive
  TorchServe models, persist BentoML runners, explore hyperparameters with
  Optuna or Ray Tune, and export trained modules to ONNX—all behind ergonomic
  Python call sites.
- Reinforcement learning harness via `spiraltorch.rl`—SpiralTorchRL keeps
  policy gradients inside Z-space tensors, exposes hypergrad-enabled updates,
  and streams geometric rewards without leaving Rust.
- Recommendation toolkit via `spiraltorch.rec`—SpiralTorchRec factors user/item
  lattices under open-cartesian topos guards so embeddings stay psychoid-safe
  while training entirely in Rust.

## Building wheels

The binding mirrors the Rust feature flags. Pick the backend(s) you need
and maturin will bake the appropriate artefact:

```bash
pip install maturin==1.*

# CPU + WebGPU (default)
maturin build -m bindings/st-py/Cargo.toml --release --features wgpu

# Metal (macOS)
maturin build -m bindings/st-py/Cargo.toml --release --features mps

# CUDA (NVRTC toolchain expected on PATH)
maturin build -m bindings/st-py/Cargo.toml --release --features cuda

# HIP / ROCm (use hip-real for the RCCL path)
maturin build -m bindings/st-py/Cargo.toml --release --features "hip hip-real"
```

## Minimal usage

### Rank-K execution

```python
>>> import spiraltorch as st
>>> x = st.Tensor(2, 4, [0.1, 0.7, -0.2, 0.4, 0.9, 0.5, 0.6, 0.0])
>>> vals, idx = st.topk2d_tensor(x, 2)
>>> vals.tolist()
[[0.7, 0.4], [0.9, 0.6]]
>>> [[int(i) for i in row] for row in idx.tolist()]
[[1, 3], [0, 2]]
```

### Hello SpiralSession

```bash
python examples/hello_session.py
```

Aligns a barycenter with a hypergrad tape, prepares a Sequential module, and
finishes a roundtable epoch entirely from Python.

```python
from spiraltorch import Tensor, Hypergrad, LanguageWaveEncoder

z = Tensor(2, 4, [0.1, 0.2, 0.3, 0.4, 0.9, 0.8, 0.7, 0.6])
encoder = LanguageWaveEncoder(-1.0, 0.5)
wave = encoder.encode_z_space("SpiralTorch in Rust")

tape = Hypergrad(-1.0, 0.05, *z.shape())
tape.accumulate_pair(z, wave)
tape.apply(z)
print(z.tolist())
```

```python
from spiraltorch import SpiralSession, Tensor

session = SpiralSession(device="wgpu", curvature=-1.0, hyper_learning_rate=0.05)
densities = [Tensor(1, 2, [0.7, 0.3]), Tensor(1, 2, [0.2, 0.8])]
bary = session.barycenter(densities)
hyper = session.hypergrad(*bary.density.shape())
session.align_hypergrad(hyper, bary)
print(bary.objective, hyper.gradient())
```

```python
import spiraltorch as st
from spiraltorch.nn import Linear, MeanSquaredError, Sequential

session = st.SpiralSession(device="wgpu", curvature=-1.0)
trainer = session.trainer()
schedule = trainer.roundtable(
    rows=1,
    cols=2,
    psychoid=True,
    psychoid_log=True,
    psi=True,
    collapse=True,
    dist=st.DistConfig(node_id="demo", mode="periodic-meta", push_interval=10.0),
)
trainer.install_blackcat_moderator(threshold=0.6, participants=1)
model = Sequential([Linear(2, 2, name="layer")])
loss = MeanSquaredError()
session.prepare_module(model)

loader = (
    st.dataset.from_vec([
        (st.Tensor(1, 2, [0.0, 1.0]), st.Tensor(1, 2, [0.0, 1.0])),
        (st.Tensor(1, 2, [1.0, 0.0]), st.Tensor(1, 2, [1.0, 0.0])),
    ])
    .shuffle(0xC0FFEE)
    .batched(2)
    .prefetch(2)
)

stats = session.train_epoch(trainer, model, loss, loader, schedule)
print(f"roundtable avg loss {stats.average_loss:.6f} over {stats.batches} batches")
print(st.get_psychoid_stats())
```

### SpiralLightning harness

Python callers can skip manual trainer plumbing by instantiating the new
`SpiralLightning` helper. It prepares modules (honouring the session topos),
keeps the roundtable schedule cached, and collects epoch reports for you.

```python
import spiraltorch as st
from spiraltorch import SpiralSession
from spiraltorch.nn import Linear, MeanSquaredError

session = SpiralSession(device="wgpu", curvature=-1.0)
lightning = session.lightning(rows=1, cols=2, auto_prepare=True)
model = Linear(2, 2, name="layer")
loss = MeanSquaredError()

dataset = [
    (st.Tensor(1, 2, [0.0, 1.0]), st.Tensor(1, 2, [0.0, 1.0])),
    (st.Tensor(1, 2, [1.0, 0.0]), st.Tensor(1, 2, [1.0, 0.0])),
]

reports = lightning.fit(model, loss, [dataset])
for epoch, stats in enumerate(reports, start=1):
    print(f"epoch {epoch}: avg loss={stats.average_loss:.6f}")

# Switch back to manual preparation mid-run if you need custom tape control
lightning.set_auto_prepare(False)
session.prepare_module(model)

# Stage training plans inherit the previous configuration by default
plan = [
    {"label": "warmup", "epochs": [dataset]},
    {
        "label": "refine",
        "config": {"top_k": 4, "auto_prepare": False},
        "epochs": [dataset],
    },
# Run Optuna on a SpiralTorch training loop
def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    # ... wire lr into a SpiralSession run ...
    return final_loss

study = optuna_optimize(objective, n_trials=25, direction="minimize")

# Dispatch Ray Tune sweeps without leaving the SpiralTorch API surface
def train_spiral(lr: float):
    # ... execute a SpiralSession epoch and report Ray-compatible metrics ...
    return {"loss": 0.42}

analysis = ray_tune_run(
    trainable=lambda config: train_spiral(config["lr"]),
    config={"lr": [1e-3, 5e-4, 1e-4]},
    num_samples=5,
)

print("TorchServe bundle:", archive_path)
print("Bento artifact:", bento_ref)
print("Best Optuna trial:", study.best_trial.value)
print("Best Ray Tune result:", analysis.get_best_config(metric="loss", mode="min"))
```

## SpiralTorchRL quickstart

`spiraltorch.rl` packages the policy-gradient harness from the Rust side so
Python notebooks can lean on SpiralTorchRL without reimplementing Z-space
plumbing. Policies keep their weight updates inside hypergrad tapes and expose
the discounted-return baseline used during training.

```python
from spiraltorch import Tensor
from spiraltorch.rl import PolicyGradient

policy = PolicyGradient(state_dim=4, action_dim=2, learning_rate=0.02, discount=0.97)
policy.enable_hypergrad(curvature=-1.0, learning_rate=0.05)

state = Tensor(1, 4, [0.2, 0.4, -0.1, 0.3])
action, probs = policy.select_action(state)
policy.record_transition(state, action, reward=1.0)

report = policy.finish_episode()
print(f"reward={report.total_reward:.2f} baseline={report.mean_return:.2f} hypergrad={report.hypergrad_applied}")
print("weights", policy.weights().tolist())
```

Chrono telemetry is shared through the global hub, so recording resonance
histories on the session side automatically feeds loop signals back into the
policy geometry. Call `session.resonate_over_time(...)`/`session.timeline(...)`
from Python to keep the hub warm; the Rust learner will tighten its clamps,
adjust Λ₂₄ pressure, and publish loop gain/softening diagnostics the next time
you finish an episode with geometry enabled.

Each roundtable summary now contributes to a distributed `LoopbackEnvelope`
queue. The Python side doesn’t need to manage it directly—whenever a summary
or collapse pulse fires, the bindings push the latest SpiralK script hint,
softlogic Z-bias, and PSI total into the hub. `SpiralPolicyGradient` drains the
queue before processing resonance snapshots, blends the envelopes into a single
chrono signal, and keeps the strongest script around so the controller can
rewrite its own clamps on the next pass.

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

report = lightning.fit_plan(model, loss, plan)
print(report.best_stage_label(), report.best_epoch().average_loss)
```

The `DistConfig` connects the local roundtable to a meta layer that exchanges
`MetaSummary` snapshots with peers. `install_blackcat_moderator` spins up a
moderator runtime that scores summaries, publishes Blackcat minutes, and funnels
evidence into the embedded meta conductor—all without exposing ψ readings to the
outside world.

```python
from spiraltorch import SpiralSession, Tensor

session = SpiralSession(device="wgpu", curvature=-1.0)
seed = Tensor(1, 2, [0.4, 0.6])
generator = Tensor(1, 2, [0.1, -0.2])
direction = Tensor(1, 2, [0.05, 0.07])
kernel = Tensor(2, 2, [1.0, 0.5, -0.25, 1.25])

weights = [0.6, 0.4]
densities = [Tensor(1, 2, [0.6, 0.4]), Tensor(1, 2, [0.5, 0.5])]

trace = session.trace(seed)
trace.deform(generator, direction)
trace.via(kernel)
trace.with_barycenter_from(weights, densities)
trace.with_infinity([densities[0].clone()], [])
resonance = trace.resonate()
print(resonance.homotopy_flow().tolist())
```

Temporal telemetry is available directly from Python. Record frames with
`session.resonate_over_time(resonance, dt)` and animate the geometry through the
new helpers. Use `timeline_summary` for rolling drift/energy stats,
`timeline_harmonics` to analyse spectral drift, `loop_signal` for a ready-made
bundle (complete with SpiralK hints when `kdsl` is enabled), and `session.speak(...)`
for a ready-to-plot amplitude trace while `timeline_story` narrates the same
window:

```python
frame = session.resonate_over_time(resonance, dt=0.1)
print(frame.timestamp, frame.total_energy, frame.curvature_drift)

frames = session.timeline(timesteps=64)
summary = session.timeline_summary(timesteps=64)
harmonics = session.timeline_harmonics(timesteps=128, bins=20)
loop_signal = session.loop_signal(timesteps=128)
times, energy, drift = session.animate_resonance(timesteps=64)
wave = session.speak(timesteps=64, temperature=0.6)
story, highlights = session.timeline_story(timesteps=128, temperature=0.65)
print(session.describe())
print(st.describe_timeline(frames))
if harmonics and harmonics.dominant_energy:
    print("Energy harmonic", harmonics.dominant_energy.frequency)
if loop_signal and loop_signal.spiralk_script:
    print("SpiralK loop hint:\n", loop_signal.spiralk_script)

encoder = LanguageWaveEncoder(session.curvature(), 0.55)
wave = encoder.speak(frames)

import spiraltorch as st
from spiraltorch import TextResonator
narrator = TextResonator(session.curvature(), 0.55)
print(narrator.describe_resonance(resonance))
print(narrator.describe_timeline(frames))
print(narrator.describe_frame(frames[-1]))
audio = narrator.speak(frames)
```

Atlas projections collect those temporal statistics, maintainer diagnostics,
and loopback envelopes into one object. Grab the latest `AtlasFrame` via
`session.atlas()`, inspect its metrics/notes, and narrate it with
`session.atlas_story(...)` or `st.describe_atlas(...)`:

```python
atlas = session.atlas()
if atlas:
    print(atlas.timestamp, atlas.maintainer_status)
    for metric in atlas.metrics():
        print(metric.name, metric.value)
    for district in atlas.districts():
        print("district", district.name, district.mean, district.span)
    story = session.atlas_story(temperature=0.6)
    if story:
        print(story[0])
        print(story[1])
    print(st.describe_atlas(atlas))

route = session.atlas_route(limit=6)
print("atlas history", route.length, [frame.timestamp for frame in route.frames])
```

The `SpiralSession` maintainer surfaces clamp and density suggestions directly
from the temporal stream. Configure it via the builder or tweak thresholds at
runtime:

```python
builder.maintainer(jitter_threshold=0.25, clamp_max=2.8)
session = builder.build()

print(session.maintainer_config())
report = session.self_maintain()
print(report.spiralk_script)
if report.should_rewrite():
    session.configure_maintainer(pressure_step=0.2)
    print("Maintainer escalated:", report.diagnostic)
if report.drift_peak:
    print("Drift harmonic", report.drift_peak.frequency, report.drift_peak.magnitude)
pulse = session.collapse_pulse()
if pulse:
    print("Collapse pulse", pulse.command, pulse.step)
```

```python
from spiraltorch import SpiralSession, Tensor, TensorBiome
from spiraltorch.nn import ZSpaceProjector, LanguageWaveEncoder
from spiraltorch.sot import generate_plan

session = SpiralSession(device="wgpu", curvature=-1.0)
seed = Tensor(1, 8, [0.2] * 8)
trace = session.trace(seed, sot={"steps": 64, "radial_growth": 0.08})
plan = trace.sot_plan or generate_plan(64, radial_growth=0.08)

topos = session.topos()
encoder = LanguageWaveEncoder(session.curvature(), 0.5)
projector = ZSpaceProjector(topos, encoder)

spiral_tensor = plan.as_tensor()
canvas = projector.project_spiral(plan)
print(spiral_tensor.shape(), canvas.shape())

biome = plan.grow_biome(topos)
biome.absorb_weighted("canvas", canvas, weight=2.0)
stacked = biome.stack()
meaning = projector.reimport_biome(biome)
print("stacked", stacked.shape(), "reimported", meaning.shape())
```
