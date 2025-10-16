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
