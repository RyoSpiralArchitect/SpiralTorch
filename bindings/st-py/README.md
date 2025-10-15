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
- `TensorBiome` to cultivate open-topos rewrites and harvest guarded tensors
  that can be re-imported into Z-space.
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
- Non-commutative differential traces via `SpiralSession.trace(...)` which emit
  `SpiralDifferentialTrace` builders and `DifferentialResonance` snapshots to
  blend homotopy flows, functor derivatives, recursive barycenter energies, and
  \(\infty\)-tower projections—optionally wiring the result straight into a
  `Hypergrad` tape.
- SoT-3Dφ spiral planners (`spiraltorch.sot`) that collapse to Z-space tensors
  and are stitched directly into `SpiralSession.trace(...)` for geometry-aware
  exploration loops.
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

session = SpiralSession(device="wgpu", curvature=-1.0)
seed = Tensor(1, 8, [0.2] * 8)
trace = session.trace(seed, sot={"steps": 64, "radial_growth": 0.08})
plan = trace.sot_plan

if plan is None:
    raise RuntimeError("SoT planner was disabled")

topos = session.topos()
encoder = LanguageWaveEncoder(session.curvature(), 0.5)
projector = ZSpaceProjector(topos, encoder)

spiral_tensor = plan.as_tensor()
canvas = projector.project_spiral(plan)
print(spiral_tensor.shape(), canvas.shape())

biome = TensorBiome(topos)
biome.absorb("spiral", spiral_tensor)
biome.absorb("canvas", canvas)
meaning = projector.reimport_biome(biome)
print("reimported", meaning.shape())
```
