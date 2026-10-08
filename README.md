# 🌀🕯️ SpiralTorch 🕯️🌀

**Language, geometry, and learning inside the Z-space.**

SpiralTorch is a Rust-first research ML stack with Python and browser/WASM
clients. Native tensors, autograd, learning modules, and geometric mechanisms
share a Rust core; WGPU provides explicit GPU execution paths.
NumPy, PyTorch, and Transformers are optional, not mandatory dependencies.

[Manifesto](docs/spiraltorch_manifesto.md) · [Z-space](docs/zspace_intro.md) ·
[Documentation](docs/README.md) · [Model zoo](models/README.md)

[![PyPI](https://img.shields.io/pypi/v/spiraltorch.svg?label=spiraltorch)](https://pypi.org/project/spiraltorch/)

## Start With Python

```bash
python -m pip install -U spiraltorch
```

```python
import spiraltorch as st
from spiraltorch.nn import Linear, Sequential

print("CPU:", st.describe_device("cpu")["backend"])
model = Sequential()
model.add(Linear(2, 2, name="head"))
x = st.Tensor(1, 2, [0.25, 0.75])
print(model(x).tolist())
```

Wheels support CPython 3.8+, Linux x86_64, Windows x86_64, and macOS 14+
(universal2). They include WGPU plus CPU fallback; installing a wheel does not
guarantee that a compatible GPU is available or that every operation is resident.
See [installation and CPU-only builds](docs/reference/installation.md).

**Published wheel versus source:** this README follows the source tree.
New resident, geometry, or learning APIs may require a source build; consult the
[Python changelog](bindings/st-py/CHANGELOG.md) and the
[release tag](https://github.com/RyoSpiralArchitect/SpiralTorch/releases) before
assuming that an example is available in your installed wheel.

## Choose a Path

| I want to... | Start here |
| --- | --- |
| Learn the native Python API | [Getting started](docs/getting-started.md), [Python recipes](docs/python/recipes.md) |
| Train or fine-tune language models | [Model zoo](models/README.md), [HF/Z-space optimizer experiments](docs/hf_zspace_optimizer_ablation.md) |
| Connect geometric mechanisms to learning | [Geometric learning bridge](docs/geometric_learning_bridge.md), [geometry toolkits](docs/reference/geometry.md) |
| Keep NN computation on the GPU | [Resident forward](docs/module_resident_forward.md), [resident training](docs/resident_nn_training.md) |
| Use a browser as a real client | [WASM bindings](bindings/st-wasm/README.md), [browser demos](docs/reference/wasm.md) |
| Work on vision or GNNs | [Vision clients](docs/resident_vision_training_clients.md), [GNN and geometry reference](docs/reference/geometry.md) |
| Interoperate with PyTorch | [DLPack](docs/dlpack_interop.md), [migration guide](docs/pytorch-migration-guide.md) |
| Inspect performance evidence | [Matched backend benchmarks](docs/backend_pytorch_benchmarks.md) |

## What Makes It SpiralTorch?

Z-space, roundtables, desire, topos resonators, SpiralK, GoldenRetriever, and
BlackCat remain part of the research vocabulary. They are not substitutes for
execution or learning evidence. Follow their concrete APIs and examples in the
[toolkit reference](docs/reference/geometry.md),
[training recipes](docs/reference/learning.md), and
[kernel guide](docs/reference/kernels.md).

Rust owns the shared numerical and execution contracts. Python orchestrates
experiments and ecosystem integration; WASM exposes browser-side clients.
Compare speed on matched conventional computation, and evaluate geometric
interventions separately through learning, stability, and ablation studies.

## Build and Contribute

```bash
# Default Python binding build: WGPU-first, with CPU fallback.
maturin build -m bindings/st-py/Cargo.toml --release --locked

# Explicit CPU-only build, retaining the standard Python surface.
maturin build -m bindings/st-py/Cargo.toml --release --locked \
  --no-default-features --features python-default,cpu
```

[Build prerequisites](docs/reference/installation.md) ·
[Workspace crates](docs/development/workspace_crates.md) ·
[Ecosystem roadmap](docs/ecosystem_roadmap.md) ·
[Contributor reference](docs/reference/project.md) ·
[Repository statistics](docs/repository-stats.md)

## Releases

Small, reviewed milestones should reach tags and PyPI rather than accumulating
indefinitely on main. The [release runbook](docs/ops/release.md#release-cadence)
keeps version bumps, immutable tags, installed-wheel checks, and verified
publication together. A merged PR or a tag alone is not a published wheel.

## License

[GNU AGPL-3.0-or-later](LICENSE%20.txt).
See the [licensing policy](docs/licensing.md) for attribution and commercial
licensing information.
