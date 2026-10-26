# 🌀🕯️ SpiralTorch 🕯️🌀

**Come in. Leave a light on for the strange result.**

SpiralTorch is a Rust-first research ML stack with Python and browser/WASM
clients. Native tensors, autograd, learning modules, and geometric mechanisms
share a Rust core; WGPU provides explicit GPU execution paths.
NumPy, PyTorch, and Transformers are optional, not mandatory dependencies.

A tensor is enough to begin. Further in, the paths bend through **Z-space**: the project's
vocabulary for geometric coordinates, learning controls, and telemetry. The rest of the workshop can wait. 🕯️

[Manifesto](docs/spiraltorch_manifesto.md) · [Z-space](docs/zspace_intro.md) ·
[Documentation](docs/README.md) · [Model zoo](models/README.md)

[![PyPI](https://img.shields.io/pypi/v/spiraltorch.svg?label=spiraltorch)](https://pypi.org/project/spiraltorch/)

## Start With Python

```bash
python -m pip install -U spiraltorch
```

Save this as `first_spiral.py` and run `python first_spiral.py`:

```python
import spiraltorch as st

# A tiny computation: y = 1² + (-2)². Its gradient is 2x.
x = st.AutogradTensor.variable(st.Tensor(1, 2, [1.0, -2.0]))
x.hadamard(x).sum().backward()
assert x.grad().tolist() == [[2.0, -4.0]]
print(x.grad().tolist())  # [[2.0, -4.0]]
```

This uses host tensors and Rust-owned differentiation. It needs no model files,
API key, NumPy, PyTorch, or GPU. Run it outside the source checkout so Python
imports your installed package. See [getting started](docs/getting-started.md)
for a first neural layer and learning loop.

Wheels support CPython 3.8+, Linux x86_64, Windows x86_64, and macOS 14+
(universal2). They include WGPU plus CPU fallback; installing a wheel does not
guarantee that a compatible GPU is available or that every operation is resident.
See [installation and CPU-only builds](docs/reference/installation.md).

**Published wheel versus source:** this README follows the source tree.
New resident, geometry, or learning APIs may require a source build; consult the
[Python changelog](bindings/st-py/CHANGELOG.md) and the
[release tag](https://github.com/RyoSpiralArchitect/SpiralTorch/releases) before
assuming that an example is available in your installed wheel.

## What Can I Explore?

Pick a thread. Follow it as far into the machinery as you like.

| Path | What it offers | Boundary to know |
| --- | --- | --- |
| [Tensors and learning](docs/python/recipes.md) | Native autograd, neural layers, optimizers and checkpoints | Start on CPU; GPU residency is explicit |
| [Geometric experiments](docs/geometric_learning_bridge.md) | WaveGate, Topos and fractional/elliptic mechanisms with learning recipes | Evaluate each intervention with controls; geometry is not a quality guarantee |
| [Byte language models](docs/python/byte-language-models.md) | Resident training, paired studies, saved geometry and bounded generation | Requires the documented execution route; exact GPU replay has known failures |
| [Native GPT-2 (source builds)](docs/python/native-hf-boundary.md) | CPU inference and geometry-only learning through frozen base weights | Newer than the published 0.4.32 wheels; not arbitrary HF architectures or full-model fine-tuning |
| [Python and browser clients](docs/README.md#python-and-browser-clients) | Rust-owned contracts exposed through Python and WASM | Optional framework bridges and browser features have their own requirements |

The language-model direction carries **weights, learned geometry and an explicit
operating policy together**. [Model artifacts and serving gates](docs/python/byte-serving-gates.md)
connect paired evaluation to local activation and rollback, with
[bounded byte generation](docs/python/byte-generation.md). These are local
research mechanisms, not yet a durable deployment service. The guides and
[package changelog](bindings/st-py/CHANGELOG.md) identify each API's release boundary.

## Environments and Verification Limits

WGPU uses Metal, Vulkan or DX12 where supported. A wheel installation, an available
device and a fully resident workload are three different things. Consult the
[backend matrix](docs/backend_matrix.md) and
[resident training contract](docs/resident_nn_training.md) before choosing a route.
The separate MPS feature is a placeholder; it is not the WGPU/Metal path.

**Release snapshot — 2026-10-26:** 0.4.32 wheels omit the newer native GPT-2 APIs.
[PR #2297](https://github.com/RyoSpiralArchitect/SpiralTorch/pull/2297#issuecomment-6407901854)
retains a macOS WGPU concurrent-study divergence; merge, tagging, final-wheel
qualification and publication remain held. Historical
[#2289](https://github.com/RyoSpiralArchitect/SpiralTorch/issues/2289) closed at a
bounded investigation cutoff, with its cause unresolved. Closure is neither a
repair nor release qualification. The [evidence guide](docs/README.md#research-and-evidence)
links the separate investigation and artifact-qualification records.

Working APIs, language-quality results and final-wheel acceptance need different
evidence: see [matched benchmarks and learning controls](docs/README.md#research-and-evidence).

## Native Autograd, Two Ways

**Rust:** the same squared norm as the Python example, with no Python runtime.

```rust
use st_tensor::{AutogradTensor, PureResult, Tensor};

fn main() -> PureResult<()> {
    let x = AutogradTensor::variable(Tensor::from_vec(1, 2, vec![1.0, -2.0])?)?;
    x.hadamard(&x)?.sum()?.backward()?;
    let gradient = x.grad().expect("leaf gradient");
    assert_eq!(gradient.data(), &[2.0, -4.0]);
    println!("{:?}", gradient.data());
    Ok(())
}
```

From a source checkout, run the checked-in example on CPU:

```bash
cargo run --locked -p st-tensor --release --no-default-features \
  --features cpu,faer --example readme_autograd
```

Both versions are host-tensor examples, not resident GPU graphs. Continue with the
[Rust nonlinear learning example](crates/st-tensor/examples/autograd_xor.rs),
[Python learning recipes](docs/python/recipes.md), or the
[geometric learning bridge](docs/geometric_learning_bridge.md).

## What Makes It SpiralTorch?

Above, Here, Beneath. Gradients gather at roundtables; the workshop has names like desire,
Topos resonators, and BlackCat. You do not need every name to begin. Follow one into
the [geometry toolkit](docs/reference/geometry.md), [learning recipes](docs/reference/learning.md), or [kernel guide](docs/reference/kernels.md).
Change something small. Look at what moved.

The [documentation hub](docs/README.md) keeps language, vision, browser clients,
and the records of experiments that did—and did not—work within reach.

## Build and Contribute

For unreleased APIs or a custom backend, use a source build. First clone this
repository, enter its root, and install the
[build prerequisites](docs/reference/installation.md#build-from-source-cargo)
(Rust, platform build tools and `maturin>=1.9,<2`). Then choose one:

```bash
# Default Python binding build: WGPU-first, with CPU fallback.
maturin build -m bindings/st-py/Cargo.toml --release --locked

# Explicit CPU-only build, retaining the standard Python surface.
maturin build -m bindings/st-py/Cargo.toml --release --locked \
  --no-default-features --features python-default,cpu
```

[Workspace crates](docs/development/workspace_crates.md) ·
[Ecosystem roadmap](docs/ecosystem_roadmap.md) ·
[Contributor reference](docs/reference/project.md) ·
[Repository statistics](docs/repository-stats.md)

## Releases

The [release runbook](docs/ops/release.md#release-cadence) connects reviewed
milestones to immutable tags, installed-wheel checks and verified publication.
A merged PR or a tag alone is not a published wheel.

## License

[GNU AGPL-3.0-or-later](LICENSE%20.txt). See the [licensing policy](docs/licensing.md)
for attribution and commercial licensing information.
