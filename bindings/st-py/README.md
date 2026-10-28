# SpiralTorch Python bindings

**A Rust-first learning stack, with a Python door into Z-space.**

Native tensors, autograd, neural modules, geometry, and runtime contracts live
in Rust. Python connects them to experiments and other ML libraries without
making NumPy, PyTorch, Transformers, or provider SDKs mandatory dependencies.

## Install

```bash
python -m pip install -U spiraltorch
```

Source package version: **Version 0.4.33**. Wheels target CPython 3.8+, Linux x86_64,
Windows x86_64, and macOS 14+ universal2. The default wheel includes WGPU with
CPU fallback; a compatible GPU and the appropriate execution route are still
required for GPU use.
Source metadata does not confirm publication; check the
[PyPI release history](https://pypi.org/project/spiraltorch/#history) for available
versions before using APIs introduced after your installed version.

Source version metadata does not prove publication; check the
[PyPI release history](https://pypi.org/project/spiraltorch/#history) for available
wheels. This source tree can be ahead of PyPI. The documentation links below follow
main and identify source-only paths where applicable. Check the
[package changelog](https://github.com/RyoSpiralArchitect/SpiralTorch/blob/main/bindings/st-py/CHANGELOG.md)
and your installed version before using a new API.

## First Model

```python
import spiraltorch as st
from spiraltorch.nn import Linear, Sequential

print("CPU:", st.describe_device("cpu")["backend"])
model = Sequential()
model.add(Linear(2, 2, name="head"))
print(model(st.Tensor(1, 2, [0.25, 0.75])).tolist())
```

## Native Gradients

Differentiate a squared norm with Rust-owned autograd, without NumPy or PyTorch:

```python
import spiraltorch as st

x = st.AutogradTensor.variable(st.Tensor(1, 2, [1.0, -2.0]))
x.hadamard(x).sum().backward()
assert x.grad().tolist() == [[2.0, -4.0]]
print(x.grad().tolist())
```

This uses host tensors. Continue with [learning recipes](https://github.com/RyoSpiralArchitect/SpiralTorch/blob/main/docs/python/recipes.md),
[geometric learning](https://github.com/RyoSpiralArchitect/SpiralTorch/blob/main/docs/geometric_learning_bridge.md),
or [direct Rust use](https://github.com/RyoSpiralArchitect/SpiralTorch#native-autograd-two-ways).

## Find Your Entry Point

| Surface | Purpose |
| --- | --- |
| `Tensor`, `AutogradTensor` | Native tensors and Rust-owned reverse-mode differentiation |
| `nn`, `optim`, `SpiralSession` | Modules, training, optimization, and runtime planning |
| `nn.ByteDecoderPlan`, `nn.ByteDecoderGeometryPlan` | Explicit resident byte-LM learning, geometry mixtures, and model checkpoints |
| `nn.ByteCorpusStudy` | Shared Rust/WASM paired studies, geometry controls, and request-bound pause/resume |
| `SafeTensorArchive`, `NdTensor` | Native bounded checkpoint-tensor loading without Python safetensors |
| `NativeCausalLm`, `NativeCausalLmTrainer`, `NativeCausalLmGenerator` | Supported GPT-2 CPU inference, attached-geometry learning, and resumable generation |
| `NativeGeometryRuntime`, `NativeGeometryAdamW` | Rust-owned geometry evaluation, gradients, and optimizer state |
| `ecosystem` | Explicit tensor exchange with other ML libraries |
| `ApiLLMZSpaceRuntime` | Connect hosted-model responses to Z-space runtime traces |
| `runtime_import_preflight_report` | Check optional HF/Torch/PEFT dependencies before a larger run |

- [Getting started](https://github.com/RyoSpiralArchitect/SpiralTorch/blob/main/docs/getting-started.md): tensors, models, and learning loops.
- [Python recipes](https://github.com/RyoSpiralArchitect/SpiralTorch/blob/main/docs/python/recipes.md): sessions, trainers, geometry, API inference, and ecosystem bridges.
- [Capability reference](https://github.com/RyoSpiralArchitect/SpiralTorch/blob/main/docs/python/capabilities.md): runtime contracts, numerical APIs, optional surfaces, and source-only additions.
- [Model zoo](https://github.com/RyoSpiralArchitect/SpiralTorch/blob/main/models/README.md): runnable language-model and other learning examples.
- [Resident NN training](https://github.com/RyoSpiralArchitect/SpiralTorch/blob/main/docs/resident_nn_training.md): explicit GPU-resident execution and weight handoff.
- [Geometric byte language models](https://github.com/RyoSpiralArchitect/SpiralTorch/blob/main/docs/python/byte-language-models.md): Rust-owned QKV learning, four metrics, query routing, and exact model resume.
- [Paired byte-corpus studies](https://github.com/RyoSpiralArchitect/SpiralTorch/blob/main/docs/python/byte-corpus-studies.md): the shared experiment runner and geometry-control preparation, available from 0.4.31.
- [DLPack interoperability](https://github.com/RyoSpiralArchitect/SpiralTorch/blob/main/docs/dlpack_interop.md): ownership and copy policies for tensor exchange.

## Native Language Models

The 0.4.33 source exposes `NativeCausalLm`, `NativeCausalLmTrainer` and
`NativeCausalLmGenerator` for supported GPT-2 CPU execution, geometry-only
learning, corpus evaluation and generation without Torch or NumPy. Base weights
remain frozen. The importer is not an arbitrary HF architecture loader or PEFT
executor, and this CPU path is separate from resident WGPU byte-LM training.
See the [native/HF execution boundary](https://github.com/RyoSpiralArchitect/SpiralTorch/blob/main/docs/python/native-hf-boundary.md)
for supported configurations, native learning/generation recipes and remaining
foreign-model dependencies. These newer APIs are not in the immutable 0.4.32
wheels; check the installed version and publication history above.

Supported affine, WaveGate, elliptic, Topos and fractional geometry recipes can
also execute and learn directly through `NativeGeometryRuntime`, without Torch,
Transformers or NumPy. See
[native geometry training](https://github.com/RyoSpiralArchitect/SpiralTorch/blob/main/docs/python/native-geometry-runtime.md)
for saved-state reuse, Rust-owned updates and the still-optional HF boundary.

## Optional HF Integration

Use the optional HF clients when execution should remain in Transformers/Torch.
They are explicit interoperability paths, not hidden fallbacks for unsupported
native models:

```bash
python -m pip install 'spiraltorch[hf-runtime]'
```

Optional runtime packages do not download a model or start training. See the
[HF optimizer study](https://github.com/RyoSpiralArchitect/SpiralTorch/blob/main/docs/hf_zspace_optimizer_ablation.md)
for matched experiments, supported routes, and limitations. Geometric
interventions and conventional backend speed are separate evaluation questions.

## Build, Test, and Release

Use the [wheel build and validation guide](https://github.com/RyoSpiralArchitect/SpiralTorch/blob/main/docs/python/build-and-validation.md)
for source builds, an explicit CPU-only feature set, and smoke commands.
The [release runbook](https://github.com/RyoSpiralArchitect/SpiralTorch/blob/main/docs/ops/release.md)
keeps immutable tags, signed wheels, and verified PyPI publication together.

[Full documentation](https://github.com/RyoSpiralArchitect/SpiralTorch/blob/main/docs/README.md) ·
[Z-space introduction](https://github.com/RyoSpiralArchitect/SpiralTorch/blob/main/docs/zspace_intro.md) ·
[GNU AGPL-3.0-or-later](https://github.com/RyoSpiralArchitect/SpiralTorch/blob/main/LICENSE%20.txt)
