# SpiralTorch documentation

[Project entry](../README.md) | [Python package](../bindings/st-py/README.md)

Start with one working path, then explore the stranger machinery. These docs
follow the source tree; the [Python changelog](../bindings/st-py/CHANGELOG.md)
and [release tags](https://github.com/RyoSpiralArchitect/SpiralTorch/releases)
identify shipped versions. Source-only examples are not promises about an
older installed wheel.

## First Steps

- [Getting started](getting-started.md): installation, native tensors, and first models.
- [Installation and backends](reference/installation.md): source builds, CPU-only wheels, and troubleshooting.
- [Example gallery](example-gallery.md) and [model zoo](../models/README.md): runnable entry points.
- [Z-space introduction](zspace_intro.md) and [manifesto](spiraltorch_manifesto.md): the project's vocabulary and motivation.

## Language and Geometric Learning

- [HF/Z-space optimizer experiments](hf_zspace_optimizer_ablation.md): matched learning studies and their limits.
- [Geometric learning bridge](geometric_learning_bridge.md): connect mechanisms to an actual learning path.
- [Geometry adapter stack](geometry_adapter_stack.md): composing geometric interventions.
- [Training reference](reference/learning.md): language models, GoldenRetriever, RL, Rec, coherence, and Rust training examples.
- [Geometric toolkits](reference/geometry.md): topos, GNNs, desire, quantum overlays, and telemetry.
- [SpiralK and kernel heuristics](reference/kernels.md): optional tuning and execution machinery.

## Native and Resident Execution

- [Autograd contract](autograd_contract.md): differentiation and mutation boundaries.
- [Native API tour](reference/native-api.md): tensors, sessions, checkpoints, and core Python examples.
- [Resident model forward](module_resident_forward.md) and [NN training](resident_nn_training.md): explicit GPU handles and training handoff.
- [Resident classification](module_resident_classification.md), [microbatches](module_resident_microbatch.md), [clipping](module_resident_gradient_clip.md), and [momentum](module_resident_momentum.md).
- [Vision clients](resident_vision_training_clients.md) and [input pipeline](resident_vision_input.md).
- [Backend matrix](backend_matrix.md) and [matched PyTorch benchmarks](backend_pytorch_benchmarks.md).

## Python and Browser Clients

- [Python recipes](python/recipes.md): sessions, API inference, native trainers, and ecosystem bridges.
- [Python capabilities](python/capabilities.md): detailed numerical, protocol, and optional API surfaces.
- [Python builds and smoke commands](python/build-and-validation.md): maintainer validation recipes, not an automatic command batch.
- [DLPack](dlpack_interop.md) and [PyTorch migration](pytorch-migration-guide.md).
- [WASM bindings](../bindings/st-wasm/README.md), [browser demos](reference/wasm.md), and [resident WebGPU matmul](resident_webgpu_matmul.md).

## Maintainers

- [Release operations](ops/release.md#release-cadence): version, tag, installed-wheel validation, and PyPI publication.
- [Workspace crates](development/workspace_crates.md), [ecosystem roadmap](ecosystem_roadmap.md), and [project reference](reference/project.md).
- [Repository statistics](repository-stats.md): generated counts, separate from performance claims.

Keep the root and package READMEs short. Put new examples in the relevant guide
or runnable example file, and add one entry link rather than another feature
history to the README. If moving a CI-executed Python fence, retain it in the
`Run README and reference Python blocks` step in `.github/workflows/ci.yml`.
