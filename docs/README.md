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

## How the Stack Fits Together

Rust owns the numerical rules and the state that makes a run meaningful.
Python orchestrates experiments and optional ecosystem integrations; WASM is
another client of the shared core. A client binding does not imply that every
operation or model runs on every device.

| Layer | Role | Read next |
| --- | --- | --- |
| Tensors and derivatives | `st-tensor` owns values, reverse-mode graphs and gradient rules | [Autograd contract](autograd_contract.md) |
| Models and saved state | `st-nn` owns layers, explicit resident plans and typed language-model state | [Resident training](resident_nn_training.md), [language lifecycle](language_mlops.md) |
| Geometry and control | `st-core` / `st-frac` supply mechanisms; learning paths bind their parameters and recipes | [Learning bridge](geometric_learning_bridge.md), [adapter stack](geometry_adapter_stack.md) |
| Execution and planning | WGPU provides explicit device execution; SpiralK and runtime policy govern supported routes | [Installation](reference/installation.md), [kernel guide](reference/kernels.md) |

For language experiments, choose the execution owner before following a recipe:

- **Resident byte LM:** native GPU model/study state, geometry-bearing artifacts,
  local serving admission and rollback. See [language MLOps](language_mlops.md).
- **Native GPT-2:** supported frozen decoder on CPU, with geometry-only learning
  and generation; newer than published 0.4.32. See the
  [native/HF boundary](python/native-hf-boundary.md).
- **Optional HF client:** Transformers/Torch execute the base model; supported
  geometric operators and artifact validation remain Rust-owned. See
  [portable adapters](python/geometry-artifacts.md) and
  [HF training](python/hf-geometry-training.md).

These paths have different checkpoint and runtime contracts. Saving learned
geometry preserves executable parameters and recipes; it does not make their
model formats interchangeable or establish a language-quality gain.

## Language and Geometric Learning

- [Language MLOps](language_mlops.md): train, retain geometry, recover, select saved models and measure serving changes.
- [HF/Z-space optimizer experiments](hf_zspace_optimizer_ablation.md): matched learning studies and their limits.
- [Geometric learning bridge](geometric_learning_bridge.md): connect mechanisms to an actual learning path.
- [Durable byte-LM runs](python/byte-study-runs.md): sealed geometric checkpoints and explicit recovery through the Rust owner.
- [Geometry adapter stack](geometry_adapter_stack.md): composing geometric interventions.
- [HF geometry Trainer](python/hf-geometry-training.md): train geometric adapters through HF clipping, checkpoints and resume.
- [Torch-free GPT-2 geometry learning](python/native-causal-lm.md): Rust-owned checkpoint import, inference and language-loss gradients through a frozen decoder; CPU, source-only after the fixed 0.4.32 candidate.
- [Native weights](python/native-weights.md) and [geometry runtime](python/native-geometry-runtime.md): owning tensor views and standalone geometric updates without an optional ML framework.
- [Native geometry AdamW](python/native-geometry-adamw.md): shared Rust adaptive updates, clipping and exact optimizer/artifact resume for geometry and language-loss learning; source-only after 0.4.32.
- [Document-safe HF geometry data](python/hf-geometry-data.md): Rust-owned contexts, spawned workers and an offline pretrained-model example.
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

## Research and Evidence

Choose the evidence that matches the question:

- **Does this conventional computation run faster?** Read the
  [matched backend benchmarks](backend_pytorch_benchmarks.md), including shapes,
  transfers and measured environments.
- **Does a geometric intervention help learning?** Start with the
  [geometric learning bridge](geometric_learning_bridge.md) and
  [HF/Z-space ablations](hf_zspace_optimizer_ablation.md). A working forward or
  gradient is not a demonstrated language-quality improvement.
- **Can this run be continued or selected for local serving?** Follow the
  [study records and recovery contract](python/byte-study-runs.md),
  [model artifacts and serving gates](python/byte-serving-gates.md), and
  [language MLOps design](language_mlops.md).
- **Is this exact wheel qualified for publication?** Consult the
  [bounded qualification policy](ops/byte-release-qualification.md) and
  [release receipts](ops/release.md). Source checks and later passing runs do
  not explain earlier numerical failures.

Two useful worked evidence records:

- [Fixed book controls](../benchmarks/results/2026-10-18-hf-book-controls/README.md):
  the declared WaveGate insertion underperformed the ordinary channel-affine
  control in all three seeds, despite working learning and exact reconstructed
  inference. No configuration was promoted.
- [Native language integration](../benchmarks/results/2026-10-19-native-language-integration/README.md):
  installed development-wheel checks, numerical comparison and fixed-policy
  continuation. The real-input probe used a previously trained-on chapter;
  private inputs limit independent replay. It is not a publication receipt,
  held-out quality result or repair of an earlier GPU mismatch.

As of 2026-10-26, historical
[#2289](https://github.com/RyoSpiralArchitect/SpiralTorch/issues/2289) is closed
because its bounded investigation ended; the cause remains unresolved.
[PR #2297's retained first failure](https://github.com/RyoSpiralArchitect/SpiralTorch/pull/2297#issuecomment-6407901854)
records a new macOS WGPU concurrent-study divergence and an owner-held stop on
merge, tagging, final-wheel qualification and publication. These are separate
records, not interchangeable evidence of a fix. Follow their current decisions
before release work; this index grants no qualification or rerun approval.

The [manifesto](spiraltorch_manifesto.md) describes the research ambition;
[Z-space](zspace_intro.md) introduces its vocabulary. Read design documents,
experimental scripts and historical reports alongside their stated contracts
and limitations. They are part of the research record, not a promise that every
path is available in a published package.

## Maintainers

- [Release operations](ops/release.md#release-cadence): version, tag, installed-wheel validation, and PyPI publication.
- [Workspace crates](development/workspace_crates.md), [ecosystem roadmap](ecosystem_roadmap.md), and [project reference](reference/project.md).
- [Repository statistics](repository-stats.md): generated counts, separate from performance claims.

Keep the root and package READMEs short. Put new examples in the relevant guide
or runnable example file, and add one entry link rather than another feature
history to the README. If moving a CI-executed Python fence, retain it in the
`Run README and reference Python blocks` step in `.github/workflows/ci.yml`.
