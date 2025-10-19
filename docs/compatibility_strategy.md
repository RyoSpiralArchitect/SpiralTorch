# SpiralTorch Compatibility Strategy

SpiralTorch's Rust-first runtime already interoperates with multiple host languages and
execution backends. This document summarizes how we plan to embrace compatibility so
existing practitioners can migrate incrementally while still benefiting from SpiralTorch's
performance, memory safety, and collaborative tooling.

## Guiding Principles
- **Meet users where they are.** Re-use familiar entry points from PyTorch and TensorFlow
  so model authors can bring code, checkpoints, and habits with minimal rewrites.
- **Expose SpiralTorch advantages.** Highlight deterministic execution, fearless
  concurrency, and predictable resource usage as differentiators, even when mirroring
  external APIs.
- **Stay honest about gaps.** Document unsupported operators or features up front and
  provide mitigation strategies or contribution opportunities.

## API Surface Compatibility
- **PyTorch-inspired trainers.** Offer Rust and Python APIs that accept module, optimizer,
  and dataloader traits mirroring `torch.nn.Module`, `torch.optim.Optimizer`, and
  `torch.utils.data.DataLoader`. Maintain a migration guide showing one-to-one mappings
  for common callbacks, gradient hooks, and mixed precision toggles.
- **TensorFlow graph bridge.** Ship `tf.experimental.register_checkpoint_format("spiraltorch")`
  tooling that can emit SpiralTorch checkpoints from TensorFlow training loops. Provide a
  `tf2st` CLI wrapper that translates `tf.function` graphs into KD-SL schedules with
  automatic layout conversions and naming heuristics that survive SavedModel exports.
- **JAX function parity.** Maintain a crate-level feature (`compat-jax`) that exposes a
  `st::jax` namespace in Python, mirroring `jax.numpy` semantics for array creation,
  transformation primitives, and autodiff entry points. Use pynative dispatch and shape
  tracing to compile SpiralTorch kernels directly from JAX pytrees.
- **Tensor semantics.** Ensure eager tensor operations honor broadcasting, dtype casting,
  and gradient semantics consistent with PyTorch 2.x. Provide fuzz tests that compare
  `st_tensor` outputs against `torch` for canonical operators, and cross-check against
  `tf.experimental.numpy` and `jax.numpy` for numerics-sensitive ops.
- **Module conversion utilities.** Expand the existing ONNX exporter with a CLI that can
  ingest TorchScript, SavedModel, GGUF, and `flax.linen` checkpoints. Document the
  supported operator set and how to register custom shims.

## Data and Pipeline Interoperability
- **Dataset adapters.** Supply loaders that can consume Hugging Face datasets, TFRecord,
  and common parquet/csv formats directly from Rust and Python. Reuse Polars and
  Arrow-based pipelines where possible to avoid double buffering. Add `tf.data.Dataset`
  iterators and JAX input pipeline shims via `tfds.as_numpy` to ease dual-runtime
  experimentation.
- **Mixed runtime workflows.** Publish examples that delegate preprocessing to PyTorch
  dataloaders while running forward/backward passes in SpiralTorch. Document the
  zero-copy `Tensor::to_dlpack`/`Tensor::from_dlpack` exchange path, surface the
  Python `PyTensor.__dlpack__`/`__dlpack_device__` hooks, and show how to profile the
  boundary overhead. Ship ready-to-import helpers in the Python binding (`spiraltorch.Tensor`
  plus module-level `from_dlpack`/`to_dlpack`) so that PyTorch, TensorFlow, JAX, and NumPy
  callers can swap tensors without writing glue code. The `spiraltorch.compat`
  namespace now includes `compat.torch`, `compat.jax`, `compat.tensorflow`, and
  `compat.numpy` bridges that import the respective frameworks and call their
  DLPack shims directly, plus an `compat.auto` module that detects dlpack-ready
  objects and prints upgrade hints when something is missing.
- **Telemetry bridges.** Map SpiralTorch's observability events to tensorboard, Weights &
  Biases, JAX's `jax.profiler`, and OpenTelemetry spans so teams can keep their existing dashboards.

## Backend and Deployment Parity
- **Device coverage matrix.** Keep `docs/backend_matrix.md` updated with the minimum set
  of kernels, memory allocators, and graph transforms required for parity on CPU, CUDA,
  HIP/ROCm, WGPU, and MPS. Highlight backend-specific caveats so migration planners can
  sequence their rollout. Capture TensorFlow XLA and JAX GPU compatibility notes to inform
  shared driver/runtime expectations.
- **Inference gateways.** Provide drop-in adapters for TorchServe, BentoML, and FastAPI.
  Ensure they can host SpiralTorch modules via the same packaging and manifest metadata
  that PyTorch, TensorFlow Serving, and JAX/Flax exporters expect.
- **Edge distribution.** Pair the Rust binary runtime with embedded Python wheels so
  hybrid stacks can target constrained devices without retooling existing deployment
  scripts. Provide sample Bazel rules for TensorFlow/JAX monorepos that want to embed the
  SpiralTorch runtime inside existing build graphs.

## Validation & Tooling
- **Cross-framework testing.** Automate nightly parity tests that execute reference
  training loops in SpiralTorch, PyTorch, TensorFlow 2.x, and JAX/Flax, comparing loss
  curves, gradients, and checkpoint contents. Publish the diff reports for community review.
- **Developer ergonomics.** Maintain code generation templates for creating new
  compatibility shims. Provide lint rules that flag deviations from the documented
  migration guides, including schema validators for SavedModel signatures and JAX pytree
  structures.
- **Feedback loops.** Create GitHub issue templates for compatibility regressions and
  monthly discussions focused on interop pain points. Summaries feed back into the
  roadmap and release notes, with tags indicating PyTorch, TensorFlow, or JAX impact.

## Measuring Success
- Track the percentage of PyTorch tutorials that have an officially maintained
  SpiralTorch port.
- Measure migration time by surveying early adopters and capturing blockers.
- Monitor the number of hybrid deployments (PyTorch + SpiralTorch) reported by the
  community to ensure incremental adoption remains feasible.
