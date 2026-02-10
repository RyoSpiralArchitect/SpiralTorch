# SpiralTorch (Python) changelog

## 0.4.7

- `nn`: add training mode handling, norm-state propagation, and gradient clipping support in module training workflows.
- `spiralk`: re-export `MaxwellFingerprint`, `MeaningGate`, `SequentialZ`, `MaxwellPulse`, `MaxwellProjector`, and expectation helpers at the top level so `import spiraltorch as st` quickstarts work as documented.
- Wheels CI: add Maxwell quickstart coverage in smoke tests and remove brittle trace/live-server checks from the cross-platform `Wheels` workflow.

## 0.4.1

- Model zoo: add runnable Python baselines for `vision_conv_pool_classification.py` and `zspace_vae_reconstruction.py`.
- `nn`: expose `MellinBasis` + `ZSpaceVae` for the Coherence VAE reconstruction loop.

## 0.4.0

- DesirePipeline: surface `indices`/`probabilities`/`logit_offsets` in `step(...)` so callers can apply desire feedback during sampling.
- Model zoo: new attentionless character LLM baseline (`llm_char_wave_rnn_mixer.py`) built from `WaveRnn` + `ZSpaceMixer` + `WaveGate`.
- `nn`: expose `WaveGate`, `WaveRnn`, `ZSpaceMixer`, and `FeatureReorder2d` in the Python bindings.
- Dev: when running from a source checkout, auto-load the freshest native extension from `target/` after `cargo build -p spiraltorch-py`.
- Telemetry: add `trainer_events_to_atlas_route(...)` helpers and export them from `spiraltorch`.

## 0.3.9

- WGPU: remove unsupported WGSL `enable ...;` directives and harden compute-pipeline creation against validation panics.
- SpiralK: add `SpiralKContext.eval_with_trace(...)` and `RankPlan.rewrite_with_spiralk_explain(...)` for notebook-friendly explainability.
- Add `write_kdsl_trace_html(...)` / `write_kdsl_trace_jsonl(...)` + demo viewer.

## 0.3.8

- Fix `serve_zspace_trace(...)` live viewer (no PyO3 cross-thread `PluginQueue` panic).
- Fix HTML viewers (`write_zspace_trace_html`, live viewer) f-string escaping for JS template literals.
- Add wheel smoke coverage for trace viewers.

## 0.3.7

- Stream Z-space trace events into Python (payload + timestamp on `spiraltorch.plugin` custom events).
- Add `ZSpaceCoherenceSequencer.install_trace_recorder(...)` + `ZSpaceTraceRecorder` bindings.
- Add `write_zspace_trace_html(...)` / `serve_zspace_trace(...)` for notebook-friendly visualization (HTML + live SSE viewer).
- Add `zspace_trace_to_atlas_route(...)` to merge traces into `telemetry.AtlasRoute`.

## 0.3.6

- Route `nn.LayerNorm` forward through the tensor-layer `layer_norm` op (WGPU kernel available when the backend is enabled).
- Expose `Tensor.layer_norm_affine` / `Tensor.layer_norm_affine_add` for direct call sites in notebooks.
- Build `wgpu` with native backends enabled (Metal/Vulkan/DX12) so GPU adapter discovery works in release wheels.

## 0.3.5

- Add ops sugar (`ops.signature`, flexible `ops.register`) and allow variadic inputs for Python-registered operators.
- Add `ops.describe` plus varargs support for `ops.execute`.
- Add `plugin.record(...)` JSONL recorder plus queue helpers for notebook-friendly observability.
- Expand `nn` bindings with additional losses (HyperbolicCrossEntropy/CrossEntropy, Focal, Contrastive, Triplet) and broader `ModuleTrainer` coverage.
- Make `nn.save_json/load_json` accept state dicts and return them when called with `None` (plus bincode equivalents).
- Add `nn.save/nn.load` helpers with auto format detection and manifest generation.
- Allow `nn.Sequential.add` to accept NonLiner/Scaler/Dropout/Pool2d/ZPooling/ZConv/ZConv6DA layers (NCHW for 2d ops, NCDHW for ZConv6DA).
- Strengthen unittest smoke coverage for ops/plugin/serialization.

## 0.3.4

- Add `spiraltorch.ops` to register custom operators in Python and run them via the Rust registry.
- Upgrade `spiraltorch.plugin` with queue/listen helpers plus batch subscribe/unsubscribe.
- Add `spiraltorch.nn.save_json/load_json` (+ bincode) for module snapshots.
- Start the model zoo with Python + Rust MLP regression examples.
- Route core tensor ops (`add`, `matmul`, `row_softmax`, …) into plugin events via the new `st-tensor` observer hook.
- Expose `st_nn::ModuleTrainer` as `spiraltorch.nn.ModuleTrainer` with `RoundtableConfig`, `RoundtableSchedule`, and `EpochStats`.
