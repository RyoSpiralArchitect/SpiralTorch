# SpiralTorch Python bindings

This package exposes a thin, dependency-light bridge to SpiralTorch's
Rust-first training stack. The wheel ships the same Z-space tensors,
hypergrad tapes, and unified rank planners that power the Rust API without
making NumPy, PyTorch, Transformers, or Hugging Face tooling mandatory
dependencies.

## Install

```bash
pip install -U spiraltorch
```

The published wheel is WGPU-first with CPU fallback. In plain Python terms,
start with four handles:

- `spiraltorch.Tensor` for dependency-light native tensors.
- `spiraltorch.nn` for modules, losses, trainers, LoRA adapters, and checked
  checkpoint handoff.
- `spiraltorch.ecosystem` when a PyTorch/JAX/CuPy/TensorFlow tensor needs to
  cross the Z-space membrane.
- `spiraltorch.ApiLLMZSpaceRuntime` when hosted/API-model LLM responses should
  become Z-space partial traces without requiring the OpenAI SDK or any other
  hosted-model package at install time; if `openai` is installed, use
  `runtime.call_openai_responses(...)` or `spiraltorch.make_openai_chat_invoke(...)`;
  if `anthropic` is installed, use `runtime.call_anthropic_messages(...)` or
  `spiraltorch.make_anthropic_messages_invoke(...)`. Provider keys are read
  from the environment by their SDKs. Runtime traces can be persisted with
  `runtime.write_jsonl(...)`, batched with `runtime.run_prompts(...)`, replayed
  across providers with `spiraltorch.run_api_llm_prompt_suite_matrix(...)`, and summarized with
  `spiraltorch.summarize_api_llm_trace_events(...)`, or compared with
  `spiraltorch.compare_api_llm_trace_runs(...)`.
- `spiraltorch.runtime_import_preflight_report(...)` when a Transformers,
  Torch, PEFT, or dataset dependency contract should be recorded before a
  heavier fine-tune run.

```bash
python - <<'PY'
import spiraltorch as st
from spiraltorch.nn import Linear, Sequential

print("runtime:", st.describe_device("cpu")["backend"])

model = Sequential()
model.add(Linear(2, 2, name="head"))
print("forward:", model(st.Tensor(1, 2, [0.25, 0.75])).tolist())

head = Linear(2, 2, name="head")
native = dict(head.state_dict())
external = {
    "lm_head.weight": native["head::weight"],
    "lm_head.bias": native["head::bias"],
}
report = head.state_dict_compatibility_with_key_map(
    external,
    {"lm_head.weight": "head::weight", "lm_head.bias": "head::bias"},
)
print("checkpoint compatible:", report["compatible"])

runtime = st.runtime_import_preflight_report(
    runtime_import_presets=["hf-runtime"],
    required_runtime_import_presets=["hf-runtime"],
    runtime_device_backends=["wgpu"],
)
print("HF runtime ready:", runtime["runtime_import_preflight_passed"])
print("WGPU status:", runtime["runtime_device_report_statuses"])
PY
```

## What's included

- `Tensor`, `ComplexTensor`, and `OpenTopos` for dependency-free
  geometry experiments.
- Native neural layers via `spiraltorch.nn`—`Linear`, `Embedding`,
  `Sequential`, losses, `ModuleTrainer`, `LoraLinear`, and `ZSpaceProjector`.
- Checked checkpoint handoff helpers for exact or key-mapped `state_dict`
  reports, subset loads, overlap resize/projection preflight, and HF-style
  checkpoint presets.
- `LanguageWaveEncoder` + `Hypergrad` so Python callers can stream Z-space
  text, accumulate gradients, and project back into the Poincaré ball.
- `TensorBiome` to cultivate open-topos rewrites, weight shoots, stack the
  harvest, and guard tensors that can be re-imported into Z-space.
- Unified planning helpers (`plan`, `plan_topk`, `describe_device`,
  `probe_gpu_path`) that
  reuse the same heuristics as the Rust executors.
- ROCm probing (`hip_probe`) so Python callers can reflect the stubbed
  device hints shared with the Rust runtime.
- Z-space barycentre solver (`z_space_barycenter`) to mix colour-field
  priors and chart couplings directly from Python.
- Loss-monotone barycenter intermediates (`BarycenterIntermediate`) that plug
  into `Hypergrad.accumulate_barycenter_path` so tapes converge along the
  same Z-space corridor as the solver.
- Lightweight runtime orchestration via `SpiralSession` so callers can record
  backend intent, inspect device preflight evidence, and reuse the same
  `RankPlan` helpers as the Rust executors.
- Hosted/API-model LLM runtime bridge via `ApiLLMZSpaceRuntime` so an
  OpenAI-compatible response mapping, SDK response object, or arbitrary API
  callable can be converted into Z-space metrics, usage/latency telemetry, and
  posterior confidence without making hosted SDKs mandatory dependencies. The
  optional OpenAI and Anthropic adapters are lazy: they import provider SDKs only
  when called, then feed Responses, chat-completion, or Messages API results into
  the same trace path. API LLM trace JSONL helpers mirror the trainer/transformers
  trace workflow so hosted-model runs can be compared without re-running the API
  call; use `run_api_llm_prompt_suite(...)` for a multi-prompt bipolar/Z-space
  smoke, or `run_api_llm_prompt_suite_matrix(...)` to replay the same prompts
  across OpenAI, Anthropic, gateway, or local callables. Pass
  `request_kwargs={"route": {...}}` when each provider needs different request
  controls, such as OpenAI output-token caps versus Claude adaptive-thinking
  `output_config.effort`. For Claude 5/Opus 4.8 adaptive-thinking routes, size
  `max_tokens` for thinking plus visible output before interpreting
  `completion_rate` or `empty_text_rate`, then
  `compare_api_llm_trace_runs(...)` to pick candidates by route score, latency,
  token use, confidence, runtime readiness, refusal rate, empty-text rate, and
  attached WASM context signals such as browser-side loss and WebGPU readiness.
  Comparison rows also expose `quality_score`, `efficiency_score`, normalized
  `latency_cost` / `token_cost`, and `health_penalty`; use `near_best` to inspect
  routes that are close enough that the tradeoff matters more than the rank.
  Trace summaries also include deterministic text-quality guards:
  `prompt_coverage`, `prompt_echo_rate`, `response_signal_rate`,
  `repetition_rate`, and `text_quality_score`. For route selection, comparison
  payloads include `selection_profiles` for `balanced`, `quality`, `grounded`,
  `efficiency`, and `latency` routing. Use
  `compare_api_llm_matrix_reports(...)` to compare repeated live provider
  `report.json` sweeps and inspect profile-winner stability across runs; the
  `api_llm_live_provider_matrix_sweep.py` example can run several token-budget
  pairs and produce that comparison in one command. Pass `--resume-existing`
  when expanding a sweep so completed budget pairs are reused instead of
  re-calling provider APIs. Browser-side WASM learning reports can also be
  loaded with `load_wasm_report(...)`, summarized with `summarize_wasm_report(...)`,
  converted into reusable context via `api_llm_wasm_context_partials(...)`, and
  passed as `context_partials=` to `ApiLLMZSpaceRuntime` or
  `run_api_llm_prompt_suite(...)`; see
  `examples/api_llm_wasm_context_runtime.py` for a keyless end-to-end bridge, or
  `examples/openai_api_llm_wasm_context_runtime.py` for a live OpenAI Responses
  smoke that persists the selected WASM context handoff and writes trace JSONL, or
  pass `--wasm-report report.json` to `examples/api_llm_live_provider_matrix.py`
  and `examples/api_llm_live_provider_matrix_sweep.py` when live OpenAI/Claude
  route comparisons should carry the same browser-side learning signal. Use
  `collect_wasm_report_paths(...)` or `build_wasm_report_context(...)` directly,
  or pass `--wasm-report-glob`, `--wasm-report-dir`, `--wasm-report-recursive`,
  and `--wasm-max-reports` to collect repeated browser runs, compare them by
  loss, and feed only the strongest reports into the provider matrix. Persist
  the selected handoff for later FT/notebook/API runs with
  `write_wasm_report_context_artifact(...)`, then reload its partials with
  `load_wasm_report_context_artifact(...)`.
- Language desire controls via `st.nn.DesirePipeline`, `DesireTrainerBridge`,
  `DesireRoundtableBridge`, and downstream hook adapters so notebooks can
  inspect phase/temperature/entropy offsets without making the symbolic kernel
  internals part of the stable public surface.
- Native trainer harnesses via `spiraltorch.nn.ModuleTrainer` and
  `RoundtableConfig` for quick notebook experiments without leaving the Rust
  training loop.
- Event observability via `spiraltorch.plugin`—subscribe, listen queues, or
  record JSONL streams with `plugin.record(...)`.
- Python plugin registry via `spiraltorch.plugin.register_python_plugin(...)`
  (and `plugin.load_entrypoints(...)` / `plugin.load_path(...)` / `plugin.reload_path(...)` / `plugin.watch_path(...)` for discovery + hot reload).
  The `spiral-plugin` CLI can introspect plugin graphs (`list`, `graph`, `dot`, `explain`, `validate`).
- Custom operator registration via `spiraltorch.ops` with flexible `register`
  calls, `ops.signature(...)`, and a human-friendly `ops.describe(...)`.
- Built-in module + state-dict serialization helpers (`spiraltorch.nn.save_json` /
  `spiraltorch.nn.load_json`, plus bincode equivalents) for `Linear`,
  `Sequential`, and core layer modules; pass `None` to `load_json` to get a
  state dict back. The higher-level `spiraltorch.nn.save` / `load` helpers
  auto-detect JSON vs bincode and emit a compact manifest alongside weights.
- Expanded loss surface: `MeanSquaredError`, `HyperbolicCrossEntropy`
  (`CrossEntropy` alias), `FocalLoss`, `ContrastiveLoss`, and `TripletLoss`.
- Direct access to the core A/B/C roundtable trainer via
  `spiraltorch.nn.ModuleTrainer` (`RoundtableConfig`, `RoundtableSchedule`,
  `EpochStats`) including `prepare/step/zero`, optional realgrad toggles, and
  curvature-scheduler controls plus spectral/coherence bridge toggles (with
  tunable `SpectralLearningRatePolicy`) for
  long-running adaptive training loops.
- Attentionless sequence layers via `spiraltorch.nn`—`WaveRnn`, `WaveGate`,
  `ZSpaceMixer`, and `FeatureReorder2d` for Conv/RNN-style language baselines.
- Coherence VAE primitives via `spiraltorch.nn`—`MellinBasis`, `ZSpaceVae`,
  and `ZSpaceTextVae` for encoder+decoder reconstruction training loops,
  batch metrics, SGD/Adam/RMSProp optimizer state, and Atlas-ready telemetry.
- Streaming dataset helpers via `spiraltorch.dataset`—build a
  shuffle/batch/prefetch pipeline entirely in Rust using the native
  `DataLoader`.
- Trace and artifact utilities via `spiraltorch.zspace_trace`,
  `spiraltorch.trainer_trace`, and Atlas adapters so JSONL telemetry can be
  loaded, summarized, compared, and rendered from Python.
- Top-level runtime import helpers so FT notebooks can expand HF/PEFT presets,
  emit install hints, probe `torch` / `transformers` / `tokenizers` /
  `datasets` / `accelerate` / `safetensors` evidence, write JSON reports, and
  gate optional dependency contracts without pulling those packages into
  SpiralTorch's required dependency set.
- SoT-3Dφ spiral planners (`spiraltorch.sot`) that collapse to Z-space tensors,
  grow full TensorBiomes via `SoT3DPlan.grow_biome(...)`, and feed
  geometry-aware experiments or trace artifacts without requiring a Python
  session-side trace builder.
- Z-space projector bindings (`spiraltorch.nn.ZSpaceProjector`) so spiral
  trajectories can be rendered onto the canvas or reused inside sequential
  transformer stacks.
- Atlas adapters (`spiraltorch.zspace_atlas`) to convert JSONL traces + trainer
  events into `telemetry.AtlasRoute` summaries.
- Deployment and optimisation bridges via `spiraltorch.integrations`: archive
  TorchServe models, persist BentoML runners, explore hyperparameters with
  Optuna or Ray Tune, and emit deployment JSON artefacts (ONNX/TFLite planned) - all behind ergonomic
  Python call sites.
  Use the `spiral-export` CLI to generate export artefacts.
- Ecosystem helpers via `spiraltorch.ecosystem` to shuttle tensors between
  PyTorch, JAX, CuPy, and TensorFlow through zero-copy DLPack bridges.
- Reinforcement learning harness via `spiraltorch.spiral_rl`—SpiralTorchRL keeps
  policy gradients inside Z-space tensors, exposes hypergrad-enabled updates,
  and streams geometric rewards without leaving Rust.
- Recommendation toolkit via `spiraltorch.rec`—SpiralTorchRec factors user/item
  lattices under open-cartesian topos guards so embeddings stay psychoid-safe
  while training entirely in Rust.
- Model-zoo orchestration via `spiraltorch.model_zoo`—discover recipes, filter
  by task/family, resolve script paths, rank recommendations with
  `suggest_models(...)`/`recommend_model(...)`, and run models with a stable
  Python API or the `spiral-model-zoo` CLI.
- Stream telemetry interop via `vision.ChronoSnapshot`,
  `vision.ZSpaceStreamFrame`, `vision.StreamedVolume`, and
  `vision.ZSpaceStreamFrameAggregator` so Python can attach chrono summaries,
  aggregate live frame streams, and ingest temporal updates without dropping to
  Rust glue code.
- Online stream-loop helpers `vision.vision_online_step(...)` and
  `vision.stream_vision_training(...)` to wire frame streams into
  `SpiralTorchVision` + `ZSpaceTrainer` loops directly from Python.

## Tokenizerless FT diagnostics

The examples in `examples/byte_lm_*.py` provide a bounded byte-LM fine-tune
diagnostic surface for local HF/PyTorch-style checkpoints without making Torch,
safetensors, or Transformers hard dependencies of the binding. Start with
`examples/byte_lm_profile_smoke.py --hf-state-dict <path> --key-preset auto` to
run checkpoint preflight, native LoRA/source/profile comparisons, promotion
manifests, and dry-run continuation plans before scaling into heavier training
runs. `spiraltorch.nn.Linear`, `Embedding`, and `LoraLinear` expose checked
exact and key-mapped load reports, while `ZSpaceProjector` can be inserted when
you want a bounded residual projection instead of silently trusting an imported
state dict. For a practical Transformers fine-tune readiness smoke, add
`--ft-readiness-preset hf-wgpu-balanced`; this turns on checkpoint audit,
Transformers trace, produced-manifest validation, same-process
`transformers`/`torch`/`tokenizers` co-import evidence, the
Transformers/trainer runtime bridge gate, `describe_device("wgpu")` runtime
readiness evidence, and WGPU run-summary/promotion gates.
The recipe expands to `--runtime-contract-preset hf-runtime --wgpu-readiness-preset balanced`;
use `hf-wgpu-observed` to only require WGPU metrics/report presence or
`hf-wgpu-strict` for a high-readiness gate that also requires WGPU runtime-ready
evidence. Lower-level runtime/WGPU presets, explicit
`--runtime-device-report-backend`, and explicit run, promotion, or manifest
WGPU thresholds override the recipe defaults. Add
`--transformers-audit` when a local Transformers
config/tokenizer should be co-imported into the same JSONL evidence without
making Transformers mandatory.
For pre-FT inference evidence, `examples/byte_lm_transformers_trace.py` loads a
local Transformers model, records prompt-level next-token top-k logits and
hidden-state summaries, co-imports config/tokenizer/model runtime metadata, and
can attach `--zspace-project` projection metrics. Add
`--runtime-contract-preset hf-runtime` to require same-process
`transformers`/`torch`/`tokenizers` co-import evidence without going through the
profile ladder, or use
`checkpoint_preflight.py --transformers-runtime-contract-preset hf-runtime` for
the matching checkpoint audit shortcut. Add
`--require-runtime-metadata-match` when comparing traces to fail fast on
config/tokenizer/model swaps before reading prompt-level drift.

For notebook or CI preflight without a training script, either run the CLI:

```bash
spiral-runtime-preflight \
  --preset hf-runtime \
  --require \
  --runtime-device-backend wgpu \
  --json-out ft-runtime.json
```

or call the same contract from Python:

```python
import spiraltorch as st

report = st.runtime_import_preflight_report(
    runtime_import_presets=["hf-runtime"],
    required_runtime_import_presets=["hf-runtime"],
    runtime_device_backends=["wgpu"],
)
st.write_runtime_import_preflight_report(report, "ft-runtime.json")
```

## Building wheels

The binding mirrors the Rust feature flags. Pick the backend(s) you need
and maturin will bake the appropriate artefact:

```bash
pip install maturin==1.*

# macOS wheel targets (Apple Silicon requires >= 11.0):
# - macOS 11+ (broad compatibility): export MACOSX_DEPLOYMENT_TARGET=11.0
# - macOS 14+ (separate wheel build): export MACOSX_DEPLOYMENT_TARGET=14.0
export MACOSX_DEPLOYMENT_TARGET=11.0

# Default binding build (WGPU-first; CPU fallback remains available)
maturin build -m bindings/st-py/Cargo.toml --release --locked

# Release-equivalent (matches PyPI wheels: default WGPU route + logic/kdsl)
maturin build -m bindings/st-py/Cargo.toml --release --locked --features logic,kdsl

# CPU-only (drop the default WGPU route but keep the standard Python surface)
maturin build -m bindings/st-py/Cargo.toml --release --locked --no-default-features --features python-default

# Add CUDA or HIP alongside the default WGPU-first wheel
maturin build -m bindings/st-py/Cargo.toml --release --locked --features cuda,logic,kdsl
maturin build -m bindings/st-py/Cargo.toml --release --locked --features hip,logic,kdsl

# Backend-specific builds without the default WGPU route
maturin build -m bindings/st-py/Cargo.toml --release --locked --no-default-features --features python-default,cuda,logic,kdsl
maturin build -m bindings/st-py/Cargo.toml --release --locked --no-default-features --features python-default,hip,logic,kdsl

# Install the wheel you just built
pip install --force-reinstall --no-cache-dir target/wheels/spiraltorch-*.whl
```

## Smoke tests (no pytest required)

```bash
PYTHONNOUSERSITE=1 python3 -s -m unittest bindings/st-py/tests/test_unittest_smoke.py
```

After installing a wheel, optional Hugging Face/FT dependencies can be checked
without launching a training job:

```bash
spiral-runtime-preflight \
  --preset hf-finetune \
  --require \
  --runtime-device-backend wgpu \
  --json-out ft-runtime.json
spiral-runtime-preflight --preset hf-peft --require --json
```

## Minimal usage

### Model Zoo discovery + launch

```python
import spiraltorch as st

entries = st.model_zoo.list_models(task="classification")
print("classification recipes:", [entry.key for entry in entries[:5]])

suggested = st.model_zoo.suggest_models(
    "llm_char",
    task="language-modeling",
    prefer_tags=["coherence"],
)
print("suggested:", [entry.key for entry in suggested[:3]])

zspace_stream = st.model_zoo.suggest_models(
    focus="zspace_stream",
    available_only=True,
    limit=3,
)
print("zspace stream track:", [entry.key for entry in zspace_stream])

cmd = st.model_zoo.build_model_command("mlp_regression", "--help")
print("command:", " ".join(cmd))
```

```bash
spiral-model-zoo focuses
spiral-model-zoo list --task language-modeling
spiral-model-zoo suggest llm_char --task language-modeling --prefer-tag coherence
spiral-model-zoo suggest --focus zspace_stream --available-only
spiral-model-zoo run zspace_stream_online_vision -- --steps 16 --flush-every 2
spiral-model-zoo run zspace_stream_frame_aggregator -- --steps 12 --native-frames
spiral-model-zoo run mlp_regression -- --help
```

### Rank-K execution

```python
import spiraltorch as st

plan = st.plan_topk(rows=2, cols=4, k=2, backend="auto")
print(plan.kind, plan.effective_backend)
print("tile/workgroup:", plan.tile, plan.workgroup)
print(plan.to_unison_script().splitlines()[0])
print(plan.fft_spiralk_hint().splitlines()[0])
```

### Hello SpiralSession

```bash
python bindings/st-py/examples/hello_session.py
```

Aligns a barycenter with a hypergrad tape, prepares a Sequential module, and
finishes a roundtable epoch entirely from Python.

```python
from spiraltorch import Tensor, Hypergrad, LanguageWaveEncoder

encoder = LanguageWaveEncoder(-1.0, 0.5)
wave = encoder.encode_z_space("SpiralTorch in Rust")
rows, cols = wave.shape()
z = Tensor(rows, cols, [0.0] * (rows * cols))

tape = Hypergrad(-1.0, 0.05, *z.shape())
tape.accumulate_pair(z, wave)
tape.apply(z)
print(z.tolist())
```

### Canvas projector quickstart

```bash
python bindings/st-py/examples/canvas_projector_quickstart.py
```

Renders a radial energy field into an RGBA surface (written as
`spiraltorch_canvas.ppm`) and prints the row-wise FFT power spectrum tensor
shape.

### Atlas telemetry quickstart

```bash
python bindings/st-py/examples/atlas_quickstart.py
```

Builds an `AtlasFrame` from a Python dict and prints the district aggregation.

```python
import spiraltorch as st

route = st.telemetry.AtlasRoute()
route.push_bounded(
    st.telemetry.AtlasFrame.from_metrics({"psi.total": 1.0}, timestamp=0.0),
    bound=32,
)
route.push_bounded(
    st.telemetry.AtlasFrame.from_metrics({"psi.total": 1.5}, timestamp=1.0),
    bound=32,
)
print("summary:", route.summary()["frames"], "frames")
print("psi perspective:", route.perspective_for("Concourse", focus_prefixes=["psi."])["guidance"])
```

### Z-space + optim quickstart

```bash
python bindings/st-py/examples/zspace_optim_quickstart.py
```

```python
import spiraltorch as st

opt = st.optim.Amegagrad((1, 3), curvature=-0.9, hyper_learning_rate=0.03, real_learning_rate=0.02)
weights = st.Tensor(1, 3, [0.2, -0.1, 0.05])

opt.accumulate_wave(st.Tensor(1, 3, [0.4, -0.6, 0.2]))
opt.step(weights)  # tunes rates via DesireGradientControl + applies both tapes
print("weights:", weights.tolist())

trainer = st.ZSpaceTrainer(z_dim=4)
loss = trainer.step({"speed": 0.2, "memory": 0.1, "stability": 0.9, "gradient": opt.real.gradient()})
print("z:", trainer.state, "loss:", loss)
```

### AmegagradSession quickstart

```bash
python bindings/st-py/examples/amegagrad_session_quickstart.py
```

### Canvas → Atlas → Session quickstart

```bash
python bindings/st-py/examples/canvas_atlas_session_quickstart.py
```

Runs a small closed-loop demo:

- `AmegagradSession` updates weights
- `CanvasProjector` renders + emits a loopback patch
- `CanvasProjector.emit_atlas_frame(...)` streams metrics into `AtlasRoute`

### Text → optim → zspace quickstart

```bash
python bindings/st-py/examples/text_optim_zspace_quickstart.py
```

```python
import spiraltorch as st

encoder = st.LanguageWaveEncoder(-1.0, 0.5)
rows, cols = encoder.encode_z_space("SpiralTorch").shape()
opt = st.optim.Amegagrad((rows, cols), curvature=encoder.curvature())
weights = st.Tensor(rows, cols, [0.0] * (rows * cols))

opt.absorb_text(encoder, "Z-space wants to be steerable.")  # pads/truncates to (rows, cols)
opt.step(weights)
print("weights (head):", [v for row in weights.tolist() for v in row][:5])

trainer = st.ZSpaceTrainer(z_dim=4)
trainer.step({"speed": 0.2, "memory": 0.1, "stability": 0.9, "gradient": opt.real.gradient()})
```

### Z-space inference quickstart

```bash
python bindings/st-py/examples/zspace_inference_quickstart.py
```

```python
import spiraltorch as st

trainer = st.ZSpaceTrainer(z_dim=4)
loss = trainer.step_partial({"speed": 0.2, "memory": 0.1, "stability": 0.9, "gradient": [0.1, 0.0, 0.0, 0.0]})
print("z:", trainer.state, "loss:", loss)
print("last inference:", trainer.last_inference.residual, trainer.last_inference.confidence)
```

### SoT-3Dφ → TensorBiome quickstart

```bash
python bindings/st-py/examples/sot_biome_quickstart.py
```

### SpiralK KDSl plan rewrite quickstart

```bash
python bindings/st-py/examples/spiralk_plan_rewrite_quickstart.py
```

### Maxwell-coded envelopes → SpiralK hints quickstart

```bash
python bindings/st-py/examples/maxwell_spiralk_bridge_quickstart.py
```

### Streaming Z-space trainer quickstart

```bash
python bindings/st-py/examples/zspace_stream_training_quickstart.py
```

### Ecosystem bridges

SpiralTorch tensors can flow into PyTorch or JAX without copies thanks to the
`spiraltorch.ecosystem` helpers. CuPy round-trips also accept optional CUDA
streams so you can coordinate asynchronous pipelines, and the helpers can
resolve friendly stream aliases on demand:

```python
import spiraltorch as st
from spiraltorch.ecosystem import (
    tensor_to_cupy,
    tensor_to_jax,
    tensor_to_tensorflow,
    tensor_to_torch,
    cupy_to_tensor,
    jax_to_tensor,
    tensorflow_to_tensor,
    torch_to_tensor,
)

spiral = st.Tensor(2, 2, [1.0, 2.0, 3.0, 4.0])

try:
    import torch

    torch_tensor = tensor_to_torch(spiral, dtype=torch.float32)
    roundtrip = torch_to_tensor(torch_tensor)
    print("torch:", roundtrip.shape())
except Exception as exc:
    print("torch bridge skipped:", type(exc).__name__)

try:
    jax_array = tensor_to_jax(spiral)
    spiral_again = jax_to_tensor(jax_array)
    print("jax:", spiral_again.shape())
except Exception as exc:
    print("jax bridge skipped:", type(exc).__name__)

try:
    # stream can be an explicit cupy.cuda.Stream, or a lazy alias such as
    # "current" (resolve the active stream) or "null" (select the default stream).
    cupy_array = tensor_to_cupy(spiral, stream="current")
    spiral_from_cupy = cupy_to_tensor(cupy_array, stream="current")
    print("cupy:", spiral_from_cupy.shape())
except Exception as exc:
    print("cupy bridge skipped:", type(exc).__name__)

try:
    tf_tensor = tensor_to_tensorflow(spiral)
    spiral_from_tf = tensorflow_to_tensor(tf_tensor)
    print("tensorflow:", spiral_from_tf.shape())
except Exception as exc:
    print("tensorflow bridge skipped:", type(exc).__name__)
```

```python
import spiraltorch as st
from spiraltorch.nn import Linear, MeanSquaredError, Sequential

trainer = st.nn.ModuleTrainer(
    backend="cpu",
    curvature=-1.0,
    hyper_learning_rate=1e-2,
    fallback_learning_rate=1e-2,
)
schedule = trainer.roundtable(
    2,
    1,
    st.nn.RoundtableConfig(top_k=1, mid_k=1, bottom_k=1, here_tolerance=1e-5),
)
model = Sequential()
model.add(Linear(2, 1, name="layer"))
model.attach_hypergrad(curvature=-1.0, learning_rate=1e-2)

loss = MeanSquaredError()
x = st.Tensor.rand(2, 2, seed=3)
y = st.Tensor.rand(2, 1, seed=4)
stats = trainer.train_epoch(model, loss, [(x, y)], schedule)

print(f"roundtable avg loss {stats.average_loss:.6f} over {stats.batches} batch")
```

### Desire pipeline orchestration

```python
import spiraltorch as st

pipeline = st.nn.DesirePipeline(vocab_size=2, concepts=2)
step = pipeline.step([1.2, -0.4], previous_token=0, concept=[0.6, 0.4])
print("phase", step["phase"], "entropy", step["entropy"])

adapter = st.build_desire_adapter_from_downstream_hook(
    {
        "geometry_bias_coherence": {"score": 0.7},
        "top_probability": 0.8,
    }
)
pipeline.ingest_geometry_bias(adapter["geometry_bias_signal"], source="zspace")
print("geometry bias:", pipeline.geometry_bias_metrics())
```

### Native trainer harness

Python callers can keep the training loop small while still using the Rust
roundtable trainer. For heavier HPO/serving flows, use this loop as the inner
objective and wrap it with your Optuna/Ray/BentoML/TorchServe tool of choice.

```python
import spiraltorch as st
from spiraltorch.nn import Linear, MeanSquaredError, ModuleTrainer, RoundtableConfig, Sequential

dataset = [
    (st.Tensor(1, 2, [0.0, 1.0]), st.Tensor(1, 1, [1.0])),
    (st.Tensor(1, 2, [1.0, 0.0]), st.Tensor(1, 1, [0.0])),
]

for label, lr in [("warmup", 1e-2), ("refine", 5e-3)]:
    trainer = ModuleTrainer(
        backend="cpu",
        curvature=-1.0,
        hyper_learning_rate=lr,
        fallback_learning_rate=lr,
    )
    schedule = trainer.roundtable(
        1,
        1,
        RoundtableConfig(top_k=1, mid_k=1, bottom_k=1, here_tolerance=1e-5),
    )
    model = Sequential()
    model.add(Linear(2, 1, name="layer"))
    model.attach_hypergrad(curvature=-1.0, learning_rate=lr)
    stats = trainer.train_epoch(model, MeanSquaredError(), dataset, schedule)
    print(label, f"avg_loss={stats.average_loss:.6f}")
```

## SpiralTorchRL quickstart

`spiraltorch.spiral_rl` packages the Rust-side reinforcement-learning harness so
Python notebooks can select actions, update native agents, and inspect compact
state dictionaries without reimplementing the loop in Python.

### Legacy `rl` imports

Older notebooks sometimes `import rl` directly. The Python binding now
discovers whether the native wheel exposes `spiraltorch.rl` before wiring a
lazy import hook. If another library has already populated `sys.modules["rl"]`
we leave it untouched; otherwise importing `rl` defers to the SpiralTorch
module on demand. Wheels built without SpiralTorchRL skip the hook entirely so
third-party modules remain unaffected.

```python
from spiraltorch.spiral_rl import stAgent

agent = stAgent(state_dim=4, action_dim=2, discount=0.97, learning_rate=0.02)

state = 0
next_state = 1
action = agent.select_action(state)
agent.update(state, int(action), reward=1.0, next_state=next_state)

print("action:", action)
print("epsilon:", agent.state_dict()["epsilon"])
```

The generic `spiraltorch.rl.Agent` wrapper exposes the same loop with an
explicit config object and exploration schedule:

```python
from spiraltorch.rl import Agent, AgentConfig, EpsilonGreedy

config = AgentConfig(
    "dqn",
    state_dim=4,
    action_dim=2,
    gamma=0.97,
    lr=0.02,
    exploration=EpsilonGreedy(0.2, 0.05, 100),
    seed=7,
)
agent = Agent(config)
action = agent.select_action(0)
agent.update(0, int(action), 1.0, 1)
print(agent.algo, agent.state_dict()["epsilon"])
```

The RL surface is intentionally compact today: keep state/action loops native,
then export `state_dict()` for audit or handoff. Geometry-aware policy traces
remain Rust-first until their Python facade is stable enough to document as a
copy-paste path.

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

report = rec.train_epoch(ratings)
print(report.rmse, report.samples)
print("score:", rec.predict(0, 0))
print("top-k:", rec.recommend_top_k(0, 3))
```

### SpiralSession backend planning

`SpiralSession` is intentionally small: it records the requested backend,
captures device preflight evidence, and exposes planner helpers that match the
Rust runtime. Use it as the first runtime object before deciding whether a run
should stay on CPU, ask for WGPU, or escalate into a heavier training recipe.

```python
from spiraltorch import SpiralSession

session = SpiralSession(backend="wgpu")
print(session.requested_backend, "->", session.effective_backend)
print("runtime:", session.device_preflight["runtime_status"])

plan = session.plan_topk(rows=8, cols=64, k=4)
print(plan.kind, plan.effective_backend, plan.tile)
```
