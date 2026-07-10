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
- `spiraltorch.text` for contextual Lagrangian gates plus token-level semantic
  scale stacks via `token_scale_stack` and `token_coherence_levels`, useful for
  FT/runtime probes over local-HF embeddings.
- `spiraltorch.frac` for Rust-backed fractional/Mellin/Z-space probes, including
  `fft_real`, `fft_complex32`, `fft_radix2`, and `fft_radix4` from
  `st-frac::fft` for lightweight spectrum checks during WASM, telemetry, and
  local-HF inference experiments.
- `spiraltorch.safety` for Rust-backed Drift-Response Linguistics metrics,
  including `drl_analyse_word`, `drl_trainer_penalty`, and frame summaries that
  can be injected into FT telemetry, prompt/runtime drift reports, or API-model
  routing traces.
- `spiraltorch.kv` for planner-choice persistence helpers, including
  `kv_choice_from_rank_plan`, Redis-compatible choice keys, validated JSON SET
  option payloads, and `kv_redis_*` calls when built with `--features kv-redis`.
- `spiraltorch.wgpu` for GPU-free WGPU kernel catalog and selection reports,
  including `wgpu_kernel_catalog`, `wgpu_kernel_report_from_rank_plan`, and
  softmax/rank-k dispatch descriptors for runtime trace cards.
- `spiraltorch.vision` for Rust-backed `ImageTensor`, `TransformPipeline`,
  in-memory vision datasets/dataloaders, lightweight classification models,
  static dataset/model catalogs, and transform GPU-coverage audit reports that
  can be reused by FT, WASM, and runtime probe scripts.
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

### Rust-to-Python exposure queue

The native exposure queue is intentionally ordered by immediate experiment
value:

1. `st-frac::fft` spectrum helpers, now exposed as `st.frac.fft_real`,
   `st.frac.fft_complex32`, `st.frac.fft_radix2`, and `st.frac.fft_radix4`.
2. `spiral-safety::drift_response` DRL metrics, now exposed as
   `st.safety.drl_analyse_word`, `st.safety.drl_trainer_penalty`, and related
   summary helpers for FT telemetry penalties, prompt/runtime drift reports, and
   safety-aware training traces.
3. `st-kv` JSON/choice persistence, now exposed as `st.kv_choice_from_rank_plan`,
   `st.kv_rank_choice_key`, `st.kv_json_set_options`, and `kv_redis_*` helpers
   when the binding is built with `--features kv-redis`, so Python experiments
   can reuse the same Redis-backed rank/choice stores as Rust workers.
4. `st-backend-wgpu` kernel descriptor/report helpers, now exposed as
   `st.wgpu_kernel_catalog`, `st.wgpu_kernel_report_from_rank_plan`, and
   `st.wgpu_softmax_kernel_report` for WGPU-first runtime selection without
   requiring direct Rust inspection or a live GPU device.
5. `st-vision` image preprocessing, dataset, dataloader, and lightweight model
   helpers, now exposed as `st.ImageTensor`, `st.TransformPipeline`,
   `st.TensorVisionDataset`, `st.VisionDataLoader`,
   `st.vision_create_classification_model`, and catalog/audit helpers for
   FT/WASM/runtime probes without dropping into Rust.
6. `st-text::semantics` token helpers, now exposed as `st.token_scale_stack`
   and `st.token_coherence_levels` so local-HF embeddings can be inspected with
   the same semantic scale-stack implementation as Rust.
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
  `report.json` sweeps and inspect profile-winner stability plus carried WASM
  context loss/WebGPU readiness and context-consistency status across runs; the
  `api_llm_live_provider_matrix_sweep.py` example can run several token-budget
  pairs and produce that comparison in one command. Pass `--resume-existing`
  when expanding a sweep so completed budget pairs are reused instead of
  re-calling provider APIs. Topos sweep reports can be
  distilled into route rewards with `api_llm_topos_sweep_route_rewards(...)`
  and learned by any stAgent-shaped loop via
  `train_stagent_topos_route_policy(...)`; the
  `examples/api_llm_topos_stagent_route_policy.py` demo runs this keylessly or
  reuses an existing sweep report with `--report report.json`. Browser-side
  WASM learning reports can also be
  loaded with `load_wasm_report(...)`, summarized with `summarize_wasm_report(...)`,
  converted into reusable context via `api_llm_wasm_context_partials(...)`, and
  passed as `context_partials=` to `ApiLLMZSpaceRuntime` or
  `run_api_llm_prompt_suite(...)`; see
  `examples/api_llm_wasm_context_runtime.py` for a keyless end-to-end bridge, or
  `examples/openai_api_llm_wasm_context_runtime.py` for a live OpenAI Responses
  smoke that can prepend bounded context with `--include-context-prompt`, persists
  the selected WASM context handoff, and writes trace JSONL, or
  pass `--wasm-report report.json` to `examples/api_llm_live_provider_matrix.py`
  and `examples/api_llm_live_provider_matrix_sweep.py` when live OpenAI/Claude
  route comparisons should carry the same browser-side learning signal. Use
  `collect_wasm_report_paths(...)` or `build_wasm_report_context(...)` directly,
  or pass `--wasm-report-glob`, `--wasm-report-dir`, `--wasm-report-recursive`,
  and `--wasm-max-reports` to collect repeated browser runs, compare them by
  loss plus audited readiness, and feed only the strongest reports into the
  provider matrix. Use `audit_wasm_report(...)` or
  `audit_wasm_report_context(...)` before promotion; context artifacts also
  carry the readiness status, learning-progress score, risk flags, and audit
  recommendations. Persist the selected handoff for later FT/notebook/API runs with
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
- Vision preprocessing via `vision.ImageTensor` and
  `vision.TransformPipeline`: resize, center-crop, deterministic horizontal
  flip, normalize, audit GPU transform coverage, and inspect canonical
  dataset/model catalogs from Python.
- Vision mini-pipelines via `vision.TensorVisionDataset`,
  `vision.VisionDataLoader`, and `vision.VisionModel`: build small in-memory
  batches, apply Rust transforms during loading, stack image batches, and run
  lightweight classification forward/feature extraction from Python.
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
  --preset hf-full-finetune \
  --require \
  --runtime-device-backend wgpu \
  --json-out ft-runtime.json
```

or call the same contract from Python:

```python
import spiraltorch as st

report = st.runtime_import_preflight_report(
    runtime_import_presets=["hf-full-finetune"],
    required_runtime_import_presets=["hf-full-finetune"],
    runtime_device_backends=["wgpu"],
)
st.write_runtime_import_preflight_report(report, "ft-runtime.json")

ft_report = st.hf_finetune_preflight_report(
    runtime_device_backends=["wgpu", "cpu"],
)
print(ft_report["model_profile_id"], ft_report["hf_model_name"])
print(ft_report["hf_finetune_rust_surfaces"])
```

With no explicit `model_name`, the generic HF preflight resolves the default
model profile (`causal-lm-local-smoke`) and records its model/tokenizer/family
metadata. Pass `model_name=...` for a one-off override, or `model_configs=` plus
`model_profile=` to pin another config-driven route.

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
  --preset hf-full-finetune \
  --require \
  --runtime-device-backend wgpu \
  --json-out ft-runtime.json
spiral-runtime-preflight --preset hf-peft --require --json
spiral-runtime-preflight --preset hf-peft-finetune --require --json
```

Install the stronger local Hugging Face fine-tuning dependency surface with
`pip install "spiraltorch[hf-full-finetune]"` or compose a narrower surface with
`pip install "spiraltorch[hf-finetune,hf-peft]"`. The older
`spiraltorch[hf-gpt2-ft]` extra remains available as a compatibility alias for
historical scripts. Use `hf-runtime` for inference-only
`transformers`/`torch`/`tokenizers` checks, `hf-finetune` for the lighter
`datasets`/`accelerate`/`safetensors` contract, `hf-peft` for adapter-only
workflows, `hf-peft-finetune` for PEFT plus the dataset/Trainer stack, and
`hf-trl-sft` when a TRL SFT loop should be importable in the same environment.

For a real local causal-LM FT run with a larger dataset, treat this as a hard
dependency boundary rather than a suggestion: the Rust wheel already exposes the
default `nn`, `text`, `logic`, `spiral_rl`, and WGPU-backed tensor/runtime
surface, while Python must bring the HF data/model stack. In practice,
`st-tensor`/`st-nn` map to `torch` plus WGPU readiness evidence, `st-text` and
`st-logic` map to `transformers`/`tokenizers`, large local corpora require
`datasets` + `pyarrow`, training orchestration requires `accelerate` and
`safetensors`, and adapter/evaluation experiments should have `peft` and
`evaluate` available before the first long run starts.

The generic HF bridge turns that boundary into an executable run card. The
historical `hf_gpt2_*` scripts still work, but new runs should prefer the
`hf_*` entrypoints plus a model profile so the same path can target the default
`causal-lm-local-smoke` profile, GPT-2/DistilGPT-2 baselines, Pythia, Qwen,
tiny CI models, or another local `AutoModelForCausalLM` profile. Keep
model-specific settings in
`bindings/st-py/examples/hf_finetune_model_configs.example.json` or a copied
config file rather than baking them into the script name. The generic
`spiral-hf-finetune`, `spiral-hf-finetune-sweep`, and
`spiral-hf-zspace-generation-control-sweep` entrypoints default to the built-in
`causal-lm-local-smoke` profile when no `--model-name`, `--model-profile`, or
`--model-configs` is supplied; pass any of those flags to pin a specific model
explicitly. Profiles can carry model/tokenizer names, model family and
parameter-scale labels, training shape, dataset/revision/streaming defaults,
full-FT or LoRA mode, adapter rank/targets, generation/Z-Space softmax knobs,
activation hook selectors, and local runtime policy such as remote-code trust,
disk guards, dataloader pinning, tokenizer-estimate policy, or required
SpiralTorch backends.
Profiles may also use `extends` to create model-neutral aliases or override only
one nested section without duplicating model-specific settings:

```bash
spiral-hf-profile --list

spiral-hf-profile \
  --model-profile causal-lm-local-smoke

spiral-hf-profile \
  --model-configs bindings/st-py/examples/hf_finetune_model_configs.example.json \
  --model-profile pythia-70m-local-smoke

spiral-hf-profile \
  --model-configs bindings/st-py/examples/hf_finetune_model_configs.example.json \
  --model-profile qwen2-0.5b-local-smoke \
  --cli-args

spiral-hf-profile \
  --model-configs bindings/st-py/examples/hf_finetune_model_configs.example.json \
  --model-profile qwen2-0.5b-local-smoke \
  --runtime-contract

spiral-hf-profile \
  --runtime-contract-artifact runs/hf-finetune-qwen2-zspace-ft/runtime-contract.json

spiral-hf-profile \
  --model-configs bindings/st-py/examples/hf_finetune_model_configs.example.json \
  --model-profile qwen2-0.5b-local-smoke \
  --preflight \
  --mode inference

spiral-hf-profile \
  --model-configs bindings/st-py/examples/hf_finetune_model_configs.example.json \
  --model-profile qwen2-0.5b-local-smoke \
  --preflight \
  --mode finetune \
  --require

spiral-hf-profile \
  --model-configs bindings/st-py/examples/hf_finetune_model_configs.example.json \
  --model-profile qwen2-0.5b-local-smoke \
  --launch-plan \
  --mode finetune \
  --output-dir runs/hf-finetune-qwen2-zspace-ft \
  --zspace-probe \
  --bundle-dir runs/hf-finetune-qwen2-zspace-ft

spiral-hf-profile \
  --inspect-bundle runs/hf-finetune-qwen2-zspace-ft \
  --refresh-preflight

PYTHONPATH=bindings/st-py python bindings/st-py/examples/hf_finetune_bridge.py \
  --model-configs bindings/st-py/examples/hf_finetune_model_configs.example.json \
  --model-profile causal-lm-local-smoke \
  --metadata-only \
  --allow-remote \
  --zspace-probe \
  --run-card ft-run-card.json
```

Use `spiral-hf-profile --list --json` or
`st.hf_finetune_model_profile_catalog(...)` when automation needs the same
profile catalog as structured data before choosing a model/run shape. Use
`st.hf_finetune_model_profile_runtime_contract(...)` when FT, local inference,
or Z-Space generation code needs a single profile-derived contract containing
the selected model/tokenizer, activation hook selectors, Z-Space generation
knobs, runtime preset, rough token-estimate policy, focused generation-control
CLI args, and expanded local-inference runtime CLI args. The same contract is
embedded into profile-backed inference-distortion runtime plans and checkpoint
generation-control reports, so downstream artifacts can be replayed or audited
without re-resolving the model config. Use
`spiral-hf-profile --runtime-contract-artifact ...` or
`st.hf_finetune_model_profile_runtime_contract_from_artifact(...)` to recover
that contract from a saved run card/report/contract JSON when you are stitching
a later FT, local-HF inference probe, or Z-Space generation-control run back together. Add
`--preflight` to probe the selected profile's inference, finetune, PEFT, or
TRL-SFT runtime preset before launching a long run; add `--require` when that
probe should act as a CI/local gate instead of an observational report. Add
`--launch-plan` when you want the resolved command, expanded profile args, and
preflight result in one artifact before deciding whether to run metadata-only
smoke, local inference diagnostics, or a full FT job; pair it with `--out` and
`--lines-out` to keep a reproducible run manifest next to the eventual run card.
Add `--script-out` when you also want an executable shell handoff that replays
the reviewed command without re-resolving the profile. For normal runs,
`--bundle-dir` writes all three artifacts together:
`profile-launch-plan.json`, `profile-launch-plan.lines`, and
`profile-launch-plan.sh`. Use `--inspect-bundle` before replaying the shell
handoff to verify that the bundle is complete, executable, and still matches the
planned command. Add `--refresh-preflight` to re-probe the current runtime
imports/devices from the stored plan, and `--require-refresh-preflight` when
that refreshed runtime check should become a hard gate.

After installing from a wheel, use the installed console entrypoint instead of
the repo path:

```bash
spiral-hf-finetune \
  --model-profile causal-lm-local-smoke \
  --metadata-only \
  --allow-remote \
  --zspace-probe \
  --run-card ft-run-card.json
```

After the contract passes, add `--train --max-train-samples 50000 --block-size
128 --output-dir runs/gpt2-small-zspace-ft` to run a real local
`AutoModelForCausalLM` / `Trainer` fine-tune. Train runs write
`spiraltorch-hf-finetune-run-card.json` and
`spiraltorch-hf-finetune-trainer-trace.jsonl` by default when launched through
the generic `hf_*` wrappers, capturing
train/log/evaluate/save/end events and summarizing loss/eval-loss telemetry
back into the run card; add `--trainer-telemetry --trainer-desire-gain 1.0
--trainer-psi-gain 1.0` to inject bounded desire/psi frames into those events
and compare them across run cards. When an inference-distortion handoff is
attached, the bridge auto-enables those trace telemetry frames so
`hf_ft.inference_distortion.*` can be correlated with loss and eval events even
if `--trainer-telemetry` was not passed explicitly; the run card records both
`trainer_telemetry_requested` and the resolved enabled/auto-reason fields. Pass
`--no-trainer-trace` only when that audit trail is too noisy. Use
`--require-runtime-device-ready-backend wgpu`
when the SpiralTorch WGPU surface must be available before the run starts.

For an adapter run on Apple Silicon or a model that is too expensive to update
fully, select one of the built-in LoRA profiles. `--mode auto` resolves these
profiles to the Trainer-ready `hf-peft-finetune` preflight, while the bridge attaches PEFT
before constructing `Trainer`:

```bash
spiral-hf-profile \
  --model-profile qwen2-0.5b-lora-local-smoke \
  --launch-plan \
  --train \
  --bundle-dir runs/qwen2-lora-plan

spiral-hf-finetune \
  --model-profile qwen2-0.5b-lora-local-smoke \
  --allow-remote \
  --train \
  --train-file data/corpus.txt \
  --validation-fraction 0.02 \
  --output-dir runs/qwen2-lora
```

The sample config also includes `causal-lm-lora-local-smoke`,
`gpt2-lora-local-smoke`, and `smollm2-135m-lora-local-smoke`. For an unlisted
family, pass repeated `--lora-target-module` flags or add an `adapter` section
with `target_modules` to the profile. `--gradient-checkpointing` disables model
KV caching and lowers activation memory. The run card records the resolved
target modules, matched layer count, trainable/frozen parameter counts and
ratio, PEFT version, artifact kind, and whether `adapter_config.json` was
actually saved.

An adapter-only output can now be continued as a new Trainer run without
merging or attaching a second LoRA. This is a weights-only warm start: the
adapter remains trainable, while optimizer and scheduler state are reset for a
new corpus or learning-rate schedule:

```bash
spiral-hf-finetune \
  --model-name runs/qwen2-lora \
  --finetune-mode lora \
  --train \
  --train-file data/continued-corpus.txt \
  --output-dir runs/qwen2-lora-continued
```

Local adapters are auto-detected. For a Hub adapter id, add
`--model-artifact-kind peft-adapter`; model profiles can persist the same
choice as `"artifact_kind": "peft-adapter"`. The run card records
`finetune_start_report.mode=adapter_warm_start`, the resolved base/tokenizer,
active adapter, runtime PEFT config, and whether a new adapter was attached.

Every successful LoRA save now writes
`spiraltorch-hf-adapter-lineage.json`. Its path-independent SHA-256 identity is
derived from `adapter_config.json` plus all adapter weight files, and the node
records its verified parent, root, depth, start mode, and run-card digest. The
bridge refuses to continue an adapter in place: input and output directories
must not overlap so the parent remains immutable.

Require before/after evidence and stop the run from becoming a promotion
candidate when eval loss regresses:

```bash
spiral-hf-finetune \
  --model-name runs/qwen2-lora \
  --finetune-mode lora \
  --train \
  --train-file data/continued-corpus.txt \
  --eval-before-train \
  --adapter-promotion-gate \
  --adapter-promotion-max-eval-loss-regression 0.02 \
  --output-dir runs/qwen2-lora-candidate
```

The bound is `eval_after - eval_before`: `0` rejects any regression, while a
negative value requires improvement. The gate also requires a matching
lineage/run-card digest, a finite trainer loss, successful adapter save, and a
content change from a local parent. Add
`--adapter-promotion-require-generation-change` with `--generation-prompt` when
changed generation is part of the promotion contract. Reports are written to
`spiraltorch-hf-adapter-promotion.json`; a blocked or evidence-incomplete gate
returns a nonzero exit status while preserving both artifacts for inspection.

Existing adapters can use the same contracts without rerunning training:

```bash
spiral-hf-adapter-lineage \
  --adapter runs/qwen2-lora-candidate \
  --parent-adapter runs/qwen2-lora \
  --run-card runs/qwen2-lora-candidate/spiraltorch-hf-finetune-run-card.json

spiral-hf-adapter-promote \
  --candidate runs/qwen2-lora-candidate \
  --parent-adapter runs/qwen2-lora \
  --run-card runs/qwen2-lora-candidate/spiraltorch-hf-finetune-run-card.json
```

The importable equivalents are `st.hf_adapter_fingerprint(...)`,
`st.write_hf_adapter_lineage(...)`, and
`st.hf_adapter_promotion_report(...)`. A remote parent reference remains
explicitly unverified until its adapter files are available locally, so it
cannot silently satisfy the default promotion gate.

Audit several promoted generations as one chain, then feed its selected tip
directly back into the normal scale-up command builder:

```bash
spiral-hf-adapter-chain runs/qwen2-study \
  --out runs/qwen2-study/promotion-chain.json \
  --require-continuation-ready

spiral-hf-scale-up runs/qwen2-study/promotion-chain.json \
  --output-dir runs/qwen2-generation-next \
  --write-command runs/qwen2-study/next-generation.json \
  --require-ready
```

The chain report re-fingerprints every local adapter, verifies parent/root/depth
and ancestor continuity, checks each non-root promotion report and run-card
digest, preserves rejected branches, and selects only a unique deepest ready
tip. Equal-depth tips return `ambiguous` until `--select-adapter-id` resolves
the fork. New FT run cards carry their exact `launch_command`, so each promoted
generation can produce the next one without a sweep artifact. For older run
cards, repeat `--command-artifact <scale-up-command.json>`; a local seed from
before lineage manifests existed may be inferred only when its live fingerprint
matches the declared root ID (`--no-infer-roots` disables that compatibility
path). Python can use `st.hf_adapter_promotion_chain_report(...)`,
`st.write_hf_adapter_promotion_chain(...)`,
`st.load_hf_adapter_promotion_chain(...)`, and then
`st.hf_finetune_scale_up_command(chain, ...)` for the same flow.

Make the chain stop itself when additional adapter generations no longer earn
their cost:

```bash
spiral-hf-adapter-chain runs/qwen2-study \
  --max-lineage-depth 6 \
  --target-eval-loss 2.40 \
  --min-eval-improvement 0.002 \
  --plateau-patience 2 \
  --out runs/qwen2-study/promotion-chain.json \
  --require-continuation-ready
```

These gates are opt-in. `max lineage depth` stops at the selected depth,
`target eval loss` stops after reaching the target, and the plateau gate stops
after the configured number of consecutive generations have
`eval_before_loss - eval_after_loss` below the minimum. Missing eval evidence
blocks continuation instead of being treated as improvement. A stopped report
keeps `chain_ready=True` and `continuation_artifacts_ready=True` for audit, but
sets `continuation_ready=False`; `spiral-hf-scale-up` then returns
`promotion_chain_stopped_by_policy` without emitting a launch command. Python
can inspect or persist the same decision with
`st.hf_adapter_continuation_policy_report(...)`,
`st.write_hf_adapter_continuation_policy(...)`, and
`st.load_hf_adapter_continuation_policy(...)`.

Close the loop with a resumable executor. Start with a dry-run; it writes an
atomic state artifact containing the live chain audit, policy decision,
resolved command, and preflight without launching training:

```bash
spiral-hf-adapter-executor runs/qwen2-study \
  --output-root runs/qwen2-study/executor \
  --state runs/qwen2-study/executor/state.json \
  --max-lineage-depth 6 \
  --min-eval-improvement 0.002 \
  --plateau-patience 2
```

After inspecting that state, execute one promoted generation:

```bash
spiral-hf-adapter-executor runs/qwen2-study \
  --output-root runs/qwen2-study/executor \
  --state runs/qwen2-study/executor/state.json \
  --max-lineage-depth 6 \
  --min-eval-improvement 0.002 \
  --plateau-patience 2 \
  --run --max-generations 1
```

Repeat the same command to resume from the new deepest eligible adapter, or
raise `--max-generations` explicitly for several synchronous generations in
one invocation. The executor holds a single-writer lock for the output root,
flushes `running` state before every subprocess, then records the child PID,
hostname, working directory, and a unique combined stdout/stderr log under
`executor-logs/`. Logs are created owner-only (`0600`) on POSIX. Output is also
mirrored to the terminal by default; use
`--no-tee-output` to keep the terminal quiet without losing the durable log.
While output flows, byte-count and last-output heartbeats are flushed to state
at a bounded interval. After the process exits, the executor re-fingerprints
the output and requires the expected parent ID, lineage depth, promotion-ready
node, and selected chain tip before advancing. A failed command remains in
history and retries the same depth only when no partial output collides;
unresolved interrupted runs fail closed unless explicitly audited with
`--retry-interrupted`. A recorded live local PID or unverified remote process
blocks recovery even with that flag, preventing duplicate training. Step and
sample multipliers default to `1.0`, so repeated generations do not grow their
resource budget silently.

Inspect a live or completed executor without changing its state:

```bash
spiral-hf-adapter-executor-status \
  runs/qwen2-study/executor/state.json \
  --require-healthy
```

The status surface reports state age, active attempt, local PID liveness,
single-writer lock presence, output presence, and log size. PID liveness is an
observation rather than a process-identity guarantee; remote-host and legacy
attempts are reported as unverified instead of being guessed healthy.
`--require-healthy` also rejects a missing active lock, durable subprocess log,
or promoted output. Python can drive the same state machine with
`st.run_hf_adapter_continuation_executor(...)` and inspect it with
`st.load_hf_adapter_continuation_executor(...)` or
`st.hf_adapter_continuation_executor_status_report(...)`.

Use `--resume-from-checkpoint` only for an exact Trainer continuation that
should restore optimizer, scheduler, and RNG state. SpiralTorch audits
`trainer_state.json` and those state files before loading. A checkpoint saved
at its original `max_steps` may have an exhausted scheduler; extending that
exact resume can advance `global_step` while keeping the learning rate at zero.
Inspect the decision from Python before launching:

```python
import spiraltorch as st

resume = st.hf_finetune_checkpoint_resume_report(
    "runs/qwen2-lora/checkpoint-1000",
    requested_max_steps=2000,
)
print(st.hf_finetune_checkpoint_resume_lines(resume)[0])
print(resume["recommendation"], resume["recommended_args"])
```

When `scheduler_extension_risk` is true, use that checkpoint as
`--model-name ... --finetune-mode lora` without
`--resume-from-checkpoint` to warm-start its adapter weights under a fresh
optimizer/scheduler.

The same preparation path is importable without eagerly importing PEFT:

```python
import spiraltorch as st

model, report = st.prepare_hf_finetune_model(
    model,
    mode="lora",
    model_family="qwen2",
    rank=16,
    alpha=32,
    gradient_checkpointing=True,
)
print(report["parameter_report_after"]["trainable_parameter_ratio"])
```

For an adapter already loaded with
`load_hf_causal_lm_artifact(..., is_trainable=True)`, pass
`preloaded_adapter=True` to reuse it safely. The preparation API rejects
double attachment and rejects full FT on a PEFT-wrapped model until it is
merged.

Adapter-only outputs are also first-class inference artifacts. Local
`adapter_config.json` directories are detected automatically; SpiralTorch loads
the recorded base model, prefers tokenizer files saved beside the adapter, and
attaches PEFT only for that route:

```python
model, tokenizer, config, load_report = st.load_hf_causal_lm_artifact(
    "runs/qwen2-lora",
    loader_kwargs={"local_files_only": True},
)
print(st.summarize_hf_causal_lm_artifact(load_report))
```

The same loader now backs `byte_lm_transformers_trace.py`, the local branch of
`spiral-zspace-inference-distortion-probe`, Z-Space generation-control sweeps,
and Transformers checkpoint audit. Use `--model-artifact-kind peft-adapter`
for a remote adapter id that cannot be detected from a local directory.

Inspect an adapter without loading Transformers, or merge it atomically into a
standalone full-model directory for runtimes that do not ship PEFT:

```bash
spiral-hf-adapter-export \
  --adapter runs/qwen2-lora \
  --inspect-only

spiral-hf-adapter-export \
  --adapter runs/qwen2-lora \
  --output-dir runs/qwen2-lora-merged
```

Merge export refuses a non-empty output directory and never overwrites the
source adapter. The output contains normal model/tokenizer files plus
`spiraltorch-hf-merged-export.json` with base-model, PEFT, parameter, and merge
provenance.

For a larger local corpus, bypass Hub datasets and feed files directly:

```bash
PYTHONPATH=bindings/st-py python bindings/st-py/examples/hf_finetune_bridge.py \
  --model-configs bindings/st-py/examples/hf_finetune_model_configs.example.json \
  --model-profile qwen2-0.5b-local-smoke \
  --corpus-scan \
  --train \
  --train-file data/corpus-000.txt \
  --train-file data/corpus-001.txt \
  --validation-fraction 0.02 \
  --dataset-format text \
  --max-train-samples 50000 \
  --block-size 128 \
  --output-dir runs/hf-finetune-qwen2-zspace-ft
```

The run card stores `corpus_file_report` with file counts, total bytes, missing
files, and a lightweight path/size/mtime fingerprint so repeat FT runs can be
compared before reading model metrics. With `--corpus-scan`, it also stores
`corpus_scan_report`: streamed line counts, nonempty/empty-line shape, rough
GPT-2 token estimates, short sample texts, and per-file scan hashes. Use
`--corpus-scan-max-bytes-per-file` when you want a bounded preview of very
large corpora instead of a full pre-train pass over every byte. When
`--allow-remote` is explicit, the bridge temporarily relaxes local Hugging Face
offline environment flags for model/tokenizer downloads; model-load,
tokenization, `TrainingArguments`, and Trainer failures are written into the run
card so long-run blockers leave structured evidence instead of only a traceback.
After tokenization/grouping, `dataset_fit_report` records whether train/eval
splits produced enough fixed-size blocks; empty eval blocks are reported and
disabled instead of producing evaluation-looking traces without `eval_loss`.
Add `--generation-prompt "SpiralTorch is" --generation-max-new-tokens 32` to
store deterministic before/after text samples, generated continuation text,
token deltas, and hashes in the same run card. Use `--generation-do-sample`
with `--generation-temperature` and optional `--generation-top-k` when you want
qualitative samples to include sampling variance. Add
`--generation-zspace-softmax --generation-ngram-size 3 --generation-ngram-window
96 --generation-ngram-repression-strength 0.75` to route post-FT generation
through SpiralTorch's entropy thermostat plus phrase-scale repetition
repression.
Add `--eval-before-train` when you want a numeric baseline before updates; the
bridge writes `eval_before_train` plus final `eval_after_train` reports with
`eval_loss`, perplexity, raw metrics, and skipped/error status. Pass
`--no-eval-after-train` only when a long run must avoid the final eval pass.
For heavier local eval loops, add `--max-eval-blocks N` to cap grouped eval
blocks, use `--eval-after-train-policy skip-if-final-step-eval` when
`max_steps` already lands on an eval boundary, and leave
`--dataloader-pin-memory auto` so MPS/CPU runs avoid unsupported pin-memory
warnings while CUDA can still opt in.
After several runs, call
`st.compare_hf_finetune_run_cards([run_a, run_b, ...])` to flatten
run-card JSON into eval-loss/perplexity deltas, generation-sample changes,
dataset-fit status, and trainer metrics so tuning choices can be ranked without
hand-reading each artifact.

Before another FT pass, the same desire/psi pressure can be tested at inference
time:

```bash
PYTHONPATH=bindings/st-py python bindings/st-py/examples/zspace_inference_distortion_probe.py \
  --local-model runs/gpt2-small-zspace-ft \
  --prompt "SpiralTorch is a tensor and geometry runtime that" \
  --activation-name-contains transformer.h.0 \
  --out runs/zspace-inference-distortion-probe.json
```

When a previous FT/profile artifact already carries a model-profile runtime
contract, pass it directly instead of repeating model-specific flags:

```bash
PYTHONPATH=bindings/st-py python bindings/st-py/examples/zspace_inference_distortion_probe.py \
  --runtime-contract-artifact runs/hf-finetune-qwen2-zspace-ft/runtime-contract.json \
  --prompt "SpiralTorch is a tensor and geometry runtime that" \
  --out runs/zspace-inference-distortion-probe.json
```

The probe builds one `api_llm_zspace_inference_distortion_adapter(...)` and
uses it three ways: local HF logits receive `ZSpaceRepressionLogitsProcessor`
kwargs, matching activation hooks record and can gently intervene in selected
modules, and API-model-shaped calls receive request overrides plus bounded
Z-space context telemetry. Omit `--local-model` for a keyless fake API smoke, or
add `--api-provider openai-responses|openai-chat|anthropic --api-model <model>`
to reuse the same distortion adapter with a live provider. Provider adapters
filter request overrides against each SDK method signature and record
`api_request_dropped_keys`; they also retry once-per-parameter when a model
rejects an otherwise SDK-shaped request option such as `temperature` or `top_p`,
recording those server-side drops as `retry_dropped_keys`. Responses/Chat/Messages
surface differences therefore stay auditable instead of silently changing the
experiment. Load the artifact with
`st.load_zspace_inference_distortion_probe(...)`, flatten it with
`st.summarize_zspace_inference_distortion_probe(...)`, or print compact status
lines with `st.summarize_zspace_inference_distortion_probe_lines(...)`. When
several pressure/coherence settings have been tried, call
`st.compare_zspace_inference_distortion_probes(...)` or
`st.summarize_zspace_inference_distortion_probe_comparison_lines(...)` to rank
the artifacts by local text changes, top-token changes, activation evidence,
API non-empty response, request-drop/retry-drop compatibility, and distortion
energy. Comparisons also compute `api_compatibility_score` so equal-effect
candidates prefer visible text with fewer server-side request retries before
falling back to labels.
See `examples/zspace_inference_distortion_local_gpt2_openai_sample.json` for a
sanitized local-GPT-2 plus OpenAI Responses run that keeps the local hook/logits
evidence while omitting API keys, response ids, and absolute local paths.
`examples/zspace_inference_distortion_local_gpt2_gpt5nano_sample.json` shows the
same pre-FT probe against `gpt-5-nano`, including model-rejected request knobs
captured as `retry_dropped_keys` and the `--api-reasoning-effort minimal`
`--api-text-verbosity low` route that keeps visible text flowing.
After a sweep, replay its recommended setting directly with
`--from-sweep-report runs/zspace-inference-distortion-sweep/sweep-report.json`;
the probe imports the saved prompt/runtime plus recommended distortion config,
while any explicitly passed CLI flag overrides the handoff value.

For a reproducible pre-FT comparison, run the same probe over a small
desire/psi/coherence grid:

```bash
PYTHONPATH=bindings/st-py python bindings/st-py/examples/zspace_inference_distortion_sweep.py \
  --local-model runs/gpt2-small-zspace-ft \
  --prompt "SpiralTorch is a tensor and geometry runtime that" \
  --activation-name-contains transformer.h.0 \
  --desire-pressure-values 0.45,0.8 \
  --psi-total-values 0.5,0.75 \
  --coherence-values 0.35,0.55 \
  --out-dir runs/zspace-inference-distortion-sweep
```

`--runtime-contract-artifact <run-card-or-contract.json>` works here too; the
sweep report keeps that artifact reference in its replay commands while still
letting explicit CLI flags override any recovered local-model/tokenizer/hook
defaults.

The sweep writes `sweep-plan.json`, one probe artifact per setting,
`sweep-report.json`, and `sweep-report.md` with comparison rows, a recommended
probe/config, replay commands, and compact summary lines. It uses the keyless
fake API route by default, so the local-HF hook path can be checked first. Add
`--api-provider openai-responses|openai-chat|anthropic` plus `--api-model
<model>` to replay the exact same distortion grid against a live API model. For
interrupted or paid-provider sweeps, add `--resume-existing` to reuse successful
probe artifacts, `--report-only` to rebuild `sweep-report.json` and
`sweep-report.md` without touching local/API models, or promote one or more
single-probe artifacts directly:

```bash
PYTHONPATH=bindings/st-py python bindings/st-py/examples/zspace_inference_distortion_sweep.py \
  --from-probe runs/inference-distortion/live-local-api-probe.json \
  --from-probe-label live-openai-check \
  --out-dir runs/zspace-inference-distortion-sweep
```

The `--from-probe` path preserves the saved prompt/runtime/config, local
generation-control evidence, activation-hook report, and provider request-filter
audit, so a live API probe can be promoted into a reusable pre-FT handoff without
paying for another provider call. Use `--force` only when every row should be
rerun intentionally. Existing grid artifacts are reused only when the saved
prompt, distortion config, and runtime/provider settings match the current
sweep. From Python, call `st.load_zspace_inference_distortion_sweep(...)`,
`st.summarize_zspace_inference_distortion_sweep(...)`, or
`st.summarize_zspace_inference_distortion_sweep_lines(...)` to recover the
recommended config, request overrides, logits-processor kwargs, and focused
single-probe/sweep CLI args; single-probe summaries also expose the same
effect/risk scores used to rank FT handoff candidates. If you already have one
or more probe artifacts,
`st.zspace_inference_distortion_sweep_report_from_probes(...)` promotes them to
the same sweep-shaped report in memory, preserving request overrides,
logits-processor kwargs, activation-hook settings, and provider request-filter
audit for downstream FT handoff.

For the first real FT pass on a new corpus, use the sweep runner to make that
comparison reproducible:

```bash
PYTHONPATH=bindings/st-py python bindings/st-py/examples/hf_finetune_sweep.py \
  --model-configs bindings/st-py/examples/hf_finetune_model_configs.example.json \
  --model-profile causal-lm-local-smoke \
  --train-file data/corpus-000.txt \
  --validation-fraction 0.02 \
  --corpus-scan \
  --generation-prompt "SpiralTorch is" \
  --generation-from-inference-distortion \
  --eval-before-train \
  --zspace-probe \
  --trainer-telemetry \
  --inference-distortion-probe runs/inference-distortion/live-local-api-probe.json \
  --block-size-values 64,128 \
  --learning-rate-values 0.0001,0.00005 \
  --seed-values 7,13 \
  --out-dir runs/hf-causal-lm-zspace-sweep
```

For a LoRA profile or adapter continuation, the sweep can enforce the same
lineage/eval promotion contract on every row:

```bash
spiral-hf-finetune-sweep \
  --model-configs bindings/st-py/examples/hf_finetune_model_configs.example.json \
  --model-profile qwen2-0.5b-lora-local-smoke \
  --train-file data/corpus.txt \
  --validation-fraction 0.02 \
  --eval-before-train \
  --adapter-promotion-gate \
  --adapter-promotion-max-eval-loss-regression 0.02 \
  --seed-values 7,13 \
  --out-dir runs/qwen2-lora-promotion-sweep
```

`sweep-plan.json` records the resolved artifact kind, FT mode, and promotion
policy before any model is loaded. After execution, run-card summaries expose
lineage depth, promotion status, eval regression, and failed/missing checks.
Blocked or evidence-incomplete rows remain in the comparison for audit, but
only `promotion_ready=True` rows can become the selected run, be reused by
`--resume-existing`, or seed `scale-up-command.json`. If no row is ready, the
sweep summary reports `no_promotion_ready_runs`, scale-up reports
`adapter_promotion_not_ready`, and the non-dry-run CLI exits nonzero.

For a promotion-ready LoRA winner, `scale-up-command.json` now defaults to
`adapter_continuation=auto`: the selected run's adapter directory replaces the
original `--model-name`, `--model-artifact-kind peft-adapter` and
`--finetune-mode lora` are made explicit, and the expected child lineage depth
is recorded before launch. `spiral-hf-scale-up sweep-report.json`
therefore continues the winning weights rather than merely rerunning their
configuration. Pass `--adapter-continuation replay` for the old configuration
replay behavior, or `--adapter-continuation continue` to require adapter
continuation for a non-gated LoRA run. Exact `--resume-from-checkpoint` remains
the stronger path in `auto` mode; combining it with explicit `continue` is
rejected as ambiguous. Scale-up preflight checks local adapter config/weights,
input-output separation, lineage adapter ID/depth, and promotion report
ID/readiness. The sanitized
[`hf_adapter_scale_up_continuation_sample.json`](examples/hf_adapter_scale_up_continuation_sample.json)
records a real tiny-GPT2 depth-1 to depth-2 continuation whose parent
fingerprint, weight change, eval bound, and standalone promotion revalidation
all passed.

For checkpoint-level local inference, use the generic generation-control
wrapper so the fine-tuned checkpoint stays the model path while the profile can
provide tokenizer, decode defaults, Z-Space softmax grids, and runtime access
policy such as `allow_remote` / `trust_remote_code`. The same profile reference
is forwarded into generated sweep and generation-curve commands, so downstream
artifacts keep model/dataset metadata without retyping it:

```bash
spiral-hf-zspace-generation-control-sweep \
  --dry-run \
  --model-configs bindings/st-py/examples/hf_finetune_model_configs.example.json \
  --model-profile pythia-70m-local-smoke \
  --prompt "SpiralTorch is" \
  --out runs/pythia-zspace-generation-control-sweep.json

spiral-hf-checkpoint-generation-control \
  --run-dir runs/local-causal-lm-zspace-ft \
  --checkpoint checkpoint-2048 \
  --model-configs bindings/st-py/examples/hf_finetune_model_configs.example.json \
  --model-profile local-causal-lm-template \
  --dry-run

PYTHONPATH=bindings/st-py python bindings/st-py/examples/hf_checkpoint_generation_control.py \
  --run-dir runs/local-causal-lm-zspace-ft \
  --checkpoint checkpoint-2048 \
  --model-configs bindings/st-py/examples/hf_finetune_model_configs.example.json \
  --model-profile local-causal-lm-template \
  --dry-run
```

When a long FT run is still live, `spiral-hf-generation-curve` can join
checkpoint generation-control sweeps with a trainer trace before the final run
card exists. Pass the same `--model-configs/--model-profile` to stamp model and
dataset metadata into the curve artifact instead of hand-writing `--model-name`:

```bash
spiral-hf-generation-curve \
  runs/local-causal-lm-zspace-ft/prompt-spiral-checkpoint-2048-generation-control-sweep.json \
  --trainer-trace-jsonl runs/local-causal-lm-zspace-ft/spiraltorch-hf-finetune-trainer-trace.jsonl \
  --run-dir runs/local-causal-lm-zspace-ft \
  --model-configs bindings/st-py/examples/hf_finetune_model_configs.example.json \
  --model-profile local-causal-lm-template \
  --out runs/local-causal-lm-zspace-ft/generation-curve.json \
  --lines-out runs/local-causal-lm-zspace-ft/generation-curve.lines
```

It writes `sweep-plan.json` before launching runs and `sweep-report.json` after
the run cards are available, plus `scale-up-command.json` with the
distortion-adjusted next-run command when a completed candidate is available,
including the same
`st.compare_hf_finetune_run_cards(...)` comparison payload. Add
`--inference-distortion-probe` after one local/API inference-distortion probe, or
`--inference-distortion-sweep-report` after a ranked distortion grid, to stamp
the recommended prompt/runtime/config into each FT run card and train-begin trace
event; `sweep-plan.json` and dry-run `sweep-report.json` embed the same handoff
so the scale-up plan can be audited before loading a model. Any attached
handoff also enables trainer trace telemetry automatically unless
`--no-trainer-trace` is set, so the same handoff appears as
`hf_ft.inference_distortion.*` telemetry keys on each frame;
summaries surface the recommended probe, effect/risk, desire pressure, psi total,
API route, and provider request keys that were dropped or sent beside
eval/generation metrics. The bridge preflight line and sweep summaries also
show whether telemetry was explicitly requested or auto-enabled from the
handoff, which makes dry-run plans honest before the model is loaded. Trace
summaries carry the handoff's risk/API retry-drop counts plus logits repression
strengths beside loss/eval telemetry, so post-run comparisons can distinguish
"it trained well" from "it trained well under an aggressive decode field."
`distortion_adjusted_eval_loss` and the sweep `scale_up_candidate_*` fields let
the next longer run prefer near-best eval loss with lower distortion pressure
instead of blindly following raw eval loss alone. When adapter promotion is
required, this ranking runs only after promotion-ready filtering. Those fields
include the candidate run card, trace path, output directory, and replay command
when the sweep report has matching run metadata. From Python,
`st.hf_finetune_scale_up_command(summary_or_report)` turns that candidate into a
shell-safe longer-run command, defaulting to doubled `--max-steps` and
`--max-train-samples` while writing a fresh run card and trainer trace under a
`-scaleup` output directory. The GPT-2-specific
`st.hf_gpt2_finetune_scale_up_command(...)` name remains as a compatibility
alias, but new code should use the generic Hugging Face FT name. The sweep CLI
writes the same payload to `scale-up-command.json` and surfaces its
status/preview in the embedded summary. To inspect or execute that next run
without hand-copying shell text:

```bash
spiral-hf-scale-up \
  runs/hf-finetune-sweep/sweep-report.json \
  --write-command runs/hf-finetune-sweep/scale-up-command-long.json \
  --max-steps 2000 \
  --max-train-samples 200000

spiral-hf-scale-up \
  runs/hf-finetune-sweep/scale-up-command-long.json \
  --run
```

When running from a source checkout without installing the console script, use
`PYTHONPATH=bindings/st-py python bindings/st-py/examples/hf_finetune_scale_up.py`
with the same arguments.

When the source is a saved `scale-up-command*.json`, the CLI replays that saved
command as-is unless you pass explicit overrides such as `--max-steps` or
`--output-dir`. Each replay also includes a lightweight preflight over the
resolved command's executable, bridge script, input files, and output parents;
add `--require-ready` to fail before `--run` when an input artifact has gone
missing. When `--write-command` is combined with `--run`, the written artifact is
updated after execution so it includes `run_returncode` beside the preflight.
From Python, use `st.hf_finetune_scale_up_preflight_report(...)` or
`st.hf_finetune_scale_up_preflight_lines(...)` on a command artifact, sweep
report, or command list to run the same check in notebooks and CI.

Add
`--generation-from-inference-distortion` with a
generation prompt to reuse the handoff's `recommended_processor_kwargs` as the
Z-Space/repression logits processor for before/after generated samples; omit it
when you want to hand-tune `--generation-zspace-*` and repression flags instead.
Run-card and sweep summaries expose the applied handoff probe plus key processor
values such as entropy target and repression strengths beside generation-control
telemetry, plus ready-to-replay bridge CLI args for turning the same inferred
distortion into explicit `--generation-zspace-*` / repression flags, so the
source of a decode change remains auditable.
Dry-run `sweep-plan.json` / `sweep-report.json` also include the planned
handoff-driven generation processor values before any model is loaded.
Add
`--dry-run` to inspect commands without loading Transformers, or
`--require-wgpu-ready` when the SpiralTorch WGPU surface should gate each run.
The report also embeds a compact `summary`; from notebooks or CI, call
`st.load_hf_finetune_sweep_report(...)`,
`st.summarize_hf_finetune_sweep_report(...)`, or
`st.summarize_hf_finetune_sweep_report_lines(...)` to recover the
selected scale-up candidate plus its command/run-card/trace/output directory,
top eval-loss rows, failed runs, and
dry-run/partial/complete status without hand-reading the full artifact. For
longer local runs, add `--resume-existing` to reuse successful per-run cards and
continue only missing or failed rows after an interruption; add `--force` with
that same command when you intentionally want to rerun every row. From Python,
use `st.hf_finetune_inference_distortion_handoff_report(...)` to inspect a
probe or sweep recommendation before launching the FT bridge; the handoff and
run-card/sweep summaries include ready-to-replay generation bridge CLI args and
the chosen probe's `api_compatibility_score` when the recommendation carries
logits-processor kwargs. Use
`st.hf_finetune_inference_distortion_handoff_lines(...)` when you want a
compact, copy-friendly status/replay readout in notebooks or CI logs; bridge
preflight/run-card artifacts and sweep `sweep-plan.json` / `sweep-report.json`
preserve those same lines beside the structured handoff payload, and run-card /
sweep summary helpers surface the same lines plus replay-arg previews. Handoff
reports also include source-selector args such as `--inference-distortion-probe`,
automatic generation handoff args with `--generation-from-inference-distortion`,
and explicit Z-Space generation bridge args when you want to replay the same
processor settings without relying on the automatic override; shell-safe display
strings are included alongside the structured arg lists.
After a longer run is producing checkpoint/status artifacts, use the generic
installed ops CLIs to archive what happened without remembering GPT-2-specific
filenames:

```bash
spiral-hf-run-status --run-dir runs/hf-finetune-qwen2-zspace-ft --max-steps 2000
spiral-hf-trace-summary \
  runs/hf-finetune-qwen2-zspace-ft/spiraltorch-hf-finetune-trainer-trace.jsonl \
  --max-steps 2000
spiral-hf-monitor-snapshot \
  --run-dir runs/hf-finetune-qwen2-zspace-ft \
  --run-status-history-jsonl runs/hf-finetune-qwen2-zspace-ft/status.jsonl
spiral-hf-run-artifacts --run-dir runs/hf-finetune-qwen2-zspace-ft
spiral-hf-run-ops --run-dir runs/hf-finetune-qwen2-zspace-ft --dry-run
```

`spiral-hf-run-status` watches the generic run card/trainer trace names by
default, falling back to legacy GPT-2 filenames when only those exist.
`spiral-hf-monitor-snapshot` folds live status/watch histories into one compact
readout for handoffs and CI logs. The archive commands write
`hf-finetune-run-artifact-manifest.json` and
`hf-finetune-run-ops-snapshot.json` by default. They still read legacy
`spiraltorch-hf-gpt2-ft-*` run cards/traces, but prefer the generic
`spiraltorch-hf-finetune-*` artifacts when both exist.

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
trace = agent.select_action_trace(state)
action = int(trace["action"])
agent.update(state, int(action), reward=1.0, next_state=next_state)

print("action:", action)
print("policy:", agent.policy_report(state))
print("trace:", trace)
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
trace = agent.select_action_trace(0)
action = int(trace["action"])
agent.update(0, int(action), 1.0, 1)
print(agent.algo, agent.policy_report(0), agent.state_dict()["epsilon"])
```

The RL surface is intentionally compact today: keep state/action loops native,
use `policy_report(state)` or `select_action_trace(state)` to audit Q-values,
epsilon, and greedy-vs-exploratory choices, then export `state_dict()` for
handoff.

For API-model topological routing, use
`api_llm_topos_sweep_route_rewards(report, profile="grounded")` to convert a
`run_api_llm_topos_sweep(...)` report into bounded route rewards, then call
`train_stagent_topos_route_policy(report, agent, profile="grounded")` to update
an stAgent-shaped policy and capture the selected route trace. Use
`api_llm_topos_route_policy_selection(policy, report=..., topos_profiles=...)`
to recover the selected route record and, when raw profiles are available,
rebuild the request/runtime-route payload for the next hosted-model call.

## Open-topos learning and inference hints

`topos_control_signal()` turns an open-cartesian guard into one compact pressure
signal, while `topos_training_hints()` and `topos_inference_hints()` split the
same signal into named controls for local learning loops and hosted-model
runtime requests.

```python
import spiraltorch as st

topos = st.hypergrad_topos(max_depth=10, max_volume=100)
signal = st.topos_control_signal(topos, observed_depth=4, visited_volume=25)
training = st.topos_training_hints(signal)
adapter = st.topos_runtime_adapter(signal, request_options={"base_temperature": 0.8})

trainer = st.ZSpaceTrainer(z_dim=4, topos_control_gain=0.5)
trainer.step(st.z.metrics(speed=0.0, memory=0.0, stability=0.0, telemetry={"topos": signal}))

print("gradient bias:", training["gradient_bias_scale"])
print("runtime temperature:", adapter["request"]["temperature"])
```

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
