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

Release wheels include the Python payloads behind every `spiral-hf-*` and
`spiral-zspace-inference-distortion-*` console command. These commands do not
depend on a source checkout's `bindings/st-py/examples` directory; release
validation rejects a wheel when a direct or transitive CLI payload is missing.

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

Historical lineage manifests without a composite fine-tune replay identity stay
readable. Once a generation adopts that identity, however, each descendant must
carry a ready `adopted` or `enforced` contract with a newly observed composite
ID. Reusing or dropping the parent ID blocks the transition because the adapter
input and effective recipe are generation-scoped. Scale-up and the continuation
executor therefore remove the source `--expected-training-recipe-id` and
`--expected-finetune-replay-id`, record both source IDs as `reissued`, and expose
the new contracts through pending plans, attempts, postflight, and live status.

Make the chain stop itself when additional adapter generations no longer earn
their cost:

```bash
spiral-hf-adapter-chain runs/qwen2-study \
  --max-lineage-depth 6 \
  --target-eval-loss 2.40 \
  --min-eval-improvement 0.002 \
  --max-distortion-pressure-index 0.45 \
  --min-desire-stability 0.60 \
  --max-psi-total 0.80 \
  --plateau-patience 2 \
  --out runs/qwen2-study/promotion-chain.json \
  --require-continuation-ready
```

These gates are opt-in. `max lineage depth` stops at the selected depth,
`target eval loss` stops after reaching the target, and the plateau gate stops
after the configured number of consecutive generations have
`eval_before_loss - eval_after_loss` below the minimum. The optional geometry
gates stop when distortion pressure or maximum trainer psi rises above its
limit, or when mean desire stability falls below its minimum. All three
thresholds use the normalized `[0, 1]` range. Distortion evidence can come from
the run card's inference-distortion handoff; desire and psi evidence come from
trainer telemetry produced by `--trainer-telemetry`. When desire or psi gates
are active, the continuation executor automatically adds that switch and
removes a conflicting `--no-trainer-trace` from the next scale-up command. The
resulting telemetry command contract is sealed into the generation plan and
recomputed by preflight; a distortion-only gate leaves trainer telemetry
unchanged. After a gated generation exits, executor postflight reads the exact
trainer trace, requires at least one telemetry event and every configured
desire/psi aggregate, cross-checks them against the audited chain node, and
records content-addressed trace and evidence IDs. Missing trace evidence fails
postflight, while present evidence outside a policy threshold remains a valid
receipt and produces a normal continuation stop. Successful receipts are
digest-checked whenever executor state is loaded, so later trace mutation fails
closed. Once a geometry gate is configured, missing or out-of-range evidence
blocks continuation rather than silently passing. A depth-zero seed is allowed
to launch its first measured generation without telemetry; that child must then
provide every configured signal before another generation can run. A stopped
report keeps
`chain_ready=True` and
`continuation_artifacts_ready=True` for audit, but sets
`continuation_ready=False`; `spiral-hf-scale-up` then returns
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
  --max-distortion-pressure-index 0.45 \
  --min-desire-stability 0.60 \
  --max-psi-total 0.80 \
  --plateau-patience 2
```

After inspecting that state, execute one promoted generation:

```bash
spiral-hf-adapter-executor runs/qwen2-study \
  --output-root runs/qwen2-study/executor \
  --state runs/qwen2-study/executor/state.json \
  --max-lineage-depth 6 \
  --min-eval-improvement 0.002 \
  --max-distortion-pressure-index 0.45 \
  --min-desire-stability 0.60 \
  --max-psi-total 0.80 \
  --plateau-patience 2 \
  --run --max-generations 1
```

An audited root adapter at lineage depth zero does not need a promotion report:
the executor validates its lineage manifest and fingerprint, then adds
`--adapter-promotion-gate` to the child command so every later generation must
produce promotion evidence before it can become the selected tip.

Every audited child command also receives
`--expected-parent-adapter-id`, `--expected-parent-lineage-depth`, and
`--expected-root-adapter-id`. The bridge re-fingerprints adapter config and
weights before loading the model and immediately after load, then refuses to
touch the dataset or Trainer if either observation differs from the selected
chain tip. Executor plan, pending, attempt, postflight, and status artifacts
retain this input-identity contract and its observed evidence.

Scale-up separately content-addresses every local training input: the model
profile file, ordered train/validation files, inference-distortion probe or
sweep report, and a recursively fingerprinted resume checkpoint. The bundle ID
includes each input's role, order, content digest, byte size, and file count but
not its absolute path, so an unchanged study can move as one tree without
changing identity. The generated child command pins it with
`--expected-training-input-id`; the bridge verifies the bundle during preflight
and again after model load, before dataset materialization or Trainer work.
Ordinary artifact replay preserves the prior expectation instead of silently
accepting changed bytes, while an explicit resume-checkpoint override issues a
new bundle contract. Run cards, lineage transitions, scale-up artifacts, and
executor state/status retain both observations. Python can build or render the
same report with `st.hf_finetune_input_identity_report(...)` and
`st.hf_finetune_input_identity_lines(...)`. Hashing is intentionally fail-closed
and reads all covered bytes at both phases, so very large corpora or checkpoints
should budget that I/O rather than treating the gate as metadata-only.

Remote Hub corpora use a separate immutable dataset-source contract. Before
model weights load, the bridge resolves a requested dataset branch or tag to
the Hub's canonical repository id and 40-hex commit, then fingerprints that
repository commit together with the dataset config, train/eval split
expressions, and text column. For example, `wikitext@main` is rewritten to a
namespaced repository plus an immutable revision before `load_dataset(...)` is
called. The canonical launch command records both `--dataset-revision` and
`--expected-dataset-input-id`; the identity is checked again after dataset
load, and later scale-up/executor generations require the same parent/child
identity before promotion. Any config, split, column, repository, or commit
drift fails before model loading. Use
`st.hf_dataset_input_identity_report(...)` and
`st.hf_dataset_input_identity_lines(...)` to inspect the same contract. This
scope pins the Hub repository and logical selection, not mutable external URLs
that a custom dataset builder may fetch. Local `--train-file` corpora remain
covered by the byte-level local training-input identity above instead of being
double counted.

Once the requested splits are loaded and `--max-train-samples` /
`--max-eval-samples` are applied, the bridge also hashes every selected text
row in exact order. Split presence, row boundaries, whitespace, Unicode UTF-8
bytes, and the text-column name all contribute to this second identity; corpus
text itself is not retained in the report. The first metadata/training run
adopts the observed value as `--expected-dataset-materialization-id` in its
canonical launch command. Replays and later adapter generations enforce it
before tokenization or Trainer construction. This catches mutable external
builder downloads as well as changed streaming shuffle, validation fallback,
or selected-row content even when the Hub commit identity above still matches.
It applies to local and remote datasets because it fingerprints the rows that
will actually reach the tokenizer. Use
`st.hf_dataset_materialization_identity_report(...)` and
`st.hf_dataset_materialization_identity_lines(...)` for the same privacy-safe
contract; hashing cost is linear in the selected text bytes and is bounded by
the configured sample caps for streaming runs.

Training runs add a third, downstream dataset contract after tokenization,
block grouping, and `--max-eval-blocks` are complete. SpiralTorch hashes every
train/eval block in order, including all typed values in `input_ids`, `labels`,
`attention_mask`, and any model-specific columns. The report retains only
per-split digests and aggregate row/token/value counts. The first run adds
`--expected-tokenized-dataset-id` to its canonical launch command; replay and
adapter continuations verify it before model preparation or Trainer
construction. This catches tokenizer-library, special-token, grouping, mask,
label, and block-boundary drift even when the selected raw text identity and
model/tokenizer source identity are unchanged. Inspect or build the same report
with `st.hf_tokenized_dataset_identity_report(...)` and
`st.hf_tokenized_dataset_identity_lines(...)`. Use `--tokenize-only` to run the
whole dataset/tokenization audit without constructing Trainer; the resulting
run card rewrites its canonical replay command to `--train` and carries the
newly adopted tokenized identity into the real launch.

Immediately before `Trainer` construction, the bridge adds a separate effective
training-recipe identity. It reads the instantiated `TrainingArguments`, not
raw argv, so Transformers defaults and compatibility aliases are already
resolved. The payload covers optimizer and scheduler settings, effective batch
and accumulation, seed/data order, precision and checkpointing, dataloader and
evaluation control, applied full-FT/LoRA trainability, model dtype conversion,
Trainer checkpoint-resume state, the causal-LM collator, and the loss-guard
callback. Model, tokenizer, corpus, tokenized blocks, package/device runtime,
output paths, run names, and trace destinations remain in their own contracts;
moving an otherwise identical run directory therefore does not change the
recipe ID.

Use `--training-recipe-only` for a no-optimization adoption pass. It performs
the real model preparation and constructs effective `TrainingArguments`, writes
`training_recipe_identity` into the run card, then stops before callback or
`Trainer` construction. Its canonical launch command replaces that flag with
`--train` and adds `--expected-training-recipe-id`. The real launch then fails
before `Trainer(...)` if LR, schedule, batch, seed, precision, LoRA target/rank,
resume semantics, or training-control behavior drifted. Python callers can use
`st.hf_finetune_training_recipe_identity_report(...)` and
`st.hf_finetune_training_recipe_identity_lines(...)` directly.

At that same final boundary, SpiralTorch binds all independently verified
layers into one `finetune_replay_identity`: adapter input lineage when present,
local training inputs or the resolved Hub dataset source, exact selected text
rows, exact tokenized blocks, model/tokenizer runtime, software/device
execution, and the effective training recipe. The composite payload contains
only each component schema, applicability, and content-addressed ID, so report
paths and timestamps do not affect it. A `--training-recipe-only` adoption run
adds `--expected-finetune-replay-id` to its canonical `--train` command. The
individual expected-ID flags remain in that command to preserve earlier,
layer-specific failures; the composite is the final fail-closed check before
`Trainer(...)`. Use `st.hf_finetune_replay_identity_report(...)` and
`st.hf_finetune_replay_identity_lines(...)` when assembling or auditing the
same bundle outside the bridge.

Scale-up keeps these contracts strict without making corpus growth impossible.
If sample/eval shape stays unchanged, the child must reproduce both parent
dataset identities exactly. Selection changes such as `--max-train-samples`,
`--max-eval-samples`, or streaming validation size reissue both the selected
text and tokenized identities. A post-tokenization-only change such as
`--max-eval-blocks` or `--block-size` reissues only the tokenized identity and
continues enforcing the parent's raw-row digest. The planner records each
source/target flag value, and preflight independently recomputes the complete
command delta rather than trusting its reissue boolean. After the child run,
promotion-chain validation applies the same layer-specific boundary and exposes
each reissue on the transition. Content drift with no corresponding
command-shape change remains rejected.

An intentional scale-up always reissues the training-recipe layer because its
step horizon and often its batch/corpus plan are deliberately different. The
planner strips the parent's `--expected-training-recipe-id`, records the source
ID plus a `reissued` contract, and preflight rejects a command that accidentally
retains the parent ID. It also strips and reissues
`--expected-finetune-replay-id`, because changed recipe or row-selection layers
necessarily define a new complete run identity. The child adopts both newly
resolved IDs immediately before its own Trainer construction; unchanged
dataset, model/runtime, and execution identities remain independently enforced.

The model basis and tokenizer now form a second content-addressed runtime
contract. Remote base models record the resolved Hub commit plus stable config
and tokenizer semantics; tokenizer files shipped with an adapter and local
base-model config/weight files are hashed without embedding their absolute
paths. After config resolution, SpiralTorch forces that commit onto the
tokenizer and model loaders. A first audited generation adopts the observed
bundle ID and writes `--expected-runtime-input-id` into its canonical launch
command; later scale-up and executor generations enforce the same ID before
model weights load, repeat the check after load, and require parent/child
continuity in the promotion-chain transition. The reports are available as
`st.hf_causal_lm_runtime_identity_report(...)` and
`st.hf_causal_lm_runtime_identity_lines(...)`, while
`st.load_hf_causal_lm_artifact(..., expected_runtime_identity_id=...)` raises
`st.HfCausalLmRuntimeIdentityError` on drift. Local full-model identity checks
read all covered weight bytes before and after load, so large local models must
budget two additional sequential reads; remote models use the immutable Hub
commit rather than downloading and hashing weights twice.

The software and device basis forms a third fail-fast identity. Before model
weights load, the bridge fingerprints the installed SpiralTorch wheel and HF
stack from normalized distribution `RECORD` rows, Python/OS/architecture,
stable SpiralTorch backend reports, Torch CUDA/MPS capabilities, and selected
compute-affecting environment variables. Absolute install paths, hostnames,
timestamps, free-memory values, and SpiralTorch's intentionally volatile
runtime `BUILD_FINGERPRINT` are excluded. The report is repeated after model
load; a first generation adopts the observed ID and canonicalizes it as
`--expected-execution-input-id`, while later scale-up/executor generations
enforce it and require parent/child continuity before promotion. Use
`st.hf_finetune_execution_identity_report(...)` and
`st.hf_finetune_execution_identity_lines(...)` for the same public surface.
Missing wheel `RECORD` evidence is fail-closed rather than silently reduced to
package version checks.

The continuation runtime entrypoint does not depend on the repository staying
at its original absolute path. Known `hf_finetune_bridge.py`, legacy
`hf_gpt2_finetune_bridge.py`, and `spiral-hf-finetune` prefixes are rewritten
to the current interpreter's
`python -m spiraltorch.hf_finetune_entrypoint`. The state keeps the original
prefix as provenance, preflight verifies that the packaged module is
importable, and unknown custom launchers are preserved rather than guessed.
Executor plan, attempt, and status lines expose this as
`runtime=portable_module`. New run cards also record `launch_cwd`; scale-up and
executor plans resolve relative model-config, corpus, validation, distortion,
and checkpoint paths against that source directory before preflight. Explicit
absolute paths remain unchanged, unresolved inputs fail closed, and command
artifacts retain the resolution report. Runs launched through the installed
entrypoint write the canonical module prefix and working directory back into
their run cards, so subsequent plans are portable and idempotent rather than
repeatedly recovering a historical script path or depending on the caller's
current directory.

The foreground executor seals every ready generation plan before execution. Run
the command once without `--run`, review the emitted `plan_id`, and then bind the
first generation to that exact command, parent adapter, lineage/output,
continuation policy, scale-up settings, working directory, and environment
override digest. Environment values are never persisted; the state records only
the sorted override keys, their count, and the digest:

```bash
spiral-hf-adapter-executor runs/qwen2-study \
  --output-root runs/qwen2-study/executor \
  --state runs/qwen2-study/executor/state.json \
  --max-lineage-depth 6 \
  --max-steps 1024

spiral-hf-adapter-executor runs/qwen2-study \
  --output-root runs/qwen2-study/executor \
  --state runs/qwen2-study/executor/state.json \
  --max-lineage-depth 6 \
  --max-steps 1024 \
  --run --max-generations 1 \
  --require-pending-plan \
  --expected-plan-id sha256:<reviewed-plan-id>
```

If any bound input changes, `--run` stops with
`action=review_generation_plan`, preserves the reviewed pending plan, and
records the new candidate separately. Re-run the changed command without
`--run` to adopt it intentionally; the state retains both scale-up and
generation-plan history. Legacy pending plans remain readable but must also be
sealed by one plan-only invocation before execution. The Python API exposes the
same check through
`hf_adapter_continuation_executor_generation_plan_report(...)` and its concise
line formatter. `require_pending_plan` and `expected_plan_id` apply to the first
generation in one invocation. When that invocation reaches `max_generations`,
the executor audits the newly promoted tip, derives the next generation, and
persists its sealed pending plan before returning `resume_executor`. A policy
stop or failed preflight still ends without advertising an automatically
resumable boundary.

For a long run, use the same executor policy in a detached process:

```bash
spiral-hf-adapter-executor runs/qwen2-study \
  --output-root runs/qwen2-study/executor \
  --state runs/qwen2-study/executor/state.json \
  --max-lineage-depth 6 \
  --min-eval-improvement 0.002 \
  --plateau-patience 2 \
  --run --max-generations 1 \
  --require-pending-plan \
  --expected-plan-id sha256:<reviewed-plan-id> \
  --detach --no-tee-output
```

The command returns success only after it observes a new executor invocation
and verifies that the detached PID owns the output root's single-writer lock.
It writes durable launch history to
`spiraltorch-hf-adapter-continuation-executor-launch.json` under the output
root and a separate owner-only launcher log under `executor-logs/`. Repeating
the command while that owner is live is idempotent. A handoff timeout remains
visible as a nonzero start result rather than guessing that training began;
raise `--detach-handoff-timeout-seconds` for a slower environment and inspect
the launch status before retrying. Use `--launch-state` to place the history at
another path.

Resume the exact detached policy directly from its launch artifact instead of
reconstructing the original command:

```bash
spiral-hf-adapter-executor-resume \
  runs/qwen2-study/executor/spiraltorch-hf-adapter-continuation-executor-launch.json \
  --plan

spiral-hf-adapter-executor-resume \
  runs/qwen2-study/executor/spiraltorch-hf-adapter-continuation-executor-launch.json
```

The read-only plan requires a healthy terminal executor whose durable action is
`resume_executor`, an absent or stale local executor lock, the original working
directory, and a ready sealed pending generation. Every new detached launch
stores a SHA-256 replay contract binding its exact executor argv, immutable
policy argv, output root, state path, and cwd. At a generation boundary, resume
removes only the old `--expected-plan-id` / `--require-pending-plan` gate,
rebinds both to the pending generation's ID, and rechecks that ID under the
launch lock immediately before process creation. The launched argv remains
fully hashed, while all non-gate options must match the source launch exactly.
Concurrent, stale, or policy-mutating plans therefore cannot start a different
invocation. Existing launch artifacts without the newer replay digest remain
readable when their redundant stored child command exactly matches their argv,
but an old generation-limit state without a sealed pending plan is not resumed
automatically. Re-run its original full executor command once without `--run`,
`--detach`, `--expected-plan-id`, or `--require-pending-plan` to seal the pending
generation, then request resume again. Policy-stopped, live, remote, unverified,
modified, or cwd-missing histories fail closed. To
intentionally change policy, issue a new full
`spiral-hf-adapter-executor ... --detach` command instead of replaying.
An operator stop observed after preflight but before a child process starts
keeps its freshly verified planned generation, so a strict resume can bind to
it directly. An earlier stop, a stop after a promotion boundary, or a cancelled
partial child has no reusable pending plan;
after resolving any output quarantine, seal a fresh plan-only invocation before
resuming.

For bounded unattended continuation, inspect one supervision decision and then
detach the supervisor over the same launch artifact:

```bash
spiral-hf-adapter-executor-supervise \
  runs/qwen2-study/executor/spiraltorch-hf-adapter-continuation-executor-launch.json \
  --plan

spiral-hf-adapter-executor-supervise \
  runs/qwen2-study/executor/spiraltorch-hf-adapter-continuation-executor-launch.json \
  --detach \
  --max-resumes 5 \
  --poll-interval-seconds 5 \
  --timeout-seconds 0
```

For the normal operational path, the integrated runtime command folds the
executor, launcher, supervision decision, supervisor, and supervisor launcher
into one fail-closed snapshot. It can also perform the initial idempotent
handoff:

```bash
spiral-hf-adapter-executor-runtime \
  runs/qwen2-study/executor/spiraltorch-hf-adapter-continuation-executor-launch.json

spiral-hf-adapter-executor-runtime \
  runs/qwen2-study/executor/spiraltorch-hf-adapter-continuation-executor-launch.json \
  --reconcile \
  --max-resumes 5 \
  --poll-interval-seconds 5
```

The read-only report verifies that every discovered artifact points back to the
same executor launch, executor state, output root, supervisor state, and
supervisor launch history. Missing optional control artifacts describe an
`unmanaged_*` runtime; malformed or cross-wired artifacts become `invalid` and
cannot be reconciled.

Reconciliation never silently crosses a completed supervisor run. After an
operator stop, timeout, interruption, or exhausted resume budget, the command
returns `operator_restart_required`. Starting another bounded controller then
requires explicit intent:

```bash
spiral-hf-adapter-executor-runtime \
  runs/qwen2-study/executor/spiraltorch-hf-adapter-continuation-executor-launch.json \
  --reconcile \
  --restart-supervisor \
  --max-resumes 5
```

The detach call returns only after a new supervisor run and its single-owner
lock agree on the child PID. Repeating the command while that owner is verified
is an idempotent `already_running` no-op. Observe both layers independently:

```bash
spiral-hf-adapter-executor-supervisor-launch-status \
  runs/qwen2-study/executor/spiraltorch-hf-adapter-continuation-executor-supervisor-launch.json

spiral-hf-adapter-executor-supervisor-status \
  runs/qwen2-study/executor/spiraltorch-hf-adapter-continuation-executor-supervisor.json
```

To stop only the controller while leaving its current detached executor alone,
write a run-targeted cooperative request:

```bash
spiral-hf-adapter-executor-supervisor-stop \
  runs/qwen2-study/executor/spiraltorch-hf-adapter-continuation-executor-supervisor.json \
  --reason operator_pause
```

The stop command never signals a recorded PID. It writes a mode-`0600` control
artifact only when the supervisor state and live lock identify the same run;
the next poll acknowledges it as a healthy `stopped` boundary. A later detach
starts a new run ID and cannot consume the old request. Omit `--detach` when a
foreground controller is preferable.

The supervisor waits while a healthy detached owner is running and automatically
replays only an executor whose exact durable reason is
`max_generations_per_invocation_reached` and whose next generation plan verifies
as sealed. Its supervision decision, transition, detached launch, resume event,
status line, and integrated runtime report all carry the rebound generation plan
ID. A clean `stop_requested` exits as a healthy operator pause (cancelled
partial output still blocks for quarantine), continuation-policy stop exits
completed, and quarantined recovery requires an explicit manual resume.
Failed/unhealthy launches, remote or unverified owners, missing/invalid plans,
invalid handoff identity, and changed replay contracts fail closed.
`--max-resumes` bounds launches per supervisor invocation (default `1`), while
`--timeout-seconds 0` means no wall-clock timeout. A separate
single-owner supervisor lock prevents duplicate controllers, and atomic state
plus transition/resume history is written by default to
`spiraltorch-hf-adapter-continuation-executor-supervisor.json` under the output
root; use `--state` to place it elsewhere. Restarting a supervisor marks its
unfinished prior run interrupted, then derives the next action from current
launch/executor artifacts rather than replaying stale in-memory intent.
Detached launch history and logs default to
`spiraltorch-hf-adapter-continuation-executor-supervisor-launch.json` and
`supervisor-logs/` under the same output root; use `--launch-state` to relocate
the history artifact.

Alternatively, repeat the full command to change explicit settings, or raise
`--max-generations` for several synchronous generations in one invocation. The
executor holds a single-writer lock for the output root,
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

Inspect the detached launcher and its executor together, also read-only:

```bash
spiral-hf-adapter-executor-launch-status \
  runs/qwen2-study/executor/spiraltorch-hf-adapter-continuation-executor-launch.json \
  --require-healthy
```

Stop a live invocation cooperatively instead of signalling its recorded PID:

```bash
spiral-hf-adapter-executor-stop \
  runs/qwen2-study/executor/state.json \
  --reason "operator maintenance"
```

The stop command writes an owner-only, invocation-scoped request under
`executor-control/`; it never sends a signal to a PID whose identity it cannot
prove. The executor launches real training in an isolated process group,
observes the request even when training is silent, asks the owned group to
terminate, and escalates after a bounded grace period so Trainer/data-loader
workers are not left behind. An in-flight generation is recorded as
`cancelled`. If the request arrives after the command has completed, the
executor verifies and promotes that finished adapter, then stops before
launching the next depth.
Repeated stop commands are idempotent. A cancelled generation that left a
partial output directory remains fail-closed and is reported as
`cancelled_output_present`, including the exact unresolved attempt ID. Plan a
non-destructive quarantine before resuming:

```bash
spiral-hf-adapter-executor-quarantine \
  runs/qwen2-study/executor/state.json \
  --attempt-id generation-attempt-... \
  --plan

spiral-hf-adapter-executor-quarantine \
  runs/qwen2-study/executor/state.json \
  --attempt-id generation-attempt-... \
  --reason "operator inspected partial generation"
```

Quarantine never deletes the partial output. Under the executor's
single-writer lock it atomically moves the exact attempt output into a sibling
`<output-root>.executor-quarantine/` directory, outside recursive adapter-chain
discovery rooted at that executor output directory. The state retains
source/destination paths, prior attempt status,
reason, and a SHA-256 tree-metadata digest. Attempt-ID matching prevents stale
operator commands from moving a different generation; live or unverified lock
owners fail closed, stale local locks are reaped, and repeating a completed
quarantine revalidates the recorded metadata before returning idempotently.
FIFO, socket, and device entries fail closed rather than being moved.
The executor itself also blocks on a pending or inconsistent quarantine intent,
so bypassing the status CLI cannot resume training inside a half-finished move.

The same command closes an executor-crash interruption when the recorded
attempt is still `running`. SpiralTorch issues an interruption claim only for a
local subprocess whose leader PID is absent and whose isolated POSIX process
group is also absent. The claim is rechecked under the executor lock immediately
before the move and persisted in the attempt, intent, and final resolution.
Live/PID-reused processes, surviving process-group members, remote hosts,
custom runners, and platforms without process-group observation remain
fail-closed. Status reports `quarantine_interrupted_output` only when this scope
proof is ready; otherwise it reports the exact inspection/wait action.
`spiral-hf-adapter-executor-status --require-healthy`
then reports `output_quarantined` with `resume_executor`, after which the same
foreground or `--detach` executor command can recreate that generation.
Python can issue the same request with
`st.request_hf_adapter_continuation_executor_stop(...)` and load its durable
evidence with `st.load_hf_adapter_continuation_executor_stop_request(...)`.
Use `st.hf_adapter_continuation_executor_output_quarantine_report(...)` and
`st.quarantine_hf_adapter_continuation_executor_output(...)` for the equivalent
recovery flow.

The status surface reports state age, active attempt, local PID liveness,
single-writer lock presence, output presence, log size, and a pending stop
request. `stopping` is a healthy transition while the owner drains the child.
PID liveness is an observation rather than a process-identity guarantee;
remote-host and legacy attempts are reported as unverified instead of being
guessed healthy.
`--require-healthy` also rejects a missing active lock, durable subprocess log,
or promoted output. Python can drive the same state machine with
`st.run_hf_adapter_continuation_executor(...)` and inspect it with
`st.load_hf_adapter_continuation_executor(...)` or
`st.hf_adapter_continuation_executor_status_report(...)`. Detached callers can
use `st.launch_hf_adapter_continuation_executor(...)`, load launch history with
`st.load_hf_adapter_continuation_executor_launch(...)`, and combine launcher
and executor health with
`st.hf_adapter_continuation_executor_launch_status_report(...)`. Durable replay
is available through
`st.hf_adapter_continuation_executor_resume_report(...)` and
`st.resume_hf_adapter_continuation_executor(...)`. Bounded supervision is
available through
`st.hf_adapter_continuation_executor_supervision_report(...)` and
`st.supervise_hf_adapter_continuation_executor(...)`; load its durable history
with `st.load_hf_adapter_continuation_executor_supervisor(...)`. Detach with
`st.launch_hf_adapter_continuation_executor_supervisor(...)`, inspect ownership
through `st.hf_adapter_continuation_executor_supervisor_status_report(...)` or
`st.hf_adapter_continuation_executor_supervisor_launch_status_report(...)`, and
request a controller-only stop with
`st.request_hf_adapter_continuation_executor_supervisor_stop(...)`. The unified
Python entrypoints are
`st.hf_adapter_continuation_executor_runtime_report(...)` and
`st.reconcile_hf_adapter_continuation_executor_runtime(...)`.

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
is recorded before launch. The emitted command also pins the expected parent
adapter ID/depth/root ID, and the child verifies that content fingerprint both
before and immediately after model load. Promotion-required children also get a
complete launch contract: `--train`, LoRA mode, before/after evaluation, and the
promotion gate are normalized together. Source metadata/tokenize/recipe-only
switches, `--no-eval-after-train`, or `--eval-after-train-policy never` are
removed safely, while preflight rejects any later command/metadata tampering or
an incomplete generation-change policy.
`spiral-hf-scale-up sweep-report.json`
therefore continues the winning weights rather than merely rerunning their
configuration. Pass `--adapter-continuation replay` for the old configuration
replay behavior, or `--adapter-continuation continue` to require adapter
continuation for a non-gated LoRA run. Exact `--resume-from-checkpoint` remains
the stronger path in `auto` mode; combining it with explicit `continue` is
rejected as ambiguous. Scale-up preflight checks local adapter config/weights,
input-output separation, lineage adapter ID/depth, and promotion report
ID/readiness. When the source run card records `launch_cwd`, relative model
config, corpus, validation, distortion, and checkpoint arguments are resolved
against that directory and recorded in `command_path_resolution`, allowing a
wheel-installed executor to launch from another working directory without
silently changing its inputs. The sanitized
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
