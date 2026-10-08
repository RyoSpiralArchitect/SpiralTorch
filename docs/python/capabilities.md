# Python capability reference

[Documentation index](../README.md) | [Project entry](../../README.md)

This is the detailed source-tree reference, moved from the README.
Examples retain their individual feature, device, and optional-dependency requirements.
A source-tree API is not a claim that an older PyPI wheel exposes it.
Run repository commands from the repository root.

## Contents

- [Install](#install)
- [Resident NN Inference](#resident-nn-inference)
- [Rust-owned protocol catalog](#rust-owned-protocol-catalog)
- [Shared FFT numerics](#shared-fft-numerics)
- [Stable LayerNorm statistics](#stable-layernorm-statistics)
- [Differentiable LayerNorm](#differentiable-layernorm)
- [Integer Embeddings (Source Builds)](#integer-embeddings-source-builds)
- [What's included](#whats-included)
- [Rust-to-Python exposure queue](#rust-to-python-exposure-queue)
- [Tokenizerless FT diagnostics](#tokenizerless-ft-diagnostics)
- [Rust-owned token periodicity](#rust-owned-token-periodicity)
- [Rust-owned stochastic Schrodinger dynamics](#rust-owned-stochastic-schrodinger-dynamics)
- [Blinded semantic review](#blinded-semantic-review)

## Install

```bash
pip install -U spiraltorch
```

The published wheel is WGPU-first with CPU fallback. In plain Python terms,
start with these handles:

- `spiraltorch.Tensor` for dependency-light native tensors.
- `spiraltorch.AutogradTensor` for a thin handle over the Rust-owned immutable
  reverse-mode graph. Python never rebuilds its derivative or accumulation
  rules.
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
- `spiraltorch.zspace_runtime_protocol_catalog()` when you need the exact,
  content-addressed Rust/Python/WASM surface for generation evidence,
  periodicity analysis, repetition control, and blinded semantic review before
  persisting or dispatching an artifact.

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
print("route contract:", runtime["runtime_device_route_contract_version"])
PY
```

## Resident NN Inference

Wheels built from this source expose `Linear.inference_plan(shape)` and
`Sequential.inference_plan(shape)`. Existing Linear/GELU chains can keep
weights and intermediate activations on WGPU instead of reading back every
layer:

```python
import spiraltorch as st

model = st.nn.Sequential()
model.add(st.nn.Linear(4, 7, name="up"))
model.add(st.nn.Gelu())
model.add(st.nn.Linear(7, 3, name="down"))
plan = model.inference_plan([2, 5, 4])
gpu = plan.compile_wgpu()
gpu.upload_values([0.25] * 40)
gpu.dispatch()
snapshot = gpu.snapshot()
print(snapshot.shape, snapshot.read_values())  # (2, 5, 3), one final readback
```

The same Rust-owned plan can be exported with `plan.to_json()` and imported
by `st.nn.InferencePlan.from_json(...)` or the browser's
`InferencePlan.fromJson(...)`. Source model updates require a new plan.
The code above is explicit inference, not a general N-D Tensor. The same plan's
`compile_training_wgpu()` creates a mutable GPU workspace for mean-MSE, VJP and
transactional plain SGD; see [resident training](../resident_nn_training.md)
for Python/browser use and weight-only handoff. Current-source bindings are
required; older published wheels may not have these APIs.
For mixed `Scaler`/`Relu`/`Linear`/`Gelu` graphs, use
`plan.compile_graph_training_wgpu(gradient_policy="exact")`. It keeps all
parameters and intermediate VJPs resident, exposes raw/effective gain gradients,
and exports the same v2 weight-only plan for browser training and return resume.
See [graph training clients](../resident_graph_training.md#python-and-browser-clients).
CPU-only wheels support plan transport but reject GPU compilation rather
than silently falling back. See the [resident NN guide](../resident_nn_inference.md)
for browser use, validation, and the retained PyTorch comparison boundaries.

The [original model can now accept `WgpuTensor` directly](../module_resident_forward.md):
`y = model(x_gpu)` caches the same Rust graph, follows parameter updates, and
returns an owning GPU tensor. Host `Tensor` inputs keep the existing behavior.
No automatic CPU fallback or per-layer host readback is introduced.

## Rust-owned protocol catalog

The admission-certified catalog is generated and replayed by `st-core`; Python
does not maintain a second list of protocol versions or validation rules:

```python
import spiraltorch as st

catalog = st.zspace_runtime_protocol_catalog()
assert st.validate_zspace_runtime_protocol_catalog(catalog) == catalog

for protocol in catalog["protocols"]:
    print(protocol["name"], protocol["semantic_owner"])
    for surface in protocol["clients"]:
        print(
            " ",
            surface["client"],
            surface["normal_admission"]["profile"],
            surface["normal_admission"]["limits"],
        )
```

The current catalog covers held-out generation evidence, bounded token
periodicity, stochastic Schrodinger real and complex forward/VJP, repetition-unlikelihood
planning, and the complete blinded semantic-review lifecycle. Catalog v4
records a normal-admission profile and
guarantee for every client surface; serialized Python/WASM surfaces also carry
Rust-owned byte/node/depth limits, while typed Rust admission has no serialized
budget. Trusted historical replay is always opt-in, accepts only already trusted
local evidence, and is exposed by Rust/Python only. Browser/WASM catalog entries
expose only bounded JSON routes. WASM object helpers remain trusted-local
convenience transports and are intentionally not certified as hostile-input
boundaries.

For multi-step evolution that retains the imaginary state and learns through
both quadratures, see the [complex dynamics guide](../zspace_complex_dynamics.md)
and `examples/zspace_complex_trajectory.py`.

Normal catalogued Python routes admit `dict`-backed mappings and `list`/`tuple`
sequences, including subclasses through base-container descriptors that inspect
the concrete C type without consulting `__class__`. Arbitrary `Mapping` or
`Sequence` implementations are rejected before their Python type, enumeration,
or comparison hooks can run; Rust then applies the protocol byte/node/depth and
semantic admission budgets. Opt-in trusted-legacy replay keeps its documented
historical budget exemptions, but still rejects active outer containers and
remains unsuitable for remote input.

`describe_runtime_devices()` and HF preflight orchestrate observations in Python, but committed
SpiralTorch probes are validated and projected into route evidence only by
`st-core::backend::runtime_route`. In particular, an MPS placeholder can remain honestly
`native_ready = false` while being `route_ready = true` through its WGPU surrogate. Use
`evaluate_runtime_device_route_from_probes(...)` when an orchestrator has full committed probes;
use `evaluate_runtime_device_route(...)` only for external or legacy evidence rows. Use
`validate_runtime_device_route_contract(...)` to validate a persisted payload or replay it
against its original request. The v5 contract retains canonical evidence and commits both the
request and Rust-derived output with SHA-256; reports that disagree about the same effective
backend fail closed. It also owns the payload-level `runtime_readiness` projection: explicit required
backends use an all-required gate, while an ungated request accepts any ready route. Missing
evidence remains `native_readiness = "unknown"` or `route_readiness = "unknown"`, and the
boolean `runtime_ready` projection remains fail-closed.

`describe_runtime_devices(...)` also appends the complete source `reports` for diagnostics.
Those rows are Python transport metadata and are not committed; the canonical Rust-owned
`evidence` field is the replayable source for route decisions. Python never copies readiness
fields out of native probes on this path. If a diagnostic batch mixes committed probes with
legacy or error rows, committed probes stay on the validated Rust ingress. Rust admits only
explicit, non-ready error rows into the committed diagnostic projection, so collection failures
remain visible without allowing uncommitted rows into executable selection; the whole batch is
never downgraded to compatibility semantics.
Malformed probe envelopes are recognized from their stable identity and commitment markers, then
rejected rather than reclassified as legacy evidence when `kind` or another contract field drifts.
The probe-only `route_evidence` transport key is also reserved for this validated ingress; external
compatibility rows provide readiness fields directly instead of wrapping uncommitted evidence.

Build a graph with the same `spiraltorch.autograd.v1` contract used by direct
Rust and browser clients:

```python
import spiraltorch as st

x = st.AutogradTensor.variable(st.Tensor((1, 3), data=[1.0, 2.0, -1.0]))
loss = x.hadamard(x).add(x.scale(3.0)).sum()
receipt = loss.backward()

print(x.grad().tolist())
print(receipt["contract_version"], receipt["semantic_owner"])
```

`backward()` accepts only scalar outputs; pass a shape-matched `Tensor` seed
for an explicit vector-Jacobian product. Repeated calls accumulate, while
`zero_grad_graph()` clears the complete reachable graph. These are Rust
invariants, not Python-side conventions. Use
`output.vector_jacobian_product(input, seed)` for a side-effect-free VJP; it
does not read or mutate accumulated gradients and returns zero when `input` is
disconnected.

Repeated projections can reuse a Rust-owned packed RHS without detaching its
source graph:

```python
weights = st.AutogradTensor.constant(st.Tensor(3, 2, [1, 0, 0, 1, 1, 1]))
packed = weights.prepack_rhs()  # AutogradPackedRhs; pack once for frozen weights
hidden = st.AutogradTensor.variable(st.Tensor(1, 3, [1, 2, 3]))
projected = hidden.matmul_prepacked(packed)
projected.sum().backward()
print(projected.value().tolist(), hidden.grad().tolist())
```

Trainable and non-leaf RHS nodes retain their original gradient paths too.
`source_id()`, `shape()` and `requires_grad()` describe that immutable snapshot.
After `AutogradSgd.step()`, fetch the new parameter and call `prepack_rhs()` again
if that RHS was updated; an old packed handle deliberately keeps the old value
and graph. This uses the existing Rust Tensor **prepacked Auto dispatch**, which
can choose a different backend from ordinary matmul. Packing is not a promise
of GPU residency or faster training; measure the actual workload.

Native graphs also expose `add_row(bias)`, `relu()`, `gelu()` (tanh
approximation), and `row_softmax()`. Bias gradients sum over rows; any averaging
belongs to the loss. See [the nonlinear learning example](../../examples/autograd_xor.py)
for a complete training loop without PyTorch or NumPy dependencies.

For classification, feed **logits**, not probabilities, directly into the
Rust-owned loss:

```python
logits = st.AutogradTensor.variable(st.Tensor(2, 3, [2., 0., -1., 0., 2., -1.]))
loss = logits.cross_entropy_with_logits([0, 1], label_smoothing=0.05)
loss.backward()
print(loss.item(), logits.grad().tolist())
```

`reduction="mean"` averages non-ignored rows; `ignore_index=-100` masks labels.
`"sum"` and `"none"` are also available. An all-ignored mean batch raises an
error. `row_log_softmax()` and the Tensor forward/VJP methods share the same
stable CPU kernels. `st.nn.CrossEntropyWithLogits` plugs into `ModuleTrainer`
using `(samples, 1)` integral target Tensors; strict WGPU execution is rejected
for this host Tensor route. For GPU-resident class-last logits, the same loss
now exposes [`evaluate_resident(logits, labels)`](../module_resident_classification.md)
with owning loss/gradient outputs that seed a resident graph learner.
Existing `SoftmaxCrossEntropy` remains the probability
loss and is **not** renamed. See the [classification contract](../autograd_contract.md#classification-from-logits)
and [runnable multiclass fixture](../../examples/autograd_classification.py).

Leaves and accumulated gradients are protected Rust snapshots. `value()` and
`grad()` expose read-only versioned DLPack storage; legacy export copies rather
than exposing writable graph memory. For an ordinary Tensor, `snapshot()`
explicitly isolates mutable aliases, and `is_snapshot()` reports that state.
Normal `from_dlpack` Tensor sharing is unchanged. Snapshot capture is not a
PyTorch autograd-graph transfer.

## Shared FFT numerics

Version 0.4.24 corrects radix-4 bin ordering and transform direction in
`st-frac::fft`, shared by native Python, WASM, Canvas, and temporal fusion.
The existing mixed radix-2/4 path is retained, without a Python DFT fallback.

```python
import spiraltorch as st

spectrum = st.frac.fft_real([1.0, 2.0, 3.0, 4.0])
assert spectrum == [(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0), (-2.0, -2.0)]
restored = st.frac.fft_complex32(spectrum, inverse=True)
```

Forward uses a negative exponent; inverse divides by the signal length.
Positive power-of-two lengths are accepted, including the length-one identity.
The root `st.fft_*` aliases use the same implementation. Earlier nontrivial
FFT-derived results from these paths should be recomputed, because origin-impulse
tests alone missed incorrect bin order and phase. This is not WebGPU dispatch.

## Stable LayerNorm statistics

Version 0.4.25 replaces cancellation-prone raw second moments in affine
LayerNorm. CPU uses centered f64 moments; WGPU uses a centered, scaled Welford
reduction without a CPU fallback. `nn.LayerNorm.backward()` shares the Rust
statistics rather than reconstructing a rounded f32 mean.

```python
import spiraltorch as st

x = st.Tensor(1, 4, [10000.0, 10001.0, 10002.0, 10003.0])
normalized, inverse_std = x.layer_norm_stats(epsilon=1e-5)
assert normalized.shape() == (1, 4)
assert inverse_std.shape() == (1, 1)
```

`layer_norm_stats` is a CPU helper returning two native tensors, with f64
intermediates and finite f32 outputs. It accepts logical column-major inputs,
preserves empty row batches, and rejects zero-column tensors, non-finite input,
invalid epsilon, unrepresentable inverse standard deviations, or constant rows
with zero epsilon. Affine forward keeps its existing backend selection and
fused residual-add API. Recheck earlier results on large-offset inputs; fixing
the operator does not by itself establish an LLM/FT quality improvement.

## Differentiable LayerNorm

Version 0.4.27 adds affine LayerNorm to the Rust-owned reverse-mode graph:

```python
import spiraltorch as st

x = st.AutogradTensor.variable(st.Tensor(2, 3, [0, 1, 2, 2, 1, 0]))
gamma = st.AutogradTensor.variable(st.Tensor(1, 3, [1, 1, 1]))
beta = st.AutogradTensor.variable(st.Tensor.zeros(1, 3))
x.layer_norm_affine(gamma, beta, epsilon=1e-5).sum().backward()
assert beta.grad().tolist() == [[2, 2, 2]]
```

Input, gamma and beta can all learn. Affine VJPs sum rows without hidden batch
averaging. Frozen parents skip unused gradients, shared parents accumulate
every path, and failed backward passes preserve existing gradients.
Plain `Tensor.layer_norm_affine_backward(gamma, upstream, epsilon=...)` returns
the three gradients without constructing a graph. Both use the shared Rust
CPU/f64 VJP; forward retains Tensor backend selection. The explicit Rust WGPU
backward path remains a hybrid of CPU moments and GPU utilities, not a fused
backward shader. Native `nn.LayerNorm` reuses the core with its existing
parameter-only averaging policy.

The repository's `examples/autograd_layer_norm.py` runs a 400-step native SGD
fixture. This checks learning mechanics, not LLM/FT quality.

## Integer Embeddings (Source Builds)

```python
import spiraltorch as st

table = st.AutogradTensor.variable(st.Tensor(3, 2, [1, 2, 3, 4, 5, 6]))
table.gather_rows([2, 0, 2]).sum().backward()
assert table.grad().tolist() == [[1, 1], [0, 0], [2, 2]]
```

`gather_rows` captures exact nonnegative integer IDs. Repeated IDs sum gradients;
there is no implicit batch average. `scatter_add_rows(ids, output_rows)` is the
transpose operation and also supports autograd. Invalid IDs, floats and booleans
are rejected rather than repaired. Graph semantics and both VJPs live in Rust.
Plain `Tensor` methods additionally accept `backend="cpu"`, `"auto"` (CPU), or
`"wgpu"` (strict). WGPU carries integer u32 IDs and groups scatter contributions
without float atomics; CPU sums in f64, WGPU in f32. Graph VJPs remain CPU.
Legacy `nn.Embedding` retains its float-token repair and batch-average policy
but delegates its numerical operations to these kernels.

Run `python examples/autograd_tied_embedding.py` for a 400-step tied-weight
token-transition fixture. This verifies mechanics, not language-model quality.
These APIs are not in the published 0.4.27 wheel yet.

## What's included

- `Tensor`, `AutogradTensor`, `ComplexTensor`, and `OpenTopos` for dependency-free
  geometry experiments.
- Native neural layers via `spiraltorch.nn`—`Linear`, `Embedding`,
  `Sequential`, losses, `ModuleTrainer`, `LoraLinear`, and `ZSpaceProjector`.
- Checked checkpoint handoff helpers for exact or key-mapped `state_dict`
  reports, subset loads, overlap resize/projection preflight, and HF-style
  checkpoint presets.
- `LanguageWaveEncoder` + `Hypergrad` so Python callers can stream Z-space
  text, accumulate gradients, and project back into the Poincaré ball.
- `zspace_temperature_control(...)` for atomic stateful entropy, Z-feedback,
  scale, and gradient-temperature transitions. Python transports `config` and
  `state`; `st-core::inference::temperature_control` validates the request,
  computes every adjustment, and returns the auditable `next_state`.
- `zspace_parameter_trajectory(...)` for Rust-owned factorization of an observed
  HF learning-rate control into raw, dose-matched constant, and dose-normalized
  schedules. `validate_zspace_parameter_trajectory(...)` recomputes persisted
  reports in Rust; the [four-arm ablation guide](../hf_zspace_optimizer_ablation.md)
  shows the matched Trainer workflow, fail-closed comparator, and audited
  three-seed loss-guard result with its non-efficacy boundary.
- `zspace_concept_diffusion(...)` for labelled probability-simplex heat flow.
  Rust owns observation blending, Z-bias tilt, symmetric conductivity, CFL
  substeps, and entropy/Dirichlet invariants; Python only carries graph state.
- `zspace_imaginary_time_schrodinger(...)` for positive-amplitude evolution under
  a Rust-owned Hermitian graph Hamiltonian, including spectral step bounds,
  potential-gauge normalization, and Rayleigh-energy audit fields. The payload
  reports its current `f64_cpu` execution backend and WGPU route blocker.
- `zspace_stochastic_schrodinger_forward(...)` plus
  `zspace_stochastic_schrodinger_vjp(...)` for a content-addressed real-time
  stochastic transition and analytic input/potential VJP. The standard-normal
  samples and configuration are fixed replay witnesses; Python never samples
  hidden noise or accepts an external phase for backward.
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
  `observe_runtime_device_probe`, `probe_gpu_path`) that
  reuse the same heuristics as the Rust executors. `RankPlan.contract()` exposes
  the validated Rust-owned shape, device capabilities, rich choice, and frozen
  execution policy; invalid dimensions and capability overrides fail closed
  instead of being clamped in Python. `spiralk_context()` and
  `rewrite_with_spiralk()` also delegate context construction, algorithm/mode
  interpretation, and override validation to that Rust contract.
  `observe_runtime_device_probe(...)` is the lower-level live entrypoint: Python
  supplies only a requested backend, capability overrides, and workload hints;
  Rust alone selects any MPS surrogate and commits requested/effective runtime
  evidence. Persisted v1 requests remain available for validation and replay.
  Workload-specific kernel readiness follows the same boundary:
  `evaluate_runtime_execution_plan(..., component_workloads=...)` embeds a committed
  Rust `runtime_component_capability_observation` contract. Its `ready_proof` is
  produced by `st-tensor` as either a static host contract or an accelerator
  dispatch/readback sentinel; Python never supplies or reconstructs a naked `Ready`
  capability list.
- `WgpuRank` / `wgpu.WgpuRank` for persistent exact TopK/MidK/BottomK storage,
  enqueue-only dispatch and combined value/index readback. See the
  [shared WASM/Python/Rust guide](../performance/resident_rank.md).
- `RankAdaptationSession` for bounded SpiralK candidate compilation and Black
  Cat selection over one rank workload. `choose()` returns a typed
  `RankAdaptationSelection`; its `.plan` is the selected `RankPlan` and its
  `.receipt()` carries the decision witness and client provenance.
  `observe(selection_id, elapsed_ms, correctness_passed)` credits the Rust-owned
  reward only after correctness, while `abandon(selection_id)` closes an
  unexecuted choice without posterior credit. UCB decision receipts distinguish
  `selection_attempts` from rewarded `observations`, so failed work advances
  initial exploration without being reported as learned performance. A
  correctness-failed arm is quarantined; an abandoned arm remains eligible.
  These fields use Black Cat bandit witness contract v3. Candidate receipts
  expose the effective kernel execution signature, and every nested plan carries
  the same requested/effective Python client provenance as a directly created
  `RankPlan`. Use `plan(..., strict_accelerator=True)` before comparing multiple
  candidates through a caller-owned native executor. This forbids a failed GPU
  arm from silently becoming a software run with a separate posterior; a native
  launch failure must be abandoned rather than credited. Fallback-allowed plans
  are deduplicated by the route currently executable in the Rust build.
  Strict WGPU signatures are marked `scope=declared_native`: they validate the
  shared Rust kernel geometry but do not attest device initialization. Direct
  lane hints are deduplicated on heap/workgroup routes; subgroup bitonic keeps
  the effective keep-count when a fixed subgroup width is known or declared.
- ROCm probing (`hip_probe`) so Python callers can reflect the stubbed
  device hints shared with the Rust runtime.
- Z-space barycentre solver (`z_space_barycenter`) to mix colour-field
  priors and chart couplings directly from Python.
- Source builds also expose `mean_tensors_scaled(partials, scale=1.0)` for
  signed-vector arithmetic means. For example,
  `st.mean_tensors_scaled([st.Tensor(1, 2, [1, -2]), st.Tensor(1, 2, [3, 4])])`
  returns `[[2, 1]]` via `.tolist()`. GoldenRetriever and WASM `TensorMeanBatch`
  share this Rust CPU reducer: input-order f64 addition, divide by count,
  multiply by the f32 scale promoted to f64, then round to f32. It rejects
  non-finite inputs/scales/outputs and does not dispatch a rank kernel, execute
  a probability barycenter, or construct an autograd graph.
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
   FT/WASM/runtime probes without dropping into Rust. With the wheel's `nn`
   feature (included in `python-default`), the ConvNeXt factory now constructs
   the real Rust backbone/global-pool/classifier rather than a `SimpleCnn`:

   ```python
   model = st.vision_create_classification_model("convnext_tiny", num_classes=3, seed=42)
   print(model.parameter_count())  # 27_888_003 scalar parameters
   print(model.metadata()["has_pretrained"])  # False
   # model.forward(images): nonempty list of 3x224x224 ImageTensor values
   ```

   This factory exposes ordinary inference. For explicit GPU-resident training,
   `st.vision.ResidentConvNeXtClassifier` exposes the same Rust model's
   forward/VJP/SGD and checkpoint handles with `nn,wgpu` enabled; see
   [Resident Vision Training Clients](../resident_vision_training_clients.md).
   Initialization is seeded; no pretrained weights or torchvision
   checkpoint compatibility is implied. Other legacy model kinds still use
   `SimpleCnn`. The shared Rust training path is documented in
   [Resident ConvNeXt Training](../resident_convnext_training.md).
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
  `efficiency`, and `latency` routing. Those scores, normalized latency/token
  costs, health penalties, deterministic rank, metric winners, and near-best
  membership come from the typed
  `st-core::runtime::api_llm_route_policy` contract, not a Python replica.
  Aggregate token totals are normalized per observed response before cost
  scoring, so longer trace runs are not penalized merely for containing more
  samples.
  Its v1 evidence witness treats missing latency/token measurements as unknown
  rather than free, shrinks sparse or partial evidence toward a neutral prior,
  and excludes zero-observation rows from selection. The comparison payload's
  `selection_semantics` records the exact Rust owner and score-formula version;
  Python deliberately stops if that native contract is absent or stale. Use
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
  reuses an existing sweep report with `--report report.json`. Python gathers
  provider evidence and orchestrates stAgent, while the compiled
  `st-core::runtime::topos_route_policy` contract exclusively owns profile
  normalization, scoring, deterministic tie-breaking, reward projection, and
  selected-route resolution. Its v2 evidence contract ignores client-provided
  score fields, uses a neutral prior for missing metrics, shrinks by sample
  count, excludes zero-observation routes, and revalidates the source evidence
  when a learned selection is resolved. Stored v1 reward arrays lack this v2
  witness and must be rebuilt from the original sweep rows before resolution;
  legacy rows without a positive observation `count` must be remeasured.
  The same contract is exposed to browser clients
  through `spiraltorch-wasm`; Python deliberately has no semantic fallback when
  the Rust core is unavailable. Browser-side
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
  dataset/model catalogs from Python. On WGPU-enabled native builds,
  `pipeline.enable_wgpu()` attaches the GPU dispatcher; no dispatcher is attached
  by default. `apply_resident_batch` and `VisionDataLoader.next_resident_batch`
  keep Normalize/resize/crop/flip on the GPU, while ordinary `apply` still returns
  a host image. See [Resident Vision Input](../resident_vision_input.md)
  for checked statistics, explicit target upload and failure/retry boundaries.
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

## Rust-owned token periodicity

Training objectives, generation evidence, Python, and WASM now share one bounded
periodic-suffix kernel in `st-core`. Python sends token IDs and optional proposal
state; it does not recreate period search, tie-breaking, ratios, or report
identity:

```python
import spiraltorch as st

report = st.zspace_periodicity(
    [9, 1, 2, 1, 2, 1],
    appended_token_id=2,
    maximum_period=16,
    minimum_repetitions=3,
)
assert report["periodic_suffix"]["period"] == 2
assert st.validate_zspace_periodicity(report) == report
```

Every report records the canonical request, a SHA-256 analysis identity, a
conservative comparison-work bound, and the exact selected suffix. The validator
replays the Rust analysis and rejects changes to either request or result. Token
IDs are limited to JavaScript-safe integers so persisted Python and browser
reports remain identical. This is structural token evidence only: a detected
suffix does not establish semantic degradation or predict that a loop will
continue.

The normal Python facade accepts only passive `list`/`tuple` token containers and
passive `dict` reports. It snapshots built-in storage without invoking subclass
iteration, item, length, or type hooks, then the native boundary applies the
catalogued byte/node/depth limits before serde materialization. Arbitrary
`Sequence` and `Mapping` implementations are deliberately not an ingress
contract.

## Rust-owned stochastic Schrodinger dynamics

The direct dynamics API records the complete state, potential, standard-normal
noise witness, configuration, complex quadratures, and numerical audit in a
content-addressed Rust receipt. Omitting `standard_normal` records an all-zero
witness instead of invoking an implicit RNG:

```python
import spiraltorch as st

forward = st.zspace_stochastic_schrodinger_forward(
    [1.0, 0.25, -0.5, 0.75],
    [0.2, -0.1],
    standard_normal=[0.1, -0.3, 0.2, 0.0],
    config={"time_step": 0.08, "noise_scale": 0.15},
)
assert st.validate_zspace_stochastic_schrodinger_forward(forward) == forward

vjp = st.zspace_stochastic_schrodinger_vjp(
    forward,
    [0.2, -0.4, 0.1, 0.3],
)
assert vjp["forward_id"] == forward["forward_id"]
assert st.validate_zspace_stochastic_schrodinger_vjp(vjp) == vjp
```

VJP construction first replays the complete forward receipt in Rust, then
recomputes phase from its canonical request. It differentiates `output_real`
with respect to input and potential; the noise witness and configuration remain
fixed. A changed quadrature, phase, noise witness, gradient, or audit field fails
closed. These receipts certify the stated bounded numerical transition and
derivative, not physical fidelity or training efficacy.

## Blinded semantic review

Held-out generation packets can be reviewed without rebuilding scoring or
unblinding semantics in Python. Rust can build and seal the packet itself,
validates the packet and pre-review map commitments, partial draft coverage, 1-through-5 score bounds,
complete-response receipt, and arm/seed aggregation. Python only presents
groups and atomically saves the last fully validated draft.

```bash
PACKET=docs/benchmarks/hf_periodic_baseline_replication_pythia70m_alice_semantic_review_packet_20260823.json

# reviewer-id is a pseudonymous lowercase sha256:<64 hex> identity.
spiral-hf-semantic-review inspect "$PACKET" --output packet-receipt.json
spiral-hf-semantic-review review "$PACKET" \
  --draft semantic-review-draft.json \
  --reviewer-id 'sha256:<64 lowercase hex>'

# Resume without repeating reviewer identity. Exit status 2 means incomplete.
spiral-hf-semantic-review review "$PACKET" \
  --draft semantic-review-draft.json

# Supply the separately held map only after the complete response is sealed.
spiral-hf-semantic-review unblind "$PACKET" \
  semantic-review-draft.json blinding-map.json \
  --output semantic-review-unblind.json
spiral-hf-semantic-review validate-report semantic-review-unblind.json
```

The corresponding Python surface is
`seal_zspace_semantic_review_packet()`,
`validate_zspace_semantic_review_packet()`,
`zspace_semantic_review_map_id()`,
`summarize_zspace_semantic_review_draft()`, and
`unblind_zspace_semantic_review()`. The same lifecycle is available through
the WASM JSON/object API. Python ingress is bounded before serde materialization;
the Rust contract caps aggregate JSON-encoded packet text at 32 MiB, groups/map entries at
10,000, and arm names at 128 bytes. Existing packet and map IDs are unchanged.
A historical v1 artifact above a newer aggregate packet or standalone-map
admission budget can be replayed only through the explicitly named
`*_trusted_legacy_replay` Rust/Python functions. Those opt-in functions accept
already trusted local evidence only; normal validation, the CLI, and every WASM
entry point remain bounded and must be used for untrusted or remotely supplied
input.
A structurally valid report does not prove
that the reviewer remained blind and does not establish model superiority.
