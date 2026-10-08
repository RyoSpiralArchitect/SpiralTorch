# Installation, source builds, and backend selection

[Documentation index](../README.md) | [Project entry](../../README.md)

This is the detailed source-tree reference, moved from the README.
Examples retain their individual feature, device, and optional-dependency requirements.
A source-tree API is not a claim that an older PyPI wheel exposes it.
Run repository commands from the repository root.

## Contents

- [Install (pip)](#install-pip)
- [wheel](#wheel)
- [Python orientation after install](#python-orientation-after-install)
- [Build from source (cargo)](#build-from-source-cargo)
- [Build Python wheel (maturin)](#build-python-wheel-maturin)
- [Release operations](#release-operations)
- [Backend Matrix](#backend-matrix)
- [Tests](#tests)
- [Troubleshooting & FAQ](#troubleshooting--faq)

## Install (pip)

```bash
pip install -U spiraltorch
```

- Wheels are **abi3**; you can use any CPython ≥ 3.8.
- Prebuilt wheels ship for **Windows**, **manylinux2014 (x86_64)**, and **macOS 14+ (universal2)**.
- The published wheel is built with `wgpu,logic,kdsl` (CPU always available; WGPU activates when a compatible backend is present).
- For macOS < 14 or other targets, build from source.

### wheel

Without cloning the repo, `pip install spiraltorch` gives you:

- **Core tensors + optimisers:** `st.Tensor`, autodiff, `st.optim.Amegagrad`, DLPack interop.
- **Native neural layers:** `st.nn.Linear`, `Embedding`, `Sequential`, losses,
  `ModuleTrainer`, `LoraLinear`, and `ZSpaceProjector`.
- **Checkpoint handoff:** checked `state_dict` compatibility reports,
  key-mapped loads, overlap resize/projection preflight, and HF-style presets.
- **Geometry:** `st.frac.MellinLogGrid` (Mellin mesh / verticals) + Hilbert helpers.
- **Signal tools:** `st.MaxwellFingerprint` expectation curves (visualisation-ready).
- **Z-space training utilities:** `st.ZSpaceTrainer`, `st.LanguageWaveEncoder`.
- **Rust-owned evaluation protocols:**
  `st.zspace_runtime_protocol_catalog()` exposes the content-addressed Rust,
  Python, and WASM surface for held-out generation evidence, bounded
  token-periodicity analysis, repetition-unlikelihood plans, and blinded
  semantic review. Persisted
  catalogs can be replayed with `st.validate_zspace_runtime_protocol_catalog(...)`;
  every client surface records its normal-admission profile, while serialized
  Python/WASM surfaces also carry Rust-owned byte/node/depth limits. Catalogued
  Python/WASM paths remain bounded, WASM object helpers stay trusted-local and
  outside that guarantee, and trusted legacy replay is explicit and never
  exposed to WASM.
- **Canvas + observability:** `st.canvas.CanvasProjector`, `st.telemetry.*`, HTML trace writers, `st.serve_zspace_trace`.
- **SpiralK planning:** `st.plan_topk(...)`, the Rust-owned
  `st.RankPlan.contract()` audit payload, `st.write_kdsl_trace_jsonl`, and
  `st.write_kdsl_trace_html`; invalid rank shapes and device-capability
  overrides fail closed in the Rust planner. SpiralK contexts and hard rewrites
  use that same planner contract, so Python never clamps or reinterprets rank
  choices independently.
- **API-model LLM bridge:** `st.ApiLLMZSpaceRuntime` converts hosted LLM
  responses/callables into Z-space runtime traces without requiring an API SDK
  at install time; when optional provider packages are available,
  `st.make_openai_responses_invoke(...)`, `runtime.call_openai_responses(...)`,
  `st.make_anthropic_messages_invoke(...)`, or
  `runtime.call_anthropic_messages(...)` can use provider keys from the
  environment directly. Persist runs with
  `runtime.write_jsonl("api_llm_trace.jsonl")`, reload with
  `st.load_api_llm_trace_events(...)`, and compare runs with
  `st.summarize_api_llm_trace_events(...)` or
  `st.compare_api_llm_trace_runs({"baseline": "a.jsonl", "candidate": "b.jsonl"})`.
  The runtime derives its shared gradient width from `len(z_state)` by default.
  When injecting a prebuilt Topos/WASM/geometry context, build it with that same
  `gradient_dim`, or set `gradient_dim=...` on the runtime/suite explicitly;
  mismatched active gradients fail under the Rust-owned `strict` contract.
  Use `runtime.run_prompts(...)` or `st.run_api_llm_prompt_suite(...)` for a
  multi-prompt bipolar/Z-space suite backed by OpenAI, Anthropic, or any
  compatible callable. Use `st.run_api_llm_prompt_suite_matrix(...)` when the
  same prompts should be replayed across several provider routes and compared
  from their persisted JSONL traces. Trace comparison also surfaces
  `empty_text_rate`, `refusal_rate`, `completion_rate`, and stop-detail
  categories so provider safety behavior does not hide behind raw confidence.
- **FT diagnostics:** tokenizerless byte-LM profile smokes, Transformers logit
  trace capture, runtime import audits, and WGPU readiness gates.

### Python orientation after install

If you only want to confirm the wheel and choose a runtime route, start with:

```python
import spiraltorch as st

print("spiraltorch", st.__version__)
print("cpu:", st.describe_device("cpu")["backend"])
print("wgpu:", st.describe_device("wgpu").get("backend"))

session = st.SpiralSession(backend="auto")
print("session backend:", session.backend)
print("effective backend:", getattr(session, "effective_backend", session.backend))
```

Runtime route readiness and workload-kernel readiness are deliberately separate.
The v6 `spiraltorch.runtime_execution_plan` contract lets native Rust observe the
exact declared component shapes, bias bindings, device limits, and lazy pipelines.
Those results live in a nested v2
`spiraltorch.runtime_component_capability_observation` contract that binds the
runtime probe, canonical workloads, selected Rust policy, evidence, and both
commitment hashes. Kernel support is decided by `st-tensor`, not reconstructed by
`st-core`: host readiness carries `static_host_contract`, while accelerator readiness
requires exact workload preflight plus an operation-specific device dispatch/readback
sentinel and carries `runtime_dispatch_sentinel`. Passing `component_workloads=` to
`st.evaluate_runtime_execution_plan(...)` invokes that Rust observer automatically;
WASM exposes the same contract as JSON/Object transport instead of rebuilding the
rules in JavaScript. The execution plan owns policy selection; the capability
observer measures that resolved policy and cannot replace it. Naked client-supplied
capability arrays are rejected. The commitment is reproducibility evidence, not
cryptographic hardware attestation, and
undeclared workloads remain unobserved. Once a committed plan is installed, every
declared workload is bound exactly through tensor dispatch: shape, bias, or utility
operation mismatches are rejected before a kernel call. Undeclared components remain
dynamic and continue through operation-time capability checks.
Receipt self-validation deliberately does not treat a well-formed plan hash as
authorization. Supply the original committed plan to make Rust replay the plan and
reapply its exact workload, backend, threshold, and fallback rules:

```python
import spiraltorch as st

workload = {"component": "softmax", "rows": 2, "cols": 3}
plan = st.evaluate_runtime_execution_plan(
    st.describe_device("cpu"),
    accelerator_fallback="allow",
    tensor_util_wgpu_min_values=1024,
    component_resolution="deferred",
    component_workloads=[workload],
)
receipt = {
    "kind": "spiraltorch.tensor_execution_receipt",
    "contract_version": "spiraltorch.tensor_execution_receipt.v1",
    "semantic_owner": "st-tensor::execution",
    "component": "softmax",
    "operation": "row_softmax",
    "workload": workload,
    "requested_backend": "cpu",
    "selected_backend": "cpu",
    "executed_backend": "cpu",
    "kernel_backend": "cpu",
    "route_status": "direct",
    "runtime_execution_plan_output_sha256": plan["output_sha256"],
}
validated_receipt = st.validate_tensor_execution_receipt_against_runtime_plan(
    receipt,
    plan,
)
```

WASM exposes the same operation as
`tensorExecutionReceiptValidateAgainstRuntimePlanJson` and
`tensorExecutionReceiptValidateAgainstRuntimePlanObject`; neither client rebuilds
the authorization rules.
Standalone evaluation uses `component_resolution="concrete"`, so strict plans reject
unobserved accelerator capabilities rather than guessing that a workload is ready.

For ordinary training, let `SpiralSession` capture that contract once. Rank plans,
trainers, schedules, optimizer checkpoints, and replayed sessions then inherit the
same Rust-owned commitment instead of reading mutable environment configuration
again. A session records `component_resolution="deferred"`: this does not label
unobserved kernels native; it commits the Rust policy and leaves shape-specific
capability checks to each operation:

```python
import spiraltorch as st

session = st.SpiralSession(
    backend="cpu",
    tensor_util_wgpu_min_values=37,
)
rank = session.plan_topk(rows=8, cols=64, k=4)
trainer = session.trainer()

assert rank.runtime_execution_plan_output_sha256 == session.runtime_execution_plan_output_sha256
assert trainer.runtime_execution_plan_output_sha256 == session.runtime_execution_plan_output_sha256

replayed = st.SpiralSession.from_runtime_execution_plan(session.runtime_execution_plan)
assert replayed.runtime_execution_plan_output_sha256 == session.runtime_execution_plan_output_sha256
```

The returned `runtime_execution_plan` is an isolated copy. Supplying a committed
plan together with backend, capability, or execution-config overrides fails closed,
and executable materialization rechecks the receiving Rust build and local runtime.
`backend="auto"` is only orchestration order: Python asks Rust for WGPU readiness
first and tries CPU only when Rust returns an explicit unavailable signal and the
captured fallback policy allows it. Plan validation, transport, and configuration
errors remain visible instead of silently changing the backend.
`SPIRALTORCH_STRICT_GPU=1` is captured by Rust and forbids that CPU retry; it also
keeps small tensor-utility operations on WGPU instead of applying the performance
threshold as an implicit fallback. `st.resolve_runtime_execution_config()` exposes
the exact Rust-captured policy for inspection.

When a plan is built separately, bind it before creating a training schedule so
rank planning and every tensor kernel share that same context:

```python
import spiraltorch as st

plan = st.evaluate_runtime_execution_plan(
    st.describe_device("cpu"),
    accelerator_fallback="allow",
)
trainer = st.nn.ModuleTrainer(backend="cpu")
trainer.bind_runtime_execution_plan(plan)
schedule = trainer.roundtable(rows=8, cols=64)

assert trainer.runtime_execution_plan_output_sha256 == plan["output_sha256"]
```

Plans are validated and materialized in Rust. A tampered, blocked, or locally
unavailable plan leaves the trainer unchanged, while a schedule created under a
different device/config contract is rejected before the epoch starts.

Rank planning is intentionally narrower than execution. Python and WASM can validate
a committed plan and reuse its capabilities, configuration, and parent SHA without
claiming that their local process has the corresponding tensor executor. Sessions and
trainers still cross the stricter `BackendPolicy` boundary before running any kernel.

Then pick the path that matches the job:

- **Native learning loop:** use `st.Tensor`, `st.nn`, `st.optim`, and
  `st.SpiralSession` directly when you want SpiralTorch-owned tensors,
  checkpoints, traces, and WGPU/CPU routing.
- **Interop-first experiment:** use `spiraltorch.ecosystem` or DLPack when a
  PyTorch/JAX/CuPy/TensorFlow object should cross into SpiralTorch without
  rewriting the whole training stack at once.
- **LLM / FT preflight:** run
  `bindings/st-py/examples/checkpoint_preflight.py`,
  `byte_lm_transformers_trace.py`, or `byte_lm_profile_smoke.py` against local
  checkpoint files before committing to heavier fine-tuning.
- **Dependency contract before a run:** use the CLI or the top-level Python
  helper to record optional runtime evidence without making Hugging Face
  packages hard dependencies of the wheel.
- **Source build or custom backend:** keep the wheel path for ordinary use, and
  switch to the maturin commands below only when you need local Rust changes,
  CPU-only artifacts, CUDA/HIP flags, or a release-equivalent wheel.

For the dependency contract path, either run the CLI:

```bash
spiral-runtime-preflight --preset hf-full-finetune --require --json-out ft-runtime.json
```

or keep the evidence inside Python:

```python
import spiraltorch as st

report = st.runtime_import_preflight_report(
    runtime_import_presets=["hf-full-finetune"],
    required_runtime_import_presets=["hf-full-finetune"],
)
print(report["runtime_import_preflight_passed"])

ft_report = st.hf_finetune_model_profile_preflight_report(
    profile="qwen2-0.5b-local-smoke",
    mode="full-finetune",
    runtime_device_backends=["wgpu", "cpu"],
)
print(ft_report["runtime_import_preset"])
```

---

## Build from source (cargo)

**Prereqs**

- Rust stable (`rustup`), Cargo
- macOS: Xcode CLT / Linux: build-essentials
- Optional GPU stacks: CUDA / ROCm / Vulkan as needed

**Workspace build**

```bash
# Debug (fast iteration)
cargo build --workspace

# Release (optimised)
cargo build --workspace --release

# Run tests
cargo test --workspace
```

For the complete Cargo member inventory, including packages that are workspace
members but not root `default-members`, see
[`docs/development/workspace_crates.md`](../development/workspace_crates.md).

**Per-crate**

```bash
cargo build -p st-core        # core math/runtime
cargo build -p st-nn          # neural helpers
cargo build -p st-vision      # vision kernels/pipelines
```

**Feature flags (typical)**
- `cpu` — CPU fallback (on by default in many crates)
- `wgpu` — Metal/Vulkan/DX12 backends via WGPU
- `cuda` — CUDA kernels
- `hip` — HIP planner + CPU reference contract (no executable ROCm kernels)
- `hip-real` — executable ROCm/HIP kernels

---

## Build Python wheel (maturin)

```bash
# Install maturin (once). 1.9+ is required so wheel license-files are emitted.
python -m pip install -U "maturin>=1.9,<2"

# Default binding build (WGPU-first; CPU fallback remains available)
maturin build -m bindings/st-py/Cargo.toml --release --locked

# Release-equivalent (matches PyPI wheels: default WGPU route + logic/kdsl)
maturin build -m bindings/st-py/Cargo.toml --release --locked --features logic,kdsl

# macOS 14+ universal2 (matches macOS wheels on PyPI)
export MACOSX_DEPLOYMENT_TARGET=14.0
maturin build -m bindings/st-py/Cargo.toml --release --locked --target universal2-apple-darwin --features logic,kdsl

# CPU-only (drop the default WGPU route but keep the standard Python surface)
maturin build -m bindings/st-py/Cargo.toml --release --locked --no-default-features --features python-default,cpu

# Add CUDA or real HIP alongside the default WGPU-first wheel
maturin build -m bindings/st-py/Cargo.toml --release --locked --features cuda,logic,kdsl
maturin build -m bindings/st-py/Cargo.toml --release --locked --features hip-real,logic,kdsl

# Backend-specific builds without the default WGPU route
maturin build -m bindings/st-py/Cargo.toml --release --locked --no-default-features --features python-default,cuda,logic,kdsl
maturin build -m bindings/st-py/Cargo.toml --release --locked --no-default-features --features python-default,hip-real,logic,kdsl

# Planner/reference-only HIP contract (contains no executable GPU kernels)
maturin build -m bindings/st-py/Cargo.toml --release --locked --no-default-features --features python-default,hip

# Install the wheel you just built
pip install --force-reinstall --no-cache-dir target/wheels/spiraltorch-*.whl
```

If your local `python` aborts or prints startup output from a third-party
`sitecustomize.py`, rerun the build with `PYTHONNOUSERSITE=1`. Startup output can
also confuse PyO3's interpreter probe; `python -s` disables user-site hooks for
direct Python invocations.

Linux note: for manylinux2014 wheels you either need a manylinux container (e.g. via GitHub Actions) or `maturin --compatibility manylinux2014 --zig` (requires `pip install maturin[zig]`). Building directly on Ubuntu without these may produce wheels that won’t install on older distros.

### Release operations

Manual build artifacts and published wheels are different states. Use the
[release runbook](../ops/release.md) for readiness, credential setup, signed
asset validation, immutable tags, and explicit PyPI publication. Commands live
there rather than being duplicated in installation instructions.


## Backend Matrix

> Replace feature names if your `Cargo.toml` differs (`cpu`, `wgpu`, `cuda`, `hip` are typical).

| Backend | How to build (cargo) | Notes |
|---|---|---|
| **CPU** | `cargo build -p st-core --no-default-features --features cpu` | Portable, no GPU deps |
| **Metal (macOS)** | `export WGPU_BACKEND=metal` then `cargo build -p st-core --features wgpu` | Apple GPUs via WGPU |
| **CUDA (NVIDIA)** | `export CUDA_HOME=/usr/local/cuda` then `cargo build -p st-core --features cuda` | Ensure driver & toolkit |
| **HIP/ROCm (AMD, Linux)** | `cargo build -p st-core --features hip-real` | Real dense/rank kernels; requires ROCm. `hip` alone is planner/reference-only |

**Wheel builds** mirror these. The default Python wheel is WGPU-first; use
`--no-default-features --features python-default,<backend>` for backend-specific
builds without the default WGPU route.

---


### Tests

```bash
# Rust tests
cargo test --workspace

# Python smoke
python - <<'PY'
import spiraltorch as st
x = st.Tensor(1,2,[1,0]); cap = st.to_dlpack(x); assert st.from_dlpack(cap).tolist()==[[1.0,0.0]]
print("ok")
PY
```

---

## Troubleshooting & FAQ

**Q: `AttributeError: module 'rl' has no attribute 'DqnAgent'`**
A: Use **`st.stAgent`**. The DQN surface was renamed for stability; façade provides the alias.

**Q: CUDA/ROCm link errors**
A: Verify `CUDA_HOME`, driver/toolkit versions, or ROCm installation. On CI, add toolkit paths to `LD_LIBRARY_PATH`/`DYLD_LIBRARY_PATH`.

**Q: Wheel contains stale symbols after code changes**
A: `pip uninstall -y spiraltorch && pip cache purge` → reinstall the freshly built wheel with `--no-cache-dir`.

**Q: How stable is the Python API?**
A: The shipped type stubs (`spiraltorch/__init__.pyi`) reflect the **supported** surface. New Rust exports appear dynamically via forwarding; removals/renames adopt compatibility aliases where possible.

---
