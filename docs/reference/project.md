# Project background and contributor reference

[Documentation index](../README.md) | [Project entry](../../README.md)

This is the detailed source-tree reference, moved from the README.
Examples retain their individual feature, device, and optional-dependency requirements.
A source-tree API is not a claim that an older PyPI wheel exposes it.
Run repository commands from the repository root.

## Contents

- [🚀 Latest SpiralTorch highlights](#-latest-spiraltorch-highlights)
- [Why it’s different](#why-its-different)
- [Planning the Ecosystem](#planning-the-ecosystem)
- [Julia & Go integration ](#julia--go-integration)
- [Explore the runtime interactively](#explore-the-runtime-interactively)
- [Technical notes](#technical-notes)
- [🛠️ Why Robotics Teams Are Watching](#️-why-robotics-teams-are-watching)
- [Safety & fallbacks](#safety--fallbacks)
- [Contributing](#contributing)
- [Social preview](#social-preview)
- [Troubleshooting](#troubleshooting)
- [License](#license)

geometry, and device heuristics in the same conversation. SpiralK orchestrates
the kernels, the hypergrad tape streams Z-space meaning, and the high-level
`st-nn` modules stay PyTorch-compatible without shipping NumPy or PyTorch.

The stack is comfortable living entirely in Rust—yet the Python wheel remains a
thin veneer that reuses the same planners, losses, and Z-space resonators. No
tensor shims, no translation layers, and no tracebacks.

---

## 🚀 Latest SpiralTorch highlights

- **Learning Stack (model zoo).** Runnable baselines (Rust + Python) including raw-text char LMs (no tokenizer), coherence scan/wave, and an attentionless `WaveRnn+Mixer` variant; use `--desire` to bias sampling and `--events ... --atlas` to emit telemetry routes (see `models/README.md`).
- **Fractal → Quantum RL bridge.** Stream Mellin-log fractal patches straight into the quantum overlay studio and recover policy gradients through `FractalQuantumTrainer` and friends—keeping Python fallbacks and PyO3 builds in lockstep.
- **Self-evolving SpiralK kernels.** A new diversity governor inside `SelfRewriteEngine` tracks plateauing η̄ gains, forces fresh AI rewrites when caches go stale, and surfaces telemetry via `diversity_snapshot()` so operators can keep the autonomous kernel lab on course.
- **Saga-aware kernel evolution.** The `SelfRewriteEngine` now learns multi-step hint sagas, boosting cache priority for sequenced rewrites and exposing the orbits via `saga_snapshots()` so you can audit every cosmic combo.
- **Reinforcement-tuned hint genetics.** Hint chains earn reward-weighted feedback, anomaly filters freeze outliers, recombination fuses cached winners, graph snapshots illuminate hint affinity, and `take_trace()` streams the self-evolution log when `--debug-self-evolution` is enabled.
- **Chain-policy reinforcement.** Cached hint chains now feed a lightweight Q-style policy inside `SelfRewriteEngine`, boosting cache priority via chain multipliers, exposing telemetry with `chain_policy_snapshots()`, and logging `PolicyUpdate` traces whenever the self-evolution loop reshapes its instincts.
- **Contextual bandit autopilot.** A UCB-flavoured bandit now rides alongside the cache, weighting hint reuse via `bandit_snapshots()` telemetry, steering exploration with `with_bandit_*` knobs, and leaving `BanditUpdate` traces whenever rewrites learn a new favourite arm.


## Why it’s different
 - **Training comes first:** Modules such as `Linear`, `Sequential`,
   `WaveGate`, the new `ToposResonator`, and `ZSpaceProjector` stream gradients
    into the hypergrad tape and expose a `train_epoch` loop that mirrors
    familiar `nn.Module` patterns.
  - **Open Z-space:** Gradient splits honour the A/B/C roundtable through the
    new `zspace_round` ops module so Above/Here/Beneath bands stay in sync with
    SpiralK plans without auxiliary buffers.
  - **Hilbert-grounded Mellin bridges:** `st-frac::mellin::MellinLogGrid`
    now exposes fallible APIs, a `Scalar` alias for f32/f64 toggling, exact
    lattice bit-matching, and WebGPU-backed vertical/mesh sweeps that reuse the
    same `st-frac::zspace` weights while `hilbert_inner_product` and
    `evaluate_vertical_line` surface the lattice’s Hilbert geometry directly.
  - **Three-voice consensus:** SpiralK heuristics, DSL directives, and the
    generated WASM tuner table discuss every launch decision and keep the
    transcript in the roundtable log.
  - **Rust by default, Python ready:** Every feature—from WASM tuning to
    hypergrad curvature—is implemented in Rust and exposed unchanged through the
    Python bindings when needed.
  - **Unified RL + Rec stacks:** SpiralTorchRL and SpiralTorchRec keep policy
    gradients, recommendation factors, and hypergrad tapes inside the same
    Z-space geometry so deployment-grade loops never leave Rust.
  - **Z-space-native graph reasoning:** The Rust core, backend abstraction
    layer, and Z-space operators already form the spine of a graph neural
    network stack that embeds large-scale, hierarchical graphs with the same
    fidelity as its tree-aligned geometry.
  - **Semiotic suturing at the logit level:** The new `st-nn::language`
    toolkit folds symbolic kernels, repression fields, and semantic bridges
    into a single Lagrangian so SpiralTorch can bias logits with desire,
    anchor S/s correspondences, and respect target entropies without leaving
    Z-space.
  - **Interpretability as a first-class citizen:** Hypergrad tapes, roundtable
    transcripts, and ψ telemetry double as explainability artifacts, enabling
    decision-path inspection without leaving the Z-space calculus.

**Current release:** [![PyPI](https://img.shields.io/pypi/v/spiraltorch.svg?label=spiraltorch)](https://pypi.org/project/spiraltorch/) (abi3 wheel, Python ≥3.8)
**Targets:** CPU (always), MPS, Vulkan/DX (WGPU), CUDA, HIP/ROCm

---


## Planning the Ecosystem

- Explore the [Ecosystem Roadmap](../ecosystem_roadmap.md) for high-level priorities around documentation, samples, and community building.
- Review the [Backend Feature Matrix](../backend_matrix.md) when validating device support or filing bugs that touch accelerators.
- **Interop focus.** SpiralTorch now ships a living [Compatibility Strategy](../compatibility_strategy.md) that maps out PyTorch, TensorFlow, and JAX migration paths—from trainer APIs to checkpoint conversion—so you can bring existing stacks along for the ride. The Python wheel exposes `spiraltorch.compat.torch|jax|tensorflow` helpers that exchange tensors with those runtimes through zero-copy DLPack capsules, plus ergonomic knobs for dtype/device targeting, gradient flags, and memory format tweaks.

## Julia & Go integration

- Prototype workflows for future Julia/Go bindings live in [docs/ops/julia_go_development.md](../ops/julia_go_development.md). Follow the guide for setup, lint/test commands, and release checklists.

Prefer flat-space optimisation? Reach for the new Rust-side
`st_tensor::AmegaRealgrad` tape to mirror the same API without the Poincaré
projection step—handy when Canvas Transformer energy needs to feed classical
optimisers alongside its hypergradient updates.


### Explore the runtime interactively

Prefer a guided walkthrough of the dispatcher flows? Open the
[interactive runtime explorer](../interactive/README.md) for clickable
diagrams, a narrated "story tour" of the runtime handoff, and playful
spotlights on graph-node materialisation versus return-handle delivery:

- 🎬 **Story tour.** Step through a six-beat mini adventure that explains
  how a single API call ripples through SpiralTorch, from the first FFI
  marshals to the triumphant return of tensor handles.
- 🔍 **Focus toggles.** Snap to either the graph-node materialisation
  path or the return-handle arc whenever you want to revisit a specific
  phase.
- 🌈 **Aurora mode.** Bathe the canvas in a psychedelic gradient to feel the
  runtime choreography pop—the toggle sits beside the spotlight buttons.
- ✨ **Phase constellations.** Sidebar cards cluster subsystems by stage so you
  can intuit which teams light up together as the story advances.
- 🧭 **Free roam.** Click any node or edge to read quick lore about the
  component, then resume the story exactly where you left off.

> **Update — GPU-first convolution.** `Conv2d` now routes through a WGPU im2col + GEMM path that expands the 5D activation volume entirely on the GPU before projection back into Z-space, accelerating large vision stacks on portable GPUs.
>
> **New — Conv6da with Leech enrichment.** `Conv6da` fuses six-directional adjacency with optional Leech lattice density boosts so Z-space fields aggregate neighbors with structure-aware gradients.

> **New — DLPack/compat inference bridges.** Import weights over `st.from_dlpack` or the `spiraltorch.compat` adapters and feed them straight into Z-space inference via `st.weights_partial_from_dlpack`, `st.weights_partial_from_compat`, or the higher-level `st.infer_weights_from_dlpack`. PSI telemetry is now summarised live during these projections so models can modulate confidence against streaming telemetry frames.

> **Expanded — Higher-order convolutions.** Fresh `Conv3d` and `Conv4d` modules now mirror the dilation-aware ergonomics of their 1D/2D siblings so volumetric stacks and temporal cubes slide straight into the same API.
>
> **New — Online-softmax fused attention.** A single-kernel QKᵀ + mask + softmax + V pipeline now lands in the WGPU backend, slashing bandwidth for multi-head attention while matching PyTorch semantics for drop-in migrations.

> **In flight — CUDA attention kernel.** The fused scaled dot-product path now supports causal masking, per-context sequence lengths, optional Z-bias/attention bias, and an opt-in attention-probability readback so Z-space transformers can mix ragged batches without leaving the GPU hot path.

> **In progress — Fused attention for PyTorch migrations.** The new single-kernel Q·Kᵀ + softmax + V planner keeps intermediate logits on-chip, so PyTorch users can co-train or stage migrations while retaining numerically stable attention/softmax semantics.

> **In progress — Fused attention for Torch migrations.** The new single-kernel Q·Kᵀ + softmax + V planner keeps intermediate logits on-chip, so PyTorch users can co-train or stage migrations while retaining numerically stable attention/softmax semantics.
>
> **New — Python `Tensor.scaled_dot_attention`.** The fused kernel is now exposed to Python callers with a CPU fallback, so migration experiments can drop directly into WGPU-backed attention without leaving the high-level API.

> **In flight — CUDA attention kernel.** The fused scaled dot-product path now supports causal masking, per-context sequence lengths, optional Z-bias/attention bias, and opt-in attention probability/logit readback so Z-space transformers can mix ragged batches without leaving the GPU hot path.

> **New — Z-space inference for imported checkpoints.** `spiraltorch.infer_weights_from_dlpack` and `spiraltorch.infer_with_psi` now project DLPack/compat weights, Canvas transformers, and PSI telemetry straight into the Z-space posterior. Warm-start inference can blend partial observations with live ψ health data so Rust sessions reuse PyTorch/JAX weights without leaving the SpiralTorch runtime.

> **New — PSI synchroniser learning bundles.** Multi-branch MetaMEMB runs now deliver combined heatmaps, ZPulse snapshots, Atlas fragments, PSI component breakdowns, and Golden directives via `st.psi.run_zspace_learning(...)` so Z-space learners and distributed `golden` retrievers can coordinate straight from Rust or Python.

---

## Technical notes

- [Coded-Envelope Maxwell Model (M₀^code)](../coded_envelope_maxwell_model.md) — Technical memo on the sequential detection framework that couples physical fingerprints with semantic gating.
- [Conceptual Entropy and Qualia](../conceptual_entropy_qualia.md) — SpiralTorch-oriented translation of the qualia report tracing how the term drifts across philosophy, neuroscience, and public discourse.
- [Drift-Response Linguistics for Z-space Language Training](../drift_response_linguistics.md) — Full write-up of the existential load / safe radius theory, signature geometry with timing elasticities, tipping radii, and direction-aware safe radii, plus how SpiralTorch wires DRL penalties into trainers and governance loops.
- [General Relativity Couplings inside Z-space](../general_relativity_zspace.md) — How Lorentzian curvature feeds Einstein tensors into ZPulse telemetry so gravitational signals steer the cooperative runtime.
- [Invariant barrier gating and contraction notes](../invariant_barrier_design.md) — Design cheatsheet covering safety barriers, steady amplitudes, and contraction-rate lower bounds for Spiral dynamics controllers.
- [COBOL web dispatch quickstart](../cobol_web_dispatch.md) — structure of the WASM planner and how to shuttle envelopes into MQ or CICS entrypoints.
- [SpinoTensorVector derivation notes](../stv_z_space.md), Start architecting Z-space kernels with the new  covering determinant identities, kernel causality, and minimal-norm constructions ready for implementation.

---

## 🛠️ Why Robotics Teams Are Watching

SpiralTorch’s runtime mirrors the main pressure points that robotics teams wrestle with today:

1. **Vendor-agnostic acceleration (WGPU-first).** Robots ship on Jetsons, Raspberry Pis, Apple Silicon laptops, AMD workstations, and plain x86 servers. SpiralTorch kernels are written against WGPU, so the same Z-space operators run across Metal, Vulkan, or DirectX 12 without CUDA lock-in.
2. **A control-plane grade runtime.** The mermaid diagram above is effectively a realtime robotics OS: session managers, graph planners, schedulers, command queues, and observability hooks coordinate sensing, planning, and actuation in one pipeline.
3. **Unified sensor fusion via Z-space.** Camera frames, LiDAR point clouds, IMU states, force/torque readings, and language instructions all project into a single geometric manifold. Planning loops consume multi-modal context without writing one-off fusion code per robot, while per-channel exponential smoothers and staleness monitors keep jittery or missing proprioception from destabilising downstream control.
4. **Geometric + gravitational dynamics.** Z-space now carries configurable geometry (Euclidean, curved manifolds, or GR-inspired metrics) and gravity wells (Newtonian or relativistic) so training can explore non-Euclidean kinematics, orbital behaviours, or heavy-mass safety regimes without leaving the runtime. The new Relativity Bridge mirrors the general relativity tensors used in the theory module—including Lorentzian metrics generated from symmetry ansätze—so the same conceptual primitives power both physics experiments and real robot rollouts.
5. **Instinctive behaviours with Desire Lagrangians + SpiralTorchRL.** Instead of hand-crafting reward functions, core instincts—“don’t tip over,” “seek charge,” “avoid the ledge”—are encoded as potentials. Traditional policy-gradient loops still plug in, but they optimise around explainable priors.
6. **Self-aware safety through ψ telemetry.** The runtime continuously measures kernel stability, allocator pressure, dynamics drift, and channel health. When thresholds break—whether from physical saturation or stale sensors—ψ telemetry drives self-maintain routines that pause execution or transition to safe postures before a robot ever enters an unrecoverable state.
7. **Training-grade data capture.** A built-in trajectory recorder snapshots fused observations, instinct evaluations, telemetry, and policy commands so teams can stream curated rollouts into offline RL pipelines or regression harnesses without bolting on a separate logging stack.
8. **Pluggable safety governance.** Robotics runtimes can now host drift-response safety plugins sourced from the `spiral-safety` crate. ψ telemetry, Desire energy, and channel health feed directly into SpiralTorch’s governance metrics so the same policy guardrails auditing language models can halt physical robots when hazards spike.
9. **Z-space trainers with temporal feedback.** The new `TemporalFeedbackLearner` and `ZSpaceTrainerBridge` convert runtime steps, ψ telemetry, and Desire energy into Z-space trainer metrics (speed, memory, stability, drift) and `ZSpacePartialBundle` payloads. Vision and instinctive signals stay synchronised so offline trainers can replay the exact feedback a robot experienced on the floor.
10. **CanvasProjector-aligned vision control.** `VisionFeedbackSynchronizer` aligns CanvasProjector vector fields with sensor channels, exposes alignment/energy metrics, and feeds them into the same partial bundles. Reinforcement loops now mix proprioceptive policies with live visual feedback for gradients that understand what the robot “saw” when it acted.
11. **Episode-scale dataset tooling.** `ZSpaceTrainerEpisodeBuilder` groups trainer samples into bounded episodes, reports aggregate stability/memory/drift scores, and hands Python bindings a ready-to-train bundle. Teams can checkpoint on-policy rollouts, feed them into offline RL, or compare episode health across deployments without rebuilding analysis plumbing.

Together these properties let SpiralTorch act as the perception, planning, and autonomy stack for heterogeneous robot fleets—from lab manipulators to outdoor rovers—without a rewrite per platform.



## Safety & fallbacks

- CPU remains **always available**, and Python wheels can drop the default WGPU
  route with `--no-default-features --features python-default,cpu`.
- WGPU / CUDA / HIP are **feature-gated** and degrade safely.
- Heuristic chooser always returns a **safe** `Choice` (fills mk/tile from table or conservative defaults).

---

## Contributing

See `CONTRIBUTING.md` for local setup, testing, and wheel build commands (Windows: `scripts/dev.ps1`).

Issues & PRs welcome, especially:
- Backend kernels (WGPU subgroup variants, HIP/CUDA heap/k-way merges)
- Tuner recipes & generated tables
- New SpiralK sugar (e.g., `penalty_if(...)`, device-aware bands)

Run tests/benches on your device and share logs (latency / shapes / adapter caps).
**AGPL-3.0-or-later** keeps it open and remix-able.

---

## Social preview

Upload a social preview PNG via **Repo → Settings → Social preview** (1200×630).
Suggested caption: **“SpiralTorch — WGPU-first, Self-Tuning GPU Top-K (Rank-K)”**.

---

### Troubleshooting

- **No Redis?**
  Build without `kv-redis` or leave `REDIS_URL` unset. The consensus chooser
  skips network calls and falls back to SpiralK / Generated-table safely.

- **ROCm not installed but `hip` enabled?**
  Use `--features hip` only (stub path). The **real** path needs `hip-real`
  and a working ROCm + RCCL toolchain.

- **Wheels red?**
  First try the release-equivalent wheel build: `maturin build -m bindings/st-py/Cargo.toml --release --locked --features logic,kdsl`.
  If you suspect the default WGPU route, build CPU-only instead: `maturin build -m bindings/st-py/Cargo.toml --release --locked --no-default-features --features python-default,cpu`.

---

## License

**AGPL-3.0-or-later** for every crate and Python wheel. See [`LICENSE .txt`](../../LICENSE%20.txt).
Unauthorized derivations will be treated as non-compliant with AGPL §13
