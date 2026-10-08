# Geometric and ecosystem toolkits

[Documentation index](../README.md) | [Project entry](../../README.md)

This is the detailed source-tree reference, moved from the README.
Examples retain their individual feature, device, and optional-dependency requirements.
A source-tree API is not a claim that an older PyPI wheel exposes it.
Run repository commands from the repository root.

## Contents

- [Emerging toolkits unique to SpiralTorch](#emerging-toolkits-unique-to-spiraltorch)
- [Canvas Pixel Transformer → Z-space feedback](#canvas-pixel-transformer--z-space-feedback)
- [Z-space inference autopilot (DLPack + ψ telemetry)](#z-space-inference-autopilot-dlpack--ψ-telemetry)
- [SpiralTorchVision](#spiraltorchvision)
- [Z-space-native graph neural networks](#z-space-native-graph-neural-networks)
- [Explainability through hypergrad telemetry](#explainability-through-hypergrad-telemetry)
- [Multi-modal topos safety envelopes](#multi-modal-topos-safety-envelopes)
- [Autotune telemetry for the WGPU-first roadmap](#autotune-telemetry-for-the-wgpu-first-roadmap)
- [Microlocal interface gauges](#microlocal-interface-gauges)
- [Collaborative canvas telemetry and ghost trails](#collaborative-canvas-telemetry-and-ghost-trails)
- [Maxwell-coded envelopes meet SpiralK](#maxwell-coded-envelopes-meet-spiralk)
- [Quantum Reality Studio overlays](#quantum-reality-studio-overlays)
- [Semiotic suturing, desire control, and EGW bridges](#semiotic-suturing-desire-control-and-egw-bridges)

## Emerging toolkits unique to SpiralTorch

### Canvas Pixel Transformer → Z-space feedback

- `CanvasProjector::refresh_with_vectors` now returns both the RGBA buffer and
  a colour vector field that carries normalised energy and chroma as
  Z-space-friendly coordinates.
- `FractalCanvas::vectorFieldFft(false)` surfaces the per-row FFT spectrum as
  interleaved energy/chroma pairs so Canvas Transformer pipelines can ingest
  frequency features without leaving Rust.
- `FractalCanvas::emitWasmTrail(curvature)` packages the live vector field into
  `(x, y, z, energy, r, g, b)` samples so WebGPU/WebXR experiences can stream
  Z-space particle trails without reconstructing curvature or FFT passes in
  JavaScript.
- `FractalCanvas::emitWasmTrail(curvature)` packages the live vector field into
  `(x, y, z, energy, r, g, b)` samples so WebGPU/WebXR experiences can stream
  Z-space particle trails without reconstructing curvature or FFT passes in
  JavaScript.
- `CanvasProjector::accumulate_hypergrad` and
  `CanvasProjector::accumulate_realgrad` stream the refreshed canvas tensor
  directly into SpiralTorch's Riemannian or Euclidean optimisers without
  additional copies.
- `FractalCanvas::relation()` mirrors the projector's tensor output as a
  `Float32Array` so browser call-sites can feed the raw relation into custom
  pipelines or training loops.
- `FractalCanvas::hypergradWave(curvature)` and `FractalCanvas::realgradWave()`
  surface curvature-aware hypergrad updates alongside Euclidean gradients so the
  Canvas Transformer can keep hypergrad/Realgrad buffers in sync by default.
- `FractalCanvas::gradientSummary(curvature)` condenses both tapes into shared
  L1/L2/∞ norms plus RMS/mean-absolute magnitudes so monitoring dashboards can
  watch gradient health without shipping the full relation buffers across the
  WASM boundary.
- `FractalCanvas::desireInterpretation(curvature)` lifts the paired gradient
  summaries into Desire-ready feedback metrics (pressure, balance, stability)
  so automation layers can steer the Desire Lagrangian without leaving WASM.
- `FractalCanvas::desireControl(curvature)` extends that pipeline with
  ready-to-apply Desire gradient control packets—penalty gains, bias/observation
  mixers, and tuned hyper/Realgrad learning-rate scales—mirroring the Rust
  automation layer on the browser side.
- `FractalCanvas::hypergradOperatorUniformFromControl(control)` and
  `FractalCanvas::hypergradOperatorUniformAuto(curvature)` map those Desire
  control packets directly into the WGSL uniform payload, saving JavaScript
  callers from recomputing the blend/gain heuristics before dispatching the
  GPU hypergrad operator.
- `FractalCanvas::vectorFieldFftKernel(true)` returns the ready-to-dispatch
  WGSL compute shader (including uniform layout) so WebGPU call-sites can bind
  the vector field and accumulate the spectrum fully on-GPU.
- `FractalCanvas::hypergradOperatorKernel(false)` emits the complementary WGSL
  pass that accumulates relation tensors into hypergradient buffers directly on
  the GPU, with `hypergradOperatorUniform(mix, gain)` +
  `hypergradOperatorDispatch(subgroup)` mirroring the uniform payload and
  workgroup math for WebGPU callers.
- `FractalCanvas::vectorFieldFftUniform(false)` packages the `CanvasFftParams`
  uniform (width, height, inverse flag, padding) as a `Uint32Array` so the WGSL
  kernel can be dispatched without manual byte packing.
- `FractalCanvas::vectorFieldFftLayout()` reports the byte lengths and strides
  for the `FieldSample`/`SpectrumSample` storage buffers plus the uniform block
  so WebGPU callers can allocate resources without hard-coding struct sizes.
- `FractalCanvas::vectorFieldFftDispatch(true)` computes the workgroup triplet
  for the generated WGSL so callers can hand the counts directly to
  `computePass.dispatchWorkgroups(...)` (or the Rust equivalent) without
  duplicating the ceil division logic.
- Use `CanvasProjector::emit_zspace_patch` to fold the canvas state back into
  the fractal scheduler without leaving Rust or allocating intermediate
  buffers.
- Blend chart priors with the new `z_space_barycenter` solver—available in
  Rust (`st_tensor::z_space_barycenter`) and Python (`spiraltorch.z_space_barycenter`)—to
  wire colour energy directly into the Z-space roundtable.
- Follow the barycenter's loss-monotone intermediates and feed them straight into
  the hypergradient tape with `Hypergrad.accumulate_barycenter_path` so the
  optimiser converges along the same Z-space path as the solver.

### Z-space inference autopilot (DLPack + ψ telemetry)

- `spiraltorch.weights_partial_from_dlpack` distils PyTorch/JAX/TF tensors
  captured via DLPack or `spiraltorch.compat.capture` into weighted partials,
  ready for `spiraltorch.infer_with_partials` and `spiraltorch.infer_weights_from_dlpack`.
- `spiraltorch.infer_with_psi` consults `spiraltorch.fetch_latest_psi_telemetry`
  on demand so inference blends metric overrides, Canvas partials, and live ψ
  telemetry without writing glue code.
- `spiraltorch.psi_partial_from_reading` and
  `spiraltorch.psi_partial_from_advisory` expose PSI breakdowns, Hopf regime
  flags, and tuning plans as canonical metrics that pipe directly into
  `ZSpaceInferencePipeline` or `ZSpaceInferenceRuntime`.
- `ZSpaceInferencePipeline(psi=True)` now reuses cached PSI telemetry when
  blending Canvas snapshots, coherence diagnostics, and imported checkpoints—
  ideal for warm-starting a Rust session with PyTorch weights while watching
  the ψ health score in real time.
- Drive the entire workflow from the high-level `SpiralSession` orchestrator in
  Rust (`st_nn::SpiralSession`) or Python (`spiraltorch.SpiralSession`) to pick
  devices, generate rank plans, synthesise barycentres, and align hypergrads via
  intuitive method calls.
- Launch `session.trace(tensor)` to compose non-commutative homotopy flows,
  functor linearisations, recursive barycenter gradients, and \(\infty\)-tower
  projections before calling `.resonate()` (or
  `.resonate_with_hypergrad(hypergrad)`) to surface a
  `DifferentialResonance` snapshot that binds the four differential layers
  together.
- Let the trace synthesise barycentres on demand via
  `trace.with_barycenter_from(weights, densities)` or override the coupling
  matrix with `trace.with_barycenter_with(weights, densities, Some(coupling))`
  before resonating, keeping Z-space orchestration entirely on the session.

### SpiralTorchVision

SpiralTorchVision reinterprets the Z-axis as a perceptual frequency domain,
collapsing it with spectral-window-aware projectors into tensor spaces that any
TorchVision model can consume. Temporal resonance buffers now let `ZSpaceVolume`
perform exponential moving averages across frames, while `MultiViewFusion`
registers camera descriptors so the projector can weight view-specific Z slices
before collapse. The roadmap now leans into generative feedback loops between
SpiralRNN conductors and vision modules. Read the full guide in
[docs/spiraltorchvision.md](../spiraltorchvision.md).).

### Z-space-native graph neural networks

SpiralTorch’s hyperbolic geometry grew around hierarchical graphs. The Rust
kernel, backend abstraction, and Z-space operator families already expose the
building blocks for graph neural networks that keep distortion in check while
scaling to social, molecular, or citation graphs. By composing the existing
SpiralK planners, Z-space resonators, and curvature-aware tensors, new
`st-gnn` layers can stream hypergrad updates through tree-like manifolds as a
first-class citizen in the framework—becoming a third pillar alongside
SpiralTorchRec and SpiralTorchRL. The new `st-nn::gnn` module ships a
`GraphContext` normaliser plus a `ZSpaceGraphConvolution` layer that attaches
hypergrad tapes by default and surfaces per-node flow traces via the telemetry
`GraphFlowTracer`, so graph reasoning can be trained and inspected without
leaving Z-space. The `GraphContextBuilder` allows you to dial in symmetric or
row-stochastic normalisation (and self-loop weighting) per graph before it ever
touches the tape, while the tracer now aggregates energy so higher-level tools
can see how much of the negotiation passed through each layer at a glance.
Those traces plug straight into SpiralTorch’s other pillars: `embed_into_biome`
folds propagated node states into an `OpenCartesianTopos`/`TensorBiome` pair for
RewriteMonad consumers, the flow grid can be painted onto any canvas projector,
and `fold_into_roundtable` promotes the graph manifold as a fourth participant
beside the A/B/C bands. The new `fold_with_band_energy` helper lets you blend a
fresh telemetry report with an existing roundtable split without recomputing the
schedule, keeping graph energy in lock-step with whatever SpiralK already
decided for the batch. Feed those reports into `GraphConsensusBridge` to
generate SpiralK snippets and Above/Here/Beneath multipliers—then hand the
bridge to `ModuleTrainer::enable_graph_feedback` so every optimisation step
absorbs graph telemetry before the SoftLogic weighting fires. The trainer keeps
the SpiralK hint from the last applied digest available via
`ModuleTrainer::graph_hint()`, making it trivial to stream the graph-aware
policy back into SpiralK orchestrators or external dashboards.

### Explainability through hypergrad telemetry

Every hypergrad tape, roundtable consensus log, and ψ telemetry stream doubles
as a geometric audit trail. SpiralTorch can expose these records directly to an
interpretability toolkit that maps gradient flows, consensus splits, and
telemetry spikes back to model behaviour. Visualising these pathways keeps
“why” answers native to Z-space, turning SpiralTorch’s internal instrumentation
into an Explainable AI surface without external probes.

### Multi-modal topos safety envelopes

Open topos guards now ship a unified, multi-modal façade so the same
hyperbolic safety window can simultaneously protect text, audio, vision, graph
and reinforcement-learning reward streams. The new
`st_tensor::MultiModalToposGuard` wraps an existing `OpenCartesianTopos` and
lets you tune per-modality envelopes through lightweight profiles:

```rust
use st_tensor::{
    GraphGuardProfile, ModalityProfile, MultiModalToposGuard, OpenCartesianTopos, RewardBoundary,
};

let topos = OpenCartesianTopos::new(-0.95, 1e-6, 8.0, 256, 16_384)?;
let guard = MultiModalToposGuard::new(&topos)?
    .with_text_profile(
        ModalityProfile::new(32_768, Some(0.35))?.with_permeability(0.25)?,
    )?
    .with_audio_profile(
        ModalityProfile::new(96_000, Some(1.25))?.with_permeability(0.12)?,
    )?
    .with_graph_profile(
        GraphGuardProfile::new(512, 8_192, 64, 1e-3, 0.02, None)?.with_permeability(0.15)?,
    )?
    .with_reward_boundary(RewardBoundary::new(-0.8, 0.8, 0.05)?)?;

let mut rewards = vec![1.2, 0.6, -1.1];
let signal = guard.guard_reward_trace(&mut rewards)?;
if let Some(breach) = signal.upper_breach_index {
    tracing::warn!(breach, "reward trace escaped the safe window");
}
```

Each `ModalityProfile` enforces volume limits and softly saturates values using
a tunable **permeability** so monadic biomes retain breathing room instead of
being hard-clipped at the boundary. `GraphGuardProfile` applies the same
permeable clamp to adjacency weights and tolerates a budget overshoot within
the configured permeability, reporting the overflow through
`GraphGuardReport::edge_overflow`. `RewardBoundary` surfaces the first reward
breach while clamping the trace back inside the permitted envelope. The guard
reports symmetry violations, observed reward ranges, saturation counts, and
edge overflow so downstream monitors can react without recomputing the checks in
higher-level languages.

The guard can now seed both an atlas and a biome that retain the same
permeability envelopes, making it easy to wire multi-modal checkpoints into
longer traversals:

```rust
let mut atlas = guard.atlas();
let mut biome = guard.cultivate_biome();

let mut text = Tensor::from_vec(1, 8, vec![2.5; 8])?;
atlas.guard_text_tensor("atlas_text", &mut text)?;

let vision = Tensor::from_vec(3, 3, vec![1.3; 9])?;
biome.absorb_vision("biome_vision", vision)?;
let canopy = biome.canopy()?;
tracing::info!(volume = atlas.visited_volume(), shoots = biome.len());
```

`MultiModalAtlas` shares `ToposAtlas` telemetry like `visited_volume` and
`remaining_volume` while applying modality-aware rewrites before the atlas guard
fires, so downstream geometry keeps a consistent notion of traversal depth.
`MultiModalBiome` keeps the same permeability when absorbing shoots, meaning any
monadic collapse through `canopy()` stays in lock-step with the atlas and guard
without re-deriving modality constraints.

### Autotune telemetry for the WGPU-first roadmap

SpiralTorch now ships an autotuning registry and bounded telemetry log so the
WGPU backend can remember which tile schedules performed best on each device.
We encode the hardware fingerprint using vendor, numeric device ID, subgroup
size, shared-memory budget, and driver revision, then splice in the shader
revision plus op signature to form a stable cache key—no timestamps or host
process details required. The same log
tracks throughput, bandwidth, occupancy, chosen tile, and regression fallbacks
while evicting the oldest samples once the per-key capacity is reached, keeping
the cache warm without unbounded growth.

### Microlocal interface gauges

SpiralTorch’s theory core now hosts a microlocal boundary gauge that translates
the BV/varifold correspondence directly into code. The new
`st_core::theory::microlocal::InterfaceGauge` measures local total-variation
density over shrinking metric balls, outputs the gauge-invariant `R` machine,
and only reconstructs oriented normals when an external label `c′` is supplied.
This lets SpiralTorch stabilise interface detection, switch on co-orientations
precisely when downstream pipelines inject a label, and keep curvature-ready
statistics without violating the gauge symmetry of the unlabeled limit.
Once those signatures exist, `InterfaceZLift` pushes them straight into
Z-space: it projects the perimeter mass onto a preferred Z-axis, splits the
energy into Above/Here/Beneath bands, enriches the drift with the Leech
projector, and emits a ready-to-store `SoftlogicZFeedback` pulse so runtimes can
bias their collapse heuristics without leaving the microlocal picture.

Beyond the perimeter statistics the gauge now reports a gauge-invariant
mean-curvature magnitude and, whenever `c′` fixes an orientation, the signed
mean curvature that restores the co-oriented BV picture. Collapse loops can
therefore gate on curvature without labels, and only light up the signed
variant once a label or co-orientation becomes available.

To make those bridges operational inside collapse loops the gauge now supports
multi-radius sweeps and a conductor that fuses their Z pulses with exponential
smoothing. `InterfaceGauge::analyze_multiradius` probes the same mask at
different blow-up scales (and reuses an optional `c′` label when supplied),
while `InterfaceZConductor` drives any number of gauges, aggregates the
resulting pulses, and hands back a `ZFused` packet with attribution weights and
event tags alongside the smoothed `SoftlogicZFeedback` record so runtime loops
can see which layer dominated the decision.
`MicrolocalGaugeBank` turns that loose collection into a pluggable registry.
It stores named `InterfaceGauge`s, offers builder-style helpers to register or
remove probes, runs batch analysis keyed by id, and hands the resulting lineup
directly to the conductor so runtime code can swap probe sets without rewriting
fusion logic.
`InterfaceZConductor::step` now preserves those identifiers, returning an
`InterfaceZReport` that bundles the raw `InterfaceSignature`s alongside the
matching ids and a cloned lift so downstream consumers can reuse the same
projection without re-running the gauges.
`MacroTemplateBank` mirrors that registry pattern for macro-scale designs: it
keeps named `MacroModelTemplate`s, accepts cards directly, and couples the whole
lineup to an `InterfaceZLift` to emit a bridge bank so macro kinetics can travel
with whatever microlocal gauges are currently wired into the conductor.
It can then call `drive_matched` to produce macro drives only for the gauges
present in the latest report and merge their microlocal feedback via
`feedback_from_report` before piping the result back into the conductor.

The conductor can now blend the pulses in both time and frequency: `set_frequency_config`
installs a power-of-two FFT window and per-source spectral gains so high-frequency
microlocal gradients or low-frequency desire trends can be emphasised without a
second pass, while `set_adaptive_gain_config` keeps a per-source reliability
score and nudges their gains on-line until the fused drift stabilises. A new
`set_latency_config` alpha–beta aligner adjusts timestamps using the reported
latency and emits `latency-*` events whenever the offsets are learnt or
corrected, keeping Maxwell’s block pulses in lockstep with microlocal frames.
Tests cover the spectral weighting, the adaptive loop, and the latency alignment
so the new knobs keep their invariants.

Desire loops pick up the fused Z feedback straight from the hub: the conductor
stores the latest `SoftlogicZFeedback`, and the temperature controller now
accepts that pulse to raise exploration when the drift jitters and cool the
distribution when the Z-bias settles. The default controller keeps a short
memory of recent flips and exposes `with_feedback` so runtimes can tweak the
feedback gain without rebuilding the desire machinery.

### Collaborative canvas telemetry and ghost trails

The hardened collaboration stack now layers spectator-grade UX on top of the
BroadcastChannel/localStorage bridge. Every pointer broadcast feeds a
policy-aware ghost trail buffer (`pointerTrail` events plus the
`getPointerTrail` helper) so dashboards can draw fading cursors without keeping
their own queues.
At the same time a bounded timeline recorder captures every pointer, patch, and
full-state message with Lamport clocks and origins so HUDs can play back the
last few seconds of collaboration or splice the data into attribution feeds via
`session.replay`.
The WASM README now documents the new knobs (`pointerTrailMs`,
`replayWindowMs`, `replayMaxEntries`) and ships usage snippets for replaying
frames or painting ghost trails, making it trivial to showcase collaborative
Z-space canvases right from the top-level docs.

### Maxwell-coded envelopes meet SpiralK

The coded-envelope utilities now ship with a `MaxwellSpiralKBridge` that turns
sequential Z pulses into KDSl snippets ready for the runtime. Every channel name
is sanitised for SpiralK, weights adapt to the observed Z magnitude, and
existing programs can be prepended so the hints extend a live policy rather than
replace it. Call
`push_pulse(channel, &pulse)` for each stream, then `script()` to emit the
combined `soft(maxwell.bias, …)` rules that SpiralK can ingest without custom
glue code. The workflow is
documented in the refreshed Maxwell technical note, which now includes a
section on streaming detections back into SpiralK orchestration.

Want the language desire loops to see the same detections? Enable the PSI
feature and run the new `MaxwellPsiTelemetryBridge`. It converts each pulse into
a PSI reading, optional band-energy threshold events, and a `SoftlogicZFeedback`
sample so `DesirePsiBridge` captures the Z drift alongside ψ totals without
hand-written glue.
Pair it with `MaxwellDesireBridge` to translate the very same pulse into a
concept window that the `DesireLagrangian` can consume, aligning coded-envelope
channels with vocabulary slots on the fly.

### Quantum Reality Studio overlays

The new `st-qr-studio` crate spins up a **QuantumRealityStudio** that records
Maxwell pulses, emits concept windows, and stitches narrative tags into VR/AR
overlays. Signal capture sessions enforce which laboratory rigs may publish
pulses, semantic taggers mirror the `MaxwellDesireBridge` lexicon, and overlay
frames surface glyph/intensity pairs for immersive projection. The crate now
re-exports `MaxwellPulse` (an alias for `MaxwellZPulse`) and ships overlay
builders such as `OverlayFrame::from_pairs`/`::from_glyphs_and_intensities` so
AR pipelines can zip glyph and intensity streams without writing manual
plumbing. Storyboard exports drop directly into
`tools/qr_storyboard.py`, which converts JSON/NDJSON captures into Markdown decks
grouped by channel for Desire roundtables while weaving overlay glyph stacks,
meta-narrative tags, causal ancestry, concept windows, and meaning-sheaf
signatures into a highlights column for reviewers. The companion
[Quantum Reality Playbook](../qr_playbook/README.md) provides rituals,
collaboration tips, and art-direction cues so research and cultural teams stay
synchronised.

Latest iterations expose `QuantumRealityStudio::record_pulse` so capture rigs can
stash `RecordedPulse` snapshots prior to narration, while
`infer_concept_window` and `emit_concept_window` transform either raw records or
streamed frames into serialisable concept windows suited for AR overlays and
Desire loops. `OverlayGlyph` powers `OverlayFrame::new`, while the new
convenience constructors accept glyph/intensity pairs directly and the
storyboard exporter now retains overlay stacks, narrative tags, and concept
window weights so AR HUDs can replay exactly what collaborators saw without
deriving those assets a second time.

### Semiotic suturing, desire control, and EGW bridges

SpiralTorch now ships a native semiotic optimiser that compresses Lacanian
language machinery into Z-space numerics. The `st-nn::language::DesireLagrangian`
implements the closed-form update

\[
\pi_t^*(j) \propto q_\theta(j\mid h_t)^{1/T_t}\exp\{\alpha_t\log K_{\text{syn}} + \beta_t\log K_{\text{par}} - \lambda r_j + \gamma_t g_t(j)\},
\]

so syntagmatic/ paradigmatic couplings, repression scores, and S→s drives land
as a single additive logit injection.
The `TemperatureController` keeps desire aligned with a target entropy, while
the lightweight Schrödinger lookahead adds one-to-two Doob iterations directly
from the Z-space kernels to approximate the bridge in-line with training.

Symbol/meaning suturing arrives via an entropic Gromov–Wasserstein solver that
enforces anchors and floaty signifiers together. `EntropicGwSolver` estimates
the coupling \(\Pi\) by minimising the EGW objective with Sinkhorn-style
updates, boosting anchor pairs, and handing back a `SemanticBridge` ready for
token-to-concept expectations across the tape.
Feed that bridge into the desire Lagrangian and you obtain a turn-key workflow:

1. Build sparse syntagmatic/ paradigmatic kernels and repression vectors, then
   estimate \(\Pi\) with the EGW solver (optionally seeding anchor pairs).
2. Initialise the `DesireLagrangian` with those artefacts and wire it to a
   SpiralK loop or roundtable injector.
3. Stream LM logits through `step(...)` or the phase-aware `step_with_scheduler(...)`
   to receive logit offsets, entropy telemetry, and temperature updates that
   honour the S/s suturing and desire budget.

Configure desire as a three-stage routine by combining the provided schedule
helpers. `warmup(...)` handles the observation phase (desire starts at zero and
logs avoidance), ramping towards the interference window where `alpha` nudges
avoided terms while `beta/γ` remain gentle. Once the warmups complete the
integration phase kicks in, coupling desire with the Z-space barycenter and
surfacing a hypergrad penalty that measures drift from the barycentric anchor.
For example:

```rust
let mut desire = DesireLagrangian::new(geometry, repression, semantics, controller)?
    .with_alpha_schedule(warmup(0.0, 0.1, 400))
    .with_beta_schedule(warmup(0.0, 0.05, 800))
    .with_gamma_schedule(constant(0.02))
    .with_lambda_schedule(constant(0.08));

let report = desire.step_with_scheduler(&logits, previous_token, &concept_hint)?;
match report.phase {
    DesirePhase::Observation => log_observation(report.avoidance),
    DesirePhase::Injection => reinforce_desire(report.logit_offsets),
    DesirePhase::Integration => hypergrad.push_penalty(report.hypergrad_penalty),
}
```

`DesireAvoidanceReport` exposes the dominant repressed tokens collected during
observation, while the integration phase emits the barycentric drift so a
hypergrad or self-rewrite scheduler can keep desire centred without collapse.
The schedules default to zeroed observation and grow-only ramps, so existing
callers can continue to provide manual `DesireWeights` without opt-in changes.

Every step now ships a `DesireGradientControl` alongside the interpretation so
automation layers can react without recomputing heuristics. Grab it via
`DesireLagrangian::gradient_control()` (or directly from the streamed
`DesireSolution`) to inspect the recommended hyper/Realgrad learning-rate
scales, penalty gains, and WGSL operator mix/gain before issuing GPU updates.
The control packet also captures Desire's “feel-good” tuning: exponential
learning-rate gains driven by entropy deltas (with min/max bounds and slew
limits), EMA-smoothed clipping windows anchored at the 95th percentile, Z-space
temperature coupling (`κ`) guidance, and sigmoid quality scaling hooks so
Maxwell/Microlocal evidence can raise the step size only when the gradients look
clean. Each packet carries a telemetry bitmask plus string labels (e.g.
`lr_increase`, `lr_clipped`, `temperature_suppress`, `quality_suppress`,
`lr_slew_limit`) so PSI dashboards can log _why_ the controller nudged Desire in
a given direction, and the new `control_events` field on `DesireSolution` keeps
historical replays compatible with older logs via the serde default.

When you want to feed live telemetry back into Desire, use the new
`DesireGradientControl::control_with_gain()` builder. It mirrors the ergonomic
sketch above—pipe the latest entropy estimate, Z magnitude, quality score, and
clip hints directly into the builder and call `finalise()` to obtain the packed
control:

```rust
let ctrl = DesireGradientControl::control_with_gain()
    .with_gain(gain_factor)
    .with_entropy(last_entropy)
    .with_z_coupling(z_magnitude)
    .with_quality(quality_estimate)
    .with_bounds(1e-4, 3e-3)
    .with_clip_p95_hint(p95_gradient)
    .finalise();
lag.set_gradient_control(ctrl.clone());
```

For GPU loops, call `CanvasProjector::desire_control_uniform` (or the WASM
`FractalCanvas.desireControlUniform`) to obtain a `Uint32Array` view of the
packed uniform. Each lane stores the IEEE-754 bits for the target entropy,
learning-rate envelope, clipping window, Z coupling, quality gain, and rate
scales, with the final element containing the raw telemetry mask. Reinterpret
the buffer as a `Float32Array` when uploading to WGSL so the compute shader sees
the expected 16-lane, 64-byte-aligned payload without reserialising structs for
every dispatch.

To automate the “unconscious” loop, wrap the lagrangian with
`DesireAutomation`. It samples the `SelfRewriteCfg` thresholds, tracks
hypergrad drift during the integration phase, and emits
`DesireRewriteTrigger` structures once enough evidence accumulates. Each
trigger carries the normalised avoidance vector so a SpiralK
`self-rewrite` or hypergrad scheduler can queue barycentric nudges without
hand-crafted heuristics.

```rust
use st_core::config::self_rewrite::read_cfg;
use st_nn::language::{DesireAutomatedStep, DesireAutomation};
use std::time::Instant;

let cfg = read_cfg();
let mut automation = DesireAutomation::new(desire, cfg);
let DesireAutomatedStep { solution, trigger } = automation
    .step(&logits, previous_token, &concept_hint, Instant::now())?;
if let Some(event) = trigger {
    spiralk_scheduler.queue_desire(event.report, event.mean_penalty);
}
```

`read_cfg()` now pulls from layered configuration files so operators can
separate defaults, site overrides, and run-time experiments without rebuilding
or touching environment variables. By default the loader merges
`~/.spiraltorch/config/base.toml`, `site.toml`, and `run.json` (in that order),
falling back to `~/.spiraltorch` when the `config/` directory is absent. Each
layer is optional—missing files are ignored—and environment variables such as
`SPIRAL_CONFIG_ROOT`, `SPIRAL_CONFIG_BASE`, `SPIRAL_CONFIG_SITE`, and
`SPIRAL_CONFIG_RUN` can redirect the loader to alternate locations. Per-run
JSON overrides make it easy to script experiments (for example via
`run.json` produced by an orchestrator) while keeping persistent defaults in
TOML. Every merge emits a diff event that records which keys changed and their
before/after values.

Python callers can retrieve the same diff stream via
`spiraltorch.get_config_events()`, which returns dictionaries of
`{"layer": "run", "path": "desire.self_rewrite.score_thresh", "previous": 0.02, "current": 0.05}`.
Point the module at alternate config roots (for example when replaying a site
profile) by exporting the environment variables before importing
`spiraltorch` so the layered loader observes the overrides.

Persist the stream to disk with `DesireLogbook` so the observation/injection/
integration cadence can be replayed later or shared with SpiralK rewrite
automation. The logbook writes line-delimited JSON records that contain the
entire `DesireSolution` payload plus any emitted triggers, keeping telemetry and
avoidance vectors together for offline inspection. Re-opening the logbook will
resume the ordinal counter automatically, so a long-running automation loop can
be restarted without clobbering record IDs.

```rust
use st_nn::language::{DesireAutomatedStep, DesireAutomation, DesireLogbook};
use std::time::{Instant, SystemTime};

let mut logbook = DesireLogbook::new("desire.ndjson")?;
let DesireAutomatedStep { solution, trigger } = automation
    .step(&logits, previous_token, &concept_hint, Instant::now())?;
logbook.record(&DesireAutomatedStep { solution, trigger }, SystemTime::now())?;
```

Stream the persisted decisions back with `DesireLogReplay` to build dashboards
or off-line analytics. The iterator skips blank lines and surfaces every record
as a `PureResult`, making it straightforward to plug into telemetry sinks or
trainers that ingest JSONL traces.

```rust
use st_nn::language::DesireLogReplay;

for entry in DesireLogReplay::open("desire.ndjson")? {
    let record = entry?;
    audit(record.ordinal, record.solution.phase);
}
```

Once the raw telemetry exists, braid it directly into automation, logging, and
rewrite hooks with the `DesirePipeline`. The pipeline fans each automated step
out to any number of sinks—logbooks, trigger buffers, SpiralK bridges—so graph
tooling, language desire, and self-rewrite loops evolve together without custom
glue. Attach a
`DesireTriggerBuffer` to capture emitted rewrite events while the logbook keeps
the JSONL trace alive, and optionally replay historical automation into new
consumers:

```rust
use st_nn::language::{
    DesireLogReplay, DesireLogbook, DesirePipeline, DesireTriggerBuffer,
};

let trigger_buffer = DesireTriggerBuffer::new();
let mut pipeline = DesirePipeline::builder(automation)
    .with_logbook(DesireLogbook::new("desire.ndjson")?)
    .with_sink(trigger_buffer.clone())
    .build();

let step = pipeline.step_realtime(&logits, previous_token, &concept_hint)?;
if let Some(trigger) = &step.trigger {
    spiralk_scheduler.queue_desire(trigger.report.clone(), trigger.mean_penalty);
}

pipeline.flush()?;
let replayed = pipeline.replay(DesireLogReplay::open("desire.ndjson")?)?;
let drained = trigger_buffer.drain()?; // forward to analytics or trainers
```

Python bindings currently expose **desire-derived gradient control** primitives
(via `Hypergrad.desire_control(...)` / `desire_interpretation(...)`). The full
`DesirePipeline` builder + logbook sinks remain Rust-first today.

```python
import spiraltorch as st

hyper = st.Hypergrad(curvature=-0.9, learning_rate=0.05, rows=1, cols=4)
real = st.Realgrad(learning_rate=0.01, rows=1, cols=4)
try:
    pred = st.Tensor(1, 4, [0.2, 0.1, 0.3, 0.4])
    tgt = st.Tensor(1, 4, [0.0, 1.0, 0.0, 0.0])

    hyper.accumulate_pair(pred, tgt)
    real.accumulate_pair(pred, tgt)

    real_summary = real.summary()
    control = hyper.desire_control(real_summary)
    interp = hyper.desire_interpretation(real_summary)

    print("operator mix/gain:", control.operator_mix(), control.operator_gain())
    print("pressure/balance:", interp.hyper_pressure(), interp.balance())
finally:
    hyper.reset()
    real.reset()
```

When you need to splice the stream into other runtimes, attach a
`DesireChannelSink` via `with_channel`. It emits `DesirePipelineEvent`s over a standard channel so
rewriters, trainers, or async dashboards can subscribe without bespoke glue—each
step arrives before any trigger for the same timestamp, preserving ordering for
downstream automata.

```rust
use std::sync::mpsc::channel;
use st_nn::language::{
    DesirePipeline, DesirePipelineEvent, DesireTriggerBuffer,
};

let (sender, receiver) = channel();
let mut pipeline = DesirePipeline::builder(automation)
    .with_channel(sender)
    .with_sink(DesireTriggerBuffer::new())
    .build();

let step = pipeline.step_realtime(&logits, previous_token, &concept_hint)?;
for event in receiver.try_iter() {
    match event {
        DesirePipelineEvent::Step { step, timestamp } => {
            audit(step.solution.phase, timestamp)
        }
        DesirePipelineEvent::Trigger { trigger, .. } =>
            spiralk_scheduler.queue_desire(trigger.report.clone(), trigger.mean_penalty),
    }
}
```

Global telemetry consumers can subscribe without owning the pipeline by adding
`with_telemetry()`. The `DesireTelemetrySink` records every step’s phase,
temperature, avoidance energy, and schedule weights into the shared telemetry
hub so trainers, notebooks, or external services can poll the latest state via
`get_last_desire_step`.

```rust
use st_core::telemetry::hub;
use st_nn::language::DesirePipeline;

let mut pipeline = DesirePipeline::builder(automation)
    .with_telemetry()
    .build();

let _ = pipeline.step_realtime(&logits, previous_token, &concept_hint)?;
if let Some(sample) = hub::get_last_desire_step() {
    println!("phase {:?} at T={:.3}", sample.phase, sample.temperature);
}
```

Training loops can now subscribe directly. Clone a `DesireTrainerBridge`, attach
it with `with_trainer_bridge`, and hand the same bridge to `ModuleTrainer` via
`enable_desire_pipeline`. Each step drains into a shared summary so the trainer
records phase counts, mean desire weights, and trigger temperatures alongside
band energy telemetry without custom glue.

```rust
use st_nn::language::{
    ConceptHint, DesirePipeline, DesireTrainerBridge, DesireTriggerBuffer,
};
use st_nn::trainer::ModuleTrainer;

let bridge = DesireTrainerBridge::new();
let mut pipeline = DesirePipeline::builder(automation)
    .with_trainer_bridge(&bridge)
    .with_sink(DesireTriggerBuffer::new())
    .build();

trainer.enable_desire_pipeline(bridge.clone());
let step = pipeline.step_realtime(&logits, previous_token, &concept_hint)?;
if let Some(trigger) = &step.trigger {
    println!("trigger mean penalty: {:.3}", trigger.mean_penalty);
}
```

Graph telemetry can join the same braid. Instantiate a `GraphFlowTracer`, feed it
into `GraphConsensusBridge`, and wrap the result with `DesireGraphBridge`. Every
desire step now captures the latest graph digest, letting you aggregate
Z-space desire entropy with SpiralK’s quad-band consensus or replay graph
shares into analytics dashboards via `DesireGraphSummary`.

```rust
use std::sync::{Arc, Mutex};
use st_core::telemetry::xai::{GraphFlowTracer, NodeFlowSample};
use st_nn::language::{DesireGraphBridge, DesirePipeline};
use st_nn::{BandEnergy, GraphConsensusBridge};

let tracer = Arc::new(Mutex::new(GraphFlowTracer::new()));
let graph_bridge = DesireGraphBridge::new(
    GraphConsensusBridge::new(tracer.clone()),
    BandEnergy { above: 0.4, here: 0.35, beneath: 0.25, drift: 0.0 },
);
let mut pipeline = DesirePipeline::builder(automation)
    .with_graph_bridge(&graph_bridge)
    .build();

// capture graph flows alongside desire automation
tracer.lock().unwrap().begin_layer(
    "gnn::spiral",
    -1.0,
    vec![NodeFlowSample { node_index: 0, incoming_weight: 1.0, aggregated_norm: 0.6 }],
);

let step = pipeline.step_realtime(&logits, previous_token, &concept_hint)?;
if let Some(summary) = graph_bridge.drain_summary()? {
    for (layer, share) in summary.layer_support {
        println!("graph layer {layer} captured {:.2}% of energy", share * 100.0);
    }
}
```

Roundtable consensus can now absorb desire impulses directly. Attach a
`DesireRoundtableBridge` and the pipeline will export Above/Here/Beneath
multipliers plus drift adjustments every step. Drain the summary or let
`ModuleTrainer::enable_desire_roundtable_bridge` fold it into the optimiser so
the three-way negotiation constantly reflects the latest semiotic pressure.

```rust
use st_nn::language::{DesirePipeline, DesireRoundtableBridge};

let bridge = DesireRoundtableBridge::new().with_blend(0.45);
let mut pipeline = DesirePipeline::builder(automation)
    .with_roundtable_bridge(&bridge)
    .build();

for (step, logits) in logits_stream.enumerate() {
    let now = Instant::now();
    let timestamp = SystemTime::now();
    pipeline.step_at(&logits, step % vocab, &concept_hint, now, timestamp)?;
}

if let Some(summary) = bridge.drain_summary()? {
    println!("desire barycentric → Above {:.3}", summary.mean_above);
}
```

Inside the trainer simply call:

```rust
trainer.enable_desire_roundtable_bridge(bridge.clone());
```

Every optimisation step now reports `desire_roundtable_*` metrics, while
`ModuleTrainer::desire_roundtable_summary()` returns the most recent aggregate so
Python notebooks can watch the unconscious drift in real time.

Need all three braids at once? `DesireTelemetryBundle` wires the trainer,
roundtable, and ψ bridges together so experiments can attach a single bundle to
both the pipeline and optimiser.

```rust
use st_nn::language::{
    DesirePipeline, DesirePsiBridge, DesireRoundtableBridge, DesireTelemetryBundle,
    DesireTrainerBridge,
};

let trainer_bridge = DesireTrainerBridge::new();
let roundtable_bridge = DesireRoundtableBridge::new();
let psi_bridge = DesirePsiBridge::new();
let bundle = DesireTelemetryBundle::new()
    .with_trainer_bridge(&trainer_bridge)
    .with_roundtable_bridge(&roundtable_bridge)
    .with_psi_bridge(&psi_bridge);

let mut pipeline = DesirePipeline::builder(automation)
    .with_telemetry_bundle(&bundle)
    .build();

trainer.enable_desire_telemetry(&bundle);
```

Python note: `DesireRoundtableBridge` / ψ telemetry bridges are not yet exposed in
the published wheel. For notebook-friendly feedback today, use
`Hypergrad.desire_control(...)` (shown above).

ψ telemetry can ride the same braid. Attach a `DesirePsiBridge` to fold the
latest `PsiMeter` readings, SoftLogic Z feedback, and threshold crossings into
the automation stream. The bridge can be drained directly or wired into
`ModuleTrainer::enable_desire_psi_bridge` so every optimisation step records the
aggregated ψ view alongside desire entropy and graph consensus.

```rust
use st_core::telemetry::hub;
use st_core::telemetry::psi::{PsiComponent, PsiReading};
use st_nn::language::{DesirePipeline, DesirePsiBridge};

let psi_bridge = DesirePsiBridge::new();
let mut pipeline = DesirePipeline::builder(automation)
    .with_psi_bridge(&psi_bridge)
    .build();

// seed hub telemetry before each step (normally done by ModuleTrainer)
let mut breakdown = std::collections::HashMap::new();
breakdown.insert(PsiComponent::LOSS, 0.9);
let reading = PsiReading { total: 0.9, breakdown, step: 1 };
hub::set_last_psi(&reading);

let step = pipeline.step_realtime(&logits, previous_token, &concept_hint)?;
if let Some(summary) = psi_bridge.drain_summary()? {
    println!("ψ mean total: {:.3}", summary.mean_psi_total);
}
```

The result is a single Rust-native control surface that marries KL control,
Schrödinger bridges, and entropic GW into SpiralTorch’s Z-space, ready to steer
language modules, rewrite monads, or SpiralK trainers without bespoke Python
glue.
