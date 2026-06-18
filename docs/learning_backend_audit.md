# Learning Backend Audit

Date: 2026-06-14

This note tracks the Rust backend surfaces that matter most for turning
SpiralTorch into a serious learning stack. The focus is not raw kernel breadth
alone, but whether backend choice, tensor execution, gradient accumulation,
trainer traces, and distributed synchronization form one observable contract.

## Current Dataflow

The present learning path is split across several crates:

- `examples/_shared/backend.rs` parses `auto`, `cpu`, `wgpu`, `mps`, `cuda`, and
  `hip` into `DeviceCaps`.
- `st-core::backend::runtime_probe` keeps MPS honest as a placeholder and maps
  it to WGPU or CPU planner caps.
- `st-nn::trainer::ModuleTrainer` stores `DeviceCaps` through `RankPlanner` and
  uses them for roundtable schedule decisions, not for forcing tensor kernels.
- `st-nn::module::Module` exposes `preferred_device()`, but `forward()` and
  `backward()` do not receive a backend execution context.
- Dense learning layers such as `Linear`, GNN layers, and RNN projections mostly
  call `Tensor::matmul_prepacked()` or `Tensor::matmul()`.
- `st-tensor::Tensor` then chooses WGPU, HIP, CPU SIMD, faer, or naive kernels
  from local `Auto` heuristics.
- `spiral-selfsup::trainer::TrainingDevice` provides distributed gradient and
  metric synchronization and now bridges into `ModuleTrainer` parameter
  accumulators through `st-core::distributed::AccumulatorSynchronizer`.

That means the stack can request WGPU-first at the planner or example level and
can synchronize parameter accumulators before optimizer steps, but actual tensor
execution still needs trace evidence to distinguish policy from real kernels.

## Findings

### P0: Make backend execution observable in every learning trace

Backend selection is currently a wish unless traces show the selected tensor
kernel. This change starts that path by emitting `TensorOpMeta` for learning
critical dense paths:

- `matmul` and `matmul_prepacked`
- high-order differential functor lifts: `FunctorDifferential::apply`,
  `apply_to`, and `linearise` now call `matmul_with_backend(..., Auto)` instead
  of the legacy plain `matmul()` wrapper, so theoretical/differential probes
  still emit backend-routed dense metadata.
- fused projections: `matmul_bias_relu`, `matmul_bias_gelu`,
  `matmul_bias_add_relu`, and `matmul_bias_add_gelu`
- GELU backward
- normalization backward reductions: `batch_norm_backward`,
  `zspace_batch_norm_backward`, `layer_norm_backward`, and
  `zspace_layer_norm_backward`
- GNN graph-level readout reductions: `graph_readout` and
  `graph_readout_backward`
- supervised loss reductions: MSE, categorical cross entropy, hyperbolic cross
  entropy, focal, contrastive, and triplet loss forward/backward paths
- core CPU tensor elementwise/layout/reduction/copy paths used by backward and
  optimizers: `add`, `sub`, `scale`, `hadamard`, `add_scaled`,
  `add_row_inplace`, `relu_inplace`, `gelu_inplace`, `transpose`, `reshape`,
  `sum_axis0`, `sum_axis1`, `cat_rows`, and the composite
  `mean_squared_error` reduction
- scalar diagnostics and Z-space projections that influence learning policy:
  `squared_l2_norm`, `project_to_poincare`, and `hyperbolic_distance`. These
  are still CPU scalar/projection paths, but now show up as
  `diagnostic_reduction`, `hyperbolic_projection`, or `hyperbolic_distance`
  metadata instead of disappearing behind coherence, optimizer, and Z-space
  module decisions.
- Z-space coherence policy reductions: `coherence_measure_phases` now reports
  actual `cpu` execution separately from the requested `CoherenceBackend`
  (`pure_rust`, `webgpu`, `cufft`, etc.), plus channel count, entropy,
  dominant weight, curvature/backend bias, and linguistic profile count. The
  trainer backend collector treats accelerated coherence requests that still
  run as CPU as fallback evidence.
- topology/biome reductions: `tensor_biome_absorb_weighted`,
  `tensor_biome_renormalise_weights`, and `tensor_biome_canopy` now make
  shoot counts, raw/normalised total weights, canopy shape, and topos curvature
  visible. Biome weight renormalisation is `control_cpu`/`host`, tensor absorb
  records `topos_cpu`/`auto`, and canopy collapse records `backend=hybrid` with
  explicit tensor-util accumulation/normalisation labels (`auto`, `cpu`, or
  `wgpu`) plus an explicit `rewrite_backend` of `topos_cpu`. These events now
  also carry `route_blocker`, rewrite/guard modes, and estimated accumulation,
  normalisation, rewrite, and stored-value counts, so topos CPU work is visible
  as guarded rewrite/control rather than a generic backend fallback.
  `ZSpaceProjector::reimport_biome()` therefore surfaces both the routeable
  tensor-collapse part and the remaining topos rewrite/control island before
  the following Poincare projection.
- sequencer raw-slice diagnostics: `zspace_semantic_window` and
  `zspace_maxwell_pulse_summary` now expose semantic-window token/channel
  widths, skipped/normalised weights, dominant window weight, Maxwell pulse band
  energy, z-bias, and standard error. Maxwell pulse metadata is
  `summary_cpu`/`host` trace-only state. Semantic-window early exits for empty
  coherence or zero-token runs are now `semantic_control_cpu`/`host` and
  excluded from CPU-debt residuals, while non-empty semantic-window scans now
  report `backend=hybrid` with `window_energy_backend` and
  `distribution_scale_backend` tensor-util labels (`auto`, `cpu`, or `wgpu`).
  The host-side token-slice extraction is called out separately as
  `window_energy_extraction_backend=semantic_cpu`, while the mean-absolute
  energy reduction itself now uses `Tensor::sum_abs_with_backend`.
- semantic distribution reductions: `zspace_semantic_distribution` and
  `zspace_semantic_distribution_fusion` now expose window/fusion dimensions,
  distribution mass, dominant probability, entropy, and uniform-fallback status
  before downstream desire or geometry hooks consume the result. Non-empty
  semantic distribution inference now reports `backend=hybrid`, points to the
  nested `semantic_bridge_window_distribution` event, and splits the remaining
  sparse row lookup (`semantic_sparse_scan_backend=semantic_cpu`) from routeable
  accumulation and distribution scaling. Fusion similarly splits
  `backend=hybrid` into routeable `fusion_accumulation_backend` and
  `distribution_scale_backend` tensor-util labels. The CPU work left here is
  sparse semantic lookup, empty-window control, and host input preparation, not
  the add/scale accumulation kernels.
- Lawvere probability guards: `lawvere_guard_probability_slice` now reports raw,
  guarded, renormalised, and final mass plus non-finite/negative/clipping repair
  counts for the topos-level probability slices used by barycenters, biome
  weights, and other Z-space reductions. These guards are labelled
  `control_cpu`/`host` so residual comparisons do not confuse scalar probability
  repair with routeable tensor kernels.
- language probability reductions: `desire_softmax`, `desire_normalise`,
  `gw_marginal_normalise`, `gw_marginal_normalise_in_place`, and
  `sparse_kernel_probability_row` now make Desire top-k probability formation,
  avoidance-bias normalisation, GW marginal repair, Sinkhorn row/column
  normalisation, and sparse semantic-kernel smoothing visible. Desire softmax,
  non-zero normalise, self-rewrite desire-vector normalise, GW marginal
  normalise, and sparse semantic-kernel rows now split into `backend=hybrid`,
  CPU fallback/validation islands, and tensor-util labels (`auto`, `cpu`, or
  `wgpu`) for routeable softmax, sanitise, positive-mass summation, and
  distribution scaling. Finite `desire_softmax` logits use
  `Tensor::row_softmax_with_backend`, finite `desire_normalise` and
  self-rewrite `desire_automation_vector_normalise` sanitise use
  `Tensor::relu_with_backend`, and finite non-negative GW/sparse row masses use
  `Tensor::sum_abs_with_backend`; non-finite repair remains a CPU probability
  path. Zero/empty fallbacks still report `probability_cpu`. These are
  intentionally kept in residual comparisons because large-vocabulary or dense
  semantic-kernel sweeps may turn probability formation into real language-model
  training work.
- cross-crate semantic/PSI probability reductions:
  `concept_hint_distribution`, `semantic_bridge_window_distribution`,
  `psi_heatmap_distribution`, and feature-gated `psychoid_softmax` now expose
  downstream concept-hint repair, semantic-window inference, PSI band-energy
  normalisation, and st-core psychoid KL softmax. Concept-hint distributions
  and semantic bridge window inference now split non-empty semantic-window
  paths into `backend=hybrid`, sparse semantic row lookup
  (`semantic_sparse_scan_backend=semantic_cpu`), routeable dense concept
  accumulation (`semantic_accumulation_backend` as `auto`, `cpu`, or `wgpu`),
  and `distribution_scale_backend` tensor-util labels. Finite concept-hint
  distributions now route non-negative sanitise through `Tensor::relu_with_backend`
  (`semantic_sanitize_backend` as `auto`, `cpu`, or `wgpu`), while non-finite
  repair and sparse window lookup remain explicit semantic CPU sub-backends. PSI
  heatmap analysis remains `summary_cpu`/`host`, so trainer residuals do not
  confuse branch-level heatmap summaries with routeable tensor kernels. Trainer
  trace collection now records the sparse lookup explicitly as keys such as
  `tensor_op_backend_zspace_semantic_distribution_semantic_sparse_scan_semantic_cpu`,
  `tensor_op_backend_semantic_bridge_window_distribution_semantic_sparse_scan_semantic_cpu`,
  and `tensor_op_backend_concept_hint_distribution_semantic_sparse_scan_semantic_cpu`;
  these stay in CPU debt while the paired accumulation/sanitise/scale keys can
  move to `wgpu`.
- automation/diffusion/Z-RBA normalisation: `desire_automation_vector_normalise`,
  `concept_diffusion_state_normalise`, and `zrba_metric_weights_normalise` now
  surface self-rewrite desire-vector mass, concept diffusion state mass, and
  Z-RBF metric-weight normalisation without adding per-token or per-edge trace
  explosions inside attention kernels. Desire automation non-zero finite
  sanitise and scaling now follow tensor-util routing; concept diffusion is now
  labelled as an explicit `f64_cpu` precision island because its state is
  `nalgebra::DVector<f64>`, with `precision_backend=f64_cpu`,
  `state_sum_backend=f64_cpu`, `distribution_scale_backend=f64_cpu`, and
  `route_blocker=f64_state` called out explicitly. Tiny Z-RBF metric-weight normalisation remains
  `control_cpu`/`host`.
- Z-RBA attention diagnostics: `zrba_workspace_softmax_summary` now reports the
  CPU row-wise workspace softmax/entropy loop per attention head, including row
  mass, zero-sum rows, dominant probability, and entropy aggregates without
  emitting per-row events.
- st-core telemetry reducers: feature-gated psychoid reducers now emit
  `psychoid_semantic_reducer`, `psychoid_hidden_cosine_reducer`,
  `psychoid_ritual_reducer`, `psychoid_numinous_reducer`, and
  `psychoid_symbol_reducer`; Z-space region parsing now emits
  `zspace_region_descriptor` with clamped spin/radius values and coarse
  spin/radius labels.
- psychoid CTI/dream reducers: feature-gated `psychoid_z_vector_projection`,
  `psychoid_motif_projection`, `psychoid_cti_projection`, and
  `psychoid_dream_replay_mapping` now expose motif mass, z-metric projection,
  CTI matrix projection components, and shadow-phrase-to-symbol replay mapping.
- chrono/maintainer monitoring reducers: `chrono_summary`,
  `chrono_harmonics_summary`, `timeline_maintainer_assessment`,
  `monitoring_drift_observe`, `monitoring_performance_observe`, and
  `monitoring_hub_observe` now expose time-window statistics, harmonic peaks,
  rewrite/clamp/dormancy decisions, drift alert counts, performance-regression
  targets, and hub-level alert/exporter state as CPU diagnostic metadata.
- atlas aggregation reducers: `atlas_route_summary` now exposes route-level
  frame/district/focus coverage, loop/collapse/Z drift, maintainer status and
  suggested controls, concept density, coherence, flux, and top beacon
  intensity. This closes the gap between low-level telemetry reducers and the
  atlas view used by learning monitors and run comparisons.
- dashboard export reducers: `dashboard_frame_ingest`, `dashboard_summary`, and
  `dashboard_snapshot_export` now expose dashboard frame retention, metric/event
  aggregate counts, top metric drift, and snapshot export volume. This makes the
  final operator-facing telemetry surface comparable with trainer and atlas
  traces instead of hiding behind dashboard-only state.
- noncollapse/XAI/region reducers: `noncollapse_snapshot_merge`,
  `noncollapse_hypergrad_snapshot`, `noncollapse_zpulse_snapshot`,
  `graph_flow_layer_begin`, `graph_flow_weight_update`,
  `graph_flow_elliptic_annotation`, `graph_flow_roundtable_annotation`,
  `graph_flow_drain`, `zspace_region_heatmap_report`,
  `zspace_region_delta_report`, and `zspace_region_volatility_report` now expose
  collapse-avoidance snapshot fields, hypergrad/ZPulse stability signals,
  graph-flow energy/update/roundtable annotations, and region-weight drift or
  volatility. These cover the explanation surfaces most directly tied to GNN
  learning diagnostics and region-adaptive training policy.
- attribution report bridges: generic `attribution_report_new` metadata now
  exposes XAI report shape, finite/positive/negative attribution mass, algorithm,
  layer/target/step presence, and extra metadata count. Hub storage bridges also
  emit `region_loss_report_store`, `region_loss_trend_report_store`, and
  `region_loss_volatility_report_store`, so trainer region monitors can see both
  report construction and the latest report chosen for operator-facing state.
- theory-to-telemetry observation adapters: `observation_bridge_ingest` now
  exposes microlocal gauge coverage, matched macro drives, pulse activation,
  observability efficiency, Softlogic Z feedback, and merged microlocal feedback
  hints. This makes the microlocal/macro observation bridge visible to trainer
  traces before its Softlogic feedback reaches region weighting or atlas state.
- Observability coalgebra adapters: `observability_assessment` now exposes
  empirical-vs-theoretical observation compression directly from
  `ObservationalCoalgebra::assess()`, including slot symmetry, branching factor,
  observable root cardinality, colour symmetry/singleton visibility, final
  observed and expected counts, compression gap, saturation/overflow flags, and
  min/mean/final efficiency. This lets data-prep and structural-observation
  experiments report learning-relevant compression quality even when they do not
  pass through the higher-level observation bridge.
- RG/fixed-point adapters: `rg_flow_fixed_points` now exposes Z-space RG lattice
  size, coupling dimensionality, tolerance, beta-norm envelope, Mellin attachment,
  fixed-point count, fixed scale range, and mean fixed coupling magnitude. Python
  theory calls that inspect `ScaleFixedPoint`s therefore leave a comparable trace
  footprint instead of staying outside the learning telemetry vocabulary.
- Mellin RG propagation adapters: `rg_mellin_flow_propagate` now exposes the
  older `renormalization_flow` DSL used by the Python theory facade, including
  lattice span, operator/trajectory counts, resonance and nonlinear feedback
  counts, scaling-dimension summaries, damping, initial/final coupling
  magnitudes, drift, max coupling, and non-finite trajectory counts. This makes
  narrative-depth and Mellin-resonance experiments comparable with Z-space
  fixed-point traces.
- Inflaton Z-lattice adapters: `inflaton_primordial_projection` now exposes
  primordial spectrum assembly from Hubble/epsilon Z-transform channels,
  including lattice/evaluation sizes, Planck mass, H and epsilon channel
  magnitudes, epsilon clamp count, spectrum min/max/mean, and non-finite count.
  This keeps cosmology/Z-transform probes visible to learning audits when their
  spectra are used as synthetic curricula or scale priors.
- Spiral dynamics stability adapters: `psi_spiral_metrics` now exposes the
  invariant-barrier and Hopf-control summary used by PSI monitoring, including
  growth parameters, forcing derivatives, Hopf regime, origin stability,
  audit/container gains, dimensionless clusters, hard-barrier availability and
  satisfaction, soft-barrier growth at zero container load, and noise-free
  amplitude bounds. This turns controller stability checks into traceable
  learning telemetry instead of leaving them as isolated algebra helpers.
- sync-bridge gate adapters: `sync_theorem_step` and
  `sync_family_aggregation` now expose structural/observation gate counts,
  I×K label distribution, log-e/increment/confidence/effective-gap summaries,
  hitting-time configuration, Azuma tail configuration, family fold policy, and
  dominant pair evidence. This turns the Rust-native sync theorem control loop
  into traceable learning telemetry rather than an opaque safety predicate.
- GR/Z-relativity reduction adapters: `zrelativity_dimensional_reduction`,
  `zrelativity_model_assemble`, and `zrelativity_tensor_bundle` now expose
  compactification dimensions, warp and learnable-block flags, internal volume
  and effective Newton constants, gauge/scalar/effective-metric magnitude
  summaries, field-equation shape/prefactor statistics, and tensor export
  counts. This makes the theory-to-tensor handoff used by Z-relativity modules
  visible to trainer traces instead of hiding as an offline model-construction
  step.
- Z-relativity learning module adapters: `zrelativity_module_forward` and
  `zrelativity_module_backward` now expose parameter-vector width, product and
  internal dimensions, learnable segment counts, block/gauge/scalar/warp segment
  lengths, value/gradient magnitude summaries, and
  `backend=parameter_adapter` / `requested_backend=host` so host-side
  pack/unpack work does not masquerade as tensor CPU compute debt. This confirms
  from ordinary training traces when the relativity module actually participates
  in a step, not only when its core theory tensors were assembled.
- Backend rank-planner adapters: `unison_rank_choice` now exposes
  `choose_unified_rank()` decisions across WGPU/MPS/CUDA/HIP/CPU capability
  descriptors, including rank kind, shape, selected source
  (`fallback`/KDSL/KV/generated WGPU), candidate count, score, lane/workgroup
  choices, merge strategy, FFT hints, latency-window bounds, shared-memory caps,
  and RealGrad summary pressure when present. This lets learning traces compare
  planner intent against the tensor kernel metadata collected later in the same
  training step.
- Microlocal ZPulse fusion adapters: `microlocal_conductor_step` now exposes the
  interface-gauge step that feeds Softlogic/ZPulse learning control, including
  gauge/signature counts, label presence, interface coverage, quality
  distribution, input and fused band energy, drift/Z-bias summaries,
  smoothing/band/budget policy flags, budget clamp scale, elliptic event counts,
  ZConductor attribution counts, fused Z/support/tempo/stderr, and raw-density
  statistics. This turns policy-controlled microlocal fusion into comparable
  trainer telemetry instead of leaving it as an opaque pre-learning control
  surface.
- Canonical ZPulse conductor adapters: `zpulse_conductor_step` now exposes the
  per-step fusion of Microlocal, Maxwell, Graph, Desire, GW, and RealGrad pulse
  sources, including pending/ready/retained counts, source support attribution,
  top source, fused Z/support/drift/quality/density, latency-align state,
  flip/density events, and optional frequency/adaptive-gain configuration. This
  makes the control signal that downstream learning policy consumes comparable
  across ordinary trainer steps instead of only visible inside specialised
  microlocal traces.
- ZPulse scale-persistence adapters: `zpulse_scale_stack` and
  `zpulse_coherence_levels` now expose the bridge from pulse traces into
  `st-frac` semantic scale stacks, including pulse source counts, feature
  summaries, scale/threshold/metric configuration, gate curve statistics,
  persistence mass, requested coherence levels, resolved scale count, and
  resolved ratio. This makes scale-coherence stability visible beside the fused
  Z control signal consumed by downstream learning policy.
- SpinoTensorVector ZPulse adapters: `spino_tensor_zpulse_project` now exposes
  STV-to-ZPulse materialisation, including source label, causal class, alpha/beta
  invariants, determinant summaries, band energy, support, density fluctuation,
  drift, Z-bias, quality, stderr, and Z-scale. This makes Graph/Maxwell pulses
  derived from the STV kernel algebra observable before they enter the canonical
  conductor.
- PSI synchronisation adapters: feature-gated `psi_sync_branch` now exposes the
  Rust-native SynchroMonolith branch pipeline, including phi sampling span and
  drift, gamma/lambda/drive parameters, kappa and omega estimates, RMSE,
  Arnold-tongue heatmap size and peak concentration, best rational lock, qmax
  policy, and render status. This closes the gap between old PSI branch
  synchronisation theory and the downstream heatmap/ZPulse adapters already
  visible in `st-nn`.
- Maxwell ZPulse adapters: `maxwell_z_project` and feature-gated
  `maxwell_psi_publish` now expose sequential-Z projection gates, emitted pulse
  quality, band energy, Z-bias, density fluctuation, PSI reading totals,
  Softlogic feedback, and event/attribution counts. This connects coded-envelope
  Maxwell statistics to the same trainer-visible telemetry stream used by
  microlocal fusion and region-aware learning feedback.
- CPU fallback metadata for row softmax, row hardmax/softmax fusion, layer norm,
  and scaled dot-product attention. WGPU fused kernels already emit
  kernel-level metadata as `wgpu_dense`, which trainer strictness now normalizes
  to the logical `wgpu` backend.

The metadata includes:

- `backend`: actual selected kernel path, such as `wgpu`, `hip`, `cpu_simd`,
  `faer`, or `naive`.
- `requested_backend`: caller policy, currently often `auto`.
- shape and layout metadata needed to explain fallback.

`ModuleTrainer::train_epoch()` now installs a step-scoped tensor metadata
collector for each training step. When `TrainerStep` has listeners, the resulting
`metrics.extra` payload includes `tensor_ops_total`, `tensor_backend_*`,
`tensor_op_*`, `tensor_op_backend_*`, and `tensor_policy_*` counters, so existing
trainer trace JSONL files can be compared with
`summarize_trainer_trace_events()` without a new trace format. The policy keys
make the trainer-level request visible next to actual tensor execution, e.g.
`tensor_policy_matmul_wgpu` beside `tensor_backend_wgpu` or
`tensor_backend_cpu_simd`. Layer norm, fused attention, and row-softmax policy
requests are also visible as `tensor_policy_layer_norm_*`,
`tensor_policy_attention_*`, and `tensor_policy_softmax_*`. When
`SPIRALTORCH_STRICT_GPU` is set and the trainer caps request WGPU, CUDA, or HIP,
the same collector is installed even without trace listeners and rejects steps
whose observed tensor backends never match the requested GPU backend. Missing
backend metadata is now also treated as a strict GPU failure rather than an
unproven success.
The same collector now also feeds `EpochStats.tensor_backend`, a fixed-width
`EpochTensorBackendStats` summary that is returned by `train_epoch()` even when
no `TrainerStep` listener is installed. Python `EpochStats` exposes the same
summary as `tensor_backend`, and char-LM model-zoo `metrics.jsonl` files now
persist it per epoch. `compare_char_lm_runs.py` will show tensor op, WGPU, CPU,
CPU SIMD, and fallback columns from either trainer traces or those epoch
metrics, making backend behavior visible in ordinary sweep comparison tables.
When trainer traces are present, the same comparison now also renders
learning-op backend columns for matmul, prepacked matmul, row-softmax,
layer-norm, layer-norm backward, and scaled-dot attention, so char-LM/FT
comparisons can see which learning kernels actually moved to WGPU rather than
only seeing aggregate backend totals.
Logical backend counters intentionally keep normalizing `wgpu_dense` to `wgpu`
for strict validation, but kernel-family counters now preserve the raw label as
`tensor_kernel_backend_wgpu_dense` and `kernel_backend_wgpu_dense`. That lets
learning reports distinguish "the step satisfied WGPU policy" from "the dense
WGPU kernel family actually ran" without weakening strict GPU checks.
Strict validation now treats `backend=composite` and `backend=view` as
metadata-only evidence: they remain visible in `tensor_backend_*` and
`tensor_op_backend_*` counters, but they neither satisfy strict GPU on their own
nor count as CPU/GPU mismatches when a real WGPU/CUDA/HIP tensor kernel is also
observed. This keeps graph readout, CoherenceWave, ToposResonator, and
zero-copy reshape wrappers readable without turning logical layer boundaries
into false strict-GPU failures.

Next steps:

1. Extend tensor metadata to remaining elementwise/reduction paths that do not
   yet emit requested-vs-actual backend fields outside the loss stack, next by
   reviewing remaining st-core telemetry reducers that can influence learning
   monitors without surfacing metadata, especially any theory reducers still
   absent from trainer traces after the observation/RG/sync/GR pass.
2. Extend strict GPU parity tests to WGPU-enabled CI or local device jobs.

### P0: Thread a real execution policy from trainer to layers

`DeviceCaps` influences schedule and telemetry. A first execution-policy bridge
now exists in `st-nn::execution`: `BackendPolicy::from_device_caps()` derives
dense, prepacked matmul, layer-normalization, fused-attention, and row-softmax
backends, and `ModuleTrainer::train_epoch()` installs that policy in a
thread-local guard for the active step. `Linear`,
`GraphContext`, `ZSpaceGraphConvolution`, `WaveRnn`, `SpiralRnn`, Conv2d
lowering gradients, and Z-RBA attention projections now route their
dense/prepacked matmuls through that policy instead of unconditional `Auto`.
`LayerNorm` also routes tensor affine layer normalization through the same
policy. Its backward path now also uses the affine-correct LayerNorm VJP:
the row-wise mean/dot reductions are formed from `grad_output * gamma`, not raw
`grad_output`, so learned non-uniform gamma values no longer corrupt
`grad_input`; a finite-difference test locks that contract. The affine
normalization family now also follows the shared-parameter update contract:
`LayerNorm`, `BatchNorm1d`, `ZSpaceLayerNorm`, and `ZSpaceBatchNorm1d` scale
gamma/beta gradients by `1 / rows` or `1 / batch`, keep input gradients
unaveraged, and expose that scale in normalization backward metadata so
duplicated batches no longer inflate affine updates. Z-RBA mean
aggregation now routes its geometry-biased per-head attention through
`Tensor::scaled_dot_attention_with_backend()`, exposing `scaled_dot_attention`
backend metadata in learning traces. `ZSpaceSoftmax`
now preserves its curvature/temperature adaptation while delegating row
probabilities to `Tensor::row_softmax_with_backend()`, so the NLP-facing
softmax path emits tensor backend metadata too. The normal non-adaptive
`ZSpaceSoftmax` path used by the char-LM examples now sends the full batch
through one row-softmax call, while the entropy-targeted adaptive path keeps the
row-wise retry loop where each row can choose a different effective
temperature.

Trainer step traces now also include `batch_input_*`, `batch_target_*`,
`batch_prediction_*`, `batch_loss_*`, and `batch_grad_output_*` shape metrics.
That makes row-concatenated mini-batches visible when comparing NLP token rows,
GNN node rows, and self-supervised pair rows in one trace vocabulary. Observed
trainer-step runs also emit finite/non-finite counts and finite L1/L2/Linf
scales for those tensors, making prediction blow-ups, invalid loss values, and
bad loss gradients visible before they become opaque parameter-gradient health
signals.
When a `TrainerStep` listener is active, the trainer also snapshots parameters
around the optimizer step and emits `optim_param_update_*` metrics plus the
learning rates used for that step. This gives trace runs a direct update-ratio
signal for LR, curvature, collapse, and spectral-policy experiments without
paying the copy cost in unobserved training. The same trace includes
per-parameter hotspot aggregates such as max update L2, max update ratio, active
parameter count, and zero-update parameter ratio, so under-trained or
over-dominant parameter groups can be spotted from the run comparison table.
The direct `Parameter` update path now validates fallback learning rates before
adapter scaling or accumulator mutation, so non-positive and non-finite rates
cannot silently perturb gradients when callers bypass `ZSpaceOptimizer`.
Fallback Euclidean updates plus HyperGrad/RealGrad tape applies and direct
gradient scaling now preflight the computed delta, next parameter value, or
scaled accumulator before any in-place write or accumulator reset. HyperGrad
also scratch-builds raw accumulator candidates before topos saturation, so
huge-but-finite local LR factors cannot leave weights or gradient buffers
half-poisoned with `inf`.
`ZSpaceOptimizer`, `SpectralLrAdapter`, and `ModuleTrainer` now expose copyable
state snapshots for learning-rate, mode, clipping, spectral-policy, and adapter
EMA state. `TrainerStep` traces emit the same optimizer-state vocabulary before
each update, and trainer-level global LR scaling now resets the spectral adapter
just like `ZSpaceOptimizer::scale_learning_rate()` so local LR decisions do not
reuse stale pre-scale gradient statistics.
Global LR scaling now also prevalidates every multiplied optimizer/trainer rate
before mutating state, while the underlying HyperGrad/RealGrad tapes ignore
overflowing scale factors, so a finite-but-huge schedule factor cannot poison
future steps with `inf` learning rates.
Rank planner feedback follows the same snapshot rule: `choose_unified_rank()`
captures the current RealGrad summary once when it builds a `RankScenario`, while
plain `RankScenario::new()` remains deterministic for tests and static planning.
That prevents concurrent telemetry updates from perturbing one planner decision
halfway through latency-window tuning, and keeps adaptive rank choices
reproducible enough to compare in learning traces.

The tensor core now preserves zero-sized axes in the basic empty-batch path:
`Tensor::from_vec`, `zeros`, `from_fn`, `view`, `reshape`, and `cat_rows` can
carry shapes such as `(0, cols)` without erasing the column contract. Loss
reductions defensively return zero loss and empty gradients for zero-sized
predictions before dividing by their reduction counts, so empty batches from
Rust/Python facades can flow through training without becoming divide-by-zero
or construction failures. Some compute kernels still reject zero axes when the
operation has no sensible sampling or normalization semantics. Probability
epsilons are also sanitized before clamp-based losses use them, avoiding
`min > max` clamp panics from overly large or non-finite user-provided epsilon
values.
Loss reductions now reject non-finite predictions or targets before clamp-based
probability math can hide `NaN`, and they validate finite intermediate terms,
sums, scalar losses, and gradient entries before handing tensors back to the
trainer. MSE, categorical CE, focal, hyperbolic CE, contrastive, and triplet
losses therefore fail at the supervised objective boundary when distance
squares, margin scores, scaled logits, or probability gradients overflow.
Backend-routed loss reductions also relabel downstream tensor non-finite errors
back to the loss-specific gradient boundary, so trainer diagnostics report
`mse_gradient` or `contrastive_loss_gradient` instead of a generic
`scale_output` tail. Contrastive loss now evaluates positive and negative terms
only when their clamped labels can contribute, so a margin-satisfied negative
pair with a huge finite distance yields zero loss/gradient instead of a false
distance-square overflow.
Hyperbolic CE also uses stable softplus/sigmoid forms for large logits, so
large but finite confident predictions no longer turn into `inf` merely because
`exp(logit)` overflowed.
The core scaled-dot attention op now has an explicit empty-volume contract too:
zero contexts, zero sequence length, or zero head dimension preserve the
validated output shape and emit `scaled_dot_attention` metadata without
dispatching a GPU kernel. This keeps lower-level attention probes consistent
with the higher-level Z-RBF empty-sequence path.
The same attention boundary now rejects non-finite scale factors, query/key/value
buffers, Z-bias tensors, attention-bias tensors, and backend outputs before they
can become ordinary activations. The CPU fallback also validates dot products,
logits, denominators, weights, and accumulators during the online softmax loop,
so finite-but-overflowing attention scores fail as `PureResult` instead of being
rounded into misleading zero or one-hot attention output.
The row-wise probability fusion path is covered as well: softmax+hardmax and
softmax+hardmax+spiral preserve zero-row and zero-column shapes, returning empty
payload tensors and default spiral consensus metrics instead of depending on a
caller-side special case.
Layer normalisation is now consistent at the tensor-core level for empty
batches: `layer_norm_affine(_add)` validates affine and residual shapes, emits
`layer_norm` metadata, and returns `(0, cols)` without forcing higher layers to
special-case the op. Fused `matmul_bias_gelu` also has regression coverage for
zero-row outputs and zero-inner products, preserving the useful bias-only GELU
semantics for degenerate MLP projections.
The residual fused MLP path follows the same rule: `matmul_bias_add_relu`
preserves empty-row outputs and zero-inner residual+bias ReLU semantics. On the
data ingress side, fixed and dynamic `DataLoader` batching now have regression
coverage for zero-row samples, so the loader preserves column contracts and
continues making progress rather than erasing empty examples or stalling row
budget grouping.
The same empty-batch contract is starting to move up into learning layers:
`Linear` and `Scaler` backward now validate feature shapes, return empty input
gradients for zero-row batches, and avoid accumulating parameter updates from
nonexistent examples. `Linear` now also validates inputs, loaded weights/biases,
forward outputs, upstream gradients, weight/bias reductions, and input gradients
at the relevant cache/accumulator boundary, keeping dense projection failures
local to the offending minibatch. `ZSpaceGraphConvolution`
also rejects node-row mismatches at aggregation time, so graph models fail with
a shape error instead of letting an empty or row-concatenated graph silently
reach batch-normalized projection updates. `WaveRnn` and `SpiralRnn` follow the
same pattern for recurrent backward passes, returning empty input gradients
without touching parameter accumulators when a zero-row sequence batch reaches
the cached backward path.
Trainable convolution layers (`Conv1d`, `Conv2d`, `Conv3d`, `Conv4d`, and
`Conv6da`) now follow that rule as well, so empty vision/sequence batches do
not divide by batch size or install zero-valued weight/bias gradients.
`Embedding` keeps emitting CPU backend metadata for zero-token batches while
skipping weight updates, and `NonLiner` now returns empty gradients without
creating zero-valued affine parameter accumulators.
Normalization layers now follow the same zero-row contract: `LayerNorm`,
`BatchNorm1d`, `ZSpaceLayerNorm`, and `ZSpaceBatchNorm1d` return empty outputs
and input gradients without updating running statistics, affine accumulators, or
Z-space projector gain from nonexistent observations. Z-space normalization
still caches empty telemetry, and normalization backward continues to emit CPU
metadata so strict traces can see the empty pass-through explicitly.
Activation/probability layers are being pinned to that contract too: `Relu`
now rejects non-finite forward inputs, backward inputs, and upstream gradients
before its scalar mask can hide a bad activation as an inactive unit. `Gelu` now
passes zero-row and zero-column tensors through forward/backward while emitting
explicit identity `gelu_backward` metadata for empty gradients. Its scalar forward
path and backend/fallback backward outputs now reject non-finite inputs,
overflowed tanh-approximation intermediates, non-finite upstream gradients, and
non-finite VJP outputs instead of returning ordinary activation tensors.
`ZSpaceSoftmax` has regression coverage for zero-row logits preserving shape and
empty entropy/temperature diagnostics.
Z-RBF attention now handles empty ZTensor sequences directly, returning empty
mean/variance tensors plus zeroed per-head telemetry instead of dividing kernel
statistics by zero or calling scaled-dot attention with sequence length zero.
The Z-RBF attention workspace is now fail-fast as well: metric weights,
ARD length scales, `mu`/`sigma`, query/key/value projections, scalar dot-score
products, kernel logits, row-softmax probabilities, entropy telemetry,
per-head means, variance accumulation, and final output projection are
validated before the attention output is returned. This makes attention blowups
show up as `zrba_attention_*` errors rather than drifting into variance
telemetry or downstream covariance reconstruction.
The residual Beta gate follows the same boundary in the Z-RBA forward path:
`try_forward()` validates spectral features, stable softplus logits, alpha/beta
concentration, expected value, variance, Beta samples, and EMA updates before
the residual gate can scale `mu` or `sigma`. Z-RBA residual scaling then validates
the gated tensors and residual additions before covariance reconstruction sees
them.
The self-supervised InfoNCE entry points also reject empty and zero-feature
pairs at construction time, and backward now verifies that the prediction shape
matches the forward cache before reading cached feature rows. This turns stale
or mismatched backward calls into shape errors rather than panic-prone or
silently truncated gradients.

The NLP input side is now visible as well: `Embedding` still performs CPU
gather/scatter-add, but forward and backward emit explicit `embedding_forward`
and `embedding_backward` metadata with token counts plus non-finite, rounded,
low-clamped, and high-clamped token repair counters. The metadata also reports
unique and repeated token indices, scatter-add volume, gradient scale, and the
gradient reduction backend used for batch normalisation. This makes token-index
data quality, token reuse pressure, and the remaining CPU embedding path
visible in trainer traces before a future GPU gather/scatter backend exists.
Embedding now also validates finite table weights, gathered outputs, upstream
embedding gradients, scatter-add contributions, and scaled weight gradients
before touching the parameter accumulator, so bad token tensors can still be
counted as repairs while bad numeric payloads fail at the learning boundary.
`ModuleTrainer` also aggregates those events into `tensor_embedding_*` metrics,
with token repairs, unique indices, and repeated indices included in the Python
trace spotlight. Char/self-supervised comparisons now expose
`embedding_fwd_cpu` and `embedding_bwd_cpu` as explicit learning-op columns.

Normalization backward paths are still CPU implementations, but they now emit
explicit backend metadata instead of disappearing from strict traces. That makes
normalization-heavy models honest under WGPU-first experiments: a WGPU policy can
show WGPU forward layer norm with CPU backward normalization reductions instead
of looking silently GPU-complete.

GNN graph-level readout pooling and its backward broadcast now emit explicit CPU
metadata as well. This keeps graph-regressor traces honest when node-message
matmuls use policy-driven kernels but the graph-level reduction remains a
hand-written CPU aggregation.

This deliberately avoids changing the broad `Module::forward()` and
`Module::backward()` signatures while still making the dominant char-LM, RNN,
GNN, vision-backward, and Z-RBA projection paths policy-aware. Explicit WGPU
prepacked matmul is also wired in `st-tensor`, so a WGPU policy can reach the
same kernel family that Auto already used opportunistically.

Next steps:

1. Extend the same policy/trace contract to remaining graph-level diagnostics
   and coherence aggregations that still compute entirely by hand.
2. Decide whether a future explicit context parameter is still needed once the
   thread-local bridge covers the important learning paths.
3. Add WGPU-feature parity tests for policy-forced prepacked matmul on machines
   with an available adapter.

### P0: Connect `TrainingDevice` to parameter accumulators

`spiral-selfsup` already has a clean `TrainingDevice` abstraction for
all-reduce and metric aggregation. The main trainer does not use it, and
`Parameter` accumulators remain local.
The older in-memory `st-core::distributed::DistributedTrainer` now rejects
non-positive/non-finite learning rates, non-finite broadcast payloads, sync
gradients, async gradients, and reduced-gradient overflow before mutating
parameters. That makes the distributed side safe enough to use as a bridge test
target when wiring synchronization into `ModuleTrainer`.
Its sync and async parameter commits are now scratch-built as well: finite
reduced gradients still must produce finite update deltas and finite next
parameters before any shard is committed, so huge-but-finite LR/gradient pairs
cannot partially write `inf` into old distributed trainer state.
`st-core::distributed::AmebaAutograd` now applies the same finite-value contract
one layer earlier: tolerance, agent learning rates, non-negative damping,
initial weights, seed gradients, local update deltas, and forwarded gradients
are checked before weights are committed. Non-finite, sign-inverting, or
overflowed distributed messages therefore cannot partially mutate agent weights
before surfacing an error.
`ModuleTrainer` now has an optional `st-core::distributed::AccumulatorSynchronizer`
slot, and `spiral-selfsup` devices implement that bridge without creating a
crate cycle. Before each optimizer step the trainer synchronizes every active
`Parameter` accumulator buffer: hypergrad, realgrad, or the Euclidean fallback
gradient. The latest sync pass is exposed as `TrainerAccumulatorSyncStats` and
emitted into `TrainerStep` traces as `optim_accumulator_sync_*`, so distributed
runs can be compared against local runs without a new trace format.
Accumulator synchronization now commits atomically at the `Parameter` boundary:
device callbacks write into a scratch buffer, the synchronized values are checked
for finiteness, and only then are the real gradient/hypergrad/realgrad buffers
updated. A failed or overflowed distributed sync can therefore abort the trainer
step without poisoning either weights or local accumulators.
`spiral-selfsup` also now carries an integration smoke that installs real
`DistributedDevice` handles into two `ModuleTrainer` instances and verifies
their post-step parameter values match a single local trainer fed the averaged
accumulator. That makes the bridge more than a unit hook: it proves the
distributed all-reduce semantics survive the trainer's local LR adapter and
optimizer step.

Next steps:

1. Add Python-facing controls for configuring a trainer training device.
2. Extend sync stats with strategy/error counters once multiple strategies exist.
3. Add a full `train_epoch()` distributed smoke once roundtable band replay has a
   stable deterministic fixture that will not accidentally multiply gradients.

### P1: Deepen WGPU-first prepacked learning projections

`Linear`, GNN, and RNN paths sensibly cache packed weights. Auto mode can route
prepacked matmul through WGPU when available, and explicit WGPU prepacked matmul
is now wired so trainer policy can reach that path. The remaining work is
less about selecting the kernel and more about proving learning behavior,
packed-layout stability, and fallback policy under realistic batches.

Next steps:

1. Stress the prepacked WGPU route under `Linear`, GNN, and RNN training steps.
2. Track packed layout and tile choice in trainer traces.
3. Add CPU/WGPU parity tests for `Linear::forward`, `Linear::backward`, and GNN
   projection steps under identical seeds.

### P1: Separate HIP planning from real HIP execution

HIP has useful runtime probing and optional real kernels, but feature `hip`
without `st-backend-hip/hip-real` can still look available from environment
markers. That is helpful for planner experimentation but dangerous for learning
benchmarks if reported as actual acceleration.

There is already a useful `SPIRALTORCH_STRICT_GPU` path for CUDA/HIP rank-k
execution in `st-core::backend::{cuda_exec,hip_exec}`. Dense learning matmul
now shares the same truthy environment contract for GPU-attempt failures in
`matmul` and `matmul_prepacked`; fused matmul and non-matmul dense ops still
need the same treatment. HIP rank-k strictness is also gated by `hip-real`,
while non-real HIP rank-k remains a software path by construction.
`st-core::backend::wgpu_exec` is now honest in the same direction: the executor
is compiled under `wgpu-rt`, borrows registered host launch buffers, attempts
real WGPU TopK/MidK/BottomK launches when a `WgpuCtx` is installed, and
otherwise uses the finite-safe software rank-k reference in non-strict mode.
Under `SPIRALTORCH_STRICT_GPU=1`, missing WGPU context or failed real rank
launch returns an explicit "fallback disabled" error. An opt-in
`SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1` parity test now exercises strict real
TopK, MidK, and BottomK when an adapter is available. The real WGPU TopK/BottomK
bridge is deliberately guarded to exact shapes through
`wgpu_rank_exact_support{,_for}` (`k <= cols`, and `cols <= 256` when `k > 1`)
because the current kernel keeps one lane winner before final selection. The
real MidK bridge uses an exact small-row workgroup sort and is guarded to
`k <= cols <= 256`. Wider rows fall back unless strict mode asks to fail. The
runtime parity coverage includes both the original 2×5/k=2 smoke and a 2×8/k=4
TopK/MidK/BottomK comparison against the software reference. That prevents
planner-level WGPU intent from being mistaken for actual WGPU rank execution.
`st-tensor` keeps the same opt-in guard for tests that intentionally probe
missing WGPU runtime fallback, so default test runs validate the metadata
contract without risking adapter-initialisation aborts.
The char-LM fine-tune harness now also records this distinction in `run.json`
via `tensor_policy` and `roundtable_backend_audit`, so learning runs can be
compared by actual WGPU rank readiness instead of only by the requested backend
label. The same backend runtime preparation now runs through the shared
model-zoo training entrypoints for char coherence scan/wave, WaveRNN, vision
conv-pool, GNN band tracing, and the lightning self-supervised smoke so WGPU
selection installs the rank runtime before training starts instead of only
changing `DeviceCaps`. `tools/compare_char_lm_runs.py` and
`tools/run_char_lm_sweep.py` surface
the same data as `tensor_policy_*` and `rt_wgpu_*` columns/manifest fields, so
sweep tables can separate learning quality from backend truth (`not_requested`,
`runtime_context_missing`, `fallback_shape`, or `exact_runtime_ready`).
`run_char_lm_sweep.py --backends cpu,wgpu --cargo-features wgpu ...` now runs
the same char-LM grid across both requested backends and includes the backend in
each run directory name. A 1-epoch smoke showed identical validation quality for
CPU and WGPU in this tiny setting while `tensor_wgpu` increased on the WGPU row
and `rt_wgpu_statuses` now reaches `exact_runtime_ready` after the WGPU backend
entrypoint installs the shared `wgpu_rt::WgpuCtx`. That cleanly separates
dense/tensor WGPU usage from real rank-k runtime readiness while keeping both
signals in the same learning comparison table.

A broader 6-run smoke now exercises the shared char-LM matrix:

```bash
python3 -S -s tools/run_char_lm_sweep.py models/samples/spiral_demo_en.txt \
  --run-root target/tmp/char_lm_arch_backend_matrix_smoke \
  --architectures finetune,scan,wave \
  --features token-bigram \
  --head-priors learned-unigram \
  --backends cpu,wgpu \
  --seeds 7 \
  --cargo-features wgpu \
  --epochs 1 \
  --batches 1 \
  --batch 2 \
  --steps 4 \
  --embed-dim 4 \
  --hidden 8 \
  --memory 4 \
  --gen 8 \
  --eval-samples 4 \
  --preset smoke \
  --continue-on-error
```

The matrix finished with `failed=false`, six successful `summary.json` /
`run.json` pairs, and trainer trace summaries for every run. The WGPU rows for
`llm_char_finetune`, `llm_char_coherence_scan`, and
`llm_char_coherence_wave` all reported
`above:exact_runtime_ready,here:exact_runtime_ready,beneath:exact_runtime_ready`,
while the corresponding CPU rows reported `not_requested`. In this tiny smoke
the CPU/WGPU final validation NLL matched per architecture, which is a
correctness sanity check rather than evidence of useful acceleration.

The same reporting contract now exists outside char-LM through
`tools/backend_sweep_meta.py`. `run_wave_rnn_sweep.py`,
`run_vision_conv_sweep.py`, `run_lstm_sweep.py`, `run_selfsup_sweep.py`, and
`run_gnn_band_trace_sweep.py` load each run's `run.json`, add
`policy_matmul`, `policy_softmax`, `rt_wgpu_initialized`, `rt_wgpu_ctx`,
`rt_wgpu_ready`, and `rt_wgpu_statuses` columns to `compare.md`, and preserve
`backend_runtime`, `tensor_policy`, and `roundtable_wgpu` in `sweep.json`.
The LSTM sweep adds focused columns for the gate/BPTT backend split and paired
WGPU projection/reduction routes, so backward-scan changes can be compared
without reinterpreting raw trace keys. The backward path now isolates reverse
recurrence as `bptt_scan_kernel=lstm_backward_scan.cpu_fused_loop` with
`bptt_scan_lowering=host_reverse_recurrent_scan` on the CPU/reference seam, and
the WGPU feature path attempts `bptt_scan_kernel=lstm_backward_scan.wgsl` with
`bptt_scan_lowering=wgpu_single_workgroup_hidden_parallel_recurrence` before falling back
with `bptt_scan_fallback_reason`. Forward gate activation can likewise route
through the tensor-util `lstm_forward_gate_step` WGPU helper. This gives the
future parallel WGPU scan a single replacement seam instead of an inline loop
hidden in `backward()`.
`tools/run_lstm_scan_profile_grid.py` wraps that sweep across `steps x hidden`
shape cells and writes `grid.md` / `grid.json`, preserving preflight/failure
evidence while aggregating scan elapsed time, value counts, dispatch count, and
estimated BPTT work per scan step. Each `grid.json` run also carries a
machine-readable `scan_profile` object so follow-up analysis can read
`lstm_scan_us`, `lstm_scan_gate_values`, `lstm_scan_workgroup`,
`lstm_scan_parallel_lanes`, `lstm_scan_parallel_axis`,
`lstm_scan_backend`, `lstm_scan_kernel`, `lstm_scan_lowering`,
`lstm_scan_fallback`, and `lstm_est_bptt_ops_per_us` without parsing
markdown. Shape averages also include `lstm_scan_backend_counts` and
`lstm_scan_fallback_counts` so mixed route/fallback cells are visible before
reading per-run rows.
`tools/compare_char_lm_runs.py` now surfaces the same forward/backward LSTM gate
WGPU hits plus estimated gate/BPTT CPU-debt, WGPU-op splits, and
`lstm_scan_backend/kernel/lowering/fallback`, so char-LM rows can show whether
the new gate helper is actually reducing recurrent CPU debt and whether the
backward recurrent scan stayed on the intended route. Aggregate char-LM tables
and top-ranked aggregate summaries also include scan backend/fallback counts
when scan route evidence is present.
The char-LM fine-tune example now accepts `--recurrent spiral|lstm`, and
`tools/run_char_lm_sweep.py --architectures lstm ...` routes through the same
raw-text harness with a stateless batched LSTM window adapter. New LSTM
artifacts persist `recurrent="lstm"` in both `run.json` and weights metadata,
while older metadata defaults to the original `spiral` recurrent path.
`tools/compare_char_lm_runs.py --aggregate` groups rows by architecture,
recurrent core, backend, priors, features, mode, and shape/training dimensions,
reporting multi-seed mean/std NLL plus backend-debt means. `run_char_lm_sweep.py`
now includes this aggregate table in generated `compare.md` files by default,
and also writes the same rows plus aggregate/top-aggregate selections to
`compare.json`, making `--architectures finetune,lstm --seeds ...` a direct
recurrent-core comparison that downstream scripts can consume without parsing
Markdown. Aggregate rows carry `route_status` directly, and
`tools/summarize_char_lm_compare.py` can emit a route-aware shortlist, including
`--route-clean-only` and `--prefer-clean-route` views that keep scan fallback
rows from masquerading as clean winners. It accepts one or more `compare.json`
paths, so independent sweeps can be ranked together while preserving each row's
source artifact; pass a sweep directory to use its direct `compare.json`, or
`--recursive` on a parent directory to collect many sweep outputs. Its Markdown
and JSON outputs include both all-candidate and selected-row route-status counts
before the shortlist, so a sweep bundle with many hidden fallbacks is visible
even when the displayed rows prefer clean routes. `--prefer-clean-route` only
penalises missing scan-route evidence for rows where the LSTM scan route is
applicable; non-LSTM architectures with `no_scan_route` remain rank-neutral so
mixed architecture comparisons are not distorted by route metadata that does
not apply to them. The shortlist carries head
prior/residual, feature/mode, shape, and training-budget columns so top rows can
be reproduced without opening the raw aggregate table. It also includes
aggregate trace latency/update-ratio and CPU-debt columns, so equal-quality rows
can be compared by execution cost directly in the shortlist. When matching
`spiral` and `lstm` aggregate rows share the same backend, head, feature, shape,
and training budget, the summary also emits a paired recurrent-delta table and
JSON payload with candidate-minus-baseline quality, latency, and CPU-debt
deltas. These paired rows include quality/cost status fields plus an
`efficiency_verdict` such as `candidate_quality_neutral_cost_better`, allowing
automation to distinguish a genuine efficiency win from a cheaper but
quality-regressed run. The JSON and Markdown outputs now derive a
`paired_recurrent_recommendations` shortlist from those deltas, keeping
quality-improved or quality-neutral cost wins near the top while leaving
quality-regressed candidates in the full audit table only. Recommendations also
require the candidate's own `delta_nll_mean` to be improved or neutral, so a
cheaper recurrent route that merely matches a non-learning baseline does not
look like a learning win. It can also rank by
alternate metrics such as
`--sort-metric final_vs_bigram` or, through the sweep wrapper,
`--compare-summary-sort-metric final_vs_bigram`, which is useful when absolute
NLLs across corpora are less informative than beating a simple conditional
baseline. The sweep wrapper writes the preferred-clean shortlist to
`compare_summary.md` and `compare_summary.json` by default, exposes
`--compare-summary-limit`,
`--compare-summary-route-clean-only`, and
`--compare-summary-no-prefer-clean-route`, and can fail summary generation with
`--compare-summary-fail-on-route-status scan_fallback` when a sweep should be
treated as route-regressed. It can also fail on paired recurrent regressions
with `--compare-summary-fail-on-paired-quality-status regressed`, or on a
specific `efficiency_verdict` with
`--compare-summary-fail-on-efficiency-verdict candidate_quality_regressed`.
The generated `compare_summary.json` and parent `sweep.json` both carry
route-status counts, paired recurrent deltas, paired recurrent
recommendations, bigram guard deltas/recommendations, bigram rank guard
deltas/recommendations, baseline-difficulty hotspots, and gate failure maps, so
automation can detect route, efficiency, guard, or eval-slice regressions
without scraping logs.
Aggregate summaries also surface `unigram_nll_mean` and `bigram_nll_mean`,
making eval-slice baseline difficulty visible next to model NLL and
`final_vs_bigram_mean`. They also carry `final_windows_mean`,
`unigram_windows_mean`, and `bigram_windows_mean`, so requested
`eval_samples` can be distinguished from the actual number of validation
windows evaluated. The baseline-difficulty rows rank cases where the bigram
baseline is stronger than the unigram baseline and the model still lags that
bigram line, making hard evaluation slices visible before adding more optimizer
or guard sweeps.
The same sweep wrapper accepts `--step-values`,
`--hidden-values`, and `--embed-dim-values`; explicit shape-grid runs include
those dimensions in run names and aggregate keys so unlike windows or hidden
sizes are not averaged together. It also accepts `--epoch-values` and
`--batches-values` for training-budget grids, `--eval-samples-values` for
evaluation-size grids, `--val-start-values` for validation-slice location
grids, plus `--lr-values` for optimizer rate grids, so smoke, medium, and
longer learning runs can be compared without collapsing different optimization
budgets, eval-set sizes, or validation slices into one aggregate. It also
accepts `--head-residual-scale-values` so prior/no-prior and residual-logit
pressure sweeps can be compared without averaging unlike head settings
together; `0` is accepted as a pure-prior ablation where residual logits and
their backward path are intentionally muted. Top
aggregate rows still rank by validation NLL first, but now use trace latency
and CPU-debt tie-breaks so equal-quality recurrent shapes surface the cheaper
execution path.

A focused recurrent-depth probe around `steps=12 hidden=16 lr=0.02
head_residual_scale=0.5` shows why the absolute-learning guard matters. With
`epochs=10 batches=12 eval_samples=32`, the LSTM route kept pair quality neutral
while improving its own `delta_nll_mean` by roughly `-0.0024`, and reduced
trace latency to about `0.47x` plus CPU debt to about `0.25x` of the spiral
baseline. Re-running the same shape with `eval_samples=64` and
`batches=12,24` kept the relative LSTM-vs-spiral quality neutral, but both
candidate pairs had positive candidate `delta_nll_mean`; those rows now stay in
`paired_recurrent_deltas` and are intentionally omitted from
`paired_recurrent_recommendations`. This prevents a cheaper route from being
promoted when the underlying learning run did not actually improve.

A follow-up `eval_samples=64 batches=24` learning-rate sweep (`lr=0.0025,
0.005, 0.01, 0.02`, three seeds) narrowed the failure mode: lowering the rate
from `0.02` to `0.0025..0.005` moved candidate `delta_nll_mean` from regressed
to neutral, but did not yet produce an improved learning status. The LSTM route
still retained the same cost advantage (`cpu_debt_ratio` around `0.25`) and
quality stayed pair-neutral, so the next quality push should test longer
budgets or a rank/top-k guard rather than simply increasing the step size at
`lr=0.02`.

The rank/top-k guard was tested next on the same `eval_samples=64`,
`batches=24`, `epochs=10`, `steps=12`, `hidden=16` setup with
`lr=0.0025,0.005`, `bigram_guard=0,0.1`, and `bigram_guard_k=5` over three
seeds. The guard did not produce a guard-specific recommendation and did not
move the candidate recurrent rows from neutral into improved learning status;
`bigram_guard=0.1` slightly reduced CPU debt for LSTM but left NLL/top-k
quality effectively tied. That makes longer/wider budget probes at
`lr=0.0025..0.005` a better next step than increasing guard weight first.

That budget probe then compared `batches=24` versus `48` at `lr=0.0025,0.005`
with `eval_samples=64` and two seeds. Increasing batches did not push the
candidate rows into improved learning status: `lr=0.0025` stayed neutral at
both budgets, while `lr=0.005` became regressed at `batches=48`. The recurrent
cost advantage remained intact, but this result argues against simple budget
extension as the next quality lever. The next probe should inspect whether the
`eval_samples=64` split itself is noisier/different, or whether the training
objective needs a stronger signal than the current bigram/top-k guard.

An `--eval-samples-values 32,64,128` probe at `lr=0.0025`,
`batches=24`, `epochs=10`, `steps=12`, and `hidden=16` confirmed that eval
sample size is an important axis rather than just a reporting detail. With two
seeds, `eval_samples=32` gave the LSTM candidate an improved
`delta_nll_mean` around `-0.0007`, `eval_samples=64` moved to neutral around
`+0.0003`, and `eval_samples=128` was also neutral around `-0.0002` but had
`final_vs_bigram_mean` around `+0.0207`. The new aggregate baseline columns
make the reason explicit: `eval_samples=128` had `bigram_nll_mean` around
`2.9728` versus model `final_nll_mean` around `2.9936`, while
`eval_samples=64` had a weaker bigram baseline (`bigram_nll_mean` around
`3.0392`) and therefore a large negative `final_vs_bigram_mean`. The generated
baseline-difficulty hotspot table now ranks the `eval_samples=128` slice first
as `bigram_stronger_than_unigram` and `model_lags_bigram`, while 32 and 64 are
classified as `bigram_weaker_than_unigram` and `model_beats_bigram`. Future
learning probes should therefore grid evaluation size explicitly before
interpreting absolute NLL deltas as optimizer-only behavior. The coverage
columns add an important nuance: the requested `eval_samples=128` case actually
evaluated about 86 windows because the validation split had fewer available
windows, so the hard row is best read as a near-full-validation-slice result
rather than a simple 128-window sample. The char-LM examples now expose
`--val-start-fraction`, and the sweep wrapper exposes `--val-start-values`, so
future probes can test whether that hard near-full slice is specific to the
historical tail holdout or remains hard across earlier corpus slices. The
compare artifacts carry `val_start`, `val_start_actual_mean`, and window-count
columns so these split-location probes do not get averaged into unrelated
tail-holdout rows. The exact follow-up grid is captured as
`tools/run_char_lm_sweep.py --recipe val-start-hardness`, which expands to the
two-seed SpiralRNN/LSTM `val_start=0,0.5,1` probe below.

A follow-up `--val-start-values 0,0.5,1` probe kept the same
`eval_samples=128`, `lr=0.0025`, `batches=24`, `epochs=10`, `steps=12`, and
`hidden=16` setup over two seeds. The hard slice was not tail-specific:
all three validation locations still classified as
`bigram_stronger_than_unigram` and `model_lags_bigram`. The severity instead
formed a corpus-position gradient: `val_start=0` had the largest
`final_vs_bigram_mean` around `+0.2857`, `val_start=0.5` was around `+0.0886`,
and the historical tail `val_start=1.0` was the weakest hard case at
`+0.0207`. LSTM remained pair-quality neutral versus SpiralRNN and retained
roughly `0.25x` CPU debt across slices, but the middle slice had a slight
trace-latency regression (`trace_step_ms_ratio` around `1.03`) while the head
and tail slices stayed cost-recommended. The next quality lever should
therefore target beating a strong conditional bigram baseline across corpus
positions, not only avoiding a noisy tail holdout.

Char-LM summaries now persist both smoothed train-token unigram and previous-token
bigram validation baselines, plus `final_vs_bigram_nll_delta` and
`best_vs_bigram_nll_delta`, so context-learning probes can distinguish beating
a frequency prior from beating a simple conditional baseline. The same raw-text,
coherence-scan, and coherence-wave examples now accept `--head-prior bigram`
and `--head-prior learned-bigram`, using a context-token capture layer plus a
softmax-adjacent contextual logit prior so the bigram baseline can also become
an explicit head route rather than only an evaluation row. Validation metrics
now also persist target log-prob lift, rank lift, KL, and top-5 overlap against
the smoothed bigram row, making residual context gains above the previous-token
baseline visible in `summary.json`, `metrics.jsonl`, `compare.md`, and
route-aware compare summaries. Compare aggregation now also lifts
`final_top5_bigram_overlap` into aggregate and route-aware summary rows, and
those summaries can rank by `final_bigram_logprob_lift`,
`final_bigram_rank_lift`, or `final_top5_bigram_overlap` when the probe needs a
rank/top-k guard rather than pure NLL ordering. A one-seed residual-scale grid over
`--head-prior bigram,learned-bigram` and
`--head-residual-scale-values 0,0.25,0.5,1,2` showed the expected pure-prior anchor at scale `0`
(`bigram` matched the smoothed baseline exactly) and monotonic NLL/log-prob
lift improvement as residual scale increased, with `head_residual_scale=2`
reaching roughly `final_vs_bigram_mean=-0.0190`. The same runs also showed
negative target-rank lift once residual logits were enabled, so the next
learning-stack step should not only maximize target log-probability above the
bigram row; it should add rank/top-k-aware diagnostics or objectives to keep
contextual residuals from perturbing useful previous-token ordering.
The hardest `val_start=0` two-seed probe is now captured as
`tools/run_char_lm_sweep.py --recipe hard-bigram-prior`. In that setting,
`learned-unigram` still lagged the conditional bigram baseline
(`final_vs_bigram_mean` around `+0.2857`), while both `bigram` and
`learned-bigram` moved the same slice just past the baseline
(`final_vs_bigram_mean` around `-0.0008..-0.0010`) and improved during
training. The ranking diagnostics kept the remaining problem honest:
top-5 overlap stayed near `88%`, but target-rank lift remained around `-2.9`,
so the prior solved the hard-slice NLL gap before it solved ordering
preservation. Evaluation metrics now also persist the absolute unigram/bigram
baseline target rank plus a signed rank-debt view (`model_rank - baseline_rank`)
so the compare tables can distinguish "the baseline itself ranked this target
low" from "the learned residual pushed the target below a useful baseline
ordering." Sweep summaries can sort by `final_bigram_rank_debt` when the
objective is to minimize that ordering debt rather than maximize NLL lift.
The raw-text, coherence-scan, and coherence-wave examples now also expose
`--bigram-rank-guard F --bigram-rank-guard-margin F`. This adds a pairwise
log-probability hinge over candidates that the train-bigram baseline ranked no
higher than the true target, using the existing `--bigram-topk-guard-k` as the
comparison budget. The sweep wrapper now accepts
`--bigram-rank-guard-margin-values`, so weight/margin scans can be recorded in
the manifest instead of being run as disconnected one-offs. The first
hard-slice LSTM probe with the earlier probability-space hinge confirmed that
the objective is delicate: `rank_guard=0.5` was too weak to move the one-seed
aggregate, while `rank_guard=50` nudged `final_bigram_rank_debt_mean` from
about `3.15` to `3.12` but regressed top-5 overlap from about `88.14%` to
`86.28%` and slightly worsened the NLL/bigram gap. Treat this objective as an
experimental rank-debt pressure term; the next useful sweep should search
log-prob margins and intermediate weights before promoting it to a default.
The first log-prob-hinge sweep over `rank_guard=0,0.05,0.1,0.5` and
`margin=0,0.05` found a healthier middle point: `rank_guard=0.1`,
`margin=0.05` lowered `final_bigram_rank_debt_mean` from about `3.15` to
`3.13` while keeping NLL, `final_vs_bigram_mean`, and top-5 overlap unchanged
at the probe's displayed precision. Heavier `rank_guard=0.5` moved in the
wrong direction (`rank_debt` around `3.19`), so the useful region appears
small and should be searched locally rather than scaled aggressively. That
local search is captured as
`tools/run_char_lm_sweep.py --recipe hard-rank-guard-local`. Route-aware
compare summaries now also emit `Bigram Rank Guard Deltas` and `Bigram Rank
Guard Recommendations` whenever a rank-guard sweep includes the
`bigram_rank_guard=0` baseline at the same margin, ranking clean rank-debt
improvements separately from NLL/top-5 tradeoffs. The follow-up confirmation
grid is captured as
`tools/run_char_lm_sweep.py --recipe hard-rank-guard-confirm`, which keeps the
hardest learned-bigram LSTM slice fixed and checks `rank_guard=0,0.05,0.1` at
`margin=0.05` across two seeds before treating the local sweet spot as stable.
That confirmation run did not validate the one-seed sweet spot:
`rank_guard=0.05` was neutral, while `rank_guard=0.1` worsened mean
`final_bigram_rank_debt_mean` from about `2.90` to `2.915` with NLL,
`final_vs_bigram_mean`, and top-5 overlap unchanged at displayed precision.
The new `Bigram Rank Guard Seed Deltas` section makes the failure mode explicit:
seed `7` still improved rank debt by about `-0.02`, but seed `13` regressed by
about `+0.05`. `Bigram Rank Guard Stability` rolls those seed-level deltas up
as counts and mean/min/max rank-debt deltas; this confirmation run is therefore
classified as `rank_guard_seed_mixed` rather than a stable rank improvement.
The next diagnostic grid is captured as
`tools/run_char_lm_sweep.py --recipe hard-rank-guard-instability-map`, which
keeps the learned-bigram LSTM configuration fixed and maps
`rank_guard=0,0.05,0.1` at `margin=0.05` across `val_start=0,0.5,1` and two
seeds. Its purpose is to determine whether the mixed result is tied to the
hardest head slice, to seed sensitivity, or to rank-guard pressure itself before
increasing rank-guard weight. The first instability-map run found no stable
rank-guard recommendation: `val_start=0` kept `rank_guard=0.05` neutral and
`0.1` mixed, `val_start=0.5` regressed for both `0.05` and `0.1`, and
`val_start=1` was regressed at `0.05` and mixed at `0.1`. Aggregate NLL and
`final_vs_bigram_mean` stayed unchanged at displayed precision, but mean rank
debt moved in the wrong direction outside the isolated seed-7 improvements.
The next objective-side experiment should therefore soften the rank target or
change which competitors are guarded, rather than simply increasing
rank-guard weight.
The first softer alternative is now implemented as `--bigram-soft-guard F` on
the raw-text, coherence-scan, and coherence-wave char-LM examples. When enabled,
the guarded target stores the full smoothed previous-token bigram row alongside
the one-hot target and optional top-k/rank sections; `BigramTopKGuardedCrossEntropy`
adds a weighted full-distribution CE term instead of another pairwise rank
hinge. The sweep wrapper exposes `--bigram-soft-guard-values` and
`tools/run_char_lm_sweep.py --recipe hard-soft-guard-local`, which holds the
hard learned-bigram LSTM slice at `bigram_guard=0.1`, disables the rank hinge,
and compares `soft_guard=0,0.01,0.05,0.1` across two seeds. The first 8-run
probe did not produce a soft-guard recommendation: NLL and
`final_vs_bigram_mean` stayed neutral at displayed precision, but mean
`final_bigram_rank_debt_mean` worsened from about `2.90` at `soft_guard=0` to
about `2.905`, `2.925`, and `2.94` as the soft weight increased. The new
`Bigram Soft Guard Deltas`, `Bigram Soft Guard Seed Deltas`, and
`Bigram Soft Guard Stability` sections make that visible as
`soft_guard_alignment_regressed` / `soft_guard_seed_regressed`, with
`soft_guard=0.01` already mixed across seeds and larger weights regressing both
seeds. Treat the full-distribution soft guard as a diagnostic knob for now, not
a promoted training default. The follow-up
`tools/run_char_lm_sweep.py --recipe hard-soft-guard-micro-local` reduced the
search to `soft_guard=0,0.001,0.003,0.005,0.01` on the same two-seed hard
slice. That still produced no recommendation: every non-zero micro weight kept
NLL neutral but raised mean rank debt by about `+0.005`, with seed `7` neutral
and seed `13` regressed by about `+0.01`. This closes the "just lower the
weight" explanation for the current hard slice. The next objective-side step
should move away from direct full-row imitation: either anneal the soft term
after early stabilization, constrain only local competitor bands, or introduce
a separate KL diagnostic that does not backpropagate into target-rank debt.
The local-competitor alternative is now exposed as
`--bigram-rank-guard-band F` on the raw-text, coherence-scan, and coherence-wave
char-LM examples. `0` keeps the historical rank-guard competitor set, while a
positive band only keeps train-bigram competitors whose probability is no higher
than the target and within `target_probability - competitor_probability <= F`.
The sweep wrapper exposes `--bigram-rank-guard-band-values` and
`tools/run_char_lm_sweep.py --recipe hard-rank-band-local`, holding the hard
learned-bigram LSTM slice at `bigram_guard=0.1`, `rank_guard=0.1`, and
`margin=0.05` while comparing bands `0,0.001,0.003,0.005,0.01` across two
seeds. The first 10-run probe completed cleanly under
`target/tmp/char_lm_rank_band_local_probe`: NLL, `final_vs_bigram_mean`, and
top-5 overlap stayed unchanged at displayed precision, but the unbounded
historical band remained best on mean rank debt (`2.915` versus `2.93..2.94`
for positive bands). Seed-level movement was mixed: seed `7` regressed from
`3.13` to about `3.20..3.21`, while seed `13` improved from `2.70` to
`2.66..2.67`. Treat local bands as a diagnostic for competitor-set sensitivity,
not yet as a promoted objective. The next promising objective-side variants are
adaptive band schedules or non-backpropagating KL/rank diagnostics rather than a
fixed narrow band. Compare summaries now emit `Bigram Rank Band Deltas`,
`Bigram Rank Band Seed Deltas`, and `Bigram Rank Band Stability`, using
`band=0` as the baseline, so future band sweeps will surface aggregate
regressions and seed-mixed behavior without manual JSON inspection.
The rank guard also now records `bigram_rank_guard_coverage` in `run.json`, and
compare summaries expose it as `rank_cov_*` columns. A one-run coverage smoke at
`band=0.003` showed why the fixed narrow band is risky on the current sample:
the average unbounded candidate pool was about `41.17`, but the band retained
only about `1.26` candidates before top-k truncation and `1.07` guarded
candidates after truncation, with a `zero_guarded_candidate_ratio` around
`0.574`. This supports treating narrow bands as sparse diagnostic pressure
rather than a default objective until an adaptive schedule can avoid emptying
the rank comparison set.
That adaptive escape hatch is now a first-class knob:
`--bigram-rank-guard-min-candidates N` keeps the fixed band as the preferred
competitor set, then fills from the unbounded train-bigram competitors until at
least `N` guarded candidates are available, capped by
`--bigram-topk-guard-k`. The default `0` preserves the historical fixed-band
behavior. `tools/run_char_lm_sweep.py --recipe hard-rank-band-adaptive-local`
pins the hard learned-bigram LSTM slice at `band=0.003` and compares
`min_candidates=0,1,2,3` across seeds `7,13`. Coverage now also records
`min_candidates`, `mean_effective_rank_band`, `adaptive_fill_ratio`, and
`mean_adaptive_filled_candidates`, so future probes can distinguish "the band
was naturally populated" from "the adaptive fill rescued an otherwise sparse
guard window." Compare summaries now emit `Bigram Rank Min Recommendations`,
`Bigram Rank Min Deltas`, `Bigram Rank Min Stability`, and
`Bigram Rank Min Seed Deltas` with `min=0` as the baseline, including coverage
deltas for guarded candidates, zero-guard ratio, and adaptive-filled
competitors.
The first 8-run adaptive probe completed under
`target/tmp/char_lm_rank_adaptive_band_probe`. With `band=0.003`, `min=0`
matched the earlier sparse coverage (`mean_guarded_candidates=1.0738`,
`zero_guarded_candidate_ratio=0.5743`, aggregate rank debt `2.9400`). `min=1`
removed empty guard windows without moving the displayed objective metrics.
`min=2` and `min=3` raised guarded candidates to `2.3859` and `3.1782`, with
aggregate rank debt improving to `2.9300` and `2.9200` respectively while NLL,
`final_vs_bigram`, and top-5 bigram overlap stayed unchanged at displayed
precision. Seed-level movement is still mixed: seed `7` improved from `3.21`
rank debt at `min=0` to `3.14` at `min=3`, while seed `13` moved from `2.67`
to `2.70`. Treat adaptive fill as a stronger candidate than fixed narrow bands,
but not yet a promoted default until a larger seed/text sweep confirms the
tradeoff. That follow-up is now encoded as
`tools/run_char_lm_sweep.py --recipe hard-rank-band-adaptive-confirm`, which
keeps `band=0.003`, compares `min_candidates=0,1,2,3`, and expands the probe to
seeds `7,13,21,34` across validation starts `0,0.5,1`. Its summary should be
read primarily through the `Bigram Rank Min *` sections: a stable candidate
needs lower zero-guard ratio without regressing rank debt across seeds and
validation-start slices.
The 48-run confirmation completed under
`target/tmp/char_lm_rank_adaptive_confirm` with no failed runs and no compare
summary failure. Across validation starts, `min=1` is the stable safe candidate:
it eliminated zero-guard windows (`zero_guarded_candidate_ratio` deltas around
`-0.5743`, `-0.5249`, and `-0.5323`) while keeping max seed-level rank-debt
delta at `0.0000` for starts `0`, `0.5`, and `1`. `min=2` and `min=3` sometimes
improved aggregate rank debt (`min=3` at start `0`: `-0.0100`; `min=2` at start
`1`: `-0.0125`) but remained seed-mixed and can regress individual seeds
(`max_bigram_rank_debt_delta` up to `0.0500`). The compare summary now also
emits `Bigram Rank Min Stable Recommendations`; in this run all stable
recommendations chose `min=1`. Treat `min=1` as the next conservative objective
candidate, and keep `min=2/3` as exploratory pressure until a wider text/seed
sweep shows the rank-debt gains are not just local variance.
For follow-up checks, `tools/run_char_lm_sweep.py --recipe
hard-rank-band-adaptive-safe` narrows that confirmation grid to the fixed-band
baseline (`min=0`) versus the stable adaptive candidate (`min=1`) across the
same four seeds and three validation starts. This 24-run recipe is the
conservative promotion gate before widening the experiment to more texts or
architectures.
The next architecture-widening gate is encoded as
`tools/run_char_lm_sweep.py --recipe hard-rank-band-adaptive-safe-arches`.
It keeps the same baseline-versus-`min=1` comparison, but expands from LSTM to
the coherence-scan and coherence-wave examples over seeds `7,13` and validation
starts `0,0.5,1` (36 runs), with `memory=12` pinned so the scan/wave memory
window stays valid for `steps=12`. Read this as a portability check for the
objective: `min=1` should keep zero-guard windows eliminated without
architecture-specific rank-debt regressions before it becomes a default
learning-stack pressure.
That 36-run architecture-widening check completed under
`target/tmp/char_lm_rank_adaptive_safe_arches_memory_fixed` with no failed runs
and no compare-summary failure. The `memory=12` pin fixed the initial
scan/wave launch issue where their default memory window (`16`) exceeded
`steps=12`. Across LSTM, coherence-scan, and coherence-wave, every validation
start (`0`, `0.5`, `1`) produced
`rank_min_seed_stably_improved` for `min=1`: zero-guard ratio deltas were
`-0.5743`, `-0.5249`, and `-0.5323`, while max seed-level rank-debt delta
remained `0.0000`. This keeps `min=1` as the conservative cross-architecture
candidate; the next promotion gate should widen the text/corpus axis rather
than adding more pressure to the same sample.
That corpus-axis gate is now encoded as
`tools/run_char_lm_sweep.py --recipe hard-rank-band-adaptive-safe-corpus`.
It keeps the same LSTM/scan/wave, `min=0` versus `min=1`, two-seed, and
three-validation-start grid, but is intended to be run against a broader text
path such as `models/samples/spiral_corpus_en/` instead of the single
`spiral_demo_en.txt` sample. Its compare summary limit is widened to `12` so
all architecture-by-validation-start stable recommendations remain visible.
The first 36-run corpus-axis check completed under
`target/tmp/char_lm_rank_adaptive_safe_corpus` with no failed runs and no
compare-summary failure. It preserved the main coverage win: across all
architectures and validation starts, `min=1` eliminated zero-guard windows
(`mean_rank_cov_zero_ratio_delta` from about `-0.6486` to `-0.6940`) and raised
guarded candidates by the same amount. The corpus result is intentionally more
nuanced than the single-sample architecture check: validation starts `0.5` and
`1` were `rank_min_seed_stably_improved` for LSTM, coherence-scan, and
coherence-wave, but validation start `0` was `rank_min_seed_mixed` for all
three architectures. The mixed case was driven by seed `7`, where top-5 bigram
overlap dropped by `-1.87pp`; NLL and `final_vs_bigram` stayed unchanged, and
rank debt was neutral or slightly improved depending on architecture.
That frontier is now isolated as
`tools/run_char_lm_sweep.py --recipe
hard-rank-band-adaptive-safe-corpus-frontier`, which keeps the broader corpus
input but focuses on validation start `0` across seeds `7,13,21,34` (24 runs).
The 24-run frontier confirmation completed under
`target/tmp/char_lm_rank_adaptive_safe_corpus_frontier` with no failed runs.
It confirmed that validation start `0` remains seed-mixed rather than a
two-seed accident: LSTM and coherence-wave had `3/4` improved seeds and `1/4`
regressed seed, while coherence-scan had `2/4` improved and `2/4` regressed.
The persistent regression signal is still narrow: mean NLL and
`final_vs_bigram` deltas stayed `0.0000`, zero-guard ratio improved by
`-0.6720`, and mean top-5 overlap moved `-0.4675pp`; coherence-scan also saw a
small max rank-debt regression of `+0.0100`. Treat `min=1` as a strong
conservative candidate for non-front validation slices and as a diagnostic
pressure for corpus-front slices until either top-5 tolerance is explicitly
accepted or the frontier objective is refined.
The first frontier refinement is encoded as
`tools/run_char_lm_sweep.py --recipe
hard-rank-band-adaptive-safe-corpus-frontier-topk`. It keeps the corpus
validation-start `0` slice, focuses on the previously diagnostic seeds `7` and
`21`, and compares `min=0` versus `min=1` at top-k guard strengths
`0.1,0.2,0.5` across LSTM, coherence-scan, and coherence-wave (36 runs). That
probe completed under
`target/tmp/char_lm_rank_adaptive_safe_corpus_frontier_topk` with no failed
runs and no compare-summary failure. It reproduced the weak-guard issue at
`bigram_topk_guard=0.1` (seed `7` still dropped top-5 overlap by `-1.87pp`, and
coherence-scan seed `21` still carried the `+0.0100` rank-debt regression), but
both `0.2` and `0.5` made all three architectures
`rank_min_seed_stably_improved`: top-5 overlap deltas returned to `0.0000`,
zero-guard ratio still improved by `-0.6720`, and max rank-debt delta was never
positive. A full-seed confirmation pass then separated the two candidates. At
`bigram_topk_guard=0.2`,
`target/tmp/char_lm_rank_adaptive_safe_corpus_frontier_topk_confirm` completed
24/24 runs and kept top-5, NLL, and `final_vs_bigram` deltas at `0.0000`, but
coherence-wave seed `34` still produced a small rank-debt regression
(`+0.0100`), making the architecture-level verdict mixed. Re-running the same
gate with `bigram_topk_guard=0.5` under
`target/tmp/char_lm_rank_adaptive_safe_corpus_frontier_topk_confirm_guard05`
made LSTM, coherence-scan, and coherence-wave all
`rank_min_seed_stably_improved` across seeds `7,13,21,34`: zero-guard ratio
delta stayed `-0.6720`, top-5/NLL/`final_vs_bigram` deltas stayed `0.0000`,
and max rank-debt delta was `0.0000`. The confirm recipe
`hard-rank-band-adaptive-safe-corpus-frontier-topk-confirm` now defaults to
`bigram_topk_guard=0.5`; treat that as the current corpus-frontier candidate
before widening back to all validation starts.
The all-validation-start widening gate is encoded as
`tools/run_char_lm_sweep.py --recipe
hard-rank-band-adaptive-safe-corpus-topk-confirm`. Running that gate at
`bigram_topk_guard=0.5` under
`target/tmp/char_lm_rank_adaptive_safe_corpus_topk_confirm` completed 72/72
runs, but showed that the frontier-safe guard is too blunt as a single
all-slice default: validation start `0` stayed stable for all three
architectures, while starts `0.5` and `1` produced five small rank-debt
regressions (`+0.0100`) across scan/wave/LSTM despite unchanged NLL and top-5.
Re-running the same gate with `bigram_topk_guard=0.2` under
`target/tmp/char_lm_rank_adaptive_safe_corpus_topk_confirm_guard02` improved
the balance: 72/72 runs succeeded, scan and LSTM were stable across all
validation starts, and the only remaining regressions were coherence-wave seed
`34` at starts `0` and `1` (`+0.0100` rank debt, top-5/NLL/`final_vs_bigram`
unchanged). The all-start confirm recipe now defaults to `0.2`, while the
frontier-specific confirm recipe remains at `0.5`; treat the next refinement as
a wave-specific or validation-start-aware guard schedule rather than a single
global top-k guard.
That wave-specific refinement is now supported directly by
`tools/run_char_lm_sweep.py --bigram-topk-guard-schedule`, using
`validation_start:guard` pairs while still passing ordinary
`--bigram-topk-guard` values to each Rust example. The middle-guard probe
`hard-rank-band-adaptive-wave-topk-middle` filled the missing `0.3` and `0.4`
evidence for coherence-wave: guard `0.3` was stable at validation starts `0.5`
and `1` but still regressed seed `13` at start `0`; guard `0.4` was stable at
start `1`, mixed at start `0`, and had a top-5 regression at start `0.5`.
The resulting schedule candidate
`0:0.5,0.5:0.2,1:0.3` is encoded as
`hard-rank-band-adaptive-wave-topk-schedule-confirm`. Its 24-run confirmation
under `target/tmp/char_lm_rank_adaptive_wave_topk_schedule` succeeded with all
three validation starts `rank_min_seed_stably_improved`, no regressed seed
deltas, zero-guard ratio deltas preserved (`-0.6720`, `-0.6940`, `-0.6486`),
and top-5/NLL/`final_vs_bigram` deltas at `0.0000`. This makes a
validation-start-aware top-k guard schedule the cleanest current path for
coherence-wave, while LSTM/scan can keep the simpler all-start `0.2` candidate.
The combined architecture-aware gate is now encoded as
`hard-rank-band-adaptive-architecture-topk-schedule-confirm`, using
`--bigram-topk-guard-arch-schedule
"*@0:0.2,*@0.5:0.2,*@1:0.2,wave@0:0.5,wave@1:0.3"`. The wildcard keeps
LSTM/scan on the all-start `0.2` candidate and only overrides coherence-wave at
the frontier and late-validation starts. Its 72-run confirmation under
`target/tmp/char_lm_rank_adaptive_arch_topk_schedule` completed with no failed
runs, no compare-summary failure, and all nine architecture-by-validation-start
rows `rank_min_seed_stably_improved`: no regressed seed deltas, max rank-debt
delta `0.0000`, top-5/NLL/`final_vs_bigram` deltas `0.0000`, and the same
zero-guard coverage gains. This is the current strongest corpus-axis promotion
candidate for adaptive `min=1` rank fill.
The first scale sanity gate,
`hard-rank-band-adaptive-architecture-topk-schedule-scale`, keeps that
architecture-aware schedule but raises the shape/training budget to
`steps=16`, `embed_dim=12`, `hidden=24`, `memory=16`, `epochs=12`,
`batches=32`, and `eval_samples=192` over seeds `7,13` (36 runs). It completed
under `target/tmp/char_lm_rank_adaptive_arch_topk_schedule_scale` with no
failed runs, no compare-summary failure, and again all nine
architecture-by-validation-start rows `rank_min_seed_stably_improved`. No seed
deltas regressed; max rank-debt delta stayed `0.0000`; top-5/NLL/
`final_vs_bigram` deltas stayed `0.0000`; and the zero-guard coverage gains
held at `-0.6720`, `-0.6940`, and `-0.6486`. This moves the schedule from a
small-shape corpus gate to a viable medium-shape learning-stack candidate.
The complementary seed pass,
`hard-rank-band-adaptive-architecture-topk-schedule-scale-confirm`, repeats the
same medium-shape gate over seeds `21,34` (36 runs). It completed under
`target/tmp/char_lm_rank_adaptive_arch_topk_schedule_scale_confirm` with no
failed runs, no compare-summary failure, and all nine rows again
`rank_min_seed_stably_improved`: no regressed seed deltas, max rank-debt delta
`0.0000`, and top-5/NLL/`final_vs_bigram` deltas `0.0000`. Together with the
`7,13` pass, the architecture-aware schedule now has medium-shape coverage
across seeds `7,13,21,34`.
`tools/summarize_char_lm_compare.py` now accepts `--merge-evidence-sources` so
split confirmation shards can be read as one promotion-evidence set without
changing the default source-by-source compare view. Merging the two medium
artifacts into
`target/tmp/char_lm_rank_adaptive_arch_topk_schedule_scale_split_merged_summary.json`
produced nine `merged:2` stability rows, all with `seed_pairs=4`,
`rank_min_seed_stably_improved`, zero regressed seed deltas, max rank-debt delta
`0.0000`, and top-5/NLL/`final_vs_bigram` deltas `0.0000`.
For a one-shot reproduction of that medium-shape evidence, use
`hard-rank-band-adaptive-architecture-topk-schedule-scale-full`. It keeps the
same shape, training budget, architecture-aware guard schedule, and compare
summary settings, but expands the seed grid to `7,13,21,34` in a single 72-run
manifest. The split `scale` plus `scale-confirm` runs are the current completed
evidence; the `scale-full` recipe is the promotion gate to rerun when a single
artifact is needed.
The next corpus-widening gate is now encoded as
`hard-rank-band-adaptive-architecture-topk-schedule-corpus-widen`. It keeps the
same medium shape, training budget, architecture-aware guard schedule, and
`min=0/1` comparison, but is intended to be run with multiple positional text
or corpus paths so the vocabulary and bigram prior are learned from a wider
input distribution. A dry-run manifest under
`target/tmp/char_lm_rank_adaptive_arch_topk_schedule_corpus_widen_dry_run`
validated the first 36-run shard over
`models/samples/spiral_corpus_en` plus `models/samples/spiral_demo_en.txt`:
seeds `7,13`, `steps=16`, `embed_dim=12`, `hidden=24`, `memory=16`,
`epochs=12`, `batches=32`, `eval_samples=192`, all three validation starts,
and the same architecture-aware top-k guard set `[0.2, 0.3, 0.5]`.
A two-run runtime smoke with the same widened input, narrowed to LSTM, seed
`7`, validation start `0`, `epochs=2`, and `batches=4`, completed under
`target/tmp/char_lm_rank_adaptive_arch_topk_schedule_corpus_widen_smoke`.
That smoke verified the subset execution path, compare summary generation, and
effective schedule reporting (`bigram_topk_guards=[0.2]` for LSTM-only). In the
smoke result, `min=1` removed the rank-coverage zero ratio
(`mean_rank_cov_zero_ratio_delta=-0.7121`) while leaving NLL,
`final_vs_bigram`, top-5 overlap, and rank debt neutral.
Complementary and one-shot gates are available as
`hard-rank-band-adaptive-architecture-topk-schedule-corpus-widen-confirm`
(seeds `21,34`) and
`hard-rank-band-adaptive-architecture-topk-schedule-corpus-widen-full`
(seeds `7,13,21,34`). Once both corpus-widen shards complete, merge their
`compare.json` files with `tools/summarize_char_lm_compare.py
--merge-evidence-sources` to get one promotion-evidence view across all four
seeds. The same split-shard handoff can now be done from the runner by adding
`--compare-summary-extra-compare-json PREVIOUS_COMPARE_OR_SWEEP_DIR` and
`--compare-summary-merge-evidence-sources` to the second shard; the resulting
`sweep.json` records both the extra compare inputs and the merge flag under
`compare_summary`, the exact replay command under `compare_summary.command`,
plus the summarizer-resolved compare inputs under `compare_summary_sources`.
The runner also writes an executable `compare_summary_command.sh` next to the
compare artifacts and records it as `compare_summary_command_path`, so a merged
evidence summary can be replayed without reconstructing CLI flags by hand. The
script records and `cd`s to the same repository root stored as
`compare_summary.command_cwd`, so relative extra compare inputs resolve the same
way they did during the runner invocation.
Because those options request a specific merged-evidence
artifact, compare-summary generation failures from missing or invalid extra
inputs now mark the sweep failed instead of leaving a successful manifest with
no merged evidence. Replaying the completed confirm shard with
`--skip-existing --max-new-runs 0`, the previous shard as
`--compare-summary-extra-compare-json`, `--compare-summary-merge-evidence-sources`,
and a fail gate on `no_rank_min_evidence,needs_tuning,partial_promote_needs_tuning`
updated
`target/tmp/char_lm_rank_adaptive_arch_topk_schedule_corpus_widen_confirm/sweep.json`
without launching new runs. The manifest now records
`compare_summary.merge_evidence_sources=true` and
`compare_summary.extra_compare_paths=[target/tmp/char_lm_rank_adaptive_arch_topk_schedule_corpus_widen]`.
The compare artifacts now also carry raw rank/top-5 diagnostic columns
alongside the display-rounded cells, and the rank-min summary prefers those raw
values when classifying seed deltas. Replaying the merged corpus-widen evidence
with raw precision tightened the gate from display-rounded
`promote_with_bounded_watch` to `partial_promote_needs_tuning`: four rows stayed
strict promotions, four stayed bounded promotions, and LSTM at validation start
`0.5` became `rank_min_seed_mixed` because seeds `13` and `21` each regressed
rank debt by about `+0.0052` while still eliminating zero-coverage windows. The
focused follow-up recipe is
`hard-rank-band-adaptive-architecture-topk-schedule-corpus-widen-hotspot`, which
keeps the widened-corpus medium shape but narrows to LSTM, validation start
`0.5`, seeds `7,13,21,34`, and top-k guard values `0.2,0.3,0.4,0.5` against
`bigram_rank_guard_min_candidates=0,1`. Running the hotspot over guards
`0.3`, `0.4`, and `0.5` showed a clean monotonic repair: guard `0.3` stayed
`rank_min_seed_mixed` with two improved and two regressed seeds, guard `0.4`
became `rank_min_seed_bounded_mixed` with three improved and one regressed
seed, and guard `0.5` became `rank_min_seed_stably_improved` with all four
seeds improved and max rank-debt delta `0.0000`. The corpus-widen,
corpus-widen-confirm, and corpus-widen-full recipes first promoted that hotspot
directly via `lstm@0.5:0.5` while preserving the existing coherence-wave
overrides.
Re-running the promoted `hard-rank-band-adaptive-architecture-topk-schedule-corpus-widen-full`
gate in chunks under
`target/tmp/char_lm_rank_adaptive_arch_topk_schedule_corpus_widen_promoted_full`
completed all 72 runs with no failed runs, no compare-summary failure,
`run_limit_reached=false`, and no next resume point. The full promoted gate
passes as `promote_with_bounded_watch`: nine recommendation rows, five strict
promotions, four bounded promotions, and zero non-promoted rows. Strict rows
are LSTM validation starts `0`, `0.5`, and `1`, coherence-scan validation start
`1`, and coherence-wave validation start `0`. The promoted LSTM hotspot is now
fully repaired in the full recipe: validation start `0.5` uses
`bigram_guard=0.5000`, removes the zero-coverage ratio by `-0.7232`, keeps
max rank-debt delta at `0.0000`, and leaves top-5/NLL/`final_vs_bigram`
neutral across all four seeds. The bounded rows are coherence-scan validation
starts `0` and `0.5`, coherence-wave validation start `0.5`, and coherence-wave
validation start `1`; each still has three improved seeds and one regressed
seed, no non-promoted row, and neutral NLL/`final_vs_bigram`. The scan and wave
`0.5` bounded rows carry a small max rank-debt delta of `0.0052`; the wave
validation-start `1` bounded row instead keeps max rank-debt delta at `0.0000`
but has a mean top-5 overlap delta of `-0.0521pp`.
Follow-up bounded-row hotspots show that three of those four bounded rows have
a clean stricter schedule candidate. Coherence-scan validation start `0.5`
was non-monotonic across top-k guard weights (`0.3` became
`rank_min_seed_mixed`, `0.4` returned to bounded, and `0.5` became
`rank_min_seed_stably_improved`), while coherence-scan validation start `0`
and coherence-wave validation start `0.5` also became strict at guard `0.5`.
Coherence-wave validation start `1` did not become strict at guard `0.5`:
the top-5 overlap regression disappeared, but rank-debt regressed instead
(`max_bigram_rank_debt_delta=0.0104`), so the safer schedule keeps
`wave@1:0.3`. The corpus-widen schedule therefore now promotes
`scan@0:0.5`, `scan@0.5:0.5`, and `wave@0.5:0.5` in addition to
`lstm@0.5:0.5` and `wave@0:0.5`, while leaving `wave@1:0.3` as a bounded
watch row.
The updated schedule was then re-run end-to-end under
`target/tmp/char_lm_rank_adaptive_arch_topk_schedule_corpus_widen_strict_candidates_full`,
resuming after an interrupted final wave chunk with `--skip-existing`. The gate
completed all 72 planned runs with `failed=false`, `run_limit_reached=false`,
no next resume point, and `compare_summary_failed=false`. The final
rank-min promotion gate is `promote_with_bounded_watch`: nine stability rows,
eight strict promotions, one bounded promotion, and zero non-promoted rows.
The strict rows are LSTM validation starts `0`, `0.5`, and `1`,
coherence-scan validation starts `0`, `0.5`, and `1`, and coherence-wave
validation starts `0` and `0.5`; all eight have all four seeds improved,
`max_bigram_rank_debt_delta=0.0000`, neutral top-5 overlap, and neutral
NLL/`final_vs_bigram`. The only bounded row is coherence-wave validation start
`1` at guard `0.3`, with three improved seeds, one regressed seed,
`max_bigram_rank_debt_delta=0.0000`, neutral NLL/`final_vs_bigram`, and a small
mean top-5 overlap delta of `-0.0521pp`. This confirms the stricter
corpus-widen schedule while keeping `wave@1:0.3` explicitly on bounded watch.
Long corpus-widen runs can now be resumed in chunks with
`tools/run_char_lm_sweep.py --skip-existing --max-new-runs N`. The runner
reuses existing `summary.json` artifacts, records `planned_runs`,
`new_runs_started`, `run_limit_reached`, and `next_run_after_limit`, then still
renders compare artifacts for the completed subset. Replaying the interrupted
corpus-widen shard with `--skip-existing --max-new-runs 0` produced a partial
manifest with `planned_runs=36`, two existing runs collected, no new runs
started, and `next_run_after_limit` pointing at the first missing `min=1`
LSTM run.
Two further one-run resume chunks completed that first missing `min=1` pair:
LSTM, validation start `0`, seeds `7,13`. The partial corpus-widen summary now
has four completed runs and one rank-min stability row with `seed_pairs=2`,
`rank_min_seed_stably_improved`, `alignment_improved_seeds=2`,
`mean_rank_cov_zero_ratio_delta=-0.7121`, max rank-debt delta `0.0000`, and
top-5/NLL/`final_vs_bigram` deltas `0.0000`. The next chunk resumes at the
LSTM validation-start `0.5` baseline run.
Two more four-run chunks completed the remaining LSTM validation starts. The
partial corpus-widen shard now has all 12 LSTM runs complete and three
rank-min stability rows. Validation starts `0` and `1` are
`rank_min_seed_stably_improved` with `seed_pairs=2`, no regressed seeds,
zero-coverage deltas `-0.7121` and `-0.7261`, max rank-debt delta `0.0000`,
and top-5/NLL/`final_vs_bigram` deltas `0.0000`. Validation start `0.5`
still removes zero coverage (`mean_rank_cov_zero_ratio_delta=-0.7232`) with
neutral top-5/NLL/`final_vs_bigram`, but is `rank_min_seed_mixed` because seed
`13` adds a small rank-debt regression (`+0.0100`, rank-lift `-0.0100`). The
next chunk resumes at the first coherence-scan run.
Three four-run chunks then completed all coherence-scan validation starts over
seeds `7,13`. Coherence-scan now mirrors the LSTM pattern exactly: validation
starts `0` and `1` are `rank_min_seed_stably_improved` with `seed_pairs=2`,
zero regressed seeds, zero-coverage deltas `-0.7121` and `-0.7261`, max
rank-debt delta `0.0000`, and neutral top-5/NLL/`final_vs_bigram` deltas.
Validation start `0.5` still removes zero coverage
(`mean_rank_cov_zero_ratio_delta=-0.7232`) with neutral
top-5/NLL/`final_vs_bigram`, but is `rank_min_seed_mixed` because seed `13`
adds a small rank-debt regression (`+0.0100`, rank-lift `-0.0100`). The partial
corpus-widen shard now has 24/36 runs complete and six rank-min stability rows;
the next chunk resumes at the first coherence-wave run.
The remaining three chunks completed all coherence-wave validation starts over
seeds `7,13` under the architecture-specific top-k schedule
(`wave@0:0.5`, `wave@0.5:0.2`, `wave@1:0.3`). Validation start `0` is
`rank_min_seed_stably_improved` with `seed_pairs=2`, zero regressed seeds,
`mean_rank_cov_zero_ratio_delta=-0.7121`, max rank-debt delta `0.0000`, and
neutral top-5/NLL/`final_vs_bigram` deltas. Validation start `0.5` again
matches the LSTM/scan pattern: `rank_min_seed_mixed`, seed `13` rank-debt
delta `+0.0100` and rank-lift delta `-0.0100`, zero-coverage delta `-0.7232`,
and neutral top-5/NLL/`final_vs_bigram`. Validation start `1` is also mixed,
but for a different reason: rank debt/NLL/`final_vs_bigram` stay neutral while
seed `13` loses `0.2100pp` top-5 overlap, for a mean top-5 delta of
`-0.1050pp`. The first corpus-widen shard is now complete: 36/36 runs, nine
rank-min stability rows, and five stable-alignment recommendations.
The confirm corpus-widen shard
(`hard-rank-band-adaptive-architecture-topk-schedule-corpus-widen-confirm`,
seeds `21,34`) completed all 36 runs with the same corpus inputs. It produced
nine rank-min stability rows and eight stable-alignment recommendations, much
stronger than the first shard. All LSTM rows are
`rank_min_seed_stably_improved`; notably, validation start `0.5` has
zero-coverage delta `-0.7232`, max rank-debt delta `0.0000`, and neutral
top-5/NLL/`final_vs_bigram`, so the first shard's LSTM `0.5` mixed result looks
seed-local rather than deterministic.
Coherence-scan in the confirm shard is mixed only at validation start `0`,
where seed `21` adds a small rank-debt regression (`+0.0100`, rank-lift
`-0.0100`) while seed `34` stays neutral; zero-coverage delta is still
`-0.7121` and top-5/NLL/`final_vs_bigram` remain neutral. Coherence-scan
validation starts `0.5` and `1` are stable. All coherence-wave confirm rows
are stable, including the first shard's previously mixed validation starts
`0.5` and `1`: max rank-debt delta `0.0000`, top-5 delta `0.0000`, and neutral
NLL/`final_vs_bigram`.
Before raw rank diagnostic columns were added, merging the two completed
corpus-widen shards with
`tools/summarize_char_lm_compare.py --merge-evidence-sources` produced
`target/tmp/char_lm_rank_adaptive_arch_topk_schedule_corpus_widen_merged_summary.json`.
That display-rounded merged view had 72 total runs, 36 rank-min seed deltas,
nine merged stability rows, and nine promotion recommendations after adding a
separate `rank_min_seed_bounded_mixed` verdict. Its display-rounded promotion
gate decision was `promote_with_bounded_watch`: four strict rows plus five
bounded rows, zero non-promoted rows. The four strict stable rows were LSTM
validation starts `0` and `1`, coherence-scan validation start `1`, and
coherence-wave validation start `0`; each had `seed_pairs=4`,
`alignment_improved_seeds=4`, no regressed seeds, zero-coverage deltas between
`-0.7121` and `-0.7261`, max rank-debt delta `0.0000`, and neutral
NLL/`final_vs_bigram`. The remaining five rows became
`bounded_alignment_improved`: each had `alignment_improved_seeds=3`,
`alignment_regressed_seeds=1`; the bounded regression was either a `+0.0100`
rank-debt blip or, for coherence-wave validation start `1`, a mean top-5
overlap delta of `-0.0525pp` with neutral rank debt/NLL/`final_vs_bigram`.
Two-seed shard-level mixed rows remain mixed; the bounded promotion verdict is
reserved for merged evidence with at least four seed pairs and at most one
regressed seed.
`run_char_lm_sweep.py` now copies this gate into each sweep manifest as
`compare_summary_bigram_rank_min_promotion_gate`, so a completed recipe can be
checked from `sweep.json` without opening the full compare summary. The
summarizer can also fail on selected gate decisions with
`--fail-on-rank-min-promotion-decision`, and the sweep runner forwards that via
`--compare-summary-fail-on-rank-min-promotion-decision`, making promotion policy
usable as an explicit local/CI gate. The full four-seed promotion recipes now
enable that gate by default for `no_rank_min_evidence`, `needs_tuning`, and
`partial_promote_needs_tuning`, while allowing `promote` and
`promote_with_bounded_watch` to pass; the active fail decisions are recorded in
`sweep.json` under `compare_summary.fail_on_rank_min_promotion_decisions`. The
individual corpus-widen shard manifests still read
`partial_promote_needs_tuning` (`5/9` and `8/9` strict stable rows). The old
display-rounded merged evidence read `promote_with_bounded_watch` because all
remaining mixed rows satisfied the bounded-promotion criteria, but the raw
diagnostic replay now reads `partial_promote_needs_tuning` until the promoted
`lstm@0.5:0.5` schedule is rerun through the corpus-widen gate.
The top-k preservation objective remains exposed as
`--bigram-topk-guard F --bigram-topk-guard-k N`
on the Rust raw-text, coherence-scan, and coherence-wave char-LM examples. When
enabled, training targets carry the normal one-hot next token plus a smoothed
train-bigram top-k distribution for the previous token, and
`BigramTopKGuardedCrossEntropy` optimizes standard next-token CE plus the
weighted top-k preservation CE. Sweep manifests and compare aggregate keys carry
`bigram_guard` and `bigram_guard_k`, so guarded and unguarded runs do not get
averaged together. Route-aware compare summaries now also emit `Bigram Guard
Deltas` whenever a guard sweep includes the `bigram_guard=0` baseline, so NLL,
bigram lift, rank lift, and top-5 preservation can be read as direct deltas
against the unguarded objective. The delta rows also carry per-metric statuses
and a guard-specific verdict, because early corpus probes showed a real mixed
case: higher guard weights can slightly improve NLL/top-5 overlap while nudging
target-rank lift downward. A recommendation section now filters those deltas to
surface clean guard candidates separately from mixed top-k tradeoffs.
A 24-run follow-up over SpiralRNN and LSTM (`head_residual_scale=1,2`,
`bigram_guard=0,0.05,0.1`, two seeds) kept quality effectively tied between
the recurrent cores while surfacing the same guarded sweet spot. The LSTM rows
were recommended on cost: `head_residual_scale=2`, `bigram_guard=0.1` retained
the same mean NLL/bigram gap as SpiralRNN while lowering trace step time to
about `0.67x` and CPU debt to about `0.51x`; both cores showed clean
top-k-improved guard recommendations at `head_residual_scale=2`,
`bigram_guard=0.05..0.1`.
A 32-run training-budget follow-up then held `head_residual_scale=2` and
compared `bigram_guard=0` versus `0.1` across `epochs=1,2` and
`batches=8,16`. The guarded objective remained stable as budget increased:
at `epochs=2`, `batches=16`, `bigram_guard=0.1` improved mean NLL by roughly
`0.0011` over the unguarded row for both recurrent cores, without worsening
target-rank lift or top-5 overlap. The same row gave the strongest recurrent
cost recommendation: LSTM matched SpiralRNN quality while lowering trace step
time to about `0.36x` and CPU debt to about `0.51x`. This configuration is now
captured as `tools/run_char_lm_sweep.py --recipe guarded-lstm`, which expands
the validated SpiralRNN/LSTM, bigram-prior, `head_residual_scale=2`,
`bigram_guard=0,0.1`, and training-budget grid while still allowing explicit
CLI flags to override any generated axis.
A hard-slice variant is also captured as
`tools/run_char_lm_sweep.py --recipe hard-bigram-guard`, which holds
`val_start=0`, `eval_samples=128`, `head_residual_scale=0.5`, and compares
`bigram_guard=0` versus `0.1` plus `bigram_rank_guard=0` versus `0.05` for
`bigram` and `learned-bigram` priors. In the preceding top-k-only 16-run probe,
the fixed
`bigram` prior did not gain top-5 overlap from
the guard, but `learned-bigram + bigram_guard=0.1` moved top-5 overlap from
about `87.44%` to `88.14%` while nudging the NLL/bigram gap slightly further
negative. The clean guard recommendation surfaced only for the LSTM
`learned-bigram` row; target-rank lift remained roughly flat near `-2.9`, so
this recipe should be read as a top-k-preservation check rather than a complete
rank-order fix.
The same guarded objective has been smoke-tested through the coherence-scan and
coherence-wave char-LM entrypoints with `--head-prior bigram`,
`--head-residual-scale 2`, and `--bigram-topk-guard 0.1`; compare output records
`bigram_guard`, `bigram_guard_k`, and residual-scale metadata for both routes.
When shrinking those examples for very small runtime checks, keep
`--memory <= --steps`; otherwise the rolling context validation correctly
rejects the shortened sequence before training starts.
The CPU seam is covered by `lstm_backward_scan_cpu_matches_reference_gate_gradients`,
which compares its gate gradients against an independent reverse-time reference
loop before the full `backward()` path consumes them.
`run_lstm_sweep.py` also resolves implicit Cargo features per backend, so a
mixed `--backends cpu,wgpu` run keeps CPU baselines CPU-only while enabling the
`wgpu` feature only for WGPU rows. With `--continue-on-error`, failed WGPU rows
still leave `run_status`, `returncode`, `log_path`, `failure.json`, and
`sweep.json` evidence next to successful CPU rows instead of aborting the whole
comparison. Failure rows also carry `failure_kind` and `failure_detail`, which
separates compile errors from native runtime aborts such as
`signal:6:empty_log`. WGPU rows now run behind a small default preflight probe:
if WGPU setup fails once, the per-seed rows are recorded as `preflight_*`
failures pointing at the shared preflight log instead of re-triggering the same
native abort for every seed. Pass `--no-wgpu-preflight` when intentionally
debugging the direct WGPU launch path. The same process/failure primitives now
live in `backend_sweep_meta.py`, and `run_wave_rnn_sweep.py` uses the same
`run_status`, `returncode`, `failure_kind`, `failure_detail`, and WGPU preflight
contract while preserving CPU-only baselines in mixed `cpu,wgpu` sweeps.
`run_vision_conv_sweep.py` and `run_gnn_band_trace_sweep.py` now follow that
same contract too, so recurrent, vision, and graph model-zoo comparisons can
record adapter/runtime failure once and still produce compareable CPU rows.
`run_selfsup_sweep.py` now uses the same flow for InfoNCE probes, keeping
CPU-only baselines CPU-only in mixed sweeps while surfacing WGPU runtime aborts
as explicit `preflight_*` rows. `run_char_lm_sweep.py` keeps its richer
multi-architecture manifest, but now classifies failed rows with
`run_status`, `failure_kind`, and `failure_detail`, runs WGPU rows behind the
same one-shot preflight, and enables the `wgpu` Cargo feature implicitly for
WGPU rows when `--cargo-features` is omitted. Its `compare.md` keeps the
existing detailed successful-run table and appends a compact failed-run table so
preflight skips are visible without opening `sweep.json`.
Trainer summaries also aggregate the LSTM `estimated_bptt_*` and
gate-activation operation counts as `lstm_estimated_cpu_debt_ops`,
`lstm_estimated_gate_activation_ops`,
`lstm_estimated_gate_activation_cpu_debt_ops`,
`lstm_estimated_gate_activation_wgpu_ops`,
`lstm_forward_estimated_gate_activation_wgpu_ops`,
`lstm_backward_estimated_gate_activation_wgpu_ops`,
`lstm_estimated_bptt_cpu_debt_ops`, `lstm_estimated_bptt_wgpu_ops`,
`lstm_backward_estimated_bptt_ops`,
`lstm_backward_estimated_bptt_cpu_debt_ops`,
`lstm_backward_estimated_bptt_wgpu_ops`,
`lstm_backward_estimated_bptt_gate_derivative_ops`,
`lstm_backward_estimated_bptt_cell_recurrence_ops`,
`lstm_backward_estimated_bptt_state_carry_ops`, and
`lstm_backward_estimated_bptt_scan_steps`. The same scan profile now exposes
`lstm_backward_bptt_scan_elapsed_us`,
`lstm_backward_bptt_scan_gate_values`,
`lstm_backward_bptt_scan_recurrent_weight_values`,
`lstm_backward_bptt_scan_kernel_dispatches`,
`lstm_backward_bptt_scan_workgroup_size`,
`lstm_backward_bptt_scan_parallel_lanes`, and
`lstm_backward_estimated_bptt_ops_per_scan_step`, so WGPU scan changes can be
compared by shape and rough step cost rather than only by route success. CPU
debt now counts gate activation and BPTT work only when those regions actually
ran as CPU; WGPU gate activation and WGPU scan work are reported separately so
successful adapter runs no longer look like unchanged CPU LSTM debt. Smoke runs under
`target/tmp/*_backend_meta_smoke` confirm WaveRNN, vision, and GNN WGPU rows reach
`above:exact_runtime_ready,here:exact_runtime_ready,beneath:exact_runtime_ready`.
The self-supervised smoke records WGPU runtime initialization and tensor policy
but leaves `rt_wgpu_ready=-` because that entrypoint does not currently expose a
roundtable audit; this is intentional rather than silently inventing one.

The threshold-grid tools now preserve the same backend truth surface. WaveRNN,
vision conv-pool, and GNN threshold grids load `run.json` for every cell, append
`policy_matmul`, `policy_softmax`, `rt_wgpu_initialized`, `rt_wgpu_ctx`,
`rt_wgpu_ready`, and `rt_wgpu_statuses` to the per-run comparison table, and
copy `backend_runtime`, `tensor_policy`, and `roundtable_wgpu` into `grid.json`.
Minimal WGPU smokes under
`target/tmp/wave_rnn_threshold_backend_meta_smoke`,
`target/tmp/vision_conv_threshold_backend_meta_smoke`, and
`target/tmp/gnn_threshold_backend_meta_smoke` confirm that threshold routing
labels such as `wgpu` / `cpu-threshold` now sit beside real runtime readiness:
all three WGPU rows reported `rt_wgpu_ready=yes` and
`above:exact_runtime_ready,here:exact_runtime_ready,beneath:exact_runtime_ready`.
This matters for learning because threshold tuning can now distinguish "the
tensor utility op crossed the WGPU size threshold" from "the actual rank-band
runtime was ready for the training step".

The shared sweep metadata helper now also renders backend residual columns from
trainer trace summaries: `fallback_share`, `util_route_status`,
`util_route_values`, `util_route_threshold`, `cpu_residual_ops`,
`cpu_residual_share`, `cpu_residual_top`, `cpu_threshold_ops`,
`cpu_threshold_share`, `cpu_threshold_top`, `cpu_trace_ops`,
`cpu_trace_share`, `cpu_trace_top`, `cpu_control_ops`, `cpu_control_share`,
`cpu_control_top`, `cpu_runtime_fallback_ops`,
`cpu_runtime_fallback_share`, `cpu_runtime_fallback_top`, `cpu_copy_ops`,
`cpu_copy_share`, `cpu_copy_top`, `cpu_debt_ops`, `cpu_debt_share`,
`cpu_debt_top`, `wgpu_kernel_ops`, `wgpu_kernel_share`, and
`wgpu_kernel_top`.
`cpu_residual_*` intentionally groups explicit CPU, CPU-SIMD, and naive
tensor-op backends because all three represent remaining host-side work from the
perspective of a WGPU-first learning run. `tensor_util_route` is emitted as
backend-less policy metadata, so the trainer can report `cpu_threshold` or
`wgpu` size-guard decisions without inflating `tensor_ops_total` or miscounting
the route decision as a real kernel. The helper then splits residual CPU work
into `cpu_threshold_*` for threshold-protected tensor utilities,
`cpu_trace_*` for telemetry-only host reports, `cpu_control_*` for host
queue/callback/collective-arena control planes, `cpu_runtime_fallback_*` for
ops that requested WGPU but executed on CPU/naive after runtime fallback,
`cpu_copy_*` for host-side data movement such as row concatenation, and
`cpu_debt_*` for the remaining CPU/naive compute ops. The compact top-op labels
make the next kernel candidates visible
without reading the long per-op tail: in the current char-LM WGPU smoke,
`cpu_residual_top` points at `hadamard`, `add_scaled`, and `add`, but
`cpu_debt_top` correctly points at `transpose`, hypergrad accumulation, and
embedding gather/scatter; threshold grids show when small bias/readout
reductions are deliberate CPU-threshold work rather than accidental fallback.
Routeable LSTM sub-backend annotations such as input projection, recurrent
projection, bias reduction, and parameter-gradient scaling are treated as
alias-only in backend residuals because concrete `matmul`, `sum_axis0`, `scale`,
and related tensor utility events already carry the real kernel or
runtime-fallback evidence. LSTM gate activation and reverse-time BPTT are now
backend-aware as well: successful WGPU gate/scan traces add to the WGPU
estimate columns and keep `cpu_debt_*` at zero, while CPU/reference/fallback
events still surface as debt only when host recurrent work actually ran.
The same columns are now emitted by
`compare_char_lm_runs.py`, the WaveRNN/vision/selfsup/GNN sweep wrappers, and
the three tensor-utility threshold-grid tools.

Backend runtime metadata now also distinguishes feature availability from real
kernel wiring for non-WGPU targets. `st-core::backend::runtime_probe` exposes
stable helpers for `backend_feature_enabled`, `backend_real_kernels_compiled`,
`backend_placeholder`, `backend_runtime_status`, and
`backend_runtime_recommendation`. The shared model-zoo backend runtime record
copies those into `run.json` as `requested_backend_feature_enabled`,
`requested_backend_kernels_wired`, `requested_backend_placeholder`,
`requested_backend_status`, and `requested_backend_recommendation`, plus
backend-specific flags such as `hip_real_compiled`, `hip_kernels_compiled`,
`cuda_kernels_compiled`, and `mps_placeholder`. The char-LM compare table now
surfaces `backend_status`, `backend_kernels`, `backend_feature`, and `hip_real`;
the shared sweep helper surfaces `backend_status` and `backend_kernels` for the
non-char learning sweeps. A WGPU char-LM smoke under
`target/tmp/char_lm_backend_honesty_smoke` confirms CPU rows report
`backend_status=cpu`, while WGPU rows report `backend_status=kernel_wired`,
`backend_kernels=yes`, and `rt_wgpu_ready=yes`. A WaveRNN smoke under
`target/tmp/wave_rnn_backend_honesty_smoke` confirms the same shared columns are
present outside char-LM. This closes an important audit gap for HIP/MPS/CUDA:
future learning runs can now show "feature selected but kernels are placeholder
or stubbed" without relying on a separate build log.

Next steps:

1. Use `cpu_debt_top` together with `util_route_status` across char, WaveRNN,
   vision, self-supervised, and GNN smokes to choose the next WGPU kernel family
   instead of guessing from aggregate `tensor_cpu` counts.
2. Split HIP metadata into `planner_available` and `kernel_available`.
3. Emit explicit tensor meta when HIP was considered but skipped.
4. Run the opt-in WGPU TopK/MidK/BottomK runtime parity tests on adapter-backed
   machines and wire scalable MidK central-band compaction next.
5. Make benchmark and training harnesses print whether `hip-real` is compiled.
6. Add a strict HIP check that fails before training when GEMM is not real or
   when the trainer requested HIP but tensor kernels never attempted HIP.

### P1: Promote backend parity tests from kernel tests to learning tests

There are kernel-level tests, but the risky boundary is full learning behavior:
forward, backward, band replay, accumulator update, and trace output.

Next test targets:

1. `Linear` one-step CPU versus WGPU parity.
2. GNN graph-regressor one-step CPU versus WGPU parity with band replay.
3. Char-LM one mini-epoch trace that proves projection matmuls used the
   requested backend or recorded fallback.
4. Layer norm, softmax, GELU backward, and attention parity at training shapes.

### P1: Unify optimizer state with trainer update policy

`Parameter` owns hypergrad, realgrad, Euclidean fallback gradients, and packed
matmul caches. `ModuleTrainer` applies local spectral scaling through
`LocalLearningRateAdapter`. This works, but optimizer state is not yet a first
class serializable training component.

Next steps:

1. Define a trainer optimizer state snapshot for hypergrad, realgrad, adapter
   policy, and backend policy.
2. Include backend counters and fallback counts in checkpoints.
3. Add resume tests that confirm one interrupted run matches an uninterrupted
   run for a deterministic seed.

### P2: Replace backend panics with trainable errors

Some low-level WGPU paths used to panic on submit or readback failure. That is
acceptable for isolated exploratory kernels, but not for long training jobs.
The legacy `st-core::backend::wgpu_frac` wrapper is now back behind the
`wgpu-rt` feature with `Result`-returning `fracdiff_gl_wgpu` and
`specmul_frac_laplace_wgpu` entry points. Submit/readback timeouts are mapped to
`WgpuFracError::Backend`, shader/pipeline creation panics are caught as backend
errors, input/coeff/output finiteness is preflighted or validated, and the
minimal WGSL kernels are compiled under `cargo check --features wgpu-rt` instead
of leaving the old file dormant and unchecked.

Next steps:

1. Decide whether this revived wrapper should stay in st-core long term or be
   replaced by the newer `st-tensor` fractional WGPU surface.
2. Add runtime parity tests against CPU fractional ops when a WGPU adapter is
   available.
3. Include retry/fallback policy only when strict backend mode is disabled.

### P2: Harden numeric consensus paths against NaN gradients

Some lower-level consensus helpers used to rely on `partial_cmp(...).unwrap()`
in sorting/comparison paths. The compiled `st-core` backend paths now avoid the
obvious panic points: KV consensus weight medians use `total_cmp`, and temporal
spectral fusion sanitizes non-finite FFT magnitudes before dominant-frequency
selection. Trainer steps also now expose finite/non-finite gradient health and
tensor metadata sentinels in `TrainerStep.metrics.extra`, which makes early
NaN/Inf experiments diagnosable instead of silent. That is important once real
training emits NaNs or infinities during early experiments.

The Z-space coherence path now follows the same rule where it directly affects
learning feedback. Pre-discard channel ranking uses finite, non-negative scores
instead of `partial_cmp` fallback equality, geometric aggregation sanitizes
non-finite coherence weights before multiplying tensors, and PSI heatmap tongue
ranking ignores non-finite peak strengths. This keeps spectral phase labels,
local LR scales, and band feedback from being steered by accidental NaN channel
ordering. Those repairs are also surfaced through `CoherenceDiagnostics`,
`ZSpaceTrace`, and `TrainerStep.metrics.extra` so a real training run can tell
clean coherence from repaired coherence.
`summarize_trainer_trace_events()` now promotes those counters into a
`coherence_repairs` block with non-zero step counts and max/last totals, and the
char-LM sweep comparison will show repair columns when a run directory includes
a trainer trace or `trainer_trace_summary.json`.

The CPU rank-k software reference used by CUDA/HIP fallback paths now follows
that same finite-candidate contract. TopK, MidK, and BottomK selections filter
out `NaN`/`±inf` values before ordering with deterministic `total_cmp` plus
index tie-breaks, padding with `NaN/-1` only when there are not enough finite
candidates to fill the requested `k`. This keeps fallback/reference rank-k
outputs from reintroducing non-finite channels into Z-space bands after the
distributed TopK merge has already dropped non-finite shard values.

Next steps:

1. Extend NaN/Inf regression cases from software rank-k into strict GPU/runtime
   parity and consensus integration tests.
2. Decide whether non-finite gradient health should merely trace, warn, or
   actively trip a strict training guard.
3. Correlate spectral LR changes with sanitized versus clean coherence feedback
   across multi-run char-LM and graph-learning sweeps.

### P2: Bring model-zoo harnesses closer to real training workloads

The char-LM smoke harness is a good first proving ground, but the next learning
stack needs comparable runs across text, graph, and self-supervised objectives.
The self-supervised InfoNCE loss was still doing its core similarity matrix and
backward softmax as private Rust loops, which made it invisible to the trainer
backend policy and strict backend trace checks. `InfoNCELoss` now sends the
similarity matrix through `Tensor::matmul_with_backend(current_matmul_backend())`,
the positive/probability transposes through `transpose_with_backend()`, and the
backward probability row-normalisation through
`row_softmax_with_backend(current_softmax_backend())`, preserving the previous
normalisation/loss contract while exposing matmul, transpose, and softmax
metadata in `TrainerStep` traces. The backward path is now finite-difference
locked against the forward objective for both raw and L2-normalised embeddings,
so the normalisation Jacobian that maps anchor/positive gradients back to the
original prediction tensor is covered by tests instead of only by trace
metadata. A forced-WGPU regression also compares CPU/WGPU InfoNCE gradients and
requires both InfoNCE transposes to emit `tensor_util.transpose` on WGPU.

The training timeline controller is now traceable as a control-surface backend
decision too. `TimelineWarpController::apply` emits
`timeline_warp_controller_apply` metadata with the incoming learning signal,
pressure label, controller states, manual blend, applied scale/translation/
curvature deltas, and makespan before/after. That gives real LLM/graph/selfsup
sweeps a way to compare whether schedule warps came from loss spikes, memory
pressure, throughput drag, instability, or manual intervention instead of
leaving the temporal control loop as an unobserved Rust helper.
It also sanitizes config, manual overrides, and internal controller state before
each warp while finite-filtering incoming optional signals; metadata exposes
`signal_valid` and `state_sanitized`, so a single `NaN` loss/throughput/memory
observation cannot poison subsequent timeline warps.

The lower-level density scheduler is also visible now. `Scheduler::feed_density`
emits `scheduler_feed_density` metadata with raw and clamped activation,
gradient, and token-run densities, deterministic-lock status, the dominant drive
label, LR/tau before/after values, clamp bounds, and the decay/boost terms that
produced the update. This makes old adaptive LR and Z-temperature nudges
auditable inside real sweeps rather than treating them as opaque state changes.
It now sanitizes the control state before each feed as well: invalid public
LR/tau bounds are restored to finite ranges, finite out-of-range densities are
clamped, and non-finite density observations emit `density_valid=false` without
nudging LR or Z-temperature. That prevents a single `NaN` runtime density from
turning the scheduler into `NaN` for the rest of a training run.
`CurvatureScheduler` now has the same pressure-state guard: raw gradient
pressure is capped before the EMA and EMA² update, variance/relative variance
are computed in `f64`, and non-finite historical pressure state is ignored on
the next observation. Huge but finite gradients therefore no longer poison the
curvature scheduler with `inf` variance while longer trainer runs are adapting
hyperbolic curvature.

RealGrad engine pulses now join the same tensor-op metadata stream. Each
`RealGradEngine` projection emits `realgrad_projection_pulse` with sample,
spectrum, Z-space, and residual sizes; tempered convergence flags and iteration
counts; monad/Z energy split; residual and Lebesgue ratios; rolling gradient and
residual EMA state; spectrum normalisation; and transparent-gradient optics
summary when enabled. That gives training traces direct visibility into whether
RealGrad is acting as a balanced gradient projector, sparse-gradient detector,
residual-dominant warning, tempered instability signal, or optical-gradient
medium.

Legacy distributed optimizer hooks are no longer silent either. The one-bit
allreduce and ZeRO partition hook points now emit `onebit_allreduce_hook` and
`zero_partition_hook` metadata with hook registration state, world/rank, tensor
lengths, and L2 before/after summaries. That makes it safer to revive these
old extension points for real multi-rank learning because a trace can show
whether a hook was absent, active, shrinking gradients, or changing partition
shape.

The compiled distributed training helpers now expose their actual synchronization
work as well. `AmebaAutograd::propagate_round` emits `ameba_autograd_round`
metadata with processed messages, forwarded messages, tolerance absorptions,
max-hop stops, pending queue depth, aggregate signal, and update magnitude.
`DistributedTrainer` emits `distributed_trainer_sync_step`,
`distributed_trainer_async_enqueue`, and `distributed_trainer_async_merge`
metadata with worker/contribution counts, shard topology, gradient norms,
merge scale, and parameter-update magnitude. That turns the old in-memory
distributed trainer from a silent correctness helper into a traceable substrate
for future multi-rank graph, LLM, and self-supervised smoke runs.
Residual summaries classify these distributed trainer, Autograd, TopK, and
engine hook events as `cpu_control_*` rather than `cpu_debt_*`, so they remain
visible without being mistaken for routeable tensor kernels.

The older distributed TopK and lane-consensus pieces are now safer to reuse in
that path. `merge_two_shards_f32` no longer panics on NaN/Inf values during
TopK shard merge; it drops non-finite candidates, orders finite values with
`total_cmp`, and emits `distributed_topk_merge` metadata with dropped counts,
shape mismatch counts, retained L1-energy ratio, selected top value/index, and
output count. `consensus_lane_params` emits `distributed_lane_consensus`
metadata with input/output lane and whether Redis or HIP-real consensus changed
the result. It now clamps local, Redis, and HIP-real lane suggestions into the
safe `1..=4096` range before aggregation and records `input_lane_sanitized`,
`output_lane_sanitized`, and lane bounds in the same metadata. This exposes
gradient-compression health and runtime lane policy instead of leaving them as
silent, old backend helpers.

The Z-space optimiser path now exposes local and global learning-rate decisions.
`SpectralLrAdapter::scale_factor` emits `spectral_lr_scale` metadata with the
parameter name, sheet confidence, curvature/spin/energy features, smoothed
adapter state, individual scale terms, raw scale, and clamped scale. Global
optimizer scaling emits `zspace_optimizer_lr_scale`, and
`WarmupCosineScheduler::step` emits `warmup_cosine_lr_step`. These are labelled
`optimizer_control_cpu`/`host` and excluded from CPU-debt residuals: they explain
scalar learning-rate policy in real trainer traces instead of pretending to be
routeable tensor kernels or only showing the final parameter movement.
The adapter state update is now commit-safe too: non-finite spectral features,
invalid adapter configuration, or non-positive/non-finite raw scales return a
neutral `1.0` multiplier and emit `state_committed=false` without mutating the
running curvature/spin/energy EMAs.

Backend selection itself is now observable at the core runtime boundary.
`resolve_backend` emits `backend_resolution` metadata with requested, reported,
and effective backend labels plus the MPS placeholder/surrogate status when
applicable. `build_device_report` emits `backend_device_report` metadata with
lane width, subgroup support, workgroup alignment, occupancy score, preferred
TopK/compaction tiles, shared-memory availability, and MPS planner-route details.
This lets a training trace explain why a run used CPU, WGPU, CUDA, HIP, or an
MPS surrogate instead of treating backend choice as out-of-band planner state.

The temporal/spectral heuristic bridge is traceable too.
`TemporalSpectralFusion::analyse` emits `temporal_spectral_fusion` metadata with
lane-window bounds, problem dimensions, temporal span, harmonic count, fusion
grid size, dominant frequency, tempo hint, and spectral energy. This exposes the
FFT-derived lane-window signal that can influence backend heuristics, making it
possible to correlate runtime tile/merge decisions with the temporal pressure
seen during learning.

The environment KDSL bridge now reports whether heuristic overrides actually
entered the backend planner. `parse_env_dsl_plus_kind` and its explain variant
emit `kdsl_env_bridge` metadata with source presence/length, KDSL feature state,
evaluation status, problem dimensions, k-class, hard/soft override counts, and
explicit override fields. This closes an important silent path: default builds
can now show that `SPIRAL_HEUR_K` was present but ignored because the `kdsl`
feature was disabled, while KDSL builds can distinguish successful evaluation
from parse/evaluation errors.

The Redis/KV heuristic bridge now has the same audit trail. `choose_from_kv`
emits `kdsl_kv_bridge` metadata with KV feature state, Redis URL presence,
lookup status, log-bucketed key dimensions, and the selected heuristic fields
when a hit occurs. This keeps backend-policy experiments honest: a training run
can tell whether KV policy was unavailable, unconfigured, missed, errored, or
actually supplied the rank/TopK plan.

The soft consensus side of the same Redis/KV policy path is no longer silent
either. `kv_consensus_soft_rules` emits `kv_consensus_soft_rules` metadata with
logic/KV feature state, Redis URL presence, lookup status, list length, parsed
entry count, emitted soft-rule count, median weight, and log-bucketed key
dimensions. This matters for WGPU soft logic because an empty soft-rule vector
now explains whether consensus was disabled, unconfigured, missing, malformed,
or genuinely produced no usable rules.

The WGPU and Unison final-choice surfaces now close the loop. WGPU heuristic
finalization emits `wgpu_heuristic_choice` metadata with the chosen source,
score hint, problem dimensions, subgroup state, hard override fields, TopK/MidK
mode labels, and the final workgroup/lane/tile/FFT knobs. Unison rank selection
keeps its `unison_rank_choice` event but now includes candidate score slots for
baseline, environment KDSL, Redis/KV, and generated WGPU candidates. Together
these events let a trainer trace follow policy inputs through final rank-kernel
selection instead of only seeing the downstream tensor op that eventually ran.

The generated WGPU path now exposes its table lookup edge as well.
`WasmTunerTable::choose` emits `wasm_tuner_choice` metadata with hit/miss
status, workload dimensions, subgroup state, table size, matched record index,
and selected knob fields. That means a run where WGPU reports
`choice_source=generated` can now be audited back to whether the shipped tuner
table actually matched the workload or whether the generated route fell back
without table evidence.

Trainer-step traces now carry those backend-policy decisions through to the
Python sweep tooling. `TensorBackendStepTrace` aggregates policy event counts,
KDSL/KV status buckets, WGPU/Unison source buckets, and the latest WGPU/Unison
choice knobs into numeric `metrics.extra` fields, including
`backend_policy_wasm_tuner_events`. `summarize_trainer_trace_events` then
exposes a structured `backend_policy` block, and
`tools/compare_char_lm_runs.py` renders concise policy columns in comparison
tables, including `policy_tuner`. This makes real char-LM or graph-learning
sweeps able to compare loss movement against actual backend policy movement
instead of inspecting raw JSONL events by hand.
The char-LM comparison table now also exposes operation-level learning backend
columns (`matmul_*`, `prepacked_*`, `softmax_*`, `layer_norm_*`,
`gelu_bwd_*`, `batch_norm_bwd_*`, and `attention_*`). Existing WGPU finetune probes re-render with
`matmul_wgpu=50`, `prepacked_wgpu=125`, and `softmax_wgpu=5`, while
layer-norm and attention columns remain absent for that small architecture.
That gives future larger FT/LLM runs a direct checklist for whether the
activation, normalization, and attention stack is truly joining the WGPU path.

That comparison pass exposed a policy leak in the activation stack. `Gelu`
backward previously called `Tensor::gelu_backward()` directly, so a CPU trainer
policy in a WGPU feature build could still take the tensor-level auto WGPU path.
`Tensor::gelu_backward_with_backend()` now accepts `TensorUtilBackend`, and the
`Gelu` module routes backward through `current_tensor_util_backend_for_values()`.
CPU policy therefore forces `requested_backend=cpu`, while WGPU policy can still
use the fused WGPU derivative when the tensor is large enough to pass the
utility threshold. Focused `st-tensor` and `st-nn` GELU tests pass with and
without the `wgpu` feature.

A WGPU smoke sweep now confirms the value of that wiring. Running
`tools/run_char_lm_sweep.py` with `--cargo-features wgpu --backend wgpu`
against `models/samples/spiral_corpus_en` for two seeds produced stable backend
policy traces: each run saw `tensor_ops=885`, `tensor_wgpu=180`,
`tensor_wgpu_dense=5`, `tensor_cpu=705`, `policy_events=12`,
`policy_unison=3`, `policy_kdsl_env=3`, `policy_kv=3`, and
`policy_tuner=3`. The new status columns show why no generated rank candidate
was active: `policy_tuner_status=miss:3`, `policy_kdsl_env_status=empty:3`,
`policy_kv_status=feature_disabled:3`, and
`policy_unison_src=fallback:3`. In other words, this small char-LM workload is
not yet exercising the generated WGPU tuner table at all; the generated table
currently starts at larger row buckets, so `wgpu_heuristics_generated::choose`
returns no candidate and Unison ranks only the baseline fallback.

Adding a small generated tuner bucket for `rows<=255`, `cols<=4095`, `k<=128`,
and subgroup WGPU changes the same two-seed smoke run from table misses to table
hits: `policy_tuner_status` flips from `miss:3` to `hit:3`, and
`policy_candidates` increases from `1` to `2`. The generated candidate currently
ties the baseline score, so `policy_unison_src` remains `fallback:3`; that is
still useful because the generated table is now present in the candidate set,
and the next learning-oriented improvement can tune the small-shape bucket from
actual latency/loss evidence instead of merely repairing an observability gap.
The comparison tooling now also renders `policy_best_score`,
`policy_base_score`, and `policy_gen_score`; the same smoke run reports
`1.3950 / 1.3950 / 1.3950`, making the tie explicit rather than hiding it
behind the fallback source label. `unison_rank_choice` now also emits
`wgpu_generated_score_delta` and `wgpu_generated_ties_baseline`, and the compare
table renders `policy_gen_delta=0.0000` plus `policy_gen_tie=yes` for this
case. That turns the next tuning task into a measurable objective: make the
generated bucket produce a positive score delta under real latency or step-time
evidence instead of guessing from the fallback source label.
The sweep comparison also surfaces `trace_step_ms_last`,
`trace_step_ms_mean`, and `trace_step_ms_max`; in the two-seed WGPU smoke run
the one-step measurements were `798.297ms` and `716.019ms`. These are not yet
enough for a stable latency model, but they put the necessary timing evidence
next to tuner hit/miss, candidate count, backend counters, and validation loss.
That is the minimum table shape needed before doing a measured small-shape
tuner sweep.

The GNN side now has the same kind of runnable learning probe. The
`gnn_trainer_band_trace_demo` example writes a run directory with
`gnn_band_trace.json`, `trainer_trace.jsonl`, `run.json`, and `command.txt`,
and accepts `--backend`, `--run-dir`, `--events`, `--epochs`, graph count,
batch, node, feature, seed, learning-rate, and curvature controls. The JSON
payload uses schema `st.gnn.band_trace.v2` and includes run/device metadata,
training history with epoch tensor-backend counters, roundtable signal,
per-band replay coefficients, full layer flow reports, and a batched graph
readout error trace. This turns GNN training from a one-off demo into a compact
CPU/WGPU comparison harness.
Graph-level mini-batches now use `ZSpaceGraphBatchRegressor` rather than the
single-graph `ZSpaceGraphRegressor`: fixed-size graph samples are still
row-concatenated by `DataLoader::batched(N)`, but the model splits those rows
back into graph segments before pooling. The resulting prediction shape is
`(graph_count, target_cols)`, and the readout trace records `row_start`,
`row_end`, per-graph node L2, prediction L2, and per-graph MSE. A fresh CPU
smoke at `target/tmp/gnn_batch_readout_trace_smoke` used `batch=2`, `nodes=4`,
and `features=2`; the trainer trace showed `batch_input_rows=8` with
`batch_prediction_rows=2` and `batch_target_rows=2`, while
`gnn_band_trace.json` recorded `graph_count=2`, `total_rows=8`, and readout
segments `0..4` and `4..8`. This directly closes the earlier batched graph
readout risk where a row-concatenated mini-batch could be pooled as one graph.
`tools/run_gnn_band_trace_sweep.py` now carries that contract into comparison
runs. The sweep schema is `st.gnn.band_trace_sweep.v2`; run names and manifest
rows include `epochs`, train/validation graph counts, `batch`, `nodes`,
`features`, and `input_rows`, and the CLI accepts `--epoch-values`,
`--train-graph-values`, `--validation-graph-values`, `--batch-values`,
`--node-values`, and `--feature-values`. The generated `compare.md` now has a
`Group Averages` section that aggregates successful rows by those axes,
including average readout MSE, readout graph/row counts, band replay deltas,
step time, tensor backend counts, CPU debt, and backend-policy events. A 4-run
CPU smoke at `target/tmp/gnn_grid_sweep_smoke` (`seeds=9,10`,
`batch_values=1,2`, `nodes=4`, `features=2`) produced two clean groups:
`batch=1` had `avg_readout_graphs=1`, `avg_readout_rows=4`,
`avg_readout_mse=0.128179`, and `avg_cpu_debt_ops=108`, while `batch=2` had
`avg_readout_graphs=2`, `avg_readout_rows=8`, `avg_readout_mse=0.074536`, and
`avg_cpu_debt_ops=281`.

Two one-epoch smoke runs show the harness is wired through the full learning
stack. The CPU run emitted `tensor_ops_total=325`, `tensor_backend_cpu=225`,
`tensor_backend_wgpu=0`, `step_time_ms=16.503`, and readout
`mean_squared_error=0.032149`. The WGPU run emitted the same op total with
`tensor_backend_wgpu=100`, `tensor_backend_cpu=225`, `step_time_ms=362.814`,
and readout `mean_squared_error=0.187998`. The WGPU timing is dominated by
small-run overhead and should not be treated as a performance conclusion, but
it proves that graph-level learning can now be audited across readout quality,
roundtable band replay, and real backend routing in one artifact.

The probe now has a sweep wrapper as well. `tools/run_gnn_band_trace_sweep.py`
runs the GNN trace example across backends and seeds, writes one artifact
directory per run, and produces a compact `compare.md` with best/readout score,
per-band max scale delta, step time, tensor backend counters, and backend-policy
event counts. A smoke sweep over `cpu,wgpu` with seed `3` produced nearly equal
readout scores (`0.102818` CPU vs `0.103560` WGPU) while showing the expected
routing split: CPU saw `tensor_backend_cpu=225`, `tensor_backend_wgpu=-`, and
`policy_events=9`; WGPU saw `tensor_backend_cpu=225`,
`tensor_backend_wgpu=100`, and `policy_events=12`. This is the first reusable
graph-learning table where quality, band influence, and backend routing can be
compared side-by-side.

The self-supervised path now has equivalent coverage. The
`modelzoo_lightning_selfsup_minimal` example accepts run-dir, events, backend,
shape, seed, learning-rate, curvature, temperature, and normalization controls,
and writes `selfsup_trace.json`, `trainer_trace.jsonl`, `run.json`,
`weights.json`, and `command.txt`. Its trace schema
`st.selfsup.lightning_trace.v1` records InfoNCE first/last/delta, stage/epoch
telemetry, epoch tensor-backend stats, and the sanity reload output shape/norm.
Because `InfoNCELoss` already routes similarity logits through
`current_matmul_backend()`, positive/probability layout copies through the
tensor-util policy, and backward probabilities through `current_softmax_backend()`,
the trainer trace directly exposes whether the core contrastive objective used
CPU or WGPU for matmul, transpose, and row-softmax.

`tools/run_selfsup_sweep.py` lifts that example into a CPU/WGPU comparison
table. The current forced-WGPU selfsup probe
(`models/runs/current_backend_audit/selfsup_wgpu_transpose_routed`) reports
`tensor_ops=62`, `tensor_wgpu=62`, `tensor_cpu=0`, zero fallbacks,
`transpose_wgpu=2`, `matmul_scaled_wgpu=2`, `row_softmax_wgpu=1`, and
`cpu_debt_ops=0`. This gives the audit a second real learning objective,
separate from char-LM and GNN, where loss movement can be compared with backend
routing without a hidden layout-copy CPU tail.

The selfsup sweep also exposed and fixed a policy leak: in a WGPU feature build,
`BackendPolicy::from_device_caps(DeviceCaps::cpu())` previously left softmax,
layer norm, and attention on `Auto`, allowing `row_softmax_auto()` to select
WGPU even when the run requested CPU. CPU policy now pins those accelerated
ops to explicit CPU backends, while WGPU policy continues to request WGPU. The
targeted policy test passes in both normal and `--features wgpu` builds, and the
post-fix selfsup sweep confirms `tensor_policy_softmax_cpu` for CPU runs and
`tensor_policy_softmax_wgpu` for WGPU runs.

The selfsup probe now also has a local accumulator-sync mode. Passing
`--accumulator-sync local` installs a single-rank `AccumulatorSynchronizer`
through `ModuleTrainer::set_training_device()`, so the same example can prove
that optimizer accumulator statistics reach the normal `TrainerStep` trace.
`tools/run_selfsup_sweep.py` forwards this knob and renders
`sync_enabled`, `sync_world`, `sync_buffers`, and `sync_values` beside the
backend counters. A seed-7 CPU/WGPU smoke run with two batches reported
`sync_enabled=1`, `sync_world=1`, `sync_buffers=4`, and `sync_values=44` on
both backends while preserving the expected routing split: CPU stayed on
`softmax_cpu=1`, `matmul_naive=7`, and `tensor_wgpu=-`; WGPU used
`softmax_wgpu=1`, `matmul_wgpu=7`, and `tensor_wgpu=22`. The
`spiral-selfsup` two-rank accumulator test also passes, covering the averaged
distributed path separately from the single-rank trace artifact.

The multi-epoch selfsup sweep then exposed a concrete WGPU learning correctness
bug rather than only an observability gap. The first version of the table only
showed `first_info_nce` and `last_info_nce`, which hid whether the backends
started from the same loss. The probe now records a pre-training CPU reference,
selected-backend pre-training loss, total backend gap, and a split between
encoder-forward gap and InfoNCE-loss gap. Before the tensor fix, the WGPU run
showed `pretrain_backend_gap_mean=0.699070` across seeds 11, 12, and 13, and
the split isolated all of it to encoder forward (`loss_gap_mean=0.000000`).
That pointed directly at the WGPU prepacked matmul route used by `Linear`,
GNN, RNN, and convolutional forward paths.

The underlying fix had two parts. First, WGPU prepacked RHS upload now matches
the generated shader contract: F32 buffers are uploaded in the row-major
`rhs_packed[k * cols + col]` order used by the shader, while quantized buffers
use the shader's per-column packed layout. Second, the dense matmul WGSL no
longer returns early for out-of-bounds edge-tile invocations before
`workgroupBarrier()`; all invocations participate in the tile barriers and only
the final output write is bounds-guarded. The post-fix
`selfsup_sweep_pretrain_gap_split_barrier_fixed_smoke` run reduces WGPU
`pretrain_backend_gap` to a max absolute error of `0.000000119`, while
preserving WGPU routing (`matmul_wgpu=7`, `softmax_wgpu=1`,
`tensor_wgpu=22`). This is a direct learning-stack improvement because the
encoder forward baseline is now backend-parity-clean before any optimizer or
roundtable effects are interpreted.

The same run also gives the first useful small self-supervised loss movement
table. CPU and WGPU now start from matching pre-training losses. All six runs
improve at least once (`best_epoch=2` for CPU and mostly WGPU), but both
backends can rebound by the final epoch: CPU seed 11 ends `0.532394` above its
best, while WGPU seed 11 ends `0.537682` above its best. That makes the next
learning-quality target concrete: stabilize the self-supervised schedule
around the best epoch rather than interpreting final-epoch loss alone.

The fix is now guarded at both levels that matter for learning. At the tensor
level, `matmul_prepacked_forced_wgpu_matches_cpu_reference_on_edge_tiles`
forces the WGPU prepacked path on the same `(12 x 8) @ (8 x 6)` edge-tile shape
that reproduced the selfsup gap and compares it against the CPU packed
reference. At the learning layer level,
`linear_forward_forced_wgpu_prepacked_matches_cpu_reference_on_edge_tiles`
installs explicit backend policies around `Linear::forward()` and proves that
the policy-routed WGPU prepacked path matches the CPU reference. Both tests
skip only when WGPU is unavailable and run successfully on the current local
machine. A follow-up seed-21 selfsup smoke keeps the artifact-level invariant:
WGPU reports `pretrain_backend_gap=0.000000000` while still routing through
`matmul_wgpu=5`, `softmax_wgpu=1`, and `sync_enabled=1`.

The same guard pattern now covers the next learning layers that consume
prepacked matmul directly. `graph_convolution_forced_wgpu_prepacked_matches_cpu_reference_on_edge_tiles`
compares a 12-node GNN convolution under explicit CPU and WGPU policies,
covering both graph propagation and the learnable prepacked kernel.
`wave_rnn_forced_wgpu_prepacked_matches_cpu_reference_on_edge_tiles` protects
the `WaveRnn` readout path, and
`spiral_rnn_forced_wgpu_prepacked_matches_cpu_reference_on_edge_tiles` protects
the input/state/phase projections in `SpiralRnn`. The combined
`forced_wgpu_prepacked` test filter runs four parity tests successfully on the
current WGPU machine. A follow-up GNN smoke sweep with seed 8 preserves the
artifact-level invariant as well: CPU and WGPU both report
`best_score=0.129244` and `readout_mse=0.129244`, while WGPU still routes
through `tensor_wgpu=100` and `policy_events=12`.

The fused dense and convolutional routes are now part of the same correctness
net. `matmul_bias_fused_forced_wgpu_matches_cpu_reference_on_edge_tiles`
forces WGPU for the ReLU, GELU, residual+ReLU, and residual+GELU fused dense
paths on non-tile-aligned `(13 x 9) @ (9 x 7)` shapes and compares them against
CPU naive. The fused im2col shader now follows the dense shader's edge-tile
contract: out-of-bounds invocations stay alive through workgroup barriers and
only guard the final write.

That Conv2d probe found a real learning-path bug. The WGPU fused im2col path
accepted a `weight_t` buffer and an optional bias, but the shader read weights
as if they were out-major and never bound the bias at all. The fix wires a bias
storage buffer into the fused conv bind group and makes the shader read
`weights[k * out_channels + col]`, matching the transposed upload. Conv2d
forward now also converts the intermediate `(spatial, out_channels)` GEMM
matrix into the channel-major feature layout expected by pooling and by
`grad_output_to_matrix()`. The guard
`conv2d_wgpu_fused_im2col_and_grad_input_match_cpu_on_edge_tiles` now compares
CPU and WGPU forward plus fused grad-input on an edge-tile multi-channel shape,
and `conv2d_forward_uses_channel_major_layout` fixes the public layout
contract.

Conv1d forward has joined the same backend policy path. It now lowers through
im2col plus the prepacked matmul backend instead of staying on a scalar-only
loop, while preserving the empty-batch behavior. The guard
`conv1d_forward_forced_wgpu_im2col_matches_cpu_reference_on_edge_tiles` proves
that the policy-routed WGPU path matches a CPU-naive reference on a
non-trivial multi-channel shape. This makes sequence/convolutional probes more
honest: Conv1d can now exercise the same WGPU-first tensor stack as Linear,
GNN, RNN, and Conv2d.

The Conv2d/MaxPool2d model-zoo example has now been promoted from a CPU-only
demo into a backend comparison probe. It accepts `--backend`, `--run-dir`, and
`--events`, writes `vision_trace.json`, `run.json`, `weights.json`, and a
trainer-event JSONL file, and records pre-training CPU reference loss,
selected-backend loss, forward-only gap, loss-only gap, epoch losses, and
epoch tensor backend counters. `tools/run_vision_conv_sweep.py` compares those
runs in the same style as the self-supervised sweep. A seed-31 CPU/WGPU smoke
run reports exact pre-training loss parity (`pretrain_backend_gap=0.000000`)
and identical one-epoch loss (`2.675229`) while proving the WGPU route is
actually active: WGPU records `tensor_wgpu=13`, `prepacked_wgpu=7`, and
`matmul_wgpu=6`, whereas CPU records `prepacked_naive=7` and
`matmul_naive=3`. This turns the Conv2d layout and shader fixes into an
artifact-level learning invariant rather than only unit-test coverage.
The pooling half of that vision probe is now trace-visible too. `MaxPool2d`
and `AvgPool2d` emit forward/backward metadata with batch/channel dimensions,
input/output spatial shapes, kernel/stride/padding, window size, and estimated
window or scatter-add work. `MaxPool2d` now has WGPU forward/backward kernels:
forward returns both pooled values and argmax indices, while backward
recomputes the argmax relation per input cell to avoid overlapping-window
scatter races. `AvgPool2d` has the same WGPU route shape with fixed-area
padding semantics matching the CPU path; its backward kernel gathers all
covering output gradients per input cell, avoiding atomic scatter races for
overlapping windows. Comparisons expose `max_pool_fwd_wgpu` /
`max_pool_bwd_wgpu` and `avg_pool_fwd_wgpu` / `avg_pool_bwd_wgpu` beside the
CPU columns, so future vision runs can distinguish real pooling kernels from
deliberate small-shape CPU threshold choices.
The loss side of the same probe is no longer part of that island:
`HyperbolicCrossEntropy` now has direct WGPU forward/backward tensor-utility
kernels for the stable softplus/sigmoid binary objective with curvature scaling
and epsilon-clamped labels. In the forced WGPU smoke
`target/tmp/vision_conv_maxpool_wgpu_smoke`, the loss reports
`hyperbolic_cross_entropy_forward_wgpu=1` and
`hyperbolic_cross_entropy_backward_wgpu=1`, pooling reports
`max_pool2d_forward_wgpu=3` and `max_pool2d_backward_wgpu=2`, and the run
reaches `tensor_ops=46`, `tensor_wgpu=46`, no CPU backend events, zero
fallbacks, and `cpu_debt_ops=0`.

The WaveRnn sequence model-zoo example now has the same probe surface. It
accepts backend/run-dir/event knobs, writes `sequence_trace.json`, `run.json`,
`weights.json`, and trainer events, and records pre-training CPU reference,
selected-backend loss, forward-only gap, loss-only gap, epoch loss, and routing
counters. `tools/run_wave_rnn_sweep.py` produces the comparison table. A
seed-41 CPU/WGPU smoke run reports exact pre-training parity
(`pretrain_backend_gap=0.000000`) and identical one-epoch loss (`0.272757`).
The WGPU run proves the new Conv1d im2col path and WaveRnn readout are active:
`tensor_wgpu=24`, `prepacked_wgpu=18`, and `matmul_wgpu=6`, while CPU records
`prepacked_naive=14` and `matmul_naive=6`. The same artifact exposes the next
low-level learning bottleneck: WaveGate projection, reshape, bias adds,
reductions, and Poincare projection still account for `tensor_cpu=65` in the
WGPU run, so the sequence stack is backend-correct but still CPU-heavy around
geometry/projection utilities.

The sequence sweep now breaks those CPU-heavy utilities out explicitly. A
three-seed CPU/WGPU smoke (`models/runs/wave_rnn_sweep_cpu_heavy_smoke`) keeps
the correctness invariant intact across seeds 41, 42, and 43:
`avg_backend_gap=0.000000` and `avg_last_loss=0.193317` for both backends. It
also shows why this path is not yet a real accelerator path: the WGPU average
records `avg_tensor_wgpu=24`, `avg_tensor_cpu=65`, and
`avg_cpu_heavy_ops=57`, so `reshape`, `add_row_inplace`, `sum_axis0`,
`project_to_poincare`, `scale`, and `transpose` together represent about 64%
of the tensor-operation trace (`avg_cpu_heavy_share=0.640`). The one-batch
timings are small-run diagnostics rather than production throughput numbers,
but they are directionally useful: CPU averages `7.132ms` while WGPU averages
`125.897ms`, meaning dispatch/init overhead plus CPU utility round-trips swamp
the already-correct WGPU matmul route.

The implementation priority is therefore not just "GPU everything." `reshape`
is already metadata-like for RowMajor tensors and should first be made trace-
honest or copy-avoiding for non-RowMajor inputs. `transpose` is a real layout
copy and can be attacked by eliminating unnecessary transposes or adding a
layout-aware kernel path. The highest-value kernels for sequence learning are
the scalar/broadcast/reduction geometry utilities that sit directly on the
WaveGate and loss path: `add_row_inplace`, `scale`, `sum_axis0`, and
`project_to_poincare`, followed by a fused WaveGate forward/backward that can
combine gate multiply, bias, projection, and gradient reductions.

The first utility-kernel pass now exists behind an explicit tensor utility
backend. `st-tensor` exposes WGPU kernels for `scale`, row-bias
`add_row_inplace`, row-scale `mul_row`, row-affine `row_affine`, `sum_axis0`,
and `project_to_poincare`; `st-nn` carries that choice through
`BackendPolicy`, so CPU runs keep the legacy CPU utility path while WGPU runs
can opt into the new kernels. WaveRnn readout/backward, SpiralRnn recurrent
bias/reduction/scaling utilities, and WaveGate projection are wired to that
policy. The guard
`wgpu_tensor_utils_match_cpu_reference_on_sequence_shapes` compares the direct
WGPU utility kernels with CPU references, and
`tensor_utility_methods_emit_wgpu_backend_when_available` verifies the explicit
Tensor method trace.

A fresh three-seed sweep
(`models/runs/wave_rnn_sweep_wgpu_utils_smoke`) keeps the same learning
invariant: CPU and WGPU both report `avg_backend_gap=0.000000` and
`avg_last_loss=0.193317`. It also proves the routing change is real without
polluting the baseline: CPU has `avg_wgpu_utility_ops=0`, while WGPU has
`avg_wgpu_utility_ops=17` (`add_row_wgpu=4`, `sum_axis0_wgpu=3`,
`poincare_wgpu=4`, `scale_wgpu=6` per run). WGPU `avg_tensor_wgpu` rises from
24 to 41 and `avg_tensor_cpu` drops from 65 to 48, bringing
`avg_cpu_heavy_share` down from `0.640` to `0.449`. The caveat is equally
important: one-batch WGPU step time rises to `228.596ms` versus CPU
`5.286ms`, so correctness and routing improved, but small-shape dispatch and
readback overhead now dominate. The next accelerator step should therefore
fuse utilities at the learning-layer boundary rather than adding many more
single-op dispatches.

The first learning-layer fusion is now in place for the WaveRnn readout
gradient path. `tensor_utils.wgsl` adds `sum_axis0_scaled`, and WaveRnn uses it
only under the WGPU tensor-utility policy; CPU runs still emit the legacy
`sum_axis0 + scale` trace. A fresh three-seed fused smoke
(`models/runs/wave_rnn_sweep_wgpu_utils_fused_smoke`) preserves parity
(`avg_backend_gap=0.000000`, `avg_last_loss=0.193317`) and shows the fused
route explicitly: WGPU records `sum_axis0_wgpu=0`,
`sum_axis0_scaled_wgpu=3`, and `scale_wgpu=3`, reducing total tensor ops from
89 to 86 and WGPU utility ops from 17 to 14. Absolute CPU-heavy ops remain 40,
so `avg_cpu_heavy_share` reads `0.465` only because the denominator shrank.
This confirms the right pattern: fusing adjacent utility operations reduces
dispatch count without changing learning numerics, but the remaining bottleneck
is still the unfused WaveGate/frontier shape work (`reshape`, `transpose`,
residual CPU row-adds, and CPU scales). A one-seed three-batch smoke
(`models/runs/wave_rnn_sweep_wgpu_utils_fused_b3_smoke`) keeps parity and
lands at `165.279ms` on the last WGPU step, suggesting compile/first-dispatch
noise dominates the worst one-batch sample.

WaveGate forward has now joined that fused path. `tensor_utils.wgsl` adds
`wave_gate_project`, which performs row affine gating, OpenCartesianTopos
porous saturation, and Poincare projection in one WGPU dispatch. WaveGate uses
it only under the WGPU tensor-utility policy, preserving the CPU baseline and
the intermediate rewrite semantics by carrying `saturation` and `porosity`
into the shader. The direct tensor guard now covers this kernel, and
`wave_gate_forced_wgpu_forward_matches_cpu_reference` locks the layer-level
CPU/WGPU contract.

The new three-seed WaveGate-fused sweep
(`models/runs/wave_rnn_sweep_wave_gate_fused_smoke`) keeps
`avg_backend_gap=0.000000` and `avg_last_loss=0.193317`. Routing also looks
right: CPU records `wave_gate_project_wgpu=0`, while WGPU records
`wave_gate_project_wgpu=4`, `poincare_wgpu=0`, and the same fused
`sum_axis0_scaled_wgpu=3`. The average WGPU last-step time improves from the
previous fused smoke's `267.466ms` to `185.493ms`, and a one-seed three-batch
warm smoke lands at `183.545ms`. Absolute CPU-heavy trace volume remains at
40, because the removed WaveGate scalar loop was previously invisible to the
tensor-op counter. The next bottleneck is therefore no longer the WaveGate
projection dispatch itself, but the visible CPU utility tail: `reshape=14`,
`add_row_cpu=8`, `scale_cpu=9`, `sum_axis0_cpu=3`, and `transpose=6` in the
WGPU run.

That tail was routed through `TensorUtilBackend` for `Linear`, GNN, and Conv
row-bias/reduction/scale epilogues, but the first unrestricted smoke was a
useful warning: `models/runs/wave_rnn_sweep_utility_tail_routed_smoke` cut
WGPU CPU-heavy ops from 40 to 20 and raised WGPU utility ops from 14 to 34, yet
the average last-step time regressed to `342.108ms`. The functional direction
was correct, but the small tensor dispatch count dominated.

The current policy therefore keeps literal WGPU-first routing while adding a
size guard: `current_tensor_util_backend_for_values()` only dispatches WGPU
utility kernels above `SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES` (default
`1024`). The guarded smoke
(`models/runs/wave_rnn_sweep_utility_tail_threshold_smoke`) preserves parity
(`avg_backend_gap=0.000000`, `avg_last_loss=0.193317`) and returns the small
WaveRNN shape to the fused WaveGate profile: WGPU records `avg_tensor_wgpu=38`,
`avg_tensor_cpu=48`, `avg_cpu_heavy_ops=40`, `avg_wgpu_utility_ops=14`, and
`avg_step_ms_last=175.731ms`. Larger shapes can lower or clear the threshold to
explore the full utility-tail route without changing layer code.

The first learning-boundary tail fusion now targets the most common remaining
pattern directly: prepacked matmul followed by row bias. `matmul_prepacked_bias`
reuses the dense WGPU shader's `FLAG_USE_BIAS` path, preserving prepacked
weight reuse while eliminating separate `add_row_inplace` tensor utility ops.
`Linear`, GNN, Conv1d im2col forward, WaveRNN readout, and the older
`LinearModel` forward/train path now route through this fused API. The
three-seed smoke
(`models/runs/wave_rnn_sweep_prepacked_bias_fused_smoke`) preserves parity
(`avg_backend_gap=0.000000`, `avg_last_loss=0.193317`) and confirms the trace
shape: WGPU records `matmul_prepacked_bias_wgpu=12`, `add_row_wgpu=0`, and
`add_row_cpu=0`. Relative to the guarded tail smoke, WGPU tensor ops drop from
86 to 74, CPU-heavy ops drop from 40 to 32, WGPU utility ops drop from 14 to
10, and average last-step time moves from `175.731ms` to `170.713ms`. The
remaining visible WGPU-run tail is now `reshape_cpu=14`, `scale_cpu=9`,
`transpose_cpu=6`, and `sum_axis0_cpu=3`.

The next gradient-side fusion removes another small but frequent learning tail:
matmul immediately followed by batch scaling. The dense WGPU matmul uniform now
carries an `output_scale`, and `Tensor::matmul_scaled_with_backend()` emits a
single `matmul_scaled` op for CPU/Auto/WGPU. WaveRNN readout gradients,
Linear/GNN weight gradients, Conv2d gradient weights, and the older
`LinearModel::train_batch()` path now use it instead of `matmul + scale`. The
three-seed smoke
(`models/runs/wave_rnn_sweep_matmul_scaled_fused_smoke`) keeps parity
(`avg_backend_gap=0.000000`, `avg_last_loss=0.193317`) while reducing WGPU
tensor ops from 74 to 68, CPU-heavy ops from 32 to 29, and WGPU utility ops
from 10 to 7 compared with the prepacked-bias smoke. Average WGPU last-step
time moves from `170.713ms` to `135.643ms`; CPU also benefits in trace volume,
dropping from 77 to 71 tensor ops and from 45 to 39 CPU-heavy ops. The
remaining visible WGPU-run tail is now `reshape_cpu=14`, `transpose_cpu=6`,
`scale_cpu=6`, and `sum_axis0_cpu=3`.

Bias-gradient reductions now use the same fused reduction contract on CPU and
WGPU. WaveRNN, Linear, GNN, and Conv2d route bias gradients through
`sum_axis0_scaled_with_backend()` regardless of whether the threshold selects
CPU or WGPU; Conv1d's hand-rolled bias accumulator scales its Vec before
creating a Tensor, avoiding a standalone tensor `scale`. The
`models/runs/wave_rnn_sweep_bias_reduce_fused_smoke` run keeps parity
(`avg_backend_gap=0.000000`, `avg_last_loss=0.193317`) and removes the visible
`sum_axis0_cpu` and `scale_cpu` tail from the WGPU run. Relative to the
matmul-scaled smoke, WGPU tensor ops drop from 68 to 62 and CPU-heavy ops from
29 to 23; CPU drops from 71 to 62 tensor ops and from 39 to 30 CPU-heavy ops.
The short WGPU last-step average is noisier here (`135.643ms` to `148.135ms`),
so the routing result should be treated as trace/dispatch evidence rather than
latency proof. The remaining visible WGPU-run tail is now `reshape_cpu=14`,
`transpose_cpu=6`, and `sum_axis0_scaled_cpu=3`.

SpiralRnn now joins the same tensor-utility policy surface instead of keeping
its recurrent utilities as unconditional CPU/Auto calls. Forward row-bias adds
use `add_row_inplace_with_backend()`, backward bias reductions use
`sum_axis0_with_backend()`, and the final gradient averaging plus injected gate
negation use `scale_with_backend()`. The existing focused SpiralRNN tests pass
with and without `--features wgpu`, preserving the prepacked-matmul WGPU parity
test while making larger recurrent shapes eligible for threshold-protected WGPU
utility routing.

Weight-gradient matmuls no longer need to materialize a transposed input.
`dense_matmul.wgsl` now supports `FLAG_LHS_TRANSPOSE`, exposed as
`Tensor::matmul_lhs_transpose_scaled_with_backend()`. WaveRNN readout
gradients, Linear/GNN weight gradients, Conv2d gradient weights, and
`LinearModel::train_batch()` now compute `lhs.T @ rhs * scale` directly. The
`models/runs/wave_rnn_sweep_lhs_transpose_fused_smoke` run keeps parity
(`avg_backend_gap=0.000000`, `avg_last_loss=0.193317`) and confirms
`matmul_lhs_transpose_scaled_wgpu=6` with `transpose_cpu=0`. Relative to the
bias-reduce smoke, WGPU tensor ops drop from 62 to 56, CPU-heavy ops from 23
to 17, and average WGPU last-step time improves from `148.135ms` to
`127.595ms`. CPU trace volume also drops from 62 to 56 tensor ops. The
remaining visible WGPU-run tail is now almost entirely shape/reduction
bookkeeping: `reshape_cpu=14` and `sum_axis0_scaled_cpu=3`.

WaveRNN now removes two pure shape hops from the training step by taking the
final timestep directly from the flattened `WaveGate` output and by writing
readout gradients directly into `(batch * out_steps, hidden_dim)` gate-gradient
layout. The `models/runs/wave_rnn_sweep_wave_rnn_shape_direct_smoke` run keeps
parity (`avg_backend_gap=0.000000`, `avg_last_loss=0.193317`) while reducing
WGPU tensor ops from 56 to 49, CPU-heavy ops from 17 to 10, and
`reshape_cpu` from 14 to 7 compared with the lhs-transpose smoke. The remaining
WGPU-run CPU-heavy tail is now `reshape_cpu=7` and `sum_axis0_scaled_cpu=3`.
Average WGPU last-step time in this short smoke regresses/noises from
`127.595ms` to `140.984ms`, so this remains structural evidence rather than a
latency claim until a longer run separates dispatch noise from per-step cost.
The same direct layout now passes a `1 / batch` parameter-gradient scale into
the internal `WaveGate`: readout still only uses the final timestep, so gate and
bias updates no longer shrink by `out_steps` while the gate input gradient
remains the ordinary unaveraged chain-rule signal.

Zero-copy reshapes are now classified as metadata views instead of CPU work.
`Tensor::reshape()` still emits `reshape` metadata, but row-major reshapes now
report `backend=view`, `kernel=metadata`, and `zero_copy=true`; only layout
conversion reshapes report `backend=cpu`. The
`models/runs/wave_rnn_sweep_reshape_view_classified_smoke` run keeps parity
(`avg_backend_gap=0.000000`, `avg_last_loss=0.193317`) and shows
`reshape_cpu=0`, `avg_reshape_view=7`, WGPU tensor ops unchanged at 49, WGPU
CPU backend ops dropping from 18 to 11, and CPU-heavy ops dropping from 10 to
3 compared with the shape-direct smoke. The remaining WGPU-run CPU-heavy tail
is now only `sum_axis0_scaled_cpu=3`. The short smoke reports
`avg_step_ms_last=108.621ms`, but the classification change should be treated
as trace honesty rather than a kernel-speed claim.

WaveRNN readout-bias reductions now use the same tensor-utility size guard as
Linear, GNN, and Conv2d. With the default
`SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES=1024`, the small smoke
(`models/runs/wave_rnn_sweep_tensor_util_threshold_default_smoke`) keeps parity
and routes both `(4, 8)` and `(4, 1)` `sum_axis0_scaled` reductions to CPU:
WGPU utility ops drop from 7 to 4, CPU-heavy ops become 6, and average WGPU
last-step time is `124.401ms`. Forcing the threshold to 1 in
`models/runs/wave_rnn_sweep_tensor_util_threshold_forced_smoke` moves both
reductions to WGPU (`sum_axis0_scaled_wgpu=6`, `sum_axis0_scaled_cpu=0`) and
also moves small Conv1d gradient scaling to WGPU (`scale_wgpu=3`), but the
short smoke is slower (`148.548ms`), which supports keeping tiny utility work
on CPU by default. A larger default-threshold smoke
(`models/runs/wave_rnn_sweep_tensor_util_threshold_large_default_smoke`) shows
the intended scale transition: `(64, 16)=1024` WaveRNN readout-bias reduction
uses WGPU while the `(64, 1)=64` Linear head bias stays CPU, with parity intact.
`tools/run_wave_rnn_threshold_grid.py` now turns this into a reusable artifact:
the smoke grid at `models/runs/wave_rnn_tensor_util_threshold_grid_smoke`
compares thresholds 1 and 1024 across `(batch, hidden) = (4,8), (4,16),
(64,8), (64,16)`. It records the expected WaveRNN/Linear bias route beside the
actual `sum_axis0_scaled_{wgpu,cpu}` counts, step time, backend gap, and utility
totals. The boundary row (`threshold=1024`, `batch=64`, `hidden=16`) shows
`wave_bias_values=1024`, `wave_bias_route=wgpu`, `linear_bias_route=cpu-threshold`,
`sum_axis0_scaled_wgpu=3`, and `sum_axis0_scaled_cpu=3`, confirming that the
threshold is acting on tensor size rather than backend labels alone.

The same threshold-grid pattern now exists for GNN learning. `tools/run_gnn_threshold_grid.py`
drives `gnn_trainer_band_trace_demo` with
`SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES` overrides and records graph rows,
output/hidden bias values, predicted routes, actual `sum_axis0_scaled` routing,
band-pass deltas, readout MSE, graph readout CPU tails, and WGPU matmul counts.
The grid manifest is now `st.gnn.tensor_util_threshold_grid.v2`: it supports
`--dry-run`, records run names, commands, input/output/hidden value counts,
return codes, failure kinds/details, and `failed=true` without dropping the
rest of the comparison. `compare.md` also includes `run_status`, return code,
failure columns, probe-batch readout graph/row counts, validation-wide readout
MSE/graph/row counts, and group-average readout graph/row counts so batch-size
probes can prove whether the validation readout actually contains multiple
graph segments.
The v2 CPU smoke at `target/tmp/gnn_threshold_grid_v2_smoke2` compared
`thresholds=1,1024`, `batches=1,2`, `nodes=4`, and `features=2`; `batch=2`
rows recorded `readout_graphs=2`, `readout_rows=8`, and
`cpu_debt_ops=281`, while `batch=1` rows recorded `readout_graphs=1`,
`readout_rows=4`, and `cpu_debt_ops=108`. A tiny WGPU preflight probe at
`target/tmp/gnn_threshold_grid_wgpu_preflight_probe` currently records the
shared preflight failure as `failure_kind=signal` /
`failure_detail=signal:6:empty_log`, then preserves each planned grid row as a
`preflight_signal` failure with its original threshold/shape axes. That keeps
the comparison artifact readable even when the local adapter/runtime aborts
before the actual row can run.
The band-trace learning sweep at
`target/tmp/gnn_band_trace_validation_readout_sweep` confirms why the
validation-wide metric matters: probe-batch `readout_mse` varies with the first
validation mini-batch, while `validation_readout_mse` matches the trainer
`best_score` across `batch=1` and `batch=2` rows and records all four
validation graphs (`validation_readout_rows=16` for `nodes=4`).
The readout error trace now also records target mean-square energy and
`normalized_mean_squared_error`; sweep compare tables expose this as
`validation_readout_nmse` / `avg_validation_readout_nmse`, making it easier to
tell whether a fresh-seed regression is a real schedule loss or a harder target
draw.
`tools/run_gnn_band_trace_sweep.py` now also grids learning rate and roundtable
schedule axes (`--lr-values`, `--top-k-values`, `--mid-k-values`,
`--bottom-k-values`, and `--here-tolerance-values`) and includes those axes in
run names, manifests, group averages, `Top Validation Candidates`, and
`Roundtable Axis Deltas`. `RankPlan.choice` already participates in roundtable
classification and spectral feature extraction through `RoundtableChoiceProfile`
inside `zspace_round`, and the example run metadata preserves the full
roundtable backend audit/choice summary. The CPU smoke at
`target/tmp/gnn_roundtable_axis_sweep` compares `lr=0.03,0.05` and
`top_k=1,2`; the `top_k=2` rows slightly lower validation-wide MSE in that
tiny probe while reducing the Here replay delta by about `0.05`, making the
report a useful harness for tuning `RoundtableBandInfluence` before changing
the coefficient formula.
The wider smoke at `target/tmp/gnn_roundtable_axis_delta_sweep` confirms that
the same delta table now covers `top_k`, `mid_k`, `bottom_k`, and
`here_tolerance`, and missing band-delta comparisons render as `-` rather than
`inf`/`nan`. The sweep manifest now also writes a machine-readable
`comparison` payload (`st.gnn.band_trace_compare.v1`) with
`top_validation_candidates` and `roundtable_axis_deltas`, making the current
best validation-wide GNN schedule available to follow-up jobs without scraping
Markdown. Multi-seed GNN compare records now also include validation-MSE
stddev/min/max/spread plus a `validation_stability_score`, and the Markdown
report adds `Stable Validation Candidates` mirrored in
`comparison.stable_validation_candidates`; this ranks candidates by average
validation MSE plus stddev so repeatable schedules can be separated from
low-average but volatile seed wins. `--follow-up-from <previous-sweep>` now consumes that payload and
replays a ranked top candidate as the next grid default while preserving
explicit overrides such as fresh seeds or wider roundtable axes, closing the
loop from trace comparison into the next learning probe. The follow-up path can
also add `--follow-up-neighborhood` plus a selected axis list such as
`lr,top_k,bottom_k`, which fans out a compact local schedule search around the
candidate while leaving explicitly supplied `--*-values` untouched. Follow-up
reports now emit `Follow-Up Result` and `comparison.follow_up_result`, comparing
the new best candidate against the source candidate with validation-MSE, CPU
debt, and step-time deltas plus an `improved`/`matched`/`regressed` verdict.
`--follow-up-fail-on-verdict regressed,unknown` additionally writes
`Follow-Up Gate` / `comparison.follow_up_gate` and returns non-zero when the
requested verdict is hit, making the follow-up loop usable as a CI or local
promotion gate. The same compare artifact now emits `Follow-Up Promotion` /
`comparison.follow_up_promotion`, promoting the new best candidate only on an
`improved` verdict and otherwise keeping the source candidate as the next safe
seed. `--follow-up-source auto` now consumes that promotion record by default
when a later `--follow-up-from` points at the sweep, while
`--follow-up-source top-candidate` preserves the raw top-validation replay path
for exploratory overrides. The compare artifact also emits `Next Follow-Up
Command` / `comparison.follow_up_next_command`, giving chained sweeps a
machine-readable command template with the current run root already wired as
the next `--follow-up-from` source. Follow-up manifests also persist
`config.follow_up.lineage` with the parent sweep path, parent run root, selected
candidate source, and incremented generation, so multi-step GNN tuning chains can
be audited without reconstructing ancestry from shell history. Completed
follow-up comparisons now mirror that ancestry as `comparison.follow_up_chain`
plus Markdown `Follow-Up Chain` / `Follow-Up Ancestors` tables; missing or pruned
parent artifacts are recorded as ancestor rows instead of failing the new compare
report. `comparison.follow_up_chain_guidance` and the `Follow-Up Chain Guidance`
table summarize recent verdict streaks and suggest whether the next run should
continue the promotion path, rerun with fresh seeds, keep the source, or widen
the local neighborhood. `comparison.follow_up_guided_next_command` and `Guided
Next Follow-Up Command` turn that guidance into a command template, listing
placeholders such as `NEXT_RUN_ROOT` and `NEW_SEEDS` while wiring the current
sweep as `--follow-up-from`. Guidance now carries the selected candidate's
validation stability status/score/spread, so an `improved` but
`single_seed_probe` or `volatile` candidate is rerouted to fresh-seed
confirmation before the promotion path continues. If repeated improvements stay
`volatile`, guidance switches to `widen_stability_search` and adds
`--follow-up-neighborhood --seeds NEW_SEEDS`, turning the next run into a local
stability search instead of replaying the same seed-sensitive schedule. If that
neighborhood pass still improves but remains `volatile`, guidance emits
`increase_sample_budget` and adds doubled `--epoch-values`,
`--train-graph-values`, and `--validation-graph-values` so the next run can
separate real schedule signal from small-sample variance. When that larger
budget regresses on average but surfaces a more stable top candidate,
`review_stability_tradeoff` reruns the top candidate with fresh seeds before
discarding the stability win. If that review only produces a tiny average
improvement while the source remains more seed-stable, promotion switches to
`keep_source_stability_guard` and `keep_stable_source` guidance reruns the
guarded source with fresh seeds instead of chasing the volatile average-only
gain. Once the guarded or promoted candidate produces repeated stable
improvements, `explore_stable_neighborhood` anchors that schedule and adds
`--follow-up-neighborhood --seeds NEW_SEEDS`, moving the loop from confirmation
back into local schedule search without replaying the same stable point. If that
neighborhood search finds a stable winner, `confirm_stable_promotion` reruns the
promoted schedule with fresh seeds before broadening again. When that
neighborhood includes the source schedule, `Follow-Up Result` records
`source_replay_*` metrics and replay-vs-source / best-vs-replay deltas for both
raw validation MSE and target-normalized NMSE. This separates "the schedule got
worse" from "the fresh seed pair got harder"; if the source replay shifts upward
while the best neighbor matches or beats that replay,
`review_seed_shift_neighborhood` repeats the source-anchored neighborhood with
fresh seeds instead of treating the previous source score as the only baseline.
If raw MSE regresses while source replay NMSE stays flat or improves,
`review_target_scale_shift` keeps the source anchor and widens validation graph
values with fresh seeds before classifying the run as a real schedule loss.
If the replay/neighbor evidence is still volatile, `increase_seed_shift_validation_budget`
drops the neighborhood and doubles `--validation-graph-values` so the next check
can distinguish seed noise from a real promotion boundary. Repeated seed-shift
regressions switch to `widen_seed_shift_neighborhood`, keeping the local search
but requiring fresh seeds so the next surface is not tied to stale seed evidence.
If that pattern persists, `audit_seed_sensitivity` pauses schedule widening and
remeasures the source anchor with a broader seed list plus validation budget.
The run directory also writes the same template as
`next_follow_up_command.sh`, using environment-variable placeholders so the
launcher fails fast until the caller provides the next run root and any fresh
seed list.
The smoke grid at `models/runs/gnn_tensor_util_threshold_grid_smoke` compares
thresholds 1 and 1024 for `nodes=8` and `nodes=128` with `features=4`. At
`threshold=1024`, `nodes=8` keeps both GNN layer bias reductions on CPU
(`32` and `64` values, `sum_axis0_scaled_cpu=6`), while `nodes=128` routes only
the hidden layer to WGPU (`hidden_bias_values=1024`, `sum_axis0_scaled_wgpu=3`,
`sum_axis0_scaled_cpu=3`). This mirrors the WaveRNN boundary behavior and
shows the tensor-util policy applies consistently across sequence and graph
learning paths.

The threshold-grid pattern now covers the Conv2d vision path as well.
`tools/run_vision_conv_threshold_grid.py` drives
`modelzoo_vision_conv_pool_classification` with threshold overrides and records
Conv2d bias size, Linear head bias size, predicted routes, actual
`sum_axis0_scaled` routing, WGPU kernel counts, and prepacked/matmul counts.
The smoke grid at `models/runs/vision_conv_tensor_util_threshold_grid_smoke`
compares thresholds 1 and 1024 for `batch=2` and `batch=4` with `height=8`,
`width=8`, and `out_channels=4`. At `threshold=1024`, `batch=2` keeps Conv2d
and head bias reductions on CPU (`conv_bias_values=512`,
`sum_axis0_scaled_cpu=4`), while `batch=4` routes only Conv2d bias to WGPU
(`conv_bias_values=1024`, `sum_axis0_scaled_wgpu=3`,
`sum_axis0_scaled_cpu=3`). This gives the same size-based threshold evidence
across sequence, graph, and vision learning probes.

The threshold evidence now has a small multiseed check rather than only single
smokes. The three-seed grids at
`models/runs/wave_rnn_tensor_util_threshold_grid_multiseed`,
`models/runs/gnn_tensor_util_threshold_grid_multiseed`, and
`models/runs/vision_conv_tensor_util_threshold_grid_multiseed` preserve parity
or task scores while reproducing the same routing boundaries. In WaveRNN,
`threshold=1024` keeps small `(4, hidden)` and `(64, 8)` readout/head bias
reductions on CPU (`sum_axis0_scaled_cpu=6`) and routes only the `(64, 16)`
WaveRNN readout bias to WGPU (`sum_axis0_scaled_wgpu=3`,
`sum_axis0_scaled_cpu=3`); average last-step time is mixed on the tiny cases but
improves on the larger `batch=64` grids (`183.424ms -> 152.056ms` for
`hidden=8`, `171.496ms -> 165.341ms` for `hidden=16`). In GNN, the default
threshold cuts forced small utility dispatches from 6 to 0 WGPU ops for
`nodes=8`, reducing average last-step time from `289.982ms` to `216.980ms`,
while the `nodes=128` boundary keeps only the hidden-layer bias on WGPU
(`sum_axis0_scaled_wgpu=3`, `sum_axis0_scaled_cpu=3`). In Conv2d, the default
threshold is consistently faster in this smoke (`122.503ms -> 75.302ms` for
`batch=2`, `114.537ms -> 103.440ms` for `batch=4`) while routing only the
`conv_bias_values=1024` case to WGPU. The practical conclusion is that
`SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES=1024` is a defensible default for
these three probes. The threshold-grid reports now split deliberate small CPU
utility work into `cpu_threshold_utility_ops`, so these tails are no longer
misread as true CPU-heavy work.

GNN graph readout is now split into composite metadata plus real tensor-utility
ops instead of being reported as one opaque CPU readout. Mean/Sum readout
forward uses `sum_axis0(_scaled)_with_backend()`, and Mean/Sum readout backward
uses `add_row_inplace_with_backend()` over a zero tensor. The composite
`graph_readout` and `graph_readout_backward` events keep the logical graph
operation visible while the actual reduction/broadcast backend is counted by
the lower tensor utility op. `GraphContext` now routes its cached normalised
adjacency transpose through
`transpose_with_backend(current_tensor_util_backend_for_values(...))`, keeping
graph setup under the same trainer tensor-utility policy as propagation and
readout. The focused smoke at
`models/runs/gnn_tensor_util_threshold_grid_readout_composite_large_smoke`
shows the boundary: `nodes=128, features=4` leaves readout values at 512 and
therefore keeps readout broadcast/reduction on CPU (`add_row_cpu=3`,
`sum_axis0_scaled_cpu=4`, `cpu_threshold_utility_ops=7`), while `nodes=256`
reaches `output_bias_values=1024` and routes readout plus hidden bias utility
work to WGPU (`sum_axis0_scaled_wgpu=7`, `add_row_wgpu=3`,
`cpu_threshold_utility_ops=0`). The three-seed follow-up at
`models/runs/gnn_tensor_util_readout_composite_multiseed` reproduces the routing
(`nodes=256`: `avg_wgpu_utility_ops=10`,
`avg_cpu_threshold_utility_ops=0`, `avg_cpu_heavy_ops=0`,
`avg_readout_mse=0.008928`). This is a trace/routing improvement; latency
should still be judged by longer graph runs.

The refreshed multiseed compare files now show the same pattern across all
three learning probes. WaveRNN's default threshold moves small readout/head
reductions into `avg_cpu_threshold_utility_ops=6` or `3`, GNN keeps
`nodes=128` readout work at `avg_cpu_threshold_utility_ops=7`, and Conv2d keeps
small head-bias reductions at `avg_cpu_threshold_utility_ops=4` or `3`; all
three report `avg_cpu_heavy_ops=0` in these grids. That changes the next
optimization question from "remove CPU-heavy tails" to "only fuse tiny
threshold tails if longer learning runs show they dominate latency."

The char-LM/model-zoo learning path now follows the same tensor-utility policy
instead of bypassing it with direct CPU helpers. `CharFeatureEmbedding`
normalizes its bigram gradient with `scale_with_backend()`, fixed residual logit
scaling uses backend-routed `scale_with_backend()` in forward/backward, fixed
and learned logit priors broadcast with `add_row_inplace_with_backend()`, and
the learned prior delta merge uses `add_with_backend()` while its gradient uses
the fused `sum_axis0_scaled_with_backend()`.
Core trainable layers were tightened in the same pass: `Embedding` uses
`scale_with_backend()` for batch-averaged parameter gradients, while `Scaler`
now forms its gain gradient through `hadamard_with_backend()` and the fused
`sum_axis0_scaled_with_backend()`. This keeps the model-zoo/FT surface aligned
with `BackendPolicy` and avoids reintroducing CPU-only utility tails when the
larger language probes start using these layers every step.

The policy/trace split is now explicit for that threshold path. A default WGPU
char-LM smoke under `target/tmp/char_lm_softmax_bwd_default_smoke` keeps
`tensor_ops=483`, reports `util_route_status=cpu_threshold:387`, and leaves
small elementwise/reduction/layout/embedding/hypergrad/L2/loss/softmax-backward
utility work on CPU (`cpu_residual_share=0.801`; top ops `hadamard`,
`add_scaled`, and `add`) while dense matmul/softmax/rank runtime still use WGPU
where expected. The same table now separates this into
`cpu_threshold_ops=387` and true `cpu_debt_ops=0`; `transpose`, embedding
gather/scatter, hypergrad conformal accumulation, `squared_l2_norm`,
categorical cross entropy forward/backward, and fixed-temperature
`zspace_softmax_backward` are threshold-protected utility work rather than
residual debt. Forcing
`SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES=1` in
`target/tmp/char_lm_softmax_bwd_wgpu_smoke` changes the route to
`util_route_status=wgpu:383`, moves the trace to `tensor_ops=477`,
`tensor_wgpu=477`, `tensor_cpu=0`, emits
`tensor_op_backend_transpose_wgpu=24`,
`tensor_op_backend_embedding_forward_wgpu=3`,
`tensor_op_backend_embedding_backward_wgpu=2`,
`tensor_op_backend_hypergrad_accumulate_wave_wgpu=18`, and
`tensor_op_backend_squared_l2_norm_wgpu=3`, plus
`tensor_op_backend_categorical_cross_entropy_forward_wgpu=1` and
`tensor_op_backend_categorical_cross_entropy_backward_wgpu=1`, and
`tensor_op_backend_zspace_softmax_backward_wgpu=2`; both `cpu_residual_share`
and `cpu_debt_ops` drop to `0`. This proves the current char-LM fine-tune smoke
can run its traced learning tensor ops entirely on WGPU when the utility
threshold is lowered. The tiny forced run is still slower
(`trace_step_ms_last` around `1039ms` versus `258ms` default), so the default
guard remains justified while the kernels remain reachable for larger shapes or
explicit stress runs.

A second pass removed the same class of bypass from deeper learning surfaces.
`WaveScanStack` now averages multi-dilation branches through backend-routed
scale ops, `ZSpaceSoftmax` routes fixed-temperature logit scaling before
backend-aware row softmax, `ZRBA` routes residual gate scaling for `mu` and
`sigma`, and `GradScaler::scale_tensor()` follows the tensor-utility policy for
future AMP-style training. Conv3d/Conv4d backward normalization now mirrors the
Conv1d/Conv2d policy split, and the Conv2d einsum fallback broadcasts bias with
`add_row_inplace_with_backend()`. Conv6da now follows the same parameter update
contract as the lower-dimensional convolution stack: weight and bias gradients
are scaled by `1 / batch`, duplicated batches no longer double updates, and its
empty-batch path still avoids installing zero-valued accumulators. The
self-supervised downstream fine-tune example also routes its batch-normalized
head gradient, leaving the remaining direct `scale()`/`add_row_inplace()` calls
concentrated in tests or synthetic target generation rather than trainable layer
internals.

Z-RBA covariance reconstruction is now visible at the same boundary. `CovHead`
routes the `mu` mean and `sigma` diagonal variance reductions through
`sum_axis0_scaled_with_backend()`, builds centered `mu` on CPU, and routes the
sample covariance itself as `centered.T @ centered / rows` through
`matmul_lhs_transpose_scaled_with_backend()`. It then emits
`zrba_cov_head_forward` metadata as `backend=hybrid` around the remaining
low-rank/eigen/PSD projection island. The metadata records rank, effective rank,
stabiliser, reduction backend, covariance-centering backend,
covariance-accumulation backend, low-rank/eigen backend, PSD backend, the
`nalgebra_cpu` f32 eigensolver, low-rank and PSD projection modes, the
`symmetric_eigen_decomposition_and_dense_reconstruction` blocker, eigenvalue
range, condition number, and covariance/eigen work estimates.
Trainer trace collection now also records the sub-backends as
`tensor_op_backend_zrba_cov_head_forward_covariance_centering_cpu`,
`*_covariance_accumulation_wgpu`, `*_low_rank_projection_cpu_eigen`, and
`*_psd_projection_cpu_eigen`, so comparisons expose `zrba_cov_center_cpu`,
`zrba_cov_accum_wgpu`, `zrba_cov_low_rank_cpu_eigen`, and
`zrba_cov_psd_cpu_eigen` alongside `zrba_cov_hybrid`. The small
`zrba_metric_weights_normalise` and `zrba_workspace_softmax_summary` diagnostics
are labelled `control_cpu` / `summary_cpu` and excluded from CPU-debt residuals.
Longer Z-RBA runs can therefore separate "generic reductions routed through
tensor utilities" and "trace-only control diagnostics" from "CPU eigensolver
remains the real covariance backend debt." The covariance head now validates
`mu`, diagonal variance, mean/diag reductions, centered covariance products/sums,
eigensolver inputs, eigenvalues/eigenvectors, PSD outputs, and telemetry before
returning a covariance tensor, so uncertainty heads fail at the Z-RBA boundary
instead of feeding `NaN`/`inf` into nalgebra's symmetric eigensolver.
Z-RBA metrics are now fail-fast too: prediction/variance tensors, targets,
indices length, pin/quantile, row mean/variance, interval width, NLL, CRPS,
reliability bins, and OOD Spearman are validated before a telemetry bundle is
emitted. That keeps bad uncertainty diagnostics from masquerading as valid
model quality evidence during longer Z-space sweeps.

Loss forward/backward reduction tails are now split from the loss-specific CPU
math. Focal, contrastive, and triplet losses build their semantic term/gradient
tensors first and then route the mean/batch reduction through tensor utilities:
forward terms use `try_sum_axis0_scaled_with_backend()`, while backward gradients
use `scale_with_backend()` under `BackendPolicy`. MSE, categorical cross entropy,
and hyperbolic cross entropy now have direct WGPU forward/backward kernels when
tensor utility policy chooses WGPU, while their CPU fallbacks still exercise the
same routed elementwise/reduction utilities for honest trace accounting. Focal
loss backward now matches
the exact derivative of its forward objective for both positive and negative
labels, including the weighted binary-cross-entropy `gamma=0` boundary, with
finite-difference tests locking the contract. The routed loss reductions validate
and relabel final gradients after backend scaling. The char-LM comparison table
and trainer trace spotlight now include `scale_wgpu`, `scale_cpu`, `add_wgpu`,
`add_cpu`, `hadamard_wgpu`, `hadamard_cpu`, `add_scaled_wgpu`,
`add_scaled_cpu`, `sub_wgpu`, `sub_cpu`, `mul_row_wgpu`, `mul_row_cpu`,
`row_affine_wgpu`, `row_affine_cpu`, `dynamic_field_fwd_wgpu`,
`dynamic_field_bwd_wgpu`, and loss forward/backward backend columns/keys, so
the next full run can distinguish semantic CPU loss work from backend-routed
elementwise, broadcast, accumulator, field, and reduction work instead of
reporting one opaque CPU tail.

The MSE prediction-target `sub` tail is now backend-routed as well.
`tensor_utils.wgsl` has an elementwise `sub` kernel, `wgpu_dense::sub()` exposes
it, and `Tensor::sub_with_backend()` mirrors the existing strict/fallback
semantics used by `scale_with_backend()`. `MeanSquaredError` calls it using the
tensor-utility policy, so large prediction-target differences can move to WGPU
before either the forward square/reduction or backward reduction scale. The
self-supervised downstream fine-tune example and GNN readout error trace now use
the same routed subtraction. The adjacent elementwise `add` path has the same
contract now: `tensor_utils.wgsl` exposes `add`, `wgpu_dense::add()` verifies it
against the CPU reference, and `Tensor::add_with_backend()` lets deeper residual
composition follow `BackendPolicy`. `ZRBA` uses this for the gated `mu` and
`sigma` residual attention updates, SpiralRNN uses it for recurrent projection,
gate, state, and gradient merges, and CoherenceWave uses it for scan/wave
forward and backward fusion. Large residual merges therefore no longer need to
fall back to the CPU helper after their routed gate or projection scale.
The scan half of that CoherenceWave path is routed now too.
`ZSpaceCoherenceScan` forward uses a `tensor_utils.wgsl`
`zspace_coherence_scan_forward` kernel when tensor-util policy selects WGPU.
That kernel computes Z-space scores, score normalisation, weighted context
aggregation, optional query residual blending, and the cached `(batch, steps)`
weights required by backward. The backward side now has a matching
`zspace_coherence_scan_backward` tensor-util kernel that consumes those cached
weights and differentiates through the score normalisation, not just the final
weighted value scatter. Forward/backward metadata include batch, steps,
dimension, memory window, score-pair count, nonzero-weight count, residual work,
scatter-add estimates, and `score_gradient` / `normalization_gradient` flags.
Comparisons expose `coherence_scan_fwd_wgpu`, `coherence_scan_fwd_cpu`,
`coherence_scan_bwd_wgpu`, and `coherence_scan_bwd_cpu`, so char-coherence or FT
runs can distinguish fully routed scan passes from threshold or fallback CPU
paths.
The branch side of the same block is visible now as well. `WaveScan` emits
`wave_scan_forward` / `wave_scan_backward` metadata for the final-step gather
and backward scatter that wrap its routed `Conv1d` and `WaveGate` children.
Those summary copies now use `tensor_utils.wgsl` sequence-last-step
gather/scatter kernels when tensor-util policy selects WGPU, with strict-GPU
fallback checks and CPU scalar fallback for threshold-protected small tensors.
`WaveScanStack` emits composite branch-average metadata with branch count,
context size, merge backend, and estimated average/backward-add work.
Comparisons expose `wave_scan_fwd_wgpu`, `wave_scan_fwd_cpu`,
`wave_scan_bwd_wgpu`, `wave_scan_bwd_cpu`, `wave_scan_stack_fwd_composite`, and
`wave_scan_stack_bwd_composite`, so CoherenceWave runs can separate true
convolution/gate backend routing from the remaining scalar sequence-summary
boundary without misclassifying the stack wrapper itself as CPU debt.
With `SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES=1`, the smoke run
`models/runs/char_lm_wave_scan_stack_composite_probe` now reports
`wave_scan_forward_wgpu=6`, `wave_scan_backward_wgpu=4`, `fallback_share=0`,
and only `zspace_coherence_scan_forward/backward` in the CPU debt top list.
That made the next high-value CoherenceWave backend target explicit; the
forward half is now routed, leaving the coherence scorer's score-normalisation
backward VJP as the remaining CPU-heavy boundary.
The block boundary itself is trace-visible too. `ZSpaceCoherenceWaveBlock`
emits `coherence_wave_forward` and `coherence_wave_backward` composite metadata
around its scan/wave merge and final resonator path, including dim, steps,
memory, branch count, context size, and merge backend. Comparisons expose
`coherence_wave_fwd_composite` and `coherence_wave_bwd_composite`, so a
char-coherence or FT run can count the high-level CoherenceWave block separately
from its child scan, wave-stack, WaveGate, and tensor-utility kernels.
The gate/mask side now follows the same rule: `tensor_utils.wgsl` exposes
`hadamard`, `wgpu_dense::hadamard()` verifies it against the CPU reference, and
`Tensor::hadamard_with_backend()` routes recurrent gates, resonator gates, and
dropout masks through the tensor-utility policy. SpiralRNN uses it across
state-retention, injection, anchor mixing, and the corresponding backward
partials; `ToposResonator` uses it for gate forward/backward; `Dropout` uses it
for training masks. Remaining direct subtraction/addition/hadamard sites are
now mostly tests, synthetic target construction, or non-loss utility math.
`ToposResonator` also emits its own composite layer metadata now. The
underlying multiply still reports the actual `hadamard` CPU/WGPU backend, while
`topos_resonator_forward` and `topos_resonator_backward` record the trainable
gate shape, encoder attachment, requested utility backend, and gate-gradient
work. Comparisons expose `topos_res_fwd_composite` and
`topos_res_bwd_composite`, which keeps resonance-layer semantics visible without
pretending the wrapper itself is a separate GPU kernel.

Accumulator-style `add_scaled` has now joined that utility family. The WGPU
kernel evaluates `input + scale * aux`, `Tensor::add_scaled_with_backend()` uses
the same strict/fallback metadata contract as the other tensor utilities, and
the high-value learning callers route through `BackendPolicy`: parameter
gradient accumulation/application, generic band replay accumulation, GNN
roundtable support/backprop aggregation, graph readout band accumulation,
WaveScan branch merges, and SpiralRNN gradient accumulators. Remaining direct
`add_scaled()` calls are now concentrated in st-tensor theory/topology helpers
that do not yet carry an st-nn backend policy context, plus tests.
The elementwise utility family now also preflights finite outputs before
committing either CPU or WGPU results. `add`, `sub`, `hadamard`, `scale`,
`add_scaled`, and `add_row_inplace` reject overflowing output values, while
`scale`/`add_scaled` also validate their scalar factors and deltas, and
`add_row_inplace` validates broadcast bias values before mutating. The
tensor-level `mean_squared_error()` now wraps
`mean_squared_error_with_backend()`: its logical metadata is
`backend=composite`, while difference, square, and mean-reduction pieces route
through `sub_with_backend()`, `hadamard_with_backend()`, and
`try_sum_axis0_scaled_with_backend()`. `LinearModel` mirrors that split with
`forward_with_backend()` and `train_batch_with_backend()`, so the standalone
tensor learning demo can explicitly choose matmul and tensor-utility backends
instead of being locked to public Auto wrappers. A runaway trainer step
therefore fails at the tensor utility boundary instead of silently installing
`inf` into activations, gradients, parameters, accumulators, or scalar losses.
Core CPU tensor-utility fallback emitters now preserve explicit
`requested_backend` labels through `emit_tensor_util_cpu_op_meta()`. Public Auto
wrappers still report `requested_backend=auto` by design, while explicit
`TensorUtilBackend::Cpu` calls in learning paths leave `requested_backend=cpu`
on elementwise, broadcast, reduction, layout, and projection metadata.
Reduction utilities now have the same checked path. The legacy `Vec`-returning
`sum_axis0`, `sum_axis0_with_backend`, `sum_axis0_scaled_with_backend`, and
`sum_axis1` wrappers remain for compatibility, but new fallible
`try_sum_axis0*` / `try_sum_axis1` variants validate finite inputs,
accumulators, scale factors, and outputs. Bias-gradient and readout reductions
in st-nn layers, GNN readout/layers, Z-RBA covariance, SpiralRNN/WaveRNN, and
the char-LM shared modules now route through those checked variants, so a
reduction overflow reaches the trainer as `PureResult` instead of becoming an
opaque `inf` bias or gate update.
Probability row reducers now fail at the tensor boundary too.
`row_softmax_with_backend()`, `row_softmax_hardmax_with_backend()`,
`row_softmax_hardmax_spiral_with_backend()`, and `row_hardmax_with_backend()`
reject non-finite logits before CPU/WGPU dispatch and validate their probability,
mask, and spiral outputs before returning. This removes the older softmax/hardmax
habit of quietly ignoring `NaN` logits, which is useful for forgiving demos but
dangerous in trainer, attention, and contrastive-loss paths.
`ZSpaceSoftmax` now preserves that boundary at the module level: input logits,
scaled logits, probability rows, entropy/temperature adaptation, backward dots,
and softmax VJP outputs are validated before returning tensors. Its
entropy/temperature diagnostics are also committed only after the forward output
is successfully built, so a failed adaptive pass no longer leaves stale-looking
metrics from an invalid probability distribution.
LayerNorm now has the same fail-fast boundary. `layer_norm_affine(_add)` rejects
non-finite input, gamma, beta, and residual payloads even on empty batches,
validates CPU/WGPU outputs before returning, and checks CPU fallback
sum/sum-square/mean/variance/denominator/normed intermediates so huge-but-finite
activations cannot be normalised into deceptively usable tensors. Its backward
path now rejects non-finite/overflowed row-stat and normed intermediates, then
routes the reduction-heavy VJP tail through tensor utilities: `grad_output *
gamma`, norm-dot/sum reductions, correction broadcasts, final inv-std scaling,
and affine `gamma`/`beta` gradients now use `mul_row`, `hadamard`,
`sum_axis0`, `add_row`, `add_scaled`, and `try_sum_axis0_scaled` with the
current tensor-util policy. Plain `LayerNorm` still recomputes row
mean/variance/normed on CPU, so backward metadata reports
`input_gradient_backend=hybrid`, `input_gradient_reduction_backend`,
`normalization_backend=cpu`, `input_gradient_axis`, `input_gradient_formula`,
and `affine_gradient_backend`. `BatchNorm1d`, `ZSpaceBatchNorm1d`, and
`ZSpaceLayerNorm` use cached normalization state and now report routed
input-gradient/normalization sub-backends directly. Trainer trace collection
therefore separates `tensor_op_backend_*_input_gradient_hybrid`,
`tensor_op_backend_*_input_gradient_reduction_wgpu`, and
`tensor_op_backend_*_normalization_cpu`, so WGPU runs no longer hide the
remaining CPU normalization debt behind the broader hybrid normalization event.
`ZSpaceLayerNorm` now carries that rule through its projected path too:
per-row mean/variance/inv-std, normalised/projected/Jacobian/radius buffers, and
affine outputs are validated before telemetry is committed, and backward rejects
non-finite cached projection state before affine accumulators are touched. Its
projected affine gradients now use the same backend-aware reducer as plain
LayerNorm, so Z-space `gamma`/`beta` updates no longer hide inside a hand-written
CPU accumulation loop.
`BatchNorm1d` forward now follows the same rule for stateful normalization:
input/gamma/beta/running-stat payloads are validated, mean/variance/inv-std and
output intermediates are checked, and candidate running means/variances are
scratch-built before commit. Failed or overflowing batch statistics therefore no
longer poison running statistics or cached backward state; backward also rejects
non-finite inputs/gradients/cached stats before affine accumulators are touched.
BatchNorm and Z-space BatchNorm now route their `gamma`/`beta` gradient tails
through `hadamard_with_backend()` and `try_sum_axis0_scaled_with_backend()` too,
leaving only the normalization input-gradient analytic loops as intentional CPU
work. Their training-mode forward statistics use the same utility boundary:
per-feature means are `sum_axis0_scaled`, variances are centered CPU buffers
followed by backend-routed `hadamard` and `sum_axis0_scaled`, and the old
hand-written batch-axis reduction is gone from the state update path.
`ZSpaceBatchNorm1d` mirrors that boundary for the projected path: the
normalised, projected, Jacobian, radius, and output buffers are validated before
telemetry is committed, while backward rejects non-finite cached projection
state before gamma/beta accumulators are touched.
Its `adapt_projector_gain()` control loop now validates target/smoothing/gain
inputs and computes the telemetry radius mean through the checked reducer too,
so corrupted or overflowing radius telemetry fails without committing a new
projector gain.
Scaled matmul follows the same boundary now. `matmul_scaled_with_backend()` and
`matmul_lhs_transpose_scaled_with_backend()` reject non-finite scale factors
even on empty outputs, and validate the produced CPU/WGPU/HIP/faer/naive output
buffer before returning a tensor. Weight-gradient construction, self-supervised
gradient matmuls, and legacy `LinearModel` training therefore fail before an
overflowing scaled dot product becomes an ordinary gradient tensor.
Plain and prepacked dense matmul now share that commit rule.
`matmul_into_with_backend()` and `matmul_prepacked_into_with_backend()` compute
into scratch buffers, validate finite output, and only then copy into the
caller-provided destination. `matmul_prepacked_bias_with_backend()` validates
broadcast bias values and fused outputs before returning, closing the
Linear/GNN/RNN forward path that sits next to the scaled-gradient matmul
boundary.
Fused MLP projections now follow the same contract. `matmul_bias_relu` and
`matmul_bias_add_relu` validate bias/residual payloads, compute `*_into` calls in
scratch buffers, validate finite activated outputs, and only then commit to the
caller destination. The GELU fused variants validate bias/residual inputs and
their returned CPU/WGPU/HIP/faer/naive outputs before constructing tensors, so
overflowed fused activations no longer enter learning layers as ordinary finite
contracts. The standalone `Gelu` layer now mirrors that fail-fast boundary for
non-fused MLP blocks: scalar forward intermediates and backend/fallback backward
outputs are validated before a tensor is returned.

The optimizer accumulator scaling path has also been split along the same
boundary. Existing public `Parameter::scale_accumulators()` keeps its CPU-only
compatibility behavior, but now prevalidates the scaled buffers and no-ops
rather than writing `inf`/`NaN` into legacy accumulator state. Trainer
grad-scale, mixed-precision unscale, and local-LR adapter scaling now call
`scale_accumulators_with_backend_policy()`. That routes the Euclidean fallback
gradient tensor through `scale_with_backend()` and propagates strict GPU
fallback errors. Hypergrad and realgrad tapes now expose
`scale_gradient_with_backend()` as well, so their accumulator scaling uses the
same tensor utility policy while preserving hypergrad's topos saturation and
summary bookkeeping. Tape application now has explicit backend variants too:
Realgrad applies `weights += -lr * gradient` through
`add_scaled_with_backend()`, and Hypergrad keeps its per-value hyperbolic
update/saturation loop but routes the final Poincare projection through
`project_to_poincare_with_backend()`. The remaining tape-local CPU loops are now
the accumulate math and the hypergrad pre-projection update, where hyperbolic
saturation and per-value transition accounting are still semantically coupled
to each value. Those loops are now visible in trainer traces as
`hypergrad_accumulate_wave`, `hypergrad_accumulate_pair`,
`hypergrad_apply_update`, `realgrad_accumulate_wave`, and
`realgrad_accumulate_pair`, so the next fusion decision can be based on real
run frequency instead of static audit guesses.

The trainer step boundary now has the same fail-fast numeric contract. Each
`ModuleTrainer::train_epoch()` batch validates finite inputs, targets,
predictions, loss tensors, gradient outputs, band energies, final band weights,
scaled band gradients, region loss factors, weighted loss, total loss, and
parameter accumulator buffers before an optimizer step can run. Roundtable
directives are applied only after band weights have passed validation, are
cleared even when backward returns an error, and non-finite accumulators are
zeroed before returning. `GradScaler` also exposes `try_scale_loss()` so scalar
AMP loss scaling can reject non-finite inputs or overflow instead of silently
creating an `inf` scalar. The actual Above/Here/Beneath replay scaling now
routes through `GradientBands::scale_inplace()` as a fallible tensor-utility
operation too: all three scaled band tensors are scratch-built through
`scale_with_backend()` and committed only after every band succeeds, so a bad
band multiplier cannot partially rewrite the replay gradients. Validation runs
now share the same boundary: `evaluate_epoch()` installs the trainer
`BackendPolicy`, rejects non-finite input/target/prediction/loss tensors and
overflowed validation-loss sums, and still restores training mode after an
evaluation error. Early stopping therefore sees either finite validation
statistics or a real `PureResult` failure, not a silently accumulated
`inf` score.

The self-supervised InfoNCE backward path has also moved its dense gradient
mixing out of a hand-written triple loop. The expensive `P @ positive_hat` and
`P^T @ anchor_hat` terms now use `matmul_scaled_with_backend()`, and the
diagonal correction uses tensor utility `add_scaled_with_backend()`. The
probability transpose that feeds `P^T @ anchor_hat` now uses
`transpose_with_backend()`, removing the last current forced-WGPU selfsup CPU
debt (`transpose:2 -> transpose_wgpu:2`). Row normalisation still runs as a
small scalar projection loop over cached normalized rows, but the
batch-quadratic portion now follows the same backend policy as forward logits.
The remaining scalar/projection boundaries are fail-fast as well:
`InfoNCEEpochMetrics::from_batches()` rejects non-finite or overflowed epoch
loss sums, `finish_info_nce_loss()` rejects logits that become non-finite after
temperature scaling, and normalized backward rows are projected into scratch
buffers before commit so a bad anchor/positive norm cannot partially mutate the
gradient.

Z-space softmax backward is still a CPU Jacobian-vector loop, but it now follows
the forward's one-step adaptive-temperature path instead of treating the
temperature as a constant. When entropy targeting actually changes a row's
temperature without hitting tolerance or clamp boundaries, backward adds the
exact `dL/dT * dT/dH * dH/dx` contribution from the initial entropy probe before
the final softmax VJP. `zspace_softmax_backward` metadata records the active
softmax backend request, shape, curvature, adaptive-temperature flag,
`temperature_gradient=one_step_entropy_exact`, and the number of rows whose
temperature gradient was active. A finite-difference test locks that adaptive
path. This exposes `softmax_bwd_cpu` in char/self-supervised run comparisons,
making it clear when a fused backward kernel would matter more than further
forward-softmax work.

ContinuousWaveletTransform is another old Rust backend island, but its hottest
`rows * cols^2 * scales` response work no longer lives in hand-written nested
loops. Each scale now builds the Morlet response as `input @ kernel^T` through
`matmul_with_backend()`, backward builds the input gradient as
`grad_output @ weighted_kernel`, and focus/bias gradients use
`hadamard_with_backend()` plus `sum_axis0_scaled_with_backend()`. The focus and
bias parameters now follow the same batch-normalised update contract as the
other broadcast gates: duplicated rows no longer double parameter updates,
zero-row batches do not install empty updates, and the input gradient remains an
unaveraged chain-rule signal. The layer-level metadata is now
`backend=composite` with `response_path=matmul_per_scale`, `response_backend`,
`mix_backend`, and the parameter gradient scale, while the underlying `matmul`,
`add_scaled`, `add_row_inplace`, `hadamard`, and `sum_axis0_scaled` events show
which backend actually ran. A finite-difference test locks the wavelet
`grad_input` contract. The remaining debt is kernel construction/transpose
materialisation and the fact that each scale is still dispatched as a separate
matmul rather than a fused batched wavelet kernel.

The dynamic-field layer family is now trace-visible as well. Klein-Gordon,
Hamilton-Jacobi, and stochastic-Schrodinger forward/backward paths emit
`dynamic_field_*_{forward,backward}` metadata with field model, trainable
parameter count, estimated work, selected backend, and backward gradient scale.
Klein-Gordon and Hamilton-Jacobi now leave the scalar-only island:
`dynamic_klein_gordon_forward` computes the wave update on WGPU,
`dynamic_klein_gordon_backward` returns packed input/mass/spinor gradients,
`dynamic_hamilton_jacobi_forward` evaluates the temporal stencil with boundary
rows, and `dynamic_hamilton_jacobi_backward` gathers neighbouring upstream
signals without atomics before returning packed input/potential gradients.
Stochastic-Schrodinger now splits along an honest composite boundary:
`dynamic_schrodinger_forward` computes deterministic interference plus
amplitude/decoherence caches on WGPU, CPU still consumes the RNG stream and adds
noise in the original row-major order, and `dynamic_schrodinger_backward`
computes packed input/coherence gradients from the cached tensors. The existing
batch scale/accumulator contract is still applied after those kernels. Their
broadcast feature parameters now follow the same batch-normalised update
contract as the other one-row gates: input gradients remain unaveraged
chain-rule signals, parameter gradients are scaled by `1 / rows`, and zero-row
batches do not install empty mass/spinor/potential/coherence updates. The
comparison table now separates `dynamic_field_fwd_wgpu` /
`dynamic_field_bwd_wgpu` for all three field models from remaining
`dynamic_field_fwd_cpu` / `dynamic_field_bwd_cpu` residuals, while raw traces
preserve which field model is responsible and mark stochastic forward with
`rng_backend=cpu`.

Recurrent layers now protect their cached-time boundary more deliberately.
`WaveRnn` and `SpiralRnn` clear stale forward caches at the start of a new
forward pass, keep caches available when backward rejects non-finite upstream
gradients, and only consume the cache after gradients have been validated.
`WaveRnn` also delays readout accumulator commits until gate and convolution
backward have produced finite gradients, so a late recurrent failure does not
leave readout-only updates behind.

The legacy single-layer `Lstm` is also trace-visible now. Forward batches the
input projection as `input.matmul_with_backend(weight_ih, current_matmul_backend())`
and applies the input bias through tensor-util `add_row_inplace`; each timestep's
hidden recurrent projection now also uses `matmul_with_backend(weight_hh, ...)`,
and the final gate step (`sigmoid/tanh + cell/hidden update`) is a tensor-util
route with a CPU fallback and a WGPU `lstm_forward_gate_step` helper.
`lstm_forward` now reports `backend=hybrid`, `input_projection_backend`,
`bias_backend`, routeable `recurrent_backend`, `gate_activation_backend`, and
`gate_activation_fallback_reason`;
`lstm_backward` still reports `backend=hybrid`, but the routeable inner products
are now split out: `input_gradient_backend` follows the current matmul policy as
one batched sequence projection over all collected gate gradients, while
`recurrent_backend` still follows the policy inside the reverse-time recurrence.
The metadata makes that boundary explicit with
`input_gradient_mode=batched_sequence_matmul` and
`recurrent_gradient_mode=reverse_time_step_matmul`. Weight-gradient reduction reports
`parameter_gradient_reduction_backend`, bias reduction reports
`bias_gradient_backend`, and `raw_parameter_gradient_backend=hybrid` marks the
remaining mix of scalar gate derivatives plus backend-routed reductions.
Forward gate activation debt is now conditional on the selected tensor-util
route, while `bptt_backend` is either `cpu` or `wgpu` depending on the selected
scan route. CPU fallback is decomposed into `bptt_gate_derivative_backend=cpu`,
`bptt_cell_recurrence_backend=cpu`, and `bptt_state_carry_backend=cpu` with
`bptt_sequence_dependency=reverse_time_recurrence`. The reverse recurrence is
now isolated behind `bptt_scan_backend=cpu`,
`bptt_scan_kernel=lstm_backward_scan.cpu_fused_loop`, and
`bptt_scan_lowering=host_reverse_recurrent_scan`. When the matmul policy is
WGPU, LSTM now attempts the `lstm_backward_scan.wgsl` single-workgroup
hidden-parallel recurrent scan first and records either `bptt_scan_backend=wgpu` or an honest
`bptt_scan_fallback_reason`. New traces also split the precondition from the
runtime outcome with `bptt_scan_shape_supported`,
`bptt_scan_runtime_requested`, and `bptt_scan_runtime_available`, so an adapter
or runtime miss can be distinguished from a shape blocker without parsing the
fallback string. It also records scan elapsed microseconds, gate/hidden/cell
and recurrent-weight value counts, kernel-dispatch count, serial scan steps, and
estimated operations per scan step. A future parallel lowering can replace that
single scan seam while keeping the same gate-gradient contract and preserving a
before/after performance trail.
`bptt_fusion_candidate=fused_lstm_backward_scan` and
`bptt_scan_fusion_blocker=grad_h_next_and_grad_c_next_recurrence` remain in the
trace as the design intent and dependency shape.
Timesteps, input/hidden dimensions, gate width, trainable parameter count,
estimated input/recurrent/input-gradient/BPTT work, dispatch-count estimates for
the batched projections versus timestep recurrent projections, sub-estimates
for gate-derivative/cell/state-carry work, `estimated_bptt_cpu_debt_ops`,
`estimated_bptt_wgpu_ops`, and the gradient scale applied to shared parameters
remain visible.
The BPTT input gradient remains an unaveraged chain-rule signal, while the
input/recurrent kernels and gate biases now scale their accumulated updates by
`1 / timesteps` so longer scalar sequences do not silently take larger optimizer
steps. LSTM now validates explicit state loads, inputs, loaded parameters, gate
preactivations, hidden/cell states, cached forward tensors, BPTT intermediates,
input gradients, and reduced parameter gradients before committing
hidden/cell/cache state or accumulator updates; a failed forward invalidates
stale cache while preserving the previous recurrent state, and a failed backward
leaves the retryable cache intact. Trainer trace collection records these
sub-backends as keys such as `tensor_op_backend_lstm_forward_recurrent_wgpu`,
`tensor_op_backend_lstm_backward_input_gradient_wgpu`,
`tensor_op_backend_lstm_backward_parameter_gradient_reduction_wgpu`,
`tensor_op_backend_lstm_forward_gate_activation_wgpu`, and
`tensor_op_backend_lstm_backward_bptt_scan_wgpu`, so WGPU sequence runs can
separate routed input/recurrent projection, the now-batched backward input
projection, backward parameter reductions, parameter scaling, WGPU gate
activation, and WGPU backward scan work. The same trainer summaries now surface
estimated gate/BPTT operation totals, CPU-vs-WGPU splits, and scan-step counts,
so a future parallel backward-scan kernel can be judged by both event-count debt
and the size of the recurrent work it removes.

`ZSpaceMixer` now exposes and routes its old row-gate broadcast island too.
Forward and the input-gradient side use `Tensor::mul_row_with_backend()`, a
row-broadcast tensor utility backed by the WGPU `mul_row` kernel when policy and
threshold allow it. Backward gate-gradient formation still routes through
`hadamard_with_backend()` followed by `sum_axis0_scaled_with_backend()`. The
broadcast input gradient stays on the unaveraged chain-rule path, while the
single-row gate gradient now follows the same batch-normalised parameter
contract as `Scaler` and `NonLiner`; zero-row batches return empty input
gradients without installing a gate update. Mixer metadata reports
`zspace_mixer_forward`, `zspace_mixer_backward`, `broadcast_backend`, the
reduction backend, and the gate-gradient scale, and char/self-supervised
comparisons can now separate layer-level `mixer_fwd_composite` /
`mixer_bwd_composite` from the underlying `mul_row_*` and `reduce_*` counters,
while legacy `mixer_fwd_cpu` / `mixer_bwd_cpu` columns remain readable for
older traces. That same row-broadcast primitive now also
covers Scaler plus NonLiner's forward gain and backward input-gradient legs,
leaving the next larger boundary as a fused smooth-gate/activation kernel
rather than another generic row-scale dispatch.

`WaveGate` now follows the same backward-boundary rule. Its forward path was
already routed through `wave_gate_project_with_backend()` when policy requests
WGPU, and its backward row-wise VJP can now follow the same route. The WGPU
kernel recomputes the topos-rewritten affine gate, applies the row-wise Poincare
projection Jacobian-vector product, multiplies by the exact local derivative of
the OpenCartesianTopos porous saturation rewrite, and returns a packed
`grad_input` plus `grad_affine` buffer. Gate and bias parameter gradients then
reuse the existing backend-routed `hadamard_with_backend()` and
`sum_axis0_scaled_with_backend()` reductions, so the fused analytic part and the
generic reduction part remain separately visible in traces. The CPU fallback
keeps the same chain-rule path: input gradients remain unaveraged, gate and bias
parameter gradients apply and report the `1 / rows` batch-normalised update
scale, and zero-row batches return empty input gradients without installing
gate/bias updates. `wave_gate_backward` metadata records the selected backend,
reduction backend, gradient scale, curvature, saturation, porosity,
`effective_gate_rewrite=true`, `projection_gradient=true`, and
`saturation_gradient=porous_mix_exact`. Finite-difference tests still lock the
projected and porous-saturation `grad_input` contracts, and a WGPU parity test
now checks `grad_input`, gate gradient, and bias gradient against the CPU path.
Comparisons expose `wave_gate_fwd_wgpu` / `wave_gate_fwd_cpu` from the fused
projection op and `wave_gate_bwd_wgpu` / `wave_gate_bwd_cpu` from the backward
layer boundary.

`ZSpaceProjector` now lines up with that same backend contract. Forward no
longer calls the legacy `project_to_poincare()` Auto path directly; it selects
`current_tensor_util_backend_for_values()` and calls
`project_to_poincare_with_backend()`, so larger projector tensors can follow
the active trainer/device policy while small tensors stay threshold-protected.
The projector layer also emits `zspace_projector_forward` as
`backend=composite` and `zspace_projector_backward` as the remaining CPU layer
boundary around rewrite/projection gradients.
Backward no longer treats that boundary as an identity rewrite: it reconstructs
the topos-rewritten preprojection input, applies the row-wise Poincare
projection Jacobian-vector product, then multiplies by the exact local
derivative of the porous saturation rewrite. The backward metadata records
`projection_gradient=true` and `saturation_gradient=porous_mix_exact`, and a
finite-difference test locks the projected/saturated `grad_input` contract.
Comparisons expose `projector_fwd_composite` / `projector_bwd_cpu`, while
legacy `projector_fwd_cpu` remains readable for older traces and the underlying
`project_to_poincare_{wgpu,cpu}` counters still show whether the actual forward
hyperbolic projection kernel routed to WGPU or CPU. The remaining projector debt
is therefore a fused/routed backward kernel, not a hidden analytic mismatch.

`Scaler` now closes the same gap for the simplest trainable feature gate. The
forward and input-gradient paths now share the row-broadcast `mul_row` tensor
utility with `ZSpaceMixer`, while the gain gradient routes `input * grad_output`
through `hadamard_with_backend()` and batch-normalises via
`sum_axis0_scaled_with_backend()`. Layer metadata reports `scaler_forward` and
`scaler_backward` with `backend=composite`, `broadcast_backend`, and the
gradient-reduction backend; comparisons expose `scaler_fwd_composite` /
`scaler_bwd_composite` alongside the underlying `mul_row_*`, `hadamard_*`, and
`reduce_*` counters, while legacy `scaler_fwd_cpu` / `scaler_bwd_cpu` columns
remain readable for older traces. The scaler is now fail-fast at the learning
boundary too: explicit gains, loaded gains, calibration samples,
calibrated mean/variance/gain intermediates, forward broadcast outputs,
backward input gradients, and gain-gradient products are
validated before parameter/baseline/accumulator state is updated.

`NonLiner` is now split along the same learning boundary. Activation and
hyperbolic/elliptic geometry remain scalar CPU math, but backward parameter
aggregation no longer hides inside one hand-written loop: gain and slope
gradients are formed through backend-routed `hadamard_with_backend()` and all
three parameter gradients use `sum_axis0_scaled_with_backend()`. Forward now
forms preactivation with `row_affine_with_backend(input, slope, bias)`,
computes scalar activation values, then applies the learned gain with
`mul_row_with_backend(activated, gain)` before optional geometry; backward
reuses the same preactivation utility and uses `mul_row_with_backend(delta,
slope)` for the final input-gradient broadcast. Larger smooth gates can
therefore route preactivation and both row-scale legs to WGPU while activation
and curvature math remain explicit CPU semantics. The input gradient still stays on the exact chain-rule path
without batch averaging, while parameter reductions continue to report their
batch-normalisation scale through metadata and finite-difference tests lock the
Euclidean plus non-Euclidean input gradient contract. The layer is also
fail-fast at the same state boundary:
inputs, loaded gain/slope/bias tensors, pre-activations, activation outputs,
curved geometry norms/radii/JVP terms, chain-rule intermediates, input
gradients, and reduced parameter gradients are validated before ψ/radius
telemetry or parameter accumulators are committed. Layer metadata reports
`backend=composite`, activation family, activation CPU backend, geometry,
optional geometry CPU backend, curvature/Z-scale/retention, ψ drift, geometry
radius, forward broadcast backend, input-gradient backend, and gradient
reduction backend. Comparisons expose `non_liner_fwd_composite` /
`non_liner_bwd_composite` next to the utility counters, while legacy
`non_liner_fwd_cpu` / `non_liner_bwd_cpu` columns remain readable for older
traces. A larger FT trace can now distinguish "activation or geometry scalar
math is hot" from "parameter reduction still needs a fused kernel."

The stateless `Relu` layer now participates in the same trace vocabulary and can
run as a single WGPU tensor-utility kernel in both directions. Forward calls
`wgpu_dense::relu()` when tensor utility policy selects WGPU and emits
`relu_forward` metadata with active/inactive value counts; backward calls
`wgpu_dense::relu_backward()` for the input-mask VJP and otherwise falls back to
the routed CPU mask plus `hadamard_with_backend()` path. Comparisons expose
`relu_fwd_wgpu` / `relu_bwd_wgpu` beside the CPU columns, which matters because
ReLU appears across the MLP, GNN, vision, WaveRNN, and char/coherence examples.
With `SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES=1`, the focused cross-model smoke
changed the WaveRNN residual debt from ReLU-heavy (`relu_forward=3`,
`relu_backward=2`) to `wave_gate_backward=2`, moved the vision stack's ReLU
events (`relu_forward=6`, `relu_backward=4`) to WGPU while leaving max-pool and
hyperbolic-CE tails, reduced GNN ReLU events (`relu_forward=8`,
`relu_backward=6`) to WGPU, and left the self-supervised smoke with no true
compute debt after threshold-protected transpose work.

The same forced-WGPU follow-up proves the direct MSE kernels are now reached
from end-to-end learning traces. `target/tmp/wave_rnn_mse_wgpu_smoke` reports
`tensor_ops=70`, `tensor_wgpu=63`, `tensor_cpu=2`, zero fallback events,
`mse_loss_forward_wgpu=1`, and `mse_loss_backward_wgpu=1`; the only remaining
non-threshold CPU debt is `wave_gate_backward=2`. The GNN smoke at
`target/tmp/gnn_mse_wgpu_smoke` reports `tensor_ops=322`, `tensor_wgpu=267`,
`tensor_cpu=47`, zero fallback events, and the same two WGPU MSE loss events.
Its remaining non-threshold debt is now `cat_rows=4` plus `graph_flow_*` trace
annotations (`graph_flow_layer_begin=16`,
`graph_flow_roundtable_annotation=12`, `graph_flow_weight_update=12`) rather
than loss math. That makes the next GNN step clearer: classify trace-only graph
flow events separately, then decide whether `cat_rows` needs a real batched
readout kernel.

The WaveGate backward kernel closes that WaveRNN tail. With the default utility
threshold, `target/tmp/wave_rnn_wave_gate_bwd_default_smoke` keeps this tiny
shape on CPU (`wave_gate_backward_cpu=2`) and reports it as threshold-protected
work (`cpu_debt_ops=0`) rather than true residual debt. Forcing
`SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES=1` in
`target/tmp/wave_rnn_wave_gate_bwd_wgpu_smoke` moves the same events to
`wave_gate_backward_wgpu=2`, keeps `wave_gate_project_wgpu=3` and
`mse_loss_{forward,backward}_wgpu=1`, and reaches `tensor_ops=70`,
`tensor_wgpu=65`, no CPU backend events, zero fallbacks, and `cpu_debt_ops=0`.
That leaves the WaveRNN smoke with no traced CPU learning debt under forced
WGPU. The GNN follow-up at `target/tmp/gnn_trace_copy_classified_smoke` now
separates the remaining host-side work instead of reporting it as one opaque CPU
tail: `tensor_ops=322`, `tensor_wgpu=270`, `tensor_cpu=44`,
`squared_l2_norm_wgpu=6`, `cpu_trace_ops=40` for `graph_flow_*` XAI telemetry,
`cpu_copy_ops=4` for `cat_rows`, zero fallbacks, and `cpu_debt_ops=0`. That
leaves the WaveRNN and GNN forced-WGPU smokes with no traced CPU compute debt;
the vision forced-WGPU follow-up moves hyperbolic CE and max-pool to WGPU as
well, reaching `tensor_wgpu=46`, no CPU backend events, zero fallbacks, and
`cpu_debt_ops=0`. The next cross-model backend audit should therefore move past
these smoke-level compute islands and look for the next non-smoke CPU boundary,
while treating GNN `cat_rows` as a future tensor-resident/batched-readout
data-movement kernel rather than compute debt. The selfsup follow-up
`models/runs/current_backend_audit/selfsup_wgpu_transpose_routed` now joins that
bucket too: its InfoNCE transpose route reaches `tensor_wgpu=62`, no CPU backend
events, zero fallbacks, and `cpu_debt_ops=0`.

The audit now has a reproducible backlog generator as well:
`tools/audit_learning_backend_backlog.py` combines trainer trace residuals with
static Rust source heuristics for older backend boundaries. Its default static
scan now covers `st-nn`, `st-tensor`, `spiral-selfsup`, and the learning-adjacent
`st-core` distributed/engine/backend modules, so distributed gradient reducers,
autograd routing, allreduce/ZeRO hooks, and backend-control heuristics are
visible next to layer/loss tensor debt. Dynamic residuals skip CPU-only baseline
runs by default so mixed sweep directories do not report expected CPU execution
as GPU backend debt; pass `--include-cpu-runs` when baseline traces are the
thing being audited. Running it against
`models/runs/current_backend_audit` writes
`models/runs/current_backend_audit/backend_backlog.md`: the dynamic section
confirms current forced-WGPU char, WaveRNN, vision, selfsup, GNN, and LSTM
smokes have `cpu_debt_ops=0` after classifying GNN `graph_flow_*` telemetry and
`cat_rows` copy work separately. It also includes the focused
`models/runs/current_backend_audit/lstm_wgpu` sequence probe built from
`modelzoo_lstm_sequence_probe`; that run requests WGPU, records
`tensor_ops=132`, `tensor_wgpu=125`, zero fallbacks, `cpu_debt_ops=0`,
`lstm_est_cpu_debt=0`, `lstm_est_bptt_wgpu=2016`, and
`lstm_scan_rt_ok=12`. The multiseed LSTM WGPU sweep rows likewise report
`tensor_ops=124`, `tensor_wgpu=117`, zero fallbacks, `cpu_debt_ops=0`,
`lstm_est_cpu_debt=0`, `lstm_est_bptt_wgpu=864`, and `lstm_scan_rt_ok=3`.
CPU pretrain reference evaluations are no longer subscribed into the training
TensorOpMeta stream, so they do not pollute backend debt for the traced training
step. Older trace metadata can still be summarized as `lstm_est_cpu_debt`, but
new traces now make WGPU gate activation and `lstm_backward_scan.wgsl` scan work
visible as WGPU estimates instead of residual CPU debt. The LSTM sweep compare
table now carries scan profile columns as well, including `lstm_bwd_scan_us`,
`lstm_bwd_scan_gate_values`, `lstm_bwd_scan_recurrent_weight_values`,
`lstm_bwd_scan_dispatches`, `lstm_bwd_scan_serial_steps`, and
`lstm_est_bptt_ops_per_scan_step`.
The static section remains intentionally conservative and keeps source-level
follow-up candidates visible even when smoke traces are clean: the former P1
LSTM CPU island has moved from dynamic debt to a performance/parallelism seam,
while P2 candidates now include `st-core` distributed gradient/autograd CPU
metadata, engine allreduce/ZeRO hook dispatch, and ZRBA covariance's CPU
eigensolver island. These are not proven runtime debt until exercised by a trace,
but the `st-core` entries now expose their concrete host mechanisms:
`DistributedTrainer` reports `cpu_collective_arena`, `cpu_mutex_vec`, CPU
accumulation/update modes, and queue/update value estimates; `AmebaAutograd`
reports its `cpu_vecdeque` message queue, `cpu_hashmap` agent state, CPU loop
weight updates, and the stateful graph/message-queue blocker;
`distributed_topk_merge` reports host pair filtering, CPU sort/truncate mode,
and sort item estimates; engine hook points report opaque registered host
callbacks for one-bit allreduce and ZeRO partition hooks. Backend residuals now
bucket these as `cpu_control_*`, keeping distributed learning control debt
visible without pretending control-plane hooks are already tensor kernels. The
remaining semantic/topos P2 entries are now framed
the same way: sparse semantic scans and guarded topos rewrites stay visible as
places to probe when expanding beyond the current smoke models, while the LSTM
P1 should move next from the WGPU hidden-parallel scan helper toward runtime
verification and a cross-timestep parallel/backward-scan kernel rather than more
single-op dispatches.

`Dropout` is now visible at the same layer boundary. Mask generation remains a
deterministic CPU RNG path, but forward and backward emit `dropout_forward` and
`dropout_backward` metadata with training/eval mode, probability, keep scale,
retained/dropped value counts, and the backend selected for mask application.
The actual masking continues to use `hadamard_with_backend()`, so comparisons
can separate stochastic layer composition (`dropout_fwd_composite` /
`dropout_bwd_composite`, with `rng_backend=cpu`) from the underlying
`hadamard_*` utility route, while legacy `dropout_fwd_cpu` /
`dropout_bwd_cpu` columns remain readable for older traces. It now also
rejects non-finite forward inputs, backward inputs, upstream gradients, and mask
payloads before the random mask can hide a bad activation or gradient as a
dropped value; training masks are cached only after the masked output is finite.

Next steps:

1. Continue fusing learning-boundary tails rather than adding single-op
   dispatches: the `lstm_wgpu` and LSTM sweep probes now show zero dynamic
   compute debt while keeping WGPU gate/scan work visible through
   `lstm_est_bptt_wgpu` and scan runtime counters. The WGPU path has
   `lstm_backward_scan.wgsl` as a single-workgroup hidden-parallel first landing; verifying its
   step-time impact with the new scan profile columns and parallelising the scan
   is now the next recurrent learning-kernel bottleneck rather than merely
   proving the route exists.
   The remaining WaveRNN reshapes are visible as `reshape_view`, not CPU-heavy
   work, so reducing them further is an API/trace-cleanliness question.
2. Use the multiseed grids as a baseline when changing utility thresholds or
   adding fused readout/head-bias paths; new changes should improve at least one
   domain without regressing parity or exploding small-WGPU dispatches.
3. Compare `util_route_status` against `cpu_debt_top` before adding new
   kernels; threshold-protected CPU tails should first get fused at a larger
   learning boundary, not automatically forced through single-op WGPU dispatch.
4. Capture latency or step-time deltas for the small generated WGPU bucket and
   tune its `ctile`/tile fields until `wgpu_generated_score` improves over the
   baseline without regressing validation loss.
5. Treat Conv/GNN/WaveRNN/selfsup small bias/readout/layout-copy events as
   threshold-protected or data-movement work, not CPU-heavy debt; only add fused
   head/readout kernels if longer learning traces show those tiny reductions
   dominate latency.
6. Promote the self-supervised accumulator-sync comparison from local smoke and
   two-rank unit coverage into a reusable distributed run artifact, then add
   schedule/early-stop controls around `best_info_nce` and `final_minus_best`.

## Suggested PR Sequence

1. Trainer trace captures `TensorOpMeta` backend counters for matmul,
   prepacked matmul, fused dense projections, softmax/hardmax fallback, layer
   norm fallback, and attention fallback; char-LM comparisons should now use
   the operation-level columns to choose the next larger FT probe.
2. Extend `BackendPolicy` routing from `Linear`, GNN, RNN, Conv2d backward, and
   Z-RBA projections to normalization-heavy and fused-attention learning paths.
3. Broaden backend routing beyond GELU backward into normalization backward and
   remaining reduction-heavy learning paths.
4. `TrainingDevice` bridge for `ModuleTrainer` accumulator synchronization,
   including two-rank smoke coverage.
5. End-to-end parity harnesses for char-LM and graph-regressor one-step runs.
