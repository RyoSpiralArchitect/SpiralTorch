// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Open-cartesian topos guards that keep the pure tensor stack loop-free and
//! numerically stable even in the presence of extreme curvatures.
//!
//! The implementation focuses on three guarantees that were repeatedly flagged
//! as weaknesses in the original stack:
//!
//! * **Numerical safety** – All tensors are validated for finite components and
//!   projected through a saturation window so NaNs and infinities are rewritten
//!   into bounded values.
//! * **Loop freedom** – Traversals through fractal depths are capped by an
//!   "open cartesian" horizon which ensures self-referential rewrites never
//!   re-enter the same stratum.
//! * **Solver determinism** – The conjugate gradient solver exposes explicit
//!   tolerance and iteration limits so hyperbolic Jacobians cannot silently
//!   diverge.
//!
//! The module intentionally stays allocation-light so the new guards can be used
//! from both CPU-only and WASM environments without fighting the borrow checker.

use super::{fractal::FractalPatch, PureResult, Tensor, TensorError, TensorUtilBackend};
use core::{cmp, f64::consts::PI as PI64};

const DEFAULT_MODALITY_PERMEABILITY: f32 = 0.12;
const DEFAULT_GRAPH_PERMEABILITY: f32 = 0.08;

/// Stable contract identifier shared by Rust, Python, and WASM runtime-route clients.
pub const TOPOS_RUNTIME_ROUTE_CONTRACT_VERSION: &str = "spiraltorch.topos_runtime_route.v1";

/// Stable payload kind shared by Rust, Python, and WASM runtime-route clients.
pub const TOPOS_RUNTIME_ROUTE_KIND: &str = "spiraltorch.topos_runtime_route";

/// Crate/module that owns the runtime-route normalization and decision semantics.
pub const TOPOS_RUNTIME_ROUTE_SEMANTIC_OWNER: &str = "st-tensor::pure::topos";

/// Backend label used when the canonical Rust runtime-route semantics produced a payload.
pub const TOPOS_RUNTIME_ROUTE_SEMANTIC_BACKEND: &str = "rust";

/// Stable contract identifier shared by Rust, Python, and WASM control-signal clients.
pub const TOPOS_CONTROL_SIGNAL_CONTRACT_VERSION: &str = "spiraltorch.topos_control_signal.v2";

/// Stable payload kind shared by Rust, Python, and WASM control-signal clients.
pub const TOPOS_CONTROL_SIGNAL_KIND: &str = "spiraltorch.topos_control_signal";

/// Crate/module that owns open-topos control-signal semantics.
pub const TOPOS_CONTROL_SIGNAL_SEMANTIC_OWNER: &str = TOPOS_RUNTIME_ROUTE_SEMANTIC_OWNER;

/// Backend label used when the canonical Rust control semantics produced a payload.
pub const TOPOS_CONTROL_SIGNAL_SEMANTIC_BACKEND: &str = "rust";

/// Stable contract identifier for one control-to-optimizer application snapshot.
pub const TOPOS_OPTIMIZER_SNAPSHOT_CONTRACT_VERSION: &str =
    "spiraltorch.topos_optimizer_snapshot.v3";

/// Stable payload kind shared by native, Python, and WASM optimizer clients.
pub const TOPOS_OPTIMIZER_SNAPSHOT_KIND: &str = "spiraltorch.topos_optimizer_snapshot";

/// Crate/module that owns Topos optimizer snapshot semantics.
pub const TOPOS_OPTIMIZER_SNAPSHOT_SEMANTIC_OWNER: &str = TOPOS_CONTROL_SIGNAL_SEMANTIC_OWNER;

/// Backend label used when Rust produced the optimizer snapshot.
pub const TOPOS_OPTIMIZER_SNAPSHOT_SEMANTIC_BACKEND: &str = "rust";

/// Largest sequence that remains exact in JavaScript's integer number domain.
pub const TOPOS_OPTIMIZER_SNAPSHOT_MAX_SEQUENCE: u64 = (1_u64 << 53) - 1;

/// Number of axes in the optimizer-specific Topos gradient-bias basis.
pub const TOPOS_OPTIMIZER_GRADIENT_BIAS_BASIS_DIM: usize = 10;

/// Canonical scale-relative bias rule shared by every optimizer client.
pub const TOPOS_OPTIMIZER_GRADIENT_BIAS_RULE: &str =
    "g_biased[i]=g[i]+rms(g)*bias_scale*basis[i%10]";

/// Canonical scale-invariant outlier guard shared by every optimizer client.
pub const TOPOS_OPTIMIZER_GRADIENT_CLIP_RULE: &str =
    "g_clipped[i]=clamp(g_biased[i],-rms(g_biased)/(1-clip_scale),rms(g_biased)/(1-clip_scale))";

/// Canonical state transition used to damp abrupt gradient changes.
pub const TOPOS_OPTIMIZER_MOMENTUM_RULE: &str = "m_t=damping*m_(t-1)+(1-damping)*g_clipped";

/// Canonical normalization used by the Topos gradient-bias rule.
pub const TOPOS_OPTIMIZER_GRADIENT_BIAS_NORMALIZATION: &str = "raw_gradient_rms";

/// Canonical normalization used by the Topos gradient-clipping rule.
pub const TOPOS_OPTIMIZER_GRADIENT_CLIP_NORMALIZATION: &str = "biased_gradient_rms";

/// Stable contract identifier shared by Rust, Python, and WASM Z-space clients.
pub const TOPOS_ZSPACE_PROJECTION_CONTRACT_VERSION: &str = "spiraltorch.topos_zspace_projection.v2";

/// Stable payload kind for a Topos control signal projected into Z-space metrics.
pub const TOPOS_ZSPACE_PROJECTION_KIND: &str = "spiraltorch.topos_zspace_projection";

/// Crate/module that owns the Topos-to-Z-space projection semantics.
pub const TOPOS_ZSPACE_PROJECTION_SEMANTIC_OWNER: &str = TOPOS_CONTROL_SIGNAL_SEMANTIC_OWNER;

/// Backend label used when canonical Rust projection semantics produced a payload.
pub const TOPOS_ZSPACE_PROJECTION_SEMANTIC_BACKEND: &str = "rust";

/// Number of meaningful axes in the canonical Topos gradient basis.
pub const TOPOS_ZSPACE_PROJECTION_BASE_GRADIENT_DIM: usize = 6;

/// Stable identity for the six semantic control axes emitted by this projection.
pub const TOPOS_ZSPACE_PROJECTION_GRADIENT_BASIS: &str = "spiraltorch.topos.control_signal.axes.v1";

/// Exact coordinate rule used before client-requested truncation or zero-padding.
pub const TOPOS_ZSPACE_PROJECTION_GRADIENT_FORMULA: &str =
    "gradient=resize([openness,guard_strength,stability_hint,exploration_hint,depth_pressure,volume_pressure],gradient_dim,0)";

/// Ordered semantic channels in the canonical Topos control basis.
pub const TOPOS_ZSPACE_PROJECTION_GRADIENT_CHANNELS: [&str; 6] = [
    "openness",
    "guard_strength",
    "stability_hint",
    "exploration_hint",
    "depth_pressure",
    "volume_pressure",
];

/// Allocation guard for client-requested projection vectors.
pub const TOPOS_ZSPACE_PROJECTION_MAX_GRADIENT_DIM: usize = 4096;

fn validate_optimizer_learning_rate(label: &'static str, rate: f32) -> PureResult<()> {
    if !rate.is_finite() {
        return Err(TensorError::NonFiniteValue { label, value: rate });
    }
    if rate <= 0.0 {
        return Err(TensorError::NonPositiveLearningRate { rate });
    }
    Ok(())
}

fn validate_permeability(label: &'static str, permeability: f32) -> PureResult<()> {
    if !permeability.is_finite() {
        return Err(TensorError::NonFiniteValue {
            label,
            value: permeability,
        });
    }
    if !(0.0..=1.0).contains(&permeability) {
        return Err(TensorError::InvalidValue { label });
    }
    Ok(())
}

fn permeable_clamp(value: f32, limit: f32, permeability: f32) -> f32 {
    if !value.is_finite() {
        return 0.0;
    }
    if limit <= 0.0 {
        return 0.0;
    }
    let magnitude = value.abs();
    if magnitude <= limit {
        return value;
    }
    let sign = value.signum();
    if permeability <= f32::EPSILON {
        return sign * limit;
    }
    let headroom = limit * permeability;
    let softened = 1.0 - (-((magnitude - limit) / (headroom + limit))).exp();
    sign * (limit + headroom * softened.min(1.0))
}

pub(crate) fn porous_mix(value: f32, saturation: f32, porosity: f32) -> f32 {
    if !value.is_finite() {
        return 0.0;
    }
    if saturation <= 0.0 {
        return 0.0;
    }
    let limit = saturation.abs();
    let magnitude = value.abs();
    if magnitude <= limit {
        return value;
    }
    if porosity <= f32::EPSILON {
        return value.signum() * limit;
    }
    let bleed = (magnitude - limit) / (magnitude + limit);
    let absorb = (porosity * 0.25).min(1.0);
    let softened = limit * (1.0 - absorb * bleed.min(1.0)).max(0.0);
    value.signum() * softened
}

fn porous_mix_slope(value: f32, saturation: f32, porosity: f32) -> f32 {
    if !value.is_finite() || saturation <= 0.0 {
        return 0.0;
    }
    let limit = saturation.abs();
    let magnitude = value.abs();
    if magnitude <= limit {
        return 1.0;
    }
    if porosity <= f32::EPSILON {
        return 0.0;
    }
    let absorb = (porosity * 0.25).min(1.0);
    let denominator = magnitude + limit;
    if denominator <= f32::EPSILON {
        return 0.0;
    }
    -2.0 * limit * limit * absorb / (denominator * denominator)
}

fn finite_or(value: f32, default: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        default
    }
}

fn finite_non_negative(value: f32) -> f32 {
    finite_or(value, 0.0).max(0.0)
}

fn tensor_util_backend_label(backend: TensorUtilBackend) -> &'static str {
    match backend {
        TensorUtilBackend::Auto => "auto",
        TensorUtilBackend::Cpu => "cpu",
        TensorUtilBackend::GpuWgpu => "wgpu",
    }
}

fn square_root_floor(value: usize) -> usize {
    let mut root = (value as f64).sqrt() as usize;
    while root
        .checked_add(1)
        .and_then(|next| next.checked_mul(next))
        .is_some_and(|square| square <= value)
    {
        root += 1;
    }
    while root.checked_mul(root).is_none_or(|square| square > value) {
        root -= 1;
    }
    root
}

fn checked_tensor_volume(rows: usize, cols: usize) -> PureResult<usize> {
    rows.checked_mul(cols)
        .ok_or(TensorError::InvalidDimensions { rows, cols })
}

/// Runtime pressure and openness signal emitted by an open-cartesian topos.
///
/// The signal is intentionally scalar and copyable so training loops, Python
/// facades, WASM demos, and hosted-LLM inference contexts can all observe the
/// same guard state without owning the guarded tensors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ToposControlSignal {
    curvature: f32,
    tolerance: f32,
    saturation: f32,
    porosity: f32,
    max_depth: usize,
    max_volume: usize,
    observed_depth: usize,
    visited_volume: usize,
    remaining_volume: usize,
    depth_pressure: f32,
    volume_pressure: f32,
    closure_pressure: f32,
    openness: f32,
    guard_strength: f32,
    stability_hint: f32,
    exploration_hint: f32,
    learning_rate_scale: f32,
    temperature_scale: f32,
    regularization_scale: f32,
    step_damping: f32,
    sampling_focus: f32,
}

/// External topology and traversal inputs used to construct a canonical control signal.
///
/// Unlike the normalized runtime-profile ingress, these values define the topology itself.
/// Invalid or non-finite geometry is rejected rather than silently rewritten.
#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct ToposControlSignalInput {
    /// Strictly negative hyperbolic curvature.
    pub curvature: f32,
    /// Strictly positive numerical tolerance.
    pub tolerance: f32,
    /// Strictly positive saturation limit.
    pub saturation: f32,
    /// Topology permeability in `[0, 1]`.
    pub porosity: f32,
    /// Non-zero traversal-depth horizon.
    pub max_depth: usize,
    /// Non-zero visited-volume horizon.
    pub max_volume: usize,
    /// Observed traversal depth.
    pub observed_depth: usize,
    /// Observed visited volume.
    pub visited_volume: usize,
}

impl Default for ToposControlSignalInput {
    fn default() -> Self {
        Self {
            curvature: -1.0,
            tolerance: 1e-3,
            saturation: 1.0,
            porosity: 0.2,
            max_depth: 64,
            max_volume: 512,
            observed_depth: 0,
            visited_volume: 0,
        }
    }
}

/// Client representation options for the canonical Topos-to-Z-space projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct ToposZSpaceProjectionOptions {
    /// Output width. Dimensions above the six semantic axes are explicitly zero-filled.
    pub gradient_dim: usize,
}

impl Default for ToposZSpaceProjectionOptions {
    fn default() -> Self {
        Self {
            gradient_dim: TOPOS_ZSPACE_PROJECTION_BASE_GRADIENT_DIM,
        }
    }
}

impl ToposZSpaceProjectionOptions {
    fn validated(self) -> PureResult<Self> {
        if !(1..=TOPOS_ZSPACE_PROJECTION_MAX_GRADIENT_DIM).contains(&self.gradient_dim) {
            return Err(TensorError::InvalidValue {
                label: "topos_zspace_gradient_dim must be in 1..=4096",
            });
        }
        Ok(self)
    }
}

/// Canonical Z-space metrics projected from one open-topos control signal.
#[derive(Clone, Debug, PartialEq)]
pub struct ToposZSpaceProjection {
    speed: f32,
    memory: f32,
    stability: f32,
    drs: f32,
    frac: f32,
    gradient: Vec<f32>,
}

/// External optimizer hints admitted by the canonical Rust control contract.
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct ToposTrainingHintsInput {
    pub learning_rate_scale: Option<f32>,
    pub regularization_scale: Option<f32>,
    pub step_damping: Option<f32>,
    pub gradient_bias_scale: Option<f32>,
    pub clip_scale: Option<f32>,
    pub momentum_damping: Option<f32>,
}

/// Named optimizer controls projected from an open-topos pressure signal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ToposTrainingHints {
    learning_rate_scale: f32,
    regularization_scale: f32,
    step_damping: f32,
    gradient_bias_scale: f32,
    clip_scale: f32,
    momentum_damping: f32,
}

/// Gain-applied optimizer controls ready to be consumed by training loops.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ToposTrainingPlan {
    gain: f32,
    learning_rate_scale: f32,
    regularization_scale: f32,
    step_damping: f32,
    gradient_bias_scale: f32,
    clip_scale: f32,
    momentum_damping: f32,
    raw_rate_scale: f32,
    rate_scale: f32,
    effective_gradient_bias_scale: f32,
    effective_gradient_clip_scale: f32,
    effective_momentum_damping: f32,
}

impl ToposTrainingHints {
    /// Normalizes external optimizer hints against a neutral training posture.
    pub fn from_input(input: ToposTrainingHintsInput) -> Self {
        Self {
            learning_rate_scale: 1.0,
            regularization_scale: 1.0,
            step_damping: 0.0,
            gradient_bias_scale: 0.0,
            clip_scale: 1.0,
            momentum_damping: 0.0,
        }
        .with_input(input)
    }

    /// Applies a partial external hint override while preserving canonical bounds.
    pub fn with_input(&self, input: ToposTrainingHintsInput) -> Self {
        Self {
            learning_rate_scale: finite_or(
                input
                    .learning_rate_scale
                    .unwrap_or(self.learning_rate_scale),
                self.learning_rate_scale,
            )
            .clamp(0.1, 1.25),
            regularization_scale: finite_or(
                input
                    .regularization_scale
                    .unwrap_or(self.regularization_scale),
                self.regularization_scale,
            )
            .clamp(0.5, 2.0),
            step_damping: finite_or(
                input.step_damping.unwrap_or(self.step_damping),
                self.step_damping,
            )
            .clamp(0.0, 1.0),
            gradient_bias_scale: finite_or(
                input
                    .gradient_bias_scale
                    .unwrap_or(self.gradient_bias_scale),
                self.gradient_bias_scale,
            )
            .clamp(0.0, 0.35),
            clip_scale: finite_or(input.clip_scale.unwrap_or(self.clip_scale), self.clip_scale)
                .clamp(0.25, 1.0),
            momentum_damping: finite_or(
                input.momentum_damping.unwrap_or(self.momentum_damping),
                self.momentum_damping,
            )
            .clamp(0.0, 0.85),
        }
    }

    pub fn learning_rate_scale(&self) -> f32 {
        self.learning_rate_scale
    }

    pub fn regularization_scale(&self) -> f32 {
        self.regularization_scale
    }

    pub fn step_damping(&self) -> f32 {
        self.step_damping
    }

    pub fn gradient_bias_scale(&self) -> f32 {
        self.gradient_bias_scale
    }

    pub fn clip_scale(&self) -> f32 {
        self.clip_scale
    }

    pub fn momentum_damping(&self) -> f32 {
        self.momentum_damping
    }

    pub fn vector(&self) -> [f32; 6] {
        [
            self.learning_rate_scale,
            self.regularization_scale,
            self.step_damping,
            self.gradient_bias_scale,
            self.clip_scale,
            self.momentum_damping,
        ]
    }

    /// Returns the stable transport payload used by language bindings.
    pub fn payload(&self) -> ToposTrainingHintsPayload {
        ToposTrainingHintsPayload {
            learning_rate_scale: self.learning_rate_scale,
            regularization_scale: self.regularization_scale,
            step_damping: self.step_damping,
            gradient_bias_scale: self.gradient_bias_scale,
            clip_scale: self.clip_scale,
            momentum_damping: self.momentum_damping,
            vector: self.vector(),
        }
    }

    /// Applies a runtime gain and returns concrete optimizer controls.
    pub fn plan(&self, gain: f32) -> ToposTrainingPlan {
        let gain = finite_non_negative(gain);
        let raw_rate_scale = self.learning_rate_scale.clamp(0.01, 2.0);
        let rate_scale = (1.0 + gain * (raw_rate_scale - 1.0)).clamp(0.01, 2.0);
        let effective_gradient_bias_scale = (self.gradient_bias_scale * gain).clamp(0.0, 0.35);
        let effective_gradient_clip_scale = (1.0 + gain * (self.clip_scale - 1.0)).clamp(0.25, 1.0);
        let effective_momentum_damping = (self.momentum_damping * gain).clamp(0.0, 0.85);
        ToposTrainingPlan {
            gain,
            learning_rate_scale: self.learning_rate_scale,
            regularization_scale: self.regularization_scale,
            step_damping: self.step_damping,
            gradient_bias_scale: self.gradient_bias_scale,
            clip_scale: self.clip_scale,
            momentum_damping: self.momentum_damping,
            raw_rate_scale,
            rate_scale,
            effective_gradient_bias_scale,
            effective_gradient_clip_scale,
            effective_momentum_damping,
        }
    }
}

impl ToposTrainingPlan {
    pub fn gain(&self) -> f32 {
        self.gain
    }

    pub fn learning_rate_scale(&self) -> f32 {
        self.learning_rate_scale
    }

    pub fn regularization_scale(&self) -> f32 {
        self.regularization_scale
    }

    pub fn step_damping(&self) -> f32 {
        self.step_damping
    }

    pub fn gradient_bias_scale(&self) -> f32 {
        self.gradient_bias_scale
    }

    pub fn clip_scale(&self) -> f32 {
        self.clip_scale
    }

    pub fn momentum_damping(&self) -> f32 {
        self.momentum_damping
    }

    pub fn raw_rate_scale(&self) -> f32 {
        self.raw_rate_scale
    }

    pub fn rate_scale(&self) -> f32 {
        self.rate_scale
    }

    pub fn effective_gradient_bias_scale(&self) -> f32 {
        self.effective_gradient_bias_scale
    }

    pub fn effective_gradient_clip_scale(&self) -> f32 {
        self.effective_gradient_clip_scale
    }

    pub fn effective_momentum_damping(&self) -> f32 {
        self.effective_momentum_damping
    }

    pub fn vector(&self) -> [f32; 6] {
        [
            self.rate_scale,
            self.regularization_scale,
            self.step_damping,
            self.effective_gradient_bias_scale,
            self.effective_gradient_clip_scale,
            self.effective_momentum_damping,
        ]
    }

    /// Returns the stable transport payload used by language bindings.
    pub fn payload(&self) -> ToposTrainingPlanPayload {
        ToposTrainingPlanPayload {
            gain: self.gain,
            learning_rate_scale: self.learning_rate_scale,
            regularization_scale: self.regularization_scale,
            step_damping: self.step_damping,
            gradient_bias_scale: self.gradient_bias_scale,
            clip_scale: self.clip_scale,
            momentum_damping: self.momentum_damping,
            raw_rate_scale: self.raw_rate_scale,
            rate_scale: self.rate_scale,
            effective_gradient_bias_scale: self.effective_gradient_bias_scale,
            effective_gradient_clip_scale: self.effective_gradient_clip_scale,
            effective_momentum_damping: self.effective_momentum_damping,
            vector: self.vector(),
        }
    }
}

/// Builds the shared Topos basis used to bias optimizer gradients.
///
/// The function accepts `f64` so higher-level Rust runtimes can reuse the exact
/// semantics without narrowing their telemetry before the final update.
#[allow(clippy::too_many_arguments)]
pub fn topos_optimizer_gradient_bias_basis(
    closure_pressure: f64,
    volume_pressure: f64,
    depth_pressure: f64,
    guard_strength: f64,
    step_damping: f64,
    sampling_focus: f64,
    learning_rate_hint: f64,
    regularization_scale: f64,
    openness: f64,
    exploration_hint: f64,
) -> [f64; TOPOS_OPTIMIZER_GRADIENT_BIAS_BASIS_DIM] {
    [
        closure_pressure - 0.5,
        volume_pressure - 0.5,
        depth_pressure - 0.5,
        guard_strength - 0.5,
        step_damping - 0.5,
        sampling_focus - 0.5,
        1.0 - learning_rate_hint,
        regularization_scale - 1.0,
        0.5 - openness,
        0.5 - exploration_hint,
    ]
}

/// Computes the scale-relative bias amplitude used by every Topos optimizer path.
pub fn topos_optimizer_gradient_bias_amplitude(
    raw_gradient_rms: f64,
    gradient_bias_scale: f64,
) -> f64 {
    raw_gradient_rms * gradient_bias_scale
}

/// Resolves the RMS-relative clipping threshold after gradient bias is applied.
///
/// A scale of `1` is an exact no-op. Lower values progressively guard the
/// distribution tail without introducing an absolute, unit-dependent bound.
pub fn topos_optimizer_gradient_clip_threshold(
    biased_gradient_rms: f64,
    gradient_clip_scale: f64,
) -> Option<f64> {
    (gradient_clip_scale < 1.0).then(|| biased_gradient_rms / (1.0 - gradient_clip_scale))
}

/// Rust-owned gradient-state controls configured on Amega optimizer tapes.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct ToposOptimizerStateControl {
    gradient_bias_rule: &'static str,
    gradient_bias_normalization: &'static str,
    effective_gradient_bias_scale: f32,
    gradient_bias_basis_dim: usize,
    gradient_bias_basis: [f32; TOPOS_OPTIMIZER_GRADIENT_BIAS_BASIS_DIM],
    gradient_clip_rule: &'static str,
    gradient_clip_normalization: &'static str,
    effective_gradient_clip_scale: f32,
    momentum_rule: &'static str,
    effective_momentum_damping: f32,
}

/// Lossless transport form of [`ToposOptimizerStateControl`].
///
/// The live control keeps rule labels as static Rust strings. Checkpoints only
/// transport the validated numeric degrees of freedom and reconstruct those
/// labels through [`ToposOptimizerStateControl::new`].
#[derive(Clone, Copy, Debug, serde::Deserialize, PartialEq, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct ToposOptimizerStateCheckpoint {
    pub gradient_bias_scale: f32,
    pub gradient_bias_basis: [f32; TOPOS_OPTIMIZER_GRADIENT_BIAS_BASIS_DIM],
    pub gradient_clip_scale: f32,
    pub momentum_damping: f32,
}

impl ToposOptimizerStateCheckpoint {
    pub fn validate(self) -> PureResult<()> {
        self.into_control().map(|_| ())
    }

    pub fn into_control(self) -> PureResult<ToposOptimizerStateControl> {
        ToposOptimizerStateControl::new(
            self.gradient_bias_scale,
            self.gradient_bias_basis,
            self.gradient_clip_scale,
            self.momentum_damping,
        )
    }
}

impl From<ToposOptimizerStateControl> for ToposOptimizerStateCheckpoint {
    fn from(control: ToposOptimizerStateControl) -> Self {
        Self {
            gradient_bias_scale: control.gradient_bias_scale(),
            gradient_bias_basis: *control.gradient_bias_basis(),
            gradient_clip_scale: control.gradient_clip_scale(),
            momentum_damping: control.momentum_damping(),
        }
    }
}

impl Default for ToposOptimizerStateControl {
    fn default() -> Self {
        Self::neutral()
    }
}

impl ToposOptimizerStateControl {
    /// Returns a control that preserves the historical stateless gradient path.
    pub const fn neutral() -> Self {
        Self {
            gradient_bias_rule: TOPOS_OPTIMIZER_GRADIENT_BIAS_RULE,
            gradient_bias_normalization: TOPOS_OPTIMIZER_GRADIENT_BIAS_NORMALIZATION,
            effective_gradient_bias_scale: 0.0,
            gradient_bias_basis_dim: TOPOS_OPTIMIZER_GRADIENT_BIAS_BASIS_DIM,
            gradient_bias_basis: [0.0; TOPOS_OPTIMIZER_GRADIENT_BIAS_BASIS_DIM],
            gradient_clip_rule: TOPOS_OPTIMIZER_GRADIENT_CLIP_RULE,
            gradient_clip_normalization: TOPOS_OPTIMIZER_GRADIENT_CLIP_NORMALIZATION,
            effective_gradient_clip_scale: 1.0,
            momentum_rule: TOPOS_OPTIMIZER_MOMENTUM_RULE,
            effective_momentum_damping: 0.0,
        }
    }

    /// Validates an externally transported control before optimizer state adopts it.
    pub fn new(
        effective_gradient_bias_scale: f32,
        gradient_bias_basis: [f32; TOPOS_OPTIMIZER_GRADIENT_BIAS_BASIS_DIM],
        effective_gradient_clip_scale: f32,
        effective_momentum_damping: f32,
    ) -> PureResult<Self> {
        if !effective_gradient_bias_scale.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "topos_optimizer_gradient_bias_scale",
                value: effective_gradient_bias_scale,
            });
        }
        if !(0.0..=0.35).contains(&effective_gradient_bias_scale) {
            return Err(TensorError::InvalidValue {
                label: "topos_optimizer_gradient_bias_scale",
            });
        }
        if !effective_gradient_clip_scale.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "topos_optimizer_gradient_clip_scale",
                value: effective_gradient_clip_scale,
            });
        }
        if !(0.25..=1.0).contains(&effective_gradient_clip_scale) {
            return Err(TensorError::InvalidValue {
                label: "topos_optimizer_gradient_clip_scale",
            });
        }
        if !effective_momentum_damping.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "topos_optimizer_momentum_damping",
                value: effective_momentum_damping,
            });
        }
        if !(0.0..=0.85).contains(&effective_momentum_damping) {
            return Err(TensorError::InvalidValue {
                label: "topos_optimizer_momentum_damping",
            });
        }
        for value in gradient_bias_basis {
            if !value.is_finite() {
                return Err(TensorError::NonFiniteValue {
                    label: "topos_optimizer_gradient_bias_basis",
                    value,
                });
            }
            if !(-1.0..=1.0).contains(&value) {
                return Err(TensorError::InvalidValue {
                    label: "topos_optimizer_gradient_bias_basis",
                });
            }
        }
        Ok(Self {
            gradient_bias_rule: TOPOS_OPTIMIZER_GRADIENT_BIAS_RULE,
            gradient_bias_normalization: TOPOS_OPTIMIZER_GRADIENT_BIAS_NORMALIZATION,
            effective_gradient_bias_scale,
            gradient_bias_basis_dim: TOPOS_OPTIMIZER_GRADIENT_BIAS_BASIS_DIM,
            gradient_bias_basis,
            gradient_clip_rule: TOPOS_OPTIMIZER_GRADIENT_CLIP_RULE,
            gradient_clip_normalization: TOPOS_OPTIMIZER_GRADIENT_CLIP_NORMALIZATION,
            effective_gradient_clip_scale,
            momentum_rule: TOPOS_OPTIMIZER_MOMENTUM_RULE,
            effective_momentum_damping,
        })
    }

    pub fn gradient_bias_scale(&self) -> f32 {
        self.effective_gradient_bias_scale
    }

    pub fn gradient_bias_rule(&self) -> &'static str {
        self.gradient_bias_rule
    }

    pub fn gradient_bias_normalization(&self) -> &'static str {
        self.gradient_bias_normalization
    }

    pub fn gradient_bias_basis_dim(&self) -> usize {
        self.gradient_bias_basis_dim
    }

    pub fn gradient_bias_basis(&self) -> &[f32; TOPOS_OPTIMIZER_GRADIENT_BIAS_BASIS_DIM] {
        &self.gradient_bias_basis
    }

    pub fn gradient_clip_scale(&self) -> f32 {
        self.effective_gradient_clip_scale
    }

    pub fn gradient_clip_rule(&self) -> &'static str {
        self.gradient_clip_rule
    }

    pub fn gradient_clip_normalization(&self) -> &'static str {
        self.gradient_clip_normalization
    }

    pub fn momentum_damping(&self) -> f32 {
        self.effective_momentum_damping
    }

    pub fn momentum_rule(&self) -> &'static str {
        self.momentum_rule
    }

    pub fn is_neutral(&self) -> bool {
        self.effective_gradient_bias_scale == 0.0
            && self.effective_gradient_clip_scale == 1.0
            && self.effective_momentum_damping == 0.0
    }

    /// Applies the canonical RMS-relative bias, clipping, and momentum transition.
    pub fn gradient_step(
        &self,
        raw_gradient: &[f32],
        previous_momentum: &[f32],
    ) -> PureResult<ToposOptimizerGradientStep> {
        if raw_gradient.len() != previous_momentum.len() {
            return Err(TensorError::DataLength {
                expected: raw_gradient.len(),
                got: previous_momentum.len(),
            });
        }
        let mut sum_squares = 0.0f64;
        for &value in raw_gradient {
            if !value.is_finite() {
                return Err(TensorError::NonFiniteValue {
                    label: "topos_optimizer_raw_gradient",
                    value,
                });
            }
            let value = value as f64;
            sum_squares += value * value;
        }
        for &value in previous_momentum {
            if !value.is_finite() {
                return Err(TensorError::NonFiniteValue {
                    label: "topos_optimizer_previous_momentum",
                    value,
                });
            }
        }
        let raw_gradient_rms = if raw_gradient.is_empty() {
            0.0
        } else {
            (sum_squares / raw_gradient.len() as f64).sqrt() as f32
        };
        if !raw_gradient_rms.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "topos_optimizer_raw_gradient_rms",
                value: raw_gradient_rms,
            });
        }
        let gradient_bias_amplitude = topos_optimizer_gradient_bias_amplitude(
            raw_gradient_rms as f64,
            self.effective_gradient_bias_scale as f64,
        ) as f32;
        if !gradient_bias_amplitude.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "topos_optimizer_gradient_bias_amplitude",
                value: gradient_bias_amplitude,
            });
        }

        let mut biased_gradient = Vec::with_capacity(raw_gradient.len());
        let mut biased_sum_squares = 0.0f64;
        for (index, &raw) in raw_gradient.iter().enumerate() {
            let bias = gradient_bias_amplitude
                * self.gradient_bias_basis[index % TOPOS_OPTIMIZER_GRADIENT_BIAS_BASIS_DIM];
            let biased = raw + bias;
            if !biased.is_finite() {
                return Err(TensorError::NonFiniteValue {
                    label: "topos_optimizer_biased_gradient",
                    value: biased,
                });
            }
            biased_sum_squares += (biased as f64) * (biased as f64);
            biased_gradient.push(biased);
        }
        let biased_gradient_rms = if biased_gradient.is_empty() {
            0.0
        } else {
            (biased_sum_squares / biased_gradient.len() as f64).sqrt()
        };
        if !biased_gradient_rms.is_finite() {
            return Err(TensorError::InvalidValue {
                label: "topos_optimizer_biased_gradient_rms",
            });
        }
        let gradient_clip_threshold = topos_optimizer_gradient_clip_threshold(
            biased_gradient_rms,
            self.effective_gradient_clip_scale as f64,
        );
        if gradient_clip_threshold.is_some_and(|threshold| !threshold.is_finite()) {
            return Err(TensorError::InvalidValue {
                label: "topos_optimizer_gradient_clip_threshold",
            });
        }
        let mut clipped_values = 0;
        let mut clipped_gradient = Vec::with_capacity(biased_gradient.len());
        for biased in biased_gradient {
            let clipped = gradient_clip_threshold.map_or(biased, |threshold| {
                (biased as f64).clamp(-threshold, threshold) as f32
            });
            if clipped != biased {
                clipped_values += 1;
            }
            clipped_gradient.push(clipped);
        }

        let damping = self.effective_momentum_damping;
        let incoming = 1.0 - damping;
        let mut next_momentum = Vec::with_capacity(raw_gradient.len());
        for (&clipped, &previous) in clipped_gradient.iter().zip(previous_momentum.iter()) {
            let momentum = damping * previous + incoming * clipped;
            if !momentum.is_finite() {
                return Err(TensorError::NonFiniteValue {
                    label: "topos_optimizer_next_momentum",
                    value: momentum,
                });
            }
            next_momentum.push(momentum);
        }
        Ok(ToposOptimizerGradientStep {
            raw_gradient_rms,
            gradient_bias_amplitude,
            biased_gradient_rms,
            gradient_clip_threshold,
            clipped_values,
            next_momentum,
        })
    }
}

/// One validated gradient-state transition ready for a tape update.
#[derive(Clone, Debug, PartialEq)]
pub struct ToposOptimizerGradientStep {
    raw_gradient_rms: f32,
    gradient_bias_amplitude: f32,
    biased_gradient_rms: f64,
    gradient_clip_threshold: Option<f64>,
    clipped_values: usize,
    next_momentum: Vec<f32>,
}

impl ToposOptimizerGradientStep {
    pub fn raw_gradient_rms(&self) -> f32 {
        self.raw_gradient_rms
    }

    pub fn gradient_bias_amplitude(&self) -> f32 {
        self.gradient_bias_amplitude
    }

    pub fn biased_gradient_rms(&self) -> f64 {
        self.biased_gradient_rms
    }

    pub fn gradient_clip_threshold(&self) -> Option<f64> {
        self.gradient_clip_threshold
    }

    pub fn clipped_values(&self) -> usize {
        self.clipped_values
    }

    pub fn applied_gradient(&self) -> &[f32] {
        &self.next_momentum
    }

    pub fn next_momentum(&self) -> &[f32] {
        &self.next_momentum
    }

    pub fn into_next_momentum(self) -> Vec<f32> {
        self.next_momentum
    }
}

/// Named hosted-inference controls projected from an open-topos pressure signal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ToposInferenceHints {
    temperature_scale: f32,
    top_p_scale: f32,
    sampling_focus: f32,
    frequency_penalty_bias: f32,
    presence_penalty_bias: f32,
    context_weight: f32,
}

/// External inference hints admitted by the canonical Rust control contract.
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct ToposInferenceHintsInput {
    pub temperature_scale: Option<f32>,
    pub top_p_scale: Option<f32>,
    pub sampling_focus: Option<f32>,
    pub frequency_penalty_bias: Option<f32>,
    pub presence_penalty_bias: Option<f32>,
    pub context_weight: Option<f32>,
}

/// Sampling controls and provider bounds used to build a canonical inference plan.
#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct ToposInferencePlanOptions {
    pub gain: f32,
    pub base_temperature: f32,
    pub base_top_p: f32,
    pub min_temperature: f32,
    pub max_temperature: f32,
    pub min_top_p: f32,
    pub max_top_p: f32,
    pub base_frequency_penalty: f32,
    pub base_presence_penalty: f32,
}

impl Default for ToposInferencePlanOptions {
    fn default() -> Self {
        Self {
            gain: 1.0,
            base_temperature: 1.0,
            base_top_p: 1.0,
            min_temperature: 0.0,
            max_temperature: 2.0,
            min_top_p: 0.05,
            max_top_p: 1.0,
            base_frequency_penalty: 0.0,
            base_presence_penalty: 0.0,
        }
    }
}

impl ToposInferencePlanOptions {
    /// Validates provider bounds and normalizes non-finite runtime controls.
    pub fn validated(self) -> PureResult<Self> {
        for (label, value) in [
            ("topos_inference_min_temperature", self.min_temperature),
            ("topos_inference_max_temperature", self.max_temperature),
            ("topos_inference_min_top_p", self.min_top_p),
            ("topos_inference_max_top_p", self.max_top_p),
        ] {
            if !value.is_finite() {
                return Err(TensorError::NonFiniteValue { label, value });
            }
        }
        if self.min_temperature < 0.0 || self.max_temperature < self.min_temperature {
            return Err(TensorError::InvalidValue {
                label: "topos_inference_temperature_bounds",
            });
        }
        if self.min_top_p < 0.0 || self.max_top_p > 1.0 || self.max_top_p < self.min_top_p {
            return Err(TensorError::InvalidValue {
                label: "topos_inference_top_p_bounds",
            });
        }
        Ok(Self {
            gain: finite_non_negative(self.gain),
            base_temperature: finite_or(self.base_temperature, 1.0),
            base_top_p: finite_or(self.base_top_p, 1.0),
            min_temperature: self.min_temperature,
            max_temperature: self.max_temperature,
            min_top_p: self.min_top_p,
            max_top_p: self.max_top_p,
            base_frequency_penalty: finite_or(self.base_frequency_penalty, 0.0),
            base_presence_penalty: finite_or(self.base_presence_penalty, 0.0),
        })
    }
}

/// Gain and inference options used to derive the complete control payload.
#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct ToposControlPlanOptions {
    pub training_gain: f32,
    pub inference: ToposInferencePlanOptions,
}

impl Default for ToposControlPlanOptions {
    fn default() -> Self {
        Self {
            training_gain: 1.0,
            inference: ToposInferencePlanOptions::default(),
        }
    }
}

/// Hosted-inference request controls after applying base sampling parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ToposInferencePlan {
    gain: f32,
    temperature: f32,
    top_p: f32,
    frequency_penalty: f32,
    presence_penalty: f32,
    context_weight: f32,
    temperature_scale: f32,
    top_p_scale: f32,
    sampling_focus: f32,
}

/// Joint learning/inference profile projected from one open-topos signal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ToposRuntimeProfile {
    training_gain: f32,
    inference_gain: f32,
    closure_risk: f32,
    exploration_budget: f32,
    control_energy: f32,
    training_rate_scale: f32,
    training_gradient_bias_scale: f32,
    inference_temperature: f32,
    inference_top_p: f32,
    inference_context_weight: f32,
    learning_inference_balance: f32,
}

/// External inputs used to construct a normalized [`ToposRuntimeProfile`].
///
/// The defaults describe a neutral runtime profile. [`ToposRuntimeProfile::from_input`]
/// rewrites non-finite values to these defaults and clamps every bounded component, making
/// this the shared profile-ingress contract for native, Python, and WASM callers.
#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct ToposRuntimeProfileInput {
    /// Non-negative gain applied to learning controls.
    pub training_gain: f32,
    /// Non-negative gain applied to inference controls.
    pub inference_gain: f32,
    /// Estimated closure risk in `[0, 1]`.
    pub closure_risk: f32,
    /// Available exploration budget in `[0, 1]`.
    pub exploration_budget: f32,
    /// Aggregate control energy in `[0, 1]`.
    pub control_energy: f32,
    /// Effective training-rate scale, clamped to `[0.01, 2]`.
    pub training_rate_scale: f32,
    /// Effective gradient-bias scale, clamped to `[0, 0.35]`.
    pub training_gradient_bias_scale: f32,
    /// Hosted-inference temperature, clamped to `[0, 2]`.
    pub inference_temperature: f32,
    /// Hosted-inference nucleus probability, clamped to `[0.05, 1]`.
    pub inference_top_p: f32,
    /// Context contribution weight, clamped to `[0.25, 1.25]`.
    pub inference_context_weight: f32,
    /// Training-to-inference balance, clamped to `[0, 2]`.
    pub learning_inference_balance: f32,
}

impl Default for ToposRuntimeProfileInput {
    fn default() -> Self {
        Self {
            training_gain: 1.0,
            inference_gain: 1.0,
            closure_risk: 0.0,
            exploration_budget: 0.0,
            control_energy: 0.0,
            training_rate_scale: 1.0,
            training_gradient_bias_scale: 0.0,
            inference_temperature: 1.0,
            inference_top_p: 1.0,
            inference_context_weight: 1.0,
            learning_inference_balance: 1.0,
        }
    }
}

/// Named runtime posture selected from a joint learning/inference profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToposRuntimeMode {
    Balanced,
    Guarded,
    Exploratory,
    Contextual,
    TrainingFirst,
    InferenceFirst,
}

/// Component scores used to choose a [`ToposRuntimeMode`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ToposRuntimeRouteScores {
    training_score: f32,
    inference_score: f32,
    guard_score: f32,
    exploration_score: f32,
    context_score: f32,
}

/// Decision layer that names how a topos profile should steer runtime behavior.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ToposRuntimeRoute {
    profile: ToposRuntimeProfile,
    mode: ToposRuntimeMode,
    score: f32,
    scores: ToposRuntimeRouteScores,
}

/// Stable serialized view of a normalized [`ToposRuntimeProfile`].
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct ToposRuntimeProfilePayload {
    pub training_gain: f32,
    pub inference_gain: f32,
    pub closure_risk: f32,
    pub exploration_budget: f32,
    pub control_energy: f32,
    pub training_rate_scale: f32,
    pub training_gradient_bias_scale: f32,
    pub inference_temperature: f32,
    pub inference_top_p: f32,
    pub inference_context_weight: f32,
    pub learning_inference_balance: f32,
    pub vector: [f32; 6],
}

/// Stable serialized view of optimizer hints.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct ToposTrainingHintsPayload {
    pub learning_rate_scale: f32,
    pub regularization_scale: f32,
    pub step_damping: f32,
    pub gradient_bias_scale: f32,
    pub clip_scale: f32,
    pub momentum_damping: f32,
    pub vector: [f32; 6],
}

/// Stable serialized view of gain-applied optimizer controls.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct ToposTrainingPlanPayload {
    pub gain: f32,
    pub learning_rate_scale: f32,
    pub regularization_scale: f32,
    pub step_damping: f32,
    pub gradient_bias_scale: f32,
    pub clip_scale: f32,
    pub momentum_damping: f32,
    pub raw_rate_scale: f32,
    pub rate_scale: f32,
    pub effective_gradient_bias_scale: f32,
    pub effective_gradient_clip_scale: f32,
    pub effective_momentum_damping: f32,
    pub vector: [f32; 6],
}

/// Stable serialized view of hosted-inference hints.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct ToposInferenceHintsPayload {
    pub temperature_scale: f32,
    pub top_p_scale: f32,
    pub sampling_focus: f32,
    pub frequency_penalty_bias: f32,
    pub presence_penalty_bias: f32,
    pub context_weight: f32,
    pub vector: [f32; 6],
}

/// Stable serialized view of concrete hosted-inference controls.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct ToposInferencePlanPayload {
    pub gain: f32,
    pub temperature: f32,
    pub top_p: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub context_weight: f32,
    pub temperature_scale: f32,
    pub top_p_scale: f32,
    pub sampling_focus: f32,
    pub vector: [f32; 6],
}

/// Stable serialized view of runtime-route component scores.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct ToposRuntimeRouteScoresPayload {
    pub training: f32,
    pub inference: f32,
    pub guard: f32,
    pub exploration: f32,
    pub context: f32,
    pub vector: [f32; 5],
}

/// Canonical runtime-route payload shared by every language binding.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct ToposRuntimeRoutePayload {
    pub kind: &'static str,
    pub contract_version: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub mode: &'static str,
    pub mode_id: usize,
    pub score: f32,
    pub score_key: &'static str,
    pub learning_action: &'static str,
    pub inference_action: &'static str,
    pub scores: ToposRuntimeRouteScoresPayload,
    pub runtime_profile: ToposRuntimeProfilePayload,
}

/// Canonical Topos-to-Z-space projection payload shared by every language binding.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct ToposZSpaceProjectionPayload {
    pub kind: &'static str,
    pub contract_version: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub gradient_basis: &'static str,
    pub gradient_formula: &'static str,
    pub gradient_channels: [&'static str; TOPOS_ZSPACE_PROJECTION_BASE_GRADIENT_DIM],
    pub gradient_dim: usize,
    pub base_gradient_dim: usize,
    pub speed: f32,
    pub memory: f32,
    pub stability: f32,
    pub drs: f32,
    pub frac: f32,
    pub gradient: Vec<f32>,
    pub vector: [f32; 5],
}

/// Canonical control bundle shared by Rust, Python, and WASM clients.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct ToposControlSignalPayload {
    pub kind: &'static str,
    pub contract_version: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub curvature: f32,
    pub tolerance: f32,
    pub saturation: f32,
    pub porosity: f32,
    pub max_depth: usize,
    pub max_volume: usize,
    pub observed_depth: usize,
    pub visited_volume: usize,
    pub remaining_volume: usize,
    pub depth_pressure: f32,
    pub volume_pressure: f32,
    pub closure_pressure: f32,
    pub openness: f32,
    pub guard_strength: f32,
    pub stability_hint: f32,
    pub exploration_hint: f32,
    pub learning_rate_scale: f32,
    pub temperature_scale: f32,
    pub regularization_scale: f32,
    pub step_damping: f32,
    pub sampling_focus: f32,
    pub runtime_hints: [f32; 5],
    pub gradient: [f32; 6],
    pub training_hints: ToposTrainingHintsPayload,
    pub training_plan: ToposTrainingPlanPayload,
    pub inference_hints: ToposInferenceHintsPayload,
    pub inference_plan: ToposInferencePlanPayload,
    pub runtime_profile: ToposRuntimeProfilePayload,
    pub runtime_route: ToposRuntimeRoutePayload,
}

/// Learning-rate and gradient-state mutation prescribed for one optimizer step.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct ToposOptimizerApplicationPayload {
    pub scope: &'static str,
    pub control_path: &'static str,
    pub input_hyper_learning_rate: f32,
    pub input_real_learning_rate: f32,
    pub rate_scale: f32,
    pub hyper_learning_rate: f32,
    pub real_learning_rate: f32,
    #[serde(flatten)]
    pub optimizer_state: ToposOptimizerStateControl,
}

impl ToposOptimizerApplicationPayload {
    pub fn optimizer_state_control(&self) -> ToposOptimizerStateControl {
        self.optimizer_state
    }
}

/// Atomic Rust-owned record connecting one Topos observation to its prescribed optimizer mutation.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct ToposOptimizerSnapshotPayload {
    pub kind: &'static str,
    pub contract_version: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub sequence: u64,
    pub control: ToposControlSignalPayload,
    pub optimizer_application: ToposOptimizerApplicationPayload,
}

impl ToposRuntimeMode {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Balanced => "balanced",
            Self::Guarded => "guarded",
            Self::Exploratory => "exploratory",
            Self::Contextual => "contextual",
            Self::TrainingFirst => "training_first",
            Self::InferenceFirst => "inference_first",
        }
    }

    pub fn mode_id(&self) -> usize {
        match self {
            Self::Balanced => 0,
            Self::Guarded => 1,
            Self::Exploratory => 2,
            Self::Contextual => 3,
            Self::TrainingFirst => 4,
            Self::InferenceFirst => 5,
        }
    }

    pub fn score_key(&self) -> &'static str {
        match self {
            Self::Balanced | Self::Contextual => "context",
            Self::Guarded => "guard",
            Self::Exploratory => "exploration",
            Self::TrainingFirst => "training",
            Self::InferenceFirst => "inference",
        }
    }

    pub fn learning_action(&self) -> &'static str {
        match self {
            Self::Guarded => "dampen_steps",
            Self::Exploratory => "preserve_headroom",
            Self::Contextual | Self::Balanced => "keep_balanced",
            Self::TrainingFirst => "favor_learning",
            Self::InferenceFirst => "hold_learning",
        }
    }

    pub fn inference_action(&self) -> &'static str {
        match self {
            Self::Guarded => "tighten_sampling",
            Self::Exploratory => "widen_sampling",
            Self::Contextual => "raise_context_weight",
            Self::TrainingFirst => "hold_sampling",
            Self::InferenceFirst => "favor_inference",
            Self::Balanced => "keep_balanced",
        }
    }
}

impl ToposRuntimeRouteScores {
    pub fn training_score(&self) -> f32 {
        self.training_score
    }

    pub fn inference_score(&self) -> f32 {
        self.inference_score
    }

    pub fn guard_score(&self) -> f32 {
        self.guard_score
    }

    pub fn exploration_score(&self) -> f32 {
        self.exploration_score
    }

    pub fn context_score(&self) -> f32 {
        self.context_score
    }

    pub fn vector(&self) -> [f32; 5] {
        [
            self.training_score,
            self.inference_score,
            self.guard_score,
            self.exploration_score,
            self.context_score,
        ]
    }

    /// Returns the stable transport payload used by language bindings.
    pub fn payload(&self) -> ToposRuntimeRouteScoresPayload {
        ToposRuntimeRouteScoresPayload {
            training: self.training_score,
            inference: self.inference_score,
            guard: self.guard_score,
            exploration: self.exploration_score,
            context: self.context_score,
            vector: self.vector(),
        }
    }
}

impl ToposRuntimeRoute {
    fn from_profile(profile: ToposRuntimeProfile) -> Self {
        let gradient_pressure = (profile.training_gradient_bias_scale() / 0.35).clamp(0.0, 1.0);
        let rate_headroom = (profile.training_rate_scale() / 1.25).clamp(0.0, 1.0);
        let temperature_headroom = (profile.inference_temperature() / 1.5).clamp(0.0, 1.0);
        let context_pressure = (profile.inference_context_weight() / 1.25).clamp(0.0, 1.0);
        let balance_centering =
            (1.0 - (profile.learning_inference_balance() - 1.0).abs()).clamp(0.0, 1.0);
        let training_score = (0.45 * (1.0 - profile.closure_risk())
            + 0.25 * rate_headroom
            + 0.2 * (1.0 - gradient_pressure)
            + 0.1 * balance_centering)
            .clamp(0.0, 1.0);
        let inference_score = (0.3 * (1.0 - profile.closure_risk())
            + 0.25 * profile.exploration_budget()
            + 0.2 * context_pressure
            + 0.15 * temperature_headroom
            + 0.1 * profile.inference_top_p())
        .clamp(0.0, 1.0);
        let guard_score = (0.45 * profile.closure_risk()
            + 0.35 * profile.control_energy()
            + 0.2 * gradient_pressure)
            .clamp(0.0, 1.0);
        let exploration_score = (0.45 * profile.exploration_budget()
            + 0.25 * temperature_headroom
            + 0.2 * profile.inference_top_p()
            + 0.1 * (1.0 - profile.control_energy()))
        .clamp(0.0, 1.0);
        let context_score = (0.4 * context_pressure
            + 0.25 * (1.0 - profile.closure_risk())
            + 0.2 * balance_centering
            + 0.15 * (1.0 - profile.control_energy()))
        .clamp(0.0, 1.0);
        let scores = ToposRuntimeRouteScores {
            training_score,
            inference_score,
            guard_score,
            exploration_score,
            context_score,
        };
        let mode = if guard_score >= 0.58 && guard_score >= exploration_score {
            ToposRuntimeMode::Guarded
        } else if exploration_score >= 0.62 && profile.exploration_budget() >= 0.45 {
            ToposRuntimeMode::Exploratory
        } else if context_score >= 0.6 && profile.inference_context_weight() >= 0.85 {
            ToposRuntimeMode::Contextual
        } else if training_score >= inference_score + 0.08 {
            ToposRuntimeMode::TrainingFirst
        } else if inference_score >= training_score + 0.08 {
            ToposRuntimeMode::InferenceFirst
        } else {
            ToposRuntimeMode::Balanced
        };
        let score = match mode {
            ToposRuntimeMode::Balanced => {
                ((training_score + inference_score + context_score) / 3.0).clamp(0.0, 1.0)
            }
            ToposRuntimeMode::Guarded => guard_score,
            ToposRuntimeMode::Exploratory => exploration_score,
            ToposRuntimeMode::Contextual => context_score,
            ToposRuntimeMode::TrainingFirst => training_score,
            ToposRuntimeMode::InferenceFirst => inference_score,
        };
        Self {
            profile,
            mode,
            score,
            scores,
        }
    }

    pub fn mode(&self) -> ToposRuntimeMode {
        self.mode
    }

    pub fn mode_label(&self) -> &'static str {
        self.mode.label()
    }

    pub fn mode_id(&self) -> usize {
        self.mode.mode_id()
    }

    pub fn score(&self) -> f32 {
        self.score
    }

    pub fn score_key(&self) -> &'static str {
        self.mode.score_key()
    }

    pub fn learning_action(&self) -> &'static str {
        self.mode.learning_action()
    }

    pub fn inference_action(&self) -> &'static str {
        self.mode.inference_action()
    }

    pub fn scores(&self) -> ToposRuntimeRouteScores {
        self.scores
    }

    pub fn profile(&self) -> ToposRuntimeProfile {
        self.profile
    }

    /// Returns the stable contract payload consumed by native and WASM clients.
    pub fn payload(&self) -> ToposRuntimeRoutePayload {
        ToposRuntimeRoutePayload {
            kind: TOPOS_RUNTIME_ROUTE_KIND,
            contract_version: TOPOS_RUNTIME_ROUTE_CONTRACT_VERSION,
            semantic_owner: TOPOS_RUNTIME_ROUTE_SEMANTIC_OWNER,
            semantic_backend: TOPOS_RUNTIME_ROUTE_SEMANTIC_BACKEND,
            mode: self.mode_label(),
            mode_id: self.mode_id(),
            score: self.score(),
            score_key: self.score_key(),
            learning_action: self.learning_action(),
            inference_action: self.inference_action(),
            scores: self.scores.payload(),
            runtime_profile: self.profile.payload(),
        }
    }
}

impl ToposInferenceHints {
    /// Normalizes external inference hints against a neutral sampling posture.
    pub fn from_input(input: ToposInferenceHintsInput) -> Self {
        Self {
            temperature_scale: 1.0,
            top_p_scale: 1.0,
            sampling_focus: 0.0,
            frequency_penalty_bias: 0.0,
            presence_penalty_bias: 0.0,
            context_weight: 1.0,
        }
        .with_input(input)
    }

    /// Applies a partial external hint override while preserving canonical bounds.
    pub fn with_input(&self, input: ToposInferenceHintsInput) -> Self {
        Self {
            temperature_scale: finite_or(
                input.temperature_scale.unwrap_or(self.temperature_scale),
                self.temperature_scale,
            )
            .clamp(0.5, 1.5),
            top_p_scale: finite_or(
                input.top_p_scale.unwrap_or(self.top_p_scale),
                self.top_p_scale,
            )
            .clamp(0.05, 1.25),
            sampling_focus: finite_or(
                input.sampling_focus.unwrap_or(self.sampling_focus),
                self.sampling_focus,
            )
            .clamp(0.0, 1.0),
            frequency_penalty_bias: finite_or(
                input
                    .frequency_penalty_bias
                    .unwrap_or(self.frequency_penalty_bias),
                self.frequency_penalty_bias,
            )
            .clamp(-2.0, 2.0),
            presence_penalty_bias: finite_or(
                input
                    .presence_penalty_bias
                    .unwrap_or(self.presence_penalty_bias),
                self.presence_penalty_bias,
            )
            .clamp(-2.0, 2.0),
            context_weight: finite_or(
                input.context_weight.unwrap_or(self.context_weight),
                self.context_weight,
            )
            .clamp(0.25, 1.25),
        }
    }

    pub fn temperature_scale(&self) -> f32 {
        self.temperature_scale
    }

    pub fn top_p_scale(&self) -> f32 {
        self.top_p_scale
    }

    pub fn sampling_focus(&self) -> f32 {
        self.sampling_focus
    }

    pub fn frequency_penalty_bias(&self) -> f32 {
        self.frequency_penalty_bias
    }

    pub fn presence_penalty_bias(&self) -> f32 {
        self.presence_penalty_bias
    }

    pub fn context_weight(&self) -> f32 {
        self.context_weight
    }

    pub fn vector(&self) -> [f32; 6] {
        [
            self.temperature_scale,
            self.top_p_scale,
            self.sampling_focus,
            self.frequency_penalty_bias,
            self.presence_penalty_bias,
            self.context_weight,
        ]
    }

    /// Returns the stable transport payload used by language bindings.
    pub fn payload(&self) -> ToposInferenceHintsPayload {
        ToposInferenceHintsPayload {
            temperature_scale: self.temperature_scale,
            top_p_scale: self.top_p_scale,
            sampling_focus: self.sampling_focus,
            frequency_penalty_bias: self.frequency_penalty_bias,
            presence_penalty_bias: self.presence_penalty_bias,
            context_weight: self.context_weight,
            vector: self.vector(),
        }
    }

    /// Applies base sampling parameters and returns concrete hosted-LLM controls.
    pub fn plan(
        &self,
        gain: f32,
        base_temperature: f32,
        base_top_p: f32,
        base_frequency_penalty: f32,
        base_presence_penalty: f32,
    ) -> ToposInferencePlan {
        self.plan_with_options(ToposInferencePlanOptions {
            gain,
            base_temperature,
            base_top_p,
            base_frequency_penalty,
            base_presence_penalty,
            ..ToposInferencePlanOptions::default()
        })
        .expect("default Topos inference bounds are valid")
    }

    /// Applies provider bounds and returns concrete hosted-LLM controls.
    pub fn plan_with_options(
        &self,
        options: ToposInferencePlanOptions,
    ) -> PureResult<ToposInferencePlan> {
        let options = options.validated()?;
        let gain = options.gain;
        let temperature_scale = (1.0 + gain * (self.temperature_scale - 1.0)).clamp(0.5, 1.5);
        let top_p_scale = (1.0 + gain * (self.top_p_scale - 1.0)).clamp(0.05, 1.25);
        let context_weight = (1.0 + gain * (self.context_weight - 1.0)).clamp(0.25, 1.25);
        Ok(ToposInferencePlan {
            gain,
            temperature: (options.base_temperature * temperature_scale)
                .clamp(options.min_temperature, options.max_temperature),
            top_p: (options.base_top_p * top_p_scale).clamp(options.min_top_p, options.max_top_p),
            frequency_penalty: (options.base_frequency_penalty
                + gain * self.frequency_penalty_bias)
                .clamp(-2.0, 2.0),
            presence_penalty: (options.base_presence_penalty + gain * self.presence_penalty_bias)
                .clamp(-2.0, 2.0),
            context_weight,
            temperature_scale,
            top_p_scale,
            sampling_focus: self.sampling_focus,
        })
    }
}

impl ToposInferencePlan {
    pub fn gain(&self) -> f32 {
        self.gain
    }

    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    pub fn top_p(&self) -> f32 {
        self.top_p
    }

    pub fn frequency_penalty(&self) -> f32 {
        self.frequency_penalty
    }

    pub fn presence_penalty(&self) -> f32 {
        self.presence_penalty
    }

    pub fn context_weight(&self) -> f32 {
        self.context_weight
    }

    pub fn temperature_scale(&self) -> f32 {
        self.temperature_scale
    }

    pub fn top_p_scale(&self) -> f32 {
        self.top_p_scale
    }

    pub fn sampling_focus(&self) -> f32 {
        self.sampling_focus
    }

    pub fn vector(&self) -> [f32; 6] {
        [
            self.temperature,
            self.top_p,
            self.frequency_penalty,
            self.presence_penalty,
            self.context_weight,
            self.sampling_focus,
        ]
    }

    /// Returns the stable transport payload used by language bindings.
    pub fn payload(&self) -> ToposInferencePlanPayload {
        ToposInferencePlanPayload {
            gain: self.gain,
            temperature: self.temperature,
            top_p: self.top_p,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            context_weight: self.context_weight,
            temperature_scale: self.temperature_scale,
            top_p_scale: self.top_p_scale,
            sampling_focus: self.sampling_focus,
            vector: self.vector(),
        }
    }
}

impl ToposRuntimeProfile {
    /// Normalize an external profile input into the canonical runtime-route domain.
    pub fn from_input(input: ToposRuntimeProfileInput) -> Self {
        let defaults = ToposRuntimeProfileInput::default();
        Self {
            training_gain: finite_or(input.training_gain, defaults.training_gain).max(0.0),
            inference_gain: finite_or(input.inference_gain, defaults.inference_gain).max(0.0),
            closure_risk: finite_or(input.closure_risk, defaults.closure_risk).clamp(0.0, 1.0),
            exploration_budget: finite_or(input.exploration_budget, defaults.exploration_budget)
                .clamp(0.0, 1.0),
            control_energy: finite_or(input.control_energy, defaults.control_energy)
                .clamp(0.0, 1.0),
            training_rate_scale: finite_or(input.training_rate_scale, defaults.training_rate_scale)
                .clamp(0.01, 2.0),
            training_gradient_bias_scale: finite_or(
                input.training_gradient_bias_scale,
                defaults.training_gradient_bias_scale,
            )
            .clamp(0.0, 0.35),
            inference_temperature: finite_or(
                input.inference_temperature,
                defaults.inference_temperature,
            )
            .clamp(0.0, 2.0),
            inference_top_p: finite_or(input.inference_top_p, defaults.inference_top_p)
                .clamp(0.05, 1.0),
            inference_context_weight: finite_or(
                input.inference_context_weight,
                defaults.inference_context_weight,
            )
            .clamp(0.25, 1.25),
            learning_inference_balance: finite_or(
                input.learning_inference_balance,
                defaults.learning_inference_balance,
            )
            .clamp(0.0, 2.0),
        }
    }

    fn from_parts(
        signal: &ToposControlSignal,
        training_plan: ToposTrainingPlan,
        inference_plan: ToposInferencePlan,
    ) -> Self {
        let closure_risk = (0.5 * signal.closure_pressure()
            + 0.3 * signal.guard_strength()
            + 0.2 * signal.step_damping())
        .clamp(0.0, 1.0);
        let exploration_budget =
            (0.6 * signal.openness() + 0.4 * signal.exploration_hint()).clamp(0.0, 1.0);
        let gradient_pressure =
            (training_plan.effective_gradient_bias_scale() / 0.35).clamp(0.0, 1.0);
        let rate_pressure = (1.0 - training_plan.rate_scale()).max(0.0).clamp(0.0, 1.0);
        let context_pressure = inference_plan.context_weight().clamp(0.0, 1.25) / 1.25;
        let control_energy = (0.35 * closure_risk
            + 0.25 * gradient_pressure
            + 0.2 * rate_pressure
            + 0.2 * context_pressure)
            .clamp(0.0, 1.0);
        let learning_inference_balance =
            (training_plan.rate_scale() / inference_plan.temperature().max(1e-6)).clamp(0.0, 2.0);
        Self::from_input(ToposRuntimeProfileInput {
            training_gain: training_plan.gain(),
            inference_gain: inference_plan.gain(),
            closure_risk,
            exploration_budget,
            control_energy,
            training_rate_scale: training_plan.rate_scale(),
            training_gradient_bias_scale: training_plan.effective_gradient_bias_scale(),
            inference_temperature: inference_plan.temperature(),
            inference_top_p: inference_plan.top_p(),
            inference_context_weight: inference_plan.context_weight(),
            learning_inference_balance,
        })
    }

    pub fn training_gain(&self) -> f32 {
        self.training_gain
    }

    pub fn inference_gain(&self) -> f32 {
        self.inference_gain
    }

    pub fn closure_risk(&self) -> f32 {
        self.closure_risk
    }

    pub fn exploration_budget(&self) -> f32 {
        self.exploration_budget
    }

    pub fn control_energy(&self) -> f32 {
        self.control_energy
    }

    pub fn training_rate_scale(&self) -> f32 {
        self.training_rate_scale
    }

    pub fn training_gradient_bias_scale(&self) -> f32 {
        self.training_gradient_bias_scale
    }

    pub fn inference_temperature(&self) -> f32 {
        self.inference_temperature
    }

    pub fn inference_top_p(&self) -> f32 {
        self.inference_top_p
    }

    pub fn inference_context_weight(&self) -> f32 {
        self.inference_context_weight
    }

    pub fn learning_inference_balance(&self) -> f32 {
        self.learning_inference_balance
    }

    pub fn vector(&self) -> [f32; 6] {
        [
            self.control_energy,
            self.closure_risk,
            self.exploration_budget,
            self.training_rate_scale,
            self.inference_temperature,
            self.inference_context_weight,
        ]
    }

    /// Returns the stable transport payload used by language bindings.
    pub fn payload(&self) -> ToposRuntimeProfilePayload {
        ToposRuntimeProfilePayload {
            training_gain: self.training_gain,
            inference_gain: self.inference_gain,
            closure_risk: self.closure_risk,
            exploration_budget: self.exploration_budget,
            control_energy: self.control_energy,
            training_rate_scale: self.training_rate_scale,
            training_gradient_bias_scale: self.training_gradient_bias_scale,
            inference_temperature: self.inference_temperature,
            inference_top_p: self.inference_top_p,
            inference_context_weight: self.inference_context_weight,
            learning_inference_balance: self.learning_inference_balance,
            vector: self.vector(),
        }
    }

    /// Names the runtime route implied by this joint profile.
    pub fn route(&self) -> ToposRuntimeRoute {
        ToposRuntimeRoute::from_profile(*self)
    }
}

impl ToposControlSignal {
    /// Validates external topology inputs and derives the canonical control signal.
    pub fn from_input(input: ToposControlSignalInput) -> PureResult<Self> {
        let topos = OpenCartesianTopos::new(
            input.curvature,
            input.tolerance,
            input.saturation,
            input.max_depth,
            input.max_volume,
        )?
        .with_porosity(input.porosity)?;
        Ok(Self::from_observation(
            &topos,
            input.observed_depth,
            input.visited_volume,
        ))
    }

    /// Builds a signal from a topos and optional traversal observations.
    pub fn from_observation(
        topos: &OpenCartesianTopos,
        observed_depth: usize,
        visited_volume: usize,
    ) -> Self {
        let max_depth = topos.max_depth();
        let max_volume = topos.max_volume();
        let depth_pressure = if max_depth == 0 {
            1.0
        } else {
            (observed_depth as f32 / max_depth as f32).clamp(0.0, 1.0)
        };
        let volume_pressure = if max_volume == 0 {
            1.0
        } else {
            (visited_volume as f32 / max_volume as f32).clamp(0.0, 1.0)
        };
        let closure_pressure = depth_pressure.max(volume_pressure);
        let openness = (1.0 - closure_pressure).clamp(0.0, 1.0);
        let guard_strength = ((1.0 - topos.porosity()).clamp(0.0, 1.0) * 0.7
            + closure_pressure * 0.3)
            .clamp(0.0, 1.0);
        let stability_hint = (openness * (1.0 - 0.4 * topos.porosity())).clamp(0.0, 1.0);
        let exploration_hint =
            (topos.porosity() * openness + (1.0 - guard_strength) * 0.25).clamp(0.0, 1.0);
        let step_damping = (closure_pressure * guard_strength).clamp(0.0, 1.0);
        let learning_rate_scale =
            (1.0 - 0.65 * step_damping + 0.25 * exploration_hint).clamp(0.1, 1.25);
        let temperature_scale = (0.75 + 0.75 * exploration_hint + 0.25 * openness
            - 0.35 * guard_strength)
            .clamp(0.5, 1.5);
        let regularization_scale = (0.5 + guard_strength + 0.5 * closure_pressure
            - 0.25 * exploration_hint)
            .clamp(0.5, 2.0);
        let sampling_focus = (guard_strength * (1.0 - 0.5 * exploration_hint)
            + 0.25 * closure_pressure)
            .clamp(0.0, 1.0);
        Self {
            curvature: topos.curvature(),
            tolerance: topos.tolerance(),
            saturation: topos.saturation(),
            porosity: topos.porosity(),
            max_depth,
            max_volume,
            observed_depth,
            visited_volume,
            remaining_volume: max_volume.saturating_sub(visited_volume),
            depth_pressure,
            volume_pressure,
            closure_pressure,
            openness,
            guard_strength,
            stability_hint,
            exploration_hint,
            learning_rate_scale,
            temperature_scale,
            regularization_scale,
            step_damping,
            sampling_focus,
        }
    }

    pub fn curvature(&self) -> f32 {
        self.curvature
    }

    pub fn tolerance(&self) -> f32 {
        self.tolerance
    }

    pub fn saturation(&self) -> f32 {
        self.saturation
    }

    pub fn porosity(&self) -> f32 {
        self.porosity
    }

    pub fn max_depth(&self) -> usize {
        self.max_depth
    }

    pub fn max_volume(&self) -> usize {
        self.max_volume
    }

    pub fn observed_depth(&self) -> usize {
        self.observed_depth
    }

    pub fn visited_volume(&self) -> usize {
        self.visited_volume
    }

    pub fn remaining_volume(&self) -> usize {
        self.remaining_volume
    }

    pub fn depth_pressure(&self) -> f32 {
        self.depth_pressure
    }

    pub fn volume_pressure(&self) -> f32 {
        self.volume_pressure
    }

    pub fn closure_pressure(&self) -> f32 {
        self.closure_pressure
    }

    pub fn openness(&self) -> f32 {
        self.openness
    }

    pub fn guard_strength(&self) -> f32 {
        self.guard_strength
    }

    pub fn stability_hint(&self) -> f32 {
        self.stability_hint
    }

    pub fn exploration_hint(&self) -> f32 {
        self.exploration_hint
    }

    pub fn learning_rate_scale(&self) -> f32 {
        self.learning_rate_scale
    }

    pub fn temperature_scale(&self) -> f32 {
        self.temperature_scale
    }

    pub fn regularization_scale(&self) -> f32 {
        self.regularization_scale
    }

    pub fn step_damping(&self) -> f32 {
        self.step_damping
    }

    pub fn sampling_focus(&self) -> f32 {
        self.sampling_focus
    }

    /// Named optimizer controls derived from this signal.
    pub fn training_hints(&self) -> ToposTrainingHints {
        let gradient_bias_scale = (0.04 * self.closure_pressure
            + 0.16 * self.step_damping
            + 0.08 * self.sampling_focus * self.closure_pressure)
            .clamp(0.0, 0.35);
        let clip_scale = (1.0 - 0.5 * self.step_damping).clamp(0.25, 1.0);
        let momentum_damping =
            (0.75 * self.step_damping + 0.15 * self.closure_pressure).clamp(0.0, 0.85);
        ToposTrainingHints {
            learning_rate_scale: self.learning_rate_scale,
            regularization_scale: self.regularization_scale,
            step_damping: self.step_damping,
            gradient_bias_scale,
            clip_scale,
            momentum_damping,
        }
    }

    /// Gain-applied optimizer controls derived from this signal.
    pub fn training_plan(&self, gain: f32) -> ToposTrainingPlan {
        self.training_hints().plan(gain)
    }

    /// Named hosted-inference controls derived from this signal.
    pub fn inference_hints(&self) -> ToposInferenceHints {
        let top_p_scale =
            (1.0 - 0.2 * self.sampling_focus + 0.1 * self.exploration_hint).clamp(0.05, 1.25);
        let frequency_penalty_bias =
            (0.35 * self.sampling_focus + 0.2 * self.step_damping).clamp(-2.0, 2.0);
        let presence_penalty_bias =
            (0.4 * self.exploration_hint - 0.2 * self.sampling_focus).clamp(-2.0, 2.0);
        let context_weight =
            (0.5 + 0.5 * self.guard_strength + 0.25 * self.closure_pressure).clamp(0.25, 1.25);
        ToposInferenceHints {
            temperature_scale: self.temperature_scale,
            top_p_scale,
            sampling_focus: self.sampling_focus,
            frequency_penalty_bias,
            presence_penalty_bias,
            context_weight,
        }
    }

    /// Concrete hosted-inference request controls derived from this signal.
    pub fn inference_plan(
        &self,
        gain: f32,
        base_temperature: f32,
        base_top_p: f32,
        base_frequency_penalty: f32,
        base_presence_penalty: f32,
    ) -> ToposInferencePlan {
        self.inference_hints().plan(
            gain,
            base_temperature,
            base_top_p,
            base_frequency_penalty,
            base_presence_penalty,
        )
    }

    /// Joint learning/inference controls derived from this signal.
    pub fn runtime_profile(
        &self,
        training_gain: f32,
        inference_gain: f32,
        base_temperature: f32,
        base_top_p: f32,
        base_frequency_penalty: f32,
        base_presence_penalty: f32,
    ) -> ToposRuntimeProfile {
        let training_plan = self.training_plan(training_gain);
        let inference_plan = self.inference_plan(
            inference_gain,
            base_temperature,
            base_top_p,
            base_frequency_penalty,
            base_presence_penalty,
        );
        ToposRuntimeProfile::from_parts(self, training_plan, inference_plan)
    }

    /// Named runtime route derived from this signal's joint learning/inference controls.
    pub fn runtime_route(
        &self,
        training_gain: f32,
        inference_gain: f32,
        base_temperature: f32,
        base_top_p: f32,
        base_frequency_penalty: f32,
        base_presence_penalty: f32,
    ) -> ToposRuntimeRoute {
        self.runtime_profile(
            training_gain,
            inference_gain,
            base_temperature,
            base_top_p,
            base_frequency_penalty,
            base_presence_penalty,
        )
        .route()
    }

    /// Returns the canonical default control bundle shared by every client.
    pub fn payload(&self) -> ToposControlSignalPayload {
        self.payload_with_options(ToposControlPlanOptions::default(), None, None)
            .expect("default Topos control options are valid")
    }

    /// Derives one canonical control bundle from a signal, optional external hints, and plans.
    pub fn payload_with_options(
        &self,
        options: ToposControlPlanOptions,
        training_hints_input: Option<ToposTrainingHintsInput>,
        inference_hints_input: Option<ToposInferenceHintsInput>,
    ) -> PureResult<ToposControlSignalPayload> {
        let training_hints = training_hints_input.map_or_else(
            || self.training_hints(),
            |input| self.training_hints().with_input(input),
        );
        let training_plan = training_hints.plan(options.training_gain);
        let inference_hints = inference_hints_input.map_or_else(
            || self.inference_hints(),
            |input| self.inference_hints().with_input(input),
        );
        let inference_plan = inference_hints.plan_with_options(options.inference)?;
        let runtime_profile = ToposRuntimeProfile::from_parts(self, training_plan, inference_plan);
        let runtime_route = runtime_profile.route();
        Ok(ToposControlSignalPayload {
            kind: TOPOS_CONTROL_SIGNAL_KIND,
            contract_version: TOPOS_CONTROL_SIGNAL_CONTRACT_VERSION,
            semantic_owner: TOPOS_CONTROL_SIGNAL_SEMANTIC_OWNER,
            semantic_backend: TOPOS_CONTROL_SIGNAL_SEMANTIC_BACKEND,
            curvature: self.curvature,
            tolerance: self.tolerance,
            saturation: self.saturation,
            porosity: self.porosity,
            max_depth: self.max_depth,
            max_volume: self.max_volume,
            observed_depth: self.observed_depth,
            visited_volume: self.visited_volume,
            remaining_volume: self.remaining_volume,
            depth_pressure: self.depth_pressure,
            volume_pressure: self.volume_pressure,
            closure_pressure: self.closure_pressure,
            openness: self.openness,
            guard_strength: self.guard_strength,
            stability_hint: self.stability_hint,
            exploration_hint: self.exploration_hint,
            learning_rate_scale: self.learning_rate_scale,
            temperature_scale: self.temperature_scale,
            regularization_scale: self.regularization_scale,
            step_damping: self.step_damping,
            sampling_focus: self.sampling_focus,
            runtime_hints: self.runtime_hints(),
            gradient: self.gradient(),
            training_hints: training_hints.payload(),
            training_plan: training_plan.payload(),
            inference_hints: inference_hints.payload(),
            inference_plan: inference_plan.payload(),
            runtime_profile: runtime_profile.payload(),
            runtime_route: runtime_route.payload(),
        })
    }

    /// Binds one canonical control bundle to its prescribed learning-rate mutation.
    #[allow(clippy::too_many_arguments)]
    pub fn optimizer_snapshot(
        &self,
        sequence: u64,
        hyper_learning_rate: f32,
        real_learning_rate: f32,
        options: ToposControlPlanOptions,
        training_hints_input: Option<ToposTrainingHintsInput>,
        inference_hints_input: Option<ToposInferenceHintsInput>,
    ) -> PureResult<ToposOptimizerSnapshotPayload> {
        if sequence > TOPOS_OPTIMIZER_SNAPSHOT_MAX_SEQUENCE {
            return Err(TensorError::InvalidValue {
                label: "topos_optimizer_snapshot_sequence",
            });
        }
        validate_optimizer_learning_rate(
            "topos_optimizer_input_hyper_learning_rate",
            hyper_learning_rate,
        )?;
        validate_optimizer_learning_rate(
            "topos_optimizer_input_real_learning_rate",
            real_learning_rate,
        )?;

        let control =
            self.payload_with_options(options, training_hints_input, inference_hints_input)?;
        let rate_scale = control.training_plan.rate_scale;
        let scaled_hyper_learning_rate = hyper_learning_rate * rate_scale;
        let scaled_real_learning_rate = real_learning_rate * rate_scale;
        validate_optimizer_learning_rate(
            "topos_optimizer_output_hyper_learning_rate",
            scaled_hyper_learning_rate,
        )?;
        validate_optimizer_learning_rate(
            "topos_optimizer_output_real_learning_rate",
            scaled_real_learning_rate,
        )?;
        let gradient_bias_basis = topos_optimizer_gradient_bias_basis(
            control.closure_pressure as f64,
            control.volume_pressure as f64,
            control.depth_pressure as f64,
            control.guard_strength as f64,
            control.training_plan.step_damping as f64,
            control.sampling_focus as f64,
            control.training_hints.learning_rate_scale as f64,
            control.training_plan.regularization_scale as f64,
            control.openness as f64,
            control.exploration_hint as f64,
        )
        .map(|value| value as f32);
        let optimizer_state = ToposOptimizerStateControl::new(
            control.training_plan.effective_gradient_bias_scale,
            gradient_bias_basis,
            control.training_plan.effective_gradient_clip_scale,
            control.training_plan.effective_momentum_damping,
        )?;

        Ok(ToposOptimizerSnapshotPayload {
            kind: TOPOS_OPTIMIZER_SNAPSHOT_KIND,
            contract_version: TOPOS_OPTIMIZER_SNAPSHOT_CONTRACT_VERSION,
            semantic_owner: TOPOS_OPTIMIZER_SNAPSHOT_SEMANTIC_OWNER,
            semantic_backend: TOPOS_OPTIMIZER_SNAPSHOT_SEMANTIC_BACKEND,
            sequence,
            control,
            optimizer_application: ToposOptimizerApplicationPayload {
                scope: "learning_rate_and_gradient_state",
                control_path: "control.training_plan",
                input_hyper_learning_rate: hyper_learning_rate,
                input_real_learning_rate: real_learning_rate,
                rate_scale,
                hyper_learning_rate: scaled_hyper_learning_rate,
                real_learning_rate: scaled_real_learning_rate,
                optimizer_state,
            },
        })
    }

    /// Compact runtime hint vector for optimizers and inference adapters.
    pub fn runtime_hints(&self) -> [f32; 5] {
        [
            self.learning_rate_scale,
            self.temperature_scale,
            self.regularization_scale,
            self.step_damping,
            self.sampling_focus,
        ]
    }

    /// Compact gradient-like vector for Z-space inference partials.
    pub fn gradient(&self) -> [f32; 6] {
        [
            self.openness,
            self.guard_strength,
            self.stability_hint,
            self.exploration_hint,
            self.depth_pressure,
            self.volume_pressure,
        ]
    }

    /// Projects this signal into the canonical metrics consumed by Z-space inference.
    pub fn zspace_projection(
        &self,
        options: ToposZSpaceProjectionOptions,
    ) -> PureResult<ToposZSpaceProjection> {
        let options = options.validated()?;
        let mut gradient = self.gradient().to_vec();
        gradient.resize(options.gradient_dim, 0.0);
        Ok(ToposZSpaceProjection {
            speed: (self.openness * (0.7 + 0.3 * self.porosity) * self.learning_rate_scale)
                .clamp(0.0, 1.0),
            memory: self.volume_pressure.clamp(0.0, 1.0),
            stability: (self.stability_hint / self.regularization_scale.max(1.0)).clamp(0.0, 1.0),
            drs: (self.openness - self.closure_pressure - 0.25 * self.step_damping)
                .clamp(-1.0, 1.0),
            frac: self.sampling_focus.clamp(0.0, 1.0),
            gradient,
        })
    }
}

impl ToposZSpaceProjection {
    pub fn speed(&self) -> f32 {
        self.speed
    }

    pub fn memory(&self) -> f32 {
        self.memory
    }

    pub fn stability(&self) -> f32 {
        self.stability
    }

    pub fn drs(&self) -> f32 {
        self.drs
    }

    pub fn frac(&self) -> f32 {
        self.frac
    }

    pub fn gradient(&self) -> &[f32] {
        &self.gradient
    }

    pub fn vector(&self) -> [f32; 5] {
        [self.speed, self.memory, self.stability, self.drs, self.frac]
    }

    /// Returns the stable transport payload used by language bindings.
    pub fn payload(&self) -> ToposZSpaceProjectionPayload {
        ToposZSpaceProjectionPayload {
            kind: TOPOS_ZSPACE_PROJECTION_KIND,
            contract_version: TOPOS_ZSPACE_PROJECTION_CONTRACT_VERSION,
            semantic_owner: TOPOS_ZSPACE_PROJECTION_SEMANTIC_OWNER,
            semantic_backend: TOPOS_ZSPACE_PROJECTION_SEMANTIC_BACKEND,
            gradient_basis: TOPOS_ZSPACE_PROJECTION_GRADIENT_BASIS,
            gradient_formula: TOPOS_ZSPACE_PROJECTION_GRADIENT_FORMULA,
            gradient_channels: TOPOS_ZSPACE_PROJECTION_GRADIENT_CHANNELS,
            gradient_dim: self.gradient.len(),
            base_gradient_dim: TOPOS_ZSPACE_PROJECTION_BASE_GRADIENT_DIM,
            speed: self.speed,
            memory: self.memory,
            stability: self.stability,
            drs: self.drs,
            frac: self.frac,
            gradient: self.gradient.clone(),
            vector: self.vector(),
        }
    }
}

/// Numerically guards the Lawvere–Tierney topology that keeps probabilistic data j-closed.
#[derive(Clone, Copy, Debug)]
pub struct LawvereTierneyGuard {
    density_min: f32,
    density_max: f32,
    mass_tolerance: f32,
}

impl LawvereTierneyGuard {
    /// Creates a new guard ensuring densities stay within the provided window and
    /// that normalisations land within `mass_tolerance` of unit mass.
    pub fn new(density_min: f32, density_max: f32, mass_tolerance: f32) -> PureResult<Self> {
        for (label, value) in [
            ("lawvere_tierney_density_min", density_min),
            ("lawvere_tierney_density_max", density_max),
            ("lawvere_tierney_mass_tolerance", mass_tolerance),
        ] {
            if value.is_finite() {
                continue;
            }
            return Err(TensorError::NonFiniteValue { label, value });
        }
        if density_min <= 0.0 || density_max < density_min {
            return Err(TensorError::InvalidValue {
                label: "lawvere_tierney_density_window",
            });
        }
        if !(f32::EPSILON..1.0).contains(&mass_tolerance) {
            return Err(TensorError::InvalidValue {
                label: "lawvere_tierney_mass_tolerance",
            });
        }
        Ok(Self {
            density_min,
            density_max,
            mass_tolerance,
        })
    }

    /// Returns the minimum density admitted by the guard.
    pub fn density_min(&self) -> f32 {
        self.density_min
    }

    /// Returns the maximum density admitted by the guard.
    pub fn density_max(&self) -> f32 {
        self.density_max
    }

    /// Returns the tolerance allowed when projecting to unit mass.
    pub fn mass_tolerance(&self) -> f32 {
        self.mass_tolerance
    }

    fn guard_density(&self, density: f32) -> PureResult<()> {
        if !density.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "lawvere_tierney_density",
                value: density,
            });
        }
        if density <= 0.0 {
            return Err(TensorError::InvalidValue {
                label: "density_non_positive",
            });
        }
        if density < self.density_min || density > self.density_max {
            return Err(TensorError::InvalidValue {
                label: "density_window_violation",
            });
        }
        Ok(())
    }

    fn guard_mass(&self, mass: f32) -> PureResult<()> {
        if !mass.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "lawvere_tierney_mass",
                value: mass,
            });
        }
        if mass <= 0.0 {
            return Err(TensorError::InvalidValue {
                label: "lawvere_tierney_mass_non_positive",
            });
        }
        if mass < self.density_min {
            return Err(TensorError::InvalidValue {
                label: "lawvere_tierney_mass_too_small",
            });
        }
        if mass > 1.0 + self.mass_tolerance {
            return Err(TensorError::InvalidValue {
                label: "lawvere_tierney_mass_too_large",
            });
        }
        Ok(())
    }

    fn guard_cover_mass(&self, total_mass: f32) -> PureResult<()> {
        if !total_mass.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "lawvere_tierney_cover_mass",
                value: total_mass,
            });
        }
        if total_mass <= 0.0 {
            return Err(TensorError::InvalidValue {
                label: "lawvere_tierney_cover_empty",
            });
        }
        let deviation = (total_mass - 1.0).abs();
        if deviation > self.mass_tolerance {
            return Err(TensorError::InvalidValue {
                label: "lawvere_tierney_cover_mass_violation",
            });
        }
        Ok(())
    }

    /// Projects a probability slice to the guarded subtopos.
    ///
    /// The bounded proportional projection preserves zero support and relative
    /// mass wherever neither density bound is active. It commits only after the
    /// density window and unit-mass invariant hold simultaneously.
    pub fn project_slice(
        &self,
        label: &'static str,
        slice: &mut [f32],
        saturation: f32,
    ) -> PureResult<()> {
        if slice.is_empty() {
            return Err(TensorError::EmptyInput(label));
        }
        if !saturation.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "lawvere_tierney_saturation",
                value: saturation,
            });
        }
        if saturation <= 0.0 {
            return Err(TensorError::NonPositiveSaturation { saturation });
        }

        let effective_density_max = self.density_max.min(saturation);
        let mut projected = Vec::with_capacity(slice.len());
        let mut raw_sum = 0.0f64;
        let mut repaired_non_finite = 0usize;
        let mut repaired_negative = 0usize;
        let mut clipped_saturation = 0usize;
        let mut clipped_density_max = 0usize;
        for &original in slice.iter() {
            if original.is_finite() {
                raw_sum += original as f64;
            } else {
                repaired_non_finite = repaired_non_finite.saturating_add(1);
            }
            let mut guarded = if original.is_finite() { original } else { 0.0 };
            if guarded < 0.0 {
                repaired_negative = repaired_negative.saturating_add(1);
                guarded = 0.0;
            }
            if guarded > saturation {
                clipped_saturation = clipped_saturation.saturating_add(1);
                guarded = saturation;
            }
            if guarded > self.density_max {
                clipped_density_max = clipped_density_max.saturating_add(1);
                guarded = self.density_max;
            }
            projected.push(guarded);
        }

        let guarded_sum = projected.iter().map(|&value| value as f64).sum::<f64>();
        let active_indices = projected
            .iter()
            .enumerate()
            .filter_map(|(index, &value)| (value > 0.0).then_some(index))
            .collect::<Vec<_>>();
        let active_values = active_indices.len();
        let active = active_values as f64;
        let density_min = self.density_min as f64;
        let density_max = effective_density_max as f64;
        let tolerance = self.mass_tolerance as f64;
        if active_values == 0
            || active * density_min > 1.0 + tolerance
            || active * density_max < 1.0 - tolerance
        {
            return Err(TensorError::ProbabilityProjectionInfeasible {
                active_values,
                density_min: self.density_min,
                density_max: effective_density_max,
            });
        }

        let lifted_density_min = active_indices
            .iter()
            .filter(|&&index| projected[index] < self.density_min)
            .count();
        let mut scale_low = 0.0f64;
        let mut scale_high = active_indices
            .iter()
            .map(|&index| density_max / projected[index] as f64)
            .fold(0.0f64, f64::max);
        const PROJECTION_ITERATIONS: usize = 80;
        for _ in 0..PROJECTION_ITERATIONS {
            let scale = 0.5 * (scale_low + scale_high);
            let mass = active_indices
                .iter()
                .map(|&index| (projected[index] as f64 * scale).clamp(density_min, density_max))
                .sum::<f64>();
            if mass < 1.0 {
                scale_low = scale;
            } else {
                scale_high = scale;
            }
        }
        let scale = 0.5 * (scale_low + scale_high);
        for &index in &active_indices {
            projected[index] =
                (projected[index] as f64 * scale).clamp(density_min, density_max) as f32;
        }

        let renorm_sum = projected.iter().map(|&value| value as f64).sum::<f64>();
        for _ in 0..4 {
            let mass = projected.iter().map(|&value| value as f64).sum::<f64>();
            let mut delta = 1.0 - mass;
            if delta.abs() <= tolerance * 0.25 {
                break;
            }
            for &index in &active_indices {
                let current = projected[index] as f64;
                let adjustment = if delta > 0.0 {
                    delta.min(density_max - current)
                } else {
                    delta.max(density_min - current)
                };
                projected[index] = (current + adjustment) as f32;
                delta -= adjustment;
                if delta.abs() <= tolerance * 0.25 {
                    break;
                }
            }
        }

        for &index in &active_indices {
            let value = projected[index];
            if value + self.mass_tolerance < self.density_min
                || value > effective_density_max + self.mass_tolerance
            {
                return Err(TensorError::InvalidValue {
                    label: "lawvere_tierney_probability_density_window",
                });
            }
        }
        let final_sum = projected.iter().map(|&value| value as f64).sum::<f64>();
        let deviation = (final_sum - 1.0).abs();
        if deviation > tolerance {
            return Err(TensorError::InvalidValue {
                label: "lawvere_tierney_probability_mass",
            });
        }
        slice.copy_from_slice(&projected);
        crate::emit_tensor_op(
            "lawvere_guard_probability_slice",
            &[slice.len()],
            &[slice.len()],
        );
        crate::emit_tensor_op_meta("lawvere_guard_probability_slice", || {
            serde_json::json!({
                "backend": "control_cpu",
                "requested_backend": "host",
                "kind": "lawvere_tierney_probability_guard",
                "label": label,
                "values": slice.len(),
                "density_min": self.density_min,
                "density_max": self.density_max,
                "effective_density_max": effective_density_max,
                "mass_tolerance": self.mass_tolerance,
                "saturation": saturation,
                "raw_sum": raw_sum,
                "guarded_sum": guarded_sum,
                "renorm_sum": renorm_sum,
                "final_sum": final_sum,
                "mass_deviation": deviation,
                "repaired_non_finite": repaired_non_finite,
                "repaired_negative": repaired_negative,
                "clipped_saturation": clipped_saturation,
                "clipped_density_max": clipped_density_max,
                "lifted_density_min": lifted_density_min,
                "active_values": active_values,
                "projection_method": "bounded_proportional_bisection",
                "projection_iterations": PROJECTION_ITERATIONS,
                "repaired_values": repaired_non_finite
                    .saturating_add(repaired_negative)
                    .saturating_add(clipped_saturation)
                    .saturating_add(clipped_density_max)
                    .saturating_add(lifted_density_min),
                "final_rescaled": false,
            })
        });
        Ok(())
    }
}

/// Open box in a negatively curved Z-space site.
#[derive(Clone, Debug)]
pub struct ZBox {
    centers: Vec<Vec<f32>>,
    radii: Vec<f32>,
    density: f32,
}

impl ZBox {
    /// Builds a new κ-box. Each factor is described by a centre and a radius.
    pub fn new(centers: Vec<Vec<f32>>, radii: Vec<f32>, density: f32) -> PureResult<Self> {
        if centers.is_empty() || radii.is_empty() {
            return Err(TensorError::EmptyInput("zbox_factors"));
        }
        if centers.len() != radii.len() {
            return Err(TensorError::DataLength {
                expected: centers.len(),
                got: radii.len(),
            });
        }
        for center in centers.iter() {
            if center.is_empty() {
                return Err(TensorError::EmptyInput("zbox_center"));
            }
            if center.iter().any(|c| !c.is_finite()) {
                return Err(TensorError::NonFiniteValue {
                    label: "zbox_center",
                    value: f32::NAN,
                });
            }
        }
        for radius in radii.iter() {
            if !radius.is_finite() {
                return Err(TensorError::NonFiniteValue {
                    label: "zbox_radius",
                    value: *radius,
                });
            }
            if *radius <= 0.0 {
                return Err(TensorError::InvalidValue {
                    label: "zbox_radius_non_positive",
                });
            }
        }
        if !density.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "zbox_density",
                value: density,
            });
        }
        if density <= 0.0 {
            return Err(TensorError::InvalidValue {
                label: "zbox_density_non_positive",
            });
        }
        Ok(Self {
            centers,
            radii,
            density,
        })
    }

    /// Returns the number of factors composing the κ-box.
    pub fn arity(&self) -> usize {
        self.radii.len()
    }

    /// Returns the density weight assigned to the box.
    pub fn density(&self) -> f32 {
        self.density
    }

    /// Returns the dimension of the ambient Z-space for the i-th factor.
    pub fn factor_dimension(&self, index: usize) -> PureResult<usize> {
        self.centers
            .get(index)
            .map(Vec::len)
            .ok_or(TensorError::InvalidValue {
                label: "zbox_factor_index",
            })
    }

    /// Computes the total hyperbolic volume of the κ-box.
    pub fn hyperbolic_volume(&self, curvature: f32) -> PureResult<f32> {
        if !curvature.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "zbox_curvature",
                value: curvature,
            });
        }
        if curvature >= 0.0 {
            return Err(TensorError::NonHyperbolicCurvature { curvature });
        }
        let mut volume = 1.0f64;
        for (i, radius) in self.radii.iter().enumerate() {
            let dim = self.factor_dimension(i)?;
            let factor = hyperbolic_ball_volume(curvature, *radius, dim)?;
            volume *= factor as f64;
            if !volume.is_finite() || volume > f32::MAX as f64 {
                return Err(TensorError::NonFiniteValue {
                    label: "zbox_hyperbolic_volume",
                    value: f32::INFINITY,
                });
            }
        }
        Ok(volume as f32)
    }

    /// Returns the probability mass assigned to this κ-box.
    pub fn probability_mass(&self, curvature: f32) -> PureResult<f32> {
        let mass = self.density as f64 * self.hyperbolic_volume(curvature)? as f64;
        if !mass.is_finite() || mass > f32::MAX as f64 {
            return Err(TensorError::NonFiniteValue {
                label: "zbox_probability_mass",
                value: f32::INFINITY,
            });
        }
        Ok(mass as f32)
    }

    fn validate_radius_window(&self, min: f32, max: f32) -> PureResult<()> {
        if min <= 0.0 || max <= 0.0 || max < min {
            return Err(TensorError::InvalidValue {
                label: "zbox_radius_window",
            });
        }
        for radius in self.radii.iter() {
            if *radius < min || *radius > max {
                return Err(TensorError::InvalidValue {
                    label: "zbox_radius_window_violation",
                });
            }
        }
        Ok(())
    }
}

/// Guards the κ-box site attached to an open-cartesian topos.
#[derive(Clone, Debug)]
pub struct ZBoxSite {
    curvature: f32,
    radius_min: f32,
    radius_max: f32,
    guard: LawvereTierneyGuard,
}

impl ZBoxSite {
    /// Builds the default κ-box site for the provided curvature.
    pub fn default_for(curvature: f32) -> PureResult<Self> {
        if !curvature.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "zbox_site_curvature",
                value: curvature,
            });
        }
        if curvature >= 0.0 {
            return Err(TensorError::NonHyperbolicCurvature { curvature });
        }
        let guard = LawvereTierneyGuard::new(1e-6, 1e3, 1e-5)?;
        Ok(Self {
            curvature,
            radius_min: 1e-3,
            radius_max: 64.0,
            guard,
        })
    }

    /// Adjusts the admissible radius window.
    pub fn with_radius_window(mut self, min: f32, max: f32) -> PureResult<Self> {
        for (label, value) in [("zbox_radius_min", min), ("zbox_radius_max", max)] {
            if !value.is_finite() {
                return Err(TensorError::NonFiniteValue { label, value });
            }
        }
        if min <= 0.0 || max <= 0.0 || max < min {
            return Err(TensorError::InvalidValue {
                label: "zbox_radius_window",
            });
        }
        self.radius_min = min;
        self.radius_max = max;
        Ok(self)
    }

    /// Replaces the internal Lawvere–Tierney guard.
    pub fn with_guard(mut self, guard: LawvereTierneyGuard) -> Self {
        self.guard = guard;
        self
    }

    /// Returns the curvature parameter for the site.
    pub fn curvature(&self) -> f32 {
        self.curvature
    }

    /// Returns the minimum admissible κ-box radius.
    pub fn radius_min(&self) -> f32 {
        self.radius_min
    }

    /// Returns the maximum admissible κ-box radius.
    pub fn radius_max(&self) -> f32 {
        self.radius_max
    }

    /// Returns the Lawvere–Tierney guard used by the site.
    pub fn guard(&self) -> &LawvereTierneyGuard {
        &self.guard
    }

    /// Ensures a single κ-box is admissible.
    pub fn guard_box(&self, zbox: &ZBox) -> PureResult<()> {
        self.guarded_box_mass(zbox).map(|_| ())
    }

    fn guarded_box_mass(&self, zbox: &ZBox) -> PureResult<f32> {
        zbox.validate_radius_window(self.radius_min, self.radius_max)?;
        self.guard.guard_density(zbox.density())?;
        let mass = zbox.probability_mass(self.curvature)?;
        self.guard.guard_mass(mass)?;
        Ok(mass)
    }

    /// Ensures a cover of κ-boxes is admissible and mass-preserving.
    pub fn guard_cover(&self, cover: &[ZBox]) -> PureResult<()> {
        if cover.is_empty() {
            return Err(TensorError::EmptyInput("zbox_cover"));
        }
        let mut mass = 0.0f64;
        for zbox in cover.iter() {
            mass += self.guarded_box_mass(zbox)? as f64;
        }
        self.guard
            .guard_cover_mass(checked_f64_to_f32("lawvere_tierney_cover_mass", mass)?)
    }
}

fn hyperbolic_ball_volume(curvature: f32, radius: f32, dimension: usize) -> PureResult<f32> {
    if dimension == 0 {
        return Err(TensorError::EmptyInput("hyperbolic_dimension"));
    }
    if !curvature.is_finite() {
        return Err(TensorError::NonFiniteValue {
            label: "hyperbolic_curvature",
            value: curvature,
        });
    }
    if !radius.is_finite() {
        return Err(TensorError::NonFiniteValue {
            label: "hyperbolic_radius",
            value: radius,
        });
    }
    if radius <= 0.0 {
        return Err(TensorError::InvalidValue {
            label: "hyperbolic_radius_non_positive",
        });
    }
    let kappa = curvature;
    if kappa >= 0.0 {
        return Err(TensorError::NonHyperbolicCurvature { curvature: kappa });
    }
    let lambda = (-kappa).sqrt();
    let n = i32::try_from(dimension).map_err(|_| TensorError::InvalidValue {
        label: "hyperbolic_dimension_too_large",
    })?;
    let sphere_dimension = i32::try_from(dimension - 1).map_err(|_| TensorError::InvalidValue {
        label: "hyperbolic_dimension_too_large",
    })?;
    let omega = unit_sphere_surface(sphere_dimension);
    let steps = 64;
    let h = radius / steps as f32;
    let mut integral = 0.0f64;
    for i in 0..=steps {
        let t = i as f32 * h;
        let sinh_term = ((lambda * t) as f64).sinh() / lambda as f64;
        let power = if n == 1 {
            1.0
        } else {
            sinh_term.powi((n - 1).max(0))
        };
        let weight = if i == 0 || i == steps {
            1.0
        } else if i % 2 == 0 {
            2.0
        } else {
            4.0
        };
        integral += weight * power;
    }
    integral *= (h as f64) / 3.0;
    let volume = omega * integral;
    if !volume.is_finite() || volume <= 0.0 {
        return Err(TensorError::InvalidValue {
            label: "hyperbolic_volume",
        });
    }
    let volume = volume as f32;
    if !volume.is_finite() || volume <= 0.0 {
        return Err(TensorError::NonFiniteValue {
            label: "hyperbolic_volume_f32",
            value: volume,
        });
    }
    Ok(volume)
}

fn unit_sphere_surface(dimension: i32) -> f64 {
    match dimension {
        -1 => 2.0,
        0 => 2.0,
        1 => 2.0 * PI64,
        2 => 4.0 * PI64,
        d if d >= 3 => {
            let n = (d + 1) as f64;
            2.0 * PI64.powf(n / 2.0) / gamma(n / 2.0)
        }
        _ => 2.0,
    }
}

fn gamma(z: f64) -> f64 {
    if z == 0.5 {
        return PI64.sqrt();
    }
    if z == 1.0 {
        return 1.0;
    }
    if z == 2.0 {
        return 1.0;
    }
    if z.fract() == 0.5 {
        let n = (z - 0.5) as usize;
        let mut numerator = 1.0f64;
        for k in 1..=n {
            numerator *= (2 * k) as f64;
        }
        let mut denominator = 1.0f64;
        for k in 1..=n {
            denominator *= (2 * k - 1) as f64;
        }
        numerator / denominator * PI64.sqrt()
    } else if z.fract() == 0.0 {
        let n = z as usize - 1;
        (1..=n).fold(1.0f64, |acc, v| acc * v as f64)
    } else {
        lanczos_gamma(z)
    }
}

fn lanczos_gamma(z: f64) -> f64 {
    const G: f64 = 7.0;
    const P: [f64; 9] = [
        0.9999999999998099,
        676.5203681218851,
        -1259.1392167224028,
        771.3234287776531,
        -176.6150291621406,
        12.507343278686905,
        -0.13857109526572012,
        9.984369578019572e-6,
        1.5056327351493116e-7,
    ];
    if z < 0.5 {
        PI64 / ((PI64 * z).sin() * lanczos_gamma(1.0 - z))
    } else {
        let z = z - 1.0;
        let mut x = P[0];
        for (i, p) in P.iter().enumerate().skip(1) {
            x += p / (z + i as f64);
        }
        let t = z + G + 0.5;
        (2.0 * PI64).sqrt() * t.powf(z + 0.5) * (-t).exp() * x
    }
}

/// Maintains safety envelopes for tensors travelling through the pure stack.
#[derive(Clone, Debug)]
pub struct OpenCartesianTopos {
    curvature: f32,
    tolerance: f32,
    saturation: f32,
    porosity: f32,
    max_depth: usize,
    max_volume: usize,
    site: ZBoxSite,
}

/// Serializable reconstruction state for an [`OpenCartesianTopos`].
#[derive(Clone, Copy, Debug, serde::Deserialize, PartialEq, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct OpenCartesianToposCheckpoint {
    pub curvature: f32,
    pub tolerance: f32,
    pub saturation: f32,
    pub porosity: f32,
    pub max_depth: usize,
    pub max_volume: usize,
    pub site_radius_min: f32,
    pub site_radius_max: f32,
    pub guard_density_min: f32,
    pub guard_density_max: f32,
    pub guard_mass_tolerance: f32,
}

impl OpenCartesianToposCheckpoint {
    /// Validates every nested guard by reconstructing the live Rust object.
    pub fn validate(self) -> PureResult<()> {
        self.into_topos().map(|_| ())
    }

    /// Reconstructs the complete guard, including a custom site and porosity.
    pub fn into_topos(self) -> PureResult<OpenCartesianTopos> {
        let guard = LawvereTierneyGuard::new(
            self.guard_density_min,
            self.guard_density_max,
            self.guard_mass_tolerance,
        )?;
        let site = ZBoxSite::default_for(self.curvature)?
            .with_radius_window(self.site_radius_min, self.site_radius_max)?
            .with_guard(guard);
        OpenCartesianTopos::new(
            self.curvature,
            self.tolerance,
            self.saturation,
            self.max_depth,
            self.max_volume,
        )?
        .with_porosity(self.porosity)?
        .with_site(site)
    }
}

impl OpenCartesianTopos {
    /// Builds a new guard. `curvature` must remain negative, `tolerance` and
    /// `saturation` must be positive. `max_depth` and `max_volume` are expressed
    /// in absolute counts rather than logarithms so callers can wire them to
    /// dataset or topology specific limits.
    pub fn new(
        curvature: f32,
        tolerance: f32,
        saturation: f32,
        max_depth: usize,
        max_volume: usize,
    ) -> PureResult<Self> {
        if !curvature.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "topos_curvature",
                value: curvature,
            });
        }
        if !tolerance.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "topos_tolerance",
                value: tolerance,
            });
        }
        if !saturation.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "topos_saturation",
                value: saturation,
            });
        }
        if curvature >= 0.0 {
            return Err(TensorError::NonHyperbolicCurvature { curvature });
        }
        if tolerance <= 0.0 {
            return Err(TensorError::NonPositiveTolerance { tolerance });
        }
        if saturation <= 0.0 {
            return Err(TensorError::NonPositiveSaturation { saturation });
        }
        if max_depth == 0 {
            return Err(TensorError::EmptyInput("topos max depth"));
        }
        if max_volume == 0 {
            return Err(TensorError::EmptyInput("topos max volume"));
        }
        let site = ZBoxSite::default_for(curvature)?;
        Ok(Self {
            curvature,
            tolerance,
            saturation,
            porosity: 0.2,
            max_depth,
            max_volume,
            site,
        })
    }

    /// Captures all values that influence guard behavior.
    pub fn checkpoint(&self) -> OpenCartesianToposCheckpoint {
        let guard = self.site.guard();
        OpenCartesianToposCheckpoint {
            curvature: self.curvature,
            tolerance: self.tolerance,
            saturation: self.saturation,
            porosity: self.porosity,
            max_depth: self.max_depth,
            max_volume: self.max_volume,
            site_radius_min: self.site.radius_min(),
            site_radius_max: self.site.radius_max(),
            guard_density_min: guard.density_min(),
            guard_density_max: guard.density_max(),
            guard_mass_tolerance: guard.mass_tolerance(),
        }
    }

    /// Returns the curvature enforced by the topos.
    pub fn curvature(&self) -> f32 {
        self.curvature
    }

    /// Tolerance applied when inverting Jacobians or measuring residuals.
    pub fn tolerance(&self) -> f32 {
        self.tolerance
    }

    /// Returns the saturation limit used to absorb overflows.
    pub fn saturation(&self) -> f32 {
        self.saturation
    }

    /// Returns the permeability applied while saturating values.
    pub fn porosity(&self) -> f32 {
        self.porosity
    }

    /// Maximum permitted traversal depth before the guard considers the topos
    /// closed for the current rewrite.
    pub fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// Maximum tensor volume allowed inside the topos envelope.
    pub fn max_volume(&self) -> usize {
        self.max_volume
    }

    /// Returns the κ-box site guarded by this topos.
    pub fn site(&self) -> &ZBoxSite {
        &self.site
    }

    /// Replaces the κ-box site guard, returning a new topos instance.
    pub fn with_site(mut self, site: ZBoxSite) -> PureResult<Self> {
        if (site.curvature() - self.curvature).abs() > self.tolerance {
            return Err(TensorError::CurvatureMismatch {
                expected: self.curvature,
                got: site.curvature(),
            });
        }
        self.site = site;
        Ok(self)
    }

    /// Replaces the porosity used during saturation.
    pub fn with_porosity(mut self, porosity: f32) -> PureResult<Self> {
        if !porosity.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "topos_porosity",
                value: porosity,
            });
        }
        if !(0.0..=1.0).contains(&porosity) {
            return Err(TensorError::PorosityOutOfRange { porosity });
        }
        self.porosity = porosity;
        Ok(self)
    }

    /// Emits a zero-observation control signal for inference or training telemetry.
    pub fn control_signal(&self) -> ToposControlSignal {
        self.control_signal_for(0, 0)
    }

    /// Emits a control signal with explicit traversal and volume observations.
    pub fn control_signal_for(
        &self,
        observed_depth: usize,
        visited_volume: usize,
    ) -> ToposControlSignal {
        ToposControlSignal::from_observation(self, observed_depth, visited_volume)
    }

    /// Ensures the provided tensor stays finite and within the permitted volume.
    pub fn guard_tensor(&self, label: &'static str, tensor: &Tensor) -> PureResult<()> {
        let (rows, cols) = tensor.shape();
        let volume = checked_tensor_volume(rows, cols)?;
        if volume > self.max_volume {
            return Err(TensorError::TensorVolumeExceeded {
                label,
                volume,
                max_volume: self.max_volume,
            });
        }
        self.guard_slice(label, tensor.data())
    }

    /// Ensures the provided κ-box satisfies the site guard.
    pub fn guard_zbox(&self, zbox: &ZBox) -> PureResult<()> {
        self.site.guard_box(zbox)
    }

    /// Ensures a κ-box cover is admissible and mass-preserving.
    pub fn guard_cover(&self, cover: &[ZBox]) -> PureResult<()> {
        self.site.guard_cover(cover)
    }

    /// Ensures a buffer remains finite.
    pub fn guard_slice(&self, label: &'static str, slice: &[f32]) -> PureResult<()> {
        for &value in slice {
            if !value.is_finite() {
                return Err(TensorError::NonFiniteValue { label, value });
            }
        }
        Ok(())
    }

    /// Normalises a probability slice while keeping it within the topos saturation window.
    pub fn guard_probability_slice(
        &self,
        label: &'static str,
        slice: &mut [f32],
    ) -> PureResult<()> {
        self.site
            .guard()
            .project_slice(label, slice, self.saturation)
    }

    /// Normalises a probability tensor in-place.
    pub fn guard_probability_tensor(
        &self,
        label: &'static str,
        tensor: &mut Tensor,
    ) -> PureResult<()> {
        let (rows, cols) = tensor.shape();
        let volume = checked_tensor_volume(rows, cols)?;
        if volume > self.max_volume {
            return Err(TensorError::TensorVolumeExceeded {
                label,
                volume,
                max_volume: self.max_volume,
            });
        }
        let mut candidate = tensor.clone();
        self.guard_probability_slice(label, candidate.data_mut())?;
        self.guard_tensor(label, &candidate)?;
        *tensor = candidate;
        Ok(())
    }

    /// Catches runaway recursion depth before it can trigger a feedback loop.
    pub fn ensure_loop_free(&self, depth: usize) -> PureResult<()> {
        if depth >= self.max_depth {
            return Err(TensorError::LoopDetected {
                depth,
                max_depth: self.max_depth,
            });
        }
        Ok(())
    }

    /// Validates a fractal patch before it is ingested by other pure modules.
    pub fn guard_fractal_patch(&self, label: &'static str, patch: &FractalPatch) -> PureResult<()> {
        self.ensure_loop_free(patch.depth() as usize)?;
        self.guard_tensor(label, patch.relation())
    }

    /// Saturates a scalar into the finite window enforced by the topos.
    pub fn saturate(&self, value: f32) -> f32 {
        porous_mix(value, self.saturation, self.porosity)
    }

    /// Saturates a scalar and returns the exact local slope of that rewrite.
    ///
    /// The slope is the canonical reverse-mode rule for every Rust adapter that
    /// differentiates through the open-topos saturation boundary.
    pub fn saturate_with_slope(&self, value: f32) -> (f32, f32) {
        (
            porous_mix(value, self.saturation, self.porosity),
            porous_mix_slope(value, self.saturation, self.porosity),
        )
    }

    /// Saturates an entire slice in-place.
    pub fn saturate_slice(&self, slice: &mut [f32]) {
        for value in slice.iter_mut() {
            *value = self.saturate(*value);
        }
    }
}

/// Local envelope that specialises an [`OpenCartesianTopos`] guard for a specific modality.
#[derive(Clone, Copy, Debug)]
pub struct ModalityProfile {
    max_volume: usize,
    local_saturation: Option<f32>,
    permeability: f32,
}

impl ModalityProfile {
    /// Creates a new modality envelope. `max_volume` must be non-zero and `local_saturation`,
    /// when provided, must be positive.
    pub fn new(max_volume: usize, local_saturation: Option<f32>) -> PureResult<Self> {
        if max_volume == 0 {
            return Err(TensorError::EmptyInput("modality_max_volume"));
        }
        if let Some(saturation) = local_saturation {
            if !saturation.is_finite() {
                return Err(TensorError::NonFiniteValue {
                    label: "modality_local_saturation",
                    value: saturation,
                });
            }
            if saturation <= 0.0 {
                return Err(TensorError::NonPositiveSaturation { saturation });
            }
        }
        Ok(Self {
            max_volume,
            local_saturation,
            permeability: DEFAULT_MODALITY_PERMEABILITY,
        })
    }

    /// Returns the maximum tensor volume admitted by the modality.
    pub fn max_volume(&self) -> usize {
        self.max_volume
    }

    /// Returns the optional modality-local saturation cap.
    pub fn local_saturation(&self) -> Option<f32> {
        self.local_saturation
    }

    /// Returns the permeability applied when clamping modality values.
    pub fn permeability(&self) -> f32 {
        self.permeability
    }

    /// Overrides the permeability while preserving other settings.
    pub fn with_permeability(mut self, permeability: f32) -> PureResult<Self> {
        validate_permeability("modality_permeability", permeability)?;
        self.permeability = permeability;
        Ok(self)
    }

    fn effective_saturation(&self, topos: &OpenCartesianTopos) -> f32 {
        self.local_saturation
            .unwrap_or_else(|| topos.saturation())
            .min(topos.saturation())
            .max(0.0)
    }

    fn guard_volume(&self, label: &'static str, volume: usize) -> PureResult<()> {
        if volume > self.max_volume {
            return Err(TensorError::TensorVolumeExceeded {
                label,
                volume,
                max_volume: self.max_volume,
            });
        }
        Ok(())
    }

    fn rewrite_slice(
        &self,
        topos: &OpenCartesianTopos,
        label: &'static str,
        slice: &mut [f32],
    ) -> PureResult<()> {
        let saturation = self.effective_saturation(topos);
        for &value in slice.iter() {
            if !value.is_finite() {
                return Err(TensorError::NonFiniteValue { label, value });
            }
        }
        for value in slice.iter_mut() {
            *value = permeable_clamp(*value, saturation, self.permeability);
        }
        topos.guard_slice(label, slice)
    }

    fn rewrite_tensor(
        &self,
        topos: &OpenCartesianTopos,
        label: &'static str,
        tensor: &mut Tensor,
    ) -> PureResult<()> {
        let (rows, cols) = tensor.shape();
        self.guard_volume(label, checked_tensor_volume(rows, cols)?)?;
        self.rewrite_slice(topos, label, tensor.data_mut())?;
        topos.guard_tensor(label, tensor)
    }

    fn validate_for_topos(&self, topos: &OpenCartesianTopos) -> PureResult<()> {
        if self.max_volume > topos.max_volume() {
            return Err(TensorError::InvalidValue {
                label: "modality_volume_exceeds_topos",
            });
        }
        Ok(())
    }
}

/// Guard configuration for structured graph data rewritten through an open-cartesian topos.
#[derive(Clone, Copy, Debug)]
pub struct GraphGuardProfile {
    max_nodes: usize,
    max_edges: usize,
    max_degree: usize,
    symmetry_tolerance: f32,
    activation_threshold: f32,
    edge_saturation: Option<f32>,
    permeability: f32,
}

impl GraphGuardProfile {
    /// Creates a graph guard profile. Node and edge budgets must be positive; a zero
    /// maximum degree is valid for an explicitly edgeless graph.
    pub fn new(
        max_nodes: usize,
        max_edges: usize,
        max_degree: usize,
        symmetry_tolerance: f32,
        activation_threshold: f32,
        edge_saturation: Option<f32>,
    ) -> PureResult<Self> {
        if max_nodes == 0 {
            return Err(TensorError::EmptyInput("graph_max_nodes"));
        }
        if max_edges == 0 {
            return Err(TensorError::EmptyInput("graph_max_edges"));
        }
        if max_degree > max_nodes.saturating_sub(1) {
            return Err(TensorError::InvalidValue {
                label: "graph_max_degree_exceeds_nodes",
            });
        }
        if !symmetry_tolerance.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "graph_symmetry_tolerance",
                value: symmetry_tolerance,
            });
        }
        if symmetry_tolerance < 0.0 {
            return Err(TensorError::InvalidValue {
                label: "graph_symmetry_tolerance",
            });
        }
        if !activation_threshold.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "graph_activation_threshold",
                value: activation_threshold,
            });
        }
        if activation_threshold <= 0.0 {
            return Err(TensorError::InvalidValue {
                label: "graph_activation_threshold",
            });
        }
        if let Some(saturation) = edge_saturation {
            if !saturation.is_finite() {
                return Err(TensorError::NonFiniteValue {
                    label: "graph_edge_saturation",
                    value: saturation,
                });
            }
            if saturation <= 0.0 {
                return Err(TensorError::NonPositiveSaturation { saturation });
            }
        }
        Ok(Self {
            max_nodes,
            max_edges,
            max_degree,
            symmetry_tolerance,
            activation_threshold,
            edge_saturation,
            permeability: DEFAULT_GRAPH_PERMEABILITY,
        })
    }

    /// Maximum number of graph nodes admitted by the guard.
    pub fn max_nodes(&self) -> usize {
        self.max_nodes
    }

    /// Maximum number of active undirected edges admitted by the guard.
    pub fn max_edges(&self) -> usize {
        self.max_edges
    }

    /// Maximum off-diagonal neighbour degree admitted per node.
    pub fn max_degree(&self) -> usize {
        self.max_degree
    }

    /// Symmetry tolerance applied after adjacency canonicalization.
    pub fn symmetry_tolerance(&self) -> f32 {
        self.symmetry_tolerance
    }

    /// Strictly positive absolute weight that activates an edge.
    pub fn activation_threshold(&self) -> f32 {
        self.activation_threshold
    }

    /// Optional graph-local edge saturation cap.
    pub fn edge_saturation(&self) -> Option<f32> {
        self.edge_saturation
    }

    /// Returns the permeability admitted while guarding adjacency tensors.
    pub fn permeability(&self) -> f32 {
        self.permeability
    }

    fn validate_for_topos(&self, topos: &OpenCartesianTopos) -> PureResult<()> {
        let max_possible = topos.max_volume();
        let adjacency_volume = self.expected_len(self.max_nodes)?;
        if adjacency_volume > max_possible {
            return Err(TensorError::TensorVolumeExceeded {
                label: "graph_adjacency_profile",
                volume: adjacency_volume,
                max_volume: max_possible,
            });
        }
        if self.max_edges > max_possible {
            return Err(TensorError::InvalidValue {
                label: "graph_edges_exceed_topos_volume",
            });
        }
        Ok(())
    }

    fn expected_len(&self, node_count: usize) -> PureResult<usize> {
        node_count
            .checked_mul(node_count)
            .ok_or(TensorError::InvalidDimensions {
                rows: node_count,
                cols: node_count,
            })
    }

    fn saturate_value(&self, topos: &OpenCartesianTopos, value: f32) -> f32 {
        if !value.is_finite() {
            return 0.0;
        }
        let limit = self
            .edge_saturation
            .unwrap_or_else(|| topos.saturation())
            .min(topos.saturation())
            .max(0.0);
        permeable_clamp(value, limit, self.permeability)
    }

    fn guard_shape(&self, node_count: usize, len: usize) -> PureResult<usize> {
        if node_count == 0 {
            return Err(TensorError::EmptyInput("graph_nodes"));
        }
        if node_count > self.max_nodes {
            return Err(TensorError::InvalidValue {
                label: "graph_max_nodes_exceeded",
            });
        }
        let expected = self.expected_len(node_count)?;
        if len != expected {
            return Err(TensorError::DataLength { expected, got: len });
        }
        Ok(expected)
    }

    fn guard_degree(&self, degree: usize) -> PureResult<()> {
        if degree > self.max_degree {
            return Err(TensorError::InvalidValue {
                label: "graph_degree_exceeded",
            });
        }
        Ok(())
    }

    fn guard_edge_budget(&self, edge_count: usize) -> PureResult<usize> {
        if edge_count <= self.max_edges {
            return Ok(0);
        }
        let slack = ((self.max_edges as f64) * self.permeability as f64).ceil() as usize;
        let overflow = edge_count - self.max_edges;
        if overflow > slack {
            return Err(TensorError::TensorVolumeExceeded {
                label: "graph_edge_budget",
                volume: edge_count,
                max_volume: self.max_edges,
            });
        }
        Ok(overflow)
    }

    /// Overrides the permeability used when smoothing adjacency saturation and edge budgets.
    pub fn with_permeability(mut self, permeability: f32) -> PureResult<Self> {
        validate_permeability("graph_permeability", permeability)?;
        self.permeability = permeability;
        Ok(self)
    }

    /// Configures the porosity for adjacency saturation.
    ///
    /// This is an alias for [`with_permeability`](Self::with_permeability) that mirrors the
    /// terminology used by higher level guards and bindings.
    pub fn with_porosity(mut self, porosity: f32) -> PureResult<Self> {
        if !porosity.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "graph_porosity",
                value: porosity,
            });
        }
        if !(0.0..=1.0).contains(&porosity) {
            return Err(TensorError::PorosityOutOfRange { porosity });
        }
        self.permeability = porosity;
        Ok(self)
    }
}

/// Structured report emitted by [`MultiModalToposGuard::guard_graph_adjacency`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GraphGuardReport {
    /// Number of undirected edges whose absolute weight exceeded the activation threshold.
    pub edge_count: usize,
    /// Maximum node degree observed while guarding the adjacency matrix.
    pub max_degree: usize,
    /// Number of asymmetric edge pairs detected beyond the configured tolerance.
    pub symmetry_violations: usize,
    /// Number of active diagonal self-loop entries.
    pub self_loops: usize,
    /// Number of non-finite entries repaired to zero before graph measurements.
    pub repaired_non_finite: usize,
    /// Number of entries that were saturated into the admissible range.
    pub saturated_entries: usize,
    /// Number of edges exceeding the configured budget but tolerated by permeability.
    pub edge_overflow: usize,
}

/// Reward boundary used to detect runaway reinforcement learning signals.
#[derive(Clone, Copy, Debug)]
pub struct RewardBoundary {
    lower: f32,
    upper: f32,
    hysteresis: f32,
    porosity: f32,
}

impl RewardBoundary {
    /// Creates a new reward boundary. `lower` must be below `upper` and hysteresis must be
    /// non-negative.
    pub fn new(lower: f32, upper: f32, hysteresis: f32) -> PureResult<Self> {
        for (label, value) in [
            ("reward_boundary_lower", lower),
            ("reward_boundary_upper", upper),
            ("reward_boundary_hysteresis", hysteresis),
        ] {
            if !value.is_finite() {
                return Err(TensorError::NonFiniteValue { label, value });
            }
        }
        if lower >= upper {
            return Err(TensorError::InvalidValue {
                label: "reward_boundary_window",
            });
        }
        if hysteresis < 0.0 {
            return Err(TensorError::InvalidValue {
                label: "reward_boundary_hysteresis",
            });
        }
        Ok(Self {
            lower,
            upper,
            hysteresis,
            porosity: 0.0,
        })
    }

    /// Creates a new reward boundary with the provided porosity.
    pub fn with_porosity(
        lower: f32,
        upper: f32,
        hysteresis: f32,
        porosity: f32,
    ) -> PureResult<Self> {
        if !porosity.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "reward_boundary_porosity",
                value: porosity,
            });
        }
        if !(0.0..=1.0).contains(&porosity) {
            return Err(TensorError::PorosityOutOfRange { porosity });
        }
        let mut boundary = Self::new(lower, upper, hysteresis)?;
        boundary.porosity = porosity;
        Ok(boundary)
    }

    /// Returns the lower reward bound.
    pub fn lower(&self) -> f32 {
        self.lower
    }

    /// Returns the upper reward bound.
    pub fn upper(&self) -> f32 {
        self.upper
    }

    /// Returns the hysteresis applied outside each reward bound.
    pub fn hysteresis(&self) -> f32 {
        self.hysteresis
    }

    /// Returns the configured porosity.
    pub fn porosity(&self) -> f32 {
        self.porosity
    }

    fn clamp(&self, value: f32) -> f32 {
        if value > self.upper {
            self.soft_clamp(value, self.upper, true)
        } else if value < self.lower {
            self.soft_clamp(value, self.lower, false)
        } else {
            value
        }
    }

    fn soft_clamp(&self, value: f32, bound: f32, upper: bool) -> f32 {
        if self.porosity <= f32::EPSILON {
            return bound;
        }
        let span = self.upper - self.lower;
        let excess = if upper { value - bound } else { bound - value };
        let bleed = excess / (excess + span);
        let inward = span * (self.porosity * 0.25) * bleed.clamp(0.0, 1.0);
        if upper {
            bound - inward
        } else {
            bound + inward
        }
    }

    fn breached_lower(&self, value: f32) -> bool {
        value < self.lower - self.hysteresis * (1.0 + self.porosity)
    }

    fn breached_upper(&self, value: f32) -> bool {
        value > self.upper + self.hysteresis * (1.0 + self.porosity)
    }

    fn validate_for_topos(&self, topos: &OpenCartesianTopos) -> PureResult<()> {
        if self.lower < -topos.saturation() || self.upper > topos.saturation() {
            return Err(TensorError::InvalidValue {
                label: "reward_boundary_exceeds_topos_saturation",
            });
        }
        Ok(())
    }
}

/// Signal emitted when reward traces cross configured boundaries.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RewardBoundarySignal {
    /// Index of the first sample that breached the lower boundary, if any.
    pub lower_breach_index: Option<usize>,
    /// Index of the first sample that breached the upper boundary, if any.
    pub upper_breach_index: Option<usize>,
    /// Number of reward samples that were clamped back into the admissible window.
    pub clamped: usize,
    /// Minimum reward observed before clamping.
    pub min_observed: f32,
    /// Maximum reward observed before clamping.
    pub max_observed: f32,
}

impl Default for RewardBoundarySignal {
    fn default() -> Self {
        Self {
            lower_breach_index: None,
            upper_breach_index: None,
            clamped: 0,
            min_observed: f32::INFINITY,
            max_observed: f32::NEG_INFINITY,
        }
    }
}

impl RewardBoundarySignal {
    /// Finalises the observed range by normalising infinities for empty traces.
    fn finalise(mut self) -> Self {
        if !self.min_observed.is_finite() {
            self.min_observed = 0.0;
        }
        if !self.max_observed.is_finite() {
            self.max_observed = 0.0;
        }
        self
    }
}

/// Multi-modal guard that specialises an [`OpenCartesianTopos`] for audio, vision, text, graph,
/// and reinforcement-learning reward traces.
#[derive(Clone, Copy, Debug)]
pub struct MultiModalToposGuard<'a> {
    topos: &'a OpenCartesianTopos,
    text: ModalityProfile,
    audio: ModalityProfile,
    vision: ModalityProfile,
    graph: GraphGuardProfile,
    reward: RewardBoundary,
}

impl<'a> MultiModalToposGuard<'a> {
    /// Builds a multi-modal guard from explicit modality profiles and a reward boundary.
    pub fn from_profiles(
        topos: &'a OpenCartesianTopos,
        text: ModalityProfile,
        audio: ModalityProfile,
        vision: ModalityProfile,
        graph: GraphGuardProfile,
        reward: RewardBoundary,
    ) -> PureResult<Self> {
        graph.validate_for_topos(topos)?;
        reward.validate_for_topos(topos)?;
        text.validate_for_topos(topos)?;
        audio.validate_for_topos(topos)?;
        vision.validate_for_topos(topos)?;
        Ok(Self {
            topos,
            text,
            audio,
            vision,
            graph,
            reward,
        })
    }

    /// Builds a new multi-modal guard anchored to the provided topos with conservative defaults.
    pub fn new(topos: &'a OpenCartesianTopos) -> PureResult<Self> {
        let text = ModalityProfile::new(topos.max_volume().clamp(1, 16384), None)?
            .with_permeability(0.18)?;
        let audio =
            ModalityProfile::new(topos.max_volume().clamp(1, 65536), Some(topos.saturation()))?
                .with_permeability(0.1)?;
        let vision = ModalityProfile::new(topos.max_volume().clamp(1, 262144), None)?
            .with_permeability(0.15)?;
        let graph_max_nodes = cmp::min(2048, square_root_floor(topos.max_volume())).max(1);
        let graph = GraphGuardProfile::new(
            graph_max_nodes,
            cmp::min(topos.max_volume(), 131_072),
            cmp::min(1024, graph_max_nodes.saturating_sub(1)),
            1e-3,
            1e-4,
            None,
        )?
        .with_permeability(0.1)?;
        let reward = RewardBoundary::new(-topos.saturation(), topos.saturation(), 0.05)?;
        Self::from_profiles(topos, text, audio, vision, graph, reward)
    }

    /// Returns the underlying topos guard.
    pub fn topos(&self) -> &'a OpenCartesianTopos {
        self.topos
    }

    /// Returns the active text modality profile.
    pub fn text_profile(&self) -> ModalityProfile {
        self.text
    }

    /// Returns the active audio modality profile.
    pub fn audio_profile(&self) -> ModalityProfile {
        self.audio
    }

    /// Returns the active vision modality profile.
    pub fn vision_profile(&self) -> ModalityProfile {
        self.vision
    }

    /// Returns the active graph guard profile.
    pub fn graph_profile(&self) -> GraphGuardProfile {
        self.graph
    }

    /// Returns the active reward boundary.
    pub fn reward_boundary(&self) -> RewardBoundary {
        self.reward
    }

    /// Returns a rewrite monad anchored to the multi-modal guard.
    pub fn monad(&self) -> RewriteMonad<'a> {
        RewriteMonad::new(self.topos)
    }

    /// Creates a multi-modal atlas that shares this guard's modality envelopes.
    pub fn atlas(&self) -> MultiModalAtlas<'a> {
        MultiModalAtlas::new(*self)
    }

    /// Cultivates a tensor biome that preserves the guard's modality profiles.
    pub fn cultivate_biome(&self) -> MultiModalBiome {
        MultiModalBiome::new(*self)
    }

    /// Overrides the text modality profile.
    pub fn with_text_profile(mut self, profile: ModalityProfile) -> PureResult<Self> {
        profile.validate_for_topos(self.topos)?;
        self.text = profile;
        Ok(self)
    }

    /// Overrides the audio modality profile.
    pub fn with_audio_profile(mut self, profile: ModalityProfile) -> PureResult<Self> {
        profile.validate_for_topos(self.topos)?;
        self.audio = profile;
        Ok(self)
    }

    /// Overrides the vision modality profile.
    pub fn with_vision_profile(mut self, profile: ModalityProfile) -> PureResult<Self> {
        profile.validate_for_topos(self.topos)?;
        self.vision = profile;
        Ok(self)
    }

    /// Overrides the graph guard profile.
    pub fn with_graph_profile(mut self, profile: GraphGuardProfile) -> PureResult<Self> {
        profile.validate_for_topos(self.topos)?;
        self.graph = profile;
        Ok(self)
    }

    /// Overrides the reward boundary detector.
    pub fn with_reward_boundary(mut self, boundary: RewardBoundary) -> PureResult<Self> {
        boundary.validate_for_topos(self.topos)?;
        self.reward = boundary;
        Ok(self)
    }

    /// Guards language or text tensors by applying modality-specific saturation before
    /// delegating to the underlying topos guard.
    pub fn guard_text_tensor(&self, label: &'static str, tensor: &mut Tensor) -> PureResult<()> {
        self.text.rewrite_tensor(self.topos, label, tensor)
    }

    /// Guards audio waveforms represented as contiguous slices.
    pub fn guard_audio_waveform(
        &self,
        label: &'static str,
        waveform: &mut [f32],
    ) -> PureResult<()> {
        self.audio.guard_volume(label, waveform.len())?;
        self.audio.rewrite_slice(self.topos, label, waveform)
    }

    /// Guards image or vision tensors by applying the configured vision profile.
    pub fn guard_vision_tensor(&self, label: &'static str, tensor: &mut Tensor) -> PureResult<()> {
        self.vision.rewrite_tensor(self.topos, label, tensor)
    }

    /// Guards graph adjacency matrices and reports symmetry and activation statistics.
    pub fn guard_graph_adjacency(
        &self,
        adjacency: &mut [f32],
        node_count: usize,
    ) -> PureResult<GraphGuardReport> {
        let adjacency_volume = self.graph.guard_shape(node_count, adjacency.len())?;
        if adjacency_volume > self.topos.max_volume() {
            return Err(TensorError::TensorVolumeExceeded {
                label: "graph_adjacency",
                volume: adjacency_volume,
                max_volume: self.topos.max_volume(),
            });
        }
        let mut canonical = Vec::with_capacity(adjacency.len());
        let mut repaired_non_finite = 0usize;
        let mut saturated_entries = 0usize;
        for &value in adjacency.iter() {
            let finite = if value.is_finite() {
                value
            } else {
                repaired_non_finite = repaired_non_finite.saturating_add(1);
                0.0
            };
            let clamped = self.graph.saturate_value(self.topos, finite);
            if clamped != finite {
                saturated_entries = saturated_entries.saturating_add(1);
            }
            canonical.push(clamped);
        }
        self.topos.guard_slice("graph_adjacency", &canonical)?;

        let mut edge_count = 0usize;
        let mut max_degree = 0usize;
        let mut symmetry_violations = 0usize;
        let mut self_loops = 0usize;
        for i in 0..node_count {
            let diagonal = canonical[i * node_count + i];
            if diagonal.abs() >= self.graph.activation_threshold {
                self_loops = self_loops.saturating_add(1);
            }
            let mut degree = 0usize;
            for j in 0..node_count {
                if i == j {
                    continue;
                }
                let forward = canonical[i * node_count + j];
                let reverse = canonical[j * node_count + i];
                if forward.abs().max(reverse.abs()) >= self.graph.activation_threshold {
                    degree = degree.saturating_add(1);
                    if i < j {
                        edge_count = edge_count.saturating_add(1);
                    }
                }
                if i < j && (forward - reverse).abs() > self.graph.symmetry_tolerance {
                    symmetry_violations = symmetry_violations.saturating_add(1);
                }
            }
            self.graph.guard_degree(degree)?;
            max_degree = cmp::max(max_degree, degree);
        }
        let overflow = self.graph.guard_edge_budget(edge_count)?;
        adjacency.copy_from_slice(&canonical);
        Ok(GraphGuardReport {
            edge_count,
            max_degree,
            symmetry_violations,
            self_loops,
            repaired_non_finite,
            saturated_entries,
            edge_overflow: overflow,
        })
    }

    /// Guards reinforcement learning reward traces and detects boundary breaches.
    pub fn guard_reward_trace(&self, rewards: &mut [f32]) -> PureResult<RewardBoundarySignal> {
        if rewards.is_empty() {
            return Err(TensorError::EmptyInput("reward_trace"));
        }
        for &reward in rewards.iter() {
            if !reward.is_finite() {
                return Err(TensorError::NonFiniteValue {
                    label: "reward_trace",
                    value: reward,
                });
            }
        }

        let mut signal = RewardBoundarySignal::default();
        let mut guarded = Vec::with_capacity(rewards.len());
        for (idx, &reward) in rewards.iter().enumerate() {
            signal.min_observed = signal.min_observed.min(reward);
            signal.max_observed = signal.max_observed.max(reward);
            if self.reward.breached_lower(reward) && signal.lower_breach_index.is_none() {
                signal.lower_breach_index = Some(idx);
            }
            if self.reward.breached_upper(reward) && signal.upper_breach_index.is_none() {
                signal.upper_breach_index = Some(idx);
            }
            let saturated = self.topos.saturate(reward);
            let clamped = self.reward.clamp(saturated);
            if (clamped - reward).abs() > f32::EPSILON {
                signal.clamped = signal.clamped.saturating_add(1);
            }
            guarded.push(clamped);
        }
        self.topos.guard_slice("reward_trace", &guarded)?;
        rewards.copy_from_slice(&guarded);
        Ok(signal.finalise())
    }
}

/// Atlas that keeps track of modality-aware traversals across an open-cartesian topos.
#[derive(Clone, Debug)]
pub struct MultiModalAtlas<'a> {
    guard: MultiModalToposGuard<'a>,
    atlas: ToposAtlas<'a>,
}

impl<'a> MultiModalAtlas<'a> {
    pub(crate) fn from_parts(guard: MultiModalToposGuard<'a>, atlas: ToposAtlas<'a>) -> Self {
        Self { guard, atlas }
    }

    /// Creates a fresh multi-modal atlas from a guard.
    pub fn new(guard: MultiModalToposGuard<'a>) -> Self {
        Self::from_parts(guard, ToposAtlas::new(guard.topos))
    }

    /// Returns the underlying topos.
    pub fn topos(&self) -> &'a OpenCartesianTopos {
        self.guard.topos
    }

    /// Returns the rewrite monad anchored to this atlas.
    pub fn monad(&self) -> RewriteMonad<'a> {
        self.atlas.monad()
    }

    /// Returns the visited depth recorded by the underlying atlas.
    pub fn depth(&self) -> usize {
        self.atlas.depth()
    }

    /// Returns the accumulated tensor volume rewritten through this atlas.
    pub fn visited_volume(&self) -> usize {
        self.atlas.visited_volume()
    }

    /// Remaining admissible volume before the guard saturates.
    pub fn remaining_volume(&self) -> usize {
        self.atlas.remaining_volume()
    }

    /// Borrows the inner atlas.
    pub fn inner(&self) -> &ToposAtlas<'a> {
        &self.atlas
    }

    /// Borrows the inner atlas mutably.
    pub fn inner_mut(&mut self) -> &mut ToposAtlas<'a> {
        &mut self.atlas
    }

    /// Consumes the multi-modal wrapper and returns the underlying atlas.
    pub fn into_inner(self) -> ToposAtlas<'a> {
        self.atlas
    }

    /// Guards language tensors while tracking atlas traversal.
    pub fn guard_text_tensor(
        &mut self,
        label: &'static str,
        tensor: &mut Tensor,
    ) -> PureResult<()> {
        let mut candidate = tensor.clone();
        self.guard
            .text
            .rewrite_tensor(self.guard.topos, label, &mut candidate)?;
        self.atlas.guard_tensor(label, &candidate)?;
        *tensor = candidate;
        Ok(())
    }

    /// Lifts a text tensor through the atlas after applying modality saturation.
    pub fn lift_text_tensor(&mut self, label: &'static str, tensor: Tensor) -> PureResult<Tensor> {
        let mut tensor = tensor;
        self.guard
            .text
            .rewrite_tensor(self.guard.topos, label, &mut tensor)?;
        self.atlas.lift_tensor(label, tensor)
    }

    /// Guards audio waveforms using the atlas.
    pub fn guard_audio_waveform(
        &mut self,
        label: &'static str,
        waveform: &mut [f32],
    ) -> PureResult<()> {
        self.guard.audio.guard_volume(label, waveform.len())?;
        let mut candidate = waveform.to_vec();
        self.guard
            .audio
            .rewrite_slice(self.guard.topos, label, &mut candidate)?;
        self.atlas.guard_slice(label, &candidate)?;
        waveform.copy_from_slice(&candidate);
        Ok(())
    }

    /// Guards vision tensors while accounting for atlas traversal volume.
    pub fn guard_vision_tensor(
        &mut self,
        label: &'static str,
        tensor: &mut Tensor,
    ) -> PureResult<()> {
        let mut candidate = tensor.clone();
        self.guard
            .vision
            .rewrite_tensor(self.guard.topos, label, &mut candidate)?;
        self.atlas.guard_tensor(label, &candidate)?;
        *tensor = candidate;
        Ok(())
    }

    /// Lifts a vision tensor through the atlas after modality saturation.
    pub fn lift_vision_tensor(
        &mut self,
        label: &'static str,
        tensor: Tensor,
    ) -> PureResult<Tensor> {
        let mut tensor = tensor;
        self.guard
            .vision
            .rewrite_tensor(self.guard.topos, label, &mut tensor)?;
        self.atlas.lift_tensor(label, tensor)
    }

    /// Guards graph adjacency matrices and reuses the atlas guard for the resulting slice.
    pub fn guard_graph_adjacency(
        &mut self,
        adjacency: &mut [f32],
        node_count: usize,
    ) -> PureResult<GraphGuardReport> {
        let report = self.guard.guard_graph_adjacency(adjacency, node_count)?;
        self.atlas.guard_slice("graph_adjacency", adjacency)?;
        Ok(report)
    }

    /// Guards reinforcement-learning reward traces through the atlas.
    pub fn guard_reward_trace(&mut self, rewards: &mut [f32]) -> PureResult<RewardBoundarySignal> {
        let signal = self.guard.guard_reward_trace(rewards)?;
        self.atlas.guard_slice("reward_trace", rewards)?;
        Ok(signal)
    }

    /// Guards a fractal patch by delegating to the underlying atlas.
    pub fn guard_fractal_patch(
        &mut self,
        label: &'static str,
        patch: &FractalPatch,
    ) -> PureResult<()> {
        self.atlas.guard_fractal_patch(label, patch)
    }

    /// Emits the current multi-modal traversal pressure as a Z-space control signal.
    pub fn control_signal(&self) -> ToposControlSignal {
        self.atlas.control_signal()
    }
}

/// Tensor biome that preserves modality envelopes while absorbing shoots.
#[derive(Clone, Debug)]
pub struct MultiModalBiome {
    biome: TensorBiome,
    text: ModalityProfile,
    audio: ModalityProfile,
    vision: ModalityProfile,
    graph: GraphGuardProfile,
    reward: RewardBoundary,
}

impl MultiModalBiome {
    /// Cultivates a biome from the provided guard.
    pub fn new(guard: MultiModalToposGuard<'_>) -> Self {
        Self {
            biome: TensorBiome::new(guard.topos.clone()),
            text: guard.text,
            audio: guard.audio,
            vision: guard.vision,
            graph: guard.graph,
            reward: guard.reward,
        }
    }

    fn guard_ref(&self) -> MultiModalToposGuard<'_> {
        MultiModalToposGuard {
            topos: self.biome.topos(),
            text: self.text,
            audio: self.audio,
            vision: self.vision,
            graph: self.graph,
            reward: self.reward,
        }
    }

    fn absorb_with_profile(
        &mut self,
        label: &'static str,
        mut tensor: Tensor,
        profile: ModalityProfile,
        weight: f32,
    ) -> PureResult<()> {
        let guard = self.guard_ref();
        profile.rewrite_tensor(guard.topos, label, &mut tensor)?;
        self.biome.absorb_weighted(label, tensor, weight)
    }

    /// Returns the underlying open-cartesian topos.
    pub fn topos(&self) -> &OpenCartesianTopos {
        self.biome.topos()
    }

    /// Returns the text profile preserved by this biome.
    pub fn text_profile(&self) -> ModalityProfile {
        self.text
    }

    /// Returns the audio profile preserved by this biome.
    pub fn audio_profile(&self) -> ModalityProfile {
        self.audio
    }

    /// Returns the vision profile preserved by this biome.
    pub fn vision_profile(&self) -> ModalityProfile {
        self.vision
    }

    /// Returns the graph profile preserved by this biome.
    pub fn graph_profile(&self) -> GraphGuardProfile {
        self.graph
    }

    /// Returns the reward boundary preserved by this biome.
    pub fn reward_boundary(&self) -> RewardBoundary {
        self.reward
    }

    /// Returns a rewrite monad anchored to the biome's guard.
    pub fn monad(&self) -> RewriteMonad<'_> {
        self.biome.monad()
    }

    /// Creates a multi-modal atlas sharing the biome's profiles.
    pub fn atlas(&self) -> MultiModalAtlas<'_> {
        MultiModalAtlas::from_parts(self.guard_ref(), self.biome.atlas())
    }

    /// Emits a control signal derived from the tensors currently absorbed by the biome.
    pub fn control_signal(&self) -> ToposControlSignal {
        self.biome.control_signal()
    }

    /// Absorbs a text tensor into the biome.
    pub fn absorb_text(&mut self, label: &'static str, tensor: Tensor) -> PureResult<()> {
        self.absorb_text_weighted(label, tensor, 1.0)
    }

    /// Absorbs a text tensor with an explicit weight.
    pub fn absorb_text_weighted(
        &mut self,
        label: &'static str,
        tensor: Tensor,
        weight: f32,
    ) -> PureResult<()> {
        self.absorb_with_profile(label, tensor, self.text, weight)
    }

    /// Absorbs a vision tensor into the biome.
    pub fn absorb_vision(&mut self, label: &'static str, tensor: Tensor) -> PureResult<()> {
        self.absorb_vision_weighted(label, tensor, 1.0)
    }

    /// Absorbs a vision tensor with an explicit weight.
    pub fn absorb_vision_weighted(
        &mut self,
        label: &'static str,
        tensor: Tensor,
        weight: f32,
    ) -> PureResult<()> {
        self.absorb_with_profile(label, tensor, self.vision, weight)
    }

    /// Absorbs an audio tensor into the biome.
    pub fn absorb_audio(&mut self, label: &'static str, tensor: Tensor) -> PureResult<()> {
        self.absorb_audio_weighted(label, tensor, 1.0)
    }

    /// Absorbs an audio tensor with an explicit weight.
    pub fn absorb_audio_weighted(
        &mut self,
        label: &'static str,
        tensor: Tensor,
        weight: f32,
    ) -> PureResult<()> {
        self.absorb_with_profile(label, tensor, self.audio, weight)
    }

    /// Delegates graph guarding to the preserved profiles.
    pub fn guard_graph_adjacency(
        &self,
        adjacency: &mut [f32],
        node_count: usize,
    ) -> PureResult<GraphGuardReport> {
        self.guard_ref()
            .guard_graph_adjacency(adjacency, node_count)
    }

    /// Guards reward traces through the biome's guard.
    pub fn guard_reward_trace(&self, rewards: &mut [f32]) -> PureResult<RewardBoundarySignal> {
        self.guard_ref().guard_reward_trace(rewards)
    }

    /// Borrows the inner tensor biome.
    pub fn inner(&self) -> &TensorBiome {
        &self.biome
    }

    /// Borrows the inner tensor biome mutably.
    pub fn inner_mut(&mut self) -> &mut TensorBiome {
        &mut self.biome
    }

    /// Consumes the wrapper and returns the inner tensor biome.
    pub fn into_inner(self) -> TensorBiome {
        self.biome
    }

    /// Number of shoots currently stored in the biome.
    pub fn len(&self) -> usize {
        self.biome.len()
    }

    /// Whether the biome has absorbed any shoots.
    pub fn is_empty(&self) -> bool {
        self.biome.is_empty()
    }

    /// Total accumulated weight across all shoots.
    pub fn total_weight(&self) -> f32 {
        self.biome.total_weight()
    }

    /// Common tensor shape shared by all retained shoots.
    pub fn shape(&self) -> Option<(usize, usize)> {
        self.biome.shape()
    }

    /// Number of tensor values currently retained by the biome.
    pub fn stored_volume(&self) -> usize {
        self.biome.stored_volume()
    }

    /// Remaining aggregate storage volume admitted by the biome's topos.
    pub fn remaining_volume(&self) -> usize {
        self.biome.remaining_volume()
    }

    /// Returns the weights assigned to each shoot.
    pub fn weights(&self) -> &[f32] {
        self.biome.weights()
    }

    /// Harvests the canopy averaged across all shoots.
    pub fn canopy(&self) -> PureResult<Tensor> {
        self.biome.canopy()
    }

    /// Clears all shoots from the biome.
    pub fn clear(&mut self) {
        self.biome.clear();
    }

    /// Returns a view over the stored shoots.
    pub fn shoots(&self) -> &[Tensor] {
        self.biome.shoots()
    }

    /// Stacks all shoots into a dense tensor.
    pub fn stack(&self) -> PureResult<Tensor> {
        self.biome.stack()
    }

    /// Delegates fractal patch absorption to the underlying biome.
    pub fn absorb_fractal_patch(&mut self, patch: &FractalPatch) -> PureResult<()> {
        self.biome.absorb_fractal_patch(patch)
    }

    /// Renormalises the shoot weights through the preserved guard.
    pub fn renormalise_weights(&mut self) -> PureResult<()> {
        self.biome.renormalise_weights()
    }
}

/// Tracks shared topos state across tensor rewrites, fractal traversals, and measure updates.
#[derive(Debug, Clone)]
pub struct ToposAtlas<'a> {
    topos: &'a OpenCartesianTopos,
    monad: RewriteMonad<'a>,
    visited_volume: usize,
    depth: usize,
}

impl<'a> ToposAtlas<'a> {
    /// Creates a new atlas anchored to a shared open-cartesian topos.
    pub fn new(topos: &'a OpenCartesianTopos) -> Self {
        Self {
            topos,
            monad: RewriteMonad::new(topos),
            visited_volume: 0,
            depth: 0,
        }
    }

    /// Returns the underlying guard.
    pub fn topos(&self) -> &'a OpenCartesianTopos {
        self.topos
    }

    /// Returns the rewrite monad anchored to this atlas.
    pub fn monad(&self) -> RewriteMonad<'a> {
        self.monad
    }

    fn projected_volume(&self, label: &'static str, volume: usize) -> PureResult<usize> {
        let projected =
            self.visited_volume
                .checked_add(volume)
                .ok_or(TensorError::TensorVolumeExceeded {
                    label,
                    volume: usize::MAX,
                    max_volume: self.topos.max_volume(),
                })?;
        if projected > self.topos.max_volume() {
            return Err(TensorError::TensorVolumeExceeded {
                label,
                volume: projected,
                max_volume: self.topos.max_volume(),
            });
        }
        Ok(projected)
    }

    /// Guards a tensor and records the total traversed volume.
    pub fn guard_tensor(&mut self, label: &'static str, tensor: &Tensor) -> PureResult<()> {
        let (rows, cols) = tensor.shape();
        let projected = self.projected_volume(label, checked_tensor_volume(rows, cols)?)?;
        self.monad.guard_tensor(label, tensor)?;
        self.visited_volume = projected;
        Ok(())
    }

    /// Rewrites a tensor in-place while tracking the traversed volume.
    pub fn guard_tensor_mut(&mut self, label: &'static str, tensor: &mut Tensor) -> PureResult<()> {
        let (rows, cols) = tensor.shape();
        let projected = self.projected_volume(label, checked_tensor_volume(rows, cols)?)?;
        let mut candidate = tensor.clone();
        self.monad.rewrite_tensor(label, &mut candidate)?;
        self.visited_volume = projected;
        *tensor = candidate;
        Ok(())
    }

    /// Lifts an owned tensor into the atlas, returning the rewritten value.
    pub fn lift_tensor(&mut self, label: &'static str, tensor: Tensor) -> PureResult<Tensor> {
        let (rows, cols) = tensor.shape();
        let projected = self.projected_volume(label, checked_tensor_volume(rows, cols)?)?;
        let tensor = self.monad.lift_tensor(label, tensor)?;
        self.visited_volume = projected;
        Ok(tensor)
    }

    /// Guards a slice without affecting the tracked volume.
    pub fn guard_slice(&self, label: &'static str, slice: &[f32]) -> PureResult<()> {
        self.monad.guard_slice(label, slice)
    }

    /// Rewrites a mutable slice while keeping volume untouched.
    pub fn guard_slice_mut(&self, label: &'static str, slice: &mut [f32]) -> PureResult<()> {
        self.monad.rewrite_slice(label, slice)
    }

    /// Normalises a probability slice within the atlas.
    pub fn guard_probability_slice(
        &self,
        label: &'static str,
        slice: &mut [f32],
    ) -> PureResult<()> {
        self.monad.guard_probability_slice(label, slice)
    }

    /// Normalises a probability tensor within the atlas.
    pub fn guard_probability_tensor(
        &self,
        label: &'static str,
        tensor: &mut Tensor,
    ) -> PureResult<()> {
        self.monad.guard_probability_tensor(label, tensor)
    }

    /// Registers the observed depth and guards the underlying relation tensor.
    pub fn guard_fractal_patch(
        &mut self,
        label: &'static str,
        patch: &FractalPatch,
    ) -> PureResult<()> {
        let depth = patch.depth() as usize;
        self.topos.ensure_loop_free(depth)?;
        let (rows, cols) = patch.relation().shape();
        let projected = self.projected_volume(label, checked_tensor_volume(rows, cols)?)?;
        self.monad.guard_tensor(label, patch.relation())?;
        self.visited_volume = projected;
        self.depth = self.depth.max(depth);
        Ok(())
    }

    /// Updates the maximum visited depth.
    pub fn observe_depth(&mut self, depth: usize) -> PureResult<()> {
        self.topos.ensure_loop_free(depth)?;
        if depth > self.depth {
            self.depth = depth;
        }
        Ok(())
    }

    /// Returns the deepest stratum observed by the atlas.
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Returns the accumulated volume guarded by the atlas.
    pub fn visited_volume(&self) -> usize {
        self.visited_volume
    }

    /// Remaining admissible tensor volume before the atlas saturates.
    pub fn remaining_volume(&self) -> usize {
        self.topos.max_volume().saturating_sub(self.visited_volume)
    }

    /// Emits the atlas traversal pressure as a Z-space control signal.
    pub fn control_signal(&self) -> ToposControlSignal {
        self.topos
            .control_signal_for(self.depth, self.visited_volume)
    }
}

/// Minimal monadic helper that rewrites values through the enclosing topos.
#[derive(Clone, Copy, Debug)]
pub struct RewriteMonad<'a> {
    topos: &'a OpenCartesianTopos,
}

impl<'a> RewriteMonad<'a> {
    /// Wraps a guard for repeated rewrites.
    pub fn new(topos: &'a OpenCartesianTopos) -> Self {
        Self { topos }
    }

    /// Returns the underlying topos guard.
    pub fn topos(&self) -> &'a OpenCartesianTopos {
        self.topos
    }

    /// Returns the Lawvere–Tierney guard of the enclosed topos.
    pub fn lawvere_guard(&self) -> &LawvereTierneyGuard {
        self.topos.site().guard()
    }

    fn rewrite_slice_candidate(
        &self,
        label: &'static str,
        mut candidate: Vec<f32>,
    ) -> PureResult<Vec<f32>> {
        self.topos.saturate_slice(&mut candidate);
        self.topos.guard_slice(label, &candidate)?;
        Ok(candidate)
    }

    fn rewrite_tensor_candidate(
        &self,
        label: &'static str,
        mut candidate: Tensor,
    ) -> PureResult<Tensor> {
        self.topos.saturate_slice(candidate.data_mut());
        self.topos.guard_tensor(label, &candidate)?;
        Ok(candidate)
    }

    /// Rewrites a scalar by saturating it into the open-cartesian window.
    pub fn rewrite_scalar(&self, value: f32) -> f32 {
        self.topos.saturate(value)
    }

    /// Rewrites a mutable slice and validates the result.
    pub fn rewrite_slice(&self, label: &'static str, slice: &mut [f32]) -> PureResult<()> {
        let candidate = self.rewrite_slice_candidate(label, slice.to_vec())?;
        slice.copy_from_slice(&candidate);
        Ok(())
    }

    /// Guards a read-only slice without saturation.
    pub fn guard_slice(&self, label: &'static str, slice: &[f32]) -> PureResult<()> {
        self.topos.guard_slice(label, slice)
    }

    /// Rewrites a tensor and re-validates its envelope.
    pub fn rewrite_tensor(&self, label: &'static str, tensor: &mut Tensor) -> PureResult<()> {
        let candidate = self.rewrite_tensor_candidate(label, tensor.clone())?;
        *tensor = candidate;
        Ok(())
    }

    /// Guards an immutable tensor reference.
    pub fn guard_tensor(&self, label: &'static str, tensor: &Tensor) -> PureResult<()> {
        self.topos.guard_tensor(label, tensor)
    }

    /// Ensures a κ-box is admissible for the enclosed topos.
    pub fn guard_zbox(&self, zbox: &ZBox) -> PureResult<()> {
        self.topos.guard_zbox(zbox)
    }

    /// Ensures a κ-box cover remains mass-preserving.
    pub fn guard_cover(&self, cover: &[ZBox]) -> PureResult<()> {
        self.topos.guard_cover(cover)
    }

    /// Normalises a probability slice through the topos window.
    pub fn guard_probability_slice(
        &self,
        label: &'static str,
        slice: &mut [f32],
    ) -> PureResult<()> {
        self.topos.guard_probability_slice(label, slice)
    }

    /// Normalises a probability tensor through the topos window.
    pub fn guard_probability_tensor(
        &self,
        label: &'static str,
        tensor: &mut Tensor,
    ) -> PureResult<()> {
        self.topos.guard_probability_tensor(label, tensor)
    }

    /// Lifts an owned tensor into the monadic context and returns the guarded value.
    pub fn lift_tensor(&self, label: &'static str, tensor: Tensor) -> PureResult<Tensor> {
        self.rewrite_tensor_candidate(label, tensor)
    }

    /// Applies a closure to a tensor before rewriting it through the guard.
    pub fn bind_tensor<F>(
        &self,
        label: &'static str,
        mut tensor: Tensor,
        f: F,
    ) -> PureResult<Tensor>
    where
        F: FnOnce(&mut Tensor) -> PureResult<()>,
    {
        f(&mut tensor)?;
        self.rewrite_tensor_candidate(label, tensor)
    }

    /// Applies a closure to a mutable slice before rewriting it through the guard.
    pub fn bind_slice<F>(&self, label: &'static str, slice: &mut [f32], f: F) -> PureResult<()>
    where
        F: FnOnce(&mut [f32]) -> PureResult<()>,
    {
        let mut candidate = slice.to_vec();
        f(&mut candidate)?;
        let candidate = self.rewrite_slice_candidate(label, candidate)?;
        slice.copy_from_slice(&candidate);
        Ok(())
    }

    /// Cultivates a fresh tensor biome anchored to this monad's guard.
    pub fn cultivate_biome(&self) -> TensorBiome {
        TensorBiome::new(self.topos.clone())
    }

    /// Absorbs an owned tensor through both this monadic guard and the destination biome guard.
    pub fn absorb_into_biome(
        &self,
        biome: &mut TensorBiome,
        label: &'static str,
        tensor: Tensor,
    ) -> PureResult<()> {
        let tensor = self.rewrite_tensor_candidate(label, tensor)?;
        biome.absorb(label, tensor)
    }

    /// Absorbs a weighted tensor through this monad and the destination biome guard.
    pub fn absorb_weighted_into_biome(
        &self,
        biome: &mut TensorBiome,
        label: &'static str,
        tensor: Tensor,
        weight: f32,
    ) -> PureResult<()> {
        let tensor = self.rewrite_tensor_candidate(label, tensor)?;
        biome.absorb_weighted(label, tensor, weight)
    }
}

/// Organises tensors rewritten through an open topos into a living "biome".
///
/// The biome behaves like a minimal monad: every tensor absorbed into it is
/// rewritten through the enclosing `OpenCartesianTopos`, saturated into the
/// safety window, and retained as a new shoot.  When the caller is ready to
/// harvest the emergent meaning, the biome collapses all shoots into a guarded
/// canopy tensor that stays within the same topos envelope.
#[derive(Clone, Debug)]
pub struct TensorBiome {
    topos: OpenCartesianTopos,
    shoots: Vec<Tensor>,
    weights: Vec<f32>,
    total_weight: f32,
    shape: Option<(usize, usize)>,
}

impl TensorBiome {
    /// Wraps a biome around an open-cartesian topos.
    pub fn new(topos: OpenCartesianTopos) -> Self {
        Self {
            topos,
            shoots: Vec::new(),
            weights: Vec::new(),
            total_weight: 0.0,
            shape: None,
        }
    }

    fn validate_weight(weight: f32) -> PureResult<()> {
        if !weight.is_finite() || weight <= 0.0 {
            return Err(TensorError::NonPositiveWeight { weight });
        }
        Ok(())
    }

    fn projected_total_weight(&self, weight: f32) -> PureResult<f32> {
        Self::validate_weight(weight)?;
        checked_f64_to_f32(
            "tensor_biome_total_weight",
            self.total_weight as f64 + weight as f64,
        )
    }

    fn guard_storage_volume(
        &self,
        label: &'static str,
        shoot_count: usize,
        shape: (usize, usize),
    ) -> PureResult<usize> {
        let per_shoot = checked_tensor_volume(shape.0, shape.1)?;
        let volume =
            shoot_count
                .checked_mul(per_shoot)
                .ok_or(TensorError::TensorVolumeExceeded {
                    label,
                    volume: usize::MAX,
                    max_volume: self.topos.max_volume(),
                })?;
        if volume > self.topos.max_volume() {
            return Err(TensorError::TensorVolumeExceeded {
                label,
                volume,
                max_volume: self.topos.max_volume(),
            });
        }
        Ok(volume)
    }

    /// Returns the guard topos.
    pub fn topos(&self) -> &OpenCartesianTopos {
        &self.topos
    }

    /// Returns a rewrite monad anchored to the biome's guard.
    pub fn monad(&self) -> RewriteMonad<'_> {
        RewriteMonad::new(&self.topos)
    }

    /// Returns an atlas anchored to the biome's guard.
    pub fn atlas(&self) -> ToposAtlas<'_> {
        ToposAtlas::new(&self.topos)
    }

    /// Number of shoots currently living inside the biome.
    pub fn len(&self) -> usize {
        self.shoots.len()
    }

    /// Whether the biome is empty.
    pub fn is_empty(&self) -> bool {
        self.shoots.is_empty()
    }

    /// Total accumulated weight across all shoots.
    pub fn total_weight(&self) -> f32 {
        self.total_weight
    }

    /// Number of tensor values currently retained by the biome.
    pub fn stored_volume(&self) -> usize {
        self.shape
            .and_then(|(rows, cols)| rows.checked_mul(cols))
            .and_then(|volume| volume.checked_mul(self.len()))
            .unwrap_or_else(|| if self.shape.is_some() { usize::MAX } else { 0 })
    }

    /// Remaining aggregate storage volume admitted by the guard topos.
    pub fn remaining_volume(&self) -> usize {
        self.topos.max_volume().saturating_sub(self.stored_volume())
    }

    /// Emits a control signal derived from the currently retained shoots.
    pub fn control_signal(&self) -> ToposControlSignal {
        self.topos.control_signal_for(0, self.stored_volume())
    }

    /// Common tensor shape shared by all shoots once the biome is non-empty.
    pub fn shape(&self) -> Option<(usize, usize)> {
        self.shape
    }

    /// Returns the individual weights that were assigned to each shoot.
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Absorbs a tensor into the biome, rewriting it through the guard topos.
    pub fn absorb(&mut self, label: &'static str, tensor: Tensor) -> PureResult<()> {
        self.absorb_weighted(label, tensor, 1.0)
    }

    /// Absorbs a tensor with an explicit weight that skews the canopy average.
    pub fn absorb_weighted(
        &mut self,
        label: &'static str,
        tensor: Tensor,
        weight: f32,
    ) -> PureResult<()> {
        Self::validate_weight(weight)?;
        let monad = RewriteMonad::new(&self.topos);
        let tensor = monad.lift_tensor(label, tensor)?;
        let (rows, cols) = tensor.shape();
        let values = checked_tensor_volume(rows, cols)?;
        self.push_shoot(label, tensor, weight)?;
        let stored_values = self.stored_volume();
        crate::emit_tensor_op("tensor_biome_absorb_weighted", &[rows, cols], &[self.len()]);
        crate::emit_tensor_op_meta("tensor_biome_absorb_weighted", || {
            serde_json::json!({
                "backend": "topos_cpu",
                "requested_backend": "auto",
                "kind": "topos_biome_absorb",
                "rewrite_backend": "topos_cpu",
                "storage_backend": "host_vec",
                "rewrite_mode": "guarded_in_place_tensor_rewrite",
                "route_blocker": "open_cartesian_topos_rewrite",
                "label": label,
                "rows": rows,
                "cols": cols,
                "values": values,
                "shoots": self.len(),
                "weight": weight,
                "total_weight": self.total_weight,
                "curvature": self.topos.curvature(),
                "saturation": self.topos.saturation(),
                "estimated_rewrite_values": values,
                "estimated_stored_values": stored_values,
            })
        });
        Ok(())
    }

    /// Absorbs a tensor produced by a monadic builder.
    pub fn absorb_with<F>(&mut self, label: &'static str, build: F) -> PureResult<()>
    where
        F: FnOnce(RewriteMonad<'_>) -> PureResult<Tensor>,
    {
        let tensor = build(self.monad())?;
        self.absorb(label, tensor)
    }

    /// Absorbs a weighted tensor produced by a monadic builder.
    pub fn absorb_weighted_with<F>(
        &mut self,
        label: &'static str,
        weight: f32,
        build: F,
    ) -> PureResult<()>
    where
        F: FnOnce(RewriteMonad<'_>) -> PureResult<Tensor>,
    {
        Self::validate_weight(weight)?;
        let tensor = build(self.monad())?;
        self.absorb_weighted(label, tensor, weight)
    }

    fn push_shoot(&mut self, label: &'static str, tensor: Tensor, weight: f32) -> PureResult<()> {
        let total_weight = self.projected_total_weight(weight)?;
        self.topos.guard_tensor(label, &tensor)?;
        let shape = tensor.shape();
        if let Some(expected) = self.shape {
            if expected != shape {
                return Err(TensorError::ShapeMismatch {
                    left: expected,
                    right: shape,
                });
            }
        }
        let shoot_count = self
            .len()
            .checked_add(1)
            .ok_or(TensorError::TensorVolumeExceeded {
                label: "tensor_biome_storage",
                volume: usize::MAX,
                max_volume: self.topos.max_volume(),
            })?;
        self.guard_storage_volume("tensor_biome_storage", shoot_count, shape)?;
        self.shoots.push(tensor);
        self.weights.push(weight);
        self.shape = Some(shape);
        self.total_weight = total_weight;
        Ok(())
    }

    /// Applies a guarded rewrite to every shoot living inside the biome.
    pub fn bind_shoots<F>(&mut self, label: &'static str, mut f: F) -> PureResult<()>
    where
        F: FnMut(RewriteMonad<'_>, &mut Tensor) -> PureResult<()>,
    {
        let topos = self.topos.clone();
        let monad = RewriteMonad::new(&topos);
        let mut candidates = self.shoots.clone();
        for shoot in &mut candidates {
            f(monad, shoot)?;
            monad.rewrite_tensor(label, shoot)?;
        }
        let shape = candidates.first().map(Tensor::shape);
        if let Some(expected) = shape {
            for candidate in candidates.iter().skip(1) {
                let actual = candidate.shape();
                if actual != expected {
                    return Err(TensorError::ShapeMismatch {
                        left: expected,
                        right: actual,
                    });
                }
            }
            self.guard_storage_volume("tensor_biome_storage", candidates.len(), expected)?;
        }
        self.shoots = candidates;
        self.shape = shape;
        Ok(())
    }

    /// Builds a new biome by mapping the current shoots through a monadic builder.
    pub fn map_shoots<F>(&self, label: &'static str, mut f: F) -> PureResult<TensorBiome>
    where
        F: FnMut(RewriteMonad<'_>, &Tensor) -> PureResult<Tensor>,
    {
        let topos = self.topos.clone();
        let monad = RewriteMonad::new(&topos);
        let mut biome = TensorBiome::new(topos.clone());
        for (shoot, &weight) in self.shoots.iter().zip(self.weights.iter()) {
            let mapped = monad.lift_tensor(label, f(monad, shoot)?)?;
            biome.push_shoot(label, mapped, weight)?;
        }
        Ok(biome)
    }

    /// Renormalises the shoot weights through the Lawvere–Tierney guard.
    pub fn renormalise_weights(&mut self) -> PureResult<()> {
        if self.weights.is_empty() {
            return Err(TensorError::EmptyInput("tensor_biome_weights"));
        }
        let before = self.weights.clone();
        let raw_total = before.iter().map(|&weight| weight as f64).sum::<f64>();
        let topos = self.topos.clone();
        let monad = RewriteMonad::new(&topos);
        let mut weights = self.weights.clone();
        monad.guard_probability_slice("tensor_biome_weights", &mut weights)?;
        let total_weight = checked_f64_to_f32(
            "tensor_biome_total_weight",
            weights.iter().map(|&weight| weight as f64).sum::<f64>(),
        )?;
        let changed_weights = before
            .iter()
            .copied()
            .zip(weights.iter().copied())
            .filter(|(lhs, rhs)| (lhs - rhs).abs() > 1.0e-6)
            .count();
        self.weights = weights;
        self.total_weight = total_weight;
        crate::emit_tensor_op(
            "tensor_biome_renormalise_weights",
            &[before.len()],
            &[self.weights.len()],
        );
        crate::emit_tensor_op_meta("tensor_biome_renormalise_weights", || {
            serde_json::json!({
                "backend": "control_cpu",
                "requested_backend": "host",
                "kind": "topos_biome_weight_reduction",
                "weight_guard_backend": "lawvere_topos_cpu",
                "reduction_backend": "host_slice_sum",
                "guard_mode": "probability_slice_guard",
                "route_blocker": "lawvere_tierney_probability_guard",
                "weights": self.weights.len(),
                "shoots": self.len(),
                "raw_total": raw_total,
                "total_weight": self.total_weight,
                "changed_weights": changed_weights,
                "curvature": self.topos.curvature(),
                "saturation": self.topos.saturation(),
                "estimated_guard_values": self.weights.len(),
                "estimated_reduction_values": before.len(),
            })
        });
        Ok(())
    }

    /// Absorbs a fractal relation patch directly into the biome canopy.
    pub fn absorb_fractal_patch(&mut self, patch: &FractalPatch) -> PureResult<()> {
        let mut atlas = self.atlas();
        atlas.guard_fractal_patch("tensor_biome_fractal_patch", patch)?;
        let relation = atlas.lift_tensor(
            "tensor_biome_fractal_patch_relation",
            patch.relation().clone(),
        )?;
        self.absorb_weighted("tensor_biome_fractal_patch", relation, patch.weight())
    }

    /// Clears all shoots from the biome while preserving the topos.
    pub fn clear(&mut self) {
        self.shoots.clear();
        self.weights.clear();
        self.total_weight = 0.0;
        self.shape = None;
    }

    /// Harvests the biome by averaging all shoots into a guarded canopy tensor.
    pub fn canopy(&self) -> PureResult<Tensor> {
        self.canopy_with_backend(TensorUtilBackend::Auto)
    }

    /// Harvests the biome with an explicit tensor utility backend for weighted
    /// accumulation and a final normalisation after host-side `f64` weight ratios.
    pub fn canopy_with_backend(&self, backend: TensorUtilBackend) -> PureResult<Tensor> {
        let (rows, cols) = self.shape.ok_or(TensorError::EmptyInput("tensor_biome"))?;
        if self.is_empty() {
            return Err(TensorError::EmptyInput("tensor_biome"));
        }
        if !self.total_weight.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "tensor_biome_total_weight",
                value: self.total_weight,
            });
        }
        if self.total_weight <= 0.0 {
            return Err(TensorError::NonPositiveWeight {
                weight: self.total_weight,
            });
        }
        if self.shoots.len() != self.weights.len() {
            return Err(TensorError::DataLength {
                expected: self.shoots.len(),
                got: self.weights.len(),
            });
        }
        let total_weight = self
            .weights
            .iter()
            .map(|&weight| weight as f64)
            .sum::<f64>();
        if !total_weight.is_finite() || total_weight <= 0.0 {
            return Err(TensorError::NonFiniteValue {
                label: "tensor_biome_weight_sum",
                value: total_weight as f32,
            });
        }
        let mut acc = Tensor::zeros(rows, cols)?;
        let mut normalised_total = 0.0f64;
        for (shoot, &weight) in self.shoots.iter().zip(self.weights.iter()) {
            let normalised_weight = checked_f64_to_f32(
                "tensor_biome_normalised_weight",
                weight as f64 / total_weight,
            )?;
            normalised_total += normalised_weight as f64;
            acc.add_scaled_with_backend(shoot, normalised_weight, backend)?;
        }
        if !normalised_total.is_finite() || normalised_total <= 0.0 {
            return Err(TensorError::NonFiniteValue {
                label: "tensor_biome_normalised_weight_sum",
                value: normalised_total as f32,
            });
        }
        let normalisation =
            checked_f64_to_f32("tensor_biome_weight_normalisation", 1.0 / normalised_total)?;
        let mut canopy = acc.scale_with_backend(normalisation, backend)?;
        let monad = RewriteMonad::new(&self.topos);
        monad.rewrite_tensor("tensor_biome_canopy", &mut canopy)?;
        let tensor_util_backend = tensor_util_backend_label(backend);
        crate::emit_tensor_op(
            "tensor_biome_canopy",
            &[self.len(), rows, cols],
            &[rows, cols],
        );
        crate::emit_tensor_op_meta("tensor_biome_canopy", || {
            serde_json::json!({
                "backend": "hybrid",
                "requested_backend": tensor_util_backend,
                "kind": "topos_biome_canopy",
                "accumulation_backend": tensor_util_backend,
                "normalise_backend": tensor_util_backend,
                "weight_normalise_backend": "host_f64",
                "rewrite_backend": "topos_cpu",
                "rewrite_mode": "guarded_canopy_rewrite",
                "route_blocker": "open_cartesian_topos_rewrite_after_tensor_util_accumulation",
                "rows": rows,
                "cols": cols,
                "values": rows.saturating_mul(cols),
                "shoots": self.len(),
                "weights": self.weights.len(),
                "total_weight": self.total_weight,
                "curvature": self.topos.curvature(),
                "saturation": self.topos.saturation(),
                "estimated_accumulation_values": self.len().saturating_mul(rows).saturating_mul(cols),
                "estimated_normalise_values": rows.saturating_mul(cols),
                "estimated_rewrite_values": rows.saturating_mul(cols),
            })
        });
        Ok(canopy)
    }

    /// Returns a snapshot of the current shoots.
    pub fn shoots(&self) -> &[Tensor] {
        &self.shoots
    }

    /// Stacks all shoots along the row dimension, yielding a dense tensor.
    pub fn stack(&self) -> PureResult<Tensor> {
        let (rows, cols) = self.shape.ok_or(TensorError::EmptyInput("tensor_biome"))?;
        if self.is_empty() {
            return Err(TensorError::EmptyInput("tensor_biome"));
        }
        let stacked_rows =
            self.shoots
                .len()
                .checked_mul(rows)
                .ok_or(TensorError::InvalidDimensions {
                    rows: self.shoots.len(),
                    cols: rows,
                })?;
        let volume = checked_tensor_volume(stacked_rows, cols)?;
        self.guard_storage_volume("tensor_biome_stack", self.shoots.len(), (rows, cols))?;
        let mut data = Vec::with_capacity(volume);
        for shoot in &self.shoots {
            data.extend_from_slice(shoot.data());
        }
        Tensor::from_vec(stacked_rows, cols, data)
    }
}

/// Deterministic conjugate gradient solver that respects the open-cartesian guard.
pub struct ConjugateGradientSolver<'a> {
    topos: &'a OpenCartesianTopos,
    tolerance: f32,
    max_iterations: usize,
}

impl<'a> ConjugateGradientSolver<'a> {
    /// Creates a solver with an explicit tolerance and iteration cap.
    pub fn new(
        topos: &'a OpenCartesianTopos,
        tolerance: f32,
        max_iterations: usize,
    ) -> PureResult<Self> {
        if !tolerance.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "conjugate_gradient_tolerance",
                value: tolerance,
            });
        }
        if tolerance <= 0.0 {
            return Err(TensorError::NonPositiveTolerance { tolerance });
        }
        if max_iterations == 0 {
            return Err(TensorError::EmptyInput("conjugate gradient max iterations"));
        }
        Ok(Self {
            topos,
            tolerance,
            max_iterations,
        })
    }

    /// Returns the solver tolerance.
    pub fn tolerance(&self) -> f32 {
        self.tolerance
    }

    /// Returns the hard iteration cap.
    pub fn max_iterations(&self) -> usize {
        self.max_iterations
    }

    /// Solves a linear system `Ax = b` using repeated matrix-vector products.
    ///
    /// The callback receives the candidate vector and must write every component
    /// of `A * src` into `dst`. Callback output is checked for completeness and
    /// finiteness. The caller's `x` is committed only after the true residual of
    /// the guarded candidate reaches tolerance, so every failure is transactional.
    pub fn solve<F>(&self, mut matvec: F, b: &[f32], x: &mut [f32]) -> PureResult<usize>
    where
        F: FnMut(&[f32], &mut [f32]),
    {
        if b.len() != x.len() {
            return Err(TensorError::DataLength {
                expected: b.len(),
                got: x.len(),
            });
        }
        if b.is_empty() {
            return Err(TensorError::EmptyInput("conjugate gradient rhs"));
        }
        self.topos.guard_slice("cg_rhs", b)?;
        self.topos.guard_slice("cg_initial", x)?;
        let mut solution = x.to_vec();
        let mut r = vec![0.0f32; b.len()];
        let mut p = vec![0.0f32; b.len()];
        let mut ap = vec![0.0f32; b.len()];
        self.guarded_matvec(&mut matvec, &solution, &mut ap, "cg_initial_matvec")?;
        for index in 0..b.len() {
            r[index] =
                checked_f64_to_f32("cg_initial_residual", b[index] as f64 - ap[index] as f64)?;
            p[index] = r[index];
        }
        let mut rsold = dot_f64("cg_initial_residual_norm", &r, &r)?;
        let tol = self.tolerance.max(self.topos.tolerance()) as f64;
        if rsold.sqrt() <= tol {
            return Ok(0);
        }
        for iter in 0..self.max_iterations {
            self.guarded_matvec(&mut matvec, &p, &mut ap, "cg_direction_matvec")?;
            let denominator = dot_f64("cg_direction_curvature", &p, &ap)?;
            if denominator <= 0.0 {
                return Err(TensorError::ConjugateGradientBreakdown {
                    iteration: iter,
                    denominator,
                });
            }
            let alpha = rsold / denominator;
            if !alpha.is_finite() {
                return Err(TensorError::NonFiniteValue {
                    label: "conjugate_gradient_alpha",
                    value: alpha as f32,
                });
            }
            let mut projected_step = false;
            for index in 0..solution.len() {
                let raw = checked_f64_to_f32(
                    "cg_candidate",
                    solution[index] as f64 + alpha * p[index] as f64,
                )?;
                let guarded = self.topos.saturate(raw);
                projected_step |= guarded != raw;
                solution[index] = guarded;
            }
            self.guarded_matvec(&mut matvec, &solution, &mut ap, "cg_solution_matvec")?;
            for index in 0..r.len() {
                r[index] =
                    checked_f64_to_f32("cg_true_residual", b[index] as f64 - ap[index] as f64)?;
            }
            let rsnew = dot_f64("cg_true_residual_norm", &r, &r)?;
            if rsnew.sqrt() <= tol {
                self.topos.guard_slice("cg_solution", &solution)?;
                x.copy_from_slice(&solution);
                return Ok(iter + 1);
            }
            if projected_step {
                p.copy_from_slice(&r);
            } else {
                let beta = rsnew / rsold;
                if !beta.is_finite() {
                    return Err(TensorError::NonFiniteValue {
                        label: "conjugate_gradient_beta",
                        value: beta as f32,
                    });
                }
                for index in 0..p.len() {
                    p[index] = checked_f64_to_f32(
                        "cg_direction",
                        r[index] as f64 + beta * p[index] as f64,
                    )?;
                }
            }
            rsold = rsnew;
        }
        Err(TensorError::ConjugateGradientDiverged {
            residual: rsold.sqrt(),
            tolerance: tol,
        })
    }

    fn guarded_matvec<F>(
        &self,
        matvec: &mut F,
        src: &[f32],
        dst: &mut [f32],
        label: &'static str,
    ) -> PureResult<()>
    where
        F: FnMut(&[f32], &mut [f32]),
    {
        dst.fill(f32::NAN);
        matvec(src, dst);
        self.topos.guard_slice(label, dst)
    }
}

fn checked_f64_to_f32(label: &'static str, value: f64) -> PureResult<f32> {
    if !value.is_finite() || value.abs() > f32::MAX as f64 {
        let value = if value.is_nan() {
            f32::NAN
        } else if value.is_sign_negative() {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
        return Err(TensorError::NonFiniteValue { label, value });
    }
    Ok(value as f32)
}

fn dot_f64(label: &'static str, a: &[f32], b: &[f32]) -> PureResult<f64> {
    let value = a
        .iter()
        .zip(b.iter())
        .map(|(&left, &right)| left as f64 * right as f64)
        .sum::<f64>();
    if !value.is_finite() {
        return Err(TensorError::NonFiniteValue {
            label,
            value: value as f32,
        });
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pure::fractal::FractalPatch;
    use std::cell::Cell;
    use std::sync::{Arc, Mutex, OnceLock};

    #[track_caller]
    fn unwrap_ok<T, E: core::fmt::Debug>(result: Result<T, E>) -> T {
        match result {
            Ok(value) => value,
            Err(error) => panic!("expected Ok(..), got Err({error:?})"),
        }
    }

    #[track_caller]
    fn unwrap_err<T, E: core::fmt::Debug>(result: Result<T, E>) -> E {
        match result {
            Ok(_) => panic!("expected Err(..), got Ok(..)"),
            Err(error) => error,
        }
    }

    fn demo_topos() -> OpenCartesianTopos {
        unwrap_ok(OpenCartesianTopos::new(-1.0, 1e-5, 10.0, 64, 4096))
    }

    fn observer_lock() -> std::sync::MutexGuard<'static, ()> {
        static OBSERVER_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        OBSERVER_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("observer lock available")
    }

    #[test]
    fn topos_porosity_softens_extremes() {
        let rigid = unwrap_ok(OpenCartesianTopos::new(-1.0, 1e-5, 10.0, 64, 4096));
        let rigid = unwrap_ok(rigid.with_porosity(0.0));
        let porous = unwrap_ok(OpenCartesianTopos::new(-1.0, 1e-5, 10.0, 64, 4096));
        let porous = unwrap_ok(porous.with_porosity(0.9));
        let sample = rigid.saturation() * 4.0;
        let rigid_value = rigid.saturate(sample);
        let porous_value = porous.saturate(sample);
        assert!(porous_value.abs() < rigid_value.abs());
        assert!(porous_value.abs() > rigid.saturation() * 0.8);
    }

    #[test]
    fn topos_saturation_slope_matches_the_canonical_rewrite() {
        let topos = unwrap_ok(
            OpenCartesianTopos::new(-1.0, 1e-6, 1.0, 8, 8)
                .and_then(|topos| topos.with_porosity(0.8)),
        );
        let (inside, inside_slope) = topos.saturate_with_slope(0.5);
        assert_eq!(inside, 0.5);
        assert_eq!(inside_slope, 1.0);

        let value = 2.0f32;
        let (outside, outside_slope) = topos.saturate_with_slope(value);
        let epsilon = 1e-3;
        let numeric =
            (topos.saturate(value + epsilon) - topos.saturate(value - epsilon)) / (2.0 * epsilon);
        assert!(outside.abs() <= topos.saturation());
        assert!((outside_slope - numeric).abs() < 1e-4);
        assert!(outside_slope < 0.0);
    }

    #[test]
    fn runtime_profile_input_normalizes_external_values() {
        let profile = ToposRuntimeProfile::from_input(ToposRuntimeProfileInput {
            training_gain: f32::NAN,
            inference_gain: f32::INFINITY,
            closure_risk: -1.0,
            exploration_budget: 2.0,
            control_energy: f32::NAN,
            training_rate_scale: 0.0,
            training_gradient_bias_scale: 1.0,
            inference_temperature: 3.0,
            inference_top_p: 0.0,
            inference_context_weight: 0.0,
            learning_inference_balance: 3.0,
        });

        assert_eq!(profile.training_gain(), 1.0);
        assert_eq!(profile.inference_gain(), 1.0);
        assert_eq!(profile.closure_risk(), 0.0);
        assert_eq!(profile.exploration_budget(), 1.0);
        assert_eq!(profile.control_energy(), 0.0);
        assert_eq!(profile.training_rate_scale(), 0.01);
        assert_eq!(profile.training_gradient_bias_scale(), 0.35);
        assert_eq!(profile.inference_temperature(), 2.0);
        assert_eq!(profile.inference_top_p(), 0.05);
        assert_eq!(profile.inference_context_weight(), 0.25);
        assert_eq!(profile.learning_inference_balance(), 2.0);
    }

    #[test]
    fn runtime_route_payload_is_the_shared_transport_contract() {
        let route = ToposRuntimeProfile::from_input(ToposRuntimeProfileInput {
            closure_risk: 0.7,
            control_energy: 0.8,
            training_gradient_bias_scale: 0.2,
            ..ToposRuntimeProfileInput::default()
        })
        .route();
        let payload = route.payload();

        assert_eq!(payload.kind, TOPOS_RUNTIME_ROUTE_KIND);
        assert_eq!(
            payload.contract_version,
            TOPOS_RUNTIME_ROUTE_CONTRACT_VERSION
        );
        assert_eq!(payload.semantic_owner, TOPOS_RUNTIME_ROUTE_SEMANTIC_OWNER);
        assert_eq!(
            payload.semantic_backend,
            TOPOS_RUNTIME_ROUTE_SEMANTIC_BACKEND
        );
        assert_eq!(payload.mode, route.mode_label());
        assert_eq!(payload.mode_id, route.mode_id());
        assert_eq!(payload.scores.vector, route.scores().vector());
        assert_eq!(payload.runtime_profile.vector, route.profile().vector());

        let serialized = serde_json::to_value(payload).expect("runtime route serializes");
        assert_eq!(serialized["kind"], TOPOS_RUNTIME_ROUTE_KIND);
        assert_eq!(serialized["scores"]["guard"], route.scores().guard_score());
        assert_eq!(
            serialized["runtime_profile"]["closure_risk"],
            route.profile().closure_risk()
        );
    }

    #[test]
    fn control_signal_input_derives_canonical_pressure_and_hints() {
        let signal = unwrap_ok(ToposControlSignal::from_input(ToposControlSignalInput {
            curvature: -0.9,
            tolerance: 1e-4,
            saturation: 2.0,
            porosity: 0.25,
            max_depth: 10,
            max_volume: 100,
            observed_depth: 4,
            visited_volume: 25,
        }));

        assert!((signal.closure_pressure() - 0.4).abs() < 1e-6);
        assert!((signal.openness() - 0.6).abs() < 1e-6);
        assert!((signal.step_damping() - 0.258).abs() < 1e-6);
        assert!((signal.training_hints().gradient_bias_scale() - 0.0786561).abs() < 1e-6);
        assert_eq!(
            signal.runtime_route(1.0, 1.0, 1.0, 1.0, 0.0, 0.0).mode(),
            ToposRuntimeMode::Contextual
        );
    }

    #[test]
    fn control_signal_input_accepts_partial_serde_payloads() {
        let input = serde_json::from_str::<ToposControlSignalInput>(
            r#"{"max_depth":10,"max_volume":100,"observed_depth":4}"#,
        )
        .expect("partial control-signal input");
        let signal = unwrap_ok(ToposControlSignal::from_input(input));

        assert_eq!(signal.curvature(), -1.0);
        assert_eq!(signal.porosity(), 0.2);
        assert_eq!(signal.observed_depth(), 4);
        assert_eq!(signal.visited_volume(), 0);
        assert!((signal.depth_pressure() - 0.4).abs() < 1e-6);
    }

    #[test]
    fn open_topos_rejects_non_finite_geometry_at_ingress() {
        for (curvature, tolerance, saturation, expected_label) in [
            (f32::NAN, 1e-3, 1.0, "topos_curvature"),
            (-1.0, f32::INFINITY, 1.0, "topos_tolerance"),
            (-1.0, 1e-3, f32::NEG_INFINITY, "topos_saturation"),
        ] {
            let error = unwrap_err(OpenCartesianTopos::new(
                curvature, tolerance, saturation, 64, 512,
            ));
            assert!(matches!(
                error,
                TensorError::NonFiniteValue { label, .. } if label == expected_label
            ));
        }

        let error = unwrap_err(ToposControlSignal::from_input(ToposControlSignalInput {
            porosity: f32::NAN,
            ..ToposControlSignalInput::default()
        }));
        assert!(matches!(
            error,
            TensorError::NonFiniteValue {
                label: "topos_porosity",
                ..
            }
        ));
    }

    #[test]
    fn runtime_route_modes_are_reachable_with_bounded_scores() {
        let cases = [
            (
                ToposRuntimeMode::TrainingFirst,
                ToposRuntimeProfileInput {
                    training_rate_scale: 0.01,
                    inference_temperature: 0.0,
                    inference_top_p: 0.05,
                    inference_context_weight: 0.25,
                    learning_inference_balance: 0.0,
                    ..ToposRuntimeProfileInput::default()
                },
            ),
            (
                ToposRuntimeMode::Contextual,
                ToposRuntimeProfileInput {
                    training_rate_scale: 0.01,
                    inference_temperature: 0.0,
                    inference_top_p: 0.05,
                    inference_context_weight: 0.85,
                    learning_inference_balance: 0.0,
                    ..ToposRuntimeProfileInput::default()
                },
            ),
            (
                ToposRuntimeMode::Balanced,
                ToposRuntimeProfileInput {
                    training_rate_scale: 0.01,
                    inference_temperature: 0.5,
                    inference_top_p: 1.0,
                    inference_context_weight: 0.8,
                    learning_inference_balance: 0.0,
                    ..ToposRuntimeProfileInput::default()
                },
            ),
            (
                ToposRuntimeMode::InferenceFirst,
                ToposRuntimeProfileInput {
                    training_rate_scale: 0.01,
                    training_gradient_bias_scale: 0.1,
                    inference_temperature: 1.5,
                    inference_top_p: 1.0,
                    inference_context_weight: 0.8,
                    learning_inference_balance: 0.0,
                    ..ToposRuntimeProfileInput::default()
                },
            ),
            (
                ToposRuntimeMode::Exploratory,
                ToposRuntimeProfileInput {
                    exploration_budget: 0.6,
                    training_rate_scale: 0.01,
                    inference_temperature: 0.5,
                    inference_top_p: 1.0,
                    inference_context_weight: 0.25,
                    learning_inference_balance: 0.0,
                    ..ToposRuntimeProfileInput::default()
                },
            ),
            (
                ToposRuntimeMode::Guarded,
                ToposRuntimeProfileInput {
                    closure_risk: 0.2,
                    control_energy: 1.0,
                    training_rate_scale: 0.01,
                    training_gradient_bias_scale: 0.35,
                    inference_temperature: 0.0,
                    inference_top_p: 0.05,
                    inference_context_weight: 0.25,
                    learning_inference_balance: 0.0,
                    ..ToposRuntimeProfileInput::default()
                },
            ),
        ];

        for (expected_mode, input) in cases {
            let route = ToposRuntimeProfile::from_input(input).route();
            assert_eq!(route.mode(), expected_mode);
            assert!((0.0..=1.0).contains(&route.score()));
            for score in route.scores().vector() {
                assert!(score.is_finite());
                assert!((0.0..=1.0).contains(&score));
            }
        }
    }

    #[test]
    fn topos_control_signal_tracks_pressure_and_openness() {
        let topos = unwrap_ok(OpenCartesianTopos::new(-1.0, 1e-5, 10.0, 10, 100));
        let topos = unwrap_ok(topos.with_porosity(0.25));
        let signal = topos.control_signal_for(4, 25);
        assert_eq!(signal.observed_depth(), 4);
        assert_eq!(signal.visited_volume(), 25);
        assert_eq!(signal.remaining_volume(), 75);
        assert!((signal.depth_pressure() - 0.4).abs() < 1e-6);
        assert!((signal.volume_pressure() - 0.25).abs() < 1e-6);
        assert!((signal.closure_pressure() - 0.4).abs() < 1e-6);
        assert!((signal.openness() - 0.6).abs() < 1e-6);
        assert!(signal.guard_strength() > 0.0);
        assert!(signal.stability_hint() > 0.0);
        assert!(signal.learning_rate_scale() > 0.1);
        assert!(signal.learning_rate_scale() <= 1.25);
        assert!(signal.temperature_scale() >= 0.5);
        assert!(signal.temperature_scale() <= 1.5);
        assert!(signal.regularization_scale() >= 0.5);
        assert!(signal.step_damping() > 0.0);
        assert!(signal.sampling_focus() > 0.0);
        assert_eq!(signal.runtime_hints().len(), 5);
        assert_eq!(signal.gradient().len(), 6);
        let training = signal.training_hints();
        assert!((training.gradient_bias_scale() - 0.0786561).abs() < 1e-6);
        assert!((training.clip_scale() - 0.871).abs() < 1e-6);
        assert!((training.momentum_damping() - 0.2535).abs() < 1e-6);
        assert_eq!(training.vector().len(), 6);
        let training_plan = signal.training_plan(0.5);
        assert!((training_plan.raw_rate_scale() - 0.8919875).abs() < 1e-6);
        assert!((training_plan.rate_scale() - 0.9459937).abs() < 1e-6);
        assert!((training_plan.effective_gradient_bias_scale() - 0.03932805).abs() < 1e-6);
        assert!((training_plan.effective_gradient_clip_scale() - 0.9355).abs() < 1e-6);
        assert!((training_plan.effective_momentum_damping() - 0.12675).abs() < 1e-6);
        assert_eq!(training_plan.vector().len(), 6);
        let inference = signal.inference_hints();
        assert!((inference.top_p_scale() - 0.8902744).abs() < 1e-6);
        assert!((inference.frequency_penalty_bias() - 0.2854011).abs() < 1e-6);
        assert!((inference.presence_penalty_bias() + 0.03810062).abs() < 1e-6);
        assert!((inference.context_weight() - 0.9225).abs() < 1e-6);
        assert_eq!(inference.vector().len(), 6);
        let inference_plan = signal.inference_plan(0.5, 0.8, 0.9, 0.1, 0.2);
        assert!((inference_plan.temperature() - 0.741325).abs() < 1e-6);
        assert!((inference_plan.top_p() - 0.8506235).abs() < 1e-6);
        assert!((inference_plan.frequency_penalty() - 0.24270055).abs() < 1e-6);
        assert!((inference_plan.presence_penalty() - 0.18094969).abs() < 1e-6);
        assert!((inference_plan.context_weight() - 0.96125).abs() < 1e-6);
        assert_eq!(inference_plan.vector().len(), 6);
        let profile = signal.runtime_profile(0.5, 0.5, 0.8, 0.9, 0.1, 0.2);
        assert!((profile.closure_risk() - 0.4451).abs() < 1e-6);
        assert!((profile.exploration_budget() - 0.4555).abs() < 1e-6);
        assert!((profile.training_rate_scale() - 0.9459937).abs() < 1e-6);
        assert!((profile.inference_temperature() - 0.741325).abs() < 1e-6);
        assert!((profile.control_energy() - 0.34847772).abs() < 1e-6);
        assert!((profile.learning_inference_balance() - 1.276085).abs() < 1e-6);
        assert_eq!(profile.vector().len(), 6);
        let route = profile.route();
        assert_eq!(route.mode(), ToposRuntimeMode::Contextual);
        assert_eq!(route.mode_label(), "contextual");
        assert_eq!(route.mode_id(), 3);
        assert_eq!(route.score_key(), "context");
        assert_eq!(route.inference_action(), "raise_context_weight");
        assert!((route.score() - 0.68883634).abs() < 1e-6);
        assert!((route.scores().context_score() - route.score()).abs() < 1e-6);
        assert!(route.scores().training_score() > route.scores().guard_score());
        let signal_route = signal.runtime_route(0.5, 0.5, 0.8, 0.9, 0.1, 0.2);
        assert_eq!(signal_route.mode(), route.mode());
        assert!((signal_route.score() - route.score()).abs() < 1e-6);
    }

    #[test]
    fn topos_zspace_projection_owns_metrics_and_dimension_adaptation() {
        let signal = unwrap_ok(ToposControlSignal::from_input(ToposControlSignalInput {
            porosity: 0.25,
            max_depth: 10,
            max_volume: 100,
            observed_depth: 4,
            visited_volume: 25,
            ..ToposControlSignalInput::default()
        }));
        let projection =
            unwrap_ok(signal.zspace_projection(ToposZSpaceProjectionOptions { gradient_dim: 8 }));

        assert!((projection.speed() - 0.41477418).abs() < 1e-6);
        assert!((projection.memory() - 0.25).abs() < 1e-6);
        assert!((projection.stability() - 0.4201313).abs() < 1e-6);
        assert!((projection.drs() - 0.1355).abs() < 1e-6);
        assert!((projection.frac() - 0.6680031).abs() < 1e-6);
        assert_eq!(&projection.gradient()[..6], &signal.gradient());
        assert_eq!(&projection.gradient()[6..], &[0.0, 0.0]);

        let payload = projection.payload();
        assert_eq!(payload.kind, TOPOS_ZSPACE_PROJECTION_KIND);
        assert_eq!(
            payload.contract_version,
            TOPOS_ZSPACE_PROJECTION_CONTRACT_VERSION
        );
        assert_eq!(
            payload.semantic_owner,
            TOPOS_ZSPACE_PROJECTION_SEMANTIC_OWNER
        );
        assert_eq!(payload.semantic_backend, "rust");
        assert_eq!(
            payload.gradient_basis,
            TOPOS_ZSPACE_PROJECTION_GRADIENT_BASIS
        );
        assert_eq!(
            payload.gradient_formula,
            TOPOS_ZSPACE_PROJECTION_GRADIENT_FORMULA
        );
        assert_eq!(
            payload.gradient_channels,
            TOPOS_ZSPACE_PROJECTION_GRADIENT_CHANNELS
        );
        assert_eq!(payload.gradient_dim, 8);
        assert_eq!(payload.base_gradient_dim, 6);
        assert_eq!(payload.vector, projection.vector());
    }

    #[test]
    fn topos_zspace_projection_truncates_and_rejects_unsafe_dimensions() {
        let signal = unwrap_ok(ToposControlSignal::from_input(
            ToposControlSignalInput::default(),
        ));
        let projection =
            unwrap_ok(signal.zspace_projection(ToposZSpaceProjectionOptions { gradient_dim: 4 }));
        assert_eq!(projection.gradient(), &signal.gradient()[..4]);

        for gradient_dim in [0, TOPOS_ZSPACE_PROJECTION_MAX_GRADIENT_DIM + 1] {
            let error =
                unwrap_err(signal.zspace_projection(ToposZSpaceProjectionOptions { gradient_dim }));
            assert!(matches!(error, TensorError::InvalidValue { .. }));
        }
    }

    #[test]
    fn topos_zspace_projection_stays_finite_across_guard_pressure_grid() {
        for observed_depth in [0, 1, 32, 64, usize::MAX] {
            for visited_volume in [0, 1, 256, 512, usize::MAX] {
                let signal = unwrap_ok(ToposControlSignal::from_input(ToposControlSignalInput {
                    observed_depth,
                    visited_volume,
                    ..ToposControlSignalInput::default()
                }));
                let projection = unwrap_ok(
                    signal.zspace_projection(ToposZSpaceProjectionOptions { gradient_dim: 8 }),
                );
                for value in projection.vector() {
                    assert!(value.is_finite());
                }
                assert!((0.0..=1.0).contains(&projection.speed()));
                assert!((0.0..=1.0).contains(&projection.memory()));
                assert!((0.0..=1.0).contains(&projection.stability()));
                assert!((-1.0..=1.0).contains(&projection.drs()));
                assert!((0.0..=1.0).contains(&projection.frac()));
                assert_eq!(projection.gradient().len(), 8);
                assert!(projection.gradient().iter().all(|value| value.is_finite()));
            }
        }
    }

    #[test]
    fn topos_control_payload_is_one_self_consistent_default_bundle() {
        let signal = unwrap_ok(ToposControlSignal::from_input(ToposControlSignalInput {
            observed_depth: 16,
            visited_volume: 128,
            ..ToposControlSignalInput::default()
        }));
        let payload = signal.payload();

        assert_eq!(payload.kind, TOPOS_CONTROL_SIGNAL_KIND);
        assert_eq!(
            payload.contract_version,
            TOPOS_CONTROL_SIGNAL_CONTRACT_VERSION
        );
        assert_eq!(payload.semantic_owner, TOPOS_CONTROL_SIGNAL_SEMANTIC_OWNER);
        assert_eq!(
            payload.semantic_backend,
            TOPOS_CONTROL_SIGNAL_SEMANTIC_BACKEND
        );
        assert_eq!(payload.training_hints, signal.training_hints().payload());
        assert_eq!(payload.training_plan, signal.training_plan(1.0).payload());
        assert_eq!(payload.inference_hints, signal.inference_hints().payload());
        assert_eq!(
            payload.inference_plan,
            signal.inference_plan(1.0, 1.0, 1.0, 0.0, 0.0).payload()
        );
        assert_eq!(
            payload.runtime_route.runtime_profile,
            payload.runtime_profile
        );
        assert_eq!(payload.runtime_hints, signal.runtime_hints());
        assert_eq!(payload.gradient, signal.gradient());
    }

    #[test]
    fn topos_control_payload_normalizes_partial_external_hints_in_rust() {
        let signal = unwrap_ok(ToposControlSignal::from_input(ToposControlSignalInput {
            observed_depth: 32,
            visited_volume: 256,
            ..ToposControlSignalInput::default()
        }));
        let base_training = signal.training_hints();
        let base_inference = signal.inference_hints();
        let payload = unwrap_ok(signal.payload_with_options(
            ToposControlPlanOptions {
                training_gain: 0.5,
                inference: ToposInferencePlanOptions {
                    gain: 1.0,
                    base_temperature: 0.8,
                    base_top_p: 0.95,
                    min_temperature: 0.6,
                    max_temperature: 0.7,
                    min_top_p: 0.8,
                    max_top_p: 0.85,
                    ..ToposInferencePlanOptions::default()
                },
            },
            Some(ToposTrainingHintsInput {
                learning_rate_scale: Some(10.0),
                regularization_scale: Some(f32::NAN),
                gradient_bias_scale: Some(-1.0),
                clip_scale: Some(0.0),
                ..ToposTrainingHintsInput::default()
            }),
            Some(ToposInferenceHintsInput {
                temperature_scale: Some(10.0),
                top_p_scale: Some(0.0),
                presence_penalty_bias: Some(f32::NAN),
                context_weight: Some(0.0),
                ..ToposInferenceHintsInput::default()
            }),
        ));

        assert_eq!(payload.training_hints.learning_rate_scale, 1.25);
        assert_eq!(
            payload.training_hints.regularization_scale,
            base_training.regularization_scale()
        );
        assert_eq!(payload.training_hints.gradient_bias_scale, 0.0);
        assert_eq!(payload.training_hints.clip_scale, 0.25);
        assert_eq!(payload.training_plan.gain, 0.5);
        assert!((payload.training_plan.raw_rate_scale - 1.25).abs() < 1e-6);
        assert!((payload.training_plan.rate_scale - 1.125).abs() < 1e-6);
        assert!((payload.training_plan.effective_gradient_clip_scale - 0.625).abs() < 1e-6);
        assert_eq!(payload.inference_hints.temperature_scale, 1.5);
        assert_eq!(payload.inference_hints.top_p_scale, 0.05);
        assert_eq!(payload.inference_hints.context_weight, 0.25);
        assert_eq!(
            payload.inference_hints.presence_penalty_bias,
            base_inference.presence_penalty_bias()
        );
        assert_eq!(payload.inference_plan.temperature, 0.7);
        assert_eq!(payload.inference_plan.top_p, 0.8);
        assert_eq!(
            payload.runtime_route.runtime_profile,
            payload.runtime_profile
        );
    }

    #[test]
    fn topos_optimizer_snapshot_binds_one_control_bundle_to_optimizer_state() {
        let signal = unwrap_ok(ToposControlSignal::from_input(ToposControlSignalInput {
            observed_depth: 7,
            visited_volume: 31,
            ..ToposControlSignalInput::default()
        }));
        let snapshot = unwrap_ok(signal.optimizer_snapshot(
            42,
            0.04,
            0.02,
            ToposControlPlanOptions {
                training_gain: 0.5,
                ..ToposControlPlanOptions::default()
            },
            Some(ToposTrainingHintsInput {
                learning_rate_scale: Some(0.6),
                clip_scale: Some(0.8),
                ..ToposTrainingHintsInput::default()
            }),
            None,
        ));

        assert_eq!(snapshot.kind, TOPOS_OPTIMIZER_SNAPSHOT_KIND);
        assert_eq!(
            snapshot.contract_version,
            TOPOS_OPTIMIZER_SNAPSHOT_CONTRACT_VERSION
        );
        assert_eq!(
            snapshot.semantic_owner,
            TOPOS_OPTIMIZER_SNAPSHOT_SEMANTIC_OWNER
        );
        assert_eq!(
            snapshot.semantic_backend,
            TOPOS_OPTIMIZER_SNAPSHOT_SEMANTIC_BACKEND
        );
        assert_eq!(snapshot.sequence, 42);
        assert_eq!(snapshot.control.observed_depth, 7);
        assert_eq!(snapshot.control.visited_volume, 31);
        assert_eq!(
            snapshot.optimizer_application.rate_scale,
            snapshot.control.training_plan.rate_scale
        );
        assert_eq!(
            snapshot.optimizer_application.scope,
            "learning_rate_and_gradient_state"
        );
        assert_eq!(
            snapshot.optimizer_application.control_path,
            "control.training_plan"
        );
        assert!(
            (snapshot.optimizer_application.hyper_learning_rate
                - 0.04 * snapshot.control.training_plan.rate_scale)
                .abs()
                < 1e-6
        );
        assert!(
            (snapshot.optimizer_application.real_learning_rate
                - 0.02 * snapshot.control.training_plan.rate_scale)
                .abs()
                < 1e-6
        );

        let serialized = serde_json::to_value(snapshot).expect("snapshot is serializable");
        assert_eq!(serialized["control"]["semantic_backend"], "rust");
        assert_eq!(
            serialized["optimizer_application"]["control_path"],
            "control.training_plan"
        );
        assert_eq!(
            serialized["optimizer_application"]["gradient_bias_rule"],
            TOPOS_OPTIMIZER_GRADIENT_BIAS_RULE
        );
        assert_eq!(
            serialized["optimizer_application"]["gradient_bias_normalization"],
            TOPOS_OPTIMIZER_GRADIENT_BIAS_NORMALIZATION
        );
        assert_eq!(
            serialized["optimizer_application"]["effective_gradient_bias_scale"],
            serialized["control"]["training_plan"]["effective_gradient_bias_scale"]
        );
        assert_eq!(
            serialized["optimizer_application"]["effective_gradient_clip_scale"],
            serialized["control"]["training_plan"]["effective_gradient_clip_scale"]
        );
        assert_eq!(
            serialized["optimizer_application"]["gradient_clip_rule"],
            TOPOS_OPTIMIZER_GRADIENT_CLIP_RULE
        );
        assert_eq!(
            serialized["optimizer_application"]["gradient_clip_normalization"],
            TOPOS_OPTIMIZER_GRADIENT_CLIP_NORMALIZATION
        );
        assert_eq!(
            serialized["optimizer_application"]["effective_momentum_damping"],
            serialized["control"]["training_plan"]["effective_momentum_damping"]
        );
        assert_eq!(
            serialized["optimizer_application"]["gradient_bias_basis_dim"],
            TOPOS_OPTIMIZER_GRADIENT_BIAS_BASIS_DIM
        );
        assert_eq!(
            serialized["optimizer_application"]["gradient_bias_basis"]
                .as_array()
                .expect("gradient-bias basis array")
                .len(),
            TOPOS_OPTIMIZER_GRADIENT_BIAS_BASIS_DIM
        );
    }

    #[test]
    fn optimizer_state_control_is_neutral_and_zero_safe() {
        let raw = [2.0, -1.0];
        let previous = [7.0, -8.0];
        let neutral = ToposOptimizerStateControl::neutral();
        let neutral_step = unwrap_ok(neutral.gradient_step(&raw, &previous));
        assert_eq!(neutral_step.applied_gradient(), raw.as_slice());

        let controlled = unwrap_ok(ToposOptimizerStateControl::new(
            0.35,
            [1.0; TOPOS_OPTIMIZER_GRADIENT_BIAS_BASIS_DIM],
            0.25,
            0.5,
        ));
        let zeros = [0.0, 0.0];
        let zero_step = unwrap_ok(controlled.gradient_step(&zeros, &zeros));
        assert_eq!(zero_step.raw_gradient_rms(), 0.0);
        assert_eq!(zero_step.gradient_bias_amplitude(), 0.0);
        assert_eq!(zero_step.biased_gradient_rms(), 0.0);
        assert_eq!(zero_step.gradient_clip_threshold(), Some(0.0));
        assert_eq!(zero_step.clipped_values(), 0);
        assert_eq!(zero_step.applied_gradient(), zeros.as_slice());
    }

    #[test]
    fn optimizer_state_control_applies_rms_bias_and_momentum_equation() {
        let mut basis = [0.0; TOPOS_OPTIMIZER_GRADIENT_BIAS_BASIS_DIM];
        basis[0] = 1.0;
        basis[1] = -1.0;
        let control = unwrap_ok(ToposOptimizerStateControl::new(0.2, basis, 1.0, 0.25));
        let step = unwrap_ok(control.gradient_step(&[3.0, 4.0], &[1.0, -1.0]));
        let rms = (12.5f32).sqrt();
        let amplitude = rms * 0.2;
        assert!((step.raw_gradient_rms() - rms).abs() < 1e-6);
        assert!((step.gradient_bias_amplitude() - amplitude).abs() < 1e-6);
        assert!((step.applied_gradient()[0] - (0.25 + 0.75 * (3.0 + amplitude))).abs() < 1e-6);
        assert!((step.applied_gradient()[1] - (-0.25 + 0.75 * (4.0 - amplitude))).abs() < 1e-6);
        assert_eq!(step.applied_gradient(), step.next_momentum());
    }

    #[test]
    fn optimizer_state_control_clips_outliers_relative_to_biased_rms() {
        let zeros = [0.0; TOPOS_OPTIMIZER_GRADIENT_BIAS_BASIS_DIM];
        let clipped = unwrap_ok(ToposOptimizerStateControl::new(0.0, zeros, 0.5, 0.0));
        let unguarded = unwrap_ok(ToposOptimizerStateControl::new(0.0, zeros, 1.0, 0.0));
        let mut raw = [0.0; TOPOS_OPTIMIZER_GRADIENT_BIAS_BASIS_DIM];
        raw[0] = 10.0;
        let previous = [0.0; TOPOS_OPTIMIZER_GRADIENT_BIAS_BASIS_DIM];

        let step = unwrap_ok(clipped.gradient_step(&raw, &previous));
        let unguarded_step = unwrap_ok(unguarded.gradient_step(&raw, &previous));
        let expected_rms = 10.0f64 / (TOPOS_OPTIMIZER_GRADIENT_BIAS_BASIS_DIM as f64).sqrt();
        assert!((step.biased_gradient_rms() - expected_rms).abs() < 1e-6);
        assert!(
            (step.gradient_clip_threshold().expect("active clip") - expected_rms / 0.5).abs()
                < 1e-12
        );
        assert_eq!(step.clipped_values(), 1);
        assert!(step.applied_gradient()[0] < raw[0]);
        assert_eq!(unguarded_step.applied_gradient(), raw.as_slice());
        assert_eq!(unguarded_step.gradient_clip_threshold(), None);

        let scaled_raw = raw.map(|value| value * 10.0);
        let scaled_step = unwrap_ok(clipped.gradient_step(&scaled_raw, &previous));
        for (scaled, base) in scaled_step
            .applied_gradient()
            .iter()
            .zip(step.applied_gradient())
        {
            assert!((*scaled - 10.0 * base).abs() < 1e-5);
        }
    }

    #[test]
    fn topos_optimizer_snapshot_guards_sequence_and_learning_rate_boundaries() {
        let signal = unwrap_ok(ToposControlSignal::from_input(
            ToposControlSignalInput::default(),
        ));
        let max_sequence = unwrap_ok(signal.optimizer_snapshot(
            TOPOS_OPTIMIZER_SNAPSHOT_MAX_SEQUENCE,
            0.04,
            0.02,
            ToposControlPlanOptions::default(),
            None,
            None,
        ));
        assert_eq!(max_sequence.sequence, TOPOS_OPTIMIZER_SNAPSHOT_MAX_SEQUENCE);

        let unsafe_sequence = unwrap_err(signal.optimizer_snapshot(
            TOPOS_OPTIMIZER_SNAPSHOT_MAX_SEQUENCE + 1,
            0.04,
            0.02,
            ToposControlPlanOptions::default(),
            None,
            None,
        ));
        assert!(matches!(
            unsafe_sequence,
            TensorError::InvalidValue {
                label: "topos_optimizer_snapshot_sequence"
            }
        ));

        for rate in [0.0, -0.01] {
            let error = unwrap_err(signal.optimizer_snapshot(
                1,
                rate,
                0.02,
                ToposControlPlanOptions::default(),
                None,
                None,
            ));
            assert!(matches!(error, TensorError::NonPositiveLearningRate { .. }));
        }
        let non_finite = unwrap_err(signal.optimizer_snapshot(
            1,
            f32::NAN,
            0.02,
            ToposControlPlanOptions::default(),
            None,
            None,
        ));
        assert!(matches!(
            non_finite,
            TensorError::NonFiniteValue {
                label: "topos_optimizer_input_hyper_learning_rate",
                ..
            }
        ));

        let overflow = unwrap_err(signal.optimizer_snapshot(
            1,
            f32::MAX,
            0.02,
            ToposControlPlanOptions::default(),
            Some(ToposTrainingHintsInput {
                learning_rate_scale: Some(1.25),
                clip_scale: Some(1.25),
                ..ToposTrainingHintsInput::default()
            }),
            None,
        ));
        assert!(matches!(
            overflow,
            TensorError::NonFiniteValue {
                label: "topos_optimizer_output_hyper_learning_rate",
                ..
            }
        ));
    }

    #[test]
    fn topos_control_payload_rejects_invalid_provider_bounds() {
        let signal = unwrap_ok(ToposControlSignal::from_input(
            ToposControlSignalInput::default(),
        ));
        let reversed = unwrap_err(signal.payload_with_options(
            ToposControlPlanOptions {
                inference: ToposInferencePlanOptions {
                    min_temperature: 1.0,
                    max_temperature: 0.5,
                    ..ToposInferencePlanOptions::default()
                },
                ..ToposControlPlanOptions::default()
            },
            None,
            None,
        ));
        assert!(matches!(
            reversed,
            TensorError::InvalidValue {
                label: "topos_inference_temperature_bounds"
            }
        ));

        let non_finite = unwrap_err(signal.payload_with_options(
            ToposControlPlanOptions {
                inference: ToposInferencePlanOptions {
                    min_top_p: f32::NAN,
                    ..ToposInferencePlanOptions::default()
                },
                ..ToposControlPlanOptions::default()
            },
            None,
            None,
        ));
        assert!(matches!(
            non_finite,
            TensorError::NonFiniteValue {
                label: "topos_inference_min_top_p",
                ..
            }
        ));
    }

    #[test]
    fn biome_control_signal_reflects_retained_volume() {
        let topos = unwrap_ok(OpenCartesianTopos::new(-1.0, 1e-5, 10.0, 64, 8));
        let mut biome = TensorBiome::new(topos);
        unwrap_ok(biome.absorb(
            "signal_a",
            unwrap_ok(Tensor::from_vec(1, 2, vec![0.1, 0.2])),
        ));
        unwrap_ok(biome.absorb(
            "signal_b",
            unwrap_ok(Tensor::from_vec(1, 2, vec![0.3, 0.4])),
        ));
        let signal = biome.control_signal();
        assert_eq!(biome.stored_volume(), 4);
        assert_eq!(signal.visited_volume(), 4);
        assert!((signal.volume_pressure() - 0.5).abs() < 1e-6);
        assert!((signal.openness() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn reward_boundary_porosity_delays_breach() {
        let topos = demo_topos();
        let strict_boundary = unwrap_ok(RewardBoundary::with_porosity(-0.5, 0.5, 0.05, 0.0));
        let porous_boundary = unwrap_ok(RewardBoundary::with_porosity(-0.5, 0.5, 0.05, 0.8));
        let strict_guard = unwrap_ok(MultiModalToposGuard::new(&topos));
        let strict_guard = unwrap_ok(strict_guard.with_reward_boundary(strict_boundary));
        let porous_guard = unwrap_ok(MultiModalToposGuard::new(&topos));
        let porous_guard = unwrap_ok(porous_guard.with_reward_boundary(porous_boundary));
        let mut strict_trace = vec![0.56f32];
        let mut porous_trace = vec![0.56f32];
        let strict_signal = unwrap_ok(strict_guard.guard_reward_trace(&mut strict_trace));
        let porous_signal = unwrap_ok(porous_guard.guard_reward_trace(&mut porous_trace));
        assert!(strict_signal.upper_breach_index.is_some());
        assert!(porous_signal.upper_breach_index.is_none());
        assert!(strict_trace[0] <= 0.5 + 1e-6);
        assert!(porous_trace[0] < 0.5);
    }

    #[test]
    fn graph_profile_rejects_invalid_porosity() {
        let profile = unwrap_ok(GraphGuardProfile::new(8, 12, 4, 1e-3, 0.05, Some(1.0)));
        let err = unwrap_err(profile.with_porosity(1.5));
        assert!(matches!(err, TensorError::PorosityOutOfRange { .. }));
    }

    #[test]
    fn lawvere_guard_reports_precise_non_finite_parameters() {
        let error = unwrap_err(LawvereTierneyGuard::new(f32::NAN, 1.0, 1e-5));
        assert!(matches!(
            error,
            TensorError::NonFiniteValue {
                label: "lawvere_tierney_density_min",
                ..
            }
        ));

        let error = unwrap_err(LawvereTierneyGuard::new(1e-6, 1.0, 0.0));
        assert!(matches!(error, TensorError::InvalidValue { .. }));
    }

    #[test]
    fn lawvere_projection_preserves_mass_and_density_window_together() {
        let guard = unwrap_ok(LawvereTierneyGuard::new(0.1, 0.4, 1e-5));
        let mut slice = [0.9f32, 0.1, 0.1];
        unwrap_ok(guard.project_slice("bounded_simplex", &mut slice, 1.0));

        assert!((slice.iter().sum::<f32>() - 1.0).abs() <= guard.mass_tolerance());
        assert!(slice
            .iter()
            .all(|&value| (guard.density_min()..=guard.density_max()).contains(&value)));
        assert!((slice[0] - 0.4).abs() < 1e-5);
    }

    #[test]
    fn lawvere_projection_preserves_support_ratios_and_is_idempotent() {
        let guard = unwrap_ok(LawvereTierneyGuard::new(0.01, 0.8, 1e-5));
        let mut slice = [0.0f32, 0.4, 0.2, 0.1];
        unwrap_ok(guard.project_slice("proportional_simplex", &mut slice, 1.0));
        let once = slice;
        unwrap_ok(guard.project_slice("proportional_simplex", &mut slice, 1.0));

        assert_eq!(slice[0], 0.0);
        assert!((slice[1] / slice[2] - 2.0).abs() < 1e-5);
        assert!((slice[2] / slice[3] - 2.0).abs() < 1e-5);
        for (left, right) in slice.iter().zip(once.iter()) {
            assert!((left - right).abs() < 1e-6);
        }
    }

    #[test]
    fn lawvere_projection_is_transactional_when_support_is_infeasible() {
        let guard = unwrap_ok(LawvereTierneyGuard::new(0.05, 0.2, 1e-5));
        let mut slice = [0.5f32, 0.3, 0.2];
        let original = slice;
        let error = unwrap_err(guard.project_slice("infeasible_simplex", &mut slice, 1.0));

        assert!(matches!(
            error,
            TensorError::ProbabilityProjectionInfeasible {
                active_values: 3,
                ..
            }
        ));
        assert_eq!(slice, original);
    }

    #[test]
    fn lawvere_projection_rejects_invalid_saturation_without_mutation() {
        let guard = unwrap_ok(LawvereTierneyGuard::new(0.1, 1.0, 1e-5));
        let mut slice = [0.6f32, 0.4];
        let original = slice;
        let error = unwrap_err(guard.project_slice("invalid_saturation", &mut slice, f32::NAN));

        assert!(matches!(
            error,
            TensorError::NonFiniteValue {
                label: "lawvere_tierney_saturation",
                ..
            }
        ));
        assert_eq!(slice, original);
    }

    #[test]
    fn zbox_rejects_invalid_density_index_and_overflowing_volume() {
        let error = unwrap_err(ZBox::new(vec![vec![0.0]], vec![1.0], 0.0));
        assert!(matches!(error, TensorError::InvalidValue { .. }));

        let zbox = unwrap_ok(ZBox::new(vec![vec![0.0, 0.0, 0.0]], vec![64.0], 1.0));
        let error = unwrap_err(zbox.factor_dimension(1));
        assert!(matches!(error, TensorError::InvalidValue { .. }));
        let error = unwrap_err(zbox.hyperbolic_volume(f32::NAN));
        assert!(matches!(error, TensorError::NonFiniteValue { .. }));
        let error = unwrap_err(zbox.hyperbolic_volume(-1.0));
        assert!(matches!(error, TensorError::NonFiniteValue { .. }));
    }

    #[test]
    fn zbox_cover_respects_mass() {
        let topos = demo_topos();
        let centers_a = vec![vec![0.0f32, 0.0]];
        let centers_b = vec![vec![0.25f32, -0.1]];
        let radii = vec![0.4f32];
        let base = unwrap_ok(ZBox::new(centers_a.clone(), radii.clone(), 1.0));
        let volume = unwrap_ok(base.hyperbolic_volume(topos.curvature()));
        let density = 0.5 / volume;
        let box_a = unwrap_ok(ZBox::new(centers_a.clone(), radii.clone(), density));
        let box_b = unwrap_ok(ZBox::new(centers_b.clone(), radii.clone(), density));
        unwrap_ok(topos.guard_cover(&[box_a.clone(), box_b.clone()]));
        let heavy_density = 0.8 / volume;
        let heavy = unwrap_ok(ZBox::new(centers_b, radii, heavy_density));
        let err = unwrap_err(topos.guard_cover(&[box_a, heavy]));
        assert!(matches!(err, TensorError::InvalidValue { .. }));
    }

    #[test]
    fn topos_rejects_non_finite_values() {
        let topos = demo_topos();
        let tensor = unwrap_ok(Tensor::from_vec(1, 2, vec![1.0, f32::INFINITY]));
        let err = unwrap_err(topos.guard_tensor("nonfinite", &tensor));
        assert!(matches!(err, TensorError::NonFiniteValue { .. }));
    }

    #[test]
    fn biome_absorbs_and_harvests() {
        let topos = demo_topos();
        let mut biome = TensorBiome::new(topos.clone());
        let big = topos.saturation() * 2.0;
        unwrap_ok(biome.absorb(
            "biome_shoot_a",
            unwrap_ok(Tensor::from_vec(1, 2, vec![big, 0.5])),
        ));
        unwrap_ok(biome.absorb(
            "biome_shoot_b",
            unwrap_ok(Tensor::from_vec(1, 2, vec![-big, 1.0])),
        ));
        let canopy = unwrap_ok(biome.canopy());
        assert_eq!(canopy.shape(), (1, 2));
        let data = canopy.data();
        assert!((data[0] - 0.0).abs() < 1e-6);
        assert!((data[1] - 0.75).abs() < 1e-6);
        assert_eq!(biome.total_weight(), 2.0);
    }

    #[test]
    fn biome_detects_shape_mismatch() {
        let topos = demo_topos();
        let mut biome = TensorBiome::new(topos);
        unwrap_ok(biome.absorb(
            "biome_shape_a",
            unwrap_ok(Tensor::from_vec(2, 1, vec![0.1, 0.2])),
        ));
        let err = unwrap_err(biome.absorb(
            "biome_shape_b",
            unwrap_ok(Tensor::from_vec(1, 2, vec![0.1, 0.2])),
        ));
        assert!(matches!(err, TensorError::ShapeMismatch { .. }));
    }

    #[test]
    fn biome_rejects_aggregate_volume_without_mutation() {
        let topos = unwrap_ok(OpenCartesianTopos::new(-1.0, 1e-5, 10.0, 8, 2));
        let mut biome = TensorBiome::new(topos);
        unwrap_ok(biome.absorb("volume_a", unwrap_ok(Tensor::from_vec(1, 1, vec![1.0]))));
        unwrap_ok(biome.absorb("volume_b", unwrap_ok(Tensor::from_vec(1, 1, vec![2.0]))));
        let shoots_before = biome.shoots().to_vec();
        let weights_before = biome.weights().to_vec();

        let error =
            unwrap_err(biome.absorb("volume_c", unwrap_ok(Tensor::from_vec(1, 1, vec![3.0]))));

        assert!(matches!(error, TensorError::TensorVolumeExceeded { .. }));
        assert_eq!(biome.shoots(), shoots_before);
        assert_eq!(biome.weights(), weights_before);
        assert_eq!(biome.stored_volume(), 2);
        assert_eq!(biome.total_weight(), 2.0);
        assert_eq!(biome.shape(), Some((1, 1)));
    }

    #[test]
    fn biome_rejects_total_weight_overflow_without_mutation() {
        let topos = unwrap_ok(OpenCartesianTopos::new(-1.0, 1e-5, 10.0, 8, 2));
        let mut biome = TensorBiome::new(topos);
        unwrap_ok(biome.absorb_weighted(
            "weight_a",
            unwrap_ok(Tensor::from_vec(1, 1, vec![1.0])),
            f32::MAX,
        ));

        let error = unwrap_err(biome.absorb_weighted(
            "weight_b",
            unwrap_ok(Tensor::from_vec(1, 1, vec![2.0])),
            f32::MAX,
        ));

        assert!(matches!(
            error,
            TensorError::NonFiniteValue {
                label: "tensor_biome_total_weight",
                ..
            }
        ));
        assert_eq!(biome.len(), 1);
        assert_eq!(biome.weights(), &[f32::MAX]);
        assert_eq!(biome.total_weight(), f32::MAX);
        assert_eq!(unwrap_ok(biome.canopy()).data(), &[1.0]);
    }

    #[test]
    fn biome_canopy_avoids_intermediate_weight_overflow() {
        let topos = demo_topos();
        let mut biome = TensorBiome::new(topos);
        let weight = f32::MAX * 0.25;
        unwrap_ok(biome.absorb_weighted(
            "large_weight_a",
            unwrap_ok(Tensor::from_vec(1, 1, vec![8.0])),
            weight,
        ));
        unwrap_ok(biome.absorb_weighted(
            "large_weight_b",
            unwrap_ok(Tensor::from_vec(1, 1, vec![4.0])),
            weight,
        ));

        let canopy = unwrap_ok(biome.canopy());

        assert!((canopy.data()[0] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn biome_weighted_canopy_respects_shoot_weights() {
        let topos = demo_topos();
        let mut biome = TensorBiome::new(topos);
        unwrap_ok(biome.absorb_weighted(
            "weighted_a",
            unwrap_ok(Tensor::from_vec(1, 1, vec![1.0])),
            1.0,
        ));
        unwrap_ok(biome.absorb_weighted(
            "weighted_b",
            unwrap_ok(Tensor::from_vec(1, 1, vec![3.0])),
            3.0,
        ));
        let canopy = unwrap_ok(biome.canopy());
        assert_eq!(canopy.data(), &[2.5]);
        assert_eq!(biome.weights(), &[1.0, 3.0]);
        assert!((biome.total_weight() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn biome_absorb_renormalise_and_canopy_emit_backend_meta() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = crate::set_thread_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let topos = demo_topos();
        let mut biome = TensorBiome::new(topos);
        unwrap_ok(biome.absorb_weighted(
            "meta_weighted_a",
            unwrap_ok(Tensor::from_vec(1, 2, vec![1.0, 2.0])),
            2.0,
        ));
        unwrap_ok(biome.absorb_weighted(
            "meta_weighted_b",
            unwrap_ok(Tensor::from_vec(1, 2, vec![3.0, 4.0])),
            3.0,
        ));
        unwrap_ok(biome.renormalise_weights());
        let canopy = unwrap_ok(biome.canopy());
        crate::set_thread_meta_observer(previous);

        assert_eq!(canopy.shape(), (1, 2));
        let events = events.lock().unwrap();
        let absorbs = events
            .iter()
            .filter(|(op_name, _)| *op_name == "tensor_biome_absorb_weighted")
            .collect::<Vec<_>>();
        assert_eq!(absorbs.len(), 2);
        assert_eq!(absorbs[0].1["backend"], "topos_cpu");
        assert_eq!(absorbs[0].1["kind"], "topos_biome_absorb");
        assert_eq!(absorbs[0].1["rewrite_backend"], "topos_cpu");
        assert_eq!(absorbs[0].1["storage_backend"], "host_vec");
        assert_eq!(
            absorbs[0].1["rewrite_mode"],
            "guarded_in_place_tensor_rewrite"
        );
        assert_eq!(
            absorbs[0].1["route_blocker"],
            "open_cartesian_topos_rewrite"
        );
        assert_eq!(absorbs[0].1["label"], "meta_weighted_a");
        assert_eq!(absorbs[0].1["shoots"], 1);
        assert_eq!(absorbs[1].1["shoots"], 2);
        assert_eq!(absorbs[1].1["estimated_stored_values"], 4);

        let renorm = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "tensor_biome_renormalise_weights" && data["weights"] == 2
            })
            .expect("tensor_biome_renormalise_weights metadata event");
        assert_eq!(renorm.1["backend"], "control_cpu");
        assert_eq!(renorm.1["requested_backend"], "host");
        assert_eq!(renorm.1["kind"], "topos_biome_weight_reduction");
        assert_eq!(renorm.1["weight_guard_backend"], "lawvere_topos_cpu");
        assert_eq!(renorm.1["reduction_backend"], "host_slice_sum");
        assert_eq!(renorm.1["guard_mode"], "probability_slice_guard");
        assert_eq!(
            renorm.1["route_blocker"],
            "lawvere_tierney_probability_guard"
        );
        assert_eq!(renorm.1["shoots"], 2);
        assert!(renorm.1["changed_weights"].as_u64().unwrap_or(0) >= 1);
        assert!((renorm.1["total_weight"].as_f64().unwrap_or(0.0) - 1.0).abs() < 1.0e-6);

        let canopy_meta = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "tensor_biome_canopy" && data["rows"] == 1 && data["cols"] == 2
            })
            .expect("tensor_biome_canopy metadata event");
        assert_eq!(canopy_meta.1["backend"], "hybrid");
        assert_eq!(canopy_meta.1["kind"], "topos_biome_canopy");
        assert_eq!(canopy_meta.1["requested_backend"], "auto");
        assert_eq!(canopy_meta.1["accumulation_backend"], "auto");
        assert_eq!(canopy_meta.1["normalise_backend"], "auto");
        assert_eq!(canopy_meta.1["weight_normalise_backend"], "host_f64");
        assert_eq!(canopy_meta.1["rewrite_backend"], "topos_cpu");
        assert_eq!(canopy_meta.1["rewrite_mode"], "guarded_canopy_rewrite");
        assert_eq!(
            canopy_meta.1["route_blocker"],
            "open_cartesian_topos_rewrite_after_tensor_util_accumulation"
        );
        assert_eq!(canopy_meta.1["shoots"], 2);
        assert_eq!(canopy_meta.1["weights"], 2);
        assert_eq!(canopy_meta.1["estimated_accumulation_values"], 4);
        assert_eq!(canopy_meta.1["estimated_rewrite_values"], 2);
    }

    #[test]
    fn biome_canopy_with_backend_records_explicit_tensor_util_backend() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = crate::set_thread_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let topos = demo_topos();
        let mut biome = TensorBiome::new(topos);
        unwrap_ok(biome.absorb(
            "explicit_canopy_a",
            unwrap_ok(Tensor::from_vec(1, 2, vec![1.0, 2.0])),
        ));
        unwrap_ok(biome.absorb(
            "explicit_canopy_b",
            unwrap_ok(Tensor::from_vec(1, 2, vec![3.0, 4.0])),
        ));
        let canopy = unwrap_ok(biome.canopy_with_backend(TensorUtilBackend::Cpu));
        crate::set_thread_meta_observer(previous);

        assert_eq!(canopy.shape(), (1, 2));
        let events = events.lock().unwrap();
        let canopy_meta = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "tensor_biome_canopy" && data["rows"] == 1 && data["cols"] == 2
            })
            .expect("tensor_biome_canopy metadata event");
        assert_eq!(canopy_meta.1["backend"], "hybrid");
        assert_eq!(canopy_meta.1["requested_backend"], "cpu");
        assert_eq!(canopy_meta.1["accumulation_backend"], "cpu");
        assert_eq!(canopy_meta.1["normalise_backend"], "cpu");
        assert_eq!(canopy_meta.1["weight_normalise_backend"], "host_f64");
        assert_eq!(canopy_meta.1["rewrite_backend"], "topos_cpu");
    }

    #[test]
    fn biome_stack_concatenates_shoots() {
        let topos = demo_topos();
        let mut biome = TensorBiome::new(topos);
        unwrap_ok(biome.absorb("stack_a", unwrap_ok(Tensor::from_vec(1, 2, vec![0.1, 0.2]))));
        unwrap_ok(biome.absorb("stack_b", unwrap_ok(Tensor::from_vec(1, 2, vec![0.3, 0.4]))));
        let stacked = unwrap_ok(biome.stack());
        assert_eq!(stacked.shape(), (2, 2));
        assert_eq!(stacked.data(), &[0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    fn rewrite_monad_saturates_values() {
        let topos = demo_topos();
        let monad = RewriteMonad::new(&topos);
        let mut tensor = unwrap_ok(Tensor::from_vec(1, 2, vec![20.0, -20.0]));
        unwrap_ok(monad.rewrite_tensor("rewrite", &mut tensor));
        assert!(tensor.data().iter().all(|v| v.abs() <= topos.saturation()));
    }

    #[test]
    fn rewrite_monad_rejects_oversized_tensor_without_mutation() {
        let topos = unwrap_ok(OpenCartesianTopos::new(-1.0, 1e-5, 1.0, 8, 1));
        let monad = RewriteMonad::new(&topos);
        let mut tensor = unwrap_ok(Tensor::from_vec(1, 2, vec![100.0, -100.0]));
        let original = tensor.clone();

        let error = unwrap_err(monad.rewrite_tensor("oversized_rewrite", &mut tensor));

        assert!(matches!(error, TensorError::TensorVolumeExceeded { .. }));
        assert_eq!(tensor, original);
    }

    #[test]
    fn rewrite_monad_bind_slice_rolls_back_callback_failure() {
        let topos = demo_topos();
        let monad = RewriteMonad::new(&topos);
        let mut slice = [0.25f32, 0.5];
        let original = slice;

        let error = unwrap_err(monad.bind_slice("failing_bind", &mut slice, |candidate| {
            candidate[0] = 9.0;
            Err(TensorError::Generic("bind failed".to_string()))
        }));

        assert!(matches!(error, TensorError::Generic(_)));
        assert_eq!(slice, original);
    }

    #[test]
    fn probability_tensor_volume_failure_is_transactional() {
        let topos = unwrap_ok(OpenCartesianTopos::new(-1.0, 1e-5, 1.0, 8, 1));
        let mut tensor = unwrap_ok(Tensor::from_vec(1, 2, vec![2.0, 1.0]));
        let original = tensor.clone();

        let error =
            unwrap_err(topos.guard_probability_tensor("oversized_probability", &mut tensor));

        assert!(matches!(error, TensorError::TensorVolumeExceeded { .. }));
        assert_eq!(tensor, original);
    }

    #[test]
    fn rewrite_monad_surfaces_lawvere_guard() {
        let topos = demo_topos();
        let monad = RewriteMonad::new(&topos);
        let guard = monad.lawvere_guard();
        assert!(guard.density_min() > 0.0);
        assert!(guard.density_max() > guard.density_min());
    }

    #[test]
    fn rewrite_monad_lift_and_bind_tensor() {
        let topos = demo_topos();
        let monad = RewriteMonad::new(&topos);
        let lifted = unwrap_ok(monad.lift_tensor(
            "lift",
            unwrap_ok(Tensor::from_vec(1, 2, vec![topos.saturation() * 4.0, 0.25])),
        ));
        assert!(lifted
            .data()
            .iter()
            .all(|v| v.is_finite() && v.abs() <= topos.saturation()));

        let bound = unwrap_ok(monad.bind_tensor(
            "bind",
            unwrap_ok(Tensor::from_vec(1, 2, vec![0.1, 0.2])),
            |tensor| {
                let update = unwrap_ok(Tensor::from_vec(1, 2, vec![0.3, 0.4]));
                tensor.add_scaled(&update, 1.0)
            },
        ));
        assert_eq!(bound.shape(), (1, 2));
        assert!(bound.data().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn topos_normalises_probability_slices() {
        let topos = demo_topos();
        let mut slice = vec![2.0, -1.0, 0.5];
        unwrap_ok(topos.guard_probability_slice("probability_guard", &mut slice));
        assert!(slice.iter().all(|v| *v >= 0.0));
        let sum: f32 = slice.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn topos_probability_guard_emits_backend_meta() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = crate::set_thread_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let topos = demo_topos();
        let mut slice = vec![2.0, -1.0, f32::INFINITY, 0.5];
        unwrap_ok(topos.guard_probability_slice("probability_guard_meta", &mut slice));
        crate::set_thread_meta_observer(previous);

        assert!(slice.iter().all(|value| value.is_finite() && *value >= 0.0));
        assert!((slice.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        let events = events.lock().unwrap();
        let meta = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "lawvere_guard_probability_slice"
                    && data["label"] == "probability_guard_meta"
            })
            .expect("lawvere_guard_probability_slice metadata event");
        assert_eq!(meta.1["backend"], "control_cpu");
        assert_eq!(meta.1["requested_backend"], "host");
        assert_eq!(meta.1["kind"], "lawvere_tierney_probability_guard");
        assert_eq!(meta.1["values"], 4);
        assert_eq!(meta.1["repaired_non_finite"], 1);
        assert_eq!(meta.1["repaired_negative"], 1);
        assert!(meta.1["guarded_sum"].as_f64().unwrap_or(0.0) > 0.0);
        assert!((meta.1["final_sum"].as_f64().unwrap_or(0.0) - 1.0).abs() < 1e-6);
        assert!(meta.1["repaired_values"].as_u64().unwrap_or(0) >= 2);
    }

    #[test]
    fn atlas_tracks_volume_and_depth() {
        let topos = demo_topos();
        let mut atlas = ToposAtlas::new(&topos);
        let tensor = unwrap_ok(Tensor::from_vec(1, 2, vec![0.1, 0.2]));
        unwrap_ok(atlas.guard_tensor("atlas_tensor", &tensor));
        assert_eq!(atlas.visited_volume(), 2);
        assert_eq!(atlas.remaining_volume(), topos.max_volume() - 2);
        let patch = unwrap_ok(FractalPatch::new(
            unwrap_ok(Tensor::from_vec(1, 2, vec![0.3, 0.4])),
            1.0,
            1.0,
            1,
        ));
        unwrap_ok(atlas.guard_fractal_patch("atlas_patch", &patch));
        assert_eq!(atlas.depth(), 1);
        assert_eq!(atlas.visited_volume(), 4);
    }

    #[test]
    fn atlas_state_and_mutations_commit_together() {
        let topos = unwrap_ok(OpenCartesianTopos::new(-1.0, 1e-5, 1.0, 8, 1));
        let mut atlas = ToposAtlas::new(&topos);
        let invalid = unwrap_ok(Tensor::from_vec(1, 1, vec![f32::NAN]));
        let error = unwrap_err(atlas.guard_tensor("invalid_atlas_tensor", &invalid));
        assert!(matches!(error, TensorError::NonFiniteValue { .. }));
        assert_eq!(atlas.visited_volume(), 0);

        let valid = unwrap_ok(Tensor::from_vec(1, 1, vec![0.5]));
        unwrap_ok(atlas.guard_tensor("valid_atlas_tensor", &valid));
        assert_eq!(atlas.visited_volume(), 1);

        let mut candidate = unwrap_ok(Tensor::from_vec(1, 1, vec![100.0]));
        let error = unwrap_err(atlas.guard_tensor_mut("full_atlas", &mut candidate));
        assert!(matches!(error, TensorError::TensorVolumeExceeded { .. }));
        assert_eq!(candidate.data(), &[100.0]);
        assert_eq!(atlas.visited_volume(), 1);
    }

    #[test]
    fn atlas_lifts_tensor_through_monad() {
        let topos = demo_topos();
        let mut atlas = ToposAtlas::new(&topos);
        let lifted = unwrap_ok(atlas.lift_tensor(
            "atlas_lift",
            unwrap_ok(Tensor::from_vec(1, 2, vec![topos.saturation() * 5.0, 0.5])),
        ));
        assert!(lifted.data().iter().all(|v| v.abs() <= topos.saturation()));
        assert_eq!(atlas.visited_volume(), 2);
    }

    #[test]
    fn biome_absorbs_fractal_patches() {
        let topos = demo_topos();
        let mut biome = TensorBiome::new(topos.clone());
        let patch = unwrap_ok(FractalPatch::new(
            unwrap_ok(Tensor::from_vec(1, 1, vec![2.0])),
            2.0,
            1.0,
            0,
        ));
        unwrap_ok(biome.absorb_fractal_patch(&patch));
        assert_eq!(biome.len(), 1);
        let canopy = unwrap_ok(biome.canopy());
        assert_eq!(canopy.data(), &[2.0]);
    }

    #[test]
    fn biome_absorb_with_monadic_builder() {
        let topos = demo_topos();
        let mut biome = TensorBiome::new(topos.clone());
        unwrap_ok(biome.absorb_with("monadic", |monad| {
            monad.lift_tensor(
                "monadic_build",
                unwrap_ok(Tensor::from_vec(1, 2, vec![topos.saturation() * 3.0, 0.5])),
            )
        }));
        assert_eq!(biome.len(), 1);
        let canopy = unwrap_ok(biome.canopy());
        assert_eq!(canopy.shape(), (1, 2));
    }

    #[test]
    fn biome_absorb_weighted_with_monadic_builder() {
        let topos = demo_topos();
        let mut biome = TensorBiome::new(topos.clone());
        unwrap_ok(
            biome.absorb_weighted_with("weighted_monadic", 2.0, |monad| {
                monad.bind_tensor(
                    "weighted_monadic_build",
                    unwrap_ok(Tensor::from_vec(1, 1, vec![1.0])),
                    |tensor| {
                        let update = unwrap_ok(Tensor::from_vec(1, 1, vec![1.0]));
                        tensor.add_scaled(&update, 1.0)
                    },
                )
            }),
        );
        let canopy = unwrap_ok(biome.canopy());
        assert_eq!(canopy.data(), &[2.0]);
        assert!((biome.total_weight() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn biome_rejects_invalid_weight_before_running_builder() {
        let topos = demo_topos();
        let mut biome = TensorBiome::new(topos);
        let called = Cell::new(false);

        let error = unwrap_err(biome.absorb_weighted_with("invalid_weight", f32::NAN, |_| {
            called.set(true);
            Tensor::from_vec(1, 1, vec![1.0])
        }));

        assert!(matches!(error, TensorError::NonPositiveWeight { .. }));
        assert!(!called.get());
        assert!(biome.is_empty());
    }

    #[test]
    fn biome_bind_shoots_rewrites_each_shoot() {
        let topos = demo_topos();
        let mut biome = TensorBiome::new(topos.clone());
        unwrap_ok(biome.absorb(
            "bind_shoot",
            unwrap_ok(Tensor::from_vec(1, 1, vec![topos.saturation() * 4.0])),
        ));
        unwrap_ok(biome.bind_shoots("bind_shoots", |monad, shoot| {
            let update = monad.lift_tensor(
                "bind_shoot_update",
                unwrap_ok(Tensor::from_vec(1, 1, vec![0.5])),
            )?;
            shoot.add_scaled(&update, 1.0)
        }));
        let canopy = unwrap_ok(biome.canopy());
        assert!(canopy.data()[0].abs() <= topos.saturation());
    }

    #[test]
    fn biome_bind_shoots_rolls_back_late_callback_failure() {
        let topos = demo_topos();
        let mut biome = TensorBiome::new(topos);
        unwrap_ok(biome.absorb("bind_a", unwrap_ok(Tensor::from_vec(1, 1, vec![1.0]))));
        unwrap_ok(biome.absorb("bind_b", unwrap_ok(Tensor::from_vec(1, 1, vec![2.0]))));
        let shoots_before = biome.shoots().to_vec();
        let calls = Cell::new(0usize);

        let error = unwrap_err(biome.bind_shoots("failing_bind_shoots", |_, shoot| {
            calls.set(calls.get() + 1);
            shoot.data_mut()[0] += 10.0;
            if calls.get() == 2 {
                return Err(TensorError::Generic("second shoot failed".to_string()));
            }
            Ok(())
        }));

        assert!(matches!(error, TensorError::Generic(_)));
        assert_eq!(calls.get(), 2);
        assert_eq!(biome.shoots(), shoots_before);
        assert_eq!(biome.shape(), Some((1, 1)));
        assert_eq!(biome.stored_volume(), 2);
    }

    #[test]
    fn biome_bind_shoots_commits_a_common_shape_change() {
        let topos = demo_topos();
        let mut biome = TensorBiome::new(topos);
        unwrap_ok(biome.absorb("reshape_a", unwrap_ok(Tensor::from_vec(1, 1, vec![1.0]))));
        unwrap_ok(biome.absorb("reshape_b", unwrap_ok(Tensor::from_vec(1, 1, vec![2.0]))));

        unwrap_ok(biome.bind_shoots("reshape_shoots", |_, shoot| {
            let value = shoot.data()[0];
            *shoot = Tensor::from_vec(1, 2, vec![value, value + 0.5])?;
            Ok(())
        }));

        assert_eq!(biome.shape(), Some((1, 2)));
        assert_eq!(biome.stored_volume(), 4);
        assert_eq!(unwrap_ok(biome.stack()).shape(), (2, 2));
    }

    #[test]
    fn biome_bind_shoots_rejects_divergent_shapes_transactionally() {
        let topos = demo_topos();
        let mut biome = TensorBiome::new(topos);
        unwrap_ok(biome.absorb("shape_bind_a", unwrap_ok(Tensor::from_vec(1, 1, vec![1.0]))));
        unwrap_ok(biome.absorb("shape_bind_b", unwrap_ok(Tensor::from_vec(1, 1, vec![2.0]))));
        let shoots_before = biome.shoots().to_vec();
        let calls = Cell::new(0usize);

        let error = unwrap_err(biome.bind_shoots("divergent_shapes", |_, shoot| {
            calls.set(calls.get() + 1);
            let value = shoot.data()[0];
            *shoot = if calls.get() == 1 {
                Tensor::from_vec(1, 2, vec![value, value])?
            } else {
                Tensor::from_vec(2, 1, vec![value, value])?
            };
            Ok(())
        }));

        assert!(matches!(error, TensorError::ShapeMismatch { .. }));
        assert_eq!(biome.shoots(), shoots_before);
        assert_eq!(biome.shape(), Some((1, 1)));
        assert_eq!(biome.stored_volume(), 2);
    }

    #[test]
    fn biome_map_shoots_reuses_weights() {
        let topos = demo_topos();
        let mut biome = TensorBiome::new(topos.clone());
        unwrap_ok(biome.absorb_weighted(
            "map_a",
            unwrap_ok(Tensor::from_vec(1, 1, vec![1.0])),
            2.0,
        ));
        unwrap_ok(biome.absorb_weighted(
            "map_b",
            unwrap_ok(Tensor::from_vec(1, 1, vec![2.0])),
            3.0,
        ));
        let base_canopy = unwrap_ok(biome.canopy());
        let mapped = unwrap_ok(biome.map_shoots("map_transform", |monad, shoot| {
            monad.bind_tensor("map_transform_build", shoot.clone(), |tensor| {
                let update = unwrap_ok(Tensor::from_vec(1, 1, vec![1.0]));
                tensor.add_scaled(&update, 1.0)
            })
        }));
        assert_eq!(mapped.len(), biome.len());
        assert!((mapped.total_weight() - biome.total_weight()).abs() < 1e-6);
        let mapped_canopy = unwrap_ok(mapped.canopy());
        assert!((mapped_canopy.data()[0] - (base_canopy.data()[0] + 1.0)).abs() < 1e-6);
    }

    #[test]
    fn biome_renormalises_weights_preserves_canopy() {
        let topos = demo_topos();
        let mut biome = TensorBiome::new(topos.clone());
        unwrap_ok(biome.absorb_weighted(
            "renorm_a",
            unwrap_ok(Tensor::from_vec(1, 1, vec![1.0])),
            2.0,
        ));
        unwrap_ok(biome.absorb_weighted(
            "renorm_b",
            unwrap_ok(Tensor::from_vec(1, 1, vec![3.0])),
            3.0,
        ));
        let canopy_before = unwrap_ok(biome.canopy());
        unwrap_ok(biome.renormalise_weights());
        let sum: f32 = biome.weights().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!((biome.total_weight() - 1.0).abs() < 1e-6);
        let canopy_after = unwrap_ok(biome.canopy());
        assert!((canopy_before.data()[0] - canopy_after.data()[0]).abs() < 1e-6);
    }

    #[test]
    fn monad_cultivates_biome_and_absorbs_weighted() {
        let topos = demo_topos();
        let monad = RewriteMonad::new(&topos);
        let mut biome = monad.cultivate_biome();
        unwrap_ok(monad.absorb_weighted_into_biome(
            &mut biome,
            "monad_absorb",
            unwrap_ok(Tensor::from_vec(1, 1, vec![topos.saturation() * 6.0])),
            2.0,
        ));
        assert_eq!(biome.len(), 1);
        assert!((biome.total_weight() - 2.0).abs() < 1e-6);
        let canopy = unwrap_ok(biome.canopy());
        assert!(canopy.data()[0].abs() <= topos.saturation());
    }

    #[test]
    fn monad_absorption_respects_the_destination_biome_topos() {
        let source_topos = unwrap_ok(OpenCartesianTopos::new(-1.0, 1e-5, 10.0, 8, 4));
        let destination_topos = unwrap_ok(OpenCartesianTopos::new(-1.0, 1e-5, 1.0, 8, 1));
        let monad = RewriteMonad::new(&source_topos);
        let mut biome = TensorBiome::new(destination_topos);

        let error = unwrap_err(monad.absorb_into_biome(
            &mut biome,
            "destination_volume",
            unwrap_ok(Tensor::from_vec(1, 2, vec![8.0, 8.0])),
        ));
        assert!(matches!(error, TensorError::TensorVolumeExceeded { .. }));
        assert!(biome.is_empty());

        unwrap_ok(monad.absorb_into_biome(
            &mut biome,
            "destination_saturation",
            unwrap_ok(Tensor::from_vec(1, 1, vec![8.0])),
        ));
        let canopy = unwrap_ok(biome.canopy());
        assert!(canopy.data()[0].is_finite());
        assert!(canopy.data()[0].abs() <= biome.topos().saturation());
        assert!(canopy.data()[0] < 8.0);
    }

    #[test]
    fn conjugate_gradient_converges_with_guard() {
        let topos = demo_topos();
        let solver = unwrap_ok(ConjugateGradientSolver::new(&topos, 1e-5, 32));
        let matrix = [4.0f32, 1.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 2.0];
        let mut matvec = |src: &[f32], dst: &mut [f32]| {
            dst.fill(0.0);
            for row in 0..3 {
                for col in 0..3 {
                    dst[row] += matrix[row * 3 + col] * src[col];
                }
            }
        };
        let b = [1.0f32, 2.0, 3.0];
        let mut x = [0.0f32; 3];
        let iterations = unwrap_ok(solver.solve(&mut matvec, &b, &mut x));
        assert!(iterations > 0);
        let mut residual = [0.0f32; 3];
        matvec(&x, &mut residual);
        for (res, rhs) in residual.iter_mut().zip(b.iter()) {
            *res -= rhs;
        }
        let norm: f32 = residual.iter().map(|v| v * v).sum();
        assert!(norm.sqrt() <= solver.tolerance().max(topos.tolerance()));
    }

    #[test]
    fn conjugate_gradient_rejects_non_finite_tolerance() {
        let topos = demo_topos();
        let error = unwrap_err(ConjugateGradientSolver::new(&topos, f32::NAN, 8));
        assert!(matches!(
            error,
            TensorError::NonFiniteValue {
                label: "conjugate_gradient_tolerance",
                ..
            }
        ));
    }

    #[test]
    fn conjugate_gradient_rejects_partial_matvec_without_mutating_solution() {
        let topos = demo_topos();
        let solver = unwrap_ok(ConjugateGradientSolver::new(&topos, 1e-5, 8));
        let b = [1.0f32, 2.0];
        let mut x = [0.25f32, 0.5];
        let original = x;
        let error = unwrap_err(solver.solve(
            |_src, dst| {
                dst[0] = 1.0;
            },
            &b,
            &mut x,
        ));

        assert!(matches!(
            error,
            TensorError::NonFiniteValue {
                label: "cg_initial_matvec",
                ..
            }
        ));
        assert_eq!(x, original);
    }

    #[test]
    fn conjugate_gradient_reports_non_spd_breakdown_transactionally() {
        let topos = demo_topos();
        let solver = unwrap_ok(ConjugateGradientSolver::new(&topos, 1e-5, 8));
        let b = [1.0f32, 1.0];
        let mut x = [0.0f32, 0.0];
        let original = x;
        let error = unwrap_err(solver.solve(
            |src, dst| {
                dst[0] = src[0];
                dst[1] = -src[1];
            },
            &b,
            &mut x,
        ));

        assert!(matches!(
            error,
            TensorError::ConjugateGradientBreakdown { iteration: 0, .. }
        ));
        assert_eq!(x, original);
    }

    #[test]
    fn conjugate_gradient_rolls_back_when_guarded_solution_cannot_converge() {
        let topos = demo_topos();
        let solver = unwrap_ok(ConjugateGradientSolver::new(&topos, 1e-5, 2));
        assert_eq!(solver.max_iterations(), 2);
        let b = [100.0f32];
        let mut x = [1.0f32];
        let original = x;
        let error = unwrap_err(solver.solve(
            |src, dst| {
                dst.copy_from_slice(src);
            },
            &b,
            &mut x,
        ));

        assert!(matches!(
            error,
            TensorError::ConjugateGradientDiverged { .. }
        ));
        assert_eq!(x, original);
    }

    #[test]
    fn multi_modal_text_profile_clamps_values() {
        let topos = demo_topos();
        let text_profile = unwrap_ok(ModalityProfile::new(16, Some(0.25)));
        let guard = unwrap_ok(MultiModalToposGuard::new(&topos));
        let guard = unwrap_ok(guard.with_text_profile(text_profile));
        let mut tensor = unwrap_ok(Tensor::from_vec(4, 4, vec![10.0; 16]));
        unwrap_ok(guard.guard_text_tensor("multi_modal_text", &mut tensor));
        let headroom = 0.25 * (1.0 + DEFAULT_MODALITY_PERMEABILITY);
        assert!(tensor
            .data()
            .iter()
            .all(|&value| value <= headroom + 1e-6 && value >= 0.25 - 1e-6));
    }

    #[test]
    fn modality_rewrite_is_transactional_on_non_finite_input() {
        let topos = demo_topos();
        let text_profile = unwrap_ok(ModalityProfile::new(2, Some(0.25)));
        assert_eq!(text_profile.local_saturation(), Some(0.25));
        let guard = unwrap_ok(MultiModalToposGuard::new(&topos));
        let guard = unwrap_ok(guard.with_text_profile(text_profile));
        let mut tensor = unwrap_ok(Tensor::from_vec(1, 2, vec![10.0, f32::NAN]));
        let error = unwrap_err(guard.guard_text_tensor("transactional_text", &mut tensor));

        assert!(matches!(error, TensorError::NonFiniteValue { .. }));
        assert_eq!(tensor.data()[0], 10.0);
        assert!(tensor.data()[1].is_nan());
    }

    #[test]
    fn graph_profile_enforces_topology_and_activation_contracts() {
        let error = unwrap_err(GraphGuardProfile::new(4, 6, 4, 1e-3, 0.1, None));
        assert!(matches!(error, TensorError::InvalidValue { .. }));
        let error = unwrap_err(GraphGuardProfile::new(4, 6, 3, 1e-3, 0.0, None));
        assert!(matches!(error, TensorError::InvalidValue { .. }));
        let error = unwrap_err(GraphGuardProfile::new(4, 6, 3, f32::NAN, 0.1, None));
        assert!(matches!(
            error,
            TensorError::NonFiniteValue {
                label: "graph_symmetry_tolerance",
                ..
            }
        ));

        let small_topos = unwrap_ok(OpenCartesianTopos::new(-1.0, 1e-5, 1.0, 8, 9));
        let guard = unwrap_ok(MultiModalToposGuard::new(&small_topos));
        assert_eq!(guard.graph_profile().max_nodes(), 3);
        assert_eq!(guard.graph_profile().max_degree(), 2);
        assert!(guard.graph_profile().activation_threshold() > 0.0);
    }

    #[test]
    fn sparse_graph_uses_edge_budget_not_dense_storage_size() {
        let topos = unwrap_ok(OpenCartesianTopos::new(-1.0, 1e-5, 1.0, 8, 64));
        let profile = unwrap_ok(GraphGuardProfile::new(8, 1, 7, 1e-3, 0.1, Some(1.0)));
        let profile = unwrap_ok(profile.with_permeability(0.0));
        let guard = unwrap_ok(MultiModalToposGuard::new(&topos));
        let guard = unwrap_ok(guard.with_graph_profile(profile));
        let mut adjacency = vec![0.0f32; 64];
        let report = unwrap_ok(guard.guard_graph_adjacency(&mut adjacency, 8));

        assert_eq!(report.edge_count, 0);
        assert_eq!(report.max_degree, 0);
        assert_eq!(report.edge_overflow, 0);
    }

    #[test]
    fn graph_report_is_measured_after_canonicalization() {
        let topos = demo_topos();
        let profile = unwrap_ok(GraphGuardProfile::new(2, 1, 1, 1e-3, 0.05, Some(1.0)));
        let guard = unwrap_ok(MultiModalToposGuard::new(&topos));
        let guard = unwrap_ok(guard.with_graph_profile(profile));
        let mut forward = vec![f32::NAN, 10.0, 0.2, 0.0];
        let mut reverse = vec![f32::NAN, 0.2, 10.0, 0.0];

        let forward_report = unwrap_ok(guard.guard_graph_adjacency(&mut forward, 2));
        let reverse_report = unwrap_ok(guard.guard_graph_adjacency(&mut reverse, 2));

        assert_eq!(forward_report, reverse_report);
        assert_eq!(forward_report.edge_count, 1);
        assert_eq!(forward_report.max_degree, 1);
        assert_eq!(forward_report.symmetry_violations, 1);
        assert_eq!(forward_report.self_loops, 0);
        assert_eq!(forward_report.repaired_non_finite, 1);
        assert_eq!(forward_report.saturated_entries, 1);
        assert_eq!(forward[0], 0.0);
        assert_eq!(reverse[0], 0.0);
    }

    #[test]
    fn graph_guard_rolls_back_on_degree_and_edge_budget_failures() {
        let topos = demo_topos();
        let mut adjacency = vec![0.0f32, 2.0, 2.0, 2.0, 0.0, 2.0, 2.0, 2.0, 0.0];
        let original = adjacency.clone();

        let degree_profile = unwrap_ok(GraphGuardProfile::new(3, 3, 1, 1e-3, 0.1, Some(0.5)));
        let guard = unwrap_ok(MultiModalToposGuard::new(&topos));
        let degree_guard = unwrap_ok(guard.with_graph_profile(degree_profile));
        let error = unwrap_err(degree_guard.guard_graph_adjacency(&mut adjacency, 3));
        assert!(matches!(error, TensorError::InvalidValue { .. }));
        assert_eq!(adjacency, original);

        let edge_profile = unwrap_ok(GraphGuardProfile::new(3, 1, 2, 1e-3, 0.1, Some(0.5)));
        let edge_profile = unwrap_ok(edge_profile.with_permeability(0.0));
        let edge_guard = unwrap_ok(guard.with_graph_profile(edge_profile));
        let error = unwrap_err(edge_guard.guard_graph_adjacency(&mut adjacency, 3));
        assert!(matches!(error, TensorError::TensorVolumeExceeded { .. }));
        assert_eq!(adjacency, original);
    }

    #[test]
    fn multi_modal_graph_guard_reports_symmetry() {
        let topos = demo_topos();
        let graph_profile = unwrap_ok(GraphGuardProfile::new(8, 12, 4, 1e-3, 0.05, Some(1.0)));
        let guard = unwrap_ok(MultiModalToposGuard::new(&topos));
        let guard = unwrap_ok(guard.with_graph_profile(graph_profile));
        let mut adjacency = vec![0.0f32; 16];
        adjacency[1] = 0.2;
        adjacency[4] = 0.1;
        let report = unwrap_ok(guard.guard_graph_adjacency(&mut adjacency, 4));
        assert_eq!(report.edge_count, 1);
        assert_eq!(report.symmetry_violations, 1);
        assert_eq!(report.edge_overflow, 0);
        assert!(adjacency.iter().all(|value| value.abs() <= 1.0));
    }

    #[test]
    fn multi_modal_reward_boundary_detects_breaches() {
        let topos = demo_topos();
        let boundary = unwrap_ok(RewardBoundary::new(-0.5, 0.5, 0.1));
        let guard = unwrap_ok(MultiModalToposGuard::new(&topos));
        let guard = unwrap_ok(guard.with_reward_boundary(boundary));
        let mut rewards = vec![-0.9, 0.2, 0.8];
        let signal = unwrap_ok(guard.guard_reward_trace(&mut rewards));
        assert_eq!(signal.lower_breach_index, Some(0));
        assert_eq!(signal.upper_breach_index, Some(2));
        assert_eq!(signal.clamped, 2);
        assert!((signal.min_observed + 0.9).abs() < 1e-6);
        assert!((signal.max_observed - 0.8).abs() < 1e-6);
        assert!(rewards.iter().all(|value| *value >= -0.5 && *value <= 0.5));
    }

    #[test]
    fn reward_boundary_clamps_windows_on_either_side_of_zero() {
        let positive = unwrap_ok(RewardBoundary::with_porosity(1.0, 2.0, 0.1, 0.5));
        let negative = unwrap_ok(RewardBoundary::with_porosity(-2.0, -1.0, 0.1, 0.5));
        assert_eq!(positive.hysteresis(), 0.1);

        for value in [0.0f32, 3.0] {
            let clamped = positive.clamp(value);
            assert!((positive.lower()..=positive.upper()).contains(&clamped));
        }
        for value in [-3.0f32, 0.0] {
            let clamped = negative.clamp(value);
            assert!((negative.lower()..=negative.upper()).contains(&clamped));
        }
    }

    #[test]
    fn reward_guard_observes_raw_runaway_before_topos_saturation() {
        let topos = unwrap_ok(OpenCartesianTopos::new(-1.0, 1e-5, 1.0, 8, 8));
        let boundary = unwrap_ok(RewardBoundary::new(-0.98, 0.98, 0.0));
        let guard = unwrap_ok(MultiModalToposGuard::new(&topos));
        let guard = unwrap_ok(guard.with_reward_boundary(boundary));
        let mut rewards = [100.0f32];
        let signal = unwrap_ok(guard.guard_reward_trace(&mut rewards));

        assert_eq!(signal.upper_breach_index, Some(0));
        assert_eq!(signal.max_observed, 100.0);
        assert_eq!(signal.clamped, 1);
        assert!(rewards[0].abs() <= topos.saturation());
    }

    #[test]
    fn reward_boundary_cannot_escape_the_global_topos_envelope() {
        let topos = unwrap_ok(OpenCartesianTopos::new(-1.0, 1e-5, 1.0, 8, 8));
        let guard = unwrap_ok(MultiModalToposGuard::new(&topos));
        let boundary = unwrap_ok(RewardBoundary::new(-2.0, 2.0, 0.1));
        let error = unwrap_err(guard.with_reward_boundary(boundary));

        assert!(matches!(
            error,
            TensorError::InvalidValue {
                label: "reward_boundary_exceeds_topos_saturation"
            }
        ));
    }

    #[test]
    fn reward_guard_is_transactional_on_non_finite_tail() {
        let topos = demo_topos();
        let guard = unwrap_ok(MultiModalToposGuard::new(&topos));
        let mut rewards = [100.0f32, f32::NAN];
        let error = unwrap_err(guard.guard_reward_trace(&mut rewards));

        assert!(matches!(error, TensorError::NonFiniteValue { .. }));
        assert_eq!(rewards[0], 100.0);
        assert!(rewards[1].is_nan());
    }

    #[test]
    fn multi_modal_atlas_tracks_volume_and_lifts() {
        let topos = demo_topos();
        let guard = unwrap_ok(MultiModalToposGuard::new(&topos));
        let mut atlas = guard.atlas();
        let mut text = unwrap_ok(Tensor::from_vec(2, 4, vec![topos.saturation() * 4.0; 8]));
        unwrap_ok(atlas.guard_text_tensor("multi_modal_atlas_text", &mut text));
        assert_eq!(atlas.visited_volume(), 8);
        let text_limit = atlas.guard.text.effective_saturation(atlas.guard.topos);
        let text_headroom = text_limit * (1.0 + atlas.guard.text.permeability());
        assert!(text
            .data()
            .iter()
            .all(|&value| value.abs() <= text_headroom + 1e-5));
        let lifted = unwrap_ok(atlas.lift_vision_tensor(
            "multi_modal_atlas_vision",
            unwrap_ok(Tensor::from_vec(1, 4, vec![topos.saturation() * 3.0; 4])),
        ));
        let vision_limit = atlas.guard.vision.effective_saturation(atlas.guard.topos);
        let vision_headroom = vision_limit * (1.0 + atlas.guard.vision.permeability());
        assert!(lifted
            .data()
            .iter()
            .all(|&value| value.abs() <= vision_headroom + 1e-5));
        assert_eq!(atlas.visited_volume(), 12);
    }

    #[test]
    fn multi_modal_atlas_does_not_rewrite_when_capacity_commit_fails() {
        let topos = unwrap_ok(OpenCartesianTopos::new(-1.0, 1e-5, 1.0, 8, 1));
        let guard = unwrap_ok(MultiModalToposGuard::new(&topos));
        let mut atlas = guard.atlas();
        let mut first = unwrap_ok(Tensor::from_vec(1, 1, vec![0.5]));
        unwrap_ok(atlas.guard_text_tensor("first_text", &mut first));
        assert_eq!(atlas.visited_volume(), 1);

        let mut second = unwrap_ok(Tensor::from_vec(1, 1, vec![100.0]));
        let error = unwrap_err(atlas.guard_text_tensor("overflow_text", &mut second));
        assert!(matches!(error, TensorError::TensorVolumeExceeded { .. }));
        assert_eq!(second.data(), &[100.0]);
        assert_eq!(atlas.visited_volume(), 1);
    }

    #[test]
    fn multi_modal_biome_absorbs_with_preserved_profiles() {
        let topos = demo_topos();
        let text_profile = unwrap_ok(ModalityProfile::new(64, Some(0.5)));
        let text_profile = unwrap_ok(text_profile.with_permeability(0.2));
        let guard = unwrap_ok(MultiModalToposGuard::new(&topos));
        let guard = unwrap_ok(guard.with_text_profile(text_profile));
        let mut biome = guard.cultivate_biome();
        assert_eq!(biome.text_profile().local_saturation(), Some(0.5));
        assert_eq!(
            biome.audio_profile().max_volume(),
            guard.audio_profile().max_volume()
        );
        assert_eq!(
            biome.vision_profile().max_volume(),
            guard.vision_profile().max_volume()
        );
        assert_eq!(
            biome.graph_profile().max_nodes(),
            guard.graph_profile().max_nodes()
        );
        assert_eq!(
            biome.reward_boundary().lower(),
            guard.reward_boundary().lower()
        );
        unwrap_ok(biome.absorb_text(
            "multi_modal_biome_text",
            unwrap_ok(Tensor::from_vec(1, 4, vec![2.0; 4])),
        ));
        unwrap_ok(biome.absorb_vision_weighted(
            "multi_modal_biome_vision",
            unwrap_ok(Tensor::from_vec(1, 4, vec![5.0; 4])),
            2.0,
        ));
        assert_eq!(biome.len(), 2);
        assert!((biome.total_weight() - 3.0).abs() < 1e-6);
        assert_eq!(biome.shape(), Some((1, 4)));
        assert_eq!(biome.stored_volume(), 8);
        assert_eq!(
            biome.remaining_volume(),
            biome.topos().max_volume() - biome.stored_volume()
        );
        let canopy = unwrap_ok(biome.canopy());
        let text_limit = biome.text.effective_saturation(biome.topos());
        let text_headroom = text_limit * (1.0 + biome.text.permeability());
        let vision_limit = biome.vision.effective_saturation(biome.topos());
        let vision_headroom = vision_limit * (1.0 + biome.vision.permeability());
        let max_headroom = text_headroom.max(vision_headroom);
        assert!(canopy
            .data()
            .iter()
            .all(|&value| value.abs() <= max_headroom + 1e-5));
        let mut atlas = biome.atlas();
        let mut waveform = vec![topos.saturation() * 6.0; 4];
        unwrap_ok(atlas.guard_audio_waveform("multi_modal_biome_audio", &mut waveform));
        let audio_limit = biome.audio.effective_saturation(biome.topos());
        let audio_headroom = audio_limit * (1.0 + biome.audio.permeability());
        assert!(waveform
            .iter()
            .all(|&value| value.abs() <= audio_headroom + 1e-5));
        let _lifted = unwrap_ok(atlas.lift_text_tensor(
            "multi_modal_biome_lift",
            unwrap_ok(Tensor::from_vec(1, 4, vec![topos.saturation() * 2.5; 4])),
        ));
        assert_eq!(atlas.visited_volume(), 4);
    }
}
