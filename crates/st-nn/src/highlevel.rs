// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::module::Module;
use crate::trainer::{EpochStats, ModuleTrainer};
use crate::{BandEnergy, GradientBands, Loss, RoundtableConfig, RoundtableSchedule};
use st_core::backend::device_caps::{BackendKind, DeviceCaps};
use st_core::backend::unison_heuristics::RankKind;
use st_core::ops::rank_entry::{plan_rank, RankPlan};
use st_core::telemetry::atlas::{AtlasFragment, AtlasFrame, AtlasRoute};
use st_core::telemetry::chrono::{
    ChronoFrame, ChronoHarmonics, ChronoLoopSignal, ChronoSummary, ChronoTimeline,
    ResonanceTemporalMetrics,
};
use st_core::telemetry::hub;
use st_core::telemetry::maintainer::{Maintainer, MaintainerConfig, MaintainerReport};
use st_tensor::pure::measure::{z_space_barycenter, ZSpaceBarycenter};
use st_tensor::pure::{
    AmegaHypergrad, DifferentialResonance, FunctorDifferential, HomotopyDifferential,
    InfinityDifferential, OpenCartesianTopos, PureResult, RecursiveDifferential,
    SpiralDifferential, Tensor, TensorError,
};
use st_text::{ResonanceNarrative, TextResonator};
use std::sync::{Arc, Mutex};

/// Configuration describing how the session evaluates the Z-space barycenter objective.
#[derive(Debug, Clone)]
pub struct BarycenterConfig {
    /// Entropy regulariser weight \(\gamma_S\).
    pub entropy_weight: f32,
    /// Coupling scale \(\beta_J\).
    pub beta_j: f32,
    /// Optional coupling matrix.
    pub coupling: Option<Tensor>,
}

impl Default for BarycenterConfig {
    fn default() -> Self {
        Self {
            entropy_weight: 0.1,
            beta_j: 0.0,
            coupling: None,
        }
    }
}

impl BarycenterConfig {
    /// Creates a config with custom entropy and coupling weights.
    pub fn new(entropy_weight: f32, beta_j: f32) -> Self {
        Self {
            entropy_weight,
            beta_j,
            coupling: None,
        }
    }

    /// Overrides the coupling matrix.
    pub fn with_coupling(mut self, coupling: Tensor) -> Self {
        self.coupling = Some(coupling);
        self
    }

    /// Returns the optional coupling matrix reference.
    pub fn coupling(&self) -> Option<&Tensor> {
        self.coupling.as_ref()
    }
}

/// Builder for [`SpiralSession`].
#[derive(Debug, Clone)]
pub struct SpiralSessionBuilder {
    caps: DeviceCaps,
    curvature: f32,
    hyper_learning_rate: f32,
    fallback_learning_rate: f32,
    barycenter: BarycenterConfig,
    topos: Option<OpenCartesianTopos>,
    chrono_capacity: usize,
    maintainer: MaintainerConfig,
}

impl SpiralSessionBuilder {
    /// Creates a new builder with heuristic defaults for entropy weights and learning rates.
    pub fn new(caps: DeviceCaps) -> Self {
        Self {
            caps,
            curvature: -1.0,
            hyper_learning_rate: 0.05,
            fallback_learning_rate: 0.01,
            barycenter: BarycenterConfig::default(),
            topos: None,
            chrono_capacity: 256,
            maintainer: MaintainerConfig::default(),
        }
    }

    /// Builder convenience constructor.
    pub fn from_backend(backend: BackendKind) -> Self {
        let caps = match backend {
            BackendKind::Wgpu => DeviceCaps::wgpu(32, true, 256),
            BackendKind::Hip => DeviceCaps::hip(32, 1024, Some(64 * 1024)),
            BackendKind::Cuda => DeviceCaps::cuda(32, 1024, Some(96 * 1024)),
            BackendKind::Cpu => DeviceCaps::cpu(),
        };
        Self::new(caps)
    }

    /// Sets the hyperbolic curvature enforced by the session.
    pub fn with_curvature(mut self, curvature: f32) -> Self {
        self.curvature = curvature;
        self
    }

    /// Updates the curvature in-place.
    pub fn set_curvature(&mut self, curvature: f32) {
        self.curvature = curvature;
    }

    /// Sets the hypergradient learning rate.
    pub fn with_hyper_learning_rate(mut self, learning_rate: f32) -> Self {
        self.hyper_learning_rate = learning_rate;
        self
    }

    /// Updates the hypergradient learning rate in-place.
    pub fn set_hyper_learning_rate(&mut self, learning_rate: f32) {
        self.hyper_learning_rate = learning_rate;
    }

    /// Sets the Euclidean fallback learning rate.
    pub fn with_fallback_learning_rate(mut self, learning_rate: f32) -> Self {
        self.fallback_learning_rate = learning_rate;
        self
    }

    /// Updates the fallback learning rate in-place.
    pub fn set_fallback_learning_rate(&mut self, learning_rate: f32) {
        self.fallback_learning_rate = learning_rate;
    }

    /// Overrides the entropy regulariser \(\gamma_S\).
    pub fn with_barycenter_entropy(mut self, entropy_weight: f32) -> Self {
        self.barycenter.entropy_weight = entropy_weight;
        self
    }

    /// Updates the entropy regulariser in-place.
    pub fn set_barycenter_entropy(&mut self, entropy_weight: f32) {
        self.barycenter.entropy_weight = entropy_weight;
    }

    /// Overrides the coupling scale \(\beta_J\).
    pub fn with_barycenter_beta_j(mut self, beta_j: f32) -> Self {
        self.barycenter.beta_j = beta_j;
        self
    }

    /// Updates the coupling scale in-place.
    pub fn set_barycenter_beta_j(&mut self, beta_j: f32) {
        self.barycenter.beta_j = beta_j;
    }

    /// Installs a coupling matrix for the barycenter objective.
    pub fn with_barycenter_coupling(mut self, coupling: Tensor) -> Self {
        self.barycenter.coupling = Some(coupling);
        self
    }

    /// Overrides the maintainer configuration used for self-rewrite assessments.
    pub fn with_maintainer_config(mut self, config: MaintainerConfig) -> Self {
        self.maintainer = config;
        self
    }

    /// Updates the coupling matrix in-place.
    pub fn set_barycenter_coupling(&mut self, coupling: Option<Tensor>) {
        self.barycenter.coupling = coupling;
    }

    /// Returns the maintainer configuration reference.
    pub fn maintainer_config(&self) -> &MaintainerConfig {
        &self.maintainer
    }

    /// Updates the maintainer configuration in-place.
    pub fn set_maintainer_config(&mut self, config: MaintainerConfig) {
        self.maintainer = config;
    }

    /// Installs an open-cartesian topos guard for all hypergradient tapes.
    pub fn with_topos(mut self, topos: OpenCartesianTopos) -> Self {
        self.topos = Some(topos);
        self
    }

    /// Updates the topos guard in-place.
    pub fn set_topos(&mut self, topos: Option<OpenCartesianTopos>) {
        self.topos = topos;
    }

    /// Overrides the maximum number of temporal resonance frames retained in the session.
    pub fn with_chrono_capacity(mut self, capacity: usize) -> Self {
        self.chrono_capacity = capacity.max(1);
        self
    }

    /// Updates the chrono capacity in-place.
    pub fn set_chrono_capacity(&mut self, capacity: usize) {
        self.chrono_capacity = capacity.max(1);
    }

    /// Convenience helper to construct a guard from raw parameters.
    pub fn configure_topos(
        mut self,
        curvature: f32,
        tolerance: f32,
        saturation: f32,
        max_depth: usize,
        max_volume: usize,
    ) -> PureResult<Self> {
        self.topos = Some(OpenCartesianTopos::new(
            curvature, tolerance, saturation, max_depth, max_volume,
        )?);
        Ok(self)
    }

    /// In-place version of [`Self::configure_topos`].
    pub fn set_topos_from_params(
        &mut self,
        curvature: f32,
        tolerance: f32,
        saturation: f32,
        max_depth: usize,
        max_volume: usize,
    ) -> PureResult<()> {
        self.topos = Some(OpenCartesianTopos::new(
            curvature, tolerance, saturation, max_depth, max_volume,
        )?);
        Ok(())
    }

    fn validate(&self) -> PureResult<()> {
        if self.curvature >= 0.0 || !self.curvature.is_finite() {
            return Err(TensorError::NonHyperbolicCurvature {
                curvature: self.curvature,
            });
        }
        if self.hyper_learning_rate <= 0.0 || !self.hyper_learning_rate.is_finite() {
            return Err(TensorError::NonPositiveLearningRate {
                rate: self.hyper_learning_rate,
            });
        }
        if self.fallback_learning_rate <= 0.0 || !self.fallback_learning_rate.is_finite() {
            return Err(TensorError::NonPositiveLearningRate {
                rate: self.fallback_learning_rate,
            });
        }
        if !self.barycenter.entropy_weight.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "entropy_weight",
                value: self.barycenter.entropy_weight,
            });
        }
        if !self.barycenter.beta_j.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "beta_j",
                value: self.barycenter.beta_j,
            });
        }
        Ok(())
    }

    /// Finalises the builder and returns a [`SpiralSession`].
    pub fn build(self) -> PureResult<SpiralSession> {
        self.validate()?;
        let maintainer = self.maintainer.clone().sanitise();
        Ok(SpiralSession {
            caps: self.caps,
            curvature: self.curvature,
            hyper_learning_rate: self.hyper_learning_rate,
            fallback_learning_rate: self.fallback_learning_rate,
            barycenter: self.barycenter,
            topos: self.topos,
            chrono: Arc::new(Mutex::new(ChronoTimeline::with_capacity(
                self.chrono_capacity,
            ))),
            maintainer,
        })
    }
}

/// High-level orchestrator that stitches together Z-space barycenters, hypergrad tapes,
/// and rank planning into a single intuitive surface.
#[derive(Debug, Clone)]
pub struct SpiralSession {
    caps: DeviceCaps,
    curvature: f32,
    hyper_learning_rate: f32,
    fallback_learning_rate: f32,
    barycenter: BarycenterConfig,
    topos: Option<OpenCartesianTopos>,
    chrono: Arc<Mutex<ChronoTimeline>>,
    maintainer: MaintainerConfig,
}

/// Builder-style trace that wires homotopy, functor, recursive, and \(\infty\)-tower flows.
#[derive(Debug, Clone)]
pub struct DifferentialTrace {
    session: SpiralSession,
    seed: Tensor,
    generator: Option<Tensor>,
    direction: Option<Tensor>,
    kernel: Option<Tensor>,
    functor_source: Option<Tensor>,
    epsilon: f32,
    barycenter: Option<ZSpaceBarycenter>,
    infinity_levels: Vec<Tensor>,
    infinity_curvatures: Vec<f32>,
    topos_override: Option<OpenCartesianTopos>,
}

impl DifferentialTrace {
    fn new(session: SpiralSession, seed: Tensor) -> Self {
        let epsilon = session.hyper_learning_rate();
        Self {
            session,
            seed,
            generator: None,
            direction: None,
            kernel: None,
            functor_source: None,
            epsilon,
            barycenter: None,
            infinity_levels: Vec::new(),
            infinity_curvatures: Vec::new(),
            topos_override: None,
        }
    }

    /// Installs the homotopy generator and direction.
    pub fn deform(mut self, generator: Tensor, direction: Tensor) -> PureResult<Self> {
        if generator.shape() != self.seed.shape() {
            return Err(TensorError::ShapeMismatch {
                left: generator.shape(),
                right: self.seed.shape(),
            });
        }
        if direction.shape() != self.seed.shape() {
            return Err(TensorError::ShapeMismatch {
                left: direction.shape(),
                right: self.seed.shape(),
            });
        }
        self.generator = Some(generator);
        self.direction = Some(direction);
        Ok(self)
    }

    /// Overrides the topos used for resonance.
    pub fn across(mut self, topos: OpenCartesianTopos) -> PureResult<Self> {
        self.topos_override = Some(topos);
        Ok(self)
    }

    /// Configures the functor kernel; the seed is used as the default source.
    pub fn via(mut self, kernel: Tensor) -> PureResult<Self> {
        let seed_cols = self.seed.shape().1;
        if kernel.shape().0 != seed_cols {
            return Err(TensorError::ShapeMismatch {
                left: kernel.shape(),
                right: (seed_cols, kernel.shape().1),
            });
        }
        self.kernel = Some(kernel);
        Ok(self)
    }

    /// Configures both the kernel and an explicit source for the functor lift.
    pub fn via_with(mut self, kernel: Tensor, source: Tensor) -> PureResult<Self> {
        if source.shape().0 != self.seed.shape().0 || source.shape().1 != self.seed.shape().1 {
            return Err(TensorError::ShapeMismatch {
                left: source.shape(),
                right: self.seed.shape(),
            });
        }
        self = self.via(kernel)?;
        self.functor_source = Some(source);
        Ok(self)
    }

    /// Adjusts the finite-difference step used by the functor differential.
    pub fn functor_step(mut self, epsilon: f32) -> PureResult<Self> {
        if epsilon <= 0.0 || !epsilon.is_finite() {
            return Err(TensorError::NonPositiveLearningRate { rate: epsilon });
        }
        self.epsilon = epsilon;
        Ok(self)
    }

    /// Installs the barycenter trace that seeds the recursive differential.
    pub fn with_barycenter(mut self, barycenter: &ZSpaceBarycenter) -> PureResult<Self> {
        self.barycenter = Some(barycenter.clone());
        Ok(self)
    }

    /// Computes a barycenter using the session defaults and installs it.
    pub fn with_barycenter_from(
        mut self,
        weights: &[f32],
        densities: &[Tensor],
    ) -> PureResult<Self> {
        let barycenter = self.session.barycenter(weights, densities)?;
        self.barycenter = Some(barycenter);
        Ok(self)
    }

    /// Computes a barycenter with a custom coupling matrix and installs it.
    pub fn with_barycenter_with(
        mut self,
        weights: &[f32],
        densities: &[Tensor],
        coupling: Option<&Tensor>,
    ) -> PureResult<Self> {
        let barycenter = self.session.barycenter_with(weights, densities, coupling)?;
        self.barycenter = Some(barycenter);
        Ok(self)
    }

    /// Installs an explicit \(\infty\)-tower. Leaving `curvatures` empty reuses the session curvature.
    pub fn with_infinity(mut self, levels: Vec<Tensor>, curvatures: Vec<f32>) -> PureResult<Self> {
        if levels.is_empty() {
            return Err(TensorError::EmptyInput("infinity_levels"));
        }
        let seed_shape = self.seed.shape();
        for level in &levels {
            if level.shape() != seed_shape {
                return Err(TensorError::ShapeMismatch {
                    left: level.shape(),
                    right: seed_shape,
                });
            }
        }
        if !curvatures.is_empty() && curvatures.len() != levels.len() {
            return Err(TensorError::DataLength {
                expected: levels.len(),
                got: curvatures.len(),
            });
        }
        if let Some((_, curvature)) = curvatures
            .iter()
            .enumerate()
            .find(|(_, curvature)| !curvature.is_finite())
        {
            return Err(TensorError::NonFiniteValue {
                label: "infinity_curvature",
                value: *curvature,
            });
        }
        self.infinity_levels = levels;
        self.infinity_curvatures = curvatures;
        Ok(self)
    }

    fn build(self) -> PureResult<SpiralDifferential> {
        let session = self.session;
        let seed = self.seed;
        let generator = self
            .generator
            .ok_or_else(|| TensorError::EmptyInput("differential_generator"))?;
        let direction = self
            .direction
            .ok_or_else(|| TensorError::EmptyInput("differential_direction"))?;
        let kernel = self
            .kernel
            .ok_or_else(|| TensorError::EmptyInput("differential_kernel"))?;
        let barycenter = self
            .barycenter
            .ok_or_else(|| TensorError::EmptyInput("differential_barycenter"))?;
        let source = self.functor_source.unwrap_or_else(|| seed.clone());
        let levels = if self.infinity_levels.is_empty() {
            vec![barycenter.density.clone()]
        } else {
            self.infinity_levels
        };
        let curvatures = if self.infinity_curvatures.is_empty() {
            vec![session.curvature(); levels.len()]
        } else {
            if self.infinity_curvatures.len() != levels.len() {
                return Err(TensorError::DataLength {
                    expected: levels.len(),
                    got: self.infinity_curvatures.len(),
                });
            }
            self.infinity_curvatures
        };
        if curvatures.len() != levels.len() {
            return Err(TensorError::DataLength {
                expected: levels.len(),
                got: curvatures.len(),
            });
        }
        let homotopy = HomotopyDifferential::new(seed.clone(), generator, direction)?;
        let functor = FunctorDifferential::new(source, kernel, self.epsilon)?;
        let recursive = RecursiveDifferential::from_barycenter(&barycenter);
        let infinity = InfinityDifferential::new(levels, curvatures)?;
        let topos = self.topos_override.or_else(|| session.topos().cloned());
        Ok(SpiralDifferential::new(
            topos, homotopy, functor, recursive, infinity,
        ))
    }

    /// Finalises the trace without modifying a hypergrad tape.
    pub fn resonate(self) -> PureResult<DifferentialResonance> {
        let spiral = self.build()?;
        spiral.resonate(None)
    }

    /// Finalises the trace and aligns the provided hypergrad tape.
    pub fn resonate_with_hypergrad(
        self,
        hypergrad: &mut AmegaHypergrad,
    ) -> PureResult<DifferentialResonance> {
        let spiral = self.build()?;
        spiral.resonate(Some(hypergrad))
    }
}

impl SpiralSession {
    /// Returns a builder preloaded with heuristic defaults.
    pub fn builder(caps: DeviceCaps) -> SpiralSessionBuilder {
        SpiralSessionBuilder::new(caps)
    }

    /// Starts a differential trace anchored on the provided seed tensor.
    pub fn trace(&self, seed: Tensor) -> PureResult<DifferentialTrace> {
        if let Some(topos) = &self.topos {
            topos.guard_tensor("differential_seed", &seed)?;
        }
        Ok(DifferentialTrace::new(self.clone(), seed))
    }

    /// Reconstructs a builder seeded from this session.
    pub fn to_builder(&self) -> SpiralSessionBuilder {
        let mut builder = SpiralSessionBuilder::new(self.caps);
        builder.set_curvature(self.curvature);
        builder.set_hyper_learning_rate(self.hyper_learning_rate);
        builder.set_fallback_learning_rate(self.fallback_learning_rate);
        builder.set_barycenter_entropy(self.barycenter.entropy_weight);
        builder.set_barycenter_beta_j(self.barycenter.beta_j);
        builder.set_barycenter_coupling(self.barycenter.coupling.clone());
        builder.set_topos(self.topos.clone());
        builder.set_chrono_capacity(self.chrono_capacity());
        builder.set_maintainer_config(self.maintainer.clone());
        builder
    }

    /// Returns the backend capabilities baked into the session.
    pub fn device_caps(&self) -> DeviceCaps {
        self.caps
    }

    /// Returns the configured capacity of the temporal resonance timeline.
    pub fn chrono_capacity(&self) -> usize {
        self.chrono
            .lock()
            .map(|timeline| timeline.capacity())
            .unwrap_or(1)
    }

    /// Returns a cloned handle to the underlying chrono timeline.
    pub fn chrono_handle(&self) -> Arc<Mutex<ChronoTimeline>> {
        self.chrono.clone()
    }

    /// Returns the current maintainer configuration.
    pub fn maintainer_config(&self) -> MaintainerConfig {
        self.maintainer.clone()
    }

    /// Updates the maintainer configuration in-place.
    pub fn set_maintainer_config(&mut self, config: MaintainerConfig) {
        self.maintainer = config.sanitise();
    }

    /// Clones all recorded chrono frames in chronological order.
    pub fn chrono_frames(&self) -> Vec<ChronoFrame> {
        self.chrono
            .lock()
            .map(|timeline| timeline.frames().cloned().collect())
            .unwrap_or_default()
    }

    /// Returns a rolling summary of the temporal resonance history.
    pub fn timeline_summary(&self, window: Option<usize>) -> Option<ChronoSummary> {
        let requested = window.unwrap_or(usize::MAX);
        self.chrono.lock().ok().and_then(|timeline| {
            let available = timeline.len();
            if available == 0 {
                None
            } else {
                let window = requested.min(available).max(1);
                timeline.summarise(window)
            }
        })
    }

    /// Returns the latest atlas frame aggregated across telemetry subsystems.
    pub fn atlas(&self) -> Option<AtlasFrame> {
        hub::get_atlas_frame()
    }

    /// Returns the recent atlas route up to an optional limit.
    pub fn atlas_route(&self, limit: Option<usize>) -> AtlasRoute {
        hub::get_atlas_route(limit)
    }

    /// Generates a narrative describing the latest atlas frame when available.
    pub fn atlas_narrative(&self, temperature: f32) -> PureResult<Option<ResonanceNarrative>> {
        if let Some(frame) = self.atlas() {
            let temp = temperature.max(f32::EPSILON);
            let resonator = TextResonator::new(self.curvature, temp)?;
            Ok(Some(resonator.describe_atlas(&frame)))
        } else {
            Ok(None)
        }
    }

    /// Returns harmonic statistics for the recent temporal history.
    pub fn timeline_harmonics(
        &self,
        window: Option<usize>,
        bins: usize,
    ) -> Option<ChronoHarmonics> {
        let requested = window.unwrap_or(usize::MAX);
        let bins = bins.max(1);
        self.chrono.lock().ok().and_then(|timeline| {
            let available = timeline.len();
            if available < 2 {
                None
            } else {
                let window = requested.min(available).max(2);
                timeline.harmonics(window, bins)
            }
        })
    }

    /// Returns a combined summary and SpiralK-ready loop signal.
    pub fn loop_signal(
        &self,
        window: Option<usize>,
        bins: Option<usize>,
    ) -> Option<ChronoLoopSignal> {
        let requested = window.unwrap_or(self.maintainer.window);
        let bins = bins.unwrap_or(16).max(1);
        self.chrono.lock().ok().and_then(|timeline| {
            if timeline.is_empty() {
                None
            } else {
                let available = timeline.len();
                let window = requested.max(1).min(available);
                let harmonic_bins = bins.min(window.max(2));
                timeline.loop_signal(window, harmonic_bins)
            }
        })
    }

    /// Clears the temporal telemetry stream.
    pub fn reset_chrono(&self) {
        if let Ok(mut timeline) = self.chrono.lock() {
            timeline.reset();
        }
    }

    /// Curvature used for all hypergrad tapes.
    pub fn curvature(&self) -> f32 {
        self.curvature
    }

    /// Hypergradient learning rate.
    pub fn hyper_learning_rate(&self) -> f32 {
        self.hyper_learning_rate
    }

    /// Records the provided resonance snapshot into the temporal timeline and returns the frame.
    pub fn resonate_over_time(
        &self,
        resonance: &DifferentialResonance,
        dt: f32,
    ) -> PureResult<ChronoFrame> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(TensorError::NonPositiveLearningRate { rate: dt });
        }
        let homotopy_energy = resonance.homotopy_flow.squared_l2_norm();
        let functor_energy = resonance.functor_linearisation.squared_l2_norm();
        let recursive_energy = resonance.recursive_objective.squared_l2_norm();
        let projection_energy = resonance.infinity_projection.squared_l2_norm();
        let infinity_energy = resonance.infinity_energy.squared_l2_norm();
        let total_energy = homotopy_energy
            + functor_energy
            + recursive_energy
            + projection_energy
            + infinity_energy;
        let observed_curvature = if homotopy_energy > 0.0 {
            -homotopy_energy.sqrt()
        } else {
            self.curvature
        };
        let metrics = ResonanceTemporalMetrics {
            observed_curvature,
            total_energy,
            homotopy_energy,
            functor_energy,
            recursive_energy,
            projection_energy,
            infinity_energy,
        }
        .sanitise();
        let mut timeline = self.chrono.lock().map_err(|_| TensorError::IoError {
            message: "chrono timeline poisoned".to_string(),
        })?;
        let frame = timeline.record(dt, metrics);
        let bin_hint = self.maintainer.window.max(4).min(64);
        let signal = timeline.loop_signal(self.maintainer.window, bin_hint);
        drop(timeline);
        let atlas_signal = signal.clone();
        if let Some(ref signal) = signal {
            hub::set_chrono_loop(signal.clone());
        }
        if let Some(signal) = atlas_signal {
            let summary = signal.summary.clone();
            let harmonics = signal.harmonics.clone();
            let maintainer = self.maintain();
            let mut fragment = AtlasFragment::new();
            fragment.timestamp = Some(summary.latest_timestamp);
            fragment.summary = Some(summary.clone());
            fragment.harmonics = harmonics;
            fragment.loop_support = Some(1.0);
            fragment.push_metric("timeline.frames", summary.frames as f32);
            fragment.push_metric("timeline.energy", summary.mean_energy);
            fragment.push_metric("timeline.drift", summary.mean_drift);
            fragment.push_metric("timeline.decay", summary.mean_decay);
            fragment.maintainer_status = Some(maintainer.status);
            fragment.maintainer_diagnostic = Some(maintainer.diagnostic.clone());
            fragment.suggested_max_scale = maintainer.suggested_max_scale;
            fragment.suggested_pressure = maintainer.suggested_pressure;
            fragment.push_metric("maintainer.average_drift", maintainer.average_drift);
            fragment.push_metric("maintainer.mean_energy", maintainer.mean_energy);
            fragment.push_metric("maintainer.mean_decay", maintainer.mean_decay);
            fragment.push_note(format!("maintainer:{}", maintainer.status.as_str()));
            #[cfg(feature = "kdsl")]
            {
                if let Some(script) = signal.spiralk_script.clone() {
                    fragment.script_hint = Some(script);
                }
                if fragment.script_hint.is_none() {
                    fragment.script_hint = maintainer.spiralk_script.clone();
                } else if let Some(extra) = maintainer.spiralk_script.clone() {
                    fragment.push_note(format!("maintainer.script:{extra}"));
                }
            }
            hub::merge_atlas_fragment(fragment);
        }
        if let Some(signal) = signal {
            hub::set_chrono_loop(signal);
        }
        Ok(frame)
    }

    /// Evaluates the temporal telemetry and returns a maintenance recommendation.
    pub fn maintain(&self) -> MaintainerReport {
        let frames = self.chrono_frames();
        let maintainer = Maintainer::new(self.maintainer.clone());
        maintainer.assess(&frames)
    }

    /// Generates a natural language narrative describing the resonance.
    pub fn narrative(
        &self,
        resonance: Option<&DifferentialResonance>,
        temperature: f32,
    ) -> PureResult<ResonanceNarrative> {
        let temp = temperature.max(f32::EPSILON);
        let resonator = TextResonator::new(self.curvature, temp)?;
        if let Some(resonance) = resonance {
            Ok(resonator.describe_resonance(resonance))
        } else {
            self.timeline_narrative(None, temperature)
        }
    }

    /// Returns a short textual description of the resonance snapshot or latest frame.
    pub fn describe(
        &self,
        resonance: Option<&DifferentialResonance>,
        temperature: f32,
    ) -> PureResult<String> {
        self.narrative(resonance, temperature)
            .map(|narrative| narrative.summary)
    }

    /// Generates a narrative describing the timeline, optionally restricting to the latest frames.
    pub fn timeline_narrative(
        &self,
        window: Option<usize>,
        temperature: f32,
    ) -> PureResult<ResonanceNarrative> {
        let frames = self.chrono_frames();
        if frames.is_empty() {
            return Ok(ResonanceNarrative {
                summary: "No resonance history recorded.".to_string(),
                highlights: Vec::new(),
            });
        }
        let limit = window.unwrap_or(frames.len()).min(frames.len());
        let start = frames.len().saturating_sub(limit);
        let temp = temperature.max(f32::EPSILON);
        let resonator = TextResonator::new(self.curvature, temp)?;
        Ok(resonator.describe_timeline(&frames[start..]))
    }

    /// Encodes the recent temporal trace into an amplitude envelope.
    pub fn speak(&self, timesteps: Option<usize>, temperature: f32) -> PureResult<Vec<f32>> {
        let frames = self.chrono_frames();
        if frames.is_empty() {
            return Ok(Vec::new());
        }
        let limit = timesteps.unwrap_or(frames.len());
        let start = frames.len().saturating_sub(limit);
        let temp = temperature.max(f32::EPSILON);
        let resonator = TextResonator::new(self.curvature, temp)?;
        resonator.speak(&frames[start..])
    }

    /// Euclidean fallback learning rate.
    pub fn fallback_learning_rate(&self) -> f32 {
        self.fallback_learning_rate
    }

    /// Entropy regulariser weight.
    pub fn barycenter_entropy_weight(&self) -> f32 {
        self.barycenter.entropy_weight
    }

    /// Coupling scale \(\beta_J\).
    pub fn barycenter_beta_j(&self) -> f32 {
        self.barycenter.beta_j
    }

    /// Returns the configured coupling matrix, if any.
    pub fn barycenter_coupling(&self) -> Option<&Tensor> {
        self.barycenter.coupling()
    }

    /// Accessor for the topos guard.
    pub fn topos(&self) -> Option<&OpenCartesianTopos> {
        self.topos.as_ref()
    }

    /// Creates a hypergrad tape aligned with the session curvature and guard.
    pub fn hypergrad(&self, rows: usize, cols: usize) -> PureResult<AmegaHypergrad> {
        if let Some(topos) = &self.topos {
            AmegaHypergrad::with_topos(
                self.curvature,
                self.hyper_learning_rate,
                rows,
                cols,
                topos.clone(),
            )
        } else {
            AmegaHypergrad::new(self.curvature, self.hyper_learning_rate, rows, cols)
        }
    }

    /// Builds a module trainer preloaded with the session learning rates.
    pub fn trainer(&self) -> ModuleTrainer {
        ModuleTrainer::new(
            self.caps,
            self.curvature,
            self.hyper_learning_rate,
            self.fallback_learning_rate,
        )
    }

    /// Attaches hypergrad buffers to the provided module.
    pub fn prepare_module<M: Module>(&self, module: &mut M) -> PureResult<()> {
        if let Some(topos) = &self.topos {
            module.attach_hypergrad_with_topos(
                self.curvature,
                self.hyper_learning_rate,
                topos.clone(),
            )
        } else {
            module.attach_hypergrad(self.curvature, self.hyper_learning_rate)
        }
    }

    /// Computes the rank plan for the requested selection.
    pub fn plan_rank(&self, kind: RankKind, rows: u32, cols: u32, k: u32) -> RankPlan {
        plan_rank(kind, rows, cols, k, self.caps)
    }

    /// Produces a barycenter using the session defaults.
    pub fn barycenter(
        &self,
        weights: &[f32],
        densities: &[Tensor],
    ) -> PureResult<ZSpaceBarycenter> {
        self.barycenter_with(weights, densities, self.barycenter.coupling())
    }

    /// Computes a barycenter while overriding the coupling matrix.
    pub fn barycenter_with(
        &self,
        weights: &[f32],
        densities: &[Tensor],
        coupling: Option<&Tensor>,
    ) -> PureResult<ZSpaceBarycenter> {
        z_space_barycenter(
            weights,
            densities,
            self.barycenter.entropy_weight,
            self.barycenter.beta_j,
            coupling,
        )
    }

    /// Variant that accepts ad-hoc entropy and coupling parameters.
    pub fn barycenter_with_parameters(
        &self,
        weights: &[f32],
        densities: &[Tensor],
        entropy_weight: f32,
        beta_j: f32,
        coupling: Option<&Tensor>,
    ) -> PureResult<ZSpaceBarycenter> {
        z_space_barycenter(weights, densities, entropy_weight, beta_j, coupling)
    }

    /// Aligns a hypergrad tape with the loss-monotone barycenter interpolation.
    pub fn align_hypergrad(
        &self,
        hypergrad: &mut AmegaHypergrad,
        barycenter: &ZSpaceBarycenter,
    ) -> PureResult<()> {
        hypergrad.accumulate_barycenter_path(&barycenter.intermediates)
    }

    /// Splits a gradient tensor into roundtable bands using the session caps.
    pub fn split_bands(
        &self,
        grad_output: &Tensor,
        schedule: &RoundtableSchedule,
    ) -> PureResult<GradientBands> {
        schedule.split(grad_output)
    }

    /// Computes the per-band energy diagnostic for a gradient tensor.
    pub fn band_energy(
        &self,
        grad_output: &Tensor,
        schedule: &RoundtableSchedule,
    ) -> PureResult<BandEnergy> {
        schedule.band_energy(grad_output)
    }

    /// Runs a full epoch with the provided trainer, module, and loss surface.
    pub fn train_epoch<M, L, I>(
        &self,
        trainer: &mut ModuleTrainer,
        module: &mut M,
        loss: &mut L,
        batches: I,
        schedule: &RoundtableSchedule,
    ) -> PureResult<crate::trainer::EpochStats>
    where
        M: Module,
        L: Loss,
        I: IntoIterator<Item = (Tensor, Tensor)>,
    {
        trainer.train_epoch(module, loss, batches, schedule)
    }

    /// Runs a full epoch using a transient trainer built from the session configuration.
    pub fn train_epoch_with<M, L, I>(
        &self,
        module: &mut M,
        loss: &mut L,
        batches: I,
        schedule: &RoundtableSchedule,
    ) -> PureResult<EpochStats>
    where
        M: Module,
        L: Loss,
        I: IntoIterator<Item = (Tensor, Tensor)>,
    {
        let mut trainer = self.trainer();
        trainer.train_epoch(module, loss, batches, schedule)
    }

    /// Creates a roundtable schedule for the provided configuration.
    pub fn roundtable(&self, rows: u32, cols: u32, config: RoundtableConfig) -> RoundtableSchedule {
        self.trainer().roundtable(rows, cols, config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::linear::Linear;
    use crate::loss::MeanSquaredError;
    use st_core::telemetry::hub;
    use st_core::telemetry::maintainer::MaintainerStatus;
    use st_tensor::pure::measure::BarycenterIntermediate;

    fn toy_tensor(a: &[f32]) -> Tensor {
        Tensor::from_vec(1, a.len(), a.to_vec()).unwrap()
    }

    #[test]
    fn builder_enforces_curvature_and_rates() {
        let mut builder = SpiralSession::builder(DeviceCaps::wgpu(32, true, 256));
        builder.set_curvature(-0.5);
        builder.set_hyper_learning_rate(0.2);
        builder.set_fallback_learning_rate(0.05);
        builder.set_barycenter_entropy(0.0);
        let session = builder.build().unwrap();
        assert_eq!(session.curvature(), -0.5);
        assert_eq!(session.hyper_learning_rate(), 0.2);
        assert_eq!(session.fallback_learning_rate(), 0.05);
        assert_eq!(session.barycenter_entropy_weight(), 0.0);
    }

    #[test]
    fn session_produces_barycenter_and_hypergrad() {
        let session = SpiralSession::builder(DeviceCaps::wgpu(32, true, 256))
            .with_curvature(-1.0)
            .with_hyper_learning_rate(0.1)
            .with_barycenter_entropy(0.05)
            .build()
            .unwrap();
        let densities = vec![toy_tensor(&[0.7, 0.3]), toy_tensor(&[0.4, 0.6])];
        let weights = vec![0.5, 0.5];
        let bary = session.barycenter(&weights, &densities).unwrap();
        assert_eq!(bary.density.shape(), (1, 2));
        assert!(bary.objective.is_finite());
        let mut hypergrad = session.hypergrad(1, 2).unwrap();
        session.align_hypergrad(&mut hypergrad, &bary).unwrap();
    }

    #[test]
    fn session_plans_ranks_and_trains_module() {
        let session = SpiralSession::builder(DeviceCaps::wgpu(32, true, 256))
            .with_curvature(-1.0)
            .with_hyper_learning_rate(0.05)
            .with_fallback_learning_rate(0.02)
            .build()
            .unwrap();
        let plan = session.plan_rank(RankKind::TopK, 128, 512, 32);
        assert_eq!(plan.rows, 128);
        assert_eq!(plan.cols, 512);
        let mut module = Linear::new("toy", 1, 1).unwrap();
        session.prepare_module(&mut module).unwrap();
        let mut trainer = session.trainer();
        let schedule = trainer.roundtable(1, 1, RoundtableConfig::default());
        let mut loss = MeanSquaredError::default();
        let batches = vec![(toy_tensor(&[1.0]), toy_tensor(&[0.5])); 4];
        let stats = session
            .train_epoch(&mut trainer, &mut module, &mut loss, batches, &schedule)
            .unwrap();
        assert_eq!(stats.batches, 4);
        let energy = session.band_energy(&toy_tensor(&[0.1]), &schedule).unwrap();
        assert!(energy.here >= 0.0);
        let bands = session.split_bands(&toy_tensor(&[0.1]), &schedule).unwrap();
        assert_eq!(bands.here().shape(), (1, 1));
    }

    #[test]
    fn invalid_rates_are_rejected() {
        let mut builder = SpiralSession::builder(DeviceCaps::wgpu(32, true, 256));
        builder.set_hyper_learning_rate(-0.1);
        assert!(builder.build().is_err());
    }

    #[test]
    fn session_differential_trace_resonates() {
        let session = SpiralSession::builder(DeviceCaps::wgpu(32, true, 256))
            .with_curvature(-1.0)
            .with_hyper_learning_rate(0.05)
            .with_topos(OpenCartesianTopos::new(-1.0, 1e-4, 10.0, 8, 16).unwrap())
            .build()
            .unwrap();
        let seed = toy_tensor(&[0.4, 0.6]);
        let generator = toy_tensor(&[0.1, -0.2]);
        let direction = toy_tensor(&[0.05, 0.07]);
        let kernel = Tensor::from_vec(2, 2, vec![1.0, 0.5, -0.25, 1.25]).unwrap();
        let density = toy_tensor(&[0.55, 0.45]);
        let stage = BarycenterIntermediate {
            interpolation: 0.0,
            density: density.clone(),
            kl_energy: 0.1,
            entropy: 0.2,
            objective: 0.3,
        };
        let barycenter = ZSpaceBarycenter {
            density: density.clone(),
            kl_energy: 0.1,
            entropy: 0.2,
            coupling_energy: 0.0,
            objective: 0.3,
            effective_weight: 1.0,
            intermediates: vec![stage.clone(), stage],
        };
        let trace = session
            .trace(seed.clone())
            .unwrap()
            .deform(generator.clone(), direction.clone())
            .unwrap()
            .via(kernel.clone())
            .unwrap()
            .with_barycenter(&barycenter)
            .unwrap()
            .with_infinity(vec![density.clone(), density.clone()], vec![1.0, 0.5])
            .unwrap()
            .functor_step(0.02)
            .unwrap();
        let resonance = trace.clone().resonate().unwrap();
        assert_eq!(resonance.homotopy_flow.shape(), (1, 2));
        let mut hypergrad = session.hypergrad(1, 2).unwrap();
        let resonance_with = trace.resonate_with_hypergrad(&mut hypergrad).unwrap();
        assert_eq!(
            resonance_with.recursive_objective.shape().1,
            barycenter.intermediates.len()
        );
    }

    #[test]
    fn trace_builds_barycenter_via_session_defaults() {
        let session = SpiralSession::builder(DeviceCaps::wgpu(32, true, 256))
            .with_curvature(-1.0)
            .with_hyper_learning_rate(0.05)
            .build()
            .unwrap();
        let seed = toy_tensor(&[0.6, 0.4]);
        let generator = toy_tensor(&[0.12, -0.08]);
        let direction = toy_tensor(&[0.04, 0.02]);
        let kernel = Tensor::from_vec(2, 2, vec![1.0, -0.25, 0.5, 1.25]).unwrap();
        let densities = vec![toy_tensor(&[0.7, 0.3]), toy_tensor(&[0.5, 0.5])];
        let weights = vec![0.6, 0.4];
        let trace = session
            .trace(seed.clone())
            .unwrap()
            .deform(generator.clone(), direction.clone())
            .unwrap()
            .via(kernel.clone())
            .unwrap()
            .with_barycenter_from(&weights, &densities)
            .unwrap()
            .with_infinity(vec![densities[0].clone()], Vec::new())
            .unwrap();
        let resonance = trace.resonate().unwrap();
        assert_eq!(resonance.homotopy_flow.shape(), seed.shape());
        assert_eq!(resonance.infinity_projection.shape().0, 1);
    }

    #[test]
    fn trace_rejects_mismatched_infinity_level() {
        let session = SpiralSession::builder(DeviceCaps::wgpu(32, true, 256))
            .with_curvature(-1.0)
            .with_hyper_learning_rate(0.05)
            .build()
            .unwrap();
        let seed = toy_tensor(&[0.4, 0.6]);
        let generator = toy_tensor(&[0.1, -0.2]);
        let direction = toy_tensor(&[0.05, 0.07]);
        let kernel = Tensor::from_vec(2, 2, vec![1.0, 0.5, -0.25, 1.25]).unwrap();
        let bad_level = Tensor::from_vec(1, 3, vec![0.1, 0.2, 0.3]).unwrap();
        let err = session
            .trace(seed.clone())
            .unwrap()
            .deform(generator.clone(), direction.clone())
            .unwrap()
            .via(kernel.clone())
            .unwrap()
            .with_infinity(vec![bad_level], Vec::new())
            .unwrap_err();
        match err {
            TensorError::ShapeMismatch { .. } => {}
            other => panic!("expected shape mismatch, got {other:?}"),
        }
    }

    #[test]
    fn chrono_timeline_records_frames() {
        let session = SpiralSession::builder(DeviceCaps::cpu())
            .with_curvature(-1.0)
            .with_chrono_capacity(2)
            .build()
            .unwrap();
        let seed = toy_tensor(&[0.5, -0.2]);
        let generator = toy_tensor(&[1.0, -1.0]);
        let direction = toy_tensor(&[0.15, 0.05]);
        let kernel = Tensor::from_vec(2, 2, vec![1.0, 0.2, -0.3, 1.1]).unwrap();
        let density = toy_tensor(&[0.6, 0.4]);
        let weights = vec![1.0];
        let barycenter = session.barycenter(&weights, &[density.clone()]).unwrap();
        let resonance = session
            .trace(seed.clone())
            .unwrap()
            .deform(generator.clone(), direction.clone())
            .unwrap()
            .via(kernel.clone())
            .unwrap()
            .with_barycenter(&barycenter)
            .unwrap()
            .resonate()
            .unwrap();
        let first = session.resonate_over_time(&resonance, 0.1).unwrap();
        assert_eq!(first.step, 0);
        assert!(first.timestamp > 0.0);
        let second = session.resonate_over_time(&resonance, 0.1).unwrap();
        assert_eq!(second.step, 1);
        let frames = session.chrono_frames();
        assert_eq!(frames.len(), 2);
        assert!(frames[1].timestamp > frames[0].timestamp);
        let summary = session.timeline_summary(Some(2)).unwrap();
        assert_eq!(summary.frames, 2);
        assert!(summary.mean_energy > 0.0);
    }

    #[test]
    fn maintainer_flags_energy_growth() {
        let session = SpiralSession::builder(DeviceCaps::cpu())
            .with_curvature(-1.0)
            .with_chrono_capacity(8)
            .build()
            .unwrap();
        {
            let handle = session.chrono_handle();
            let mut timeline = handle.lock().unwrap();
            let energies = [1.0, 1.3, 1.6, 1.9];
            for energy in energies {
                let metrics = ResonanceTemporalMetrics {
                    observed_curvature: -1.0,
                    total_energy: energy,
                    homotopy_energy: energy * 0.4,
                    functor_energy: energy * 0.2,
                    recursive_energy: energy * 0.2,
                    projection_energy: energy * 0.1,
                    infinity_energy: energy * 0.1,
                }
                .sanitise();
                timeline.record(0.1, metrics);
            }
        }
        let report = session.maintain();
        assert!(report.should_rewrite());
        assert_eq!(report.status, MaintainerStatus::Rewrite);
        assert!(report.diagnostic.contains("energy growth"));
    }

    #[test]
    fn describe_returns_default_when_no_frames() {
        let session = SpiralSession::builder(DeviceCaps::cpu())
            .with_curvature(-1.0)
            .build()
            .unwrap();
        let summary = session.describe(None, 0.6).unwrap();
        assert!(summary.contains("No resonance"));
    }

    #[test]
    fn speak_generates_waveform() {
        let session = SpiralSession::builder(DeviceCaps::cpu())
            .with_curvature(-1.0)
            .with_chrono_capacity(8)
            .build()
            .unwrap();
        let resonance = DifferentialResonance {
            homotopy_flow: toy_tensor(&[0.2, -0.1]),
            functor_linearisation: toy_tensor(&[0.1, 0.05]),
            recursive_objective: toy_tensor(&[0.05, 0.02]),
            infinity_projection: toy_tensor(&[0.03, 0.01]),
            infinity_energy: toy_tensor(&[0.02, 0.01]),
        };
        session.resonate_over_time(&resonance, 0.1).unwrap();
        session.resonate_over_time(&resonance, 0.1).unwrap();
        let waveform = session.speak(Some(2), 0.6).unwrap();
        assert!(!waveform.is_empty());
        let narrative = session.describe(None, 0.6).unwrap();
        assert!(!narrative.is_empty());
    }

    #[test]
    fn loop_signal_publishes_and_reads() {
        let session = SpiralSession::builder(DeviceCaps::cpu())
            .with_curvature(-1.0)
            .with_chrono_capacity(8)
            .build()
            .unwrap();
        let resonance = DifferentialResonance {
            homotopy_flow: toy_tensor(&[0.2, -0.1]),
            functor_linearisation: toy_tensor(&[0.1, 0.05]),
            recursive_objective: toy_tensor(&[0.05, 0.02]),
            infinity_projection: toy_tensor(&[0.03, 0.01]),
            infinity_energy: toy_tensor(&[0.02, 0.01]),
        };
        session.resonate_over_time(&resonance, 0.1).unwrap();
        session.resonate_over_time(&resonance, 0.1).unwrap();
        let signal = session.loop_signal(None, None).unwrap();
        assert!(signal.summary.frames >= 2);
        let hub_signal = hub::get_chrono_loop();
        assert!(hub_signal.is_some());
    }

    #[test]
    fn atlas_collects_telemetry_fragments() {
        hub::clear_atlas();
        let session = SpiralSession::builder(DeviceCaps::cpu())
            .with_curvature(-1.0)
            .with_chrono_capacity(8)
            .build()
            .unwrap();
        let resonance = DifferentialResonance {
            homotopy_flow: toy_tensor(&[0.2, -0.1]),
            functor_linearisation: toy_tensor(&[0.1, 0.05]),
            recursive_objective: toy_tensor(&[0.05, 0.02]),
            infinity_projection: toy_tensor(&[0.03, 0.01]),
            infinity_energy: toy_tensor(&[0.02, 0.01]),
        };
        session.resonate_over_time(&resonance, 0.1).unwrap();
        session.resonate_over_time(&resonance, 0.1).unwrap();
        let atlas = session.atlas().expect("atlas");
        assert!(atlas.timestamp > 0.0);
        assert!(atlas.loop_support >= 1.0);
        assert!(atlas
            .metrics
            .iter()
            .any(|metric| metric.name == "timeline.energy"));
        let districts = atlas.districts();
        assert!(!districts.is_empty());
        assert!(districts.iter().any(|d| d.name == "Surface"));
        let route = session.atlas_route(Some(4));
        assert!(!route.is_empty());
        assert!(route.latest().is_some());
        let story = session.atlas_narrative(0.6).unwrap();
        assert!(story
            .as_ref()
            .map(|narrative| narrative.summary.contains("Atlas"))
            .unwrap_or(false));
    }

    #[test]
    fn timeline_harmonics_reports_frequency() {
        let session = SpiralSession::builder(DeviceCaps::cpu())
            .with_curvature(-1.0)
            .with_chrono_capacity(64)
            .build()
            .unwrap();
        for step in 0..48 {
            let energy = (step as f32 / 6.0).sin() + 1.2;
            let resonance = DifferentialResonance {
                homotopy_flow: toy_tensor(&[energy, -energy]),
                functor_linearisation: toy_tensor(&[0.1, 0.05]),
                recursive_objective: toy_tensor(&[0.05, 0.02]),
                infinity_projection: toy_tensor(&[0.03, 0.01]),
                infinity_energy: toy_tensor(&[0.02, 0.01]),
            };
            session.resonate_over_time(&resonance, 0.1).unwrap();
        }
        let harmonics = session.timeline_harmonics(Some(48), 16).unwrap();
        assert!(harmonics.dominant_energy.is_some());
        let story = session.timeline_narrative(Some(48), 0.6).unwrap();
        assert!(story.summary.contains("Timeline span"));
    }
}
