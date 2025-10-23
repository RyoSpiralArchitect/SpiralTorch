// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! Z-space native coherence-based sequence modeling.
//!
//! Unlike attention (Q·K^T softmax), ZSpaceCoherenceSequencer uses:
//! - Maxwell pulses for phase synchronization
//! - Desire Lagrangian for linguistic bias
//! - Hyperbolic geometry for hierarchical relationships
//! - Fractional calculus for spectral operators

use super::coherence_engine::{
    CoherenceBackend, CoherenceEngine, DomainConcept, DomainLinguisticProfile,
    LinguisticChannelReport, LinguisticContour,
};
use crate::{
    language::{ConceptHint, MaxwellDesireBridge, NarrativeHint, SemanticBridge},
    Module, PureResult, Tensor,
};
use st_core::maxwell::MaxwellZPulse;
#[cfg(feature = "psi")]
use st_core::{
    telemetry::{hub::SoftlogicZFeedback, psi::{PsiEvent, PsiReading}},
    theory::maxwell::MaxwellPsiTelemetryBridge,
};
use st_tensor::{OpenCartesianTopos, TensorError};
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, MutexGuard};

/// Identifies a stage in the Z-space sequencing pipeline for plugin callbacks.
#[derive(Debug, Clone)]
pub enum ZSpaceSequencerStage<'a> {
    /// Fired after the Euclidean input has been projected into Z-space.
    Projected {
        /// Original Euclidean input tensor.
        input: &'a Tensor,
        /// Resulting projection in Z-space.
        projected: &'a Tensor,
    },
    /// Fired after Maxwell coherence weights have been measured.
    CoherenceMeasured {
        /// Tensor residing in Z-space used for coherence measurement.
        z_space: &'a Tensor,
        /// Maxwell coherence weights per channel.
        coherence: &'a [f32],
    },
    /// Fired after geometric aggregation has produced the final tensor and diagnostics.
    Aggregated {
        /// Aggregated tensor returned by the sequencer.
        aggregated: &'a Tensor,
        /// Coherence weights used during aggregation.
        coherence: &'a [f32],
        /// Rich diagnostics captured during aggregation.
        diagnostics: &'a CoherenceDiagnostics,
    },
    /// Fired after a pre-discard policy has filtered coherence weights.
    PreDiscardApplied {
        /// Coherence weights before pre-discard was applied.
        original: &'a [f32],
        /// Coherence weights after pre-discard has been applied.
        filtered: &'a [f32],
        /// Telemetry describing the pre-discard outcome.
        telemetry: &'a PreDiscardTelemetry,
        /// Indices of channels that survived the pre-discard pass.
        survivors: &'a [usize],
        /// Indices of channels that were discarded by the pass.
        discarded: &'a [usize],
    },
    /// Fired after language bridges have fused semantics and narratives.
    LanguageBridged {
        /// Aggregated tensor emitted by the sequencer.
        aggregated: &'a Tensor,
        /// Maxwell coherence weights associated with the tensor.
        coherence: &'a [f32],
        /// Conceptual hint surfaced for downstream language models.
        concept: &'a ConceptHint,
        /// Narrative hint emitted by Maxwell desire bridges, when available.
        narrative: Option<&'a NarrativeHint>,
        /// Summarised Maxwell pulse used for PSI telemetry and feedback.
        pulse: &'a MaxwellZPulse,
    },
    /// Fired when the coherence backend has been reconfigured.
    BackendConfigured {
        /// Backend that will be used for subsequent coherence measurements.
        backend: CoherenceBackend,
    },
    /// Fired after a linguistic profile has been registered.
    LinguisticProfileRegistered {
        /// Profile that will be used to bias coherence weighting.
        profile: &'a DomainLinguisticProfile,
    },
    /// Fired when all linguistic profiles have been cleared.
    LinguisticProfilesCleared,
    /// Fired after a semantic window has been derived for language fusion.
    SemanticWindowDerived {
        /// Sliding window mapping token indices to weighted energy.
        window: &'a [(usize, f32)],
        /// Number of tokens the window was derived against.
        tokens: usize,
    },
    /// Fired after a semantic distribution has been derived from the window.
    SemanticDistributionDerived {
        /// Window used for inference.
        window: &'a [(usize, f32)],
        /// Semantic distribution normalised for downstream bridges.
        distribution: &'a [f32],
    },
    /// Fired after the canonical domain concept has been selected for bridging.
    CanonicalConceptSelected {
        /// Domain concept inferred from curvature and head count.
        concept: DomainConcept,
        /// Channel label exposed to the Maxwell desire bridge.
        channel: &'a str,
    },
    /// Fired after the Maxwell desire bridge has emitted a concept hint.
    MaxwellBridgeEmitted {
        /// Channel label supplied to the bridge.
        channel: &'a str,
        /// Summarised Maxwell pulse used to derive the hint.
        pulse: &'a MaxwellZPulse,
        /// Raw concept hint emitted prior to semantic fusion.
        hint: &'a ConceptHint,
        /// Narrative hint surfaced by the bridge, when available.
        narrative: Option<&'a NarrativeHint>,
    },
    /// Fired after a linguistic contour has been emitted for downstream stacks.
    LinguisticContourEmitted {
        /// Coherence weights that produced the contour.
        coherence: &'a [f32],
        /// Contour descriptor returned by the coherence engine.
        contour: &'a LinguisticContour,
    },
    /// Fired after coherence channels have been described.
    ChannelsDescribed {
        /// Coherence weights associated with the reports.
        coherence: &'a [f32],
        /// Reports emitted by the coherence engine.
        reports: &'a [LinguisticChannelReport],
    },
    /// Fired after a semantic window has been fused into a distribution hint.
    SemanticWindowFused {
        /// Resulting fused concept hint produced by bridges.
        concept: &'a ConceptHint,
    },
    /// Fired after PSI telemetry has been published (when enabled).
    #[cfg(feature = "psi")]
    PsiTelemetryPublished {
        /// Pulse used to publish the telemetry.
        pulse: &'a MaxwellZPulse,
        /// Reading captured by the PSI bridge, when available.
        reading: Option<&'a PsiReading>,
        /// PSI events recorded during publication.
        events: &'a [PsiEvent],
        /// Feedback emitted by the SoftLogic Z bridge.
        feedback: &'a SoftlogicZFeedback,
    },
}

/// Trait implemented by plugins that wish to observe or augment the sequencing pipeline.
pub trait ZSpaceSequencerPlugin: Send + Sync {
    /// Returns a descriptive name for the plugin.
    fn name(&self) -> &'static str;

    /// Receives callbacks for each stage of the sequencing pipeline.
    fn on_stage(&self, stage: ZSpaceSequencerStage<'_>) -> PureResult<()>;
}

/// Rich coherence diagnostics surfaced by [`ZSpaceCoherenceSequencer`].
#[derive(Clone, Debug)]
pub struct CoherenceDiagnostics {
    channel_weights: Vec<f32>,
    normalized_weights: Vec<f32>,
    normalization: f32,
    fractional_order: f32,
    dominant_channel: Option<usize>,
    mean_coherence: f32,
    z_bias: f32,
    energy_ratio: f32,
    entropy: f32,
    aggregated: Tensor,
    coherence: Vec<f32>,
    channel_reports: Vec<LinguisticChannelReport>,
    pre_discard: Option<PreDiscardTelemetry>,
}

impl CoherenceDiagnostics {
    /// Returns the raw Maxwell coherence weights.
    pub fn channel_weights(&self) -> &[f32] {
        &self.channel_weights
    }

    /// Returns the coherence weights normalised to a probability simplex.
    pub fn normalized_weights(&self) -> &[f32] {
        &self.normalized_weights
    }

    /// Returns the sum used to normalise coherence weights.
    pub fn normalization(&self) -> f32 {
        self.normalization
    }

    /// Fractional order used during bidirectional smoothing.
    pub fn fractional_order(&self) -> f32 {
        self.fractional_order
    }

    /// Index of the dominant coherence channel, if any.
    pub fn dominant_channel(&self) -> Option<usize> {
        self.dominant_channel
    }

    /// Mean coherence value observed across channels.
    pub fn mean_coherence(&self) -> f32 {
        self.mean_coherence
    }

    /// Signed Z-bias derived from the summarised Maxwell pulse.
    pub fn z_bias(&self) -> f32 {
        self.z_bias
    }

    /// Ratio of energy concentrated in the dominant channel band.
    pub fn energy_ratio(&self) -> f32 {
        self.energy_ratio
    }

    /// Shannon entropy of the coherence distribution.
    pub fn coherence_entropy(&self) -> f32 {
        self.entropy
    }

    /// Returns the aggregated tensor from the coherence pipeline.
    pub fn aggregated(&self) -> &Tensor {
        &self.aggregated
    }

    /// Returns the raw coherence weights captured during the forward pass.
    pub fn coherence(&self) -> &[f32] {
        &self.coherence
    }

    /// Returns the rich linguistic channel reports, if any were computed.
    pub fn channel_reports(&self) -> &[LinguisticChannelReport] {
        &self.channel_reports
    }

    /// Overrides the linguistic channel reports while preserving existing diagnostics.
    pub fn with_channel_reports(mut self, channel_reports: Vec<LinguisticChannelReport>) -> Self {
        self.channel_reports = channel_reports;
        self
    }

    /// Returns telemetry describing any pre-discard that was applied.
    pub fn pre_discard(&self) -> Option<&PreDiscardTelemetry> {
        self.pre_discard.as_ref()
    }

    /// Returns the number of channels that survived pre-discard.
    pub fn preserved_channels(&self) -> usize {
        self.coherence.iter().filter(|value| **value > 0.0).count()
    }

    /// Returns the number of channels removed by pre-discard when available.
    pub fn discarded_channels(&self) -> usize {
        self.pre_discard
            .as_ref()
            .map(|telemetry| telemetry.discarded())
            .unwrap_or(0)
    }

    /// Destructures the diagnostics into their core components.
    pub fn into_parts(
        self,
    ) -> (
        Tensor,
        Vec<f32>,
        Vec<LinguisticChannelReport>,
        Option<PreDiscardTelemetry>,
    ) {
        (
            self.aggregated,
            self.coherence,
            self.channel_reports,
            self.pre_discard,
        )
    }
}

/// Configuration describing how coherence weights are culled before aggregation.
#[derive(Clone, Debug)]
pub struct PreDiscardPolicy {
    dominance_ratio: f32,
    energy_floor: f32,
    min_channels: usize,
}

impl PreDiscardPolicy {
    /// Creates a new pre-discard policy. Channels that fall more than
    /// `dominance_ratio` below the dominant weight are discarded unless they
    /// clear the absolute `energy_floor` (default 0).
    pub fn new(dominance_ratio: f32) -> PureResult<Self> {
        if !dominance_ratio.is_finite() || dominance_ratio < 0.0 {
            return Err(TensorError::NonPositiveCoherence {
                coherence: dominance_ratio,
            }
            .into());
        }
        Ok(Self {
            dominance_ratio: dominance_ratio.min(1.0),
            energy_floor: 0.0,
            min_channels: 1,
        })
    }

    /// Sets an absolute floor that channels must reach to survive pre-discard.
    pub fn with_energy_floor(mut self, energy_floor: f32) -> PureResult<Self> {
        if !energy_floor.is_finite() || energy_floor < 0.0 {
            return Err(TensorError::NonPositiveCoherence {
                coherence: energy_floor,
            }
            .into());
        }
        self.energy_floor = energy_floor;
        Ok(self)
    }

    /// Ensures a minimum number of channels survive every pre-discard pass.
    pub fn with_min_channels(mut self, min_channels: usize) -> Self {
        self.min_channels = min_channels.max(1);
        self
    }

    /// Returns the configured dominance ratio.
    pub fn dominance_ratio(&self) -> f32 {
        self.dominance_ratio
    }

    /// Returns the configured absolute floor.
    pub fn energy_floor(&self) -> f32 {
        self.energy_floor
    }

    /// Returns the minimum number of channels that will survive.
    pub fn min_channels(&self) -> usize {
        self.min_channels
    }

    fn apply(&self, weights: &mut [f32], original: &[f32]) -> PureResult<PreDiscardOutcome> {
        if weights.len() != original.len() {
            return Err(TensorError::DataLength {
                expected: original.len(),
                got: weights.len(),
            }
            .into());
        }
        if weights.is_empty() {
            return Ok(PreDiscardOutcome::fallback(
                PreDiscardTelemetry::fallback(self, 0, 0, 0.0, 0.0),
                Vec::new(),
                Vec::new(),
            ));
        }

        let mut indices: Vec<usize> = (0..weights.len()).collect();
        indices.sort_by(|lhs, rhs| {
            original[*rhs]
                .partial_cmp(&original[*lhs])
                .unwrap_or(Ordering::Equal)
        });

        let mut dominant = original[indices[0]];
        if !dominant.is_finite() {
            dominant = 1e-6;
        }
        if dominant <= 0.0 {
            dominant = 1e-6;
        }
        let mut survivors = vec![false; weights.len()];
        for &idx in &indices {
            let mut weight = original[idx];
            if !weight.is_finite() {
                weight = 0.0;
            }
            let tolerance = (1.0 - self.dominance_ratio).max(0.0);
            let threshold = dominant * tolerance;
            let survives_ratio = weight >= threshold;
            let survives_floor = weight >= self.energy_floor && dominant <= self.energy_floor;
            if survives_ratio || survives_floor {
                survivors[idx] = true;
            }
        }

        let preserved = survivors.iter().filter(|flag| **flag).count();
        if preserved < self.min_channels.min(weights.len()) {
            for &idx in indices.iter().take(self.min_channels.min(weights.len())) {
                if !survivors[idx] {
                    survivors[idx] = true;
                }
            }
        }

        let mut survivor_indices = Vec::new();
        let mut discarded_indices = Vec::new();
        let mut sum = 0.0f32;
        let mut survivor_energy = 0.0f32;
        let mut discarded_energy = 0.0f32;
        for (idx, weight) in weights.iter_mut().enumerate() {
            if survivors[idx] {
                let mut value = original[idx];
                if !value.is_finite() {
                    value = 0.0;
                }
                *weight = value;
                sum += *weight;
                survivor_energy += value;
                survivor_indices.push(idx);
            } else {
                let value = if original[idx].is_finite() {
                    original[idx]
                } else {
                    0.0
                };
                discarded_energy += value;
                *weight = 0.0;
                discarded_indices.push(idx);
            }
        }
        let discarded = discarded_indices.len();

        if sum <= f32::EPSILON || !sum.is_finite() {
            weights.copy_from_slice(original);
            return Ok(PreDiscardOutcome::fallback(
                PreDiscardTelemetry::fallback(
                    self,
                    0,
                    original.len(),
                    original
                        .iter()
                        .copied()
                        .filter(|value| value.is_finite())
                        .sum(),
                    dominant,
                ),
                (0..original.len()).collect(),
                Vec::new(),
            ));
        }

        for weight in weights.iter_mut() {
            if *weight > 0.0 {
                *weight /= sum;
            }
        }

        Ok(PreDiscardOutcome::new(
            PreDiscardTelemetry::new(
                self,
                discarded,
                weights.len() - discarded,
                false,
                survivor_energy,
                discarded_energy,
                dominant,
            ),
            survivor_indices,
            discarded_indices,
        ))
    }
}

/// Telemetry describing a pre-discard pass.
#[derive(Clone, Debug)]
pub struct PreDiscardTelemetry {
    dominance_ratio: f32,
    energy_floor: f32,
    discarded: usize,
    preserved: usize,
    fallback: bool,
    survivor_energy: f32,
    discarded_energy: f32,
    dominant_weight: f32,
}

impl PreDiscardTelemetry {
    fn new(
        policy: &PreDiscardPolicy,
        discarded: usize,
        preserved: usize,
        fallback: bool,
        survivor_energy: f32,
        discarded_energy: f32,
        dominant_weight: f32,
    ) -> Self {
        Self {
            dominance_ratio: policy.dominance_ratio,
            energy_floor: policy.energy_floor,
            discarded,
            preserved,
            fallback,
            survivor_energy,
            discarded_energy,
            dominant_weight,
        }
    }

    fn fallback(
        policy: &PreDiscardPolicy,
        discarded: usize,
        preserved: usize,
        total_energy: f32,
        dominant_weight: f32,
    ) -> Self {
        Self::new(
            policy,
            discarded,
            preserved,
            true,
            total_energy,
            0.0,
            dominant_weight,
        )
    }

    /// Returns the dominance ratio used during pre-discard.
    pub fn dominance_ratio(&self) -> f32 {
        self.dominance_ratio
    }

    /// Returns the absolute energy floor used during pre-discard.
    pub fn energy_floor(&self) -> f32 {
        self.energy_floor
    }

    /// Returns the number of channels that were discarded.
    pub fn discarded(&self) -> usize {
        self.discarded
    }

    /// Returns the number of channels that survived the pass.
    pub fn preserved(&self) -> usize {
        self.preserved
    }

    /// Indicates whether the policy had to fall back to the original distribution.
    pub fn used_fallback(&self) -> bool {
        self.fallback
    }

    /// Returns the total number of channels that were considered.
    pub fn total(&self) -> usize {
        self.discarded + self.preserved
    }

    /// Returns the fraction of channels that survived the pass.
    pub fn preserved_ratio(&self) -> f32 {
        let total = self.total().max(1) as f32;
        (self.preserved as f32 / total).clamp(0.0, 1.0)
    }

    /// Returns the fraction of channels that were discarded.
    pub fn discarded_ratio(&self) -> f32 {
        1.0 - self.preserved_ratio()
    }

    /// Returns the sum of the original weights that survived the pass.
    pub fn survivor_energy(&self) -> f32 {
        self.survivor_energy
    }

    /// Returns the sum of the original weights that were discarded.
    pub fn discarded_energy(&self) -> f32 {
        self.discarded_energy
    }

    /// Returns the total original weight energy observed during the pass.
    pub fn total_energy(&self) -> f32 {
        self.survivor_energy + self.discarded_energy
    }

    /// Returns the fraction of original energy that survived.
    pub fn survivor_energy_ratio(&self) -> f32 {
        let total = self.total_energy();
        if total <= f32::EPSILON {
            0.0
        } else {
            (self.survivor_energy / total).clamp(0.0, 1.0)
        }
    }

    /// Returns the fraction of original energy that was discarded.
    pub fn discarded_energy_ratio(&self) -> f32 {
        1.0 - self.survivor_energy_ratio()
    }

    /// Returns the dominant channel weight observed before discard.
    pub fn dominant_weight(&self) -> f32 {
        self.dominant_weight
    }
}

/// Detailed outcome of a pre-discard pass, including survivor indexing.
#[derive(Clone, Debug)]
pub struct PreDiscardOutcome {
    telemetry: PreDiscardTelemetry,
    survivors: Vec<usize>,
    discarded: Vec<usize>,
}

impl PreDiscardOutcome {
    fn new(telemetry: PreDiscardTelemetry, survivors: Vec<usize>, discarded: Vec<usize>) -> Self {
        Self {
            telemetry,
            survivors,
            discarded,
        }
    }

    fn fallback(
        telemetry: PreDiscardTelemetry,
        survivors: Vec<usize>,
        discarded: Vec<usize>,
    ) -> Self {
        Self::new(telemetry, survivors, discarded)
    }

    /// Returns the telemetry emitted by the pass.
    pub fn telemetry(&self) -> &PreDiscardTelemetry {
        &self.telemetry
    }

    /// Returns indices of channels that survived.
    pub fn survivors(&self) -> &[usize] {
        &self.survivors
    }

    /// Returns indices of channels that were discarded.
    pub fn discarded(&self) -> &[usize] {
        &self.discarded
    }
}

/// Snapshot of a pre-discard pass preserved in the sequencer journal.
#[derive(Clone, Debug)]
pub struct PreDiscardSnapshot {
    step: u64,
    telemetry: PreDiscardTelemetry,
    survivors: Vec<usize>,
    discarded: Vec<usize>,
    filtered: Vec<f32>,
}

impl PreDiscardSnapshot {
    fn new(
        step: u64,
        telemetry: PreDiscardTelemetry,
        survivors: Vec<usize>,
        discarded: Vec<usize>,
        filtered: Vec<f32>,
    ) -> Self {
        Self {
            step,
            telemetry,
            survivors,
            discarded,
            filtered,
        }
    }

    /// Returns the journal step associated with the snapshot.
    pub fn step(&self) -> u64 {
        self.step
    }

    /// Returns telemetry captured during the pass.
    pub fn telemetry(&self) -> &PreDiscardTelemetry {
        &self.telemetry
    }

    /// Returns the indices that survived the pass.
    pub fn survivors(&self) -> &[usize] {
        &self.survivors
    }

    /// Returns the indices that were discarded by the pass.
    pub fn discarded(&self) -> &[usize] {
        &self.discarded
    }

    /// Returns the filtered coherence weights after pre-discard.
    pub fn filtered(&self) -> &[f32] {
        &self.filtered
    }
}

#[derive(Debug)]
struct PreDiscardJournal {
    next_step: u64,
    limit: usize,
    entries: VecDeque<PreDiscardSnapshot>,
}

impl PreDiscardJournal {
    fn new(limit: usize) -> Self {
        Self {
            next_step: 0,
            limit: limit.max(1),
            entries: VecDeque::new(),
        }
    }

    fn push(
        &mut self,
        telemetry: PreDiscardTelemetry,
        survivors: Vec<usize>,
        discarded: Vec<usize>,
        filtered: Vec<f32>,
    ) {
        let snapshot =
            PreDiscardSnapshot::new(self.next_step, telemetry, survivors, discarded, filtered);
        self.next_step = self.next_step.wrapping_add(1);
        self.entries.push_back(snapshot);
        while self.entries.len() > self.limit {
            self.entries.pop_front();
        }
    }

    fn clear(&mut self) {
        self.entries.clear();
    }

    fn set_limit(&mut self, limit: usize) {
        self.limit = limit.max(1);
        while self.entries.len() > self.limit {
            self.entries.pop_front();
        }
    }

    fn snapshots(&self) -> Vec<PreDiscardSnapshot> {
        self.entries.iter().cloned().collect()
    }
}

/// Z-space native sequence modeling via coherence and semiotic suturing.
///
/// This layer replaces attention with:
/// 1. **Coherence-based token weighting** (Maxwell pulse detection)
/// 2. **Desire-biased logits** (linguistic safety without RLHF)
/// 3. **Hyperbolic token relationships** (natural hierarchy)
/// 4. **Spectral aggregation** (fractional calculus instead of softmax)
#[derive(Clone)]
pub struct ZSpaceCoherenceSequencer {
    pub dim: usize,
    pub num_heads: usize,
    pub curvature: f32,

    coherence_engine: CoherenceEngine,
    topos: OpenCartesianTopos,
    plugins: Vec<Arc<dyn ZSpaceSequencerPlugin>>,
    pre_discard: Option<PreDiscardPolicy>,
    pre_discard_journal: Arc<Mutex<PreDiscardJournal>>,
}

impl ZSpaceCoherenceSequencer {
    /// Creates a new Z-space coherence sequencer.
    pub fn new(
        dim: usize,
        num_heads: usize,
        curvature: f32,
        topos: OpenCartesianTopos,
    ) -> PureResult<Self> {
        if num_heads == 0 {
            return Err(st_tensor::TensorError::EmptyInput("maxwell_heads").into());
        }
        if dim % num_heads != 0 {
            return Err(st_tensor::TensorError::InvalidDimensions {
                rows: dim,
                cols: num_heads,
            }
            .into());
        }
        if (topos.curvature() - curvature).abs() > 1e-6 {
            return Err(st_tensor::TensorError::CurvatureMismatch {
                expected: curvature,
                got: topos.curvature(),
            }
            .into());
        }

        let channel_count = (dim / 8).max(1).max(num_heads);
        let coherence_engine =
            CoherenceEngine::new(dim, curvature)?.with_channel_count(channel_count)?;

        Ok(Self {
            dim,
            num_heads,
            curvature,
            coherence_engine,
            topos,
            plugins: Vec::new(),
            pre_discard: None,
            pre_discard_journal: Arc::new(Mutex::new(PreDiscardJournal::new(32))),
        })
    }

    /// Projects input to Z-space and measures coherence.
    fn measure_coherence(&self, x: &Tensor) -> PureResult<Vec<f32>> {
        self.coherence_engine.measure_phases(x)
    }

    /// Projects an input tensor onto the Z-space manifold guarded by the topos.
    pub fn project_to_zspace(&self, x: &Tensor) -> PureResult<Tensor> {
        let mut projected = if self.curvature < 0.0 {
            x.project_to_poincare(self.curvature)?
        } else {
            x.clone()
        };

        self.topos.saturate_slice(projected.data_mut());
        self.topos
            .guard_tensor("zspace_coherence_project_to_zspace", &projected)?;

        Ok(projected)
    }

    /// Performs coherence-weighted geometric aggregation and surfaces diagnostics.
    fn geometric_aggregate_with_diagnostics(
        &self,
        x: &Tensor,
        coherence_weights: &[f32],
        pre_discard: Option<PreDiscardTelemetry>,
    ) -> PureResult<(Tensor, CoherenceDiagnostics)> {
        let (aggregated, normalization, fractional_order, channel_width) =
            self.compute_geometric_aggregate(x, coherence_weights)?;

        let diagnostics = self.build_coherence_diagnostics(
            &aggregated,
            coherence_weights,
            channel_width,
            normalization,
            fractional_order,
            pre_discard,
        );

        Ok((aggregated, diagnostics))
    }

    fn compute_geometric_aggregate(
        &self,
        x: &Tensor,
        coherence_weights: &[f32],
    ) -> PureResult<(Tensor, f32, f32, usize)> {
        if coherence_weights.is_empty() {
            return Err(TensorError::EmptyInput("coherence_weights").into());
        }
        if coherence_weights.len() != self.coherence_engine.num_channels() {
            return Err(TensorError::DataLength {
                expected: self.coherence_engine.num_channels(),
                got: coherence_weights.len(),
            }
            .into());
        }

        let (rows, cols) = x.shape();
        let mut aggregated = Tensor::zeros(rows, cols)?;
        let channel_width = (cols + coherence_weights.len() - 1) / coherence_weights.len();
        let normalization = coherence_weights.iter().copied().sum::<f32>().max(1e-6);
        let fractional_order = self.fractional_order();
        let input = x.data();
        {
            let output = aggregated.data_mut();
            for row in 0..rows {
                let row_start = row * cols;
                let row_slice = &input[row_start..row_start + cols];
                let row_out = &mut output[row_start..row_start + cols];
                row_out.fill(0.0);
                for (channel, &weight) in coherence_weights.iter().enumerate() {
                    let start = channel * channel_width;
                    let end = ((channel + 1) * channel_width).min(cols);
                    if start >= end {
                        continue;
                    }
                    for (dest, &value) in row_out[start..end].iter_mut().zip(&row_slice[start..end])
                    {
                        *dest += weight * value;
                    }
                }
                for value in row_out.iter_mut() {
                    *value /= normalization;
                }
                // Fractional smoothing forward pass.
                let mut accumulator = row_out.first().copied().unwrap_or(0.0);
                for value in row_out.iter_mut().skip(1) {
                    accumulator += fractional_order * (*value - accumulator);
                    *value = accumulator;
                }
                // Reverse fractional smoothing to keep symmetry.
                if cols > 1 {
                    let mut accumulator = row_out.last().copied().unwrap_or(0.0);
                    for value in row_out.iter_mut().rev().skip(1) {
                        accumulator += fractional_order * (*value - accumulator);
                        *value = accumulator;
                    }
                }
            }
        }

        self.topos.saturate_slice(aggregated.data_mut());
        self.topos
            .guard_tensor("zspace_coherence_geometric_aggregate", &aggregated)?;

        Ok((
            aggregated,
            normalization,
            fractional_order,
            channel_width.max(1),
        ))
    }

    /// Performs coherence-weighted geometric aggregation.
    pub fn geometric_aggregate(&self, x: &Tensor, coherence_weights: &[f32]) -> PureResult<Tensor> {
        let (aggregated, _, _, _) = self.compute_geometric_aggregate(x, coherence_weights)?;
        Ok(aggregated)
    }

    fn build_coherence_diagnostics(
        &self,
        aggregated: &Tensor,
        coherence_weights: &[f32],
        channel_width: usize,
        normalization: f32,
        fractional_order: f32,
        pre_discard: Option<PreDiscardTelemetry>,
    ) -> CoherenceDiagnostics {
        let channel_weights = coherence_weights.to_vec();
        let mut normalized_weights: Vec<f32> =
            channel_weights.iter().map(|value| value.max(0.0)).collect();
        let sum = normalized_weights.iter().sum::<f32>();
        if sum > 0.0 {
            for value in &mut normalized_weights {
                *value = (*value / sum).max(1e-6);
            }
        } else if !normalized_weights.is_empty() {
            let fill = 1.0 / normalized_weights.len() as f32;
            for value in &mut normalized_weights {
                *value = fill;
            }
        }

        let mean_coherence = if channel_weights.is_empty() {
            0.0
        } else {
            channel_weights.iter().copied().sum::<f32>() / channel_weights.len() as f32
        };

        let dominant_channel = channel_weights
            .iter()
            .enumerate()
            .filter(|(_, weight)| weight.is_finite())
            .max_by(|(_, lhs), (_, rhs)| lhs.partial_cmp(rhs).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx);

        let data = aggregated.data();
        let (rows, cols) = aggregated.shape();
        let total_energy = data.iter().map(|value| value.abs()).sum::<f32>().max(1e-6);
        let mut dominant_energy = 0.0f32;
        if let Some(channel) = dominant_channel {
            let start = channel * channel_width;
            let end = ((channel + 1) * channel_width).min(cols);
            if start < end {
                for row in 0..rows {
                    let offset = row * cols;
                    for value in &data[offset + start..offset + end] {
                        dominant_energy += value.abs();
                    }
                }
            }
        }
        let energy_ratio = (dominant_energy / total_energy).clamp(0.0, 1.0);

        let entropy = -normalized_weights
            .iter()
            .filter(|value| value.is_finite() && **value > 0.0)
            .map(|value| *value * value.ln())
            .sum::<f32>();

        let z_bias = self
            .summarise_maxwell_pulse(aggregated, coherence_weights)
            .z_bias;

        CoherenceDiagnostics {
            channel_weights: channel_weights.clone(),
            normalized_weights,
            normalization,
            fractional_order,
            dominant_channel,
            mean_coherence,
            z_bias,
            energy_ratio,
            entropy,
            aggregated: aggregated.clone(),
            coherence: channel_weights,
            channel_reports: Vec::new(),
            pre_discard,
        }
    }

    pub fn forward_with_diagnostics(
        &self,
        x: &Tensor,
    ) -> PureResult<(Tensor, Vec<f32>, CoherenceDiagnostics)> {
        let _ = self.topos.curvature();
        // Step 1: Project to Z-space
        let z_space = self.project_to_zspace(x)?;
        self.dispatch_plugins(|| ZSpaceSequencerStage::Projected {
            input: x,
            projected: &z_space,
        })?;

        // Step 2: Measure Maxwell coherence
        let mut coherence = self.measure_coherence(&z_space)?;
        self.dispatch_plugins(|| ZSpaceSequencerStage::CoherenceMeasured {
            z_space: &z_space,
            coherence: &coherence,
        })?;

        let pre_discard_state = self.run_pre_discard(&mut coherence)?;

        // Step 3: Geometric aggregation (replaces attention) with diagnostics
        let (aggregated, diagnostics) = self.geometric_aggregate_with_diagnostics(
            &z_space,
            &coherence,
            pre_discard_state.clone(),
        )?;

        let channel_reports = self.coherence_engine.describe_channels(&coherence)?;
        let diagnostics = diagnostics.with_channel_reports(channel_reports);
        self.dispatch_plugins(|| ZSpaceSequencerStage::Aggregated {
            aggregated: &aggregated,
            coherence: &coherence,
            diagnostics: &diagnostics,
        })?;

        // Step 4: Output from Z-space (to Euclidean)
        Ok((aggregated, coherence, diagnostics))
    }

    pub fn forward_with_coherence(&self, x: &Tensor) -> PureResult<(Tensor, Vec<f32>)> {
        let (aggregated, coherence, _) = self.forward_with_diagnostics(x)?;
        Ok((aggregated, coherence))
    }

    /// Produces a rich diagnostic snapshot that includes per-channel linguistic
    /// descriptors alongside the aggregated tensor.
    pub fn diagnostics(&self, x: &Tensor) -> PureResult<CoherenceDiagnostics> {
        let (_aggregated, _coherence, diagnostics) = self.forward_with_diagnostics(x)?;
        Ok(diagnostics)
    }

    pub fn forward(&self, x: &Tensor) -> PureResult<Tensor> {
        let (aggregated, _, _) = self.forward_with_diagnostics(x)?;
        Ok(aggregated)
    }

    /// Runs the sequencer while fusing semantic and narrative bridges so callers can
    /// project coherence weights back into language space.
    pub fn forward_with_language_bridges(
        &self,
        x: &Tensor,
        semantics: &SemanticBridge,
        maxwell_bridge: &MaxwellDesireBridge,
    ) -> PureResult<(
        Tensor,
        Vec<f32>,
        ConceptHint,
        Option<NarrativeHint>,
        MaxwellZPulse,
    )> {
        let (aggregated, coherence) = self.forward_with_coherence(x)?;
        let semantic_distribution =
            self.derive_semantic_distribution(&aggregated, &coherence, semantics)?;
        let pulse = self.summarise_maxwell_pulse(&aggregated, &coherence);
        let canonical_concept = self.canonical_domain_concept();
        let channel_label = canonical_concept.label().to_string();
        self.dispatch_plugins(|| ZSpaceSequencerStage::CanonicalConceptSelected {
            concept: canonical_concept.clone(),
            channel: channel_label.as_str(),
        })?;

        let emission = maxwell_bridge.emit(&channel_label, &pulse);
        if let Some((ref hint, ref narrative)) = emission {
            self.dispatch_plugins(|| ZSpaceSequencerStage::MaxwellBridgeEmitted {
                channel: channel_label.as_str(),
                pulse: &pulse,
                hint,
                narrative: narrative.as_ref(),
            })?;
        }

        let (concept_hint, narrative) = if let Some((hint, narrative)) = emission {
            let fused =
                Self::fuse_distributions(&semantic_distribution, &hint.as_distribution(semantics));
            let concept = ConceptHint::Distribution(fused);
            self.dispatch_plugins(|| ZSpaceSequencerStage::SemanticWindowFused {
                concept: &concept,
            })?;
            (concept, narrative)
        } else {
            let concept = ConceptHint::Distribution(semantic_distribution);
            self.dispatch_plugins(|| ZSpaceSequencerStage::SemanticWindowFused {
                concept: &concept,
            })?;
            (concept, None)
        };

        let narrative_ref = narrative.as_ref();
        self.dispatch_plugins(|| ZSpaceSequencerStage::LanguageBridged {
            aggregated: &aggregated,
            coherence: &coherence,
            concept: &concept_hint,
            narrative: narrative_ref,
            pulse: &pulse,
        })?;

        Ok((aggregated, coherence, concept_hint, narrative, pulse))
    }

    #[cfg(feature = "psi")]
    /// Runs the sequencer, fuses language bridges, and publishes PSI telemetry
    /// derived from the Maxwell pulse via the provided telemetry bridge.
    pub fn forward_with_language_and_psi(
        &self,
        x: &Tensor,
        semantics: &SemanticBridge,
        maxwell_bridge: &MaxwellDesireBridge,
        psi_bridge: &MaxwellPsiTelemetryBridge,
        psi_step: u64,
    ) -> PureResult<(
        Tensor,
        Vec<f32>,
        ConceptHint,
        Option<NarrativeHint>,
        MaxwellZPulse,
        Option<PsiReading>,
        Vec<PsiEvent>,
        SoftlogicZFeedback,
    )> {
        let (aggregated, coherence, concept_hint, narrative, pulse) =
            self.forward_with_language_bridges(x, semantics, maxwell_bridge)?;
        let (feedback, psi_reading, psi_events) = psi_bridge.publish_with_reading(&pulse, psi_step);

        self.dispatch_plugins(|| ZSpaceSequencerStage::PsiTelemetryPublished {
            pulse: &pulse,
            reading: Some(&psi_reading),
            events: psi_events.as_slice(),
            feedback: &feedback,
        })?;

        Ok((
            aggregated,
            coherence,
            concept_hint,
            narrative,
            pulse,
            Some(psi_reading),
            psi_events,
            feedback,
        ))
    }

    /// Configures the execution backend for coherence measurement.
    pub fn set_backend(&mut self, backend: CoherenceBackend) -> PureResult<()> {
        self.coherence_engine.set_backend(backend.clone());
        self.dispatch_plugins(|| ZSpaceSequencerStage::BackendConfigured {
            backend: backend.clone(),
        })?;
        Ok(())
    }

    /// Enables a pre-discard policy so low-credence channels are culled before aggregation.
    pub fn enable_pre_discard(&mut self, policy: PreDiscardPolicy) {
        self.pre_discard = Some(policy);
    }

    /// Disables any active pre-discard policy.
    pub fn disable_pre_discard(&mut self) {
        self.pre_discard = None;
    }

    /// Returns the active pre-discard policy, when configured.
    pub fn pre_discard_policy(&self) -> Option<&PreDiscardPolicy> {
        self.pre_discard.as_ref()
    }

    /// Configures how many pre-discard snapshots are retained in memory.
    pub fn configure_pre_discard_memory(&mut self, limit: usize) {
        let mut journal = self.lock_pre_discard_journal();
        journal.set_limit(limit);
    }

    /// Returns the recorded pre-discard snapshots in chronological order.
    pub fn pre_discard_snapshots(&self) -> Vec<PreDiscardSnapshot> {
        let journal = self.lock_pre_discard_journal();
        journal.snapshots()
    }

    /// Clears all retained pre-discard snapshots.
    pub fn clear_pre_discard_snapshots(&self) {
        let mut journal = self.lock_pre_discard_journal();
        journal.clear();
    }

    /// Registers a plugin that will receive callbacks across the sequencing pipeline.
    pub fn register_plugin<P>(&mut self, plugin: P)
    where
        P: ZSpaceSequencerPlugin + 'static,
    {
        self.plugins.push(Arc::new(plugin));
    }

    /// Removes all registered plugins.
    pub fn clear_plugins(&mut self) {
        self.plugins.clear();
    }

    /// Returns the descriptive names of the registered plugins.
    pub fn plugin_names(&self) -> Vec<&'static str> {
        self.plugins.iter().map(|plugin| plugin.name()).collect()
    }

    /// Registers a domain linguistic profile used to bias coherence weights.
    pub fn register_linguistic_profile(
        &mut self,
        profile: DomainLinguisticProfile,
    ) -> PureResult<()> {
        self.coherence_engine.register_linguistic_profile(profile);
        if let Some(profile) = self.coherence_engine.linguistic_profiles().last() {
            self.dispatch_plugins(|| ZSpaceSequencerStage::LinguisticProfileRegistered {
                profile,
            })?;
        }
        Ok(())
    }

    /// Removes all linguistic profiles from the underlying coherence engine.
    pub fn clear_linguistic_profiles(&mut self) -> PureResult<()> {
        self.coherence_engine.clear_linguistic_profiles();
        self.dispatch_plugins(|| ZSpaceSequencerStage::LinguisticProfilesCleared)?;
        Ok(())
    }

    /// Exposes the registered linguistic profiles.
    pub fn linguistic_profiles(&self) -> &[DomainLinguisticProfile] {
        self.coherence_engine.linguistic_profiles()
    }

    /// Returns the configured coherence backend.
    pub fn backend(&self) -> &CoherenceBackend {
        self.coherence_engine.backend()
    }

    /// Converts coherence weights into a linguistic contour descriptor that can be
    /// used by downstream vocalisation stacks.
    pub fn emit_linguistic_contour(&self, x: &Tensor) -> PureResult<LinguisticContour> {
        let mut coherence = self.measure_coherence(x)?;
        let _ = self.run_pre_discard(&mut coherence)?;
        let contour = self
            .coherence_engine
            .derive_linguistic_contour(&coherence)?;
        self.dispatch_plugins(|| ZSpaceSequencerStage::LinguisticContourEmitted {
            coherence: &coherence,
            contour: &contour,
        })?;
        Ok(contour)
    }

    /// Describes each coherence channel, surfacing dominant linguistic concepts per band.
    pub fn describe_channels(&self, x: &Tensor) -> PureResult<Vec<LinguisticChannelReport>> {
        let mut coherence = self.measure_coherence(x)?;
        let _ = self.run_pre_discard(&mut coherence)?;
        let reports = self.coherence_engine.describe_channels(&coherence)?;
        self.dispatch_plugins(|| ZSpaceSequencerStage::ChannelsDescribed {
            coherence: &coherence,
            reports: &reports,
        })?;
        Ok(reports)
    }

    /// Returns the number of Maxwell coherence channels computed by the engine.
    pub fn maxwell_channels(&self) -> usize {
        self.coherence_engine.num_channels()
    }

    /// Provides read-only access to the geometric topos used by the sequencer.
    pub fn topos(&self) -> &OpenCartesianTopos {
        &self.topos
    }

    /// Selects a canonical domain concept based on the curvature and head count.
    pub fn canonical_domain_concept(&self) -> DomainConcept {
        if self.curvature < -0.5 {
            if self.num_heads > 16 {
                DomainConcept::NeuronalPattern
            } else {
                DomainConcept::Membrane
            }
        } else if self.curvature > 0.5 {
            if self.num_heads > 8 {
                DomainConcept::DropletCoalescence
            } else {
                DomainConcept::GrainBoundary
            }
        } else if self.num_heads >= 12 {
            DomainConcept::NeuronalPattern
        } else {
            DomainConcept::Membrane
        }
    }

    fn fractional_order(&self) -> f32 {
        match self.canonical_domain_concept() {
            DomainConcept::Membrane => (1.0 / (1.0 + self.curvature.abs())).clamp(0.1, 0.85),
            DomainConcept::GrainBoundary => {
                (1.0 / (1.0 + self.curvature.abs() * 0.8)).clamp(0.15, 0.9)
            }
            DomainConcept::NeuronalPattern => {
                (1.0 / (1.0 + self.curvature.abs() * 0.6)).clamp(0.2, 0.95)
            }
            DomainConcept::DropletCoalescence => {
                (1.0 / (1.0 + self.curvature.abs() * 1.2)).clamp(0.05, 0.8)
            }
            DomainConcept::Custom(_) => (1.0 / (1.0 + self.curvature.abs())).clamp(0.1, 0.95),
        }
    }

    fn derive_semantic_distribution(
        &self,
        aggregated: &Tensor,
        coherence: &[f32],
        semantics: &SemanticBridge,
    ) -> PureResult<Vec<f32>> {
        let tokens = semantics.vocab_size();
        let window = self.derive_semantic_window(aggregated, coherence, tokens);
        self.dispatch_plugins(|| ZSpaceSequencerStage::SemanticWindowDerived {
            window: &window,
            tokens,
        })?;

        let distribution = if window.is_empty() {
            let concepts = semantics.concept_count().max(1);
            vec![1.0 / concepts as f32; concepts]
        } else {
            semantics.infer_from_window(&window, 1e-6)
        };

        self.dispatch_plugins(|| ZSpaceSequencerStage::SemanticDistributionDerived {
            window: &window,
            distribution: &distribution,
        })?;

        Ok(distribution)
    }

    fn derive_semantic_window(
        &self,
        aggregated: &Tensor,
        coherence: &[f32],
        tokens: usize,
    ) -> Vec<(usize, f32)> {
        if tokens == 0 || coherence.is_empty() {
            return Vec::new();
        }
        let (rows, cols) = aggregated.shape();
        if cols == 0 || rows == 0 {
            return Vec::new();
        }
        let token_width = (cols + tokens - 1) / tokens;
        let channel_width = (cols + coherence.len() - 1) / coherence.len().max(1);
        let mut window = Vec::with_capacity(tokens);
        let data = aggregated.data();
        for token in 0..tokens {
            let start = token * token_width;
            if start >= cols {
                break;
            }
            let end = ((token + 1) * token_width).min(cols);
            if start >= end {
                continue;
            }
            let mut energy = 0.0f32;
            for row in 0..rows {
                let offset = row * cols;
                for value in &data[offset + start..offset + end] {
                    energy += value.abs();
                }
            }
            let samples = (end - start) * rows;
            if samples == 0 {
                continue;
            }
            energy /= samples as f32;
            if energy <= 0.0 || !energy.is_finite() {
                continue;
            }
            let center = (start + end - 1) / 2;
            let channel = center / channel_width;
            let coherence_weight = coherence
                .get(channel)
                .copied()
                .unwrap_or(1.0 / coherence.len() as f32);
            let weight = (energy * coherence_weight).max(0.0);
            if weight > 0.0 {
                window.push((token, weight));
            }
        }
        let sum: f32 = window.iter().map(|(_, weight)| *weight).sum();
        if sum > 0.0 {
            for (_, weight) in &mut window {
                *weight = (*weight / sum).max(1e-6);
            }
        }
        window
    }

    fn summarise_maxwell_pulse(&self, aggregated: &Tensor, coherence: &[f32]) -> MaxwellZPulse {
        let data = aggregated.data();
        let total = data.len().max(1);
        let total_f64 = total as f64;
        let mean = data.iter().map(|v| *v as f64).sum::<f64>() / total_f64;
        let variance = data
            .iter()
            .map(|v| {
                let diff = *v as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / total_f64.max(1.0);
        let std_dev = variance.sqrt();
        let standard_error = if total_f64 > 0.0 {
            std_dev / total_f64.sqrt()
        } else {
            0.0
        };
        let z_score = if standard_error > 0.0 {
            mean / standard_error
        } else {
            0.0
        };
        let third = (coherence.len() / 3).max(1);
        let above: f32 = coherence.iter().take(third).copied().sum();
        let here: f32 = coherence.iter().skip(third).take(third).copied().sum();
        let beneath: f32 = coherence.iter().skip(third * 2).copied().sum();
        let curvature_scale = self.curvature.abs().sqrt();
        let mut z_bias = (above - beneath) * curvature_scale;
        if !z_bias.is_finite() {
            z_bias = 0.0;
        }
        MaxwellZPulse {
            blocks: aggregated.shape().0 as u64,
            mean,
            standard_error,
            z_score,
            band_energy: (above, here, beneath),
            z_bias,
        }
    }

    fn fuse_distributions(lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
        let len = lhs.len().max(rhs.len()).max(1);
        let mut fused = vec![0.0f32; len];
        for (idx, slot) in fused.iter_mut().enumerate() {
            let a = lhs.get(idx).copied().unwrap_or(1e-6);
            let b = rhs.get(idx).copied().unwrap_or(1e-6);
            *slot = a.max(0.0) + b.max(0.0);
        }
        let sum: f32 = fused.iter().sum();
        if sum > 0.0 {
            for value in &mut fused {
                *value = (*value / sum).max(1e-6);
            }
        } else {
            let fill = 1.0 / len as f32;
            for value in &mut fused {
                *value = fill;
            }
        }
        fused
    }

    fn run_pre_discard(&self, coherence: &mut Vec<f32>) -> PureResult<Option<PreDiscardTelemetry>> {
        let policy = match &self.pre_discard {
            Some(policy) => policy,
            None => return Ok(None),
        };
        let original = coherence.clone();
        let outcome = policy.apply(coherence, &original)?;
        let telemetry = outcome.telemetry().clone();
        let survivors: Vec<usize> = outcome.survivors().to_vec();
        let discarded: Vec<usize> = outcome.discarded().to_vec();
        self.dispatch_plugins(|| ZSpaceSequencerStage::PreDiscardApplied {
            original: &original,
            filtered: coherence,
            telemetry: &telemetry,
            survivors: &survivors,
            discarded: &discarded,
        })?;
        self.record_pre_discard_snapshot(
            coherence.clone(),
            telemetry.clone(),
            survivors,
            discarded,
        );
        Ok(Some(telemetry))
    }

    fn record_pre_discard_snapshot(
        &self,
        filtered: Vec<f32>,
        telemetry: PreDiscardTelemetry,
        survivors: Vec<usize>,
        discarded: Vec<usize>,
    ) {
        let mut journal = self.lock_pre_discard_journal();
        journal.push(telemetry, survivors, discarded, filtered);
    }

    fn lock_pre_discard_journal(&self) -> MutexGuard<'_, PreDiscardJournal> {
        match self.pre_discard_journal.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    fn dispatch_plugins<'a, F>(&'a self, mut stage: F) -> PureResult<()>
    where
        F: FnMut() -> ZSpaceSequencerStage<'a>,
    {
        if self.plugins.is_empty() {
            return Ok(());
        }
        let event = stage();
        for plugin in &self.plugins {
            plugin.on_stage(event.clone())?;
        }
        Ok(())
    }
}

impl Module for ZSpaceCoherenceSequencer {
    fn forward(&self, x: &Tensor) -> PureResult<Tensor> {
        ZSpaceCoherenceSequencer::forward(self, x)
    }

    fn backward(&mut self, _input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        Ok(grad_output.clone())
    }

    fn visit_parameters(
        &self,
        _visitor: &mut dyn FnMut(&crate::Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        _visitor: &mut dyn FnMut(&mut crate::Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        Ok(())
    }

    fn state_dict(&self) -> PureResult<std::collections::HashMap<String, Tensor>> {
        Ok(std::collections::HashMap::new())
    }

    fn load_state_dict(
        &mut self,
        _state: &std::collections::HashMap<String, Tensor>,
    ) -> PureResult<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::language::{MaxwellDesireBridge, SemanticBridge, SparseKernel};
    use std::sync::{Arc, Mutex};

    #[test]
    fn sequencer_forward_preserves_shape() {
        let topos = OpenCartesianTopos::new(-1.0, 1e-5, 10.0, 256, 8192).unwrap();
        let seq = ZSpaceCoherenceSequencer::new(768, 12, -1.0, topos).unwrap();

        let x = Tensor::from_vec(2, 768, vec![0.1; 768 * 2]).unwrap();
        let out = seq.forward(&x).unwrap();

        assert_eq!(out.shape(), x.shape());
    }

    #[test]
    fn forward_with_coherence_matches_forward() {
        let topos = OpenCartesianTopos::new(-0.5, 1e-5, 10.0, 256, 8192).unwrap();
        let seq = ZSpaceCoherenceSequencer::new(192, 6, -0.5, topos).unwrap();

        let mut ramp = vec![0.0f32; 192 * 3];
        for (idx, value) in ramp.iter_mut().enumerate() {
            *value = (idx as f32 % 17.0) / 17.0;
        }
        let x = Tensor::from_vec(3, 192, ramp).unwrap();

        let (with_coherence, weights) = seq.forward_with_coherence(&x).unwrap();
        let standalone = seq.forward(&x).unwrap();

        assert_eq!(with_coherence.shape(), standalone.shape());
        assert_eq!(with_coherence.data(), standalone.data());
        assert_eq!(weights.len(), seq.maxwell_channels());
        assert!(weights.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn diagnostics_surfaces_channel_reports() {
        let topos = OpenCartesianTopos::new(-0.65, 1e-5, 10.0, 256, 8192).unwrap();
        let seq = ZSpaceCoherenceSequencer::new(256, 8, -0.65, topos).unwrap();

        let mut sweep = vec![0.0f32; 256 * 4];
        for (idx, value) in sweep.iter_mut().enumerate() {
            *value = ((idx % 97) as f32).sin();
        }
        let x = Tensor::from_vec(4, 256, sweep).unwrap();

        let diagnostics = seq.diagnostics(&x).unwrap();
        assert_eq!(diagnostics.aggregated().shape(), x.shape());
        assert_eq!(diagnostics.coherence().len(), seq.maxwell_channels());
        assert_eq!(diagnostics.channel_reports().len(), seq.maxwell_channels());
    }

    #[test]
    fn pre_discard_culls_low_credence_channels() {
        let topos = OpenCartesianTopos::new(-1.0, 1e-5, 10.0, 256, 8192).unwrap();
        let mut seq = ZSpaceCoherenceSequencer::new(128, 8, -1.0, topos).unwrap();
        let policy = PreDiscardPolicy::new(0.35)
            .unwrap()
            .with_energy_floor(1e-4)
            .unwrap()
            .with_min_channels(2);
        seq.enable_pre_discard(policy);

        let mut data = vec![0.02f32; 128];
        for value in &mut data[96..] {
            *value = 0.9;
        }
        let x = Tensor::from_vec(1, 128, data).unwrap();

        let (_, _, diagnostics) = seq.forward_with_diagnostics(&x).unwrap();
        assert!(diagnostics.discarded_channels() > 0);
        assert!(diagnostics.preserved_channels() < seq.maxwell_channels());
        let telemetry = diagnostics.pre_discard().expect("telemetry missing");
        assert!(telemetry.discarded() > 0);
        assert!(!telemetry.used_fallback());
    }

    #[test]
    fn disabling_pre_discard_restores_full_distribution() {
        let topos = OpenCartesianTopos::new(-0.9, 1e-5, 10.0, 256, 8192).unwrap();
        let mut seq = ZSpaceCoherenceSequencer::new(128, 8, -0.9, topos).unwrap();
        let policy = PreDiscardPolicy::new(0.2).unwrap().with_min_channels(1);
        seq.enable_pre_discard(policy);

        let mut data = vec![0.1f32; 128];
        for (idx, value) in data.iter_mut().enumerate() {
            if idx % 5 == 0 {
                *value = 0.6;
            }
        }
        let x = Tensor::from_vec(1, 128, data).unwrap();

        let (_, _, diagnostics) = seq.forward_with_diagnostics(&x).unwrap();
        assert!(diagnostics.discarded_channels() > 0);
        seq.disable_pre_discard();
        let (_, _, diagnostics_after) = seq.forward_with_diagnostics(&x).unwrap();
        assert_eq!(diagnostics_after.discarded_channels(), 0);
        assert!(diagnostics_after.pre_discard().is_none());
    }

    #[test]
    fn pre_discard_journal_records_history() {
        let topos = OpenCartesianTopos::new(-0.85, 1e-5, 10.0, 256, 8192).unwrap();
        let mut seq = ZSpaceCoherenceSequencer::new(160, 10, -0.85, topos).unwrap();
        seq.configure_pre_discard_memory(4);
        let policy = PreDiscardPolicy::new(0.3)
            .unwrap()
            .with_energy_floor(5e-4)
            .unwrap()
            .with_min_channels(2);
        seq.enable_pre_discard(policy);

        let mut stimulus = vec![0.0f32; 160];
        for (idx, value) in stimulus.iter_mut().enumerate() {
            *value = if idx % 17 == 0 { 0.75 } else { 0.05 };
        }
        let x = Tensor::from_vec(1, 160, stimulus).unwrap();

        for _ in 0..3 {
            let _ = seq.forward_with_diagnostics(&x).unwrap();
        }

        let snapshots = seq.pre_discard_snapshots();
        assert!(!snapshots.is_empty());
        let latest = snapshots.last().unwrap();
        assert_eq!(latest.filtered().len(), seq.maxwell_channels());
        assert_eq!(latest.telemetry().total(), seq.maxwell_channels());
        assert_eq!(latest.discarded().len(), latest.telemetry().discarded());
        assert_eq!(latest.survivors().len(), latest.telemetry().preserved());
        assert!(latest.telemetry().discarded_ratio() >= 0.0);
        seq.clear_pre_discard_snapshots();
        assert!(seq.pre_discard_snapshots().is_empty());
    }

    #[test]
    fn pre_discard_telemetry_captures_energy_distribution() {
        let policy = PreDiscardPolicy::new(0.5)
            .unwrap()
            .with_energy_floor(1e-3)
            .unwrap();
        let mut weights = vec![0.8f32, 2.0e-4, 5.0e-5];
        let original = weights.clone();

        let outcome = policy.apply(&mut weights, &original).unwrap();
        let telemetry = outcome.telemetry();

        assert_eq!(telemetry.preserved(), 1);
        assert_eq!(telemetry.discarded(), 2);
        assert!((telemetry.survivor_energy() - 0.8).abs() < 1e-6);
        assert!((telemetry.discarded_energy() - 0.00025).abs() < 1e-6);
        assert!((telemetry.total_energy() - 0.80025).abs() < 1e-6);
        assert!((telemetry.survivor_energy_ratio() - (0.8 / 0.80025)).abs() < 1e-6);
        assert!((telemetry.discarded_energy_ratio() - (0.00025 / 0.80025)).abs() < 1e-6);
        assert!((telemetry.dominant_weight() - 0.8).abs() < 1e-6);

        // The mutated weights should be renormalised after discard.
        assert!((weights[0] - 1.0).abs() < 1e-6);
        assert_eq!(weights[1], 0.0);
        assert_eq!(weights[2], 0.0);
    }

    #[test]
    fn geometric_aggregate_matches_forward_path() {
        let topos = OpenCartesianTopos::new(-0.6, 1e-5, 10.0, 256, 8192).unwrap();
        let seq = ZSpaceCoherenceSequencer::new(256, 8, -0.6, topos).unwrap();

        let mut sweep = vec![0.0f32; 256 * 3];
        for (idx, value) in sweep.iter_mut().enumerate() {
            *value = (idx as f32 * 0.01).cos();
        }
        let x = Tensor::from_vec(3, 256, sweep).unwrap();

        let z_space = seq.project_to_zspace(&x).unwrap();
        let coherence = seq.measure_coherence(&z_space).unwrap();
        let aggregated_direct = seq
            .geometric_aggregate(&z_space, &coherence)
            .expect("geometric aggregation should succeed");

        let (aggregated_forward, _, _) = seq.forward_with_diagnostics(&x).unwrap();

        assert_eq!(aggregated_direct.shape(), aggregated_forward.shape());
        assert_eq!(aggregated_direct.data(), aggregated_forward.data());
    }

    #[test]
    fn projection_respects_poincare_ball_and_topos_guard() {
        let topos = OpenCartesianTopos::new(-0.75, 1e-5, 10.0, 256, 8192).unwrap();
        let seq = ZSpaceCoherenceSequencer::new(128, 8, -0.75, topos).unwrap();

        let mut exaggerated = vec![0.0f32; 128];
        for (idx, value) in exaggerated.iter_mut().enumerate() {
            *value = (idx as f32 + 1.0) * 5.0;
        }

        let x = Tensor::from_vec(1, 128, exaggerated.clone()).unwrap();
        let projected = seq.project_to_zspace(&x).unwrap();

        let norm: f32 = projected.data().iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(norm <= 1.0 + 1e-4);
        assert!(projected.data().iter().all(|v| v.is_finite()));
        assert!(projected.data()[0].abs() < exaggerated[0].abs());

        seq.topos()
            .guard_tensor("projection_respects_poincare_ball", &projected)
            .unwrap();
    }

    #[test]
    fn coherent_aggregation_emphasises_stronger_channels() {
        let topos = OpenCartesianTopos::new(-1.0, 1e-5, 10.0, 256, 8192).unwrap();
        let seq = ZSpaceCoherenceSequencer::new(128, 8, -1.0, topos).unwrap();

        let mut data = vec![0.05f32; 128];
        for value in &mut data[64..] {
            *value = 0.6;
        }
        let x = Tensor::from_vec(1, 128, data).unwrap();
        let out = seq.forward(&x).unwrap();
        let result = out.data();
        let mean_low: f32 = result[..64].iter().sum::<f32>() / 64.0;
        let mean_high: f32 = result[64..].iter().sum::<f32>() / 64.0;
        assert!(mean_high > mean_low);
    }

    #[test]
    fn linguistic_profile_registration_is_reflected() {
        let topos = OpenCartesianTopos::new(-1.0, 1e-5, 10.0, 256, 8192).unwrap();
        let mut seq = ZSpaceCoherenceSequencer::new(128, 8, -1.0, topos).unwrap();
        assert!(seq.linguistic_profiles().is_empty());
        seq.register_linguistic_profile(
            DomainLinguisticProfile::new(DomainConcept::Membrane)
                .with_emphasis(1.2)
                .unwrap(),
        )
        .unwrap();
        assert_eq!(seq.linguistic_profiles().len(), 1);
        seq.clear_linguistic_profiles().unwrap();
        assert!(seq.linguistic_profiles().is_empty());
    }

    #[test]
    fn linguistic_contour_tracks_high_frequency_bias() {
        let topos = OpenCartesianTopos::new(-1.0, 1e-5, 10.0, 256, 8192).unwrap();
        let seq = ZSpaceCoherenceSequencer::new(128, 8, -1.0, topos).unwrap();

        let mut data = vec![0.05f32; 128];
        for value in &mut data[96..] {
            *value = 0.8;
        }
        let x = Tensor::from_vec(1, 128, data).unwrap();
        let contour = seq.emit_linguistic_contour(&x).unwrap();

        assert!(contour.coherence_strength() > 0.0);
        assert!(contour.prosody_index() > 0.6);
        assert!(contour.articulation_bias() > 0.0);
    }

    #[test]
    fn channel_descriptions_surface_linguistic_bias() {
        let topos = OpenCartesianTopos::new(-1.0, 1e-5, 10.0, 256, 8192).unwrap();
        let mut seq = ZSpaceCoherenceSequencer::new(128, 8, -1.0, topos).unwrap();
        seq.register_linguistic_profile(
            DomainLinguisticProfile::new(DomainConcept::DropletCoalescence)
                .with_descriptor("fluid-lilt"),
        )
        .unwrap();
        let x = Tensor::from_vec(1, 128, vec![0.2; 128]).unwrap();
        let reports = seq.describe_channels(&x).unwrap();
        assert_eq!(reports.len(), seq.coherence_engine.num_channels());
        if let Some(report) = reports.first() {
            assert!(report.weight() >= 0.0);
            assert_eq!(report.backend().label(), seq.backend().label());
            assert_eq!(report.descriptor(), Some("fluid-lilt"));
        }
    }

    #[test]
    fn language_bridges_fuse_concepts_and_emit_narrative() {
        let topos = OpenCartesianTopos::new(-1.0, 1e-5, 10.0, 256, 8192).unwrap();
        let seq = ZSpaceCoherenceSequencer::new(128, 8, -1.0, topos).unwrap();

        let concept_kernel =
            SparseKernel::from_dense(vec![vec![0.7, 0.3], vec![0.2, 0.8]], 1e-6).unwrap();
        let semantics = SemanticBridge::from_dense(
            vec![vec![0.6, 0.4], vec![0.3, 0.7]],
            [(0usize, 0usize), (1, 1)],
            1e-6,
            concept_kernel,
        )
        .unwrap();
        let mut bridge = MaxwellDesireBridge::new()
            .with_smoothing(0.05)
            .with_magnitude_floor(0.05)
            .with_channel(
                seq.canonical_domain_concept().label(),
                vec![(0, 0.55), (1, 0.45)],
            )
            .unwrap();
        bridge
            .set_narrative_gain(seq.canonical_domain_concept().label(), 1.2)
            .unwrap();

        let mut data = vec![0.02f32; 128 * 2];
        for (idx, value) in data.iter_mut().enumerate() {
            *value += ((idx % 64) as f32) * 0.01;
        }
        let x = Tensor::from_vec(2, 128, data).unwrap();

        let (_, coherence, concept_hint, narrative, pulse) = seq
            .forward_with_language_bridges(&x, &semantics, &bridge)
            .unwrap();

        assert_eq!(coherence.len(), seq.maxwell_channels());
        assert!(coherence
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0));
        match concept_hint {
            ConceptHint::Distribution(dist) => {
                assert_eq!(dist.len(), semantics.concept_count());
                assert!(dist.iter().all(|value| *value >= 0.0));
                assert!(dist[1] > dist[0]);
            }
            ConceptHint::Window(window) => {
                let dist = semantics.infer_from_window(&window, 1e-6);
                assert_eq!(dist.len(), semantics.concept_count());
                assert!(dist[1] > dist[0]);
            }
        }
        if let Some(narrative) = narrative {
            assert_eq!(narrative.channel(), seq.canonical_domain_concept().label());
            assert!(narrative.intensity() > 0.0);
        }
        assert!(pulse.magnitude().is_finite());
        assert!(pulse.band_energy.0 >= 0.0);
        assert!(pulse.band_energy.1 >= 0.0);
        assert!(pulse.band_energy.2 >= 0.0);
    }

    #[test]
    fn plugins_observe_pipeline_stages() {
        struct RecordingPlugin {
            events: Arc<Mutex<Vec<&'static str>>>,
        }

        impl ZSpaceSequencerPlugin for RecordingPlugin {
            fn name(&self) -> &'static str {
                "recording"
            }

            fn on_stage(&self, stage: ZSpaceSequencerStage<'_>) -> PureResult<()> {
                let label = match stage {
                    ZSpaceSequencerStage::Projected { .. } => "projected",
                    ZSpaceSequencerStage::CoherenceMeasured { .. } => "coherence",
                    ZSpaceSequencerStage::PreDiscardApplied { .. } => "pre_discard",
                    ZSpaceSequencerStage::Aggregated { .. } => "aggregated",
                    ZSpaceSequencerStage::LanguageBridged { .. } => "language",
                    ZSpaceSequencerStage::BackendConfigured { .. } => "backend",
                    ZSpaceSequencerStage::LinguisticProfileRegistered { .. } => {
                        "profile_registered"
                    }
                    ZSpaceSequencerStage::LinguisticProfilesCleared => "profiles_cleared",
                    ZSpaceSequencerStage::SemanticWindowDerived { .. } => "semantic_window",
                    ZSpaceSequencerStage::SemanticDistributionDerived { .. } => {
                        "semantic_distribution"
                    }
                    ZSpaceSequencerStage::CanonicalConceptSelected { .. } => "canonical_concept",
                    ZSpaceSequencerStage::MaxwellBridgeEmitted { .. } => "maxwell_emitted",
                    ZSpaceSequencerStage::LinguisticContourEmitted { .. } => "linguistic_contour",
                    ZSpaceSequencerStage::ChannelsDescribed { .. } => "channels",
                    ZSpaceSequencerStage::SemanticWindowFused { .. } => "semantic_fused",
                    #[cfg(feature = "psi")]
                    ZSpaceSequencerStage::PsiTelemetryPublished { .. } => "psi",
                };
                self.events.lock().unwrap().push(label);
                Ok(())
            }
        }

        let topos = OpenCartesianTopos::new(-1.0, 1e-5, 10.0, 256, 8192).unwrap();
        let mut seq = ZSpaceCoherenceSequencer::new(128, 8, -1.0, topos).unwrap();
        let recorded_events = Arc::new(Mutex::new(Vec::new()));
        seq.register_plugin(RecordingPlugin {
            events: recorded_events.clone(),
        });

        let concept_kernel =
            SparseKernel::from_dense(vec![vec![0.7, 0.3], vec![0.2, 0.8]], 1e-6).unwrap();
        let semantics = SemanticBridge::from_dense(
            vec![vec![0.6, 0.4], vec![0.3, 0.7]],
            [(0usize, 0usize), (1, 1)],
            1e-6,
            concept_kernel,
        )
        .unwrap();
        let bridge = MaxwellDesireBridge::new()
            .with_channel(
                seq.canonical_domain_concept().label(),
                vec![(0, 0.5), (1, 0.5)],
            )
            .unwrap();

        let x = Tensor::from_vec(1, 128, vec![0.1; 128]).unwrap();
        seq.forward_with_language_bridges(&x, &semantics, &bridge)
            .unwrap();

        let events = recorded_events.lock().unwrap();
        assert!(events.len() == 8 || events.len() == 9);
        assert_eq!(events[0], "projected");
        assert_eq!(events[1], "coherence");
        assert_eq!(events[2], "aggregated");
        assert_eq!(events[3], "semantic_window");
        assert_eq!(events[4], "semantic_distribution");
        assert_eq!(events[5], "canonical_concept");
        let remaining = &events[6..];
        match remaining {
            ["maxwell_emitted", "semantic_fused", "language"] => {}
            ["semantic_fused", "language"] => {}
            _ => panic!("unexpected plugin ordering: {:?}", remaining),
        }
    }

    #[test]
    fn plugins_observe_backend_and_profiles() {
        struct RecordingPlugin {
            events: Arc<Mutex<Vec<&'static str>>>,
        }

        impl ZSpaceSequencerPlugin for RecordingPlugin {
            fn name(&self) -> &'static str {
                "recording"
            }

            fn on_stage(&self, stage: ZSpaceSequencerStage<'_>) -> PureResult<()> {
                let label = match stage {
                    ZSpaceSequencerStage::BackendConfigured { .. } => "backend",
                    ZSpaceSequencerStage::LinguisticProfileRegistered { .. } => {
                        "profile_registered"
                    }
                    ZSpaceSequencerStage::LinguisticProfilesCleared => "profiles_cleared",
                    _ => return Ok(()),
                };
                self.events.lock().unwrap().push(label);
                Ok(())
            }
        }

        let topos = OpenCartesianTopos::new(-0.75, 1e-5, 10.0, 256, 8192).unwrap();
        let mut seq = ZSpaceCoherenceSequencer::new(256, 8, -0.75, topos).unwrap();
        let recorded_events = Arc::new(Mutex::new(Vec::new()));
        seq.register_plugin(RecordingPlugin {
            events: recorded_events.clone(),
        });

        seq.set_backend(CoherenceBackend::Fftw).unwrap();
        seq.register_linguistic_profile(
            DomainLinguisticProfile::new(DomainConcept::Membrane).with_descriptor("membrane-test"),
        )
        .unwrap();
        seq.clear_linguistic_profiles().unwrap();

        let events = recorded_events.lock().unwrap();
        assert_eq!(events.len(), 3);
        assert_eq!(events[0], "backend");
        assert_eq!(events[1], "profile_registered");
        assert_eq!(events[2], "profiles_cleared");
    }

    #[test]
    fn plugins_observe_linguistic_descriptors() {
        struct RecordingPlugin {
            events: Arc<Mutex<Vec<&'static str>>>,
        }

        impl ZSpaceSequencerPlugin for RecordingPlugin {
            fn name(&self) -> &'static str {
                "recording"
            }

            fn on_stage(&self, stage: ZSpaceSequencerStage<'_>) -> PureResult<()> {
                let label = match stage {
                    ZSpaceSequencerStage::LinguisticContourEmitted { .. } => "contour",
                    ZSpaceSequencerStage::ChannelsDescribed { .. } => "channels",
                    _ => return Ok(()),
                };
                self.events.lock().unwrap().push(label);
                Ok(())
            }
        }

        let topos = OpenCartesianTopos::new(-0.55, 1e-5, 10.0, 256, 8192).unwrap();
        let mut seq = ZSpaceCoherenceSequencer::new(256, 8, -0.55, topos).unwrap();
        let recorded_events = Arc::new(Mutex::new(Vec::new()));
        seq.register_plugin(RecordingPlugin {
            events: recorded_events.clone(),
        });

        let data = Tensor::from_vec(2, 256, vec![0.03; 512]).unwrap();
        let contour = seq.emit_linguistic_contour(&data).unwrap();
        assert!(contour.coherence_strength() >= 0.0);
        assert!(contour.timbre_spread() >= 0.0);
        let reports = seq.describe_channels(&data).unwrap();
        assert_eq!(reports.len(), seq.maxwell_channels());

        let events = recorded_events.lock().unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0], "contour");
        assert_eq!(events[1], "channels");
    }

    #[cfg(feature = "psi")]
    #[test]
    fn language_bridges_publish_psi_telemetry() {
        use st_core::telemetry::{hub, psi::PsiComponent};
        use st_core::theory::maxwell::MaxwellPsiTelemetryBridge;

        struct RecordingPlugin {
            events: Arc<Mutex<Vec<&'static str>>>,
        }

        impl ZSpaceSequencerPlugin for RecordingPlugin {
            fn name(&self) -> &'static str {
                "recording"
            }

            fn on_stage(&self, stage: ZSpaceSequencerStage<'_>) -> PureResult<()> {
                if let ZSpaceSequencerStage::PsiTelemetryPublished { .. } = stage {
                    self.events.lock().unwrap().push("psi");
                }
                Ok(())
            }
        }

        let _guard = hub::psi_telemetry_guard();
        hub::clear_last_psi();
        hub::clear_last_psi_events();
        hub::clear_softlogic_z();

        let topos = OpenCartesianTopos::new(-1.0, 1e-5, 10.0, 256, 8192).unwrap();
        let mut seq = ZSpaceCoherenceSequencer::new(128, 8, -1.0, topos).unwrap();
        let recorded_events = Arc::new(Mutex::new(Vec::new()));
        seq.register_plugin(RecordingPlugin {
            events: recorded_events.clone(),
        });

        let concept_kernel =
            SparseKernel::from_dense(vec![vec![0.7, 0.3], vec![0.2, 0.8]], 1e-6).unwrap();
        let semantics = SemanticBridge::from_dense(
            vec![vec![0.6, 0.4], vec![0.3, 0.7]],
            [(0usize, 0usize), (1, 1)],
            1e-6,
            concept_kernel,
        )
        .unwrap();
        let mut bridge = MaxwellDesireBridge::new()
            .with_smoothing(0.05)
            .with_magnitude_floor(0.05)
            .with_channel(
                seq.canonical_domain_concept().label(),
                vec![(0, 0.55), (1, 0.45)],
            )
            .unwrap();
        bridge
            .set_narrative_gain(seq.canonical_domain_concept().label(), 1.2)
            .unwrap();

        let psi_bridge = MaxwellPsiTelemetryBridge::new()
            .with_psi_gain(1.25)
            .with_loss_gain(0.5)
            .with_band_threshold(0.0);

        let mut data = vec![0.02f32; 128 * 2];
        for (idx, value) in data.iter_mut().enumerate() {
            *value += ((idx % 64) as f32) * 0.01;
        }
        let x = Tensor::from_vec(2, 128, data).unwrap();

        let (
            _aggregated,
            coherence,
            _concept_hint,
            _narrative,
            pulse,
            reading,
            emitted_events,
            feedback,
        ) = seq
            .forward_with_language_and_psi(&x, &semantics, &bridge, &psi_bridge, 64)
            .unwrap();

        assert_eq!(coherence.len(), seq.maxwell_channels());
        let reading = reading.expect("psi reading should be available");
        assert_eq!(reading.step, 64);
        assert!(reading.total >= 0.0);
        assert!(reading.breakdown.get(&PsiComponent::BAND_ENERGY).is_some());
        assert!(emitted_events.is_empty());

        let stored_reading = hub::get_last_psi().unwrap();
        assert_eq!(stored_reading.step, reading.step);
        let stored_feedback = hub::get_softlogic_z().unwrap();
        assert!((stored_feedback.psi_total - feedback.psi_total).abs() <= 1e-6);
        assert!(feedback.weighted_loss >= 0.0);
        assert!(pulse.band_energy.0 >= 0.0);

        let events = recorded_events.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0], "psi");
    }
}
