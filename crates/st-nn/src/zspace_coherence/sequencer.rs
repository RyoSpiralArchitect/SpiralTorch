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
    telemetry::{
        hub::{self, SoftlogicZFeedback},
        psi::{PsiEvent, PsiReading},
    },
    theory::maxwell::MaxwellPsiTelemetryBridge,
};
use st_tensor::{OpenCartesianTopos, TensorError};

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

    /// Destructures the diagnostics into their core components.
    pub fn into_parts(self) -> (Tensor, Vec<f32>, Vec<LinguisticChannelReport>) {
        (self.aggregated, self.coherence, self.channel_reports)
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
}

impl ZSpaceCoherenceSequencer {
    /// Creates a new Z-space coherence sequencer.
    pub fn new(
        dim: usize,
        num_heads: usize,
        curvature: f32,
        topos: OpenCartesianTopos,
    ) -> PureResult<Self> {
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

        Ok(Self {
            dim,
            num_heads,
            curvature,
            coherence_engine: CoherenceEngine::new(dim, curvature)?,
            topos,
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
    ) -> PureResult<(Tensor, CoherenceDiagnostics)> {
        let (
            aggregated,
            normalization,
            fractional_order,
            channel_width,
        ) = self.compute_geometric_aggregate(x, coherence_weights)?;

        let diagnostics = self.build_coherence_diagnostics(
            &aggregated,
            coherence_weights,
            channel_width,
            normalization,
            fractional_order,
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

        Ok((aggregated, normalization, fractional_order, channel_width.max(1)))
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
        }
    }

    pub fn forward_with_diagnostics(
        &self,
        x: &Tensor,
    ) -> PureResult<(Tensor, Vec<f32>, CoherenceDiagnostics)> {
        let _ = self.topos.curvature();
        // Step 1: Project to Z-space
        let z_space = self.project_to_zspace(x)?;

        // Step 2: Measure Maxwell coherence
        let coherence = self.measure_coherence(&z_space)?;

        // Step 3: Geometric aggregation (replaces attention) with diagnostics
        let (aggregated, diagnostics) =
            self.geometric_aggregate_with_diagnostics(&z_space, &coherence)?;

        let channel_reports = self.coherence_engine.describe_channels(&coherence)?;
        let diagnostics = diagnostics.with_channel_reports(channel_reports);

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
            self.derive_semantic_distribution(&aggregated, &coherence, semantics);
        let pulse = self.summarise_maxwell_pulse(&aggregated, &coherence);
        let canonical_concept = self.canonical_domain_concept();
        let channel = canonical_concept.label();

        let (concept_hint, narrative) = if let Some((hint, narrative)) =
            maxwell_bridge.emit(channel, &pulse)
        {
            let fused =
                Self::fuse_distributions(&semantic_distribution, &hint.as_distribution(semantics));
            (ConceptHint::Distribution(fused), narrative)
        } else {
            (ConceptHint::Distribution(semantic_distribution), None)
        };

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
        let feedback = psi_bridge.publish(&pulse, psi_step);
        let psi_reading = hub::get_last_psi();
        let psi_events = hub::get_last_psi_events();

        Ok((
            aggregated,
            coherence,
            concept_hint,
            narrative,
            pulse,
            psi_reading,
            psi_events,
            feedback,
        ))
    }

    /// Configures the execution backend for coherence measurement.
    pub fn set_backend(&mut self, backend: CoherenceBackend) {
        self.coherence_engine.set_backend(backend);
    }

    /// Registers a domain linguistic profile used to bias coherence weights.
    pub fn register_linguistic_profile(&mut self, profile: DomainLinguisticProfile) {
        self.coherence_engine.register_linguistic_profile(profile);
    }

    /// Removes all linguistic profiles from the underlying coherence engine.
    pub fn clear_linguistic_profiles(&mut self) {
        self.coherence_engine.clear_linguistic_profiles();
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
        let coherence = self.measure_coherence(x)?;
        self.coherence_engine.derive_linguistic_contour(&coherence)
    }

    /// Describes each coherence channel, surfacing dominant linguistic concepts per band.
    pub fn describe_channels(&self, x: &Tensor) -> PureResult<Vec<LinguisticChannelReport>> {
        let coherence = self.measure_coherence(x)?;
        self.coherence_engine.describe_channels(&coherence)
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
    ) -> Vec<f32> {
        let window = self.derive_semantic_window(aggregated, coherence, semantics.vocab_size());
        if window.is_empty() {
            let concepts = semantics.concept_count().max(1);
            vec![1.0 / concepts as f32; concepts]
        } else {
            semantics.infer_from_window(&window, 1e-6)
        }
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
        );
        assert_eq!(seq.linguistic_profiles().len(), 1);
        seq.clear_linguistic_profiles();
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
        );
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

    #[cfg(feature = "psi")]
    #[test]
    fn language_bridges_publish_psi_telemetry() {
        use st_core::telemetry::{hub, psi::PsiComponent};
        use st_core::theory::maxwell::MaxwellPsiTelemetryBridge;

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

        let psi_bridge = MaxwellPsiTelemetryBridge::new()
            .with_psi_gain(1.25)
            .with_loss_gain(0.5)
            .with_band_threshold(0.0);

        let mut data = vec![0.02f32; 128 * 2];
        for (idx, value) in data.iter_mut().enumerate() {
            *value += ((idx % 64) as f32) * 0.01;
        }
        let x = Tensor::from_vec(2, 128, data).unwrap();

        let (_aggregated, coherence, _concept_hint, _narrative, pulse, reading, events, feedback) =
            seq.forward_with_language_and_psi(&x, &semantics, &bridge, &psi_bridge, 64)
                .unwrap();

        assert_eq!(coherence.len(), seq.maxwell_channels());
        let reading = reading.expect("psi reading should be available");
        assert_eq!(reading.step, 64);
        assert!(reading.total >= 0.0);
        assert!(reading.breakdown.get(&PsiComponent::BAND_ENERGY).is_some());
        assert!(events.is_empty());

        let stored_reading = hub::get_last_psi().unwrap();
        assert_eq!(stored_reading.step, reading.step);
        let stored_feedback = hub::get_softlogic_z().unwrap();
        assert!((stored_feedback.psi_total - feedback.psi_total).abs() <= 1e-6);
        assert!(feedback.weighted_loss >= 0.0);
        assert!(pulse.band_energy.0 >= 0.0);
    }
}
