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
use crate::{Module, PureResult, Tensor};
use st_tensor::{OpenCartesianTopos, TensorError};

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

    /// Performs coherence-weighted geometric aggregation.
    fn geometric_aggregate(&self, x: &Tensor, coherence_weights: &[f32]) -> PureResult<Tensor> {
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
        let channel_width = (self.dim + coherence_weights.len() - 1) / coherence_weights.len();
        let normalization = coherence_weights.iter().copied().sum::<f32>().max(1e-6);
        let canonical_concept = self.canonical_domain_concept();
        let fractional_order = match canonical_concept {
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
        };
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
        Ok(aggregated)
    }

    pub fn forward(&self, x: &Tensor) -> PureResult<Tensor> {
        let _ = self.topos.curvature();
        // Step 1: Project to Z-space
        let z_space = self.project_to_zspace(x)?;

        // Step 2: Measure Maxwell coherence
        let coherence = self.measure_coherence(&z_space)?;

        // Step 3: Geometric aggregation (replaces attention)
        let aggregated = self.geometric_aggregate(&z_space, &coherence)?;

        // Step 4: Output from Z-space (to Euclidean)
        Ok(aggregated)
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

    #[test]
    fn sequencer_forward_preserves_shape() {
        let topos = OpenCartesianTopos::new(-1.0, 1e-5, 10.0, 256, 8192).unwrap();
        let seq = ZSpaceCoherenceSequencer::new(768, 12, -1.0, topos).unwrap();

        let x = Tensor::from_vec(2, 768, vec![0.1; 768 * 2]).unwrap();
        let out = seq.forward(&x).unwrap();

        assert_eq!(out.shape(), x.shape());
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
}
