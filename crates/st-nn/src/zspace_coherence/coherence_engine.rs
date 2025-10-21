// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! Maxwell pulse-based coherence measurement for Z-space sequences.

use crate::PureResult;
use st_tensor::{Tensor, TensorError};

/// Semantic concept used to bias coherence weighting towards external domains.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum DomainConcept {
    Membrane,
    GrainBoundary,
    NeuronalPattern,
    DropletCoalescence,
    Custom(String),
}

/// Execution backend that the coherence engine should target when bridging to
/// external runtimes or accelerators.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CoherenceBackend {
    PureRust,
    Fftw,
    CuFft,
    Polars,
    Arrow,
    WebGpu,
    Custom(String),
}

impl Default for CoherenceBackend {
    fn default() -> Self {
        Self::PureRust
    }
}

impl CoherenceBackend {
    /// Returns a human readable label for the backend.
    pub fn label(&self) -> &str {
        match self {
            CoherenceBackend::PureRust => "pure_rust",
            CoherenceBackend::Fftw => "fftw",
            CoherenceBackend::CuFft => "cufft",
            CoherenceBackend::Polars => "polars",
            CoherenceBackend::Arrow => "arrow",
            CoherenceBackend::WebGpu => "webgpu",
            CoherenceBackend::Custom(label) => label.as_str(),
        }
    }

    /// Indicates whether the backend offloads work outside the pure Rust path.
    pub fn is_accelerated(&self) -> bool {
        matches!(
            self,
            CoherenceBackend::Fftw
                | CoherenceBackend::CuFft
                | CoherenceBackend::Polars
                | CoherenceBackend::Arrow
                | CoherenceBackend::WebGpu
        )
    }
}

/// Domain semantic profile that biases coherence weighting towards specific
/// harmonic structures.
#[derive(Clone, Debug)]
pub struct DomainSemanticProfile {
    concept: DomainConcept,
    emphasis: f32,
    harmonic_bias: Option<Vec<f32>>,
    descriptor: Option<String>,
}

impl DomainSemanticProfile {
    /// Creates a new semantic profile associated with a concept.
    pub fn new(concept: DomainConcept) -> Self {
        Self {
            concept,
            emphasis: 1.0,
            harmonic_bias: None,
            descriptor: None,
        }
    }

    /// Scales the overall influence of the profile.
    pub fn with_emphasis(mut self, emphasis: f32) -> PureResult<Self> {
        if emphasis <= 0.0 || !emphasis.is_finite() {
            return Err(TensorError::NonPositiveCoherence {
                coherence: emphasis,
            }
            .into());
        }
        self.emphasis = emphasis;
        Ok(self)
    }

    /// Overrides the harmonic curve applied to each channel. Values must be
    /// positive and finite.
    pub fn with_harmonic_bias(mut self, bias: Vec<f32>) -> PureResult<Self> {
        if bias.is_empty() {
            return Err(TensorError::EmptyInput("harmonic_bias").into());
        }
        for value in &bias {
            if *value <= 0.0 || !value.is_finite() {
                return Err(TensorError::NonPositiveCoherence { coherence: *value }.into());
            }
        }
        self.harmonic_bias = Some(bias);
        Ok(self)
    }

    /// Adds a human readable descriptor that can be surfaced through telemetry.
    pub fn with_descriptor(mut self, descriptor: impl Into<String>) -> Self {
        self.descriptor = Some(descriptor.into());
        self
    }

    /// Returns the descriptor when provided.
    pub fn descriptor(&self) -> Option<&str> {
        self.descriptor.as_deref()
    }

    /// Returns the associated concept.
    pub fn concept(&self) -> &DomainConcept {
        &self.concept
    }

    fn harmonic_multiplier(&self, channel_idx: usize, total_channels: usize) -> f32 {
        if let Some(ref bias) = self.harmonic_bias {
            if let Some(value) = bias.get(channel_idx) {
                return (*value * self.emphasis).max(1e-6);
            }
            if let Some(last) = bias.last() {
                return (*last * self.emphasis).max(1e-6);
            }
        }

        let total = total_channels.max(1) as f32;
        let harmonic = (channel_idx as f32 + 0.5) / total;
        let base = match self.concept {
            DomainConcept::Membrane => 1.15 - 0.35 * harmonic,
            DomainConcept::GrainBoundary => 0.85 + 0.65 * harmonic,
            DomainConcept::NeuronalPattern => 0.75 + 0.9 * (harmonic - 0.5).abs(),
            DomainConcept::DropletCoalescence => 0.95 + 0.55 * (1.0 - (2.0 * harmonic - 1.0).abs()),
            DomainConcept::Custom(_) => 1.0,
        };
        (base * self.emphasis).max(1e-6)
    }
}

/// Measures phase coherence using Maxwell pulses (instead of attention).
#[derive(Clone, Debug)]
pub struct CoherenceEngine {
    dim: usize,
    curvature: f32,
    num_channels: usize,
    backend: CoherenceBackend,
    semantic_profiles: Vec<DomainSemanticProfile>,
}

impl CoherenceEngine {
    /// Creates a new coherence engine.
    pub fn new(dim: usize, curvature: f32) -> PureResult<Self> {
        if dim == 0 {
            return Err(TensorError::InvalidDimensions { rows: 0, cols: 0 }.into());
        }
        if curvature >= 0.0 {
            return Err(TensorError::NonHyperbolicCurvature { curvature }.into());
        }
        Ok(Self {
            dim,
            curvature,
            num_channels: (dim / 64).max(1),
            backend: CoherenceBackend::default(),
            semantic_profiles: Vec::new(),
        })
    }

    /// Overrides the execution backend.
    pub fn set_backend(&mut self, backend: CoherenceBackend) {
        self.backend = backend;
    }

    /// Returns the configured backend.
    pub fn backend(&self) -> &CoherenceBackend {
        &self.backend
    }

    /// Registers a new semantic profile.
    pub fn register_domain_profile(&mut self, profile: DomainSemanticProfile) {
        self.semantic_profiles.push(profile);
    }

    /// Clears all semantic profiles.
    pub fn clear_domain_profiles(&mut self) {
        self.semantic_profiles.clear();
    }

    /// Exposes the registered semantic profiles.
    pub fn semantic_profiles(&self) -> &[DomainSemanticProfile] {
        &self.semantic_profiles
    }

    fn curvature_bias(&self) -> f32 {
        1.0 + self.curvature.abs().sqrt().min(4.0)
    }

    /// Measures phase synchronization across Maxwell channels.
    pub fn measure_phases(&self, x: &Tensor) -> PureResult<Vec<f32>> {
        let (rows, cols) = x.shape();
        if cols != self.dim {
            return Err(TensorError::ShapeMismatch {
                left: (rows, cols),
                right: (1, self.dim),
            }
            .into());
        }

        let data = x.data();
        let channel_width = (self.dim + self.num_channels - 1) / self.num_channels;
        let mut weights = Vec::with_capacity(self.num_channels);
        let curvature_bias = self.curvature_bias();
        let backend_bias = if self.backend.is_accelerated() {
            1.05
        } else {
            1.0
        };
        for channel in 0..self.num_channels {
            let start = channel * channel_width;
            let end = ((channel + 1) * channel_width).min(self.dim);
            if start >= end {
                weights.push(1e-6);
                continue;
            }
            let mut energy = 0.0f32;
            for row in 0..rows {
                let offset = row * cols;
                for value in &data[offset + start..offset + end] {
                    energy += value * value;
                }
            }
            let samples = (end - start) * rows;
            let mut weight = if samples > 0 {
                (energy / samples as f32).sqrt()
            } else {
                0.0
            };
            if !weight.is_finite() {
                weight = 0.0;
            }
            let mut semantic_bias = 1.0f32;
            for profile in &self.semantic_profiles {
                semantic_bias *= profile.harmonic_multiplier(channel, self.num_channels);
            }
            weight = (weight * semantic_bias * curvature_bias * backend_bias).max(1e-6);
            weights.push(weight);
        }

        let mut total: f32 = weights.iter().copied().sum();
        if !total.is_finite() || total <= f32::EPSILON {
            return Err(TensorError::NonPositiveCoherence { coherence: total }.into());
        }
        total = total.max(1e-6);
        for weight in &mut weights {
            *weight /= total;
        }

        Ok(weights)
    }

    /// Returns the curvature used for coherence computation.
    pub fn curvature(&self) -> f32 {
        self.curvature
    }

    /// Returns the number of Maxwell channels.
    pub fn num_channels(&self) -> usize {
        self.num_channels
    }

    /// Indicates whether the engine is configured for an accelerated backend.
    pub fn is_accelerated(&self) -> bool {
        self.backend.is_accelerated()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_sum_to_one_and_follow_energy() {
        let mut engine = CoherenceEngine::new(128, -1.0).unwrap();
        engine.set_backend(CoherenceBackend::PureRust);
        let tensor = Tensor::from_vec(
            1,
            128,
            vec![0.1; 64]
                .into_iter()
                .chain(vec![0.6; 64].into_iter())
                .collect(),
        )
        .unwrap();
        let weights = engine.measure_phases(&tensor).unwrap();
        assert_eq!(weights.len(), engine.num_channels());
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
        if engine.num_channels() > 1 {
            assert!(weights[1] > weights[0]);
        }
    }

    #[test]
    fn semantic_profile_biases_low_frequencies() {
        let mut engine = CoherenceEngine::new(128, -1.2).unwrap();
        let tensor = Tensor::from_vec(1, 128, vec![0.2; 128]).unwrap();
        let baseline = engine.measure_phases(&tensor).unwrap();
        let profile = DomainSemanticProfile::new(DomainConcept::Membrane)
            .with_emphasis(1.4)
            .unwrap();
        engine.register_domain_profile(profile);
        let biased = engine.measure_phases(&tensor).unwrap();
        if engine.num_channels() > 1 {
            assert!(biased[0] > baseline[0]);
        } else {
            assert!((biased[0] - baseline[0]).abs() < 1e-6);
        }
    }

    #[test]
    fn backend_reports_accelerated_state() {
        let mut engine = CoherenceEngine::new(64, -0.8).unwrap();
        assert!(!engine.is_accelerated());
        engine.set_backend(CoherenceBackend::Fftw);
        assert!(engine.is_accelerated());
        assert_eq!(engine.backend().label(), "fftw");
    }
}
