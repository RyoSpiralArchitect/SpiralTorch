// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! Maxwell pulse-based coherence measurement for Z-space sequences.

use crate::PureResult;
use st_tensor::{Tensor, TensorError};

/// Linguistic concept used to bias coherence weighting towards external domains.
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

/// Domain linguistic profile that biases coherence weighting towards specific
/// harmonic structures.
#[derive(Clone, Debug)]
pub struct DomainLinguisticProfile {
    concept: DomainConcept,
    emphasis: f32,
    harmonic_bias: Option<Vec<f32>>,
    descriptor: Option<String>,
}

impl DomainLinguisticProfile {
    /// Creates a new linguistic profile associated with a concept.
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

/// Linguistic bias profile used to sculpt coherence contours across tokens.
#[derive(Clone, Debug)]
pub struct DomainLinguisticProfile {
    concept: DomainConcept,
    register: f32,
    cadence: Option<Vec<f32>>,
    inflection: f32,
}

impl DomainLinguisticProfile {
    /// Creates a new linguistic profile bound to a semantic concept.
    pub fn new(concept: DomainConcept) -> Self {
        Self {
            concept,
            register: 1.0,
            cadence: None,
            inflection: 0.0,
        }
    }

    /// Adjusts the vocal register (amplitude scaling) for this profile.
    pub fn with_register(mut self, register: f32) -> PureResult<Self> {
        if register <= 0.0 || !register.is_finite() {
            return Err(TensorError::NonPositiveCoherence {
                coherence: register,
            }
            .into());
        }
        self.register = register;
        Ok(self)
    }

    /// Provides a cadence envelope applied across emitted contours.
    pub fn with_cadence(mut self, cadence: Vec<f32>) -> PureResult<Self> {
        if cadence.is_empty() {
            return Err(TensorError::EmptyInput("cadence").into());
        }
        if cadence
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(TensorError::NonPositiveCoherence {
                coherence: cadence
                    .iter()
                    .copied()
                    .find(|v| !v.is_finite() || *v <= 0.0)
                    .unwrap_or(0.0),
            }
            .into());
        }
        self.cadence = Some(cadence);
        Ok(self)
    }

    /// Tunes the melodic inflection applied after cadence scaling.
    pub fn with_inflection(mut self, inflection: f32) -> PureResult<Self> {
        if !inflection.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "inflection",
                value: inflection,
            }
            .into());
        }
        self.inflection = inflection.clamp(-2.0, 2.0);
        Ok(self)
    }

    /// Returns the underlying concept driving the profile.
    pub fn concept(&self) -> &DomainConcept {
        &self.concept
    }

    /// Exposes the configured cadence when present.
    pub fn cadence(&self) -> Option<&[f32]> {
        self.cadence.as_deref()
    }

    /// Returns the configured register multiplier.
    pub fn register(&self) -> f32 {
        self.register
    }

    fn envelope_weight(&self, token_idx: usize, total_tokens: usize, num_channels: usize) -> f32 {
        let cadence = self
            .cadence
            .as_ref()
            .and_then(|envelope| envelope.get(token_idx))
            .copied()
            .or_else(|| {
                self.cadence
                    .as_ref()
                    .and_then(|envelope| envelope.last().copied())
            })
            .unwrap_or(1.0)
            .max(1e-6);

        let harmonic = if total_tokens > 0 {
            (token_idx as f32 + 0.5) / total_tokens as f32
        } else {
            0.5
        };
        let channel_ratio = if num_channels > 0 {
            (token_idx as f32 + 1.0) / num_channels as f32
        } else {
            1.0
        };

        let base = match self.concept {
            DomainConcept::Membrane => 1.0 + 0.2 * (1.0 - harmonic),
            DomainConcept::GrainBoundary => 0.9 + 0.3 * harmonic,
            DomainConcept::NeuronalPattern => 0.8 + 0.4 * (harmonic - 0.5).abs(),
            DomainConcept::DropletCoalescence => 0.95 + 0.25 * harmonic * (1.0 - harmonic),
            DomainConcept::Custom(_) => 1.0,
        };

        let inflection = 1.0 + self.inflection * (channel_ratio - 0.5);
        (base * cadence * self.register * inflection).max(1e-6)
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
    linguistic_profiles: Vec<DomainLinguisticProfile>,
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
            linguistic_profiles: Vec::new(),
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

    /// Registers a new linguistic profile.
    pub fn register_linguistic_profile(&mut self, profile: DomainLinguisticProfile) {
        self.linguistic_profiles.push(profile);
    }

    /// Clears all linguistic profiles.
    pub fn clear_linguistic_profiles(&mut self) {
        self.linguistic_profiles.clear();
    }

    /// Exposes the registered linguistic profiles.
    pub fn linguistic_profiles(&self) -> &[DomainLinguisticProfile] {
        &self.linguistic_profiles
    }

    /// Derives a linguistic contour from the provided coherence weights.
    pub fn derive_linguistic_contour(&self, weights: &[f32]) -> PureResult<LinguisticContour> {
        if weights.is_empty() {
            return Err(TensorError::EmptyInput("linguistic_contour_weights").into());
        }
        if weights.len() != self.num_channels {
            return Err(TensorError::DataLength {
                expected: self.num_channels,
                got: weights.len(),
            }
            .into());
        }

        let mut total = 0.0f32;
        for weight in weights {
            if !weight.is_finite() || *weight < 0.0 {
                return Err(TensorError::NonPositiveCoherence { coherence: *weight }.into());
            }
            total += *weight;
        }
        if total <= 0.0 || !total.is_finite() {
            return Err(TensorError::NonPositiveCoherence { coherence: total }.into());
        }

        let norm_factor = 1.0 / total;
        let mut concentration = 0.0f32;
        let mut centroid = 0.0f32;
        for (idx, weight) in weights.iter().enumerate() {
            let normalized = *weight * norm_factor;
            concentration += normalized * normalized;
            centroid += normalized * idx as f32;
        }
        let max_idx = (self.num_channels - 1).max(1) as f32;
        let prosody = if self.num_channels > 1 {
            (centroid / max_idx).clamp(0.0, 1.0)
        } else {
            0.5
        };

        let mut variance = 0.0f32;
        for (idx, weight) in weights.iter().enumerate() {
            let normalized = *weight * norm_factor;
            let distance = idx as f32 - centroid;
            variance += normalized * distance * distance;
        }
        let timbre_spread = (variance / (self.num_channels as f32).max(1.0)).sqrt();

        let articulation_bias = (concentration * self.curvature.abs().sqrt()).clamp(0.0, 4.0);

        Ok(LinguisticContour {
            coherence_strength: concentration,
            articulation_bias,
            prosody_index: prosody,
            timbre_spread,
        })
    }

    /// Registers a new linguistic contour profile.
    pub fn register_linguistic_profile(&mut self, profile: DomainLinguisticProfile) {
        self.linguistic_profiles.push(profile);
    }

    /// Removes any registered linguistic profiles.
    pub fn clear_linguistic_profiles(&mut self) {
        self.linguistic_profiles.clear();
    }

    /// Returns the currently registered linguistic profiles.
    pub fn linguistic_profiles(&self) -> &[DomainLinguisticProfile] {
        &self.linguistic_profiles
    }

    /// Synthesises a linguistic contour given an input envelope.
    pub fn emit_linguistic_contour(&self, envelope: &[f32]) -> PureResult<Vec<f32>> {
        if envelope.is_empty() {
            return Err(TensorError::EmptyInput("linguistic_envelope").into());
        }
        if self.linguistic_profiles.is_empty() {
            return Ok(envelope.to_vec());
        }

        let total_tokens = envelope.len();
        let mut weights = vec![1.0f32; total_tokens];
        for profile in &self.linguistic_profiles {
            for (idx, weight) in weights.iter_mut().enumerate() {
                *weight *= profile.envelope_weight(idx, total_tokens, self.num_channels);
            }
        }

        let normalisation = (weights.iter().sum::<f32>() / total_tokens as f32).max(1e-6);
        let mut contour = Vec::with_capacity(total_tokens);
        for (value, weight) in envelope.iter().zip(weights.into_iter()) {
            contour.push(value * (weight / normalisation));
        }
        Ok(contour)
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
            let mut linguistic_bias = 1.0f32;
            for profile in &self.linguistic_profiles {
                linguistic_bias *= profile.harmonic_multiplier(channel, self.num_channels);
            }
            weight = (weight * linguistic_bias * curvature_bias * backend_bias).max(1e-6);
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
    fn linguistic_profile_biases_low_frequencies() {
        let mut engine = CoherenceEngine::new(128, -1.2).unwrap();
        let tensor = Tensor::from_vec(1, 128, vec![0.2; 128]).unwrap();
        let baseline = engine.measure_phases(&tensor).unwrap();
        let profile = DomainLinguisticProfile::new(DomainConcept::Membrane)
            .with_emphasis(1.4)
            .unwrap();
        engine.register_linguistic_profile(profile);
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

    #[test]
    fn linguistic_profile_envelope_weight_is_positive() {
        let profile = DomainLinguisticProfile::new(DomainConcept::Membrane)
            .with_register(1.2)
            .unwrap()
            .with_cadence(vec![0.8, 1.1, 1.3])
            .unwrap()
            .with_inflection(0.4)
            .unwrap();

        for idx in 0..6 {
            let weight = profile.envelope_weight(idx, 6, 4);
            assert!(weight.is_finite());
            assert!(weight > 0.0);
        }
    }

    #[test]
    fn emit_linguistic_contour_scales_envelope() {
        let mut engine = CoherenceEngine::new(128, -1.0).unwrap();
        engine.register_linguistic_profile(
            DomainLinguisticProfile::new(DomainConcept::NeuronalPattern)
                .with_register(1.5)
                .unwrap()
                .with_cadence(vec![1.0, 2.0, 1.0, 0.5])
                .unwrap(),
        );
        let contour = engine
            .emit_linguistic_contour(&[1.0, 1.0, 1.0, 1.0])
            .unwrap();
        assert_eq!(contour.len(), 4);
        let peak = contour.iter().copied().fold(f32::MIN, |a, b| a.max(b));
        let trough = contour.iter().copied().fold(f32::MAX, |a, b| a.min(b));
        assert!(peak > trough);
    }
}
