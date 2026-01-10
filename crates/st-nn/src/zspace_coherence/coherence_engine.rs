// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! Maxwell pulse-based coherence measurement for Z-space sequences.

use crate::PureResult;
use st_tensor::{Tensor, TensorError};
use std::fmt;

/// Linguistic concept used to bias coherence weighting towards external domains.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum DomainConcept {
    Membrane,
    GrainBoundary,
    NeuronalPattern,
    DropletCoalescence,
    Custom(String),
}

impl DomainConcept {
    /// Returns a stable human-readable label for the concept.
    pub fn label(&self) -> &str {
        match self {
            DomainConcept::Membrane => "membrane",
            DomainConcept::GrainBoundary => "grain_boundary",
            DomainConcept::NeuronalPattern => "neuronal_pattern",
            DomainConcept::DropletCoalescence => "droplet_coalescence",
            DomainConcept::Custom(label) => label.as_str(),
        }
    }
}

impl fmt::Display for DomainConcept {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.label())
    }
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

    /// Describes the capabilities unlocked by the backend for external bridges.
    pub fn capabilities(&self) -> BackendCapabilities {
        match self {
            CoherenceBackend::PureRust => BackendCapabilities {
                fft: false,
                gpu: false,
                timeseries: false,
                columnar: false,
            },
            CoherenceBackend::Fftw => BackendCapabilities {
                fft: true,
                gpu: false,
                timeseries: true,
                columnar: false,
            },
            CoherenceBackend::CuFft => BackendCapabilities {
                fft: true,
                gpu: true,
                timeseries: true,
                columnar: false,
            },
            CoherenceBackend::Polars | CoherenceBackend::Arrow => BackendCapabilities {
                fft: false,
                gpu: false,
                timeseries: true,
                columnar: true,
            },
            CoherenceBackend::WebGpu => BackendCapabilities {
                fft: true,
                gpu: true,
                timeseries: true,
                columnar: false,
            },
            CoherenceBackend::Custom(_) => BackendCapabilities {
                fft: false,
                gpu: false,
                timeseries: false,
                columnar: false,
            },
        }
    }
}

/// Capabilities exposed by a backend once the sequencer bridges out of pure Rust.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BackendCapabilities {
    pub fft: bool,
    pub gpu: bool,
    pub timeseries: bool,
    pub columnar: bool,
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
            });
        }
        self.emphasis = emphasis;
        Ok(self)
    }

    /// Overrides the harmonic curve applied to each channel. Values must be
    /// positive and finite.
    pub fn with_harmonic_bias(mut self, bias: Vec<f32>) -> PureResult<Self> {
        if bias.is_empty() {
            return Err(TensorError::EmptyInput("harmonic_bias"));
        }
        for value in &bias {
            if *value <= 0.0 || !value.is_finite() {
                return Err(TensorError::NonPositiveCoherence { coherence: *value });
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

/// Aggregated linguistic contour derived from coherence weights.
#[derive(Clone, Debug)]
pub struct LinguisticContour {
    coherence_strength: f32,
    articulation_bias: f32,
    prosody_index: f32,
    timbre_spread: f32,
}

impl LinguisticContour {
    /// Overall coherence concentration (1.0 = single channel dominance).
    pub fn coherence_strength(&self) -> f32 {
        self.coherence_strength
    }

    /// Estimated articulation drive induced by curvature-weighted coherence.
    pub fn articulation_bias(&self) -> f32 {
        self.articulation_bias
    }

    /// Normalised channel centroid capturing rising/falling prosody.
    pub fn prosody_index(&self) -> f32 {
        self.prosody_index
    }

    /// Dispersion of the harmonic emphasis, useful for timbre shaping.
    pub fn timbre_spread(&self) -> f32 {
        self.timbre_spread
    }
}

/// Per-channel report pairing the dominant linguistic concept with coherence weight.
#[derive(Clone, Debug)]
pub struct LinguisticChannelReport {
    channel: usize,
    weight: f32,
    backend: CoherenceBackend,
    dominant_concept: Option<DomainConcept>,
    emphasis: f32,
    descriptor: Option<String>,
}

impl LinguisticChannelReport {
    /// Index of the channel the report corresponds to.
    pub fn channel(&self) -> usize {
        self.channel
    }

    /// Normalised weight contributed by this channel.
    pub fn weight(&self) -> f32 {
        self.weight
    }

    /// Backend that produced the coherence sample.
    pub fn backend(&self) -> &CoherenceBackend {
        &self.backend
    }

    /// Dominant linguistic concept, when a profile biases the channel.
    pub fn dominant_concept(&self) -> Option<&DomainConcept> {
        self.dominant_concept.as_ref()
    }

    /// Effective emphasis value applied by the dominant concept or 1.0 when un-biased.
    pub fn emphasis(&self) -> f32 {
        self.emphasis
    }

    /// Optional descriptor carried from the dominant profile.
    pub fn descriptor(&self) -> Option<&str> {
        self.descriptor.as_deref()
    }
}

/// Measures phase coherence using Maxwell pulses (instead of attention).
#[derive(Clone, Debug)]
pub struct CoherenceEngine {
    dim: usize,
    curvature: f32,
    num_channels: usize,
    backend: CoherenceBackend,
    linguistic_profiles: Vec<DomainLinguisticProfile>,
}

impl CoherenceEngine {
    /// Creates a new coherence engine.
    pub fn new(dim: usize, curvature: f32) -> PureResult<Self> {
        if dim == 0 {
            return Err(TensorError::InvalidDimensions { rows: 0, cols: 0 });
        }
        if curvature >= 0.0 {
            return Err(TensorError::NonHyperbolicCurvature { curvature });
        }
        Ok(Self {
            dim,
            curvature,
            num_channels: (dim / 64).max(1),
            backend: CoherenceBackend::default(),
            linguistic_profiles: Vec::new(),
        })
    }

    /// Overrides the number of Maxwell channels used for coherence measurement.
    pub fn with_channel_count(mut self, num_channels: usize) -> PureResult<Self> {
        if num_channels == 0 {
            return Err(TensorError::EmptyInput("maxwell_channels"));
        }
        self.num_channels = num_channels;
        Ok(self)
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
            return Err(TensorError::EmptyInput("linguistic_contour_weights"));
        }
        if weights.len() != self.num_channels {
            return Err(TensorError::DataLength {
                expected: self.num_channels,
                got: weights.len(),
            });
        }

        let mut total = 0.0f32;
        for weight in weights {
            if !weight.is_finite() || *weight < 0.0 {
                return Err(TensorError::NonPositiveCoherence { coherence: *weight });
            }
            total += *weight;
        }
        if total <= 0.0 || !total.is_finite() {
            return Err(TensorError::NonPositiveCoherence { coherence: total });
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

    /// Builds per-channel reports so downstream bridges can consume coherence metadata.
    pub fn describe_channels(&self, weights: &[f32]) -> PureResult<Vec<LinguisticChannelReport>> {
        if weights.len() != self.num_channels {
            return Err(TensorError::DataLength {
                expected: self.num_channels,
                got: weights.len(),
            });
        }
        if weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
        {
            return Err(TensorError::NonPositiveCoherence { coherence: -1.0 });
        }

        let mut reports = Vec::with_capacity(self.num_channels);
        for (channel, &weight) in weights.iter().enumerate() {
            let mut dominant: Option<(DomainConcept, f32, Option<String>)> = None;
            for profile in &self.linguistic_profiles {
                let emphasis = profile.harmonic_multiplier(channel, self.num_channels);
                let descriptor = profile.descriptor().map(|desc| desc.to_owned());
                match dominant {
                    Some((_, current_emphasis, _)) if emphasis <= current_emphasis => {}
                    _ => {
                        dominant = Some((profile.concept().clone(), emphasis, descriptor));
                    }
                }
            }

            let (concept, emphasis, descriptor) = match dominant {
                Some((concept, emphasis, descriptor)) => (Some(concept), emphasis, descriptor),
                None => (None, 1.0f32, None),
            };

            reports.push(LinguisticChannelReport {
                channel,
                weight,
                backend: self.backend.clone(),
                dominant_concept: concept,
                emphasis,
                descriptor,
            });
        }

        Ok(reports)
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
            });
        }

        let data = x.data();
        let channel_width = self.dim.div_ceil(self.num_channels);
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
            return Err(TensorError::NonPositiveCoherence { coherence: total });
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
                .chain(vec![0.6; 64])
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
    fn linguistic_contour_derivation_reflects_weights() {
        let engine = CoherenceEngine::new(96, -1.0).unwrap();
        let tensor = Tensor::from_vec(
            1,
            96,
            vec![0.05; 48]
                .into_iter()
                .chain(vec![0.6; 48])
                .collect(),
        )
        .unwrap();
        let weights = engine.measure_phases(&tensor).unwrap();
        let contour = engine.derive_linguistic_contour(&weights).unwrap();

        assert!(contour.coherence_strength() > 0.0);
        assert!(contour.prosody_index() >= 0.5);
        assert!(contour.timbre_spread() >= 0.0);
    }

    #[test]
    fn backend_capabilities_match_flags() {
        let fftw = CoherenceBackend::Fftw.capabilities();
        assert!(fftw.fft);
        assert!(fftw.timeseries);
        assert!(!fftw.gpu);
        let webgpu = CoherenceBackend::WebGpu.capabilities();
        assert!(webgpu.gpu);
        assert!(webgpu.fft);
        let rust = CoherenceBackend::PureRust.capabilities();
        assert!(!rust.fft && !rust.gpu && !rust.timeseries && !rust.columnar);
    }

    #[test]
    fn describe_channels_reflects_profiles() {
        let mut engine = CoherenceEngine::new(128, -1.0).unwrap();
        let tensor = Tensor::from_vec(1, 128, vec![0.2; 128]).unwrap();
        let weights = engine.measure_phases(&tensor).unwrap();
        let reports = engine.describe_channels(&weights).unwrap();
        assert_eq!(reports.len(), engine.num_channels());
        for (idx, report) in reports.iter().enumerate() {
            assert_eq!(report.channel(), idx);
            assert_eq!(report.weight(), weights[idx]);
            assert!(report.emphasis() >= 1.0 - f32::EPSILON);
            assert!(!report.backend().label().is_empty());
        }

        engine.clear_linguistic_profiles();
        engine.register_linguistic_profile(
            DomainLinguisticProfile::new(DomainConcept::NeuronalPattern).with_descriptor("spike"),
        );
        let biased = engine.describe_channels(&weights).unwrap();
        assert_eq!(biased.len(), reports.len());
        if let Some(dominant) = biased[0].dominant_concept() {
            assert_eq!(dominant.label(), "neuronal_pattern");
            assert_eq!(biased[0].descriptor(), Some("spike"));
        }
    }
}
