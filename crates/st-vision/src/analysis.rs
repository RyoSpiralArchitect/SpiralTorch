// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Vision-centric helpers for microlocal ⇄ macrolocal persistence analysis.
//!
//! The routines here reshape `Tensor` feature maps into the semantic scale stack
//! pipeline so that Z-space imaging stacks can interrogate how meaning breaks
//! across spatial scales.

use ndarray::{ArrayViewD, IxDyn};
use ordered_float::OrderedFloat;
use st_frac::scale_stack::{ScaleSample, ScaleStack, ScaleStackError, SemanticMetric};
use st_tensor::Tensor;

/// Builds a [`ScaleStack`] from a flattened `(H·W) × C` feature tensor.
///
/// The tensor is interpreted in row-major order where consecutive rows span the
/// scanline of the image. Channels are treated as semantic embeddings and
/// compared using the selected metric while probing spatial neighbourhoods.
pub fn feature_map_scale_stack(
    feature_map: &Tensor,
    height: usize,
    width: usize,
    channels: usize,
    scales: &[f64],
    threshold: f32,
    metric: SemanticMetric,
) -> Result<ScaleStack, ScaleStackError> {
    let (rows, cols) = feature_map.shape();
    if rows != height * width || cols != channels {
        return Err(ScaleStackError::ShapeMismatch {
            expected: vec![height, width, channels],
            actual: vec![rows, cols],
        });
    }
    let shape = IxDyn(&[height, width, channels]);
    let view = ArrayViewD::from_shape(shape, feature_map.data()).map_err(|_| {
        ScaleStackError::ShapeMismatch {
            expected: vec![height, width, channels],
            actual: vec![rows, cols],
        }
    })?;
    ScaleStack::from_semantic_field(view, scales, threshold, 2, metric)
}

/// Summary statistics describing semantic persistence across the provided scale stack.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ScaleStackProfile {
    /// Scale whose gate response was maximal across the stack.
    pub dominant_scale: f64,
    /// Gate amplitude recorded at the dominant scale.
    pub dominant_gate: f32,
    /// Mean gate amplitude across all sampled scales.
    pub mean_gate: f32,
    /// Normalised Shannon entropy of the gate distribution (0 → concentrated).
    pub gate_entropy: f32,
    /// Balance score describing how evenly semantic energy was distributed (0 → peaked).
    pub multiscale_balance: f32,
}

impl ScaleStackProfile {
    fn from_samples(samples: &[(f64, f32)]) -> Self {
        if samples.is_empty() {
            return Self::default();
        }
        let mut dominant = samples[0];
        let mut total_gate = 0.0f32;
        let mut energy = 0.0f32;
        for &(scale, gate) in samples.iter() {
            if gate >= dominant.1 {
                dominant = (scale, gate);
            }
            let gate = gate.max(0.0);
            total_gate += gate;
            energy += gate * gate;
        }

        let mean_gate = total_gate / samples.len() as f32;
        let mut entropy = 0.0f32;
        if total_gate > f32::EPSILON {
            for &(_, gate) in samples.iter() {
                let weight = gate.max(0.0) / total_gate;
                if weight > f32::EPSILON {
                    entropy -= weight * (weight + f32::EPSILON).ln();
                }
            }
            // Normalise entropy by the maximum possible value (ln N).
            let max_entropy = (samples.len() as f32 + f32::EPSILON).ln();
            if max_entropy > f32::EPSILON {
                entropy /= max_entropy;
            }
        }

        let mut balance = 0.0f32;
        if total_gate > f32::EPSILON {
            let concentration = energy / (total_gate * total_gate + f32::EPSILON);
            balance = (1.0 - concentration).clamp(0.0, 1.0);
        }

        ScaleStackProfile {
            dominant_scale: dominant.0,
            dominant_gate: dominant.1,
            mean_gate,
            gate_entropy: entropy,
            multiscale_balance: balance,
        }
    }
}

/// Computes a [`ScaleStackProfile`] by inspecting the stack's aggregated samples.
pub fn scale_stack_profile(stack: &ScaleStack) -> ScaleStackProfile {
    let samples: Vec<(f64, f32)> = stack
        .samples()
        .iter()
        .map(|sample| (sample.scale, sample.gate_mean as f32))
        .collect();
    ScaleStackProfile::from_samples(&samples)
}

/// Similarity score describing how closely two stacks align in their semantic gates.
///
/// Returns `None` when the stacks contain incompatible scale support and therefore
/// cannot be compared directly.
pub fn scale_stack_similarity(lhs: &ScaleStack, rhs: &ScaleStack) -> Option<f32> {
    let lhs_samples = lhs.samples();
    let rhs_samples = rhs.samples();
    if lhs_samples.is_empty() || rhs_samples.is_empty() {
        return None;
    }

    let mut accumulator = std::collections::BTreeMap::<OrderedFloat<f64>, (f32, f32)>::new();
    for sample in lhs_samples {
        accumulator
            .entry(OrderedFloat(sample.scale))
            .or_insert((0.0, 0.0))
            .0 = sample.gate_mean as f32;
    }
    for sample in rhs_samples {
        accumulator
            .entry(OrderedFloat(sample.scale))
            .or_insert((0.0, 0.0))
            .1 = sample.gate_mean as f32;
    }

    let mut lhs_vec = Vec::with_capacity(accumulator.len());
    let mut rhs_vec = Vec::with_capacity(accumulator.len());
    for (_, (lhs_value, rhs_value)) in accumulator.into_iter() {
        lhs_vec.push(lhs_value);
        rhs_vec.push(rhs_value);
    }

    if lhs_vec.is_empty() {
        return None;
    }

    let mut dot = 0.0f32;
    let mut lhs_norm = 0.0f32;
    let mut rhs_norm = 0.0f32;
    for (&l, &r) in lhs_vec.iter().zip(rhs_vec.iter()) {
        dot += l * r;
        lhs_norm += l * l;
        rhs_norm += r * r;
    }
    if lhs_norm <= f32::EPSILON || rhs_norm <= f32::EPSILON {
        return Some(0.0);
    }
    Some((dot / (lhs_norm.sqrt() * rhs_norm.sqrt())).clamp(-1.0, 1.0))
}

/// Aggregate profile across a collection of stacks capturing coherence and contrast.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ScaleStackSummary {
    /// Mean profile aggregated across all stacks.
    pub mean_profile: ScaleStackProfile,
    /// Mean cosine similarity between individual stacks.
    pub coherence: f32,
    /// Contrast between the strongest and weakest averaged scales.
    pub mean_contrast: f32,
}

fn normalised_gate_vector(samples: &[ScaleSample]) -> Vec<f32> {
    let mut vec = Vec::with_capacity(samples.len());
    for sample in samples {
        vec.push(sample.gate_mean as f32);
    }
    let mut sum = 0.0f32;
    for value in &vec {
        sum += value.max(0.0);
    }
    if sum <= f32::EPSILON {
        return vec;
    }
    for value in &mut vec {
        *value = value.max(0.0) / sum;
    }
    vec
}

/// Generates an aggregated [`ScaleStackSummary`] for the provided stack collection.
pub fn scale_stack_summary(stacks: &[ScaleStack]) -> ScaleStackSummary {
    if stacks.is_empty() {
        return ScaleStackSummary::default();
    }

    let mut gate_accumulator = std::collections::BTreeMap::<OrderedFloat<f64>, Vec<f32>>::new();
    for stack in stacks.iter() {
        for sample in stack.samples() {
            gate_accumulator
                .entry(OrderedFloat(sample.scale))
                .or_default()
                .push(sample.gate_mean as f32);
        }
    }

    let mut aggregated = Vec::with_capacity(gate_accumulator.len());
    for (scale, values) in gate_accumulator.iter() {
        let mut sum = 0.0f32;
        for value in values {
            sum += *value;
        }
        aggregated.push((scale.0, sum / values.len() as f32));
    }

    let mean_profile = ScaleStackProfile::from_samples(&aggregated);

    let mut coherence = 0.0f32;
    let mut comparisons = 0.0f32;
    for (i, lhs) in stacks.iter().enumerate() {
        for rhs in stacks.iter().skip(i + 1) {
            if let Some(similarity) = scale_stack_similarity(lhs, rhs) {
                coherence += similarity.max(-1.0).min(1.0);
                comparisons += 1.0;
            }
        }
    }
    if comparisons > 0.0 {
        coherence /= comparisons;
    }

    let mut normalised = Vec::new();
    for stack in stacks {
        normalised.push(normalised_gate_vector(stack.samples()));
    }
    let mut mean_contrast = 0.0f32;
    if !aggregated.is_empty() {
        let mut max_gate = f32::MIN;
        let mut min_gate = f32::MAX;
        for (_, gate) in &aggregated {
            max_gate = max_gate.max(*gate);
            min_gate = min_gate.min(*gate);
        }
        if min_gate.is_finite() && max_gate.is_finite() {
            mean_contrast = (max_gate - min_gate).abs();
        }
    }

    if !normalised.is_empty() {
        let length = normalised[0].len();
        let mut avg = vec![0.0f32; length];
        for vec in &normalised {
            if vec.len() != length {
                continue;
            }
            for (dst, src) in avg.iter_mut().zip(vec.iter()) {
                *dst += *src;
            }
        }
        for value in avg.iter_mut() {
            *value /= stacks.len() as f32;
        }
        let mut max = f32::MIN;
        let mut min = f32::MAX;
        for value in avg.iter() {
            max = max.max(*value);
            min = min.min(*value);
        }
        if max.is_finite() && min.is_finite() {
            mean_contrast = mean_contrast.max(max - min);
        }
    }

    ScaleStackSummary {
        mean_profile,
        coherence,
        mean_contrast,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use st_frac::scale_stack::InterfaceMode;

    #[test]
    fn feature_map_bridge_detects_boundary() {
        let height = 2;
        let width = 2;
        let channels = 2;
        let data = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let tensor = Tensor::from_vec(height * width, channels, data).unwrap();
        let stack = feature_map_scale_stack(
            &tensor,
            height,
            width,
            channels,
            &[1.0, 2.0],
            0.25,
            SemanticMetric::Euclidean,
        )
        .unwrap();
        assert!(matches!(stack.mode(), InterfaceMode::Semantic { .. }));
        assert!(stack.samples()[0].gate_mean > 0.0);
    }

    #[test]
    fn shape_mismatch_reported() {
        let tensor = Tensor::from_vec(2, 2, vec![0.0, 0.0, 0.0, 0.0]).unwrap();
        let err = feature_map_scale_stack(&tensor, 2, 2, 3, &[1.0], 0.1, SemanticMetric::Euclidean)
            .unwrap_err();
        assert!(matches!(err, ScaleStackError::ShapeMismatch { .. }));
    }

    #[test]
    fn profile_extracts_dominant_scale() {
        let height = 2;
        let width = 2;
        let channels = 1;
        let data = vec![0.2, 0.2, 0.8, 0.9];
        let tensor = Tensor::from_vec(height * width, channels, data).unwrap();
        let stack = feature_map_scale_stack(
            &tensor,
            height,
            width,
            channels,
            &[0.5, 1.0, 2.0],
            0.1,
            SemanticMetric::Euclidean,
        )
        .unwrap();
        let profile = scale_stack_profile(&stack);
        assert!(profile.dominant_scale >= 0.5);
        assert!(profile.dominant_gate >= 0.0);
        assert!(profile.gate_entropy >= 0.0);
        assert!(profile.multiscale_balance >= 0.0);
    }

    #[test]
    fn similarity_detects_alignment() {
        let tensor_a = Tensor::from_vec(4, 1, vec![0.1, 0.2, 0.9, 0.7]).unwrap();
        let tensor_b = Tensor::from_vec(4, 1, vec![0.2, 0.3, 0.8, 0.6]).unwrap();
        let stack_a = feature_map_scale_stack(
            &tensor_a,
            2,
            2,
            1,
            &[1.0, 2.0, 4.0],
            0.05,
            SemanticMetric::Cosine,
        )
        .unwrap();
        let stack_b = feature_map_scale_stack(
            &tensor_b,
            2,
            2,
            1,
            &[1.0, 2.0, 4.0],
            0.05,
            SemanticMetric::Cosine,
        )
        .unwrap();
        let similarity = scale_stack_similarity(&stack_a, &stack_b).unwrap();
        assert!(similarity <= 1.0 && similarity >= -1.0);
        let summary = scale_stack_summary(&[stack_a, stack_b]);
        assert!(summary.mean_profile.dominant_gate >= 0.0);
        assert!(summary.coherence >= -1.0 && summary.coherence <= 1.0);
        assert!(summary.mean_contrast >= 0.0);
    }
}
