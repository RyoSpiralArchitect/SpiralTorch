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
use st_frac::scale_stack::{ScaleStack, ScaleStackError, SemanticMetric};
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
}
