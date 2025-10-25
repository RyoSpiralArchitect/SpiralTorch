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
}
