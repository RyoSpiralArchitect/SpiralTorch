// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Text-native helpers that expose the scale stack to semantic embeddings.

use ndarray::{ArrayViewD, IxDyn};
use st_frac::scale_stack::{ScaleStack, ScaleStackError, SemanticMetric};
use st_tensor::Tensor;

/// Builds a [`ScaleStack`] from token embeddings arranged as `T × D`.
///
/// Each token is treated as a spatial sample along the sequence while the
/// embedding dimension acts as the semantic feature axis.
pub fn token_scale_stack(
    embeddings: &Tensor,
    scales: &[f64],
    threshold: f32,
    metric: SemanticMetric,
) -> Result<ScaleStack, ScaleStackError> {
    let (tokens, dims) = embeddings.shape();
    let shape = IxDyn(&[tokens, dims]);
    let view = ArrayViewD::from_shape(shape, embeddings.data()).map_err(|_| {
        ScaleStackError::ShapeMismatch {
            expected: vec![tokens, dims],
            actual: vec![tokens, dims],
        }
    })?;
    ScaleStack::from_semantic_field(view, scales, threshold, 1, metric)
}

/// Convenience helper returning the coherence breakpoints for the requested levels.
pub fn token_coherence_levels(
    embeddings: &Tensor,
    scales: &[f64],
    threshold: f32,
    metric: SemanticMetric,
    levels: &[f64],
) -> Result<Vec<Option<f64>>, ScaleStackError> {
    let stack = token_scale_stack(embeddings, scales, threshold, metric)?;
    Ok(stack.coherence_profile(levels))
}

#[cfg(test)]
mod tests {
    use super::*;
    use st_frac::scale_stack::InterfaceMode;

    #[test]
    fn token_stack_flags_semantic_boundary() {
        let tensor = Tensor::from_vec(4, 2, vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]).unwrap();
        let stack =
            token_scale_stack(&tensor, &[1.0, 2.0], 0.25, SemanticMetric::Euclidean).unwrap();
        assert!(matches!(stack.mode(), InterfaceMode::Semantic { .. }));
        assert!(stack.samples()[0].gate_mean > 0.0);
    }

    #[test]
    fn token_coherence_profile_returns_levels() {
        let tensor = Tensor::from_vec(4, 2, vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]).unwrap();
        let levels = token_coherence_levels(
            &tensor,
            &[1.0, 2.0, 3.0],
            0.25,
            SemanticMetric::Euclidean,
            &[0.25, 0.5, 0.75],
        )
        .unwrap();
        assert_eq!(levels.len(), 3);
    }
}
