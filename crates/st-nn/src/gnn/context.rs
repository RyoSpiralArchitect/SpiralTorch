// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::{PureResult, Tensor, TensorError};
use st_core::telemetry::xai::NodeFlowSample;

/// Normalised adjacency wrapper that keeps Z-space compatible graph propagation.
#[derive(Debug, Clone)]
pub struct GraphContext {
    norm_adjacency: Tensor,
    norm_adjacency_t: Tensor,
    row_sums: Vec<f32>,
}

impl GraphContext {
    /// Builds a new context from a raw adjacency matrix. Self-loops of weight 1
    /// are added automatically before normalisation.
    pub fn from_adjacency(adjacency: Tensor) -> PureResult<Self> {
        Self::with_self_loops(adjacency, 1.0)
    }

    /// Builds a context from an adjacency matrix and an explicit self-loop
    /// weight. The adjacency must be square.
    pub fn with_self_loops(adjacency: Tensor, self_loop_weight: f32) -> PureResult<Self> {
        let (rows, cols) = adjacency.shape();
        if rows != cols {
            return Err(TensorError::ShapeMismatch {
                left: (rows, cols),
                right: (cols, rows),
            });
        }
        if rows == 0 {
            return Err(TensorError::EmptyInput("graph_context"));
        }
        if self_loop_weight < 0.0 {
            return Err(TensorError::NonPositiveWeight {
                weight: self_loop_weight,
            });
        }

        let mut loop_entries = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                if r == c {
                    loop_entries.push(self_loop_weight);
                } else {
                    loop_entries.push(0.0);
                }
            }
        }
        let loops = Tensor::from_vec(rows, cols, loop_entries)?;
        let adjusted = adjacency.add(&loops)?;

        let degrees = adjusted.transpose().sum_axis0();
        let mut inv_sqrt = Vec::with_capacity(degrees.len());
        for degree in degrees {
            if degree <= f32::EPSILON {
                inv_sqrt.push(0.0);
            } else {
                inv_sqrt.push(1.0 / degree.sqrt());
            }
        }

        let adjusted_data = adjusted.data();
        let mut normalised = Vec::with_capacity(adjusted_data.len());
        for r in 0..rows {
            for c in 0..cols {
                let value = adjusted_data[r * cols + c];
                if value.abs() <= f32::EPSILON {
                    normalised.push(0.0);
                } else {
                    normalised.push(value * inv_sqrt[r] * inv_sqrt[c]);
                }
            }
        }

        let norm_adjacency = Tensor::from_vec(rows, cols, normalised)?;
        let norm_adjacency_t = norm_adjacency.transpose();
        let row_sums = norm_adjacency.transpose().sum_axis0();

        Ok(Self {
            norm_adjacency,
            norm_adjacency_t,
            row_sums,
        })
    }

    /// Returns the number of nodes represented by the context.
    pub fn node_count(&self) -> usize {
        self.row_sums.len()
    }

    /// Propagates features across the normalised adjacency matrix.
    pub fn propagate(&self, features: &Tensor) -> PureResult<Tensor> {
        self.validate_features(features)?;
        self.norm_adjacency.matmul(features)
    }

    /// Propagates features while emitting explainability artefacts.
    pub fn propagate_with_trace(
        &self,
        features: &Tensor,
    ) -> PureResult<(Tensor, Vec<NodeFlowSample>)> {
        let propagated = self.propagate(features)?;
        let (_, feature_dim) = propagated.shape();
        let mut flows = Vec::with_capacity(self.node_count());
        let data = propagated.data();
        for node in 0..self.node_count() {
            let start = node * feature_dim;
            let mut norm = 0.0f32;
            for value in &data[start..start + feature_dim] {
                norm += value.abs();
            }
            flows.push(NodeFlowSample {
                node_index: node,
                incoming_weight: self.row_sums[node],
                aggregated_norm: norm,
            });
        }
        Ok((propagated, flows))
    }

    /// Propagates a gradient back through the transposed adjacency.
    pub fn propagate_transpose(&self, features: &Tensor) -> PureResult<Tensor> {
        if features.shape().0 != self.node_count() {
            return Err(TensorError::ShapeMismatch {
                left: features.shape(),
                right: (self.node_count(), features.shape().1),
            });
        }
        self.norm_adjacency_t.matmul(features)
    }

    fn validate_features(&self, features: &Tensor) -> PureResult<()> {
        if features.shape().0 != self.node_count() {
            return Err(TensorError::ShapeMismatch {
                left: features.shape(),
                right: (self.node_count(), features.shape().1),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_normalises_adjacency() {
        let adjacency = Tensor::from_vec(2, 2, vec![0.0, 1.0, 1.0, 0.0]).unwrap();
        let context = GraphContext::from_adjacency(adjacency).unwrap();
        assert_eq!(context.node_count(), 2);
        let features = Tensor::from_vec(2, 1, vec![1.0, 2.0]).unwrap();
        let propagated = context.propagate(&features).unwrap();
        let data = propagated.data();
        assert_eq!(data.len(), 2);
        assert!(data[0] > 0.0);
        assert!(data[1] > 0.0);
    }

    #[test]
    fn context_propagate_with_trace_reports_nodes() {
        let adjacency =
            Tensor::from_vec(3, 3, vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]).unwrap();
        let context = GraphContext::from_adjacency(adjacency).unwrap();
        let features = Tensor::from_vec(3, 2, vec![1.0, 0.5, 0.0, 1.0, 1.5, -0.5]).unwrap();
        let (_prop, flows) = context.propagate_with_trace(&features).unwrap();
        assert_eq!(flows.len(), 3);
        assert_eq!(flows[0].node_index, 0);
    }
}
