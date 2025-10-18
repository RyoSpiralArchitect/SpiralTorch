// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::{PureResult, Tensor, TensorError};
use st_core::telemetry::xai::NodeFlowSample;

/// Strategy used to normalise the adjacency matrix prior to propagation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphNormalization {
    /// Symmetric normalisation `D^{-1/2} A D^{-1/2}` suitable for undirected graphs.
    Symmetric,
    /// Row-stochastic normalisation `D^{-1} A` that keeps outgoing weights summing to one.
    Row,
}

impl Default for GraphNormalization {
    fn default() -> Self {
        GraphNormalization::Symmetric
    }
}

/// Normalised adjacency wrapper that keeps Z-space compatible graph propagation.
#[derive(Debug, Clone)]
pub struct GraphContext {
    norm_adjacency: Tensor,
    norm_adjacency_t: Tensor,
    row_sums: Vec<f32>,
    normalisation: GraphNormalization,
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
        Self::builder(adjacency)
            .self_loop_weight(self_loop_weight)
            .build()
    }

    /// Builds a configurable context using a builder surface.
    pub fn builder(adjacency: Tensor) -> GraphContextBuilder {
        GraphContextBuilder::new(adjacency)
    }

    /// Returns the number of nodes represented by the context.
    pub fn node_count(&self) -> usize {
        self.row_sums.len()
    }

    /// Returns the normalisation strategy used when creating the context.
    pub fn normalisation(&self) -> GraphNormalization {
        self.normalisation
    }

    /// Returns the cached row sums of the normalised adjacency.
    pub fn row_sums(&self) -> &[f32] {
        &self.row_sums
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
        let flows = self.measure_flows(&propagated)?;
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

    /// Computes explainability-friendly flow summaries for the provided features.
    pub fn measure_flows(&self, features: &Tensor) -> PureResult<Vec<NodeFlowSample>> {
        self.validate_features(features)?;
        let (_, feature_dim) = features.shape();
        let mut flows = Vec::with_capacity(self.node_count());
        let data = features.data();
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
        Ok(flows)
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

/// Builder that configures the adjacency normalisation and self-loop behaviour.
pub struct GraphContextBuilder {
    adjacency: Tensor,
    self_loop_weight: f32,
    normalisation: GraphNormalization,
}

impl GraphContextBuilder {
    fn new(adjacency: Tensor) -> Self {
        Self {
            adjacency,
            self_loop_weight: 1.0,
            normalisation: GraphNormalization::default(),
        }
    }

    /// Overrides the self-loop weight added before normalisation.
    pub fn self_loop_weight(mut self, weight: f32) -> Self {
        self.self_loop_weight = weight;
        self
    }

    /// Selects the adjacency normalisation strategy.
    pub fn normalisation(mut self, strategy: GraphNormalization) -> Self {
        self.normalisation = strategy;
        self
    }

    /// Consumes the builder, producing a fully initialised [`GraphContext`].
    pub fn build(self) -> PureResult<GraphContext> {
        GraphContext::build_context(self.adjacency, self.self_loop_weight, self.normalisation)
    }
}

impl GraphContext {
    fn build_context(
        adjacency: Tensor,
        self_loop_weight: f32,
        normalisation: GraphNormalization,
    ) -> PureResult<Self> {
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

        let mut adjusted = adjacency.data().to_vec();
        if self_loop_weight > 0.0 {
            for idx in 0..rows.min(cols) {
                let offset = idx * cols + idx;
                adjusted[offset] += self_loop_weight;
            }
        }

        let mut row_degrees = vec![0.0f32; rows];
        let mut col_degrees = vec![0.0f32; cols];
        for r in 0..rows {
            let offset = r * cols;
            for c in 0..cols {
                let value = adjusted[offset + c];
                row_degrees[r] += value;
                col_degrees[c] += value;
            }
        }

        let mut normalised = vec![0.0f32; rows * cols];
        match normalisation {
            GraphNormalization::Symmetric => {
                let mut row_scale = vec![0.0f32; rows];
                for (idx, degree) in row_degrees.iter().enumerate() {
                    row_scale[idx] = if *degree <= f32::EPSILON {
                        0.0
                    } else {
                        1.0 / degree.sqrt()
                    };
                }
                let mut col_scale = vec![0.0f32; cols];
                for (idx, degree) in col_degrees.iter().enumerate() {
                    col_scale[idx] = if *degree <= f32::EPSILON {
                        0.0
                    } else {
                        1.0 / degree.sqrt()
                    };
                }
                for r in 0..rows {
                    let offset = r * cols;
                    for c in 0..cols {
                        let value = adjusted[offset + c];
                        if value.abs() <= f32::EPSILON {
                            continue;
                        }
                        let scaled = value * row_scale[r] * col_scale[c];
                        if scaled.abs() > f32::EPSILON {
                            normalised[offset + c] = scaled;
                        }
                    }
                }
            }
            GraphNormalization::Row => {
                for r in 0..rows {
                    let offset = r * cols;
                    let degree = row_degrees[r];
                    if degree <= f32::EPSILON {
                        continue;
                    }
                    for c in 0..cols {
                        let value = adjusted[offset + c];
                        if value.abs() <= f32::EPSILON {
                            continue;
                        }
                        normalised[offset + c] = value / degree;
                    }
                }
            }
        }

        let norm_adjacency = Tensor::from_vec(rows, cols, normalised)?;
        let norm_adjacency_t = norm_adjacency.transpose();
        let mut row_sums = vec![0.0f32; rows];
        let data = norm_adjacency.data();
        for r in 0..rows {
            let offset = r * cols;
            row_sums[r] = data[offset..offset + cols].iter().sum();
        }

        Ok(Self {
            norm_adjacency,
            norm_adjacency_t,
            row_sums,
            normalisation,
        })
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

    #[test]
    fn builder_supports_row_normalisation() {
        let adjacency = Tensor::from_vec(2, 2, vec![0.0, 1.0, 1.0, 0.0]).unwrap();
        let context = GraphContext::builder(adjacency)
            .self_loop_weight(0.5)
            .normalisation(GraphNormalization::Row)
            .build()
            .unwrap();
        assert_eq!(context.normalisation(), GraphNormalization::Row);
        for &sum in context.row_sums() {
            assert!((sum - 1.0).abs() < 1e-3);
        }
    }

    #[test]
    fn builder_rejects_negative_self_loops() {
        let adjacency = Tensor::from_vec(1, 1, vec![0.0]).unwrap();
        let err = GraphContext::builder(adjacency)
            .self_loop_weight(-0.1)
            .build()
            .unwrap_err();
        assert!(matches!(err, TensorError::NonPositiveWeight { .. }));
    }
}
