// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::GraphContext;
use crate::module::{Module, Parameter};
use crate::{PureResult, Tensor, TensorError};
use st_core::telemetry::xai::{GraphFlowTracer, NodeFlowSample};
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

/// Reduction applied when combining multi-hop neighbourhood features.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationReducer {
    /// Sum all contributions respecting the configured attenuation factors.
    Sum,
    /// Average the contributions (normalising by the absolute weight mass).
    Mean,
}

/// Strategy used to mix neighbour features before applying the learnable kernel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NeighborhoodAggregation {
    /// Single-hop propagation identical to the historic implementation.
    Single,
    /// Multi-hop aggregation with optional self-inclusion and attenuation.
    MultiHop {
        hops: NonZeroUsize,
        include_self: bool,
        attenuation: f32,
        reducer: AggregationReducer,
    },
}

impl NeighborhoodAggregation {
    /// Builds a multi-hop aggregation that sums contributions without attenuation.
    pub fn multi_hop_sum(hops: NonZeroUsize) -> Self {
        NeighborhoodAggregation::MultiHop {
            hops,
            include_self: true,
            attenuation: 1.0,
            reducer: AggregationReducer::Sum,
        }
    }

    /// Builds a multi-hop aggregation that averages the hop contributions.
    pub fn multi_hop_mean(hops: NonZeroUsize) -> Self {
        NeighborhoodAggregation::MultiHop {
            hops,
            include_self: true,
            attenuation: 1.0,
            reducer: AggregationReducer::Mean,
        }
    }

    /// Overrides whether the original node features participate in the mix.
    pub fn with_include_self(mut self, include_self: bool) -> Self {
        if let NeighborhoodAggregation::MultiHop {
            include_self: ref mut flag,
            ..
        } = self
        {
            *flag = include_self;
        }
        self
    }

    /// Adjusts the attenuation factor applied to successive hops.
    pub fn with_attenuation(mut self, attenuation: f32) -> Self {
        if let NeighborhoodAggregation::MultiHop {
            attenuation: ref mut factor,
            ..
        } = self
        {
            *factor = attenuation;
        }
        self
    }

    /// Selects the reducer applied to the aggregated neighbourhood.
    pub fn with_reducer(mut self, reducer: AggregationReducer) -> Self {
        if let NeighborhoodAggregation::MultiHop {
            reducer: ref mut current,
            ..
        } = self
        {
            *current = reducer;
        }
        self
    }

    fn validate(self) -> PureResult<()> {
        if let NeighborhoodAggregation::MultiHop { attenuation, .. } = self {
            if !attenuation.is_finite() || attenuation <= 0.0 {
                return Err(TensorError::InvalidValue {
                    label: "aggregation_attenuation",
                });
            }
        }
        Ok(())
    }

    fn weights(self) -> PureResult<Vec<f32>> {
        match self {
            NeighborhoodAggregation::Single => Ok(vec![0.0, 1.0]),
            NeighborhoodAggregation::MultiHop {
                hops,
                include_self,
                attenuation,
                reducer,
            } => {
                let hop_count = hops.get();
                let mut weights = Vec::with_capacity(hop_count + 1);
                weights.push(if include_self { 1.0 } else { 0.0 });
                for hop in 1..=hop_count {
                    weights.push(attenuation.powi(hop as i32));
                }
                if matches!(reducer, AggregationReducer::Mean) {
                    let norm: f32 = weights.iter().map(|w| w.abs()).sum();
                    if norm > f32::EPSILON {
                        for weight in &mut weights {
                            *weight /= norm;
                        }
                    }
                }
                if weights.iter().all(|w| w.abs() <= f32::EPSILON) {
                    return Err(TensorError::InvalidValue {
                        label: "aggregation_weights",
                    });
                }
                Ok(weights)
            }
        }
    }
}

impl Default for NeighborhoodAggregation {
    fn default() -> Self {
        NeighborhoodAggregation::Single
    }
}

#[derive(Debug, Clone)]
struct AggregatedSupport {
    support: Tensor,
    weights: Vec<f32>,
}

impl AggregatedSupport {
    fn support(&self) -> &Tensor {
        &self.support
    }

    fn into_weights(self) -> Vec<f32> {
        self.weights
    }
}

/// Hyperbolic graph convolution that keeps gradient flows in the Z-space tape.
#[derive(Debug)]
pub struct ZSpaceGraphConvolution {
    name: String,
    context: GraphContext,
    weight: Parameter,
    bias: Parameter,
    curvature: f32,
    aggregation: NeighborhoodAggregation,
    tracer: Option<Arc<Mutex<GraphFlowTracer>>>,
}

impl ZSpaceGraphConvolution {
    /// Builds a new graph convolution with deterministic initial weights.
    pub fn new(
        name: impl Into<String>,
        context: GraphContext,
        input_dim: usize,
        output_dim: usize,
        curvature: f32,
        learning_rate: f32,
    ) -> PureResult<Self> {
        if input_dim == 0 || output_dim == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: input_dim,
                cols: output_dim,
            });
        }
        let name = name.into();
        let mut scale = 0.01f32;
        let weights = Tensor::from_fn(input_dim, output_dim, |_r, _c| {
            let value = scale;
            scale += 0.01;
            value
        })?;
        let bias = Tensor::zeros(1, output_dim)?;
        let mut weight_param = Parameter::new(format!("{name}::weight"), weights);
        weight_param.attach_hypergrad(curvature, learning_rate)?;
        let mut bias_param = Parameter::new(format!("{name}::bias"), bias);
        bias_param.attach_hypergrad(curvature, learning_rate)?;
        Ok(Self {
            name,
            context,
            weight: weight_param,
            bias: bias_param,
            curvature,
            aggregation: NeighborhoodAggregation::default(),
            tracer: None,
        })
    }

    /// Switches the neighbourhood aggregation strategy used by the layer.
    pub fn set_aggregation(&mut self, aggregation: NeighborhoodAggregation) -> PureResult<()> {
        aggregation.validate()?;
        self.aggregation = aggregation;
        Ok(())
    }

    /// Returns a new layer adopting the provided aggregation strategy.
    pub fn with_aggregation(mut self, aggregation: NeighborhoodAggregation) -> PureResult<Self> {
        self.set_aggregation(aggregation)?;
        Ok(self)
    }

    /// Attaches a shared flow tracer for interpretability tooling.
    pub fn set_tracer(&mut self, tracer: Arc<Mutex<GraphFlowTracer>>) {
        self.tracer = Some(tracer);
    }

    /// Returns the underlying graph context.
    pub fn context(&self) -> &GraphContext {
        &self.context
    }

    fn record_forward_flows(&self, flows: Vec<NodeFlowSample>) {
        if let Some(tracer) = &self.tracer {
            if let Ok(mut guard) = tracer.lock() {
                guard.begin_layer(self.name.clone(), self.curvature, flows);
            }
        }
    }

    fn record_backward_updates(&self, weight: f32, bias: f32) {
        if let Some(tracer) = &self.tracer {
            if let Ok(mut guard) = tracer.lock() {
                guard.record_weight_update(weight, Some(bias));
            }
        }
    }

    fn aggregate_support(&self, input: &Tensor) -> PureResult<AggregatedSupport> {
        let weights = self.aggregation.weights()?;
        let (rows, cols) = input.shape();
        let mut support = Tensor::zeros(rows, cols)?;
        let mut current = input.clone();
        for (idx, weight) in weights.iter().enumerate() {
            if idx > 0 {
                current = self.context.propagate(&current)?;
            }
            if weight.abs() > f32::EPSILON {
                support.add_scaled(&current, *weight)?;
            }
        }
        Ok(AggregatedSupport { support, weights })
    }

    fn backpropagate_through_aggregation(
        &self,
        grad_support: &Tensor,
        weights: &[f32],
    ) -> PureResult<Tensor> {
        let (rows, cols) = grad_support.shape();
        let mut grad_states = Vec::with_capacity(weights.len());
        for _ in 0..weights.len() {
            grad_states.push(Tensor::zeros(rows, cols)?);
        }
        for (state, &weight) in grad_states.iter_mut().zip(weights.iter()) {
            if weight.abs() > f32::EPSILON {
                state.add_scaled(grad_support, weight)?;
            }
        }
        for idx in (1..grad_states.len()).rev() {
            let propagated = self.context.propagate_transpose(&grad_states[idx])?;
            grad_states[idx - 1].add_scaled(&propagated, 1.0)?;
        }
        Ok(grad_states
            .into_iter()
            .next()
            .expect("aggregation must produce at least one state"))
    }
}

impl Module for ZSpaceGraphConvolution {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let aggregated = self.aggregate_support(input)?;
        if self.tracer.is_some() {
            let flows = self.context.measure_flows(aggregated.support())?;
            if !flows.is_empty() {
                self.record_forward_flows(flows);
            }
        }
        let mut out = aggregated.support().matmul(self.weight.value())?;
        out.add_row_inplace(self.bias.value().data())?;
        Ok(out)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        if grad_output.shape().0 != input.shape().0 {
            return Err(TensorError::ShapeMismatch {
                left: grad_output.shape(),
                right: input.shape(),
            });
        }
        let aggregated = self.aggregate_support(input)?;
        let support = aggregated.support();
        let batch = input.shape().0 as f32;
        let grad_w = support
            .transpose()
            .matmul(grad_output)?
            .scale(1.0 / batch)?;
        self.weight.accumulate_euclidean(&grad_w)?;

        let summed = grad_output.sum_axis0();
        let grad_b = Tensor::from_vec(1, summed.len(), summed)?.scale(1.0 / batch)?;
        self.bias.accumulate_euclidean(&grad_b)?;

        if self.tracer.is_some() {
            let weight_norm = l2_norm(grad_w.data());
            let bias_norm = l2_norm(grad_b.data());
            self.record_backward_updates(weight_norm, bias_norm);
        }

        let grad_support = grad_output.matmul(&self.weight.value().transpose())?;
        let weights = aggregated.into_weights();
        self.backpropagate_through_aggregation(&grad_support, &weights)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&self.weight)?;
        visitor(&self.bias)?;
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&mut self.weight)?;
        visitor(&mut self.bias)?;
        Ok(())
    }
}

fn l2_norm(data: &[f32]) -> f32 {
    data.iter().map(|v| v * v).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroUsize;

    #[test]
    fn graph_convolution_forward_runs() {
        let adjacency = Tensor::from_vec(2, 2, vec![0.0, 1.0, 1.0, 0.0]).unwrap();
        let context = GraphContext::from_adjacency(adjacency).unwrap();
        let layer = ZSpaceGraphConvolution::new("gnn", context, 3, 2, -1.0, 0.05).unwrap();
        let input = Tensor::from_vec(2, 3, vec![1.0, 0.0, -1.0, 0.5, 0.5, 0.5]).unwrap();
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), (2, 2));
    }

    #[test]
    fn graph_convolution_backward_streams_updates() {
        let adjacency =
            Tensor::from_vec(3, 3, vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]).unwrap();
        let context = GraphContext::from_adjacency(adjacency).unwrap();
        let mut layer = ZSpaceGraphConvolution::new("gnn", context, 2, 2, -1.0, 0.05).unwrap();
        let input = Tensor::from_vec(3, 2, vec![1.0, 0.0, 0.5, 0.5, -0.5, 1.0]).unwrap();
        let grad_output = Tensor::from_vec(3, 2, vec![0.1, -0.2, 0.05, 0.15, -0.1, 0.2]).unwrap();
        let grad_input = layer.backward(&input, &grad_output).unwrap();
        assert_eq!(grad_input.shape(), (3, 2));
    }

    #[test]
    fn graph_convolution_records_traces() {
        let adjacency = Tensor::from_vec(2, 2, vec![0.0, 1.0, 1.0, 0.0]).unwrap();
        let context = GraphContext::from_adjacency(adjacency).unwrap();
        let tracer = Arc::new(Mutex::new(GraphFlowTracer::new()));
        let mut layer = ZSpaceGraphConvolution::new("gnn", context, 2, 1, -1.0, 0.05).unwrap();
        layer.set_tracer(tracer.clone());
        let input = Tensor::from_vec(2, 2, vec![1.0, 0.5, -0.5, 1.0]).unwrap();
        let grad_output = Tensor::from_vec(2, 1, vec![0.1, -0.2]).unwrap();
        let _ = layer.forward(&input).unwrap();
        let _ = layer.backward(&input, &grad_output).unwrap();
        let reports = tracer
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .layers()
            .to_vec();
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].layer, "gnn");
        assert!(reports[0].weight_update_magnitude.is_some());
    }

    #[test]
    fn graph_convolution_supports_multi_hop_sum() {
        let adjacency =
            Tensor::from_vec(3, 3, vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]).unwrap();
        let context = GraphContext::from_adjacency(adjacency).unwrap();
        let aggregation = NeighborhoodAggregation::multi_hop_sum(NonZeroUsize::new(2).unwrap())
            .with_include_self(true)
            .with_attenuation(0.5);
        let layer = ZSpaceGraphConvolution::new("multi", context, 2, 2, -1.0, 0.05)
            .unwrap()
            .with_aggregation(aggregation)
            .unwrap();
        let input = Tensor::from_vec(3, 2, vec![1.0, 0.0, 0.5, 1.0, -0.5, 0.5]).unwrap();
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), (3, 2));
        assert!(output.data().iter().any(|v| v.abs() > 0.0));
    }

    #[test]
    fn graph_convolution_multi_hop_backward_is_stable() {
        let adjacency = Tensor::from_vec(
            4,
            4,
            vec![
                0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
            ],
        )
        .unwrap();
        let context = GraphContext::from_adjacency(adjacency).unwrap();
        let mut layer = ZSpaceGraphConvolution::new("stable", context, 3, 2, -1.0, 0.05).unwrap();
        let aggregation = NeighborhoodAggregation::multi_hop_mean(NonZeroUsize::new(3).unwrap())
            .with_include_self(false)
            .with_attenuation(0.75);
        layer.set_aggregation(aggregation).unwrap();
        let input = Tensor::from_vec(
            4,
            3,
            vec![
                1.0, 0.5, -0.5, 0.5, 1.0, 0.5, -1.0, 0.25, 0.75, 0.75, -0.25, 1.25,
            ],
        )
        .unwrap();
        let grad_output =
            Tensor::from_vec(4, 2, vec![0.1, -0.2, 0.05, 0.15, -0.1, 0.2, 0.075, -0.05]).unwrap();
        let grad_input = layer.backward(&input, &grad_output).unwrap();
        assert_eq!(grad_input.shape(), (4, 3));
        assert!(grad_input.data().iter().all(|v| v.is_finite()));
    }
}
