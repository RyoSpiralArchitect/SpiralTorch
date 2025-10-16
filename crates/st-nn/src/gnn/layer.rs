// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::GraphContext;
use crate::module::{Module, Parameter};
use crate::{PureResult, Tensor, TensorError};
use st_core::telemetry::xai::{GraphFlowTracer, NodeFlowSample};
use std::sync::{Arc, Mutex};

/// Hyperbolic graph convolution that keeps gradient flows in the Z-space tape.
#[derive(Debug)]
pub struct ZSpaceGraphConvolution {
    name: String,
    context: GraphContext,
    weight: Parameter,
    bias: Parameter,
    curvature: f32,
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
            tracer: None,
        })
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
            let mut guard = tracer.lock().unwrap_or_else(|poison| poison.into_inner());
            guard.begin_layer(self.name.clone(), self.curvature, flows.clone());
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
}

impl Module for ZSpaceGraphConvolution {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let (support, flows) = if self.tracer.is_some() {
            self.context.propagate_with_trace(input)?
        } else {
            (self.context.propagate(input)?, Vec::new())
        };
        if !flows.is_empty() {
            self.record_forward_flows(flows);
        }
        let mut out = support.matmul(self.weight.value())?;
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
        let support = self.context.propagate(input)?;
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
        let grad_input = self.context.propagate_transpose(&grad_support)?;
        Ok(grad_input)
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
        let reports = tracer.lock().unwrap().layers().to_vec();
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].layer, "gnn");
        assert!(reports[0].weight_update_magnitude.is_some());
    }
}
