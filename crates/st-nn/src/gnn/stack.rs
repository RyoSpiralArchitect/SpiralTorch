// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::{GraphContext, NeighborhoodAggregation, ZSpaceGraphConvolution};
use crate::layers::activation::Relu;
use crate::module::{Module, Parameter};
use crate::{PureResult, Tensor, TensorError};
use st_core::telemetry::xai::GraphFlowTracer;
use std::cell::RefCell;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

/// Activation applied after each convolutional stage in a graph network stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphActivation {
    /// Leave the convolution output untouched.
    Identity,
    /// Apply a ReLU activation.
    Relu,
}

impl Default for GraphActivation {
    fn default() -> Self {
        GraphActivation::Identity
    }
}

/// Describes a single layer inside a [`ZSpaceGraphNetwork`].
#[derive(Debug, Clone, PartialEq)]
pub struct GraphLayerSpec {
    output_dim: NonZeroUsize,
    aggregation: NeighborhoodAggregation,
    activation: GraphActivation,
}

impl GraphLayerSpec {
    /// Builds a new layer specification targeting the provided output dimension.
    pub fn new(output_dim: NonZeroUsize) -> Self {
        Self {
            output_dim,
            aggregation: NeighborhoodAggregation::default(),
            activation: GraphActivation::default(),
        }
    }

    /// Overrides the aggregation strategy used by the layer.
    pub fn with_aggregation(mut self, aggregation: NeighborhoodAggregation) -> Self {
        self.aggregation = aggregation;
        self
    }

    /// Selects the activation applied after the convolution.
    pub fn with_activation(mut self, activation: GraphActivation) -> Self {
        self.activation = activation;
        self
    }
}

/// Builder that assembles a [`ZSpaceGraphNetwork`] with consistent dimensions.
#[derive(Debug, Clone)]
pub struct ZSpaceGraphNetworkBuilder {
    context: GraphContext,
    input_dim: NonZeroUsize,
    curvature: f32,
    learning_rate: f32,
    layers: Vec<GraphLayerSpec>,
    tracer: Option<Arc<Mutex<GraphFlowTracer>>>,
}

impl ZSpaceGraphNetworkBuilder {
    /// Initialises the builder with the shared graph context and input dimension.
    pub fn new(
        context: GraphContext,
        input_dim: NonZeroUsize,
        curvature: f32,
        learning_rate: f32,
    ) -> Self {
        Self {
            context,
            input_dim,
            curvature,
            learning_rate,
            layers: Vec::new(),
            tracer: None,
        }
    }

    /// Adds another layer specification to the network.
    pub fn push_layer(&mut self, layer: GraphLayerSpec) {
        self.layers.push(layer);
    }

    /// Registers a shared flow tracer that each layer will use to emit telemetry.
    pub fn set_tracer(&mut self, tracer: Arc<Mutex<GraphFlowTracer>>) {
        self.tracer = Some(tracer);
    }

    /// Consumes the builder while installing a shared flow tracer.
    pub fn with_tracer(mut self, tracer: Arc<Mutex<GraphFlowTracer>>) -> Self {
        self.set_tracer(tracer);
        self
    }

    /// Consumes the builder and returns a fully initialised network.
    pub fn build(self, name: impl Into<String>) -> PureResult<ZSpaceGraphNetwork> {
        if self.layers.is_empty() {
            return Err(TensorError::EmptyInput("graph_network_layers"));
        }
        let ZSpaceGraphNetworkBuilder {
            context,
            input_dim,
            curvature,
            learning_rate,
            layers: layer_specs,
            tracer,
        } = self;
        let name = name.into();
        let mut current_dim = input_dim.get();
        let mut layers = Vec::with_capacity(layer_specs.len());
        for (idx, spec) in layer_specs.into_iter().enumerate() {
            let mut layer = ZSpaceGraphConvolution::new(
                format!("{name}::layer{idx}"),
                context.clone(),
                current_dim,
                spec.output_dim.get(),
                curvature,
                learning_rate,
            )?;
            layer.set_aggregation(spec.aggregation)?;
            if let Some(tracer) = tracer.as_ref() {
                layer.set_tracer(tracer.clone());
            }
            layers.push(GraphLayer {
                conv: layer,
                activation: GraphActivationRuntime::from_spec(spec.activation),
            });
            current_dim = spec.output_dim.get();
        }
        Ok(ZSpaceGraphNetwork {
            layers,
            cache: RefCell::new(None),
        })
    }
}

#[derive(Debug)]
struct GraphLayer {
    conv: ZSpaceGraphConvolution,
    activation: GraphActivationRuntime,
}

#[derive(Debug)]
enum GraphActivationRuntime {
    Identity,
    Relu(Relu),
}

impl GraphActivationRuntime {
    fn from_spec(spec: GraphActivation) -> Self {
        match spec {
            GraphActivation::Identity => GraphActivationRuntime::Identity,
            GraphActivation::Relu => GraphActivationRuntime::Relu(Relu::new()),
        }
    }

    fn needs_cache(&self) -> bool {
        !matches!(self, GraphActivationRuntime::Identity)
    }

    fn forward(&self, input: Tensor) -> PureResult<Tensor> {
        match self {
            GraphActivationRuntime::Identity => Ok(input),
            GraphActivationRuntime::Relu(layer) => layer.forward(&input),
        }
    }

    fn backward(&mut self, input: Option<&Tensor>, grad_output: Tensor) -> PureResult<Tensor> {
        match self {
            GraphActivationRuntime::Identity => Ok(grad_output),
            GraphActivationRuntime::Relu(layer) => {
                let Some(pre_activation) = input else {
                    return Err(TensorError::InvalidValue {
                        label: "graph_activation_input",
                    });
                };
                layer.backward(pre_activation, &grad_output)
            }
        }
    }
}

#[derive(Default, Debug)]
struct ForwardCache {
    conv_inputs: Vec<Tensor>,
    activation_inputs: Vec<Option<Tensor>>,
}

impl ForwardCache {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            conv_inputs: Vec::with_capacity(capacity),
            activation_inputs: Vec::with_capacity(capacity),
        }
    }
}

/// Stack of [`ZSpaceGraphConvolution`] layers that preserves intermediate states for backprop.
#[derive(Debug)]
pub struct ZSpaceGraphNetwork {
    layers: Vec<GraphLayer>,
    cache: RefCell<Option<ForwardCache>>,
}

impl ZSpaceGraphNetwork {
    /// Returns `true` if the stack contains no layers.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    fn take_cache(&self) -> Option<ForwardCache> {
        self.cache.replace(None)
    }

    fn store_cache(&self, cache: ForwardCache) {
        self.cache.replace(Some(cache));
    }
}

impl Module for ZSpaceGraphNetwork {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let mut state = input.clone();
        let mut cache = ForwardCache::with_capacity(self.layers.len());
        for layer in &self.layers {
            cache.conv_inputs.push(state.clone());
            let conv_output = layer.conv.forward(&state)?;
            let activation_input = if layer.activation.needs_cache() {
                Some(conv_output.clone())
            } else {
                None
            };
            let activated = layer.activation.forward(conv_output)?;
            cache.activation_inputs.push(activation_input);
            state = activated;
        }
        self.store_cache(cache);
        Ok(state)
    }

    fn backward(&mut self, _input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        if self.layers.is_empty() {
            return Err(TensorError::EmptyInput("graph_network_layers"));
        }
        let mut grad = grad_output.clone();
        let cache = self.take_cache().ok_or(TensorError::InvalidValue {
            label: "graph_network_cache",
        })?;
        if cache.conv_inputs.len() != self.layers.len()
            || cache.activation_inputs.len() != self.layers.len()
        {
            return Err(TensorError::InvalidValue {
                label: "graph_network_cache_depth",
            });
        }
        for idx in (0..self.layers.len()).rev() {
            let layer = &mut self.layers[idx];
            let activation_input = cache.activation_inputs[idx].as_ref();
            let grad_after_activation = layer.activation.backward(activation_input, grad)?;
            grad = layer
                .conv
                .backward(&cache.conv_inputs[idx], &grad_after_activation)?;
        }
        Ok(grad)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        for layer in &self.layers {
            layer.conv.visit_parameters(visitor)?;
        }
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        for layer in &mut self.layers {
            layer.conv.visit_parameters_mut(visitor)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use st_core::telemetry::xai::GraphFlowTracer;
    use std::num::NonZeroUsize;
    use std::sync::{Arc, Mutex};

    fn sample_context() -> GraphContext {
        let adjacency =
            Tensor::from_vec(3, 3, vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]).unwrap();
        GraphContext::from_adjacency(adjacency).unwrap()
    }

    #[test]
    fn builder_rejects_empty_stack() {
        let builder = ZSpaceGraphNetworkBuilder::new(
            sample_context(),
            NonZeroUsize::new(2).unwrap(),
            -1.0,
            0.05,
        );
        assert!(builder.build("empty").is_err());
    }

    #[test]
    fn stack_forward_backward_runs() {
        let mut builder = ZSpaceGraphNetworkBuilder::new(
            sample_context(),
            NonZeroUsize::new(2).unwrap(),
            -1.0,
            0.05,
        );
        builder.push_layer(GraphLayerSpec::new(NonZeroUsize::new(3).unwrap()));
        builder.push_layer(
            GraphLayerSpec::new(NonZeroUsize::new(2).unwrap())
                .with_activation(GraphActivation::Relu),
        );
        let mut network = builder.build("stack").unwrap();
        let input = Tensor::from_vec(3, 2, vec![1.0, 0.5, 0.25, -0.75, 0.0, 1.0]).unwrap();
        let output = network.forward(&input).unwrap();
        assert_eq!(output.shape(), (3, 2));
        let grad_output = Tensor::from_vec(3, 2, vec![0.1, -0.2, 0.05, 0.15, -0.1, 0.2]).unwrap();
        let grad_input = network.backward(&input, &grad_output).unwrap();
        assert_eq!(grad_input.shape(), (3, 2));
        assert!(grad_input.data().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn stack_records_flow_traces_with_shared_tracer() {
        let tracer = Arc::new(Mutex::new(GraphFlowTracer::new()));
        let mut builder = ZSpaceGraphNetworkBuilder::new(
            sample_context(),
            NonZeroUsize::new(2).unwrap(),
            -1.0,
            0.05,
        )
        .with_tracer(tracer.clone());
        builder.push_layer(GraphLayerSpec::new(NonZeroUsize::new(3).unwrap()));
        builder.push_layer(
            GraphLayerSpec::new(NonZeroUsize::new(2).unwrap())
                .with_activation(GraphActivation::Relu),
        );
        let mut network = builder.build("stack_trace").unwrap();
        let input = Tensor::from_vec(3, 2, vec![1.0, 0.25, 0.5, -0.5, -0.25, 1.0]).unwrap();
        let grad_output = Tensor::from_vec(3, 2, vec![0.1, -0.2, 0.05, 0.15, -0.1, 0.2]).unwrap();
        let _ = network.forward(&input).unwrap();
        let _ = network.backward(&input, &grad_output).unwrap();
        let guard = tracer.lock().unwrap_or_else(|poison| poison.into_inner());
        assert_eq!(guard.layers().len(), 2);
        assert_eq!(guard.layers()[0].layer, "stack_trace::layer0");
        assert_eq!(guard.layers()[1].layer, "stack_trace::layer1");
        assert!(guard.total_energy() > 0.0);
    }
}
