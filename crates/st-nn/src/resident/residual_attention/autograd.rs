//! Reusable residual execution. Parameter ownership belongs to the outer model.
use super::*;
use crate::resident::attention::{AttentionAutograd, AttentionTape};
use st_backend_wgpu::{
    resident_matmul::{MatmulAccumulation, MatmulKernel, MatmulTile},
    resident_tensor::{ResidentTensor, TensorDevice, TensorError},
    resident_training::{
        graph::{GraphForward, ResidentGraphAutograd},
        TrainingError,
    },
    runtime::WgpuRuntime,
};

pub(crate) struct ResidualTape {
    pre: GraphForward,
    attention: AttentionTape,
    feed_forward: GraphForward,
    prediction: ResidentTensor,
}

impl ResidualTape {
    pub(crate) fn prediction(&self) -> &ResidentTensor {
        &self.prediction
    }
}

pub(crate) struct ResidualGradients {
    pub input: ResidentTensor,
    pub parameters: Vec<ResidentTensor>,
    pub z_bias: Option<ResidentTensor>,
    pub pair_bias: Option<ResidentTensor>,
}

pub(crate) struct ResidualAutograd {
    pre: ResidentGraphAutograd,
    attention: AttentionAutograd,
    feed_forward: ResidentGraphAutograd,
    pre_parameters: usize,
    parameters: usize,
}

impl ResidualAttentionPlan {
    pub(crate) fn parameter_values(&self) -> Result<Vec<GraphParameter>, InferenceError> {
        let pre = self.pre.graph_definition()?;
        let [qkv, output] = self.attention.graph_parts()?;
        let feed = self.feed_forward.graph_definition()?;
        Ok([pre, qkv, output, feed]
            .into_iter()
            .flat_map(|graph| graph.parameters().to_vec())
            .collect())
    }
}

impl ResidualAutograd {
    pub(crate) fn new(
        plan: &ResidualAttentionPlan,
        runtime: WgpuRuntime,
        tile: MatmulTile,
        kernel: MatmulKernel,
        accumulation: MatmulAccumulation,
    ) -> Result<Self, InferenceError> {
        let pre = plan.pre.graph_definition()?;
        let feed = plan.feed_forward.graph_definition()?;
        Ok(Self {
            pre_parameters: pre.parameters().len(),
            parameters: pre.parameters().len() + 4 + feed.parameters().len(),
            pre: ResidentGraphAutograd::new(runtime.clone(), pre, tile, kernel, accumulation)?,
            attention: AttentionAutograd::new(
                &plan.attention,
                runtime.clone(),
                tile,
                kernel,
                accumulation,
            )?,
            feed_forward: ResidentGraphAutograd::new(runtime, feed, tile, kernel, accumulation)?,
        })
    }

    pub(crate) fn input_layout(&self) -> &NdLayout {
        self.pre.input_layout()
    }
    pub(crate) fn output_layout(&self) -> &NdLayout {
        self.feed_forward.output_layout()
    }
    pub(crate) fn attention_spec(&self) -> AttentionSpec {
        self.attention.attention_spec()
    }
    pub(crate) fn tensor_device(&self) -> &TensorDevice {
        self.pre.tensor_device()
    }

    pub(crate) fn validate_biases(
        &self,
        z_bias: Option<&ResidentTensor>,
        pair_bias: Option<&ResidentTensor>,
    ) -> Result<(), InferenceError> {
        self.attention_spec().validate_bias_shapes(
            z_bias.map(|b| b.layout().shape()),
            pair_bias.map(|b| b.layout().shape()),
        )?;
        for tensor in [z_bias, pair_bias].into_iter().flatten() {
            if !tensor
                .device()
                .runtime()
                .context()
                .shares_handles_with(self.tensor_device().runtime().context())
            {
                return Err(TensorError::DeviceMismatch.into());
            }
        }
        Ok(())
    }

    pub(crate) fn validate_input(
        &self,
        input: &ResidentTensor,
        z_bias: Option<&ResidentTensor>,
        pair_bias: Option<&ResidentTensor>,
    ) -> Result<(), InferenceError> {
        if input.layout().shape() != self.input_layout().shape() {
            return Err(InferenceError::InvalidLayout);
        }
        self.validate_biases(z_bias, pair_bias)?;
        if !input
            .device()
            .runtime()
            .context()
            .shares_handles_with(self.tensor_device().runtime().context())
        {
            return Err(TensorError::DeviceMismatch.into());
        }
        Ok(())
    }

    pub(crate) fn set_parameters(
        &mut self,
        values: &[ResidentTensor],
    ) -> Result<(), InferenceError> {
        if values.len() != self.parameters {
            return Err(TrainingError::ParameterLayout.into());
        }
        let n = self.pre_parameters;
        self.pre.set_parameter_tensors(&values[..n])?;
        self.attention.set_parameters(&values[n..n + 4])?;
        self.feed_forward.set_parameter_tensors(&values[n + 4..])?;
        Ok(())
    }

    pub(crate) fn forward(
        &mut self,
        input: &ResidentTensor,
        z_bias: Option<&ResidentTensor>,
        pair_bias: Option<&ResidentTensor>,
    ) -> Result<ResidualTape, InferenceError> {
        self.validate_input(input, z_bias, pair_bias)?;
        self.pre.set_input_tensor(input)?;
        let pre = self.pre.forward()?;
        let attention = self
            .attention
            .forward(pre.prediction(), z_bias, pair_bias)?;
        let residual = input.add(attention.prediction())?;
        self.feed_forward.set_input_tensor(&residual)?;
        let feed_forward = self.feed_forward.forward()?;
        let prediction = residual.add(feed_forward.prediction())?;
        Ok(ResidualTape {
            pre,
            attention,
            feed_forward,
            prediction,
        })
    }

    pub(crate) fn backward(
        &mut self,
        tape: &ResidualTape,
        cotangent: &ResidentTensor,
    ) -> Result<ResidualGradients, InferenceError> {
        let guarded = self
            .tensor_device()
            .guard_together(&[cotangent, tape.prediction()])?;
        let cotangent = &guarded[0];
        let feed_forward = self.feed_forward.backward(&tape.feed_forward, cotangent)?;
        let residual = cotangent.add(feed_forward.input_gradient())?;
        let attention = self.attention.backward(&tape.attention, &residual)?;
        let pre = self.pre.backward(&tape.pre, &attention.input)?;
        let input = residual.add(pre.input_gradient())?;
        let parameters: Vec<_> = pre
            .parameter_gradients()
            .iter()
            .chain(&attention.parameters)
            .chain(feed_forward.parameter_gradients())
            .cloned()
            .collect();
        let has_z = attention.z_bias.is_some();
        let has_pair = attention.pair_bias.is_some();
        let mut family = vec![input];
        family.extend(parameters);
        family.extend(attention.z_bias);
        family.extend(attention.pair_bias);
        let mut family = self
            .tensor_device()
            .guard_together(&family.iter().collect::<Vec<_>>())?;
        let pair_bias = if has_pair { family.pop() } else { None };
        let z_bias = if has_z { family.pop() } else { None };
        let input = family.remove(0);
        Ok(ResidualGradients {
            input,
            parameters: family,
            z_bias,
            pair_bias,
        })
    }
}
