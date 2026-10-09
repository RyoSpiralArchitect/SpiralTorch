use super::*;
use crate::resident::attention::{AttentionAutograd, AttentionTape};
use st_backend_wgpu::{
    resident_matmul::{MatmulAccumulation, MatmulKernel, MatmulTile},
    resident_tensor::{ResidentTensor, TensorDevice},
    resident_training::{
        graph::{GraphForward, ResidentGraphAutograd},
        parameters::{
            ResidentParameterGradients, ResidentParameterSnapshot, ResidentParameterUpdate,
            ResidentParameters,
        },
        TrainingError,
    },
    runtime::WgpuRuntime,
};

/// Owning output plus this model's latest reusable forward tape.
pub struct ResidentResidualAttentionForward {
    parameters: ResidentParameterSnapshot,
    submission: u64,
    pre: GraphForward,
    attention: AttentionTape,
    feed_forward: GraphForward,
    prediction: ResidentTensor,
}

impl ResidentResidualAttentionForward {
    pub fn prediction(&self) -> &ResidentTensor {
        &self.prediction
    }

    pub fn parameter_revision(&self) -> u64 {
        self.parameters.revision()
    }
}

/// Exact arbitrary-cotangent derivatives, including both residual paths.
/// Geometry biases remain caller-owned and are not part of this model's SGD.
pub struct ResidentResidualAttentionVjp {
    input: ResidentTensor,
    parameters: Vec<ResidentTensor>,
    bound: ResidentParameterGradients,
    z_bias: Option<ResidentTensor>,
    pair_bias: Option<ResidentTensor>,
}

impl ResidentResidualAttentionVjp {
    pub fn input_gradient(&self) -> &ResidentTensor {
        &self.input
    }

    pub fn parameter_gradients(&self) -> &[ResidentTensor] {
        &self.parameters
    }

    pub fn z_bias_gradient(&self) -> Option<&ResidentTensor> {
        self.z_bias.as_ref()
    }

    pub fn pair_bias_gradient(&self) -> Option<&ResidentTensor> {
        self.pair_bias.as_ref()
    }
}

/// One parameter owner and all-or-none update across both branches and attention.
/// Ordering is pre-graph parameters, fused QKV weight/bias, output weight/bias,
/// then feed-forward graph parameters. Original plans and host Modules are frozen.
pub struct ResidentResidualAttentionTraining {
    pre: ResidentGraphAutograd,
    attention: AttentionAutograd,
    feed_forward: ResidentGraphAutograd,
    parameters: ResidentParameters,
    pre_parameters: usize,
    bound_revision: u64,
    forwards: u64,
    latest: Option<u64>,
}

impl ResidualAttentionPlan {
    pub fn compile_training_wgpu(
        &self,
        runtime: WgpuRuntime,
    ) -> Result<ResidentResidualAttentionTraining, InferenceError> {
        self.compile_training_wgpu_with_options(
            runtime,
            Default::default(),
            MatmulKernel::Scalar,
            Default::default(),
        )
    }

    pub fn compile_training_wgpu_with_options(
        &self,
        runtime: WgpuRuntime,
        tile: MatmulTile,
        kernel: MatmulKernel,
        accumulation: MatmulAccumulation,
    ) -> Result<ResidentResidualAttentionTraining, InferenceError> {
        require_uncommitted_route()?;
        let attention =
            AttentionAutograd::new(&self.attention, runtime.clone(), tile, kernel, accumulation)?;
        let device = attention.tensor_device();
        let pre = self.pre.graph_definition()?;
        let feed_forward = self.feed_forward.graph_definition()?;
        let [qkv, output] = self.attention.graph_parts()?;
        let values = [&pre, &qkv, &output, &feed_forward]
            .into_iter()
            .flat_map(|graph| graph.parameters())
            .map(|p| device.upload(&p.shape, &p.values))
            .collect::<Result<Vec<_>, _>>()?;
        let parameters = ResidentParameters::new(values)?;
        Ok(ResidentResidualAttentionTraining {
            pre_parameters: pre.parameters().len(),
            pre: ResidentGraphAutograd::new(runtime.clone(), pre, tile, kernel, accumulation)?,
            attention,
            feed_forward: ResidentGraphAutograd::new(
                runtime,
                feed_forward,
                tile,
                kernel,
                accumulation,
            )?,
            parameters,
            bound_revision: 0,
            forwards: 0,
            latest: None,
        })
    }
}

impl ResidentResidualAttentionTraining {
    pub fn input_layout(&self) -> &NdLayout {
        self.pre.input_layout()
    }

    pub fn output_layout(&self) -> &NdLayout {
        self.feed_forward.output_layout()
    }

    pub fn attention_spec(&self) -> AttentionSpec {
        self.attention.attention_spec()
    }

    pub fn tensor_device(&self) -> &TensorDevice {
        self.parameters.tensor_device()
    }

    pub fn parameter_snapshot(&self) -> ResidentParameterSnapshot {
        self.parameters.snapshot()
    }

    pub fn forward(
        &mut self,
        input: &ResidentTensor,
        z_bias: Option<&ResidentTensor>,
        pair_bias: Option<&ResidentTensor>,
    ) -> Result<ResidentResidualAttentionForward, InferenceError> {
        require_uncommitted_route()?;
        if input.layout().shape() != self.input_layout().shape() {
            return Err(InferenceError::InvalidLayout);
        }
        self.attention_spec().validate_bias_shapes(
            z_bias.map(|b| b.layout().shape()),
            pair_bias.map(|b| b.layout().shape()),
        )?;
        for tensor in [Some(input), z_bias, pair_bias].into_iter().flatten() {
            if !tensor
                .device()
                .runtime()
                .context()
                .shares_handles_with(self.tensor_device().runtime().context())
            {
                return Err(st_backend_wgpu::resident_tensor::TensorError::DeviceMismatch.into());
            }
        }
        let submission = self
            .forwards
            .checked_add(1)
            .ok_or(TrainingError::Overflow)?;
        self.latest = None;
        let parameters = self.parameters.snapshot();
        if self.bound_revision != parameters.revision() {
            let n = self.pre_parameters;
            self.pre.set_parameter_tensors(&parameters.values()[..n])?;
            self.attention
                .set_parameters(&parameters.values()[n..n + 4])?;
            self.feed_forward
                .set_parameter_tensors(&parameters.values()[n + 4..])?;
            self.bound_revision = parameters.revision();
        }
        self.pre.set_input_tensor(input)?;
        let pre = self.pre.forward()?;
        let attention = self
            .attention
            .forward(pre.prediction(), z_bias, pair_bias)?;
        let residual = input.add(attention.prediction())?;
        self.feed_forward.set_input_tensor(&residual)?;
        let feed_forward = self.feed_forward.forward()?;
        let prediction = residual.add(feed_forward.prediction())?;
        self.forwards = submission;
        self.latest = Some(submission);
        Ok(ResidentResidualAttentionForward {
            parameters,
            submission,
            pre,
            attention,
            feed_forward,
            prediction,
        })
    }

    pub fn backward(
        &mut self,
        forward: &ResidentResidualAttentionForward,
        cotangent: &ResidentTensor,
    ) -> Result<ResidentResidualAttentionVjp, InferenceError> {
        require_uncommitted_route()?;
        if self.latest != Some(forward.submission)
            || !self.parameters.is_current(&forward.parameters)
        {
            return Err(TrainingError::StaleForward.into());
        }
        let guarded = self
            .tensor_device()
            .guard_together(&[cotangent, forward.prediction()])?;
        let cotangent = &guarded[0];
        let feed_forward = self
            .feed_forward
            .backward(&forward.feed_forward, cotangent)?;
        let residual = cotangent.add(feed_forward.input_gradient())?;
        let attention = self.attention.backward(&forward.attention, &residual)?;
        let pre = self.pre.backward(&forward.pre, &attention.input)?;
        let input = residual.add(pre.input_gradient())?;
        let parameters: Vec<_> = pre
            .parameter_gradients()
            .iter()
            .chain(&attention.parameters)
            .chain(feed_forward.parameter_gradients())
            .cloned()
            .collect();
        let has_z_bias = attention.z_bias.is_some();
        let has_pair_bias = attention.pair_bias.is_some();
        let mut family = vec![input];
        family.extend(parameters);
        family.extend(attention.z_bias);
        family.extend(attention.pair_bias);
        let mut family = self
            .tensor_device()
            .guard_together(&family.iter().collect::<Vec<_>>())?;
        let pair_bias = if has_pair_bias { family.pop() } else { None };
        let z_bias = if has_z_bias { family.pop() } else { None };
        let input = family.remove(0);
        let parameters = family;
        let bound = forward.parameters.bind_gradients(parameters.clone())?;
        Ok(ResidentResidualAttentionVjp {
            input,
            parameters,
            bound,
            z_bias,
            pair_bias,
        })
    }

    /// One decision across every branch. A receipt read, not submission or the
    /// attempted revision, proves acceptance. There is no implicit gradient mean.
    pub fn sgd(
        &mut self,
        gradients: &ResidentResidualAttentionVjp,
        rate: f32,
    ) -> Result<ResidentParameterUpdate, InferenceError> {
        require_uncommitted_route()?;
        let update = self.parameters.sgd(&gradients.bound, rate)?;
        self.latest = None;
        Ok(update)
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
