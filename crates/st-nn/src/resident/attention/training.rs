//! One parameter owner and one update decision across both attention projections.
use super::*;
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

/// Owning output and same-owner/version token for the most recent forward.
/// The output survives later calls; its reusable graph tape does not.
pub struct ResidentAttentionForward {
    parameters: ResidentParameterSnapshot,
    submission: u64,
    qkv: GraphForward,
    output: GraphForward,
    heads: [ResidentTensor; 3],
    z_bias: Option<ResidentTensor>,
    pair_bias: Option<ResidentTensor>,
}

impl ResidentAttentionForward {
    pub fn prediction(&self) -> &ResidentTensor {
        self.output.prediction()
    }
    pub fn parameter_revision(&self) -> u64 {
        self.parameters.revision()
    }
}

/// Input, fused projection parameters, and optional logical geometry derivatives.
/// Runtime biases remain caller-owned. Their gradients are returned, not silently
/// optimized here; broadcast/view adjoints are explicit caller responsibilities.
pub struct ResidentAttentionVjp {
    input: ResidentTensor,
    parameters: Vec<ResidentTensor>,
    bound: ResidentParameterGradients,
    z_bias: Option<ResidentTensor>,
    pair_bias: Option<ResidentTensor>,
}

impl ResidentAttentionVjp {
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

/// Fused QKV -> attention -> output projection with resident forward/VJP and
/// all-or-none plain SGD. The source plan and host Modules are never mutated.
/// Parameter order is fused QKV weight/bias, then output weight/bias; individual
/// Q/K/V projections occupy consecutive columns of the fused weight matrix.
pub struct ResidentAttentionTraining {
    qkv: ResidentGraphAutograd,
    output: ResidentGraphAutograd,
    parameters: ResidentParameters,
    bound_revision: u64,
    spec: AttentionSpec,
    forwards: u64,
    latest: Option<u64>,
}

impl AttentionInferencePlan {
    /// Initialize a separate trainable owner from this immutable plan. No source
    /// Module synchronization, loss choice, optimizer history or readback is added.
    pub fn compile_training_wgpu(
        &self,
        runtime: WgpuRuntime,
    ) -> Result<ResidentAttentionTraining, InferenceError> {
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
    ) -> Result<ResidentAttentionTraining, InferenceError> {
        require_uncommitted_route()?;
        st_backend_wgpu::resident_tensor::attention::validate_limits(
            self.spec,
            &runtime.context().device().limits(),
        )?;
        let qkv = self.qkv.graph_definition()?;
        let output = self.output.graph_definition()?;
        let device = TensorDevice::new(runtime.clone())?;
        let values = qkv
            .parameters()
            .iter()
            .chain(output.parameters())
            .map(|p| device.upload(&p.shape, &p.values))
            .collect::<Result<Vec<_>, _>>()?;
        let parameters = ResidentParameters::new(values)?;
        Ok(ResidentAttentionTraining {
            qkv: ResidentGraphAutograd::new(runtime.clone(), qkv, tile, kernel, accumulation)?,
            output: ResidentGraphAutograd::new(runtime, output, tile, kernel, accumulation)?,
            parameters,
            bound_revision: 0,
            spec: self.spec,
            forwards: 0,
            latest: None,
        })
    }
}

impl ResidentAttentionTraining {
    pub fn input_layout(&self) -> &NdLayout {
        self.qkv.input_layout()
    }
    pub fn output_layout(&self) -> &NdLayout {
        self.output.output_layout()
    }
    pub fn tensor_device(&self) -> &TensorDevice {
        self.parameters.tensor_device()
    }
    pub fn parameter_snapshot(&self) -> ResidentParameterSnapshot {
        self.parameters.snapshot()
    }
    pub fn attention_spec(&self) -> AttentionSpec {
        self.spec
    }

    pub fn forward(
        &mut self,
        input: &ResidentTensor,
        z_bias: Option<&ResidentTensor>,
        pair_bias: Option<&ResidentTensor>,
    ) -> Result<ResidentAttentionForward, InferenceError> {
        require_uncommitted_route()?;
        if input.layout().shape() != self.input_layout().shape() {
            return Err(InferenceError::InvalidLayout);
        }
        self.spec.validate_bias_shapes(
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
            self.qkv.set_parameter_tensors(&parameters.values()[..2])?;
            self.output
                .set_parameter_tensors(&parameters.values()[2..])?;
            self.bound_revision = parameters.revision();
        }
        self.qkv.set_input_tensor(input)?;
        let qkv = self.qkv.forward()?;
        let [batch, heads, sequence, dim] = self.spec.query_shape();
        let projected = qkv
            .prediction()
            .reshape(&[batch, sequence, 3, heads, dim])?;
        let head = |index| projected.select(2, index)?.permute(&[0, 2, 1, 3]);
        let heads = [head(0)?, head(1)?, head(2)?];
        let merged = heads[0].scaled_dot_attention_merged_heads(
            &heads[1],
            &heads[2],
            self.spec.scale(),
            self.spec.mask(),
            z_bias,
            pair_bias,
        )?;
        self.output.set_input_tensor(&merged)?;
        let output = self.output.forward()?;
        self.forwards = submission;
        self.latest = Some(submission);
        Ok(ResidentAttentionForward {
            parameters,
            submission,
            qkv,
            output,
            heads,
            z_bias: z_bias.cloned(),
            pair_bias: pair_bias.cloned(),
        })
    }

    /// Repeated cotangents may use the current tape. Each result owns its values;
    /// later backwards cannot overwrite gradients already submitted to SGD.
    pub fn backward(
        &mut self,
        forward: &ResidentAttentionForward,
        cotangent: &ResidentTensor,
    ) -> Result<ResidentAttentionVjp, InferenceError> {
        require_uncommitted_route()?;
        if self.latest != Some(forward.submission)
            || !self.parameters.is_current(&forward.parameters)
        {
            return Err(TrainingError::StaleForward.into());
        }
        let output = self.output.backward(&forward.output, cotangent)?;
        let [batch, heads, sequence, dim] = self.spec.query_shape();
        let seed = output
            .input_gradient()
            .reshape(&[batch, sequence, heads, dim])?
            .permute(&[0, 2, 1, 3])?;
        let attention = forward.heads[0].scaled_dot_attention_vjp(
            &forward.heads[1],
            &forward.heads[2],
            &seed,
            self.spec.scale(),
            self.spec.mask(),
            forward.z_bias.as_ref(),
            forward.pair_bias.as_ref(),
        )?;
        let branches =
            [&attention.query, &attention.key, &attention.value].map(|g| g.permute(&[0, 2, 1, 3]));
        let [query, key, value] = branches;
        let joined = ResidentTensor::concatenate(&[&query?, &key?, &value?], 2)?
            .reshape(self.qkv.output_layout().shape())?;
        let qkv = self.qkv.backward(&forward.qkv, &joined)?;
        let parameters: Vec<_> = qkv
            .parameter_gradients()
            .iter()
            .chain(output.parameter_gradients())
            .cloned()
            .collect();
        let bound = forward.parameters.bind_gradients(parameters.clone())?;
        Ok(ResidentAttentionVjp {
            input: qkv.input_gradient().clone(),
            parameters,
            bound,
            z_bias: attention.z_bias,
            pair_bias: attention.pair_bias,
        })
    }

    /// Submit exactly one decision across QKV and output parameters. Acceptance
    /// is only proven by reading the returned receipt; zero rate still validates.
    pub fn sgd(
        &mut self,
        gradients: &ResidentAttentionVjp,
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
