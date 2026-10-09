//! One parameter owner and one update decision across both attention projections.
use super::*;
use st_backend_wgpu::{
    resident_matmul::{MatmulAccumulation, MatmulKernel, MatmulTile},
    resident_tensor::{ResidentTensor, TensorDevice},
    resident_training::{
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
    tape: AttentionTape,
}

impl ResidentAttentionForward {
    pub fn prediction(&self) -> &ResidentTensor {
        self.tape.prediction()
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
    graph: AttentionAutograd,
    parameters: ResidentParameters,
    bound_revision: u64,
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
        let graph = AttentionAutograd::new(self, runtime, tile, kernel, accumulation)?;
        let [qkv, output] = self.graph_parts()?;
        let device = graph.tensor_device();
        let values = qkv
            .parameters()
            .iter()
            .chain(output.parameters())
            .map(|p| device.upload(&p.shape, &p.values))
            .collect::<Result<Vec<_>, _>>()?;
        let parameters = ResidentParameters::new(values)?;
        Ok(ResidentAttentionTraining {
            graph,
            parameters,
            bound_revision: 0,
            forwards: 0,
            latest: None,
        })
    }
}

impl ResidentAttentionTraining {
    pub fn input_layout(&self) -> &NdLayout {
        self.graph.input_layout()
    }
    pub fn output_layout(&self) -> &NdLayout {
        self.graph.output_layout()
    }
    pub fn tensor_device(&self) -> &TensorDevice {
        self.parameters.tensor_device()
    }
    pub fn parameter_snapshot(&self) -> ResidentParameterSnapshot {
        self.parameters.snapshot()
    }
    pub fn attention_spec(&self) -> AttentionSpec {
        self.graph.attention_spec()
    }

    pub fn forward(
        &mut self,
        input: &ResidentTensor,
        z_bias: Option<&ResidentTensor>,
        pair_bias: Option<&ResidentTensor>,
    ) -> Result<ResidentAttentionForward, InferenceError> {
        require_uncommitted_route()?;
        self.graph.validate_input(input, z_bias, pair_bias)?;
        let submission = self
            .forwards
            .checked_add(1)
            .ok_or(TrainingError::Overflow)?;
        self.latest = None;
        let parameters = self.parameters.snapshot();
        if self.bound_revision != parameters.revision() {
            self.graph.set_parameters(parameters.values())?;
            self.bound_revision = parameters.revision();
        }
        let tape = self.graph.forward(input, z_bias, pair_bias)?;
        self.forwards = submission;
        self.latest = Some(submission);
        Ok(ResidentAttentionForward {
            parameters,
            submission,
            tape,
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
        let attention = self.graph.backward(&forward.tape, cotangent)?;
        let parameters = attention.parameters;
        let bound = forward.parameters.bind_gradients(parameters.clone())?;
        Ok(ResidentAttentionVjp {
            input: attention.input,
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
