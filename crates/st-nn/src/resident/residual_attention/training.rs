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

/// Owning output plus this model's latest reusable forward tape.
pub struct ResidentResidualAttentionForward {
    parameters: ResidentParameterSnapshot,
    submission: u64,
    tape: ResidualTape,
}

impl ResidentResidualAttentionForward {
    pub fn prediction(&self) -> &ResidentTensor {
        self.tape.prediction()
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
    graph: ResidualAutograd,
    parameters: ResidentParameters,
    bound_revision: Option<u64>,
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
        let graph = ResidualAutograd::new(self, runtime, tile, kernel, accumulation)?;
        let device = graph.tensor_device();
        let values = self
            .parameter_values()?
            .iter()
            .map(|p| device.upload(&p.shape, &p.values))
            .collect::<Result<Vec<_>, _>>()?;
        let parameters = ResidentParameters::new(values)?;
        Ok(ResidentResidualAttentionTraining {
            graph,
            parameters,
            bound_revision: None,
            forwards: 0,
            latest: None,
        })
    }
}

impl ResidentResidualAttentionTraining {
    pub fn input_layout(&self) -> &NdLayout {
        self.graph.input_layout()
    }

    pub fn output_layout(&self) -> &NdLayout {
        self.graph.output_layout()
    }

    pub fn attention_spec(&self) -> AttentionSpec {
        self.graph.attention_spec()
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
        self.graph.validate_input(input, z_bias, pair_bias)?;
        let submission = self
            .forwards
            .checked_add(1)
            .ok_or(TrainingError::Overflow)?;
        self.latest = None;
        let parameters = self.parameters.snapshot();
        if self.bound_revision != Some(parameters.revision()) {
            self.graph.set_parameters(parameters.values())?;
            self.bound_revision = Some(parameters.revision());
        }
        let tape = self.graph.forward(input, z_bias, pair_bias)?;
        self.forwards = submission;
        self.latest = Some(submission);
        Ok(ResidentResidualAttentionForward {
            parameters,
            submission,
            tape,
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
        let gradients = self.graph.backward(&forward.tape, cotangent)?;
        let parameters = gradients.parameters;
        let bound = forward.parameters.bind_gradients(parameters.clone())?;
        Ok(ResidentResidualAttentionVjp {
            input: gradients.input,
            parameters,
            bound,
            z_bias: gradients.z_bias,
            pair_bias: gradients.pair_bias,
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
