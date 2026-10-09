//! Reusable attention execution without an optimizer or parameter owner.
use super::*;
use st_backend_wgpu::{
    resident_matmul::{MatmulAccumulation, MatmulKernel, MatmulTile},
    resident_tensor::{ResidentTensor, TensorDevice},
    resident_training::{
        graph::{GraphForward, ResidentGraphAutograd},
        TrainingError,
    },
    runtime::WgpuRuntime,
};

pub(crate) struct AttentionTape {
    qkv: GraphForward,
    output: GraphForward,
    heads: [ResidentTensor; 3],
    z_bias: Option<ResidentTensor>,
    pair_bias: Option<ResidentTensor>,
}

impl AttentionTape {
    pub(crate) fn prediction(&self) -> &ResidentTensor {
        self.output.prediction()
    }
}

pub(crate) struct AttentionGradients {
    pub input: ResidentTensor,
    pub parameters: Vec<ResidentTensor>,
    pub z_bias: Option<ResidentTensor>,
    pub pair_bias: Option<ResidentTensor>,
}

pub(crate) struct AttentionAutograd {
    qkv: ResidentGraphAutograd,
    output: ResidentGraphAutograd,
    spec: AttentionSpec,
}

impl AttentionInferencePlan {
    pub(crate) fn graph_parts(&self) -> Result<[GraphDefinition; 2], InferenceError> {
        Ok([
            self.qkv.graph_definition()?,
            self.output.graph_definition()?,
        ])
    }
}

impl AttentionAutograd {
    pub(crate) fn new(
        plan: &AttentionInferencePlan,
        runtime: WgpuRuntime,
        tile: MatmulTile,
        kernel: MatmulKernel,
        accumulation: MatmulAccumulation,
    ) -> Result<Self, InferenceError> {
        st_backend_wgpu::resident_tensor::attention::validate_limits(
            plan.spec,
            &runtime.context().device().limits(),
        )?;
        let [qkv, output] = plan.graph_parts()?;
        Ok(Self {
            qkv: ResidentGraphAutograd::new(runtime.clone(), qkv, tile, kernel, accumulation)?,
            output: ResidentGraphAutograd::new(runtime, output, tile, kernel, accumulation)?,
            spec: plan.spec,
        })
    }

    pub(crate) fn input_layout(&self) -> &NdLayout {
        self.qkv.input_layout()
    }

    pub(crate) fn output_layout(&self) -> &NdLayout {
        self.output.output_layout()
    }

    pub(crate) fn tensor_device(&self) -> &TensorDevice {
        self.qkv.tensor_device()
    }

    pub(crate) fn attention_spec(&self) -> AttentionSpec {
        self.spec
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
        Ok(())
    }

    // Only model-owned snapshots with fixed shapes reach this private boundary.
    pub(crate) fn set_parameters(
        &mut self,
        values: &[ResidentTensor],
    ) -> Result<(), InferenceError> {
        if values.len() != 4 {
            return Err(TrainingError::ParameterLayout.into());
        }
        self.qkv.set_parameter_tensors(&values[..2])?;
        self.output.set_parameter_tensors(&values[2..])?;
        Ok(())
    }

    pub(crate) fn forward(
        &mut self,
        input: &ResidentTensor,
        z_bias: Option<&ResidentTensor>,
        pair_bias: Option<&ResidentTensor>,
    ) -> Result<AttentionTape, InferenceError> {
        self.validate_input(input, z_bias, pair_bias)?;
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
        Ok(AttentionTape {
            qkv,
            output: self.output.forward()?,
            heads,
            z_bias: z_bias.cloned(),
            pair_bias: pair_bias.cloned(),
        })
    }

    pub(crate) fn backward(
        &mut self,
        tape: &AttentionTape,
        cotangent: &ResidentTensor,
    ) -> Result<AttentionGradients, InferenceError> {
        let output = self.output.backward(&tape.output, cotangent)?;
        let [batch, heads, sequence, dim] = self.spec.query_shape();
        let seed = output
            .input_gradient()
            .reshape(&[batch, sequence, heads, dim])?
            .permute(&[0, 2, 1, 3])?;
        let attention = tape.heads[0].scaled_dot_attention_vjp(
            &tape.heads[1],
            &tape.heads[2],
            &seed,
            self.spec.scale(),
            self.spec.mask(),
            tape.z_bias.as_ref(),
            tape.pair_bias.as_ref(),
        )?;
        let [query, key, value] =
            [&attention.query, &attention.key, &attention.value].map(|g| g.permute(&[0, 2, 1, 3]));
        let joined = ResidentTensor::concatenate(&[&query?, &key?, &value?], 2)?
            .reshape(self.qkv.output_layout().shape())?;
        let qkv = self.qkv.backward(&tape.qkv, &joined)?;
        Ok(AttentionGradients {
            input: qkv.input_gradient().clone(),
            parameters: qkv
                .parameter_gradients()
                .iter()
                .chain(output.parameter_gradients())
                .cloned()
                .collect(),
            z_bias: attention.z_bias,
            pair_bias: attention.pair_bias,
        })
    }
}
