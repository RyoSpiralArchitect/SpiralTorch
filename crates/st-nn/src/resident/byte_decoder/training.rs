use super::*;
use crate::resident::residual_attention::{ResidualAutograd, ResidualTape};
use st_backend_wgpu::{
    resident_matmul::{MatmulAccumulation, MatmulKernel, MatmulTile},
    resident_tensor::{
        embedding::{ResidentEmbeddingForward, ResidentEmbeddingIndices},
        loss::ResidentLoss,
        ResidentTensor, TensorDevice, TensorError as GpuTensorError,
    },
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
use st_kernel_contracts::classification::CrossEntropySpec;

#[derive(Clone)]
pub struct ResidentByteBatch {
    indices: ResidentEmbeddingIndices,
    targets: ResidentTensor,
}

impl ResidentByteBatch {
    pub fn targets(&self) -> &ResidentTensor {
        &self.targets
    }
    pub fn shape(&self) -> &[usize] {
        self.indices.layout().shape()
    }
}

/// Optional caller-owned score biases. A causal mask cannot prevent leakage
/// from biases computed using future bytes. The producer owns that contract.
#[derive(Clone, Copy, Default)]
pub struct ByteDecoderBias<'a> {
    pub z_bias: Option<&'a ResidentTensor>,
    pub pair_bias: Option<&'a ResidentTensor>,
}

pub struct ByteDecoderBiasGradient {
    z_bias: Option<ResidentTensor>,
    pair_bias: Option<ResidentTensor>,
}

impl ByteDecoderBiasGradient {
    pub fn z_bias(&self) -> Option<&ResidentTensor> {
        self.z_bias.as_ref()
    }
    pub fn pair_bias(&self) -> Option<&ResidentTensor> {
        self.pair_bias.as_ref()
    }
}

pub struct ResidentByteDecoderForward {
    parameters: ResidentParameterSnapshot,
    submission: u64,
    token: ResidentEmbeddingForward,
    position: ResidentEmbeddingForward,
    blocks: Vec<ResidualTape>,
    head: GraphForward,
    targets: ResidentTensor,
}

impl ResidentByteDecoderForward {
    pub fn prediction(&self) -> &ResidentTensor {
        self.head.prediction()
    }
    pub fn parameter_revision(&self) -> u64 {
        self.parameters.revision()
    }
    /// The shifted labels are captured from this forward's own byte batch.
    pub fn next_byte_loss(&self, spec: CrossEntropySpec) -> Result<ResidentLoss, InferenceError> {
        require_uncommitted_route()?;
        Ok(self
            .prediction()
            .cross_entropy_with_logits(&self.targets, spec)?)
    }
}

pub struct ResidentByteDecoderVjp {
    parameters: Vec<ResidentTensor>,
    bound: ResidentParameterGradients,
    embedding_output: ResidentTensor,
    biases: Vec<ByteDecoderBiasGradient>,
}

impl ResidentByteDecoderVjp {
    pub fn parameter_gradients(&self) -> &[ResidentTensor] {
        &self.parameters
    }
    /// Gradient of the summed byte/position vectors, not a derivative of byte IDs.
    pub fn embedding_output_gradient(&self) -> &ResidentTensor {
        &self.embedding_output
    }
    pub fn bias_gradients(&self) -> &[ByteDecoderBiasGradient] {
        &self.biases
    }
}

/// One owner/revision and one SGD decision across tables, all blocks and head.
pub struct ResidentByteDecoder {
    parameters: ResidentParameters,
    parameter_layout: ByteDecoderParameterLayout,
    positions: ResidentEmbeddingIndices,
    blocks: Vec<ResidualAutograd>,
    head: ResidentGraphAutograd,
    bound_revision: Option<u64>,
    forwards: u64,
    latest: Option<u64>,
}

impl ByteDecoderPlan {
    pub fn compile_training_wgpu(
        &self,
        runtime: WgpuRuntime,
    ) -> Result<ResidentByteDecoder, InferenceError> {
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
    ) -> Result<ResidentByteDecoder, InferenceError> {
        require_uncommitted_route()?;
        let device = TensorDevice::new(runtime.clone())?;
        let mut values = vec![
            device.upload(&self.token.shape, &self.token.values)?,
            device.upload(&self.position.shape, &self.position.values)?,
        ];
        let mut blocks = Vec::new();
        for block in &self.blocks {
            for p in block.parameter_values()? {
                values.push(device.upload(&p.shape, &p.values)?);
            }
            blocks.push(ResidualAutograd::new(
                block,
                runtime.clone(),
                tile,
                kernel,
                accumulation,
            )?);
        }
        let head = self.head.graph_definition()?;
        for p in head.parameters() {
            values.push(device.upload(&p.shape, &p.values)?);
        }
        if values.len() != self.parameters.len() {
            return Err(TrainingError::ParameterLayout.into());
        }
        let [batch, sequence, _] = <[usize; 3]>::try_from(self.input_layout().shape()).unwrap();
        let positions: Vec<_> = (0..batch).flat_map(|_| 0..sequence).collect();
        Ok(ResidentByteDecoder {
            positions: device.upload_embedding_indices(
                &[batch, sequence],
                self.position.shape[0],
                &positions,
            )?,
            parameters: ResidentParameters::new(values)?,
            parameter_layout: self.parameters.clone(),
            blocks,
            head: ResidentGraphAutograd::new(runtime, head, tile, kernel, accumulation)?,
            bound_revision: None,
            forwards: 0,
            latest: None,
        })
    }
}

impl ResidentByteDecoder {
    pub fn tensor_device(&self) -> &TensorDevice {
        self.parameters.tensor_device()
    }
    pub fn input_layout(&self) -> &NdLayout {
        self.blocks[0].input_layout()
    }
    pub fn output_layout(&self) -> &NdLayout {
        self.head.output_layout()
    }
    pub fn parameter_layout(&self) -> &ByteDecoderParameterLayout {
        &self.parameter_layout
    }
    pub fn parameter_snapshot(&self) -> ResidentParameterSnapshot {
        self.parameters.snapshot()
    }

    pub fn prepare_batch(&self, batch: &ByteLmBatch) -> Result<ResidentByteBatch, InferenceError> {
        require_uncommitted_route()?;
        if batch.shape.as_slice() != &self.input_layout().shape()[..2] {
            return Err(InferenceError::ByteDecoder(
                "batch shape differs from the compiled model",
            ));
        }
        let ids: Vec<_> = batch.input.iter().map(|&b| usize::from(b)).collect();
        let targets: Vec<_> = batch.targets.iter().map(|&b| f32::from(b)).collect();
        Ok(ResidentByteBatch {
            indices: self.tensor_device().upload_embedding_indices(
                &batch.shape,
                BYTE_LM_VOCAB,
                &ids,
            )?,
            targets: self.tensor_device().upload(&batch.shape, &targets)?,
        })
    }

    pub fn forward(
        &mut self,
        batch: &ResidentByteBatch,
    ) -> Result<ResidentByteDecoderForward, InferenceError> {
        self.forward_with_external_biases(
            batch,
            &vec![ByteDecoderBias::default(); self.blocks.len()],
        )
    }

    /// Advanced seam: causality is conditional on each external bias producer.
    /// Use `forward` for the self-contained causal byte model.
    pub fn forward_with_external_biases(
        &mut self,
        batch: &ResidentByteBatch,
        biases: &[ByteDecoderBias<'_>],
    ) -> Result<ResidentByteDecoderForward, InferenceError> {
        require_uncommitted_route()?;
        if batch.shape() != &self.input_layout().shape()[..2] || biases.len() != self.blocks.len() {
            return Err(InferenceError::ByteDecoder(
                "batch or bias count differs from the compiled model",
            ));
        }
        if !batch
            .indices
            .device()
            .runtime()
            .context()
            .shares_handles_with(self.tensor_device().runtime().context())
        {
            return Err(GpuTensorError::DeviceMismatch.into());
        }
        for (block, bias) in self.blocks.iter().zip(biases) {
            block.validate_biases(bias.z_bias, bias.pair_bias)?;
        }
        let submission = self
            .forwards
            .checked_add(1)
            .ok_or(TrainingError::Overflow)?;
        self.latest = None;
        let parameters = self.parameters.snapshot();
        if self.bound_revision != Some(parameters.revision()) {
            for (block, range) in self.blocks.iter_mut().zip(self.parameter_layout.blocks()) {
                block.set_parameters(&parameters.values()[range.clone()])?;
            }
            self.head
                .set_parameter_tensors(&parameters.values()[self.parameter_layout.head()])?;
            self.bound_revision = Some(parameters.revision());
        }
        let token = parameters.values()[0].embedding(&batch.indices)?;
        let position = parameters.values()[1].embedding(&self.positions)?;
        let mut input = token.prediction().add(position.prediction())?;
        let mut blocks = Vec::new();
        for (block, bias) in self.blocks.iter_mut().zip(biases) {
            let tape = block.forward(&input, bias.z_bias, bias.pair_bias)?;
            input = tape.prediction().clone();
            blocks.push(tape);
        }
        self.head.set_input_tensor(&input)?;
        let head = self.head.forward()?;
        self.forwards = submission;
        self.latest = Some(submission);
        Ok(ResidentByteDecoderForward {
            parameters,
            submission,
            token,
            position,
            blocks,
            head,
            targets: batch.targets.clone(),
        })
    }

    pub fn backward(
        &mut self,
        forward: &ResidentByteDecoderForward,
        cotangent: &ResidentTensor,
    ) -> Result<ResidentByteDecoderVjp, InferenceError> {
        require_uncommitted_route()?;
        if self.latest != Some(forward.submission)
            || !self.parameters.is_current(&forward.parameters)
        {
            return Err(TrainingError::StaleForward.into());
        }
        let guarded = self
            .tensor_device()
            .guard_together(&[cotangent, forward.prediction()])?;
        let head = self.head.backward(&forward.head, &guarded[0])?;
        let mut input = head.input_gradient().clone();
        let mut block_parameters = Vec::new();
        let mut biases = Vec::new();
        for (block, tape) in self.blocks.iter_mut().zip(&forward.blocks).rev() {
            let gradient = block.backward(tape, &input)?;
            input = gradient.input;
            block_parameters.push(gradient.parameters);
            biases.push(ByteDecoderBiasGradient {
                z_bias: gradient.z_bias,
                pair_bias: gradient.pair_bias,
            });
        }
        block_parameters.reverse();
        biases.reverse();
        let mut parameters = Vec::new();
        parameters.extend(block_parameters.into_iter().flatten());
        parameters.extend(head.parameter_gradients().iter().cloned());
        self.bind_vjp(forward, input, parameters, biases)
    }

    fn bind_vjp(
        &self,
        forward: &ResidentByteDecoderForward,
        input: ResidentTensor,
        trailing_parameters: Vec<ResidentTensor>,
        mut biases: Vec<ByteDecoderBiasGradient>,
    ) -> Result<ResidentByteDecoderVjp, InferenceError> {
        let mut parameters = vec![
            forward.token.backward(&input)?,
            forward.position.backward(&input)?,
        ];
        parameters.extend(trailing_parameters);
        let count = parameters.len();
        if count != self.parameter_layout.len() {
            return Err(TrainingError::ParameterLayout.into());
        }
        // Join AFTER embedding accumulation: a late repeated-ID overflow must
        // invalidate every earlier head/block/input/geometry derivative too.
        let mut family = parameters;
        family.push(input);
        for bias in &biases {
            family.extend(bias.z_bias.iter().cloned());
            family.extend(bias.pair_bias.iter().cloned());
        }
        let mut family = self
            .tensor_device()
            .guard_together(&family.iter().collect::<Vec<_>>())?
            .into_iter();
        let parameters: Vec<_> = family.by_ref().take(count).collect();
        let embedding_output = family.next().unwrap();
        for bias in &mut biases {
            if bias.z_bias.is_some() {
                bias.z_bias = family.next();
            }
            if bias.pair_bias.is_some() {
                bias.pair_bias = family.next();
            }
        }
        let bound = forward.parameters.bind_gradients(parameters.clone())?;
        Ok(ResidentByteDecoderVjp {
            parameters,
            bound,
            embedding_output,
            biases,
        })
    }

    pub fn sgd(
        &mut self,
        gradients: &ResidentByteDecoderVjp,
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
