//! Forward-only mixed NN graphs. Parameters, bindings and activations are
//! prepared once; each dispatch submits the whole graph without host transfers.

use crate::{
    resident_dense::{DenseDispatch, DenseError, DenseKernel},
    resident_matmul::{MatmulAccumulation, MatmulKernel, MatmulShape, MatmulTile},
    resident_tensor::{
        pointwise::PointwisePlan, storage_limit, ResidentTensor, TensorDevice, TensorError,
    },
    runtime::{self, WgpuContext, WgpuRuntime, WgpuRuntimeError},
};
use st_kernel_contracts::{
    graph::{GraphDefinition, GraphStage},
    layout::NdLayout,
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GraphInferenceError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Kernel(#[from] DenseError),
    #[error(transparent)]
    Runtime(#[from] WgpuRuntimeError),
    #[error("graph input requires {expected} elements, received {actual}")]
    InputLength { expected: usize, actual: usize },
    #[error("resident input shape must equal the graph's logical N-D input shape")]
    InputShape,
    #[error("upload an input before graph dispatch")]
    MissingInput,
    #[error("dispatch the current input before capturing graph output")]
    StaleOutput,
    #[error("graph generation or dispatch counter exhausted")]
    CounterOverflow,
    #[error(
        "non-finite graph stage {stage}, flag mask {flags:#x}; stage_count denotes upstream input"
    )]
    NonFinite { stage: usize, flags: u32 },
    #[error("invalid graph readback")]
    Readback,
}

enum Node {
    Linear(DenseDispatch),
    Pointwise {
        plan: Box<PointwisePlan>,
        binding: wgpu::BindGroup,
        flags: wgpu::Buffer,
    },
}

/// An inference workspace, not a training workspace with zero learning rate.
/// It has no target, gradient, loss, optimizer or preactivation-tape buffers.
pub struct ResidentGraph {
    definition: GraphDefinition,
    activations: Vec<wgpu::Buffer>,
    nodes: Vec<Node>,
    kernel: Option<DenseKernel>,
    validation: wgpu::Buffer,
    readbacks: runtime::ReadbackPool,
    generation: u64,
    submitted_dispatches: u64,
    output_generation: Option<u64>,
    input_source: Option<ResidentTensor>,
    // Retire device resources before their owning runtime.
    device: TensorDevice,
}

impl ResidentGraph {
    pub fn new(
        runtime: WgpuRuntime,
        definition: GraphDefinition,
        tile: MatmulTile,
        kernel: MatmulKernel,
        accumulation: MatmulAccumulation,
    ) -> Result<Self, GraphInferenceError> {
        let device = TensorDevice::new(runtime)?;
        let context = device.runtime().context();
        let gpu = context.device();
        let limits = gpu.limits();
        for len in definition
            .layouts()
            .iter()
            .map(NdLayout::len)
            .chain(definition.parameters().iter().map(|p| p.values.len()))
            .chain([definition.stages().len() + 1])
        {
            storage_limit(len, &limits)?;
        }
        let rows =
            definition.input_layout().len() / definition.input_layout().shape().last().unwrap();
        let shapes = definition
            .stages()
            .iter()
            .map(|stage| match stage {
                GraphStage::Linear { weight, .. } => {
                    let dims = &definition.parameters()[*weight].shape;
                    let shape =
                        MatmulShape::new(rows, dims[0], dims[1]).map_err(DenseError::from)?;
                    shape
                        .validate(&limits, tile, kernel)
                        .map_err(DenseError::from)?;
                    Ok(Some(shape))
                }
                GraphStage::Pointwise { .. } => Ok(None),
            })
            .collect::<Result<Vec<_>, DenseError>>()?;
        let kernel = if shapes.iter().any(Option::is_some) {
            Some(DenseKernel::new(gpu, tile, kernel, accumulation)?)
        } else {
            None
        };
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let activations = definition
            .layouts()
            .iter()
            .map(|layout| {
                runtime::empty_buffer::<f32>(gpu, "graph.forward.activation", layout.len(), usage)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let parameters = definition
            .parameters()
            .iter()
            .map(|p| {
                runtime::upload_slice(
                    gpu,
                    "graph.forward.parameter",
                    &p.values,
                    wgpu::BufferUsages::STORAGE,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let validation = runtime::empty_buffer::<u32>(
            gpu,
            "graph.forward.validation",
            definition.stages().len() + 1,
            usage,
        )?;
        let empty_flags = runtime::upload_slice(
            gpu,
            "graph.forward.empty_flags",
            &[0u32],
            wgpu::BufferUsages::STORAGE,
        )?;
        let unused = runtime::upload_slice(
            gpu,
            "graph.forward.unused",
            &[0f32],
            wgpu::BufferUsages::STORAGE,
        )?;
        let mut nodes = Vec::with_capacity(definition.stages().len());
        for (i, stage) in definition.stages().iter().enumerate() {
            nodes.push(match stage {
                GraphStage::Linear { weight, bias, gelu } => {
                    Node::Linear(kernel.as_ref().unwrap().bind(
                        gpu,
                        shapes[i].unwrap(),
                        &activations[i],
                        &activations[i + 1],
                        &parameters[*weight],
                        &parameters[*bias],
                        &unused,
                        &validation,
                        i as u32,
                        *gelu,
                    )?)
                }
                GraphStage::Pointwise {
                    chain,
                    parameters: ids,
                } => {
                    let mut layouts = vec![definition.layouts()[i].clone()];
                    for &id in ids {
                        layouts.push(
                            NdLayout::contiguous(&definition.parameters()[id].shape)
                                .map_err(TensorError::from)?,
                        );
                    }
                    let plan =
                        Box::new(PointwisePlan::new(device.clone(), chain.clone(), layouts)?);
                    let inputs: Vec<_> = std::iter::once(&activations[i])
                        .chain(ids.iter().map(|&id| &parameters[id]))
                        .collect();
                    let flags = runtime::empty_buffer::<u32>(
                        gpu,
                        "graph.forward.pointwise_flags",
                        1,
                        usage,
                    )?;
                    let binding =
                        plan.bind_into(&inputs, &activations[i + 1], &empty_flags, &flags);
                    Node::Pointwise {
                        plan,
                        binding,
                        flags,
                    }
                }
            });
        }
        let readbacks = runtime::ReadbackPool::new::<u32>(
            context.clone(),
            definition
                .output_layout()
                .len()
                .checked_add(nodes.len() + 1)
                .ok_or(TensorError::Limit("graph readback"))?,
        )?;
        Ok(Self {
            definition,
            activations,
            nodes,
            kernel,
            validation,
            readbacks,
            generation: 0,
            submitted_dispatches: 0,
            output_generation: None,
            input_source: None,
            device,
        })
    }

    pub fn input_layout(&self) -> &NdLayout {
        self.definition.input_layout()
    }
    pub fn output_layout(&self) -> &NdLayout {
        self.definition.output_layout()
    }
    pub fn stage_count(&self) -> usize {
        self.nodes.len()
    }
    pub fn parameter_count(&self) -> usize {
        self.definition.parameters().len()
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }
    pub fn submitted_dispatches(&self) -> u64 {
        self.submitted_dispatches
    }
    pub fn tensor_device(&self) -> &TensorDevice {
        &self.device
    }
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        self.device.runtime().adapter_info()
    }

    /// Validate the whole host input before replacing the current generation.
    pub fn upload(&mut self, values: &[f32]) -> Result<(), GraphInferenceError> {
        if values.len() != self.input_layout().len() {
            return Err(GraphInferenceError::InputLength {
                expected: self.input_layout().len(),
                actual: values.len(),
            });
        }
        if !values.iter().all(|v| v.is_finite()) {
            return Err(TensorError::NonFinite.into());
        }
        let generation = self
            .generation
            .checked_add(1)
            .ok_or(GraphInferenceError::CounterOverflow)?;
        self.device.runtime().context().queue().write_buffer(
            &self.activations[0],
            0,
            bytemuck::cast_slice(values),
        );
        self.generation = generation;
        self.output_generation = None;
        self.input_source = None;
        Ok(())
    }

    /// Accept a strided/broadcast view on the same queue. Packing and the copy
    /// into stable workspace storage are GPU-only; inherited errors stay deferred.
    pub fn set_input_tensor(&mut self, input: &ResidentTensor) -> Result<(), GraphInferenceError> {
        input.require_context(self.device.runtime().context())?;
        if input.layout().shape() != self.input_layout().shape() {
            return Err(GraphInferenceError::InputShape);
        }
        let generation = self
            .generation
            .checked_add(1)
            .ok_or(GraphInferenceError::CounterOverflow)?;
        let input = input.contiguous()?;
        let context = self.device.runtime().context();
        let mut encoder = context.device().create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(
            input.values(),
            0,
            &self.activations[0],
            0,
            self.activations[0].size(),
        );
        context.queue().submit(Some(encoder.finish()));
        self.generation = generation;
        self.output_generation = None;
        self.input_source = Some(input);
        Ok(())
    }

    /// One submission, no per-dispatch buffer/binding allocation or readback.
    /// Returns the dispatch counter, which advances even for later-rejected output.
    pub fn dispatch(&mut self) -> Result<u64, GraphInferenceError> {
        if self.generation == 0 {
            return Err(GraphInferenceError::MissingInput);
        }
        let dispatch = self
            .submitted_dispatches
            .checked_add(1)
            .ok_or(GraphInferenceError::CounterOverflow)?;
        let context = self.device.runtime().context();
        let mut encoder = context.device().create_command_encoder(&Default::default());
        encoder.clear_buffer(&self.validation, 0, None);
        if let Some(input) = &self.input_source {
            encoder.copy_buffer_to_buffer(
                input.flags(),
                0,
                &self.validation,
                self.nodes.len() as u64 * 4,
                4,
            );
        }
        for (i, node) in self.nodes.iter().enumerate() {
            match node {
                Node::Linear(binding) => {
                    self.kernel.as_ref().unwrap().encode(&mut encoder, binding)
                }
                Node::Pointwise {
                    plan,
                    binding,
                    flags,
                } => {
                    encoder.clear_buffer(flags, 0, None);
                    plan.encode_bound(&mut encoder, binding);
                    // Preserve each stage's guard even when a later operation masks overflow.
                    encoder.copy_buffer_to_buffer(flags, 0, &self.validation, i as u64 * 4, 4);
                }
            }
        }
        context.queue().submit(Some(encoder.finish()));
        self.output_generation = Some(self.generation);
        self.submitted_dispatches = dispatch;
        Ok(dispatch)
    }

    fn require_output(&self) -> Result<(), GraphInferenceError> {
        if self.output_generation != Some(self.generation) {
            return Err(GraphInferenceError::StaleOutput);
        }
        Ok(())
    }

    /// Freeze values and all stage guards together. Reading is explicitly separate.
    pub fn snapshot(&self) -> Result<GraphReadback, GraphInferenceError> {
        self.require_output()?;
        let context = self.device.runtime().context();
        let staging = self.readbacks.checkout("graph.forward.snapshot");
        let output = self.activations.last().unwrap();
        let mut encoder = context.device().create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(output, 0, staging.buffer(), 0, output.size());
        encoder.copy_buffer_to_buffer(
            &self.validation,
            0,
            staging.buffer(),
            output.size(),
            self.validation.size(),
        );
        context.queue().submit(Some(encoder.finish()));
        Ok(GraphReadback {
            staging,
            layout: self.output_layout().clone(),
            stages: self.nodes.len(),
            generation: self.generation,
            dispatch: self.submitted_dispatches,
            context: context.clone(),
        })
    }

    /// An immutable GPU-only capture for further N-D operations or another graph.
    /// This is an on-device copy, not a mutable alias or a CPU readback.
    pub fn output_tensor(&self) -> Result<ResidentTensor, GraphInferenceError> {
        self.require_output()?;
        Ok(self.device.capture(
            self.output_layout(),
            self.activations.last().unwrap(),
            &self.validation,
        )?)
    }
}

pub struct GraphReadback {
    staging: runtime::ReadbackLease,
    layout: NdLayout,
    stages: usize,
    generation: u64,
    dispatch: u64,
    context: WgpuContext,
}

impl GraphReadback {
    pub fn layout(&self) -> &NdLayout {
        &self.layout
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }
    pub fn dispatch(&self) -> u64 {
        self.dispatch
    }

    fn decode(bytes: &[u8], len: usize, stages: usize) -> Result<Vec<f32>, GraphInferenceError> {
        if len
            .checked_add(stages)
            .and_then(|n| n.checked_add(1))
            .and_then(|n| n.checked_mul(4))
            != Some(bytes.len())
        {
            return Err(GraphInferenceError::Readback);
        }
        for stage in 0..=stages {
            let start = (len + stage) * 4;
            let flags = u32::from_le_bytes(bytes[start..start + 4].try_into().unwrap());
            if flags != 0 {
                return Err(GraphInferenceError::NonFinite { stage, flags });
            }
        }
        #[allow(clippy::chunks_exact_to_as_chunks)]
        let values: Vec<_> = bytes[..len * 4]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        if !values.iter().all(|v| v.is_finite()) {
            return Err(TensorError::NonFinite.into());
        }
        Ok(values)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn read(mut self) -> Result<Vec<f32>, GraphInferenceError> {
        let bytes = self.staging.read(
            &self.context,
            std::time::Duration::from_secs(30),
            "graph.forward.snapshot",
        )?;
        Self::decode(&bytes, self.layout.len(), self.stages)
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn read_async(self) -> Result<Vec<f32>, GraphInferenceError> {
        let bytes = self
            .staging
            .read_async(self.context, "graph.forward.snapshot")
            .await?;
        Self::decode(&bytes, self.layout.len(), self.stages)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn readback_checks_all_stages_and_upstream_guard() {
        let mut words = vec![1f32.to_bits(), 2f32.to_bits(), 0, 0, 0];
        assert_eq!(
            GraphReadback::decode(bytemuck::cast_slice(&words), 2, 2).unwrap(),
            [1., 2.]
        );
        for stage in 0..=2 {
            words[2 + stage] = 0x80000000;
            assert!(
                matches!(GraphReadback::decode(bytemuck::cast_slice(&words), 2, 2),
                Err(GraphInferenceError::NonFinite { stage: s, .. }) if s == stage)
            );
            words[2 + stage] = 0;
        }
        assert!(GraphReadback::decode(&[], 2, 2).is_err());
        assert!(GraphReadback::decode(&[], usize::MAX, 2).is_err());
        words[0] = f32::NAN.to_bits();
        assert!(GraphReadback::decode(bytemuck::cast_slice(&words), 2, 2).is_err());
    }
}
