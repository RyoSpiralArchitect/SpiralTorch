//! A checked dense chain: persistent weights and activations, one submission,
//! no intermediate copies or readbacks. Shared by native and browser clients.

use crate::{
    resident_matmul::{MatmulError, MatmulShape, MatmulTile},
    resident_tensor::{ResidentTensor, TensorDevice, TensorError},
    runtime::{self, WgpuContext, WgpuRuntime, WgpuRuntimeError},
    shader_sources::{checked_dense_matmul_source, MatmulAccumulation, MatmulKernel},
};
use bytemuck::{Pod, Zeroable};
use st_kernel_contracts::layout::{NdLayout, NdLayoutError};
use thiserror::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenseActivation {
    None,
    Gelu,
}

/// Owned parameter snapshot. No live host alias is read after construction.
#[derive(Clone, Debug)]
pub struct DenseLayer {
    pub inner: usize,
    pub cols: usize,
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
    pub activation: DenseActivation,
}

#[derive(Debug, Error)]
pub enum DenseError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error("dense inference requires a nonempty contiguous last-axis layout at offset zero")]
    InvalidLayout,
    #[error("dense chain requires at least one layer and u32-addressable stage count")]
    EmptyChain,
    #[error("invalid dimensions or parameter lengths at stage {0}")]
    InvalidStage(usize),
    #[error("non-finite host value in {0}")]
    NonFiniteInput(&'static str),
    #[error("input requires {expected} elements, received {actual}")]
    InputLength { expected: usize, actual: usize },
    #[error("upload an input before dispatch")]
    MissingInput,
    #[error("dispatch the current input before requesting a snapshot")]
    StaleOutput,
    #[error("input generation exhausted")]
    GenerationOverflow,
    #[error("non-finite intermediate at dense stage {stage}, flag mask {flags:#x}")]
    NonFiniteIntermediate { stage: usize, flags: u32 },
    #[error("invalid dense readback length")]
    InvalidReadback,
    #[error("non-finite final output despite empty device validation flags")]
    NonFiniteOutput,
    #[error(transparent)]
    Layout(#[from] NdLayoutError),
    #[error(transparent)]
    Matmul(#[from] MatmulError),
    #[error(transparent)]
    Runtime(#[from] WgpuRuntimeError),
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct Uniforms {
    pub rows: u32,
    pub cols: u32,
    pub inner: u32,
    pub flags: u32,
    pub output_scale: f32,
    pub validation_index: u32,
    pub padding: [u32; 2],
}

pub(crate) fn validate_layers(
    limits: &wgpu::Limits,
    input_layout: &NdLayout,
    layers: &[DenseLayer],
    tile: MatmulTile,
    kernel: MatmulKernel,
) -> Result<(Vec<MatmulShape>, NdLayout), DenseError> {
    if input_layout.rank() == 0
        || input_layout.is_empty()
        || !input_layout.is_contiguous()
        || input_layout.offset() != 0
    {
        return Err(DenseError::InvalidLayout);
    }
    if layers.is_empty() || u32::try_from(layers.len()).is_err() {
        return Err(DenseError::EmptyChain);
    }
    let mut width = *input_layout.shape().last().unwrap();
    let rows = input_layout.len() / width;
    let mut shapes = Vec::with_capacity(layers.len());
    for (i, layer) in layers.iter().enumerate() {
        if layer.inner != width
            || layer.inner.checked_mul(layer.cols) != Some(layer.weights.len())
            || layer.bias.len() != layer.cols
        {
            return Err(DenseError::InvalidStage(i));
        }
        if !layer
            .weights
            .iter()
            .chain(&layer.bias)
            .all(|v| v.is_finite())
        {
            return Err(DenseError::NonFiniteInput("parameters"));
        }
        let shape = MatmulShape::new(rows, width, layer.cols)?;
        shape.validate(limits, tile, kernel)?;
        shapes.push(shape);
        width = layer.cols;
    }
    let mut dimensions = input_layout.shape().to_vec();
    *dimensions.last_mut().unwrap() = width;
    Ok((shapes, NdLayout::contiguous(&dimensions)?))
}

pub(crate) fn dense_layout(device: &wgpu::Device, tape: bool) -> wgpu::BindGroupLayout {
    let entries: Vec<_> = (0..if tape { 9 } else { 8 })
        .map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: if binding == 6 {
                    wgpu::BufferBindingType::Uniform
                } else {
                    wgpu::BufferBindingType::Storage {
                        read_only: ![2, 7, 8].contains(&binding),
                    }
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("dense.bindings"),
        entries: &entries,
    })
}

pub(crate) fn dense_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    source: String,
) -> wgpu::ComputePipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("dense.layout"),
        bind_group_layouts: &[layout],
        push_constant_ranges: &[],
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("dense.checked.shader"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("dense.checked"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: "main",
        compilation_options: Default::default(),
    })
}

/// Shared forward-only kernel. Both dense chains and mixed graphs bind their
/// persistent activation buffers here; neither needs a backward tape.
pub(crate) struct DenseKernel {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    tile: MatmulTile,
}

pub(crate) struct DenseDispatch {
    binding: wgpu::BindGroup,
    groups: [u32; 2],
}

impl DenseKernel {
    pub(crate) fn new(
        device: &wgpu::Device,
        tile: MatmulTile,
        kernel: MatmulKernel,
        accumulation: MatmulAccumulation,
    ) -> Result<Self, DenseError> {
        let limits = device.limits();
        if limits.max_storage_buffers_per_shader_stage < 7 || limits.max_bindings_per_bind_group < 8
        {
            return Err(MatmulError::DeviceLimit("checked dense bindings").into());
        }
        let layout = dense_layout(device, false);
        let source = checked_dense_matmul_source(tile.dimensions(), kernel, accumulation)
            .map_err(|_| MatmulError::UnsupportedKernelTile)?;
        let pipeline = dense_pipeline(device, &layout, source);
        Ok(Self {
            layout,
            pipeline,
            tile,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn bind(
        &self,
        device: &wgpu::Device,
        shape: MatmulShape,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        weight: &wgpu::Buffer,
        bias: &wgpu::Buffer,
        unused: &wgpu::Buffer,
        validation: &wgpu::Buffer,
        stage: u32,
        gelu: bool,
    ) -> Result<DenseDispatch, DenseError> {
        let (rows, inner, cols) = shape.dimensions();
        let params = Uniforms {
            rows: rows as u32,
            cols: cols as u32,
            inner: inner as u32,
            flags: 1 | if gelu { 4 } else { 0 },
            output_scale: 1.0,
            validation_index: stage,
            padding: [0; 2],
        };
        let uniform = runtime::upload_slice(
            device,
            "dense.params",
            &[params],
            wgpu::BufferUsages::UNIFORM,
        )?;
        let resources = [
            input, weight, output, bias, unused, unused, &uniform, validation,
        ];
        let entries: Vec<_> = resources
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let binding = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dense.stage"),
            layout: &self.layout,
            entries: &entries,
        });
        let [tm, tn, _] = self.tile.dimensions();
        Ok(DenseDispatch {
            binding,
            groups: [(cols as u32).div_ceil(tn), (rows as u32).div_ceil(tm)],
        })
    }

    pub(crate) fn encode(&self, encoder: &mut wgpu::CommandEncoder, dispatch: &DenseDispatch) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("dense.stage.pass"),
            timestamp_writes: None,
        });
        self.encode_in_pass(&mut pass, dispatch);
    }

    pub(crate) fn encode_in_pass<'a>(
        &'a self,
        pass: &mut wgpu::ComputePass<'a>,
        dispatch: &'a DenseDispatch,
    ) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &dispatch.binding, &[]);
        pass.dispatch_workgroups(dispatch.groups[0], dispatch.groups[1], 1);
    }
}

pub struct ResidentDense {
    buffers: Vec<wgpu::Buffer>,
    validation: wgpu::Buffer,
    bindings: Vec<DenseDispatch>,
    kernel: DenseKernel,
    readbacks: runtime::ReadbackPool,
    input_layout: NdLayout,
    output_layout: NdLayout,
    shapes: Vec<MatmulShape>,
    generation: u64,
    input_ready: bool,
    output_generation: Option<u64>,
    input_source: Option<ResidentTensor>,
    // Retire all resources before the last owning device handle.
    runtime: WgpuRuntime,
}

impl ResidentDense {
    pub fn new(
        runtime: WgpuRuntime,
        input_layout: NdLayout,
        layers: &[DenseLayer],
        tile: MatmulTile,
        kernel: MatmulKernel,
        accumulation: MatmulAccumulation,
    ) -> Result<Self, DenseError> {
        let context = runtime.context();
        let device = context.device();
        let limits = device.limits();
        if limits.max_storage_buffers_per_shader_stage < 7 || limits.max_bindings_per_bind_group < 8
        {
            return Err(MatmulError::DeviceLimit("checked dense bindings").into());
        }
        let (shapes, output_layout) =
            validate_layers(&limits, &input_layout, layers, tile, kernel)?;
        let snapshot_len = output_layout
            .len()
            .checked_add(layers.len())
            .ok_or(NdLayoutError::Overflow)?;
        let readbacks = runtime::ReadbackPool::new::<u32>(context.clone(), snapshot_len)?;
        let storage = wgpu::BufferUsages::STORAGE;
        let mut buffers = vec![runtime::empty_buffer::<f32>(
            device,
            "dense.input",
            input_layout.len(),
            storage | wgpu::BufferUsages::COPY_DST,
        )?];
        for shape in &shapes {
            let (rows, _, cols) = shape.dimensions();
            buffers.push(runtime::empty_buffer::<f32>(
                device,
                "dense.activation",
                rows * cols,
                storage | wgpu::BufferUsages::COPY_SRC,
            )?);
        }
        let validation = runtime::empty_buffer::<u32>(
            device,
            "dense.validation",
            layers.len(),
            storage | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        )?;
        let kernel = DenseKernel::new(device, tile, kernel, accumulation)?;
        let dummy = runtime::upload_slice(device, "dense.unused", &[0.0f32], storage)?;
        let mut bindings = Vec::with_capacity(layers.len());
        for (i, layer) in layers.iter().enumerate() {
            let weights = runtime::upload_slice(device, "dense.weights", &layer.weights, storage)?;
            let bias = runtime::upload_slice(device, "dense.bias", &layer.bias, storage)?;
            // The producer output IS the consumer input, not a device-to-device copy.
            bindings.push(kernel.bind(
                device,
                shapes[i],
                &buffers[i],
                &buffers[i + 1],
                &weights,
                &bias,
                &dummy,
                &validation,
                i as u32,
                layer.activation == DenseActivation::Gelu,
            )?);
        }
        Ok(Self {
            buffers,
            validation,
            bindings,
            kernel,
            readbacks,
            input_layout,
            output_layout,
            shapes,
            generation: 0,
            input_ready: false,
            output_generation: None,
            input_source: None,
            runtime,
        })
    }

    pub fn input_layout(&self) -> &NdLayout {
        &self.input_layout
    }
    pub fn output_layout(&self) -> &NdLayout {
        &self.output_layout
    }
    pub fn stage_count(&self) -> usize {
        self.shapes.len()
    }
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        self.runtime.adapter_info()
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn upload(&mut self, values: &[f32]) -> Result<(), DenseError> {
        if values.len() != self.input_layout.len() {
            return Err(DenseError::InputLength {
                expected: self.input_layout.len(),
                actual: values.len(),
            });
        }
        if !values.iter().all(|value| value.is_finite()) {
            return Err(DenseError::NonFiniteInput("activations"));
        }
        let generation = self
            .generation
            .checked_add(1)
            .ok_or(DenseError::GenerationOverflow)?;
        self.runtime.context().queue().write_buffer(
            &self.buffers[0],
            0,
            bytemuck::cast_slice(values),
        );
        self.generation = generation;
        self.input_source = None;
        self.input_ready = true;
        self.output_generation = None;
        Ok(())
    }

    /// Copy a logical N-D view on-device, retaining its deferred finite guard.
    /// There is no host transfer; shape/device failures leave the old input intact.
    pub fn set_input_tensor(&mut self, input: &ResidentTensor) -> Result<(), DenseError> {
        input.require_context(self.runtime.context())?;
        if input.layout().shape() != self.input_layout.shape() {
            return Err(DenseError::InvalidLayout);
        }
        let generation = self
            .generation
            .checked_add(1)
            .ok_or(DenseError::GenerationOverflow)?;
        let input = input.contiguous()?;
        let context = self.runtime.context();
        let mut encoder = context.device().create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(
            input.values(),
            0,
            &self.buffers[0],
            0,
            self.input_layout.len() as u64 * 4,
        );
        context.queue().submit(Some(encoder.finish()));
        self.input_source = Some(input);
        self.generation = generation;
        self.input_ready = true;
        self.output_generation = None;
        Ok(())
    }

    /// Enqueue the complete chain. Errors detected on-device are returned by readback.
    pub fn dispatch(&mut self) -> Result<u64, DenseError> {
        if !self.input_ready {
            return Err(DenseError::MissingInput);
        }
        let context = self.runtime.context();
        let mut encoder = context.device().create_command_encoder(&Default::default());
        encoder.clear_buffer(&self.validation, 0, None);
        if let Some(input) = &self.input_source {
            encoder.copy_buffer_to_buffer(input.flags(), 0, &self.validation, 0, 4);
        }
        for binding in &self.bindings {
            self.kernel.encode(&mut encoder, binding);
        }
        context.queue().submit(Some(encoder.finish()));
        self.output_generation = Some(self.generation);
        Ok(self.generation)
    }

    /// Snapshot output and every stage's error flags together, at request time.
    pub fn snapshot(&self) -> Result<DenseReadback, DenseError> {
        if self.output_generation != Some(self.generation) {
            return Err(DenseError::StaleOutput);
        }
        let context = self.runtime.context();
        let staging = self.readbacks.checkout("dense.snapshot");
        let output = self.buffers.last().unwrap();
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
        Ok(DenseReadback {
            staging,
            layout: self.output_layout.clone(),
            stages: self.stage_count(),
            generation: self.generation,
            context: context.clone(),
        })
    }

    /// Immutable GPU output for further N-D operations, including all stage guards.
    pub fn tensor_snapshot(&self, device: &TensorDevice) -> Result<ResidentTensor, DenseError> {
        if self.output_generation != Some(self.generation) {
            return Err(DenseError::StaleOutput);
        }
        if !device
            .runtime()
            .context()
            .shares_handles_with(self.runtime.context())
        {
            return Err(TensorError::DeviceMismatch.into());
        }
        Ok(device.capture(
            &self.output_layout,
            self.buffers.last().unwrap(),
            &self.validation,
        )?)
    }
}

/// Owned output snapshot; remains valid after re-upload, redispatch or workspace drop.
pub struct DenseReadback {
    staging: runtime::ReadbackLease,
    layout: NdLayout,
    stages: usize,
    generation: u64,
    context: WgpuContext,
}

impl DenseReadback {
    pub fn layout(&self) -> &NdLayout {
        &self.layout
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }

    fn decode(bytes: &[u8], elements: usize, stages: usize) -> Result<Vec<f32>, DenseError> {
        if elements.checked_add(stages).and_then(|n| n.checked_mul(4)) != Some(bytes.len()) {
            return Err(DenseError::InvalidReadback);
        }
        for stage in 0..stages {
            let start = (elements + stage) * 4;
            let flags = u32::from_le_bytes(bytes[start..start + 4].try_into().unwrap());
            if flags != 0 {
                return Err(DenseError::NonFiniteIntermediate { stage, flags });
            }
        }
        #[allow(clippy::chunks_exact_to_as_chunks)]
        let values: Vec<f32> = bytes[..elements * 4]
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        if !values.iter().all(|value| value.is_finite()) {
            return Err(DenseError::NonFiniteOutput);
        }
        Ok(values)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn read(mut self) -> Result<Vec<f32>, DenseError> {
        let bytes = self.staging.read(
            &self.context,
            std::time::Duration::from_secs(30),
            "dense.snapshot",
        )?;
        Self::decode(&bytes, self.layout.len(), self.stages)
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn read_async(self) -> Result<Vec<f32>, DenseError> {
        let bytes = self
            .staging
            .read_async(self.context, "dense.snapshot")
            .await?;
        Self::decode(&bytes, self.layout.len(), self.stages)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_shader_validates_for_existing_kernel_options() {
        for kernel in [MatmulKernel::Scalar, MatmulKernel::Register2x2] {
            for accumulation in [
                MatmulAccumulation::Sequential,
                MatmulAccumulation::Tiled,
                MatmulAccumulation::Compensated,
            ] {
                let source = checked_dense_matmul_source([8, 8, 16], kernel, accumulation).unwrap();
                let module = naga::front::wgsl::parse_str(&source).unwrap();
                naga::valid::Validator::new(
                    naga::valid::ValidationFlags::all(),
                    naga::valid::Capabilities::all(),
                )
                .validate(&module)
                .unwrap();
                assert!(source.contains("atomicOr"));
                assert!(source.contains("let cubic = square * x"));
            }
        }
    }

    #[test]
    fn snapshot_decode_checks_all_stages_and_cardinality() {
        let mut words = vec![1.0f32.to_bits(), 2.0f32.to_bits(), 0, 0, 0];
        assert_eq!(
            DenseReadback::decode(bytemuck::cast_slice(&words), 2, 3).unwrap(),
            vec![1.0, 2.0]
        );
        words[3] = 4;
        assert!(matches!(
            DenseReadback::decode(bytemuck::cast_slice(&words), 2, 3),
            Err(DenseError::NonFiniteIntermediate { stage: 1, flags: 4 })
        ));
        assert!(matches!(
            DenseReadback::decode(&[], 2, 3),
            Err(DenseError::InvalidReadback)
        ));
        assert!(matches!(
            DenseReadback::decode(&[], usize::MAX, 1),
            Err(DenseError::InvalidReadback)
        ));
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(matches!(
                DenseReadback::decode(bytemuck::cast_slice(&[value.to_bits(), 0u32]), 1, 1),
                Err(DenseError::NonFiniteOutput)
            ));
        }
    }
}
