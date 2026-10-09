//! Resident embedding values and exact table VJPs, with immutable integer IDs.
use super::*;
use st_kernel_contracts::indexing::{embedding_layout, grouped_rows, IndexingError};

/// Host-prepared, once-uploaded exact IDs and stable reverse grouping. All
/// forwards/pullbacks reuse this device-owned plan without activation readback.
/// This is not a GPU-produced integer tensor or a differentiable index selector.
#[derive(Clone, Debug)]
pub struct ResidentEmbeddingIndices {
    device: TensorDevice,
    layout: NdLayout,
    rows: usize,
    transport: Shared<wgpu::Buffer>,
}

impl ResidentEmbeddingIndices {
    pub fn layout(&self) -> &NdLayout {
        &self.layout
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn device(&self) -> &TensorDevice {
        &self.device
    }
}

/// Immutable lookup tape. A pullback cannot accidentally use different IDs or
/// escape a failed table/forward, including failures outside selected rows.
#[derive(Clone, Debug)]
pub struct ResidentEmbeddingForward {
    indices: ResidentEmbeddingIndices,
    table_shape: Vec<usize>,
    prediction: ResidentTensor,
}

impl ResidentEmbeddingForward {
    pub fn prediction(&self) -> &ResidentTensor {
        &self.prediction
    }

    /// Sum duplicate contributions in token order; never divide by batch size.
    /// One invocation owns each table element, with no floating-point atomics.
    pub fn backward(&self, cotangent: &ResidentTensor) -> Result<ResidentTensor, TensorError> {
        if cotangent.layout.shape() != self.prediction.layout.shape() {
            return Err(IndexingError::CotangentShape.into());
        }
        let guarded = self
            .indices
            .device
            .guard_together(&[cotangent, &self.prediction])?;
        execute(
            &self.indices,
            &guarded[0],
            &NdLayout::contiguous(&self.table_shape)?,
            self.table_shape[1],
            true,
        )
    }
}

impl TensorDevice {
    /// Prepare exact integer IDs once. CPU work is O(rows + tokens), and the
    /// stable CSR transport is O(rows + tokens); no dense one-hot matrix exists.
    pub fn upload_embedding_indices(
        &self,
        shape: &[usize],
        rows: usize,
        indices: &[usize],
    ) -> Result<ResidentEmbeddingIndices, TensorError> {
        let layout = NdLayout::contiguous(shape)?;
        if layout.len() != indices.len() {
            return Err(IndexingError::Length.into());
        }
        let gpu = self.runtime().context().device();
        let len = indices
            .len()
            .checked_mul(2)
            .and_then(|len| len.checked_add(rows))
            .and_then(|len| len.checked_add(1))
            .ok_or(IndexingError::Allocation)?;
        storage_limit(len, &gpu.limits())?;
        validate_view(&layout, layout.len(), &gpu.limits())?;
        let (offsets, positions) = grouped_rows(indices, rows)?;
        let mut transport = Vec::new();
        transport
            .try_reserve_exact(len)
            .map_err(|_| IndexingError::Allocation)?;
        transport.extend(
            indices
                .iter()
                .chain(&offsets)
                .chain(&positions)
                .map(|&v| v as u32),
        );
        Ok(ResidentEmbeddingIndices {
            device: self.clone(),
            layout,
            rows,
            transport: Shared::new(runtime::upload_slice(
                gpu,
                "embedding.indices_u32",
                &transport,
                wgpu::BufferUsages::STORAGE,
            )?),
        })
    }
}

impl ResidentTensor {
    /// Table [V,C] and IDs [B,T] produce [B,T,C]. Views and broadcasts are
    /// supported. Values and the table VJP remain resident on this same queue.
    pub fn embedding(
        &self,
        indices: &ResidentEmbeddingIndices,
    ) -> Result<ResidentEmbeddingForward, TensorError> {
        let layout = embedding_layout(self.layout.shape(), indices.layout.shape(), indices.rows)?;
        let prediction = execute(indices, self, &layout, self.layout.shape()[1], false)?;
        Ok(ResidentEmbeddingForward {
            indices: indices.clone(),
            table_shape: self.layout.shape().to_vec(),
            prediction,
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    cols: u32,
    tokens: u32,
    rows: u32,
    len: u32,
    groups_x: u32,
    groups: u32,
    input_offset: u32,
    flag_words: u32,
}

fn shader_source() -> String {
    include_str!("shaders/embedding.wgsl")
        .replace("ROUNDED_ADD", include_str!("../shaders/rounded_add.wgsl"))
        .replace("INVALID_TENSOR_FLAG", &format!("{INVALID_TENSOR_FLAG}u"))
}

#[derive(Debug)]
pub(super) struct EmbeddingKernels {
    layout: wgpu::BindGroupLayout,
    forward: wgpu::ComputePipeline,
    backward: wgpu::ComputePipeline,
}

impl EmbeddingKernels {
    fn new(gpu: &wgpu::Device) -> Self {
        let layout = gpu.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("embedding.layout"),
            entries: &(0..6)
                .map(|binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: if binding == 5 {
                            wgpu::BufferBindingType::Uniform
                        } else {
                            wgpu::BufferBindingType::Storage {
                                read_only: ![2, 4].contains(&binding),
                            }
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect::<Vec<_>>(),
        });
        let pipeline_layout = gpu.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("embedding.pipeline_layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let module = gpu.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("embedding.shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source().into()),
        });
        let pipeline = |entry_point| {
            gpu.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point,
                compilation_options: Default::default(),
            })
        };
        let forward = pipeline("gather");
        let backward = pipeline("pullback");
        Self {
            layout,
            forward,
            backward,
        }
    }
}

fn preflight(layout: &NdLayout, limits: &wgpu::Limits) -> Result<[u32; 3], TensorError> {
    validate_view(layout, layout.len(), limits)?;
    if limits.max_uniform_buffers_per_shader_stage < 1
        || limits.max_uniform_buffer_binding_size < std::mem::size_of::<Params>() as u32
    {
        return Err(TensorError::Limit("embedding uniform"));
    }
    grid(layout.len(), limits)
}

fn execute(
    indices: &ResidentEmbeddingIndices,
    input: &ResidentTensor,
    output_layout: &NdLayout,
    cols: usize,
    backward: bool,
) -> Result<ResidentTensor, TensorError> {
    let device = &indices.device;
    let context = device.runtime().context();
    input.require_context(context)?;
    let gpu = context.device();
    let limits = gpu.limits();
    validate_view(&input.layout, (input.values().size() / 4) as usize, &limits)?;
    storage_limit((input.flags().size() / 4) as usize, &limits)?;
    let [x, y, groups] = preflight(output_layout, &limits)?;
    let output = device.allocate_output(output_layout)?;
    let kernels = device
        .0
        .embedding
        .get_or_init(|| EmbeddingKernels::new(gpu));
    let mut encoder = gpu.create_command_encoder(&Default::default());
    let input = input.contiguous_into(&mut encoder)?;
    let params = Params {
        cols: cols as u32,
        tokens: indices.layout.len() as u32,
        rows: indices.rows as u32,
        len: output_layout.len() as u32,
        groups_x: x,
        groups,
        input_offset: input.layout.offset() as u32,
        flag_words: (input.flags().size() / 4) as u32,
    };
    let params = runtime::upload_slice(
        gpu,
        "embedding.params",
        &[params],
        wgpu::BufferUsages::UNIFORM,
    )?;
    let buffers = [
        input.values(),
        &indices.transport,
        output.values(),
        input.flags(),
        output.flags(),
        &params,
    ];
    let binding = gpu.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("embedding.binding"),
        layout: &kernels.layout,
        entries: &buffers
            .iter()
            .enumerate()
            .map(|(i, buffer)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>(),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("embedding.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(if backward {
            &kernels.backward
        } else {
            &kernels.forward
        });
        pass.set_bind_group(0, &binding, &[]);
        pass.dispatch_workgroups(x, y, 1);
    }
    context.queue().submit(Some(encoder.finish()));
    Ok(output)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
