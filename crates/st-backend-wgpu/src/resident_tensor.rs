//! Immutable N-D GPU storage and explicit operations. Views share storage;
//! evaluation stays on the owning queue until an explicit snapshot is read.

use crate::runtime::{self, Shared, WgpuContext, WgpuRuntime, WgpuRuntimeError};
use st_kernel_contracts::{
    elementwise::ElementwiseOp,
    layout::{broadcast_shape, NdLayout, NdLayoutError},
};
use thiserror::Error;

pub(crate) mod capture;
pub(crate) mod guard_capture;
pub mod loss;
pub mod pointwise;

/// An upstream tensor failed its finite-value contract. NN flags retain this bit.
pub const INVALID_TENSOR_FLAG: u32 = 0x8000_0000;

#[derive(Debug, Error)]
pub enum TensorError {
    #[error(transparent)]
    Pointwise(#[from] st_kernel_contracts::pointwise::PointwiseError),
    #[error(transparent)]
    Layout(#[from] NdLayoutError),
    #[error(transparent)]
    Runtime(#[from] WgpuRuntimeError),
    #[error("tensor data length does not match its shape")]
    Length,
    #[error("tensor contains a non-finite value or inherits a failed operation")]
    NonFinite,
    #[error("tensors must use the same device and queue")]
    DeviceMismatch,
    #[error("operation requires a different number of operands")]
    Operands,
    #[error("loss predictions and targets must have the same logical shape")]
    LossShape,
    #[error("tensor exceeds portable addressing or device limits: {0}")]
    Limit(&'static str),
    #[error("tensor view addresses outside its storage")]
    StorageBounds,
    #[error("invalid tensor readback")]
    Readback,
}

#[derive(Debug)]
struct Kernels {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    runtime: WgpuRuntime,
    mse: std::sync::OnceLock<loss::MseKernels>,
}

/// One reusable elementwise pipeline on an existing WGPU runtime. No device
/// discovery, fallback, or host synchronization occurs during tensor operations.
#[derive(Clone, Debug)]
pub struct TensorDevice(Shared<Kernels>);

fn source() -> String {
    substitute_ops(include_str!("shaders/resident_tensor.wgsl").replace(
        "CHECKED_ELEMENTWISE",
        include_str!("shaders/checked_elementwise.wgsl"),
    ))
}

pub(crate) fn substitute_ops(source: String) -> String {
    // Standalone kernels own one flag word. Graph pointwise kernels substitute
    // their metadata-selected stage slot before this shared expansion.
    let mut source = source
        .replace("CHECKED_FLAG_INDEX", "0u")
        .replace("INVALID_TENSOR_FLAG", &format!("{INVALID_TENSOR_FLAG}u"));
    for (name, op) in [
        ("OP_ADD", ElementwiseOp::Add),
        ("OP_MULTIPLY", ElementwiseOp::Multiply),
        ("OP_RELU", ElementwiseOp::Relu),
        ("OP_GELU", ElementwiseOp::Gelu),
    ] {
        source = source.replace(name, &format!("{}u", op as u32));
    }
    source
}

pub(crate) fn storage_limit(elements: usize, limits: &wgpu::Limits) -> Result<(), TensorError> {
    let bytes = runtime::checked_byte_len::<u32>("resident tensor", elements.max(1))?;
    if elements > u32::MAX as usize
        || bytes > limits.max_buffer_size
        || bytes > u64::from(limits.max_storage_buffer_binding_size)
    {
        return Err(TensorError::Limit("storage buffer"));
    }
    Ok(())
}

fn validate_view(
    layout: &NdLayout,
    storage_len: usize,
    limits: &wgpu::Limits,
) -> Result<(), TensorError> {
    if layout.required_storage_len()? > storage_len {
        return Err(TensorError::StorageBounds);
    }
    if layout.len() > u32::MAX as usize
        || layout.offset() > u32::MAX as usize
        || layout
            .shape()
            .iter()
            .chain(layout.strides())
            .any(|&v| v > u32::MAX as usize)
    {
        return Err(TensorError::Limit("layout address"));
    }
    storage_limit(storage_len, limits)?;
    storage_limit(
        layout
            .rank()
            .checked_mul(3)
            .and_then(|n| n.checked_add(9))
            .ok_or(TensorError::Limit("metadata"))?,
        limits,
    )
}

fn grid(len: usize, limits: &wgpu::Limits) -> Result<[u32; 3], TensorError> {
    let groups = (len as u32).div_ceil(256).max(1);
    let x = groups.min(limits.max_compute_workgroups_per_dimension);
    if x == 0 || groups.div_ceil(x) > limits.max_compute_workgroups_per_dimension {
        return Err(TensorError::Limit("dispatch grid"));
    }
    Ok([x, groups.div_ceil(x), groups])
}

struct Operand<'a> {
    values: &'a wgpu::Buffer,
    flags: &'a wgpu::Buffer,
    layout: &'a NdLayout,
}

impl TensorDevice {
    pub fn new(runtime: WgpuRuntime) -> Result<Self, TensorError> {
        let device = runtime.context().device();
        let limits = device.limits();
        if limits.max_compute_invocations_per_workgroup < 256
            || limits.max_compute_workgroup_size_x < 256
            || limits.max_storage_buffers_per_shader_stage < 7
            || limits.max_bindings_per_bind_group < 7
            || limits.max_bind_groups < 1
        {
            return Err(TensorError::Limit("elementwise pipeline"));
        }
        let entries: Vec<_> = (0..7)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: ![2, 6].contains(&binding),
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tensor.layout"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tensor.pipeline_layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tensor.shader"),
            source: wgpu::ShaderSource::Wgsl(source().into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tensor.elementwise"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: "main",
            compilation_options: Default::default(),
        });
        Ok(Self(Shared::new(Kernels {
            layout,
            pipeline,
            runtime,
            mse: std::sync::OnceLock::new(),
        })))
    }

    pub fn runtime(&self) -> &WgpuRuntime {
        &self.0.runtime
    }

    /// Private graph destination. A caller must encode all values and the guard,
    /// then submit, before exposing this immutable handle outside the backend.
    pub(crate) fn allocate_output(&self, layout: &NdLayout) -> Result<ResidentTensor, TensorError> {
        let layout = NdLayout::contiguous(layout.shape())?;
        let gpu = self.runtime().context().device();
        validate_view(&layout, layout.len(), &gpu.limits())?;
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        Ok(ResidentTensor {
            storage: Shared::new(Storage {
                values: runtime::empty_buffer::<f32>(
                    gpu,
                    "tensor.direct_output",
                    layout.len().max(1),
                    usage,
                )?,
                flags: Shared::new(runtime::empty_buffer::<u32>(
                    gpu,
                    "tensor.direct_guard",
                    1,
                    usage,
                )?),
            }),
            layout,
            device: self.clone(),
        })
    }

    pub fn upload(&self, shape: &[usize], values: &[f32]) -> Result<ResidentTensor, TensorError> {
        let layout = NdLayout::contiguous(shape)?;
        if layout.len() != values.len() {
            return Err(TensorError::Length);
        }
        validate_view(
            &layout,
            values.len(),
            &self.runtime().context().device().limits(),
        )?;
        if !values.iter().all(|v| v.is_finite()) {
            return Err(TensorError::NonFinite);
        }
        let device = self.runtime().context().device();
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        let values = runtime::upload_slice(
            device,
            "tensor.values",
            if values.is_empty() { &[0.] } else { values },
            usage,
        )?;
        let flags = runtime::upload_slice(device, "tensor.flags", &[0u32], usage)?;
        Ok(ResidentTensor {
            storage: Shared::new(Storage {
                values,
                flags: Shared::new(flags),
            }),
            layout,
            device: self.clone(),
        })
    }

    fn execute(
        &self,
        op: ElementwiseOp,
        a: Operand<'_>,
        b: Operand<'_>,
        shape: &[usize],
    ) -> Result<ResidentTensor, TensorError> {
        let context = self.runtime().context();
        let mut encoder = context.device().create_command_encoder(&Default::default());
        let output = self.encode(&mut encoder, op, a, b, shape)?;
        context.queue().submit(Some(encoder.finish()));
        Ok(output)
    }

    fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        op: ElementwiseOp,
        a: Operand<'_>,
        b: Operand<'_>,
        shape: &[usize],
    ) -> Result<ResidentTensor, TensorError> {
        let layout = NdLayout::contiguous(shape)?;
        let context = self.runtime().context();
        let device = context.device();
        let limits = device.limits();
        validate_view(a.layout, (a.values.size() / 4) as usize, &limits)?;
        validate_view(b.layout, (b.values.size() / 4) as usize, &limits)?;
        validate_view(&layout, layout.len(), &limits)?;
        storage_limit((a.flags.size() / 4) as usize, &limits)?;
        storage_limit((b.flags.size() / 4) as usize, &limits)?;
        let [x, y, groups] = grid(layout.len(), &limits)?;
        let mut meta = vec![
            layout.len() as u32,
            layout.rank() as u32,
            op as u32,
            x,
            groups,
            a.layout.offset() as u32,
            b.layout.offset() as u32,
            (a.flags.size() / 4) as u32,
            (b.flags.size() / 4) as u32,
        ];
        for values in [shape, a.layout.strides(), b.layout.strides()] {
            meta.extend(values.iter().map(|&v| v as u32));
        }
        let meta = runtime::upload_slice(
            device,
            "tensor.metadata",
            &meta,
            wgpu::BufferUsages::STORAGE,
        )?;
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        let values =
            runtime::empty_buffer::<f32>(device, "tensor.output", layout.len().max(1), usage)?;
        let flags = runtime::empty_buffer::<u32>(device, "tensor.output_flags", 1, usage)?;
        let buffers = [a.values, b.values, &values, &meta, a.flags, b.flags, &flags];
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, buffer)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let binding = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tensor.operands"),
            layout: &self.0.layout,
            entries: &entries,
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tensor.elementwise"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.0.pipeline);
            pass.set_bind_group(0, &binding, &[]);
            pass.dispatch_workgroups(x, y, 1);
        }
        Ok(ResidentTensor {
            storage: Shared::new(Storage {
                values,
                flags: Shared::new(flags),
            }),
            layout,
            device: self.clone(),
        })
    }

    /// Crate-owned bridge: freeze a mutable NN result and all its guards now.
    pub(crate) fn capture(
        &self,
        layout: &NdLayout,
        values: &wgpu::Buffer,
        flags: &wgpu::Buffer,
    ) -> Result<ResidentTensor, TensorError> {
        self.execute(
            ElementwiseOp::Identity,
            Operand {
                values,
                flags,
                layout,
            },
            Operand {
                values,
                flags,
                layout,
            },
            layout.shape(),
        )
    }

    /// Freeze within the caller's submission, retaining every upstream guard.
    pub(crate) fn capture_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        layout: &NdLayout,
        values: &wgpu::Buffer,
        flags: &wgpu::Buffer,
    ) -> Result<ResidentTensor, TensorError> {
        self.encode(
            encoder,
            ElementwiseOp::Identity,
            Operand {
                values,
                flags,
                layout,
            },
            Operand {
                values,
                flags,
                layout,
            },
            layout.shape(),
        )
    }
}

#[derive(Debug)]
struct Storage {
    values: wgpu::Buffer,
    flags: Shared<wgpu::Buffer>,
}

/// No mutable buffer escapes this handle. Clones and views cannot be invalidated
/// by later operations, source workspace reuse, or host Tensor mutation.
#[derive(Clone, Debug)]
pub struct ResidentTensor {
    storage: Shared<Storage>,
    layout: NdLayout,
    device: TensorDevice,
}

impl ResidentTensor {
    pub fn layout(&self) -> &NdLayout {
        &self.layout
    }
    pub fn device(&self) -> &TensorDevice {
        &self.device
    }
    pub fn shares_storage_with(&self, other: &Self) -> bool {
        Shared::ptr_eq(&self.storage, &other.storage)
    }

    /// Private recycling gate. No other tensor/view (or weak storage owner) can
    /// resurrect this version. Prepared operations must retain their input
    /// tensors until submission; already submitted reads precede reuse on the
    /// same queue. GPU flags are rewritten together with values, never separately.
    pub(crate) fn exclusively_owned(&mut self) -> bool {
        Shared::get_mut(&mut self.storage)
            .is_some_and(|storage| Shared::get_mut(&mut storage.flags).is_some())
    }

    fn view(&self, layout: NdLayout) -> Result<Self, TensorError> {
        validate_view(
            &layout,
            (self.storage.values.size() / 4) as usize,
            &self.device.runtime().context().device().limits(),
        )?;
        Ok(Self {
            layout,
            ..self.clone()
        })
    }
    pub fn reshape(&self, shape: &[usize]) -> Result<Self, TensorError> {
        self.view(self.layout.reshape(shape)?)
    }
    pub fn permute(&self, axes: &[usize]) -> Result<Self, TensorError> {
        self.view(self.layout.permute(axes)?)
    }
    pub fn narrow(&self, axis: usize, start: usize, length: usize) -> Result<Self, TensorError> {
        self.view(self.layout.narrow(axis, start, length)?)
    }
    pub fn broadcast_to(&self, shape: &[usize]) -> Result<Self, TensorError> {
        self.view(self.layout.broadcast_to(shape)?)
    }

    pub fn apply(&self, op: ElementwiseOp, rhs: Option<&Self>) -> Result<Self, TensorError> {
        let context = self.device.runtime().context();
        let mut encoder = context.device().create_command_encoder(&Default::default());
        let output = self.apply_into(&mut encoder, op, rhs)?;
        context.queue().submit(Some(encoder.finish()));
        Ok(output)
    }

    /// Submit the operation before preparing its terminal snapshot.
    /// Reading remains explicit and uses the same exclusive snapshot lease.
    pub fn apply_snapshot(
        &self,
        op: ElementwiseOp,
        rhs: Option<&Self>,
    ) -> Result<TensorReadback, TensorError> {
        self.apply(op, rhs)?.snapshot()
    }

    pub(crate) fn apply_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        op: ElementwiseOp,
        rhs: Option<&Self>,
    ) -> Result<Self, TensorError> {
        if op.is_binary() != rhs.is_some() {
            return Err(TensorError::Operands);
        }
        let rhs = rhs.unwrap_or(self);
        self.require_context(rhs.device.runtime().context())?;
        let shape = broadcast_shape(self.layout.shape(), rhs.layout.shape())?;
        let a = self.layout.broadcast_to(&shape)?;
        let b = rhs.layout.broadcast_to(&shape)?;
        self.device.encode(
            encoder,
            op,
            Operand {
                values: &self.storage.values,
                flags: &self.storage.flags,
                layout: &a,
            },
            Operand {
                values: &rhs.storage.values,
                flags: &rhs.storage.flags,
                layout: &b,
            },
            &shape,
        )
    }
    pub fn add(&self, rhs: &Self) -> Result<Self, TensorError> {
        self.apply(ElementwiseOp::Add, Some(rhs))
    }
    pub fn mul(&self, rhs: &Self) -> Result<Self, TensorError> {
        self.apply(ElementwiseOp::Multiply, Some(rhs))
    }
    pub fn relu(&self) -> Result<Self, TensorError> {
        self.apply(ElementwiseOp::Relu, None)
    }
    pub fn gelu(&self) -> Result<Self, TensorError> {
        self.apply(ElementwiseOp::Gelu, None)
    }

    pub fn contiguous(&self) -> Result<Self, TensorError> {
        if self.layout.is_contiguous() && self.layout.offset() == 0 {
            Ok(self.clone())
        } else {
            self.apply(ElementwiseOp::Identity, None)
        }
    }

    /// Pack only if needed, within the caller's existing GPU submission.
    pub(crate) fn contiguous_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<Self, TensorError> {
        if self.layout.is_contiguous() && self.layout.offset() == 0 {
            Ok(self.clone())
        } else {
            self.apply_into(encoder, ElementwiseOp::Identity, None)
        }
    }

    pub(crate) fn require_context(&self, context: &WgpuContext) -> Result<(), TensorError> {
        if !self.device.runtime().context().shares_handles_with(context) {
            return Err(TensorError::DeviceMismatch);
        }
        Ok(())
    }
    pub(crate) fn values(&self) -> &wgpu::Buffer {
        &self.storage.values
    }
    pub(crate) fn flags(&self) -> &wgpu::Buffer {
        &self.storage.flags
    }

    /// Capture logical values and validity now; awaiting does not re-read the source.
    pub fn snapshot(&self) -> Result<TensorReadback, TensorError> {
        let packed = self.contiguous()?;
        let context = self.device.runtime().context();
        let len = self.layout.len();
        let staging = runtime::empty_buffer::<u32>(
            context.device(),
            "tensor.snapshot",
            len.checked_add(1).ok_or(TensorError::Limit("readback"))?,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        )?;
        let mut encoder = context.device().create_command_encoder(&Default::default());
        if len > 0 {
            encoder.copy_buffer_to_buffer(packed.values(), 0, &staging, 0, len as u64 * 4);
        }
        encoder.copy_buffer_to_buffer(packed.flags(), 0, &staging, len as u64 * 4, 4);
        context.queue().submit(Some(encoder.finish()));
        Ok(TensorReadback {
            staging: runtime::ReadbackLease::unpooled(staging),
            layout: NdLayout::contiguous(self.layout.shape())?,
            context: context.clone(),
        })
    }
}

pub struct TensorReadback {
    staging: runtime::ReadbackLease,
    layout: NdLayout,
    context: WgpuContext,
}

impl TensorReadback {
    pub fn layout(&self) -> &NdLayout {
        &self.layout
    }
    fn decode(bytes: &[u8], len: usize) -> Result<Vec<f32>, TensorError> {
        if bytes.len() != (len + 1) * 4 {
            return Err(TensorError::Readback);
        }
        if u32::from_le_bytes(bytes[len * 4..].try_into().unwrap()) != 0 {
            return Err(TensorError::NonFinite);
        }
        #[allow(clippy::chunks_exact_to_as_chunks)]
        let values: Vec<f32> = bytes[..len * 4]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        if !values.iter().all(|v| v.is_finite()) {
            return Err(TensorError::NonFinite);
        }
        Ok(values)
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read(mut self) -> Result<Vec<f32>, TensorError> {
        let bytes = self.staging.read(
            &self.context,
            std::time::Duration::from_secs(30),
            "tensor.snapshot",
        )?;
        Self::decode(&bytes, self.layout.len())
    }
    #[cfg(target_arch = "wasm32")]
    pub async fn read_async(self) -> Result<Vec<f32>, TensorError> {
        let bytes = self
            .staging
            .read_async(self.context, "tensor.snapshot")
            .await?;
        Self::decode(&bytes, self.layout.len())
    }
}

#[cfg(test)]
mod tests;
