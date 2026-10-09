//! Model-independent resident parameters with versioned, all-or-none SGD.
//! Values are immutable snapshots; submitting an update never maps a buffer.

use super::*;
use crate::resident_tensor::{storage_limit, INVALID_TENSOR_FLAG};

#[derive(Clone, Debug)]
pub(super) struct ParameterVersion {
    pub(super) workspace: Shared<()>,
    pub(super) revision: u64,
}

impl ParameterVersion {
    pub(super) fn new() -> Self {
        Self {
            workspace: Shared::new(()),
            revision: 0,
        }
    }

    pub(super) fn matches(&self, other: &Self) -> bool {
        self.revision == other.revision && Shared::ptr_eq(&self.workspace, &other.workspace)
    }
}

/// One immutable parameter version. Old snapshots survive updates and owner drop.
/// Entries are independent parameters, not tied-weight aliases.
#[derive(Clone, Debug)]
pub struct ResidentParameterSnapshot {
    version: ParameterVersion,
    values: Vec<ResidentTensor>,
}

impl ResidentParameterSnapshot {
    /// Attempted-update revision, not a count of numerically accepted updates.
    pub fn revision(&self) -> u64 {
        self.version.revision
    }

    pub fn values(&self) -> &[ResidentTensor] {
        &self.values
    }

    /// Bind caller-computed derivatives to the version used to compute them.
    /// This verifies layout/device provenance, not the caller's differentiation.
    /// It does not normalize, scale, accumulate, or read gradient values.
    pub fn bind_gradients(
        &self,
        gradients: Vec<ResidentTensor>,
    ) -> Result<ResidentParameterGradients, TrainingError> {
        if gradients.len() != self.values.len() {
            return Err(TrainingError::ParameterLayout);
        }
        for (parameter, gradient) in self.values.iter().zip(&gradients) {
            gradient.require_context(parameter.device().runtime().context())?;
            if gradient.layout().shape() != parameter.layout().shape() {
                return Err(TrainingError::ParameterLayout);
            }
        }
        Ok(ResidentParameterGradients {
            version: self.version.clone(),
            values: gradients,
        })
    }
}

/// Explicit derivatives associated with one owner's parameter revision.
#[derive(Clone, Debug)]
pub struct ResidentParameterGradients {
    version: ParameterVersion,
    values: Vec<ResidentTensor>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct UpdateParams {
    len: u32,
    groups_x: u32,
    count: u32,
    index: u32,
    rate: f32,
    padding: [u32; 3],
}

struct UpdateKernels {
    layout: wgpu::BindGroupLayout,
    prepare: wgpu::ComputePipeline,
    decide: wgpu::ComputePipeline,
    commit: wgpu::ComputePipeline,
}

fn source() -> String {
    include_str!("../shaders/resident_parameters.wgsl")
        .replace(
            "SGD_CANDIDATE",
            &st_kernel_contracts::sgd::sgd_candidate_wgsl(),
        )
        .replace("INVALID_TENSOR_FLAG", &format!("{INVALID_TENSOR_FLAG}u"))
}

impl UpdateKernels {
    fn new(device: &wgpu::Device) -> Result<Self, TrainingError> {
        let limits = device.limits();
        if limits.max_storage_buffers_per_shader_stage < 8
            || limits.max_bindings_per_bind_group < 9
            || limits.max_uniform_buffers_per_shader_stage < 1
        {
            return Err(TensorError::Limit("resident parameter update bindings").into());
        }
        let entries: Vec<_> = (0..9)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 8 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: [0, 1, 4, 5].contains(&binding),
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("parameters.bindings"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("parameters.pipeline_layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("parameters.shader"),
            source: wgpu::ShaderSource::Wgsl(source().into()),
        });
        let pipeline = |entry_point| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point,
                compilation_options: Default::default(),
            })
        };
        Ok(Self {
            prepare: pipeline("prepare"),
            decide: pipeline("decide"),
            commit: pipeline("commit"),
            layout,
        })
    }
}

/// Owns the current immutable GPU parameter version. A rejected device update
/// retains every old value, but invalidates old gradient tokens, as graph SGD does.
/// This is plain SGD; optimizer history and model/host synchronization are explicit
/// responsibilities of the caller, not silently reconstructed here.
pub struct ResidentParameters {
    current: ResidentParameterSnapshot,
    device: TensorDevice,
    kernels: UpdateKernels,
}

impl ResidentParameters {
    /// Adopt restored values with a fresh owner identity and an explicit attempted
    /// revision. No old gradient token becomes valid. Model schema and optimizer
    /// compatibility must be validated by the checkpoint loader before this call.
    pub fn from_restored_values(
        values: Vec<ResidentTensor>,
        attempted_revision: u64,
    ) -> Result<Self, TrainingError> {
        let mut owner = Self::new(values)?;
        owner.current.version.revision = attempted_revision;
        Ok(owner)
    }

    /// Adopt nonempty tensors on one device without uploading or reading values.
    pub fn new(values: Vec<ResidentTensor>) -> Result<Self, TrainingError> {
        let first = values.first().ok_or(TrainingError::ParameterLayout)?;
        let device = first.device().clone();
        let context = device.runtime().context();
        for value in &values {
            value.require_context(context)?;
            if value.layout().is_empty() {
                return Err(TrainingError::ParameterLayout);
            }
        }
        let flag_count = values.len().checked_add(2).ok_or(TrainingError::Overflow)?;
        storage_limit(flag_count, &context.device().limits())?;
        let kernels = UpdateKernels::new(context.device())?;
        Ok(Self {
            current: ResidentParameterSnapshot {
                version: ParameterVersion::new(),
                values,
            },
            device,
            kernels,
        })
    }

    pub fn snapshot(&self) -> ResidentParameterSnapshot {
        self.current.clone()
    }

    /// Check both owner identity and attempted-update revision without reading values.
    pub fn is_current(&self, snapshot: &ResidentParameterSnapshot) -> bool {
        self.current.version.matches(&snapshot.version)
    }

    pub fn tensor_device(&self) -> &TensorDevice {
        &self.device
    }

    /// Validate, enqueue all candidates, decide once, then select every next value.
    /// No host mapping occurs. Only an explicit update readback proves acceptance.
    /// Rate zero still validates all derivatives and retains original value bits.
    pub fn sgd(
        &mut self,
        gradients: &ResidentParameterGradients,
        rate: f32,
    ) -> Result<ResidentParameterUpdate, TrainingError> {
        let rate = SgdStep::new(rate).map_err(|_| TrainingError::LearningRate)?;
        self.sgd_validated(gradients, |_| rate.rate())
    }

    /// One rate per parameter tensor, in snapshot order, with one atomic decision.
    /// A zero rate freezes that tensor's bits without detaching differentiation.
    /// Frozen entries still validate their values, gradients and inherited guards.
    /// Invalid rates or a wrong count fail before submission or revision changes.
    /// Rates are caller-owned step inputs, not persistent optimizer state.
    pub fn sgd_with_rates(
        &mut self,
        gradients: &ResidentParameterGradients,
        rates: &[f32],
    ) -> Result<ResidentParameterUpdate, TrainingError> {
        if rates.len() != self.current.values.len() {
            return Err(TrainingError::ParameterLayout);
        }
        let rates = rates
            .iter()
            .map(|&rate| SgdStep::new(rate).map_err(|_| TrainingError::LearningRate))
            .collect::<Result<Vec<_>, _>>()?;
        self.sgd_validated(gradients, |index| rates[index].rate())
    }

    fn sgd_validated(
        &mut self,
        gradients: &ResidentParameterGradients,
        rate: impl Fn(usize) -> f32,
    ) -> Result<ResidentParameterUpdate, TrainingError> {
        if !self.current.version.matches(&gradients.version) {
            return Err(TrainingError::ParameterVersion);
        }
        let revision = self
            .current
            .version
            .revision
            .checked_add(1)
            .ok_or(TrainingError::Overflow)?;
        let context = self.device.runtime().context();
        let gpu = context.device();
        let count = self.current.values.len();
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        let flags = Shared::new(runtime::empty_buffer::<u32>(
            gpu,
            "parameters.update_flags",
            count + 2,
            usage,
        )?);
        let mut outputs = Vec::with_capacity(count);
        let mut passes = Vec::with_capacity(count);
        let mut encoder = gpu.create_command_encoder(&Default::default());
        for (index, (parameter, gradient)) in self
            .current
            .values
            .iter()
            .zip(&gradients.values)
            .enumerate()
        {
            let parameter = parameter.contiguous_into(&mut encoder)?;
            let gradient = gradient.contiguous_into(&mut encoder)?;
            let len = parameter.layout().len();
            let groups = groups(len, &gpu.limits())?;
            let output = self.device.allocate_output(parameter.layout())?;
            let candidate = runtime::empty_buffer::<f32>(gpu, "parameters.candidate", len, usage)?;
            let params = runtime::upload_slice(
                gpu,
                "parameters.update_config",
                &[UpdateParams {
                    len: len as u32,
                    groups_x: groups[0],
                    count: count as u32,
                    index: index as u32,
                    rate: rate(index),
                    padding: [0; 3],
                }],
                wgpu::BufferUsages::UNIFORM,
            )?;
            let binding = binding(
                gpu,
                &self.kernels.layout,
                &[
                    parameter.values(),
                    gradient.values(),
                    &candidate,
                    output.values(),
                    parameter.flags(),
                    gradient.flags(),
                    output.flags(),
                    &flags,
                    &params,
                ],
            );
            passes.push((binding, groups));
            outputs.push(output);
        }
        for (binding, groups) in &passes {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.kernels.prepare);
            pass.set_bind_group(0, binding, &[]);
            pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
        }
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.kernels.decide);
            pass.set_bind_group(0, &passes[0].0, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        for (binding, groups) in &passes {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.kernels.commit);
            pass.set_bind_group(0, binding, &[]);
            pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
        }
        context.queue().submit(Some(encoder.finish()));
        self.current.values = outputs;
        self.current.version.revision = revision;
        Ok(ResidentParameterUpdate {
            device: self.device.clone(),
            flags,
            count,
            revision,
        })
    }
}

/// Immutable acceptance flags for one submitted update, independent of reuse.
pub struct ResidentParameterUpdate {
    device: TensorDevice,
    flags: Shared<wgpu::Buffer>,
    count: usize,
    revision: u64,
}

impl ResidentParameterUpdate {
    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn snapshot(&self) -> Result<ResidentParameterUpdateReadback, TrainingError> {
        Ok(ResidentParameterUpdateReadback {
            raw: readback::capture_new(self.device.runtime().context(), &[&self.flags])?,
            count: self.count,
            revision: self.revision,
        })
    }

    /// Capture this update's flags and one guarded scalar in the same submission.
    /// The caller associates the observation with the update; this does not
    /// prove that the supplied scalar produced its gradients. A rejected update
    /// takes precedence over invalid scalar data, as in separate receipt reads.
    pub fn snapshot_with_scalar(
        &self,
        scalar: &ResidentTensor,
    ) -> Result<ResidentParameterScalarReadback, TrainingError> {
        let context = self.device.runtime().context();
        scalar.require_context(context)?;
        if scalar.layout().len() != 1 {
            return Err(TensorError::Length.into());
        }
        let count = self.count.checked_add(2).ok_or(TrainingError::Overflow)?;
        let mut encoder = context.device().create_command_encoder(&Default::default());
        let batch = runtime::ReadbackBatch::encode_spans(
            context,
            &[
                (&self.flags, 0, count),
                (scalar.values(), scalar.layout().offset(), 1),
                (scalar.flags(), 0, 1),
            ],
            "parameters.scalar_snapshot",
            &mut encoder,
        )?;
        context.queue().submit(Some(encoder.finish()));
        Ok(ResidentParameterScalarReadback {
            batch,
            count: self.count,
            revision: self.revision,
        })
    }
}

/// Frozen update acceptance and a finite scalar, with one map when the small
/// combined payload fits a staging buffer. Negative finite scalars are valid.
pub struct ResidentParameterScalarReadback {
    batch: runtime::ReadbackBatch<u32>,
    count: usize,
    revision: u64,
}

impl ResidentParameterScalarReadback {
    pub fn staging_buffer_count(&self) -> usize {
        self.batch.staging_buffer_count()
    }

    fn decode(
        words: &[Vec<u32>],
        count: usize,
        revision: u64,
    ) -> Result<(u64, f32), TrainingError> {
        let expected = count.checked_add(2).ok_or(TrainingError::InvalidReadback)?;
        if words.len() != 3
            || words[0].len() != expected
            || words[1].len() != 1
            || words[2].len() != 1
        {
            return Err(TrainingError::InvalidReadback);
        }
        readback::validation_flags(bytemuck::cast_slice(&words[0]), count)?;
        let scalar = f32::from_bits(u32::from_le(words[1][0]));
        if words[2][0] != 0 || !scalar.is_finite() {
            return Err(TensorError::NonFinite.into());
        }
        Ok((revision, scalar))
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn read(self) -> Result<(u64, f32), TrainingError> {
        Self::decode(&self.batch.read()?, self.count, self.revision)
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn read_async(self) -> Result<(u64, f32), TrainingError> {
        Self::decode(&self.batch.read_async().await?, self.count, self.revision)
    }
}

pub struct ResidentParameterUpdateReadback {
    raw: readback::RawSnapshot,
    count: usize,
    revision: u64,
}

impl ResidentParameterUpdateReadback {
    /// Returns the attempted revision only after all parameter flags pass.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read(self) -> Result<u64, TrainingError> {
        readback::validation_flags(&self.raw.read()?, self.count)?;
        Ok(self.revision)
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn read_async(self) -> Result<u64, TrainingError> {
        readback::validation_flags(&self.raw.read_async().await?, self.count)?;
        Ok(self.revision)
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
