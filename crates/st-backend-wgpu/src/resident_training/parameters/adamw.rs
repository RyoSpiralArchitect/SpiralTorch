use super::*;
use st_kernel_contracts::adamw::{AdamWConfig, AdamWStep, ADAMW_CANDIDATE_WGSL};
pub use st_kernel_contracts::adamw::{AdamWSlot, AdamWState};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AdamWParams {
    len: u32,
    groups_x: u32,
    count: u32,
    index: u32,
    rate: f32,
    decay: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    padding: [u32; 3],
}

struct AdamWKernels {
    prepare_layout: wgpu::BindGroupLayout,
    commit_layout: wgpu::BindGroupLayout,
    prepare: wgpu::ComputePipeline,
    decide: wgpu::ComputePipeline,
    commit: wgpu::ComputePipeline,
}

fn source(commit: bool) -> String {
    let body = if commit {
        include_str!("../shaders/adamw_commit.wgsl")
    } else {
        include_str!("../shaders/adamw_prepare.wgsl")
    };
    (ADAMW_CANDIDATE_WGSL.to_owned() + include_str!("../shaders/adamw_params.wgsl") + body)
        .replace("INVALID_TENSOR_FLAG", &format!("{INVALID_TENSOR_FLAG}u"))
}

impl AdamWKernels {
    fn new(device: &wgpu::Device) -> Result<Self, TrainingError> {
        let limits = device.limits();
        if limits.max_storage_buffers_per_shader_stage < 8
            || limits.max_bindings_per_bind_group < 9
            || limits.max_uniform_buffers_per_shader_stage < 1
        {
            return Err(TensorError::Limit("resident AdamW bindings").into());
        }
        let layout = |commit: bool| {
            let uniform = if commit { 8 } else { 7 };
            let entries: Vec<_> = (0..=uniform)
                .map(|binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: if binding == uniform {
                            wgpu::BufferBindingType::Uniform
                        } else {
                            wgpu::BufferBindingType::Storage {
                                read_only: if commit {
                                    [0, 1, 2, 5].contains(&binding)
                                } else {
                                    [0, 1, 2, 4, 5].contains(&binding)
                                },
                            }
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect();
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("parameters.adamw.bindings"),
                entries: &entries,
            })
        };
        let prepare_layout = layout(false);
        let commit_layout = layout(true);
        let pipeline = |layout: &wgpu::BindGroupLayout, commit: bool, entry_point| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("parameters.adamw.pipeline"),
                bind_group_layouts: &[layout],
                push_constant_ranges: &[],
            });
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("parameters.adamw.shader"),
                source: wgpu::ShaderSource::Wgsl(source(commit).into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: Some(&layout),
                module: &module,
                entry_point,
                compilation_options: Default::default(),
            })
        };
        Ok(Self {
            prepare: pipeline(&prepare_layout, false, "prepare"),
            decide: pipeline(&prepare_layout, false, "decide"),
            commit: pipeline(&commit_layout, true, "commit"),
            prepare_layout,
            commit_layout,
        })
    }
}

pub(super) struct AdamWWorkspace {
    config: AdamWConfig,
    // Integer storage preserves signed zero and the full u32 clock on rejection.
    slots: Vec<Shared<wgpu::Buffer>>,
    kernels: AdamWKernels,
}

fn words(len: usize, multiples: usize) -> Result<usize, TrainingError> {
    let n = len
        .checked_mul(multiples)
        .and_then(|n| n.checked_add(1))
        .ok_or(TrainingError::Overflow)?;
    u32::try_from(n).map_err(|_| TrainingError::Overflow)?;
    Ok(n)
}

impl ResidentParameters {
    pub fn adamw_config(&self) -> Option<AdamWConfig> {
        self.adamw.as_ref().map(|w| w.config)
    }

    /// Start explicit zero history at the current weights. Refuses to reset an
    /// existing optimizer silently. Rate/decay schedules remain caller inputs.
    pub fn enable_adamw(&mut self, config: AdamWConfig) -> Result<(), TrainingError> {
        if self.adamw.is_some() {
            return Err(TrainingError::OptimizerState("AdamW already enabled"));
        }
        let gpu = self.device.runtime().context().device();
        let kernels = AdamWKernels::new(gpu)?;
        let slots = self
            .current
            .values
            .iter()
            .map(|p| {
                let count = words(p.layout().len(), 2)?;
                storage_limit(count, &gpu.limits())?;
                Ok(Shared::new(runtime::empty_buffer::<u32>(
                    gpu,
                    "adamw.zero_state",
                    count,
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                )?))
            })
            .collect::<Result<_, TrainingError>>()?;
        self.adamw = Some(AdamWWorkspace {
            config,
            slots,
            kernels,
        });
        Ok(())
    }

    pub fn restore_adamw(&mut self, state: &AdamWState) -> Result<(), TrainingError> {
        if self.adamw.is_some() {
            return Err(TrainingError::OptimizerState("AdamW already enabled"));
        }
        state.validate(
            &self
                .current
                .values
                .iter()
                .map(|p| p.layout().shape().to_vec())
                .collect::<Vec<_>>(),
            self.current.version.revision,
        )?;
        let gpu = self.device.runtime().context().device();
        let kernels = AdamWKernels::new(gpu)?;
        let slots = state
            .slots
            .iter()
            .map(|s| {
                storage_limit(words(s.first.len(), 2)?, &gpu.limits())?;
                let data: Vec<_> = s
                    .first
                    .iter()
                    .chain(&s.second)
                    .map(|x| x.to_bits())
                    .chain(std::iter::once(s.steps))
                    .collect();
                Ok(Shared::new(runtime::upload_slice(
                    gpu,
                    "adamw.restored_state",
                    &data,
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                )?))
            })
            .collect::<Result<_, TrainingError>>()?;
        self.adamw = Some(AdamWWorkspace {
            config: state.config,
            slots,
            kernels,
        });
        Ok(())
    }

    pub fn adamw_snapshot(&self) -> Result<Option<ResidentAdamWReadback>, TrainingError> {
        let Some(state) = &self.adamw else {
            return Ok(None);
        };
        let shapes: Vec<_> = self
            .current
            .values
            .iter()
            .map(|p| p.layout().shape().to_vec())
            .collect();
        let sources = state
            .slots
            .iter()
            .zip(&self.current.values)
            .map(|(s, p)| Ok((s.as_ref(), words(p.layout().len(), 2)?)))
            .collect::<Result<Vec<_>, TrainingError>>()?;
        Ok(Some(ResidentAdamWReadback {
            config: state.config,
            shapes,
            revision: self.current.version.revision,
            batch: runtime::ReadbackBatch::copy(
                self.device.runtime().context(),
                &sources,
                "adamw.snapshot",
            )?,
        }))
    }

    pub fn adamw(
        &mut self,
        gradients: &ResidentParameterGradients,
        step: AdamWStep,
    ) -> Result<ResidentParameterUpdate, TrainingError> {
        self.adamw_with_steps(gradients, &vec![step; self.current.values.len()])
    }

    /// One all-or-none transaction for values, both moments and per-tensor clocks.
    /// An invalid/frozen derivative still rejects the entire model. This never maps
    /// a GPU buffer; read the returned receipt explicitly to establish acceptance.
    pub fn adamw_with_steps(
        &mut self,
        gradients: &ResidentParameterGradients,
        steps: &[AdamWStep],
    ) -> Result<ResidentParameterUpdate, TrainingError> {
        let state = self
            .adamw
            .as_ref()
            .ok_or(TrainingError::OptimizerState("AdamW not enabled"))?;
        if !self.current.version.matches(&gradients.version) {
            return Err(TrainingError::ParameterVersion);
        }
        let count = self.current.values.len();
        if steps.len() != count {
            return Err(TrainingError::ParameterLayout);
        }
        let revision = self
            .current
            .version
            .revision
            .checked_add(1)
            .ok_or(TrainingError::Overflow)?;
        let context = self.device.runtime().context();
        let gpu = context.device();
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        let flags = Shared::new(runtime::empty_buffer::<u32>(
            gpu,
            "adamw.flags",
            count + 2,
            usage,
        )?);
        let mut encoder = gpu.create_command_encoder(&Default::default());
        let mut values = Vec::with_capacity(count);
        let mut slots = Vec::with_capacity(count);
        let mut passes = Vec::with_capacity(count);
        for (index, (p, g)) in self
            .current
            .values
            .iter()
            .zip(&gradients.values)
            .enumerate()
        {
            let p = p.contiguous_into(&mut encoder)?;
            let g = g.contiguous_into(&mut encoder)?;
            let len = p.layout().len();
            let candidate_words = words(len, 3)?;
            storage_limit(candidate_words, &gpu.limits())?;
            let groups = groups(len, &gpu.limits())?;
            let next = self.device.allocate_output(p.layout())?;
            let next_slot = Shared::new(runtime::empty_buffer::<u32>(
                gpu,
                "adamw.next_state",
                words(len, 2)?,
                usage,
            )?);
            let candidate =
                runtime::empty_buffer::<u32>(gpu, "adamw.candidate", candidate_words, usage)?;
            let config = runtime::upload_slice(
                gpu,
                "adamw.config",
                &[AdamWParams {
                    len: len as u32,
                    groups_x: groups[0],
                    count: count as u32,
                    index: index as u32,
                    rate: steps[index].rate(),
                    decay: steps[index].weight_decay(),
                    beta1: state.config.beta1(),
                    beta2: state.config.beta2(),
                    epsilon: state.config.epsilon(),
                    padding: [0; 3],
                }],
                wgpu::BufferUsages::UNIFORM,
            )?;
            let prepare = binding(
                gpu,
                &state.kernels.prepare_layout,
                &[
                    p.values(),
                    g.values(),
                    &state.slots[index],
                    &candidate,
                    p.flags(),
                    g.flags(),
                    &flags,
                    &config,
                ],
            );
            let commit = binding(
                gpu,
                &state.kernels.commit_layout,
                &[
                    p.values(),
                    &state.slots[index],
                    &candidate,
                    next.values(),
                    &next_slot,
                    p.flags(),
                    next.flags(),
                    &flags,
                    &config,
                ],
            );
            passes.push((prepare, commit, groups));
            values.push(next);
            slots.push(next_slot);
        }
        for (prepare, _, groups) in &passes {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&state.kernels.prepare);
            pass.set_bind_group(0, prepare, &[]);
            pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
        }
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&state.kernels.decide);
            pass.set_bind_group(0, &passes[0].0, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        for (_, commit, groups) in &passes {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&state.kernels.commit);
            pass.set_bind_group(0, commit, &[]);
            pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
        }
        context.queue().submit(Some(encoder.finish()));
        self.current.values = values;
        self.current.version.revision = revision;
        self.adamw.as_mut().unwrap().slots = slots;
        Ok(ResidentParameterUpdate {
            device: self.device.clone(),
            flags,
            count,
            revision,
        })
    }
}

/// Immutable optimizer capture. Model values must be captured from the same
/// owner revision by a complete training-checkpoint API, not inferred from this.
pub struct ResidentAdamWReadback {
    config: AdamWConfig,
    shapes: Vec<Vec<usize>>,
    revision: u64,
    batch: runtime::ReadbackBatch<u32>,
}

impl ResidentAdamWReadback {
    fn decode(
        config: AdamWConfig,
        shapes: Vec<Vec<usize>>,
        revision: u64,
        data: Vec<Vec<u32>>,
    ) -> Result<AdamWState, TrainingError> {
        if data.len() != shapes.len() {
            return Err(TrainingError::InvalidReadback);
        }
        let mut slots = Vec::new();
        for (shape, data) in shapes.iter().zip(data) {
            let len: usize = shape.iter().product();
            if data.len() != words(len, 2)? {
                return Err(TrainingError::InvalidReadback);
            }
            slots.push(AdamWSlot {
                shape: shape.clone(),
                first: data[..len]
                    .iter()
                    .map(|x| f32::from_bits(u32::from_le(*x)))
                    .collect(),
                second: data[len..2 * len]
                    .iter()
                    .map(|x| f32::from_bits(u32::from_le(*x)))
                    .collect(),
                steps: u32::from_le(data[2 * len]),
            });
        }
        let state = AdamWState { config, slots };
        state.validate(&shapes, revision)?;
        Ok(state)
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read(self) -> Result<AdamWState, TrainingError> {
        Self::decode(self.config, self.shapes, self.revision, self.batch.read()?)
    }
    #[cfg(target_arch = "wasm32")]
    pub async fn read_async(self) -> Result<AdamWState, TrainingError> {
        Self::decode(
            self.config,
            self.shapes,
            self.revision,
            self.batch.read_async().await?,
        )
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
