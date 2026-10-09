//! Resident causal complex-state filtering, chart coordinates and full BPTT.
use super::*;
use st_kernel_contracts::causal_wave::{CausalWaveError, CausalWaveSpec};

#[derive(Clone, Debug)]
pub struct ResidentCausalWaveForward {
    spec: CausalWaveSpec,
    drive: ResidentTensor,
    initial: ResidentTensor,
    packed: ResidentTensor,
    features: ResidentTensor,
    final_state: ResidentTensor,
}

#[derive(Clone, Debug)]
pub struct ResidentCausalWaveVjp {
    drive: ResidentTensor,
    raw_decay: ResidentTensor,
    raw_phase: ResidentTensor,
    initial_state: ResidentTensor,
}

impl ResidentCausalWaveVjp {
    pub fn drive(&self) -> &ResidentTensor {
        &self.drive
    }
    pub fn raw_decay(&self) -> &ResidentTensor {
        &self.raw_decay
    }
    pub fn raw_phase(&self) -> &ResidentTensor {
        &self.raw_phase
    }
    pub fn initial_state(&self) -> &ResidentTensor {
        &self.initial_state
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    batch: u32,
    steps: u32,
    cols: u32,
    pairs: u32,
    values: u32,
    state_values: u32,
    groups_x: u32,
    groups: u32,
    radius: f32,
    drive_offset: u32,
    decay_offset: u32,
    phase_offset: u32,
    initial_offset: u32,
    cotangent_offset: u32,
    terminal_offset: u32,
    padding: u32,
}

#[derive(Debug)]
pub(super) struct CausalWaveKernels {
    forward_layout: wgpu::BindGroupLayout,
    backward_layout: wgpu::BindGroupLayout,
    forward: wgpu::ComputePipeline,
    project: wgpu::ComputePipeline,
    project_vjp: wgpu::ComputePipeline,
    backward: wgpu::ComputePipeline,
    reduce: wgpu::ComputePipeline,
}

fn source(template: &str) -> String {
    template
        .replace("COMMON", include_str!("shaders/causal_wave_common.wgsl"))
        .replace("ROUNDED_ADD", include_str!("../shaders/rounded_add.wgsl"))
        .replace(
            "WIDE_ARITHMETIC",
            include_str!("shaders/layer_norm_wide.wgsl"),
        )
        .replace("INVALID_TENSOR_FLAG", &format!("{INVALID_TENSOR_FLAG}u"))
        .replace(
            "MAX_DECAY",
            &format!("{:?}", st_kernel_contracts::causal_wave::MAX_DECAY),
        )
        .replace("PHASE_LIMIT", &format!("{:?}", std::f32::consts::PI))
}

impl CausalWaveKernels {
    fn new(gpu: &wgpu::Device) -> Self {
        let layout = |count: u32, writable: &[u32]| {
            gpu.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("causal_wave.layout"),
                entries: &(0..=count)
                    .map(|binding| wgpu::BindGroupLayoutEntry {
                        binding,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: if binding == count {
                                wgpu::BufferBindingType::Uniform
                            } else {
                                wgpu::BufferBindingType::Storage {
                                    read_only: !writable.contains(&binding),
                                }
                            },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    })
                    .collect::<Vec<_>>(),
            })
        };
        let forward_layout = layout(7, &[4, 5, 6]);
        let backward_layout = layout(8, &[5, 6, 7]);
        let module = |template| {
            gpu.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("causal_wave.shader"),
                source: wgpu::ShaderSource::Wgsl(source(template).into()),
            })
        };
        let f = module(include_str!("shaders/causal_wave.wgsl"));
        let b = module(include_str!("shaders/causal_wave_backward.wgsl"));
        let pipeline = |layout: &wgpu::BindGroupLayout, module: &wgpu::ShaderModule, entry| {
            let layout = gpu.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("causal_wave.pipeline"),
                bind_group_layouts: &[layout],
                push_constant_ranges: &[],
            });
            gpu.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module,
                entry_point: entry,
                compilation_options: Default::default(),
            })
        };
        Self {
            forward: pipeline(&forward_layout, &f, "scan"),
            project: pipeline(&forward_layout, &f, "project"),
            project_vjp: pipeline(&backward_layout, &b, "project_vjp"),
            backward: pipeline(&backward_layout, &b, "pullback"),
            reduce: pipeline(&backward_layout, &b, "reduce_parameters"),
            forward_layout,
            backward_layout,
        }
    }
}

fn preflight(
    spec: CausalWaveSpec,
    tensors: &[&ResidentTensor],
) -> Result<(usize, usize), TensorError> {
    let context = tensors[0].device.runtime().context();
    let limits = context.device().limits();
    for t in tensors {
        t.require_context(context)?;
        validate_view(&t.layout, (t.values().size() / 4) as usize, &limits)?;
    }
    preflight_sizes(spec, &limits)
}

fn preflight_sizes(
    spec: CausalWaveSpec,
    limits: &wgpu::Limits,
) -> Result<(usize, usize), TensorError> {
    if limits.max_storage_buffers_per_shader_stage < 8
        || limits.max_bindings_per_bind_group < 9
        || limits.max_uniform_buffers_per_shader_stage < 1
        || limits.max_uniform_buffer_binding_size < std::mem::size_of::<Params>() as u32
    {
        return Err(TensorError::Limit("causal wave bindings"));
    }
    let add = |a: usize, b: usize| a.checked_add(b).ok_or(CausalWaveError::Overflow);
    let forward = add(
        add(
            spec.len(),
            spec.pairs()
                .checked_mul(5)
                .ok_or(CausalWaveError::Overflow)?,
        )?,
        spec.state_len(),
    )?;
    let backward = add(
        add(
            spec.len(),
            spec.state_len()
                .checked_mul(2)
                .ok_or(CausalWaveError::Overflow)?,
        )?,
        spec.shape()[2],
    )?;
    let adjoints = spec.len().checked_mul(4).ok_or(CausalWaveError::Overflow)?;
    for len in [forward, backward, adjoints] {
        storage_limit(len, limits)?;
    }
    for work in [
        spec.shape()[0] * spec.pairs(),
        spec.shape()[0] * spec.shape()[1],
        spec.pairs(),
    ] {
        grid(work, limits)?;
    }
    Ok((forward, backward))
}

fn params(spec: CausalWaveSpec) -> Params {
    let [batch, steps, cols] = spec.shape();
    Params {
        batch: batch as u32,
        steps: steps as u32,
        cols: cols as u32,
        pairs: spec.pairs() as u32,
        values: spec.len() as u32,
        state_values: spec.state_len() as u32,
        radius: spec.radius(),
        groups_x: 0,
        groups: 0,
        drive_offset: 0,
        decay_offset: 0,
        phase_offset: 0,
        initial_offset: 0,
        cotangent_offset: 0,
        terminal_offset: 0,
        padding: 0,
    }
}

fn guard(
    device: &TensorDevice,
    upstream: &ResidentTensor,
    encoder: &mut wgpu::CommandEncoder,
) -> Result<Shared<wgpu::Buffer>, TensorError> {
    let flags = Shared::new(runtime::empty_buffer::<u32>(
        device.runtime().context().device(),
        "causal_wave.guard",
        1,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    )?);
    encoder.copy_buffer_to_buffer(upstream.flags(), 0, &flags, 0, 4);
    Ok(flags)
}

fn dispatch(
    device: &TensorDevice,
    encoder: &mut wgpu::CommandEncoder,
    layout: &wgpu::BindGroupLayout,
    pipeline: &wgpu::ComputePipeline,
    buffers: &[&wgpu::Buffer],
    mut params: Params,
    work: usize,
) -> Result<(), TensorError> {
    let gpu = device.runtime().context().device();
    let [x, y, groups] = grid(work, &gpu.limits())?;
    params.groups_x = x;
    params.groups = groups;
    let uniform = runtime::upload_slice(
        gpu,
        "causal_wave.params",
        &[params],
        wgpu::BufferUsages::UNIFORM,
    )?;
    let mut entries: Vec<_> = buffers
        .iter()
        .enumerate()
        .map(|(binding, b)| wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: b.as_entire_binding(),
        })
        .collect();
    entries.push(wgpu::BindGroupEntry {
        binding: buffers.len() as u32,
        resource: uniform.as_entire_binding(),
    });
    let binding = gpu.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("causal_wave.bind"),
        layout,
        entries: &entries,
    });
    let mut pass = encoder.begin_compute_pass(&Default::default());
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &binding, &[]);
    pass.dispatch_workgroups(x, y, 1);
    Ok(())
}

impl ResidentTensor {
    /// Input [B,T,2P], two parameter vectors [P], raw initial state [B,2P].
    /// Empty axes are rejected. Sequence and batch never share recurrent state.
    pub fn causal_zspace_wave(
        &self,
        raw_decay: &Self,
        raw_phase: &Self,
        initial_state: &Self,
        curvature: f32,
    ) -> Result<ResidentCausalWaveForward, TensorError> {
        let shape =
            <[usize; 3]>::try_from(self.layout.shape()).map_err(|_| CausalWaveError::Shape)?;
        let spec = CausalWaveSpec::new(shape, curvature)?;
        if raw_decay.layout.shape() != [spec.pairs()]
            || raw_phase.layout.shape() != [spec.pairs()]
            || initial_state.layout.shape() != [shape[0], shape[2]]
        {
            return Err(CausalWaveError::Length.into());
        }
        let inputs = [self, raw_decay, raw_phase, initial_state];
        let (len, _) = preflight(spec, &inputs)?;
        let device = &self.device;
        let joined = device.guard_together(&inputs)?;
        let context = device.runtime().context();
        let mut encoder = context.device().create_command_encoder(&Default::default());
        let packed_inputs = joined
            .iter()
            .map(|t| t.contiguous_into(&mut encoder))
            .collect::<Result<Vec<_>, _>>()?;
        let flags = guard(device, &joined[0], &mut encoder)?;
        let packed = device
            .allocate_output_with_guard(&NdLayout::contiguous(&[len])?, Some(flags.clone()))?;
        let features =
            device.allocate_output_with_guard(&NdLayout::contiguous(&shape)?, Some(flags))?;
        let mut p = params(spec);
        p.drive_offset = packed_inputs[0].layout.offset() as u32;
        p.decay_offset = packed_inputs[1].layout.offset() as u32;
        p.phase_offset = packed_inputs[2].layout.offset() as u32;
        p.initial_offset = packed_inputs[3].layout.offset() as u32;
        let kernels = device
            .0
            .causal_wave
            .get_or_init(|| CausalWaveKernels::new(context.device()));
        let buffers = [
            packed_inputs[0].values(),
            packed_inputs[1].values(),
            packed_inputs[2].values(),
            packed_inputs[3].values(),
            packed.values(),
            features.values(),
            features.flags(),
        ];
        dispatch(
            device,
            &mut encoder,
            &kernels.forward_layout,
            &kernels.forward,
            &buffers,
            p,
            shape[0] * spec.pairs(),
        )?;
        dispatch(
            device,
            &mut encoder,
            &kernels.forward_layout,
            &kernels.project,
            &buffers,
            p,
            shape[0] * shape[1],
        )?;
        context.queue().submit(Some(encoder.finish()));
        let mut family = vec![&features, &packed];
        family.extend(packed_inputs.iter());
        let family = device.guard_together(&family)?;
        let features = family[0].clone();
        let packed = family[1].clone();
        let final_state = packed
            .narrow(0, spec.len() + spec.pairs() * 5, spec.state_len())?
            .reshape(&[shape[0], shape[2]])?;
        Ok(ResidentCausalWaveForward {
            spec,
            drive: packed_inputs[0].clone(),
            initial: packed_inputs[3].clone(),
            packed,
            features,
            final_state,
        })
    }
}

impl ResidentCausalWaveForward {
    pub fn features(&self) -> &ResidentTensor {
        &self.features
    }
    pub fn final_state(&self) -> &ResidentTensor {
        &self.final_state
    }
    pub fn backward(
        &self,
        feature_cotangent: &ResidentTensor,
        terminal_cotangent: &ResidentTensor,
    ) -> Result<ResidentCausalWaveVjp, TensorError> {
        let spec = self.spec;
        let [batch, steps, cols] = spec.shape();
        if feature_cotangent.layout.shape() != spec.shape()
            || terminal_cotangent.layout.shape() != [batch, cols]
        {
            return Err(CausalWaveError::Length.into());
        }
        let (_, len) = preflight(
            spec,
            &[&self.features, feature_cotangent, terminal_cotangent],
        )?;
        let device = &self.features.device;
        let context = device.runtime().context();
        let joined =
            device.guard_together(&[&self.features, feature_cotangent, terminal_cotangent])?;
        let mut encoder = context.device().create_command_encoder(&Default::default());
        let seed = joined[1].contiguous_into(&mut encoder)?;
        let terminal = joined[2].contiguous_into(&mut encoder)?;
        let flags = guard(device, &joined[0], &mut encoder)?;
        let state_grad = device.allocate_output_with_guard(
            &NdLayout::contiguous(&[spec.len() * 4])?,
            Some(flags.clone()),
        )?;
        let packed =
            device.allocate_output_with_guard(&NdLayout::contiguous(&[len])?, Some(flags))?;
        let mut p = params(spec);
        p.drive_offset = self.drive.layout.offset() as u32;
        p.initial_offset = self.initial.layout.offset() as u32;
        p.cotangent_offset = seed.layout.offset() as u32;
        p.terminal_offset = terminal.layout.offset() as u32;
        let kernels = device
            .0
            .causal_wave
            .get()
            .expect("forward compiled causal wave kernels");
        let buffers = [
            self.drive.values(),
            self.packed.values(),
            self.initial.values(),
            seed.values(),
            terminal.values(),
            state_grad.values(),
            packed.values(),
            packed.flags(),
        ];
        dispatch(
            device,
            &mut encoder,
            &kernels.backward_layout,
            &kernels.project_vjp,
            &buffers,
            p,
            batch * steps,
        )?;
        dispatch(
            device,
            &mut encoder,
            &kernels.backward_layout,
            &kernels.backward,
            &buffers,
            p,
            batch * spec.pairs(),
        )?;
        dispatch(
            device,
            &mut encoder,
            &kernels.backward_layout,
            &kernels.reduce,
            &buffers,
            p,
            spec.pairs(),
        )?;
        context.queue().submit(Some(encoder.finish()));
        let family = device.guard_together(&[&packed, &seed, &terminal])?;
        let packed = &family[0];
        let parameter_offset = spec.len() + 2 * spec.state_len();
        Ok(ResidentCausalWaveVjp {
            drive: packed.narrow(0, 0, spec.len())?.reshape(&spec.shape())?,
            initial_state: packed
                .narrow(0, spec.len(), spec.state_len())?
                .reshape(&[batch, cols])?,
            raw_decay: packed.narrow(0, parameter_offset, spec.pairs())?,
            raw_phase: packed.narrow(0, parameter_offset + spec.pairs(), spec.pairs())?,
        })
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
