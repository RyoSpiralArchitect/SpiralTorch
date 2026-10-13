//! Resident headwise softmax mixtures with component and router pullbacks.
use super::*;
use st_kernel_contracts::mixture::{HeadwiseMixtureSpec, MixtureError};

mod querywise;
pub use querywise::{ResidentQuerywiseMixture, ResidentQuerywiseMixtureVjp};

#[derive(Clone, Debug)]
pub struct ResidentHeadwiseMixture {
    spec: HeadwiseMixtureSpec,
    input: ResidentTensor,
    weights: ResidentTensor,
    output: ResidentTensor,
}

#[derive(Clone, Debug)]
pub struct ResidentHeadwiseMixtureVjp {
    components: ResidentTensor,
    logits: ResidentTensor,
}
impl ResidentHeadwiseMixtureVjp {
    pub fn components(&self) -> &ResidentTensor {
        &self.components
    }
    pub fn logits(&self) -> &ResidentTensor {
        &self.logits
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    batch: u32,
    heads: u32,
    positions: u32,
    components: u32,
    groups_x: u32,
    groups: u32,
    pad0: u32,
    pad1: u32,
}

#[derive(Debug)]
pub(super) struct MixtureKernels {
    forward_layout: wgpu::BindGroupLayout,
    backward_layout: wgpu::BindGroupLayout,
    weights: wgpu::ComputePipeline,
    forward: wgpu::ComputePipeline,
    components: wgpu::ComputePipeline,
    logits: wgpu::ComputePipeline,
}

fn source(template: &str) -> String {
    template
        .replace("COMMON", include_str!("shaders/mixture_common.wgsl"))
        .replace("ROUNDED_ADD", include_str!("../shaders/rounded_add.wgsl"))
        .replace("EXACT_ROUTER", include_str!("shaders/mixture_exact.wgsl"))
        .replace("INVALID_TENSOR_FLAG", &format!("{INVALID_TENSOR_FLAG}u"))
}

impl MixtureKernels {
    fn new(gpu: &wgpu::Device) -> Self {
        let layout = |count: u32, writable: u32| {
            gpu.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("mixture.bindings"),
                entries: &(0..=count)
                    .map(|binding| wgpu::BindGroupLayoutEntry {
                        binding,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: if binding == count {
                                wgpu::BufferBindingType::Uniform
                            } else {
                                wgpu::BufferBindingType::Storage {
                                    read_only: binding < writable,
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
        let forward_layout = layout(5, 2);
        let backward_layout = layout(6, 3);
        let module = |template| {
            gpu.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("mixture.shader"),
                source: wgpu::ShaderSource::Wgsl(source(template).into()),
            })
        };
        let f = module(include_str!("shaders/mixture.wgsl"));
        let b = module(include_str!("shaders/mixture_backward.wgsl"));
        let pipeline = |layout: &wgpu::BindGroupLayout, module: &wgpu::ShaderModule, entry| {
            let layout = gpu.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("mixture.pipeline"),
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
            weights: pipeline(&forward_layout, &f, "weights"),
            forward: pipeline(&forward_layout, &f, "forward"),
            components: pipeline(&backward_layout, &b, "components"),
            logits: pipeline(&backward_layout, &b, "logits"),
            forward_layout,
            backward_layout,
        }
    }
}

fn preflight(spec: HeadwiseMixtureSpec, limits: &wgpu::Limits) -> Result<(), TensorError> {
    if limits.max_storage_buffers_per_shader_stage < 6
        || limits.max_bindings_per_bind_group < 7
        || limits.max_uniform_buffers_per_shader_stage < 1
        || limits.max_uniform_buffer_binding_size < std::mem::size_of::<Params>() as u32
    {
        return Err(TensorError::Limit("mixture bindings"));
    }
    for len in [spec.len(), spec.output_len(), spec.router_len()] {
        storage_limit(len, limits)?;
        grid(len, limits)?;
    }
    Ok(())
}

fn guard(
    device: &TensorDevice,
    upstream: &ResidentTensor,
    encoder: &mut wgpu::CommandEncoder,
) -> Result<Shared<wgpu::Buffer>, TensorError> {
    let flags = Shared::new(runtime::empty_buffer::<u32>(
        device.runtime().context().device(),
        "mixture.guard",
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
    spec: HeadwiseMixtureSpec,
    work: usize,
) -> Result<(), TensorError> {
    let gpu = device.runtime().context().device();
    let [x, y, groups] = grid(work, &gpu.limits())?;
    let [batch, heads, positions, components] = spec.shape();
    let p = Params {
        batch: batch as u32,
        heads: heads as u32,
        positions: positions as u32,
        components: components as u32,
        groups_x: x,
        groups,
        pad0: 0,
        pad1: 0,
    };
    let uniform = runtime::upload_slice(gpu, "mixture.params", &[p], wgpu::BufferUsages::UNIFORM)?;
    let mut entries: Vec<_> = buffers
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: b.as_entire_binding(),
        })
        .collect();
    entries.push(wgpu::BindGroupEntry {
        binding: buffers.len() as u32,
        resource: uniform.as_entire_binding(),
    });
    let binding = gpu.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mixture.bind"),
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
    /// Mix [B,H,P,K] with softmax router logits [H,K], yielding [B,H,P].
    /// All K components participate; a failed zero-weight component still
    /// invalidates the whole operation. K is portable and bounded to 2..=8.
    pub fn headwise_mixture(&self, logits: &Self) -> Result<ResidentHeadwiseMixture, TensorError> {
        let shape = <[usize; 4]>::try_from(self.layout.shape()).map_err(|_| MixtureError::Shape)?;
        let spec = HeadwiseMixtureSpec::new(shape)?;
        let [batch, heads, positions, components] = shape;
        if logits.layout.shape() != [heads, components] {
            return Err(MixtureError::Length.into());
        }
        let device = &self.device;
        let context = device.runtime().context();
        logits.require_context(context)?;
        preflight(spec, &context.device().limits())?;
        let joined = device.guard_together(&[self, logits])?;
        let mut encoder = context.device().create_command_encoder(&Default::default());
        let input = joined[0].contiguous_into(&mut encoder)?;
        let logits = joined[1].contiguous_into(&mut encoder)?;
        let flags = guard(device, &joined[0], &mut encoder)?;
        let output = device.allocate_output_with_guard(
            &NdLayout::contiguous(&[batch, heads, positions])?,
            Some(flags),
        )?;
        let weights = device.allocate_output_with_guard(
            &NdLayout::contiguous(&[heads, components])?,
            Some(output.storage.flags.clone()),
        )?;
        let kernels = device
            .0
            .mixture
            .get_or_init(|| MixtureKernels::new(context.device()));
        let buffers = [
            input.values(),
            logits.values(),
            weights.values(),
            output.values(),
            output.flags(),
        ];
        dispatch(
            device,
            &mut encoder,
            &kernels.forward_layout,
            &kernels.weights,
            &buffers,
            spec,
            heads,
        )?;
        dispatch(
            device,
            &mut encoder,
            &kernels.forward_layout,
            &kernels.forward,
            &buffers,
            spec,
            spec.output_len(),
        )?;
        context.queue().submit(Some(encoder.finish()));
        let family = device.guard_together(&[&output, &weights, &input, &logits])?;
        Ok(ResidentHeadwiseMixture {
            spec,
            input: family[2].clone(),
            weights: family[1].clone(),
            output: family[0].clone(),
        })
    }
}

impl ResidentHeadwiseMixture {
    pub fn output(&self) -> &ResidentTensor {
        &self.output
    }
    pub fn weights(&self) -> &ResidentTensor {
        &self.weights
    }
    pub fn backward(
        &self,
        seed: &ResidentTensor,
    ) -> Result<ResidentHeadwiseMixtureVjp, TensorError> {
        if seed.layout.shape() != self.output.layout.shape() {
            return Err(MixtureError::Length.into());
        }
        let device = &self.output.device;
        let context = device.runtime().context();
        seed.require_context(context)?;
        preflight(self.spec, &context.device().limits())?;
        let joined = device.guard_together(&[&self.output, seed])?;
        let mut encoder = context.device().create_command_encoder(&Default::default());
        let seed = joined[1].contiguous_into(&mut encoder)?;
        let flags = guard(device, &joined[0], &mut encoder)?;
        let components = device
            .allocate_output_with_guard(&NdLayout::contiguous(&self.spec.shape())?, Some(flags))?;
        let [_, heads, _, count] = self.spec.shape();
        let logits = device.allocate_output_with_guard(
            &NdLayout::contiguous(&[heads, count])?,
            Some(components.storage.flags.clone()),
        )?;
        let kernels = device
            .0
            .mixture
            .get()
            .expect("forward compiled mixture kernels");
        let buffers = [
            self.input.values(),
            self.weights.values(),
            seed.values(),
            components.values(),
            logits.values(),
            components.flags(),
        ];
        dispatch(
            device,
            &mut encoder,
            &kernels.backward_layout,
            &kernels.components,
            &buffers,
            self.spec,
            self.spec.len(),
        )?;
        dispatch(
            device,
            &mut encoder,
            &kernels.backward_layout,
            &kernels.logits,
            &buffers,
            self.spec,
            heads,
        )?;
        context.queue().submit(Some(encoder.finish()));
        let family = device.guard_together(&[&components, &logits, &seed])?;
        Ok(ResidentHeadwiseMixtureVjp {
            components: family[0].clone(),
            logits: family[1].clone(),
        })
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
