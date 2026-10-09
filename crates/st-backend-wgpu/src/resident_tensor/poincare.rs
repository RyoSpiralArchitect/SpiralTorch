//! Resident causal metric score bias. Coordinates are not projected or clipped.
use super::*;
use st_kernel_contracts::euclidean::{EuclideanBiasError, EuclideanBiasSpec};
use st_kernel_contracts::poincare::{PoincareBiasSpec, PoincareError};
mod pair_spec;
use pair_spec::PairSpec;

#[derive(Clone, Debug)]
pub struct ResidentPoincareBiasForward {
    spec: PoincareBiasSpec,
    tape: PairForward,
}

#[derive(Clone, Debug)]
pub struct ResidentEuclideanBiasForward {
    spec: EuclideanBiasSpec,
    tape: PairForward,
}

#[derive(Clone, Debug)]
struct PairForward {
    spec: PairSpec,
    coordinates: ResidentTensor,
    raw_gain: ResidentTensor,
    cache: ResidentTensor,
    scores: ResidentTensor,
}

#[derive(Clone, Debug)]
pub struct ResidentPairBiasVjp {
    coordinates: ResidentTensor,
    raw_gain: ResidentTensor,
}

pub type ResidentPoincareBiasVjp = ResidentPairBiasVjp;

impl ResidentPairBiasVjp {
    pub fn coordinates(&self) -> &ResidentTensor {
        &self.coordinates
    }
    pub fn raw_gain(&self) -> &ResidentTensor {
        &self.raw_gain
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    batch: u32,
    steps: u32,
    cols: u32,
    heads: u32,
    coordinates: u32,
    pairs: u32,
    scores: u32,
    groups_x: u32,
    groups: u32,
    coordinates_offset: u32,
    gain_offset: u32,
    seed_offset: u32,
    curvature_magnitude: f32,
    distance_scale: f32,
    metric_kind: u32,
    padding: u32,
}

fn source() -> String {
    include_str!("shaders/poincare.wgsl")
        .replace("ROUNDED_ADD", include_str!("../shaders/rounded_add.wgsl"))
        .replace(
            "WIDE_ARITHMETIC",
            include_str!("shaders/layer_norm_wide.wgsl"),
        )
        .replace("INVALID_TENSOR_FLAG", &format!("{INVALID_TENSOR_FLAG}u"))
}

#[derive(Debug)]
pub(super) struct PoincareKernels {
    layout: wgpu::BindGroupLayout,
    pairs: wgpu::ComputePipeline,
    scores: wgpu::ComputePipeline,
    pair_seeds: wgpu::ComputePipeline,
    coordinates_vjp: wgpu::ComputePipeline,
    gain_vjp: wgpu::ComputePipeline,
}

impl PoincareKernels {
    fn new(gpu: &wgpu::Device) -> Self {
        let layout = gpu.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("poincare.layout"),
            entries: &(0..=6)
                .map(|binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: if binding == 6 {
                            wgpu::BufferBindingType::Uniform
                        } else {
                            wgpu::BufferBindingType::Storage {
                                read_only: ![2, 4, 5].contains(&binding),
                            }
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect::<Vec<_>>(),
        });
        let module = gpu.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("poincare.shader"),
            source: wgpu::ShaderSource::Wgsl(source().into()),
        });
        let pipeline_layout = gpu.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("poincare.pipeline"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = |entry| {
            gpu.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: entry,
                compilation_options: Default::default(),
            })
        };
        Self {
            pairs: pipeline("prepare_pairs"),
            scores: pipeline("scores"),
            pair_seeds: pipeline("prepare_pair_seeds"),
            coordinates_vjp: pipeline("coordinates_vjp"),
            gain_vjp: pipeline("gain_vjp"),
            layout,
        }
    }
}

fn sizes(spec: impl Into<PairSpec>, limits: &wgpu::Limits) -> Result<(usize, usize), TensorError> {
    let spec = spec.into();
    if limits.max_storage_buffers_per_shader_stage < 6
        || limits.max_bindings_per_bind_group < 7
        || limits.max_uniform_buffers_per_shader_stage < 1
        || limits.max_uniform_buffer_binding_size < std::mem::size_of::<Params>() as u32
    {
        return Err(TensorError::Limit("Poincare bindings"));
    }
    let cache = spec
        .pairs_len()
        .checked_mul(16)
        .ok_or(PoincareError::Overflow)?;
    let gradients = spec
        .coordinates_len()
        .checked_add(spec.heads())
        .and_then(|len| spec.pairs_len().checked_mul(4)?.checked_add(len))
        .ok_or(PoincareError::Overflow)?;
    for count in [cache, gradients, spec.scores_len()] {
        storage_limit(count, limits)?;
    }
    for work in [
        spec.pairs_len(),
        spec.scores_len(),
        spec.coordinates_len(),
        spec.heads(),
    ] {
        grid(work, limits)?;
    }
    Ok((cache, gradients))
}

fn preflight(spec: PairSpec, tensors: &[&ResidentTensor]) -> Result<(usize, usize), TensorError> {
    let context = tensors[0].device.runtime().context();
    let limits = context.device().limits();
    for t in tensors {
        t.require_context(context)?;
        validate_view(&t.layout, (t.values().size() / 4) as usize, &limits)?;
    }
    sizes(spec, &limits)
}

fn params(spec: PairSpec) -> Params {
    let [batch, steps, cols] = spec.shape();
    Params {
        batch: batch as u32,
        steps: steps as u32,
        cols: cols as u32,
        heads: spec.heads() as u32,
        coordinates: spec.coordinates_len() as u32,
        pairs: spec.pairs_len() as u32,
        scores: spec.scores_len() as u32,
        groups_x: 0,
        groups: 0,
        coordinates_offset: 0,
        gain_offset: 0,
        seed_offset: 0,
        curvature_magnitude: match spec {
            PairSpec::Poincare(s) => -s.curvature(),
            PairSpec::Euclidean(_) => 0.,
        },
        distance_scale: match spec {
            PairSpec::Poincare(_) => 0.,
            PairSpec::Euclidean(s) => s.scale(),
        },
        metric_kind: u32::from(matches!(spec, PairSpec::Euclidean(_))),
        padding: 0,
    }
}

fn flags(
    device: &TensorDevice,
    upstream: &ResidentTensor,
    encoder: &mut wgpu::CommandEncoder,
) -> Result<Shared<wgpu::Buffer>, TensorError> {
    let flags = Shared::new(runtime::empty_buffer::<u32>(
        device.runtime().context().device(),
        "poincare.guard",
        1,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    )?);
    encoder.copy_buffer_to_buffer(upstream.flags(), 0, &flags, 0, 4);
    Ok(flags)
}

fn dispatch(
    device: &TensorDevice,
    encoder: &mut wgpu::CommandEncoder,
    kernels: &PoincareKernels,
    pipeline: &wgpu::ComputePipeline,
    buffers: &[&wgpu::Buffer],
    mut p: Params,
    work: usize,
) -> Result<(), TensorError> {
    let gpu = device.runtime().context().device();
    let [x, y, groups] = grid(work, &gpu.limits())?;
    p.groups_x = x;
    p.groups = groups;
    let uniform = runtime::upload_slice(gpu, "poincare.params", &[p], wgpu::BufferUsages::UNIFORM)?;
    let mut entries: Vec<_> = buffers
        .iter()
        .enumerate()
        .map(|(binding, b)| wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: b.as_entire_binding(),
        })
        .collect();
    entries.push(wgpu::BindGroupEntry {
        binding: 6,
        resource: uniform.as_entire_binding(),
    });
    let group = gpu.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("poincare.bind"),
        layout: &kernels.layout,
        entries: &entries,
    });
    let mut pass = encoder.begin_compute_pass(&Default::default());
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &group, &[]);
    pass.dispatch_workgroups(x, y, 1);
    Ok(())
}

impl ResidentTensor {
    /// `[B,T,C]` open-ball coordinates and `[H]` gains -> `[B,H,T,T]` bias.
    /// Future bias is zero, not -infinity: still apply a structural causal mask.
    pub fn causal_poincare_bias(
        &self,
        raw_gain: &Self,
        curvature: f32,
    ) -> Result<ResidentPoincareBiasForward, TensorError> {
        let shape =
            <[usize; 3]>::try_from(self.layout.shape()).map_err(|_| PoincareError::Shape)?;
        if raw_gain.layout.shape().len() != 1 {
            return Err(PoincareError::Shape.into());
        }
        let spec = PoincareBiasSpec::new(shape, raw_gain.layout.shape()[0], curvature)?;
        Ok(ResidentPoincareBiasForward {
            spec,
            tape: self.causal_pair_bias(raw_gain, spec.into())?,
        })
    }

    /// `-softplus(raw_gain[h]) * scale * ||x_q - x_k||^2` for k <= q.
    /// Future entries are zero, not a replacement for a causal attention mask.
    pub fn causal_euclidean_bias(
        &self,
        raw_gain: &Self,
        scale: f32,
    ) -> Result<ResidentEuclideanBiasForward, TensorError> {
        let shape =
            <[usize; 3]>::try_from(self.layout.shape()).map_err(|_| EuclideanBiasError::Shape)?;
        if raw_gain.layout.shape().len() != 1 {
            return Err(EuclideanBiasError::Shape.into());
        }
        let spec = EuclideanBiasSpec::new(shape, raw_gain.layout.shape()[0], scale)?;
        Ok(ResidentEuclideanBiasForward {
            spec,
            tape: self.causal_pair_bias(raw_gain, spec.into())?,
        })
    }

    fn causal_pair_bias(
        &self,
        raw_gain: &Self,
        spec: PairSpec,
    ) -> Result<PairForward, TensorError> {
        let (cache_len, _) = preflight(spec, &[self, raw_gain])?;
        let device = &self.device;
        let context = device.runtime().context();
        let joined = device.guard_together(&[self, raw_gain])?;
        let mut encoder = context.device().create_command_encoder(&Default::default());
        let coordinates = joined[0].contiguous_into(&mut encoder)?;
        let raw_gain = joined[1].contiguous_into(&mut encoder)?;
        let flag = flags(device, &joined[0], &mut encoder)?;
        let cache = device
            .allocate_output_with_guard(&NdLayout::contiguous(&[cache_len])?, Some(flag.clone()))?;
        let scores = device
            .allocate_output_with_guard(&NdLayout::contiguous(&spec.score_shape())?, Some(flag))?;
        let mut p = params(spec);
        p.coordinates_offset = coordinates.layout.offset() as u32;
        p.gain_offset = raw_gain.layout.offset() as u32;
        let kernels = device
            .0
            .poincare
            .get_or_init(|| PoincareKernels::new(context.device()));
        let buffers = [
            coordinates.values(),
            raw_gain.values(),
            cache.values(),
            raw_gain.values(),
            scores.values(),
            scores.flags(),
        ];
        dispatch(
            device,
            &mut encoder,
            kernels,
            &kernels.pairs,
            &buffers,
            p,
            spec.pairs_len(),
        )?;
        dispatch(
            device,
            &mut encoder,
            kernels,
            &kernels.scores,
            &buffers,
            p,
            spec.scores_len(),
        )?;
        context.queue().submit(Some(encoder.finish()));
        let family = device.guard_together(&[&scores, &cache, &coordinates, &raw_gain])?;
        Ok(PairForward {
            spec,
            coordinates,
            raw_gain,
            cache: family[1].clone(),
            scores: family[0].clone(),
        })
    }
}

impl ResidentPoincareBiasForward {
    pub fn scores(&self) -> &ResidentTensor {
        &self.tape.scores
    }
    pub fn spec(&self) -> PoincareBiasSpec {
        self.spec
    }
    pub fn backward(&self, seed: &ResidentTensor) -> Result<ResidentPoincareBiasVjp, TensorError> {
        self.tape.backward(seed)
    }
}

impl ResidentEuclideanBiasForward {
    pub fn scores(&self) -> &ResidentTensor {
        &self.tape.scores
    }
    pub fn spec(&self) -> EuclideanBiasSpec {
        self.spec
    }
    pub fn backward(&self, seed: &ResidentTensor) -> Result<ResidentPairBiasVjp, TensorError> {
        self.tape.backward(seed)
    }
}

impl PairForward {
    fn backward(&self, seed: &ResidentTensor) -> Result<ResidentPairBiasVjp, TensorError> {
        let s = self.spec;
        if seed.layout.shape() != s.score_shape() {
            return Err(match s {
                PairSpec::Poincare(_) => PoincareError::Length.into(),
                PairSpec::Euclidean(_) => EuclideanBiasError::Length.into(),
            });
        }
        let (_, len) = preflight(s, &[&self.scores, seed])?;
        let device = &self.scores.device;
        let context = device.runtime().context();
        let joined = device.guard_together(&[&self.scores, seed])?;
        let mut encoder = context.device().create_command_encoder(&Default::default());
        let seed = joined[1].contiguous_into(&mut encoder)?;
        let flag = flags(device, &joined[0], &mut encoder)?;
        let packed =
            device.allocate_output_with_guard(&NdLayout::contiguous(&[len])?, Some(flag))?;
        let mut p = params(s);
        p.coordinates_offset = self.coordinates.layout.offset() as u32;
        p.gain_offset = self.raw_gain.layout.offset() as u32;
        p.seed_offset = seed.layout.offset() as u32;
        let kernels = device
            .0
            .poincare
            .get()
            .expect("Poincare forward compiled kernels");
        let buffers = [
            self.coordinates.values(),
            self.raw_gain.values(),
            self.cache.values(),
            seed.values(),
            packed.values(),
            packed.flags(),
        ];
        // The private tail belongs to this backward, never to the retained
        // forward cache. Reduce heads once per pair without host readback.
        dispatch(
            device,
            &mut encoder,
            kernels,
            &kernels.pair_seeds,
            &buffers,
            p,
            s.pairs_len(),
        )?;
        dispatch(
            device,
            &mut encoder,
            kernels,
            &kernels.coordinates_vjp,
            &buffers,
            p,
            s.coordinates_len(),
        )?;
        dispatch(
            device,
            &mut encoder,
            kernels,
            &kernels.gain_vjp,
            &buffers,
            p,
            s.heads(),
        )?;
        context.queue().submit(Some(encoder.finish()));
        let family = device.guard_together(&[&packed, &seed])?;
        Ok(ResidentPairBiasVjp {
            coordinates: family[0]
                .narrow(0, 0, s.coordinates_len())?
                .reshape(&s.shape())?,
            raw_gain: family[0].narrow(0, s.coordinates_len(), s.heads())?,
        })
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
