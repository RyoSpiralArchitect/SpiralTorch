//! Lazily prepared global clipping; disabled learners keep the old update path.
use super::*;
use st_kernel_contracts::gradient_clip::{
    MAX_SCALE_FACTORS, NORM_FLOOR, SCALE_CHUNK, SCALE_EPSILON,
};

pub(super) struct Clipping {
    passes: Vec<Pass>,
    gradients: Vec<Pass>,
}

impl Clipping {
    pub(super) fn new(g: &ResidentGraphTraining) -> Result<Self, TrainingError> {
        let device = g.device.runtime().context().device();
        let limits = device.limits();
        let counts: Vec<_> = g
            .definition
            .parameters()
            .iter()
            .map(|p| p.values.len().div_ceil(256))
            .collect();
        let partial_count = counts
            .iter()
            .try_fold(0usize, |a, &b| a.checked_add(b))
            .ok_or(TrainingError::Overflow)?;
        let partial_len = partial_count
            .checked_mul(2)
            .ok_or(TrainingError::Overflow)?;
        storage_limit(partial_len.max(1), &limits)?;
        let partials = runtime::empty_buffer::<f32>(
            device,
            "learner.clip.partials",
            partial_len.max(1),
            wgpu::BufferUsages::STORAGE,
        )?;
        let factors = runtime::empty_buffer::<f32>(
            device,
            "learner.clip.factors",
            1 + MAX_SCALE_FACTORS,
            wgpu::BufferUsages::STORAGE,
        )?;
        let unused_read = runtime::empty_buffer::<f32>(
            device,
            "learner.clip.unused_read",
            1,
            wgpu::BufferUsages::STORAGE,
        )?;
        let unused_aux = runtime::empty_buffer::<f32>(
            device,
            "learner.clip.unused_aux",
            1,
            wgpu::BufferUsages::STORAGE,
        )?;
        let entries: Vec<_> = (0..9)
            .map(|i| composition::entry(i, i >= 7, i < 4))
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("learner.clip.layout"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("learner.clip.pipeline_layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let extra = include_str!("../../../../shaders/graph_gradient_clip.wgsl")
            .replace("CLIP_NORM_FLOOR", &format!("{NORM_FLOOR:e}"))
            .replace("CLIP_SCALE_CHUNK", &format!("{SCALE_CHUNK:e}"))
            .replace("CLIP_SCALE_EPSILON", &format!("{SCALE_EPSILON:e}"));
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("learner.clip.shader"),
            source: wgpu::ShaderSource::Wgsl((training_scalar_source() + &extra).into()),
        });
        let pipeline = |entry| {
            Shared::new(
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: entry,
                    compilation_options: Default::default(),
                }),
            )
        };
        let reduce_partials = pipeline("clip_partials");
        let reduce = pipeline("clip_reduce");
        let prepare = pipeline("clip_prepare");
        let gradient = pipeline("clip_gradient");
        let element = |pipeline: &Shared<wgpu::ComputePipeline>,
                       mut p: Params,
                       buffers: [&wgpu::Buffer; 6],
                       single: bool|
         -> Result<Pass, TrainingError> {
            let grid = if single {
                [1, 1, 1]
            } else {
                groups(p.len as usize, &limits)?
            };
            p.groups_x = grid[0];
            let uniform = runtime::upload_slice(
                device,
                "learner.clip.params",
                &[p],
                wgpu::BufferUsages::UNIFORM,
            )?;
            Ok(Pass {
                pipeline: pipeline.clone(),
                groups: grid,
                binding: binding(
                    device,
                    &layout,
                    &[
                        buffers[0],
                        buffers[1],
                        buffers[2],
                        buffers[3],
                        buffers[4],
                        buffers[5],
                        &g.validation,
                        &uniform,
                        &g.step_config,
                    ],
                ),
            })
        };
        let rows = g.input_layout().len() / g.input_layout().shape().last().unwrap();
        let base = Params {
            rows: rows as u32,
            cols: 0,
            len: 0,
            stage: (g.nodes.len() + 1) as u32,
            stages: (g.nodes.len() + 2) as u32,
            gelu: 0,
            groups_x: 1,
            partials: u32::try_from(partial_count).map_err(|_| TrainingError::Overflow)?,
        };
        let mut passes = Vec::new();
        let mut offset = 0;
        for (id, p) in g.definition.parameters().iter().enumerate() {
            let params = Params {
                len: p.values.len() as u32,
                cols: offset,
                gelu: u32::from(p.role == ParameterRole::Gain),
                partials: counts[id] as u32,
                ..base
            };
            passes.push(element(
                &reduce_partials,
                params,
                [
                    &g.raw_gradients[id],
                    &unused_read,
                    &unused_read,
                    &unused_read,
                    &partials,
                    &unused_aux,
                ],
                false,
            )?);
            offset += counts[id] as u32;
        }
        passes.push(element(
            &reduce,
            base,
            [
                &partials,
                &unused_read,
                &unused_read,
                &unused_read,
                &factors,
                &unused_aux,
            ],
            true,
        )?);
        let mut gradients = Vec::new();
        for (id, p) in g.definition.parameters().iter().enumerate() {
            let params = Params {
                len: p.values.len() as u32,
                gelu: u32::from(p.role == ParameterRole::Gain),
                stage: g.definition.parameter_owners()[id] as u32,
                ..base
            };
            gradients.push(element(
                &gradient,
                params,
                [
                    &g.parameters[id],
                    &factors,
                    &g.raw_gradients[id],
                    &unused_read,
                    &g.candidates[id],
                    &g.effective_gradients[id],
                ],
                false,
            )?);
            passes.push(element(
                &prepare,
                params,
                [
                    &g.parameters[id],
                    &factors,
                    &g.raw_gradients[id],
                    &unused_read,
                    &g.candidates[id],
                    &g.effective_gradients[id],
                ],
                false,
            )?);
        }
        Ok(Self { passes, gradients })
    }

    pub(super) fn encode(&self, g: &ResidentGraphTraining, encoder: &mut wgpu::CommandEncoder) {
        g.encode_passes(encoder, &self.passes, &mut Default::default());
    }

    pub(super) fn encode_gradients(
        &self,
        g: &ResidentGraphTraining,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let prefix = self.passes.len() - self.gradients.len();
        g.encode_passes(encoder, &self.passes[..prefix], &mut Default::default());
        g.encode_passes(encoder, &self.gradients, &mut Default::default());
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
