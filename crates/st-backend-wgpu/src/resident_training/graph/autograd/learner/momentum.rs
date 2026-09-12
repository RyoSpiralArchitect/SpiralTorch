//! Fixed-storage Topos EMA, with history and weights committed together.
use super::*;
use crate::resident_tensor::capture::PreparedCapture;
use st_kernel_contracts::momentum::EMA_MOMENTUM_WGSL;

pub(super) struct Momentum {
    values: Vec<wgpu::Buffer>,
    next: Vec<wgpu::Buffer>,
    capture: Option<PreparedCapture>,
    prepare: Vec<Pass>,
    clipped_prepare: Option<Vec<Pass>>,
    commit: Vec<Pass>,
    config: wgpu::Buffer,
    layout: wgpu::BindGroupLayout,
    prepare_pipeline: Shared<wgpu::ComputePipeline>,
}

fn parameter_params(g: &ResidentGraphTraining, id: usize) -> Params {
    let p = &g.definition.parameters()[id];
    Params {
        rows: (g.input_layout().len() / g.input_layout().shape().last().unwrap()) as u32,
        cols: 0,
        len: p.values.len() as u32,
        stage: g.definition.parameter_owners()[id] as u32,
        stages: (g.nodes.len() + 2) as u32,
        gelu: u32::from(p.role == ParameterRole::Gain),
        groups_x: 1,
        partials: 0,
    }
}

fn element(
    g: &ResidentGraphTraining,
    layout: &wgpu::BindGroupLayout,
    config: &wgpu::Buffer,
    pipeline: &Shared<wgpu::ComputePipeline>,
    mut p: Params,
    buffers: [&wgpu::Buffer; 6],
) -> Result<Pass, TrainingError> {
    let device = g.device.runtime().context().device();
    let grid = groups(p.len as usize, &device.limits())?;
    p.groups_x = grid[0];
    let uniform = runtime::upload_slice(
        device,
        "learner.momentum.params",
        &[p],
        wgpu::BufferUsages::UNIFORM,
    )?;
    Ok(Pass {
        pipeline: pipeline.clone(),
        groups: grid,
        binding: binding(
            device,
            layout,
            &[
                buffers[0],
                buffers[1],
                buffers[2],
                buffers[3],
                buffers[4],
                buffers[5],
                &g.validation,
                &uniform,
                config,
            ],
        ),
    })
}

impl Momentum {
    pub(super) fn new(g: &ResidentGraphTraining) -> Result<Self, TrainingError> {
        let device = g.device.runtime().context().device();
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let values = g
            .definition
            .parameters()
            .iter()
            .map(|p| {
                runtime::empty_buffer::<f32>(
                    device,
                    "learner.momentum.state",
                    p.values.len(),
                    usage,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let next = g
            .definition
            .parameters()
            .iter()
            .map(|p| {
                runtime::empty_buffer::<f32>(
                    device,
                    "learner.momentum.candidate",
                    p.values.len(),
                    usage,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let zero_flags =
            runtime::empty_buffer::<u32>(device, "learner.momentum.snapshot_guard", 1, usage)?;
        let layouts = g
            .definition
            .parameters()
            .iter()
            .map(|p| NdLayout::contiguous(&p.shape).map_err(TensorError::from))
            .collect::<Result<Vec<_>, _>>()?;
        let sources = layouts.iter().zip(&values).collect::<Vec<_>>();
        let capture = if sources.is_empty() {
            None
        } else {
            Some(PreparedCapture::new(&g.device, &sources, &zero_flags)?)
        };
        let config = runtime::empty_buffer::<f32>(
            device,
            "learner.momentum.config",
            4,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        )?;
        let unused_read =
            runtime::empty_buffer::<f32>(device, "learner.momentum.unused_read", 1, usage)?;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("learner.momentum.layout"),
            entries: &(0..9)
                .map(|i| composition::entry(i, i >= 7, i < 4))
                .collect::<Vec<_>>(),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("learner.momentum.pipeline_layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("learner.momentum.shader"),
            source: wgpu::ShaderSource::Wgsl(
                (training_scalar_source()
                    + include_str!("../../../../shaders/optimizer_gradient.wgsl")
                    + EMA_MOMENTUM_WGSL
                    + include_str!("../../../../shaders/graph_momentum.wgsl"))
                .into(),
            ),
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
        let prepare_pipeline = pipeline("prepare_momentum");
        let commit_pipeline = pipeline("commit_momentum");
        let mut prepare = Vec::new();
        let mut commit = Vec::new();
        for id in 0..g.parameters.len() {
            let params = parameter_params(g, id);
            prepare.push(element(
                g,
                &layout,
                &config,
                &prepare_pipeline,
                params,
                [
                    &g.parameters[id],
                    &unused_read,
                    &g.raw_gradients[id],
                    &values[id],
                    &g.candidates[id],
                    &next[id],
                ],
            )?);
            commit.push(element(
                g,
                &layout,
                &config,
                &commit_pipeline,
                params,
                [
                    &g.candidates[id],
                    &next[id],
                    &unused_read,
                    &unused_read,
                    &g.parameters[id],
                    &values[id],
                ],
            )?);
        }
        Ok(Self {
            values,
            next,
            capture,
            prepare,
            clipped_prepare: None,
            commit,
            config,
            layout,
            prepare_pipeline,
        })
    }

    // Build both configurations at their first use, never per update. The
    // global norm's factors are consumed directly by candidate preparation.
    pub(super) fn prepare_clipped(
        &mut self,
        g: &ResidentGraphTraining,
        clip: &Clipping,
    ) -> Result<(), TrainingError> {
        if self.clipped_prepare.is_none() {
            let passes = (0..g.parameters.len())
                .map(|id| {
                    element(
                        g,
                        &self.layout,
                        &self.config,
                        &self.prepare_pipeline,
                        parameter_params(g, id),
                        [
                            &g.parameters[id],
                            clip.factors(),
                            &g.raw_gradients[id],
                            &self.values[id],
                            &g.candidates[id],
                            &self.next[id],
                        ],
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            self.clipped_prepare = Some(passes);
        }
        Ok(())
    }

    pub(super) fn encode(
        &self,
        g: &ResidentGraphTraining,
        clipping: Option<&Clipping>,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        if let Some(clip) = clipping {
            clip.encode_norm(g, encoder);
            g.encode_passes(
                encoder,
                self.clipped_prepare
                    .as_ref()
                    .expect("prepared clipped momentum"),
                &mut Default::default(),
            );
        } else {
            g.encode_passes(encoder, &self.prepare, &mut Default::default());
        }
        let decision = g.parameters.len();
        g.encode_passes(
            encoder,
            &g.update_passes[decision..decision + 1],
            &mut Default::default(),
        );
        g.encode_passes(encoder, &self.commit, &mut Default::default());
    }

    pub(super) fn write_config(
        &self,
        g: &ResidentGraphTraining,
        rate: f32,
        damping: f32,
        clipping: bool,
    ) {
        let policy = if g.policy == GraphGradientPolicy::ModuleCompatible {
            1.
        } else {
            0.
        };
        g.device.runtime().context().queue().write_buffer(
            &self.config,
            0,
            bytemuck::cast_slice(&[rate, policy, damping, if clipping { 1. } else { 0. }]),
        );
    }

    pub(super) fn reset(&self, g: &ResidentGraphTraining) {
        let context = g.device.runtime().context();
        let mut encoder = context.device().create_command_encoder(&Default::default());
        for value in &self.values {
            encoder.clear_buffer(value, 0, None);
        }
        context.queue().submit(Some(encoder.finish()));
    }

    pub(super) fn snapshot(
        &self,
        g: &ResidentGraphTraining,
    ) -> Result<Vec<ResidentTensor>, TrainingError> {
        let Some(capture) = &self.capture else {
            return Ok(Vec::new());
        };
        let context = g.device.runtime().context();
        let mut encoder = context.device().create_command_encoder(&Default::default());
        let outputs = capture.encode(&mut encoder)?;
        context.queue().submit(Some(encoder.finish()));
        Ok(outputs)
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
