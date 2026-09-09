//! One graph-owned transaction for dense and broadcast pointwise parameters.
//! Uses the same GEMM and pointwise VJP kernels as the specialized public plans.
use super::*;
use crate::resident_tensor::{
    pointwise::{
        vjp::{PointwiseVjpPlan, VjpWorkspace},
        PointwisePlan,
    },
    storage_limit,
};
pub use st_kernel_contracts::graph::{
    GraphDefinition, GraphGradientPolicy, GraphParameter, GraphStage, ParameterRole,
};

mod snapshot;
pub use snapshot::{GraphParameterReadback, GraphState, GraphStateReadback};

enum Node {
    Linear {
        forward: Pass,
        backward: Vec<Pass>,
    },
    Pointwise {
        plan: Box<PointwiseVjpPlan>,
        workspace: Box<VjpWorkspace>,
        forward: wgpu::BindGroup,
        parameters: Vec<usize>,
    },
}

/// A mutable execution of a validated graph, with immutable terminal snapshots.
/// All gradients are computed before any weight, bias or gain can be committed.
pub struct ResidentGraphTraining {
    definition: GraphDefinition,
    policy: GraphGradientPolicy,
    activations: Vec<wgpu::Buffer>,
    gradients: Vec<wgpu::Buffer>,
    parameters: Vec<wgpu::Buffer>,
    raw_gradients: Vec<wgpu::Buffer>,
    effective_gradients: Vec<wgpu::Buffer>,
    nodes: Vec<Node>,
    loss_passes: Vec<Pass>,
    update_passes: Vec<Pass>,
    target: wgpu::Buffer,
    validation: wgpu::Buffer,
    pointwise_flags: wgpu::Buffer,
    loss: wgpu::Buffer,
    step_config: wgpu::Buffer,
    loss_pool: runtime::ReadbackPool,
    batch_generation: u64,
    submitted_steps: u64,
    last_step: Option<u64>,
    batch_sources: Option<[ResidentTensor; 2]>,
    device: TensorDevice,
}

fn scalar_source() -> String {
    [
        training_scalar_source(),
        include_str!("../shaders/graph_parameter.wgsl").to_owned(),
    ]
    .concat()
}

fn forward_dispatches_per_pass(backend: wgpu::Backend, count: usize) -> usize {
    // Browser immediate timings were flat and deferred reads sometimes regressed.
    // Keep its old schedule; native Metal benefits in both observation cadences.
    if backend == wgpu::Backend::Metal {
        count.max(1)
    } else {
        1
    }
}

impl ResidentGraphTraining {
    pub fn new(
        runtime: WgpuRuntime,
        definition: GraphDefinition,
        policy: GraphGradientPolicy,
        tile: MatmulTile,
        kernel: MatmulKernel,
        accumulation: MatmulAccumulation,
    ) -> Result<Self, TrainingError> {
        let context = runtime.context();
        let gpu = context.device();
        let limits = gpu.limits();
        let device = TensorDevice::new(runtime.clone())?;
        let count = definition.stages().len();
        // Per-node flags, loss/target, input, pointwise, then one global decision.
        let stages = (count + 2) as u32;
        let rows =
            definition.input_layout().len() / definition.input_layout().shape().last().unwrap();
        if limits.max_storage_buffers_per_shader_stage < 8
            || limits.max_bindings_per_bind_group < 9
            || limits.max_uniform_buffers_per_shader_stage < 2
            || limits.max_compute_workgroup_storage_size < 1024
        {
            return Err(MatmulError::DeviceLimit("graph training bindings/workgroup").into());
        }
        for layout in definition.layouts() {
            storage_limit(layout.len(), &limits)?;
            groups(layout.len(), &limits)?;
        }
        for p in definition.parameters() {
            storage_limit(p.values.len(), &limits)?;
            groups(p.values.len(), &limits)?;
        }
        storage_limit(count + 4, &limits)?;
        for node in definition.stages() {
            if let GraphStage::Linear { weight, .. } = node {
                let shape = &definition.parameters()[*weight].shape;
                let (k, n) = (shape[0], shape[1]);
                for (r, k, n) in [(rows, k, n), (k, rows, n), (rows, n, k)] {
                    MatmulShape::new(r, k, n)?.validate(&limits, tile, kernel)?;
                }
            }
        }
        let storage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let empty = |label, len| runtime::empty_buffer::<f32>(gpu, label, len, storage);
        let buffers = |label| {
            definition
                .layouts()
                .iter()
                .map(|l| empty(label, l.len()))
                .collect::<Result<Vec<_>, _>>()
        };
        let activations = buffers("graph.activation")?;
        let gradients = buffers("graph.activation_gradient")?;
        let parameters = definition
            .parameters()
            .iter()
            .map(|p| runtime::upload_slice(gpu, "graph.parameter", &p.values, storage))
            .collect::<Result<Vec<_>, _>>()?;
        let parameter_buffers = |label| {
            definition
                .parameters()
                .iter()
                .map(|p| empty(label, p.values.len()))
                .collect::<Result<Vec<_>, _>>()
        };
        let raw_gradients = parameter_buffers("graph.raw_gradient")?;
        let effective_gradients = parameter_buffers("graph.effective_gradient")?;
        let candidates = parameter_buffers("graph.candidate")?;
        let target = empty("graph.target", definition.output_layout().len())?;
        let loss = empty("graph.loss", 1)?;
        let partial_count = definition.output_layout().len().div_ceil(256);
        let partials = empty("graph.loss_partials", partial_count)?;
        let validation = runtime::empty_buffer::<u32>(gpu, "graph.validation", count + 4, storage)?;
        let pointwise_flags =
            runtime::empty_buffer::<u32>(gpu, "graph.pointwise.flags", 1, storage)?;
        let empty_flags = runtime::empty_buffer::<u32>(gpu, "graph.empty.flags", 1, storage)?;
        let step_config = runtime::upload_slice(
            gpu,
            "graph.step",
            &[
                0f32,
                if policy == GraphGradientPolicy::ModuleCompatible {
                    1.
                } else {
                    0.
                },
                0.,
                0.,
            ],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        )?;
        let unused_read = empty("graph.unused_read", 1)?;
        let unused_out = empty("graph.unused_out", 1)?;
        let unused_aux = empty("graph.unused_aux", 1)?;
        let matrix_layout = resident_dense::dense_layout(gpu, true);
        let matrix = |kind| -> Result<MatrixPipeline, TrainingError> {
            Ok(MatrixPipeline {
                pipeline: Shared::new(resident_dense::dense_pipeline(
                    gpu,
                    &matrix_layout,
                    training_dense_stage_source(tile.dimensions(), kernel, accumulation, kind)
                        .map_err(TrainingError::Shader)?,
                )),
                kind,
            })
        };
        let mut forward = [None, None];
        let mut dw = None;
        let mut dx = None;
        for node in definition.stages() {
            if let GraphStage::Linear { gelu, .. } = node {
                if forward[usize::from(*gelu)].is_none() {
                    forward[usize::from(*gelu)] =
                        Some(matrix(TrainingMatmulKind::Forward { gelu: *gelu })?);
                }
                if dw.is_none() {
                    dw = Some(matrix(TrainingMatmulKind::WeightGradient)?);
                    dx = Some(matrix(TrainingMatmulKind::InputGradient)?);
                }
            }
        }
        let entries: Vec<_> = (0..9)
            .map(|i| wgpu::BindGroupLayoutEntry {
                binding: i,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if i >= 7 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage { read_only: i < 4 }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let element_layout = gpu.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("graph.element.layout"),
            entries: &entries,
        });
        let pipeline_layout = gpu.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("graph.element.pipeline_layout"),
            bind_group_layouts: &[&element_layout],
            push_constant_ranges: &[],
        });
        let module = gpu.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("graph.element.shader"),
            source: wgpu::ShaderSource::Wgsl(scalar_source().into()),
        });
        let pipeline = |entry| {
            Shared::new(
                gpu.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: entry,
                    compilation_options: Default::default(),
                }),
            )
        };
        let delta_pipeline = pipeline("delta");
        let bias_pipeline = pipeline("bias_gradient");
        let prepare = pipeline("prepare_parameter");
        let commit = pipeline("commit_sgd");
        let element = |pipeline: &Shared<wgpu::ComputePipeline>,
                       mut p: Params,
                       ops: [&wgpu::Buffer; 6],
                       single: bool|
         -> Result<Pass, TrainingError> {
            let grid = if single {
                [1, 1, 1]
            } else {
                groups(p.len as usize, &limits)?
            };
            p.groups_x = grid[0];
            let uniform = runtime::upload_slice(
                gpu,
                "graph.element.params",
                &[p],
                wgpu::BufferUsages::UNIFORM,
            )?;
            Ok(Pass {
                pipeline: pipeline.clone(),
                binding: binding(
                    gpu,
                    &element_layout,
                    &[
                        ops[0],
                        ops[1],
                        ops[2],
                        ops[3],
                        ops[4],
                        ops[5],
                        &validation,
                        &uniform,
                        &step_config,
                    ],
                ),
                groups: grid,
            })
        };
        let params = |stage: usize, cols: usize, len: usize, gelu: bool| Params {
            rows: rows as u32,
            cols: cols as u32,
            len: len as u32,
            stage: stage as u32,
            stages,
            gelu: u32::from(gelu),
            groups_x: 0,
            partials: partial_count as u32,
        };
        let mut nodes = Vec::new();
        for (i, node) in definition.stages().iter().enumerate() {
            nodes.push(match node {
                GraphStage::Linear { weight, bias, gelu } => {
                    let (w, b) = (*weight, *bias);
                    let (k, n) = (
                        definition.parameters()[w].shape[0],
                        definition.parameters()[w].shape[1],
                    );
                    let preactivation = empty("graph.preactivation", rows * n)?;
                    let delta = empty("graph.delta", rows * n)?;
                    let forward = matrix_pass(
                        gpu,
                        &matrix_layout,
                        forward[usize::from(*gelu)].as_ref().unwrap(),
                        MatrixPass {
                            shape: MatmulShape::new(rows, k, n)?,
                            tile,
                            stage: i as u32,
                            operands: [
                                &activations[i],
                                &parameters[w],
                                &activations[i + 1],
                                &parameters[b],
                                &unused_read,
                                &unused_read,
                            ],
                            validation: &validation,
                            tape: &preactivation,
                        },
                    )?;
                    let backward = vec![
                        element(
                            &delta_pipeline,
                            params(i, n, rows * n, *gelu),
                            [
                                &preactivation,
                                &gradients[i + 1],
                                &unused_read,
                                &unused_read,
                                &delta,
                                &unused_aux,
                            ],
                            false,
                        )?,
                        matrix_pass(
                            gpu,
                            &matrix_layout,
                            dw.as_ref().unwrap(),
                            MatrixPass {
                                shape: MatmulShape::new(k, rows, n)?,
                                tile,
                                stage: i as u32,
                                operands: [
                                    &activations[i],
                                    &delta,
                                    &raw_gradients[w],
                                    &unused_read,
                                    &unused_read,
                                    &unused_read,
                                ],
                                validation: &validation,
                                tape: &unused_out,
                            },
                        )?,
                        element(
                            &bias_pipeline,
                            params(i, n, n, false),
                            [
                                &delta,
                                &unused_read,
                                &unused_read,
                                &unused_read,
                                &raw_gradients[b],
                                &unused_aux,
                            ],
                            false,
                        )?,
                        matrix_pass(
                            gpu,
                            &matrix_layout,
                            dx.as_ref().unwrap(),
                            MatrixPass {
                                shape: MatmulShape::new(rows, n, k)?,
                                tile,
                                stage: i as u32,
                                operands: [
                                    &delta,
                                    &parameters[w],
                                    &gradients[i],
                                    &unused_read,
                                    &unused_read,
                                    &unused_read,
                                ],
                                validation: &validation,
                                tape: &unused_out,
                            },
                        )?,
                    ];
                    Node::Linear { forward, backward }
                }
                GraphStage::Pointwise {
                    chain,
                    parameters: ids,
                } => {
                    let mut layouts = vec![definition.layouts()[i].clone()];
                    for &id in ids {
                        layouts.push(
                            NdLayout::contiguous(&definition.parameters()[id].shape)
                                .map_err(TensorError::from)?,
                        );
                    }
                    let plan = Box::new(PointwiseVjpPlan::new(PointwisePlan::new(
                        device.clone(),
                        chain.clone(),
                        layouts,
                    )?)?);
                    let inputs: Vec<_> = std::iter::once(&activations[i])
                        .chain(ids.iter().map(|&id| &parameters[id]))
                        .collect();
                    let destinations: Vec<_> = std::iter::once(&gradients[i])
                        .chain(ids.iter().map(|&id| &raw_gradients[id]))
                        .collect();
                    let forward = plan.forward().bind_into(
                        &inputs,
                        &activations[i + 1],
                        &empty_flags,
                        &pointwise_flags,
                    );
                    let workspace = Box::new(plan.prepare_into(
                        &inputs,
                        &gradients[i + 1],
                        &destinations,
                        &empty_flags,
                        &pointwise_flags,
                    )?);
                    Node::Pointwise {
                        plan,
                        workspace,
                        forward,
                        parameters: ids.clone(),
                    }
                }
            });
        }
        let loss_params = params(count, 0, definition.output_layout().len(), false);
        let loss_passes = vec![
            element(
                &pipeline("mse_partials"),
                loss_params,
                [
                    activations.last().unwrap(),
                    &target,
                    &unused_read,
                    &unused_read,
                    gradients.last().unwrap(),
                    &partials,
                ],
                false,
            )?,
            element(
                &pipeline("mse_reduce"),
                loss_params,
                [
                    &partials,
                    &unused_read,
                    &unused_read,
                    &unused_read,
                    &unused_out,
                    &loss,
                ],
                true,
            )?,
        ];
        let mut update_passes = Vec::new();
        for (id, p) in definition.parameters().iter().enumerate() {
            update_passes.push(element(
                &prepare,
                params(
                    definition.parameter_owners()[id],
                    0,
                    p.values.len(),
                    p.role == ParameterRole::Gain,
                ),
                [
                    &parameters[id],
                    &unused_read,
                    &raw_gradients[id],
                    &unused_read,
                    &candidates[id],
                    &effective_gradients[id],
                ],
                false,
            )?);
        }
        update_passes.push(element(
            &pipeline("decide_sgd"),
            loss_params,
            [
                &unused_read,
                &unused_read,
                &unused_read,
                &unused_read,
                &unused_out,
                &unused_aux,
            ],
            true,
        )?);
        for (id, p) in definition.parameters().iter().enumerate() {
            update_passes.push(element(
                &commit,
                params(definition.parameter_owners()[id], 0, p.values.len(), false),
                [
                    &candidates[id],
                    &unused_read,
                    &unused_read,
                    &unused_read,
                    &parameters[id],
                    &unused_aux,
                ],
                false,
            )?);
        }
        let loss_pool = runtime::ReadbackPool::new::<u32>(context.clone(), count + 5)?;
        Ok(Self {
            definition,
            policy,
            activations,
            gradients,
            parameters,
            raw_gradients,
            effective_gradients,
            nodes,
            loss_passes,
            update_passes,
            target,
            validation,
            pointwise_flags,
            loss,
            step_config,
            loss_pool,
            batch_generation: 0,
            submitted_steps: 0,
            last_step: None,
            batch_sources: None,
            device,
        })
    }

    pub fn input_layout(&self) -> &NdLayout {
        self.definition.input_layout()
    }
    pub fn output_layout(&self) -> &NdLayout {
        self.definition.output_layout()
    }
    pub fn gradient_policy(&self) -> GraphGradientPolicy {
        self.policy
    }

    pub fn stage_count(&self) -> usize {
        self.definition.stages().len()
    }

    pub fn parameter_count(&self) -> usize {
        self.definition.parameters().len()
    }
    pub fn submitted_steps(&self) -> u64 {
        self.submitted_steps
    }
    pub fn batch_generation(&self) -> u64 {
        self.batch_generation
    }
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        self.device.runtime().adapter_info()
    }

    pub fn upload_batch(&mut self, input: &[f32], target: &[f32]) -> Result<(), TrainingError> {
        for (label, values, expected) in [
            ("input", input, self.input_layout().len()),
            ("target", target, self.output_layout().len()),
        ] {
            if values.len() != expected {
                return Err(DenseError::InputLength {
                    expected,
                    actual: values.len(),
                }
                .into());
            }
            if values.iter().any(|v| !v.is_finite()) {
                return Err(DenseError::NonFiniteInput(label).into());
            }
        }
        let generation = self
            .batch_generation
            .checked_add(1)
            .ok_or(TrainingError::Overflow)?;
        let queue = self.device.runtime().context().queue();
        queue.write_buffer(&self.activations[0], 0, bytemuck::cast_slice(input));
        queue.write_buffer(&self.target, 0, bytemuck::cast_slice(target));
        self.batch_generation = generation;
        self.last_step = None;
        self.batch_sources = None;
        Ok(())
    }

    pub fn upload_batch_tensors(
        &mut self,
        input: &ResidentTensor,
        target: &ResidentTensor,
    ) -> Result<(), TrainingError> {
        let context = self.device.runtime().context();
        input.require_context(context)?;
        target.require_context(context)?;
        if input.layout().shape() != self.input_layout().shape()
            || target.layout().shape() != self.output_layout().shape()
        {
            return Err(DenseError::InvalidLayout.into());
        }
        let generation = self
            .batch_generation
            .checked_add(1)
            .ok_or(TrainingError::Overflow)?;
        let input = input.contiguous()?;
        let target = target.contiguous()?;
        let mut encoder = context.device().create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(
            input.values(),
            0,
            &self.activations[0],
            0,
            self.input_layout().len() as u64 * 4,
        );
        encoder.copy_buffer_to_buffer(
            target.values(),
            0,
            &self.target,
            0,
            self.output_layout().len() as u64 * 4,
        );
        context.queue().submit(Some(encoder.finish()));
        self.batch_generation = generation;
        self.last_step = None;
        self.batch_sources = Some([input, target]);
        Ok(())
    }

    fn encode_passes(&self, encoder: &mut wgpu::CommandEncoder, passes: &[Pass]) {
        for chunk in passes.chunks(dispatches_per_pass(
            self.adapter_info().backend,
            passes.len(),
        )) {
            let mut compute = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("graph.training"),
                timestamp_writes: None,
            });
            for pass in chunk {
                pass.encode(&mut compute);
            }
        }
    }

    /// One submission, no readback: forward -> MSE -> all VJPs -> prepare/vote/commit.
    /// An enqueued attempt is not an accepted step until a guarded snapshot is read.
    pub fn step(&mut self, rate: f32) -> Result<u64, TrainingError> {
        if !rate.is_finite() || rate < 0. {
            return Err(TrainingError::LearningRate);
        }
        if self.batch_generation == 0 {
            return Err(TrainingError::MissingBatch);
        }
        let attempt = self
            .submitted_steps
            .checked_add(1)
            .ok_or(TrainingError::Overflow)?;
        let context = self.device.runtime().context();
        context
            .queue()
            .write_buffer(&self.step_config, 0, bytemuck::bytes_of(&rate));
        let mut encoder = context.device().create_command_encoder(&Default::default());
        encoder.clear_buffer(&self.validation, 0, None);
        encoder.clear_buffer(&self.pointwise_flags, 0, None);
        let n = self.nodes.len();
        if let Some([input, target]) = &self.batch_sources {
            encoder.copy_buffer_to_buffer(
                input.flags(),
                0,
                &self.validation,
                (n + 1) as u64 * 4,
                4,
            );
            encoder.copy_buffer_to_buffer(target.flags(), 0, &self.validation, n as u64 * 4, 4);
        }
        // Forward has no intervening copies. Keep the loss/VJP/decision phases
        // separate: their copies and prepare/vote/commit ordering are unchanged.
        for nodes in self
            .nodes
            .chunks(forward_dispatches_per_pass(self.adapter_info().backend, n))
        {
            let mut compute = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("graph.training.forward"),
                timestamp_writes: None,
            });
            for node in nodes {
                match node {
                    Node::Linear { forward, .. } => forward.encode(&mut compute),
                    Node::Pointwise { plan, forward, .. } => {
                        plan.forward().encode_in_pass(&mut compute, forward);
                    }
                }
            }
        }
        self.encode_passes(&mut encoder, &self.loss_passes);
        for (i, node) in self.nodes.iter().enumerate().rev() {
            match node {
                Node::Linear { backward, .. } => self.encode_passes(&mut encoder, backward),
                Node::Pointwise {
                    plan,
                    parameters,
                    workspace,
                    ..
                } => {
                    let gradients = std::iter::once(&self.gradients[i])
                        .chain(parameters.iter().map(|&id| &self.raw_gradients[id]));
                    plan.encode_prepared(&mut encoder, workspace, gradients);
                }
            }
        }
        encoder.copy_buffer_to_buffer(
            &self.pointwise_flags,
            0,
            &self.validation,
            (n + 2) as u64 * 4,
            4,
        );
        self.encode_passes(&mut encoder, &self.update_passes);
        context.queue().submit(Some(encoder.finish()));
        self.submitted_steps = attempt;
        self.last_step = Some(attempt);
        Ok(attempt)
    }

    pub fn loss_snapshot(&self) -> Result<StepReadback, TrainingError> {
        let step = self.last_step.ok_or(TrainingError::StaleStep)?;
        Ok(StepReadback {
            raw: readback::capture(
                self.device.runtime().context(),
                &self.loss_pool,
                &[&self.loss, &self.validation],
            )?,
            stages: self.nodes.len() + 2,
            step,
            batch_generation: self.batch_generation,
        })
    }
    pub fn state_snapshot(&self) -> Result<GraphStateReadback, TrainingError> {
        let step = self.last_step.ok_or(TrainingError::StaleStep)?;
        let mut buffers = vec![
            &self.loss,
            &self.validation,
            self.activations.last().unwrap(),
            &self.gradients[0],
        ];
        for i in 0..self.parameters.len() {
            buffers.extend([
                &self.parameters[i],
                &self.raw_gradients[i],
                &self.effective_gradients[i],
            ]);
        }
        Ok(GraphStateReadback {
            raw: readback::capture_new(self.device.runtime().context(), &buffers)?,
            definition: self.definition.clone(),
            policy: self.policy,
            step,
            batch_generation: self.batch_generation,
        })
    }
    pub fn parameter_snapshot(&self) -> Result<GraphParameterReadback, TrainingError> {
        let raw = if self.parameters.is_empty() {
            None
        } else {
            Some(readback::capture_new(
                self.device.runtime().context(),
                &self.parameters.iter().collect::<Vec<_>>(),
            )?)
        };
        Ok(GraphParameterReadback {
            raw,
            definition: self.definition.clone(),
        })
    }
    pub fn prediction_tensor(&self) -> Result<ResidentTensor, TrainingError> {
        self.last_step.ok_or(TrainingError::StaleStep)?;
        Ok(self.device.capture(
            self.output_layout(),
            self.activations.last().unwrap(),
            &self.validation,
        )?)
    }
    pub fn input_gradient_tensor(&self) -> Result<ResidentTensor, TrainingError> {
        self.last_step.ok_or(TrainingError::StaleStep)?;
        Ok(self
            .device
            .capture(self.input_layout(), &self.gradients[0], &self.validation)?)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn forward_pass_batching_preserves_browser_and_unmeasured_backends() {
        use super::forward_dispatches_per_pass;
        for count in [0, 1, 24, 4096] {
            assert_eq!(
                forward_dispatches_per_pass(wgpu::Backend::Metal, count),
                count.max(1)
            );
            for backend in [
                wgpu::Backend::BrowserWebGpu,
                wgpu::Backend::Empty,
                wgpu::Backend::Vulkan,
                wgpu::Backend::Dx12,
                wgpu::Backend::Gl,
            ] {
                assert_eq!(forward_dispatches_per_pass(backend, count), 1);
            }
        }
    }

    #[test]
    fn graph_parameter_shader_validates() {
        let module = naga::front::wgsl::parse_str(&super::scalar_source()).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }
}
