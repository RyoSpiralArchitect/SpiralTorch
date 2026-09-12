//! Resident Linear/GELU + mean-MSE VJP and transactional plain SGD.
//! A step enqueues work; only its owned readback proves numerical acceptance.

use crate::{
    resident_dense::{self, DenseActivation, DenseError, DenseLayer, Uniforms},
    resident_matmul::{MatmulAccumulation, MatmulError, MatmulKernel, MatmulShape, MatmulTile},
    resident_tensor::{ResidentTensor, TensorDevice, TensorError},
    runtime::{self, Shared, WgpuContext, WgpuRuntime, WgpuRuntimeError},
    shader_sources::{training_dense_stage_source, TrainingMatmulKind},
};
use bytemuck::{Pod, Zeroable};
use st_kernel_contracts::layout::NdLayout;
use thiserror::Error;

pub mod graph;
mod readback;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod stage_tests;
pub use readback::{
    LayerGradient, ParameterReadback, StepReadback, TrainingState, TrainingStateReadback,
};

#[derive(Debug, Error)]
pub enum TrainingError {
    #[error(transparent)]
    GradientClip(#[from] st_kernel_contracts::gradient_clip::GradientClipError),
    #[error(transparent)]
    Graph(#[from] st_kernel_contracts::graph::GraphError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Dense(#[from] DenseError),
    #[error(transparent)]
    Matmul(#[from] MatmulError),
    #[error(transparent)]
    Runtime(#[from] WgpuRuntimeError),
    #[error("learning rate must be finite and nonnegative")]
    LearningRate,
    #[error("upload a complete input/target batch before stepping")]
    MissingBatch,
    #[error("upload an input before forwarding the autograd graph")]
    MissingInput,
    #[error("step the current batch before requesting training results")]
    StaleStep,
    #[error("forward the current input before backward; the token must belong to this workspace and its latest forward")]
    StaleForward,
    #[error("gradient terms must contain 1..=256 same-forward contributions with finite weights")]
    GradientTerms,
    #[error("gradient weight must be finite")]
    GradientWeight,
    #[error("accumulator and gradients must belong to this learner's current parameter state")]
    AccumulatorState,
    #[error("accumulate a gradient before observing or updating from the accumulator")]
    EmptyAccumulator,
    #[error("submit an SGD update before requesting its receipt")]
    MissingUpdate,
    #[error("read the pending profile before reusing the private profiler; an abandoned or invalid profile requires a new workspace")]
    PendingProfile,
    #[error("training counter or snapshot size exhausted")]
    Overflow,
    #[error(
        "non-finite training value at stage {stage}, mask {flags:#x}; no parameters were committed"
    )]
    Rejected { stage: usize, flags: u32 },
    #[error("invalid or non-finite training readback")]
    InvalidReadback,
    #[error("training shader specialization failed: {0}")]
    Shader(&'static str),
}

#[derive(Clone, Copy)]
struct Spec {
    inner: usize,
    cols: usize,
    activation: DenseActivation,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct Params {
    rows: u32,
    cols: u32,
    len: u32,
    stage: u32,
    stages: u32,
    gelu: u32,
    groups_x: u32,
    partials: u32,
}

impl Params {
    pub(crate) fn standalone_mse(len: u32, groups_x: u32, partials: u32) -> Self {
        Self {
            rows: 0,
            cols: 0,
            len,
            stage: 0,
            stages: 0,
            gelu: 0,
            groups_x,
            partials,
        }
    }
}

struct Pass {
    pipeline: Shared<wgpu::ComputePipeline>,
    binding: wgpu::BindGroup,
    groups: [u32; 3],
}

impl Pass {
    fn encode<'a>(&'a self, pass: &mut wgpu::ComputePass<'a>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.binding, &[]);
        pass.dispatch_workgroups(self.groups[0], self.groups[1], self.groups[2]);
    }
}

fn dispatches_per_pass(backend: wgpu::Backend, dispatches: usize) -> usize {
    // Metal benefits consistently; browser deferred readback can regress.
    // Preserve the existing path on BrowserWebGpu and unmeasured backends.
    if backend == wgpu::Backend::Metal {
        dispatches.max(1)
    } else {
        1
    }
}

fn binding(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buffers: &[&wgpu::Buffer],
) -> wgpu::BindGroup {
    let entries: Vec<_> = buffers
        .iter()
        .enumerate()
        .map(|(i, buffer)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: buffer.as_entire_binding(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("dense.training.binding"),
        layout,
        entries: &entries,
    })
}

fn groups(len: usize, limits: &wgpu::Limits) -> Result<[u32; 3], TrainingError> {
    let count = u32::try_from(len)
        .map_err(|_| TrainingError::Overflow)?
        .div_ceil(256);
    let x = count.min(limits.max_compute_workgroups_per_dimension);
    if x == 0 {
        return Err(TrainingError::Overflow);
    }
    let y = count.div_ceil(x);
    if y > limits.max_compute_workgroups_per_dimension {
        return Err(MatmulError::DeviceLimit("training element dispatch").into());
    }
    Ok([x, y, 1])
}

struct MatrixPipeline {
    pipeline: Shared<wgpu::ComputePipeline>,
    kind: TrainingMatmulKind,
}

struct MatrixPass<'a> {
    shape: MatmulShape,
    tile: MatmulTile,
    stage: u32,
    operands: [&'a wgpu::Buffer; 6],
    validation: &'a wgpu::Buffer,
    tape: &'a wgpu::Buffer,
}

fn matrix_pass(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    pipeline: &MatrixPipeline,
    descriptor: MatrixPass<'_>,
) -> Result<Pass, TrainingError> {
    let MatrixPass {
        shape,
        tile,
        stage,
        operands,
        validation,
        tape,
    } = descriptor;
    let (rows, inner, cols) = shape.dimensions();
    let params = Uniforms {
        rows: rows as u32,
        cols: cols as u32,
        inner: inner as u32,
        flags: pipeline.kind.flags(),
        output_scale: 1.,
        validation_index: stage,
        padding: [pipeline.kind.validation_mask(), 0],
    };
    let uniform = runtime::upload_slice(
        device,
        "training.gemm.params",
        &[params],
        wgpu::BufferUsages::UNIFORM,
    )?;
    let buffers = [
        operands[0],
        operands[1],
        operands[2],
        operands[3],
        operands[4],
        operands[5],
        &uniform,
        validation,
        tape,
    ];
    let [tm, tn, _] = tile.dimensions();
    Ok(Pass {
        pipeline: pipeline.pipeline.clone(),
        binding: binding(device, layout, &buffers),
        groups: [(cols as u32).div_ceil(tn), (rows as u32).div_ceil(tm), 1],
    })
}

/// Mutable GPU copy of an existing plan, never a second host model definition.
pub struct ResidentDenseTraining {
    activations: Vec<wgpu::Buffer>,
    gradients: Vec<wgpu::Buffer>,
    weights: Vec<wgpu::Buffer>,
    biases: Vec<wgpu::Buffer>,
    weight_gradients: Vec<wgpu::Buffer>,
    bias_gradients: Vec<wgpu::Buffer>,
    target: wgpu::Buffer,
    validation: wgpu::Buffer,
    loss: wgpu::Buffer,
    step_config: wgpu::Buffer,
    passes: Vec<Pass>,
    loss_pool: runtime::ReadbackPool,
    input: NdLayout,
    output: NdLayout,
    specs: Vec<Spec>,
    batch_generation: u64,
    submitted_steps: u64,
    last_step: Option<u64>,
    batch_sources: Option<[ResidentTensor; 2]>,
    runtime: WgpuRuntime,
}

pub(crate) fn training_scalar_source() -> String {
    [
        include_str!("shaders/gelu_derivative.wgsl"),
        include_str!("shaders/dense_training.wgsl"),
    ]
    .concat()
}

impl ResidentDenseTraining {
    pub fn new(
        runtime: WgpuRuntime,
        input: NdLayout,
        layers: &[DenseLayer],
        tile: MatmulTile,
        kernel: MatmulKernel,
        accumulation: MatmulAccumulation,
    ) -> Result<Self, TrainingError> {
        let context = runtime.context();
        let device = context.device();
        let limits = device.limits();
        let (shapes, output) =
            resident_dense::validate_layers(&limits, &input, layers, tile, kernel)?;
        let stages = u32::try_from(layers.len().checked_add(2).ok_or(TrainingError::Overflow)?)
            .map_err(|_| TrainingError::Overflow)?
            - 2;
        if limits.max_storage_buffers_per_shader_stage < 8
            || limits.max_bindings_per_bind_group < 9
            || limits.max_uniform_buffers_per_shader_stage < 2
            || limits.max_compute_workgroup_size_x < 256
            || limits.max_compute_invocations_per_workgroup < 256
            || limits.max_compute_workgroup_storage_size < 1024
        {
            return Err(MatmulError::DeviceLimit("resident training bindings/workgroup").into());
        }
        let rows = shapes[0].dimensions().0;
        for shape in &shapes {
            let (r, k, n) = shape.dimensions();
            MatmulShape::new(k, r, n)?.validate(&limits, tile, kernel)?;
            MatmulShape::new(r, n, k)?.validate(&limits, tile, kernel)?;
            groups(r * n, &limits)?;
            groups(k * n, &limits)?;
        }
        let storage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let empty = |label, len| runtime::empty_buffer::<f32>(device, label, len, storage);
        let mut activations = vec![empty("training.input", input.len())?];
        let mut gradients = vec![empty("training.dx", input.len())?];
        let mut preactivations = Vec::new();
        let mut deltas = Vec::new();
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        let mut weight_gradients = Vec::new();
        let mut bias_gradients = Vec::new();
        let mut candidate_weights = Vec::new();
        let mut candidate_biases = Vec::new();
        for layer in layers {
            activations.push(empty("training.activation", rows * layer.cols)?);
            gradients.push(empty("training.input_gradient", rows * layer.cols)?);
            preactivations.push(empty("training.preactivation", rows * layer.cols)?);
            deltas.push(empty("training.delta", rows * layer.cols)?);
            weights.push(runtime::upload_slice(
                device,
                "training.weight",
                &layer.weights,
                storage,
            )?);
            biases.push(runtime::upload_slice(
                device,
                "training.bias",
                &layer.bias,
                storage,
            )?);
            weight_gradients.push(empty("training.dw", layer.weights.len())?);
            bias_gradients.push(empty("training.db", layer.cols)?);
            candidate_weights.push(empty("training.candidate_weight", layer.weights.len())?);
            candidate_biases.push(empty("training.candidate_bias", layer.cols)?);
        }
        let target = empty("training.target", output.len())?;
        let validation =
            runtime::empty_buffer::<u32>(device, "training.validation", layers.len() + 2, storage)?;
        let loss = empty("training.loss", 1)?;
        let partial_count = output.len().div_ceil(256);
        let partials = empty("training.loss_partials", partial_count)?;
        let step_config = runtime::upload_slice(
            device,
            "training.step",
            &[0f32; 4],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        )?;
        let unused_read = empty("training.unused_read", 1)?;
        let unused_out = empty("training.unused_out", 1)?;
        let unused_aux = empty("training.unused_aux", 1)?;
        let matrix_layout = resident_dense::dense_layout(device, true);
        let matrix = |kind| -> Result<MatrixPipeline, TrainingError> {
            let source = training_dense_stage_source(tile.dimensions(), kernel, accumulation, kind)
                .map_err(TrainingError::Shader)?;
            Ok(MatrixPipeline {
                pipeline: Shared::new(resident_dense::dense_pipeline(
                    device,
                    &matrix_layout,
                    source,
                )),
                kind,
            })
        };
        // Compile each required activation once, shared by all matching stages.
        let mut forward = [None, None];
        for layer in layers {
            let gelu = layer.activation == DenseActivation::Gelu;
            let slot = &mut forward[usize::from(gelu)];
            if slot.is_none() {
                *slot = Some(matrix(TrainingMatmulKind::Forward { gelu })?);
            }
        }
        let dw = matrix(TrainingMatmulKind::WeightGradient)?;
        let dx = matrix(TrainingMatmulKind::InputGradient)?;
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
        let element_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("training.element.layout"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("training.element.pipeline_layout"),
            bind_group_layouts: &[&element_layout],
            push_constant_ranges: &[],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("training.element.shader"),
            source: wgpu::ShaderSource::Wgsl(training_scalar_source().into()),
        });
        let element_pipeline = |entry| {
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
        let delta_pipeline = element_pipeline("delta");
        let bias_pipeline = element_pipeline("bias_gradient");
        let prepare = element_pipeline("prepare_sgd");
        let commit = element_pipeline("commit_sgd");
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
                "training.element.params",
                &[p],
                wgpu::BufferUsages::UNIFORM,
            )?;
            let buffers = [
                buffers[0],
                buffers[1],
                buffers[2],
                buffers[3],
                buffers[4],
                buffers[5],
                &validation,
                &uniform,
                &step_config,
            ];
            Ok(Pass {
                pipeline: pipeline.clone(),
                binding: binding(device, &element_layout, &buffers),
                groups: grid,
            })
        };
        let params = |stage: usize, r: usize, n: usize, len: usize| Params {
            rows: r as u32,
            cols: n as u32,
            len: len as u32,
            stage: stage as u32,
            stages,
            gelu: u32::from(
                stage < layers.len() && layers[stage].activation == DenseActivation::Gelu,
            ),
            groups_x: 0,
            partials: partial_count as u32,
        };
        let mut passes = Vec::new();
        for (i, &shape) in shapes.iter().enumerate() {
            passes.push(matrix_pass(
                device,
                &matrix_layout,
                forward[usize::from(layers[i].activation == DenseActivation::Gelu)]
                    .as_ref()
                    .unwrap(),
                MatrixPass {
                    shape,
                    tile,
                    stage: i as u32,
                    operands: [
                        &activations[i],
                        &weights[i],
                        &activations[i + 1],
                        &biases[i],
                        &unused_read,
                        &unused_read,
                    ],
                    validation: &validation,
                    tape: &preactivations[i],
                },
            )?);
        }
        let loss_params = params(
            layers.len(),
            rows,
            *output.shape().last().unwrap(),
            output.len(),
        );
        passes.push(element(
            &element_pipeline("mse_partials"),
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
        )?);
        passes.push(element(
            &element_pipeline("mse_reduce"),
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
        )?);
        for (i, &shape) in shapes.iter().enumerate().rev() {
            let (r, k, n) = shape.dimensions();
            passes.push(element(
                &delta_pipeline,
                params(i, r, n, r * n),
                [
                    &preactivations[i],
                    &gradients[i + 1],
                    &unused_read,
                    &unused_read,
                    &deltas[i],
                    &unused_aux,
                ],
                false,
            )?);
            passes.push(matrix_pass(
                device,
                &matrix_layout,
                &dw,
                MatrixPass {
                    shape: MatmulShape::new(k, r, n)?,
                    tile,
                    stage: i as u32,
                    operands: [
                        &activations[i],
                        &deltas[i],
                        &weight_gradients[i],
                        &unused_read,
                        &unused_read,
                        &unused_read,
                    ],
                    validation: &validation,
                    tape: &unused_out,
                },
            )?);
            passes.push(element(
                &bias_pipeline,
                params(i, r, n, n),
                [
                    &deltas[i],
                    &unused_read,
                    &unused_read,
                    &unused_read,
                    &bias_gradients[i],
                    &unused_aux,
                ],
                false,
            )?);
            passes.push(matrix_pass(
                device,
                &matrix_layout,
                &dx,
                MatrixPass {
                    shape: MatmulShape::new(r, n, k)?,
                    tile,
                    stage: i as u32,
                    operands: [
                        &deltas[i],
                        &weights[i],
                        &gradients[i],
                        &unused_read,
                        &unused_read,
                        &unused_read,
                    ],
                    validation: &validation,
                    tape: &unused_out,
                },
            )?);
        }
        for (i, layer) in layers.iter().enumerate() {
            passes.push(element(
                &prepare,
                params(i, layer.inner, layer.cols, layer.weights.len()),
                [
                    &weights[i],
                    &biases[i],
                    &weight_gradients[i],
                    &bias_gradients[i],
                    &candidate_weights[i],
                    &candidate_biases[i],
                ],
                false,
            )?);
        }
        passes.push(element(
            &element_pipeline("decide_sgd"),
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
        for (i, layer) in layers.iter().enumerate() {
            passes.push(element(
                &commit,
                params(i, layer.inner, layer.cols, layer.weights.len()),
                [
                    &candidate_weights[i],
                    &candidate_biases[i],
                    &unused_read,
                    &unused_read,
                    &weights[i],
                    &biases[i],
                ],
                false,
            )?);
        }
        let loss_pool = runtime::ReadbackPool::new::<u32>(context.clone(), layers.len() + 3)?;
        Ok(Self {
            activations,
            gradients,
            weights,
            biases,
            weight_gradients,
            bias_gradients,
            target,
            validation,
            loss,
            step_config,
            passes,
            loss_pool,
            input,
            output,
            specs: layers
                .iter()
                .map(|l| Spec {
                    inner: l.inner,
                    cols: l.cols,
                    activation: l.activation,
                })
                .collect(),
            batch_generation: 0,
            submitted_steps: 0,
            last_step: None,
            batch_sources: None,
            runtime,
        })
    }

    pub fn input_layout(&self) -> &NdLayout {
        &self.input
    }
    pub fn output_layout(&self) -> &NdLayout {
        &self.output
    }
    pub fn stage_count(&self) -> usize {
        self.specs.len()
    }
    pub fn submitted_steps(&self) -> u64 {
        self.submitted_steps
    }
    pub fn batch_generation(&self) -> u64 {
        self.batch_generation
    }
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        self.runtime.adapter_info()
    }

    /// Validate the whole batch before changing any queue input or generation.
    pub fn upload_batch(&mut self, input: &[f32], target: &[f32]) -> Result<(), TrainingError> {
        for (label, values, expected) in [
            ("input", input, self.input.len()),
            ("target", target, self.output.len()),
        ] {
            if values.len() != expected {
                return Err(DenseError::InputLength {
                    expected,
                    actual: values.len(),
                }
                .into());
            }
            if !values.iter().all(|v| v.is_finite()) {
                return Err(DenseError::NonFiniteInput(label).into());
            }
        }
        let generation = self
            .batch_generation
            .checked_add(1)
            .ok_or(TrainingError::Overflow)?;
        let queue = self.runtime.context().queue();
        queue.write_buffer(&self.activations[0], 0, bytemuck::cast_slice(input));
        queue.write_buffer(&self.target, 0, bytemuck::cast_slice(target));
        self.batch_generation = generation;
        self.batch_sources = None;
        self.last_step = None;
        Ok(())
    }

    /// Prepare a resident N-D batch without host readback. Upstream failures are
    /// included in the existing all-layer SGD decision on every step using it.
    pub fn upload_batch_tensors(
        &mut self,
        input: &ResidentTensor,
        target: &ResidentTensor,
    ) -> Result<(), TrainingError> {
        input.require_context(self.runtime.context())?;
        target.require_context(self.runtime.context())?;
        if input.layout().shape() != self.input.shape()
            || target.layout().shape() != self.output.shape()
        {
            return Err(DenseError::InvalidLayout.into());
        }
        let generation = self
            .batch_generation
            .checked_add(1)
            .ok_or(TrainingError::Overflow)?;
        let input = input.contiguous()?;
        let target = target.contiguous()?;
        let context = self.runtime.context();
        let mut encoder = context.device().create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(
            input.values(),
            0,
            &self.activations[0],
            0,
            self.input.len() as u64 * 4,
        );
        encoder.copy_buffer_to_buffer(
            target.values(),
            0,
            &self.target,
            0,
            self.output.len() as u64 * 4,
        );
        context.queue().submit(Some(encoder.finish()));
        self.batch_sources = Some([input, target]);
        self.batch_generation = generation;
        self.last_step = None;
        Ok(())
    }

    /// Enqueue forward, mean MSE, exact VJP, candidate validation, then all-layer commit.
    /// Zero rate is allowed for derivative probes; invalid derivatives still reject.
    pub fn step(&mut self, learning_rate: f32) -> Result<u64, TrainingError> {
        if !learning_rate.is_finite() || learning_rate < 0. {
            return Err(TrainingError::LearningRate);
        }
        if self.batch_generation == 0 {
            return Err(TrainingError::MissingBatch);
        }
        let attempt = self
            .submitted_steps
            .checked_add(1)
            .ok_or(TrainingError::Overflow)?;
        let context = self.runtime.context();
        context
            .queue()
            .write_buffer(&self.step_config, 0, bytemuck::bytes_of(&learning_rate));
        let mut encoder = context.device().create_command_encoder(&Default::default());
        encoder.clear_buffer(&self.validation, 0, None);
        if let Some([input, target]) = &self.batch_sources {
            encoder.copy_buffer_to_buffer(input.flags(), 0, &self.validation, 0, 4);
            encoder.copy_buffer_to_buffer(
                target.flags(),
                0,
                &self.validation,
                self.specs.len() as u64 * 4,
                4,
            );
        }
        let chunk_size =
            dispatches_per_pass(self.runtime.adapter_info().backend, self.passes.len());
        for chunk in self.passes.chunks(chunk_size) {
            // Compute usage scopes are per dispatch, so wgpu retains the resource barriers.
            let mut compute = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dense.training.step"),
                timestamp_writes: None,
            });
            for pass in chunk {
                pass.encode(&mut compute);
            }
        }
        context.queue().submit(Some(encoder.finish()));
        self.submitted_steps = attempt;
        self.last_step = Some(attempt);
        Ok(attempt)
    }

    pub fn loss_snapshot(&self) -> Result<StepReadback, TrainingError> {
        let step = self.last_step.ok_or(TrainingError::StaleStep)?;
        Ok(StepReadback {
            raw: readback::capture(
                self.runtime.context(),
                &self.loss_pool,
                &[&self.loss, &self.validation],
            )?,
            stages: self.specs.len(),
            step,
            batch_generation: self.batch_generation,
        })
    }

    /// Full diagnostic snapshot: pre-update loss/output/gradients plus post-update parameters.
    pub fn state_snapshot(&self) -> Result<TrainingStateReadback, TrainingError> {
        let step = self.last_step.ok_or(TrainingError::StaleStep)?;
        let mut buffers = vec![
            &self.loss,
            &self.validation,
            self.activations.last().unwrap(),
            &self.gradients[0],
        ];
        for i in 0..self.specs.len() {
            buffers.extend([
                &self.weights[i],
                &self.biases[i],
                &self.weight_gradients[i],
                &self.bias_gradients[i],
            ]);
        }
        Ok(TrainingStateReadback {
            raw: readback::capture_new(self.runtime.context(), &buffers)?,
            specs: self.specs.clone(),
            input: self.input.clone(),
            output: self.output.clone(),
            step,
            batch_generation: self.batch_generation,
        })
    }

    /// Pre-update prediction plus the whole step's acceptance guard, frozen on GPU.
    pub fn prediction_tensor(
        &self,
        device: &TensorDevice,
    ) -> Result<ResidentTensor, TrainingError> {
        self.step_tensor(device, &self.output, self.activations.last().unwrap())
    }

    /// Pre-update input VJP plus the whole step's guard, frozen on GPU for
    /// preceding pointwise stages. This does not commit any external parameter.
    pub fn input_gradient_tensor(
        &self,
        device: &TensorDevice,
    ) -> Result<ResidentTensor, TrainingError> {
        self.step_tensor(device, &self.input, &self.gradients[0])
    }

    fn step_tensor(
        &self,
        device: &TensorDevice,
        layout: &NdLayout,
        values: &wgpu::Buffer,
    ) -> Result<ResidentTensor, TrainingError> {
        self.last_step.ok_or(TrainingError::StaleStep)?;
        if !device
            .runtime()
            .context()
            .shares_handles_with(self.runtime.context())
        {
            return Err(TensorError::DeviceMismatch.into());
        }
        Ok(device.capture(layout, values, &self.validation)?)
    }

    /// Export parameters even before a step or after a numerically rejected step.
    pub fn parameter_snapshot(&self) -> Result<ParameterReadback, TrainingError> {
        let mut buffers = Vec::new();
        for (w, b) in self.weights.iter().zip(&self.biases) {
            buffers.extend([w, b]);
        }
        Ok(ParameterReadback {
            raw: readback::capture_new(self.runtime.context(), &buffers)?,
            specs: self.specs.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shader_sources::training_dense_matmul_source;

    #[test]
    fn only_measured_metal_coalesces_training_dispatches() {
        for count in [0, 1, 17, 59, 115] {
            assert_eq!(
                dispatches_per_pass(wgpu::Backend::Metal, count),
                count.max(1)
            );
            for backend in [
                wgpu::Backend::Empty,
                wgpu::Backend::Vulkan,
                wgpu::Backend::Dx12,
                wgpu::Backend::Gl,
                wgpu::Backend::BrowserWebGpu,
            ] {
                assert_eq!(dispatches_per_pass(backend, count), 1);
            }
        }
    }

    fn validate(source: &str) {
        let module = naga::front::wgsl::parse_str(source).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }

    #[test]
    fn training_shaders_validate_for_every_matrix_policy() {
        validate(&training_scalar_source());
        for kernel in [MatmulKernel::Scalar, MatmulKernel::Register2x2] {
            for accumulation in [
                MatmulAccumulation::Sequential,
                MatmulAccumulation::Tiled,
                MatmulAccumulation::Compensated,
            ] {
                for (save, transpose) in [(true, false), (false, false), (false, true)] {
                    let source = training_dense_matmul_source(
                        [8, 8, 16],
                        kernel,
                        accumulation,
                        save,
                        transpose,
                    )
                    .unwrap();
                    validate(&source);
                    assert_eq!(source.contains("preactivation[index] = value"), save);
                    assert_eq!(
                        source.contains("rhs_packed[col * params.inner + k]"),
                        transpose
                    );
                }
            }
        }
    }

    #[test]
    fn static_training_attributes_preserve_the_public_dynamic_contract() {
        for (kind, flags, mask) in [
            (TrainingMatmulKind::Forward { gelu: false }, 1, 0),
            (TrainingMatmulKind::Forward { gelu: true }, 5, 0),
            (TrainingMatmulKind::WeightGradient, 64, 256),
            (TrainingMatmulKind::InputGradient, 0, 512),
        ] {
            assert_eq!(kind.flags(), flags);
            assert_eq!(kind.validation_mask(), mask);
            for tile in [[8, 8, 16], [4, 16, 8], [16, 4, 32]] {
                for kernel in [MatmulKernel::Scalar, MatmulKernel::Register2x2] {
                    for accumulation in [
                        MatmulAccumulation::Sequential,
                        MatmulAccumulation::Tiled,
                        MatmulAccumulation::Compensated,
                    ] {
                        let save = matches!(kind, TrainingMatmulKind::Forward { .. });
                        let transpose = kind == TrainingMatmulKind::InputGradient;
                        let dynamic = training_dense_matmul_source(
                            tile,
                            kernel,
                            accumulation,
                            save,
                            transpose,
                        )
                        .unwrap();
                        let specialized =
                            training_dense_stage_source(tile, kernel, accumulation, kind).unwrap();
                        validate(&specialized);
                        for field in [
                            "params.flags",
                            "params.validation_mask",
                            "params.output_scale",
                        ] {
                            assert!(dynamic.contains(field));
                            assert!(!specialized.contains(field));
                        }
                        assert_eq!(specialized.contains("preactivation[index] = value"), save);
                        assert_eq!(
                            specialized.contains("rhs_packed[col * params.inner + k]"),
                            transpose
                        );
                        assert_eq!(
                            specialized.matches("record_nonfinite(").count(),
                            dynamic.matches("record_nonfinite(").count()
                        );
                        assert!(specialized.contains(&format!("flag | {mask}u")));
                    }
                }
            }
        }
    }

    #[test]
    fn element_grids_cover_more_than_one_dispatch_row() {
        let limits = wgpu::Limits {
            max_compute_workgroups_per_dimension: 3,
            ..Default::default()
        };
        assert_eq!(groups(257, &limits).unwrap(), [2, 1, 1]);
        assert_eq!(groups(4 * 256, &limits).unwrap(), [3, 2, 1]);
        assert!(groups(10 * 256, &limits).is_err());
    }
}
