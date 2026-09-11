//! Explicit custom-loss SGD on the resident graph, not a ModuleTrainer policy.
use super::*;
use st_kernel_contracts::elementwise::ElementwiseOp;

const MAX_TERMS: usize = 256;

/// Owning same-forward contributions; assembling a batch does no GPU work.
#[derive(Default)]
pub struct GraphGradientBatch {
    terms: Vec<(GraphGradients, f32)>,
}
impl GraphGradientBatch {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn len(&self) -> usize {
        self.terms.len()
    }
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }
    pub fn add(&mut self, gradients: &GraphGradients, weight: f32) -> Result<(), TrainingError> {
        if self.terms.len() == MAX_TERMS || !weight.is_finite() {
            return Err(TrainingError::GradientTerms);
        }
        if self
            .terms
            .first()
            .is_some_and(|(g, _)| !Shared::ptr_eq(&g.forward, &gradients.forward))
        {
            return Err(TrainingError::StaleForward);
        }
        self.terms.push((gradients.clone(), weight));
        Ok(())
    }
}

/// Mutable counterpart of the frozen autograd workspace. Derivatives remain
/// exact; the explicitly selected gain normalization applies only at SGD.
pub struct ResidentGraphLearner {
    autograd: ResidentGraphAutograd,
    update_validation: wgpu::Buffer,
    updates: u64,
    last_update: Option<(u64, u64)>,
}

impl ResidentGraphLearner {
    pub fn new(
        runtime: WgpuRuntime,
        definition: GraphDefinition,
        policy: GraphGradientPolicy,
        tile: MatmulTile,
        kernel: MatmulKernel,
        accumulation: MatmulAccumulation,
    ) -> Result<Self, TrainingError> {
        let graph = ResidentGraphTraining::prepare(
            runtime,
            definition,
            Preparation::Learner(policy),
            tile,
            kernel,
            accumulation,
        )?;
        let update_validation = runtime::empty_buffer::<u32>(
            graph.device.runtime().context().device(),
            "graph.learner.update_flags",
            graph.nodes.len() + 4,
            wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        )?;
        Ok(Self {
            autograd: ResidentGraphAutograd::from_prepared(graph)?,
            update_validation,
            updates: 0,
            last_update: None,
        })
    }

    pub fn input_layout(&self) -> &NdLayout {
        self.autograd.input_layout()
    }
    pub fn output_layout(&self) -> &NdLayout {
        self.autograd.output_layout()
    }
    pub fn stage_count(&self) -> usize {
        self.autograd.stage_count()
    }
    pub fn parameter_count(&self) -> usize {
        self.autograd.parameter_count()
    }
    pub fn tensor_device(&self) -> &TensorDevice {
        self.autograd.tensor_device()
    }
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        self.autograd.adapter_info()
    }
    pub fn input_generation(&self) -> u64 {
        self.autograd.input_generation()
    }
    pub fn submitted_forwards(&self) -> u64 {
        self.autograd.submitted_forwards()
    }
    pub fn submitted_backwards(&self) -> u64 {
        self.autograd.submitted_backwards()
    }
    pub fn submitted_updates(&self) -> u64 {
        self.updates
    }
    pub fn gradient_policy(&self) -> GraphGradientPolicy {
        self.autograd.graph.policy
    }
    pub fn upload(&mut self, input: &[f32]) -> Result<(), TrainingError> {
        self.autograd.upload(input)
    }
    pub fn set_input_tensor(&mut self, input: &ResidentTensor) -> Result<(), TrainingError> {
        self.autograd.set_input_tensor(input)
    }
    pub fn forward(&mut self) -> Result<GraphForward, TrainingError> {
        self.autograd.forward()
    }
    pub fn backward(
        &mut self,
        forward: &GraphForward,
        cotangent: &ResidentTensor,
    ) -> Result<GraphGradients, TrainingError> {
        self.autograd.backward(forward, cotangent)
    }

    pub fn sgd(&mut self, gradients: &GraphGradients, rate: f32) -> Result<u64, TrainingError> {
        self.sgd_weighted(&[(gradients, 1.)], rate)
    }

    pub fn sgd_batch(
        &mut self,
        batch: &GraphGradientBatch,
        rate: f32,
    ) -> Result<u64, TrainingError> {
        self.sgd_weighted(
            &batch.terms.iter().map(|(g, w)| (g, *w)).collect::<Vec<_>>(),
            rate,
        )
    }

    /// Ordered sum of weight * exact VJP, then explicit gain policy and SGD.
    /// No gradient may come from a different workspace, input or parameter state.
    /// A zero coefficient does not erase an invalid source/intermediate.
    pub fn sgd_weighted(
        &mut self,
        terms: &[(&GraphGradients, f32)],
        rate: f32,
    ) -> Result<u64, TrainingError> {
        if !rate.is_finite() || rate < 0. {
            return Err(TrainingError::LearningRate);
        }
        if terms.is_empty()
            || terms.len() > MAX_TERMS
            || terms.iter().any(|(_, weight)| !weight.is_finite())
        {
            return Err(TrainingError::GradientTerms);
        }
        let current = self
            .autograd
            .current
            .as_ref()
            .ok_or(TrainingError::StaleForward)?;
        if terms
            .iter()
            .any(|(g, _)| !Shared::ptr_eq(current, &g.forward))
        {
            return Err(TrainingError::StaleForward);
        }
        let attempt = self.updates.checked_add(1).ok_or(TrainingError::Overflow)?;
        let g = &self.autograd.graph;
        let context = g.device.runtime().context();
        let flag_count = terms.len() + g.parameters.len();
        storage_limit(flag_count, &context.device().limits())?;
        let inherited = runtime::empty_buffer::<u32>(
            context.device(),
            "graph.learner.sources",
            flag_count,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        )?;
        let scales: Vec<_> = terms
            .iter()
            .map(|(_, value)| g.device.upload(&[], &[*value]))
            .collect::<Result<_, _>>()?;
        let mut encoder = context.device().create_command_encoder(&Default::default());
        encoder.clear_buffer(&g.validation, 0, None);
        // Include every whole-VJP flag even for parameterless graphs or zero weights.
        for (i, (source, _)) in terms.iter().enumerate() {
            encoder.copy_buffer_to_buffer(source.input.flags(), 0, &inherited, i as u64 * 4, 4);
        }
        for id in 0..g.parameters.len() {
            let scaled = |i: usize, encoder: &mut wgpu::CommandEncoder| {
                let source = &terms[i].0.parameters[id];
                if terms[i].1 == 1. {
                    Ok(source.clone())
                } else {
                    source.apply_into(encoder, ElementwiseOp::Multiply, Some(&scales[i]))
                }
            };
            let mut sum = scaled(0, &mut encoder)?;
            for i in 1..terms.len() {
                let next = scaled(i, &mut encoder)?;
                sum = sum.apply_into(&mut encoder, ElementwiseOp::Add, Some(&next))?;
            }
            encoder.copy_buffer_to_buffer(
                sum.values(),
                0,
                &g.raw_gradients[id],
                0,
                g.raw_gradients[id].size(),
            );
            encoder.copy_buffer_to_buffer(
                sum.flags(),
                0,
                &inherited,
                (terms.len() + id) as u64 * 4,
                4,
            );
        }
        // Reduce all source and composition guards through the shared tensor
        // identity kernel. This scalar is only a guard carrier, never a loss.
        let guard = g.device.capture_into(
            &mut encoder,
            &NdLayout::contiguous(&[]).map_err(TensorError::from)?,
            terms[0].0.input.values(),
            &inherited,
        )?;
        encoder.copy_buffer_to_buffer(
            guard.flags(),
            0,
            &g.validation,
            (g.nodes.len() + 2) as u64 * 4,
            4,
        );
        g.encode_passes(&mut encoder, &g.update_passes, &mut Default::default());
        encoder.copy_buffer_to_buffer(
            &g.validation,
            0,
            &self.update_validation,
            0,
            g.validation.size(),
        );
        // Nothing fallible remains after changing a queue-visible learning rate.
        g.write_rate(rate);
        context.queue().submit(Some(encoder.finish()));
        self.last_update = Some((current.generation, current.submission));
        self.updates = attempt;
        self.autograd.current = None;
        Ok(attempt)
    }

    /// Receipt of the latest attempted update, retained across later forwards.
    pub fn update_snapshot(&self) -> Result<GraphUpdateReadback, TrainingError> {
        let (generation, forward) = self.last_update.ok_or(TrainingError::MissingUpdate)?;
        Ok(GraphUpdateReadback {
            raw: readback::capture_new(
                self.autograd.graph.device.runtime().context(),
                &[&self.update_validation],
            )?,
            stages: self.autograd.graph.nodes.len() + 2,
            update: self.updates,
            generation,
            forward,
        })
    }

    /// Weight-only checkpoint; rejected transactions leave every parameter intact.
    pub fn parameter_snapshot(&self) -> Result<GraphParameterReadback, TrainingError> {
        self.autograd.graph.parameter_snapshot()
    }
}

/// An owning acceptance receipt, with no invented objective value.
pub struct GraphUpdateReadback {
    raw: readback::RawSnapshot,
    stages: usize,
    update: u64,
    generation: u64,
    forward: u64,
}
impl GraphUpdateReadback {
    pub fn submitted_update(&self) -> u64 {
        self.update
    }
    pub fn input_generation(&self) -> u64 {
        self.generation
    }
    pub fn submitted_forward(&self) -> u64 {
        self.forward
    }
    fn decode(bytes: &[u8], stages: usize, update: u64) -> Result<u64, TrainingError> {
        if readback::validation_flags(bytes, stages)? != bytes.len() {
            return Err(TrainingError::InvalidReadback);
        }
        Ok(update)
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read(self) -> Result<u64, TrainingError> {
        let Self {
            raw,
            stages,
            update,
            ..
        } = self;
        let bytes = raw.read()?;
        Self::decode(&bytes, stages, update)
    }
    #[cfg(target_arch = "wasm32")]
    pub async fn read_async(self) -> Result<u64, TrainingError> {
        let Self {
            raw,
            stages,
            update,
            ..
        } = self;
        let bytes = raw.read_async().await?;
        Self::decode(&bytes, stages, update)
    }
}
