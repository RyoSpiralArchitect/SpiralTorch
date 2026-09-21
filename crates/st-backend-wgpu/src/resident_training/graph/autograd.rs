//! Loss-independent resident VJPs of a frozen graph. No optimizer is executed.
use super::*;
mod cotangent;
mod learner;
mod outputs;
mod prediction;
pub use learner::{
    GraphGradientAccumulator, GraphGradientBatch, GraphUpdateReadback, ResidentGraphLearner,
};

#[derive(Clone)]
struct ParameterState {
    workspace: Shared<()>,
    revision: u64,
}
impl ParameterState {
    fn matches(&self, other: &Self) -> bool {
        self.revision == other.revision && Shared::ptr_eq(&self.workspace, &other.workspace)
    }
}

struct ForwardIdentity {
    generation: u64,
    submission: u64,
    parameters: ParameterState,
}

/// Owning prediction and opaque identity of one workspace's latest forward.
/// The prediction survives reuse/drop; only the latest token can reuse its tape.
#[derive(Clone)]
pub struct GraphForward {
    identity: Shared<ForwardIdentity>,
    prediction: ResidentTensor,
}

impl GraphForward {
    pub fn prediction(&self) -> &ResidentTensor {
        &self.prediction
    }
    pub fn input_generation(&self) -> u64 {
        self.identity.generation
    }
    pub fn submitted_forward(&self) -> u64 {
        self.identity.submission
    }
}

/// Immutable, whole-VJP-guarded tensors in the graph's stable parameter order.
/// A submission is not numerical acceptance; guarded tensor reads prove that.
#[derive(Clone)]
pub struct GraphGradients {
    input: ResidentTensor,
    parameters: Vec<ResidentTensor>,
    forward: Shared<ForwardIdentity>,
    submission: u64,
}

impl GraphGradients {
    pub fn input_gradient(&self) -> &ResidentTensor {
        &self.input
    }
    pub fn parameter_gradients(&self) -> &[ResidentTensor] {
        &self.parameters
    }
    pub fn input_generation(&self) -> u64 {
        self.forward.generation
    }
    pub fn submitted_forward(&self) -> u64 {
        self.forward.submission
    }
    pub fn submitted_backward(&self) -> u64 {
        self.submission
    }
}

/// Separate forward and arbitrary-cotangent backward on the same GPU tape.
/// Parameters are frozen at compilation. VJPs share only intermediate scratch;
/// returned gradients own their output version. No implicit accumulation,
/// batch normalization, loss scaling, or optimizer is applied.
pub struct ResidentGraphAutograd {
    graph: ResidentGraphTraining,
    outputs: outputs::GradientOutputs,
    predictions: prediction::PredictionOutputs,
    forward_validation: wgpu::Buffer,
    cotangent_inherited: Option<wgpu::Buffer>,
    input_source: Option<ResidentTensor>,
    current: Option<Shared<ForwardIdentity>>,
    parameters: ParameterState,
    forwards: u64,
    backwards: u64,
}

impl ResidentGraphAutograd {
    pub fn new(
        runtime: WgpuRuntime,
        definition: GraphDefinition,
        tile: MatmulTile,
        kernel: MatmulKernel,
        accumulation: MatmulAccumulation,
    ) -> Result<Self, TrainingError> {
        Self::from_prepared(ResidentGraphTraining::prepare(
            runtime,
            definition,
            Preparation::Autograd,
            tile,
            kernel,
            accumulation,
        )?)
    }

    fn from_prepared(graph: ResidentGraphTraining) -> Result<Self, TrainingError> {
        let outputs = outputs::GradientOutputs::new(&graph)?;
        let predictions = prediction::PredictionOutputs::new(&graph);
        let forward_validation = runtime::empty_buffer::<u32>(
            graph.device.runtime().context().device(),
            "graph.autograd.forward_flags",
            graph.nodes.len() + 4,
            wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        )?;
        Ok(Self {
            graph,
            outputs,
            predictions,
            forward_validation,
            cotangent_inherited: None,
            input_source: None,
            current: None,
            parameters: ParameterState {
                workspace: Shared::new(()),
                revision: 0,
            },
            forwards: 0,
            backwards: 0,
        })
    }

    pub fn input_layout(&self) -> &NdLayout {
        self.graph.input_layout()
    }
    pub fn output_layout(&self) -> &NdLayout {
        self.graph.output_layout()
    }
    pub fn stage_count(&self) -> usize {
        self.graph.stage_count()
    }
    pub fn parameter_count(&self) -> usize {
        self.graph.parameter_count()
    }
    pub fn tensor_device(&self) -> &TensorDevice {
        &self.graph.device
    }
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        self.graph.adapter_info()
    }
    pub fn input_generation(&self) -> u64 {
        self.graph.batch_generation
    }
    pub fn submitted_forwards(&self) -> u64 {
        self.forwards
    }
    pub fn submitted_backwards(&self) -> u64 {
        self.backwards
    }

    /// Validate before replacing the input or invalidating the current tape.
    pub fn upload(&mut self, values: &[f32]) -> Result<(), TrainingError> {
        if values.len() != self.input_layout().len() {
            return Err(DenseError::InputLength {
                expected: self.input_layout().len(),
                actual: values.len(),
            }
            .into());
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err(DenseError::NonFiniteInput("input").into());
        }
        let generation = self
            .input_generation()
            .checked_add(1)
            .ok_or(TrainingError::Overflow)?;
        self.graph.device.runtime().context().queue().write_buffer(
            &self.graph.activations[0],
            0,
            bytemuck::cast_slice(values),
        );
        self.graph.batch_generation = generation;
        self.input_source = None;
        self.current = None;
        Ok(())
    }

    pub fn set_input_tensor(&mut self, input: &ResidentTensor) -> Result<(), TrainingError> {
        let context = self.graph.device.runtime().context();
        input.require_context(context)?;
        if input.layout().shape() != self.input_layout().shape() {
            return Err(DenseError::InvalidLayout.into());
        }
        let generation = self
            .input_generation()
            .checked_add(1)
            .ok_or(TrainingError::Overflow)?;
        let input = input.contiguous()?;
        let mut encoder = context.device().create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(
            input.values(),
            0,
            &self.graph.activations[0],
            0,
            self.input_layout().len() as u64 * 4,
        );
        context.queue().submit(Some(encoder.finish()));
        self.graph.batch_generation = generation;
        self.input_source = Some(input);
        self.current = None;
        Ok(())
    }

    /// Write an owning prediction directly with the shared forward kernels.
    pub fn forward(&mut self) -> Result<GraphForward, TrainingError> {
        if self.input_generation() == 0 {
            return Err(TrainingError::MissingInput);
        }
        let submission = self
            .forwards
            .checked_add(1)
            .ok_or(TrainingError::Overflow)?;
        let g = &self.graph;
        let context = g.device.runtime().context();
        let mut encoder = context.device().create_command_encoder(&Default::default());
        encoder.clear_buffer(&g.validation, 0, None);
        encoder.clear_buffer(&g.pointwise_flags, 0, None);
        if let Some(input) = &self.input_source {
            encoder.copy_buffer_to_buffer(
                input.flags(),
                0,
                &g.validation,
                (g.nodes.len() + 1) as u64 * 4,
                4,
            );
        }
        let prediction = self.predictions.encode(g, &mut encoder)?;
        encoder.copy_buffer_to_buffer(
            &g.validation,
            0,
            &self.forward_validation,
            0,
            g.validation.size(),
        );
        context.queue().submit(Some(encoder.finish()));
        let identity = Shared::new(ForwardIdentity {
            generation: self.input_generation(),
            submission,
            parameters: self.parameters.clone(),
        });
        self.current = Some(identity.clone());
        self.forwards = submission;
        Ok(GraphForward {
            identity,
            prediction,
        })
    }

    /// Exact VJP seeded by a same-device, same-logical-shape resident tensor.
    /// Repeat with independent band/cotangent tensors on the current forward.
    pub fn backward(
        &mut self,
        forward: &GraphForward,
        cotangent: &ResidentTensor,
    ) -> Result<GraphGradients, TrainingError> {
        let submission = self.check_backward(forward)?;
        let g = &self.graph;
        cotangent.require_context(g.device.runtime().context())?;
        if cotangent.layout().shape() != self.output_layout().shape() {
            return Err(DenseError::InvalidLayout.into());
        }
        let cotangent = cotangent.contiguous()?;
        let mut encoder = self.begin_backward();
        encoder.copy_buffer_to_buffer(
            cotangent.flags(),
            0,
            &g.validation,
            g.nodes.len() as u64 * 4,
            4,
        );
        encoder.copy_buffer_to_buffer(
            cotangent.values(),
            0,
            g.gradients.last().unwrap(),
            0,
            g.output_layout().len() as u64 * 4,
        );
        self.submit_backward(forward, submission, encoder)
    }

    fn check_backward(&self, forward: &GraphForward) -> Result<u64, TrainingError> {
        if !self
            .current
            .as_ref()
            .is_some_and(|id| Shared::ptr_eq(id, &forward.identity))
        {
            return Err(TrainingError::StaleForward);
        }
        self.backwards.checked_add(1).ok_or(TrainingError::Overflow)
    }

    fn begin_backward(&self) -> wgpu::CommandEncoder {
        let g = &self.graph;
        let context = g.device.runtime().context();
        let mut encoder = context.device().create_command_encoder(&Default::default());
        // Restore forward-only guards: a bad seed must not poison future VJPs,
        // and a good seed must never mask an already-invalid forward.
        encoder.copy_buffer_to_buffer(
            &self.forward_validation,
            0,
            &g.validation,
            0,
            g.validation.size(),
        );
        encoder.clear_buffer(&g.pointwise_flags, 0, None);
        encoder
    }

    fn submit_backward(
        &mut self,
        forward: &GraphForward,
        submission: u64,
        mut encoder: wgpu::CommandEncoder,
    ) -> Result<GraphGradients, TrainingError> {
        let g = &self.graph;
        let mut captured = self.outputs.encode(g, &mut encoder)?.into_iter();
        let input = captured.next().expect("prepared input gradient capture");
        let parameters = captured.collect();
        g.device
            .runtime()
            .context()
            .queue()
            .submit(Some(encoder.finish()));
        self.backwards = submission;
        Ok(GraphGradients {
            input,
            parameters,
            forward: forward.identity.clone(),
            submission,
        })
    }
}
