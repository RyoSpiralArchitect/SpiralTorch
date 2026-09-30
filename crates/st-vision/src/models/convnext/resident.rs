//! A compiled ConvNeXt owns GPU weights; its source Module remains an explicit host snapshot.
use super::*;
use st_backend_wgpu::{
    resident_tensor::TensorDevice,
    resident_training::{
        graph::{GraphForward, ResidentGraphAutograd},
        parameters::{
            ResidentParameterGradients, ResidentParameterSnapshot, ResidentParameterUpdate,
            ResidentParameters,
        },
        TrainingError,
    },
};
use st_nn::resident::{InferenceOp, InferencePlan, ResidentConvolutionSpec};
use st_tensor::{Layout, NdLayout};
use std::ops::Range;

mod checkpoint;
pub use checkpoint::ConvNeXtCheckpointSnapshot;

struct GraphPart {
    graph: ResidentGraphAutograd,
    parameters: Range<usize>,
    shapes: Vec<Vec<usize>>,
    bound_revision: u64,
}

impl GraphPart {
    fn compile(
        device: &TensorDevice,
        input: &[usize],
        operations: Vec<InferenceOp>,
        start: usize,
        revision: u64,
    ) -> Result<Self, InferenceError> {
        let plan = InferencePlan::from_operations(NdLayout::contiguous(input)?, operations)?;
        let shapes: Vec<_> = plan
            .graph_definition()?
            .parameters()
            .iter()
            .map(|p| p.shape.clone())
            .collect();
        let end = start
            .checked_add(shapes.len())
            .ok_or(TrainingError::Overflow)?;
        Ok(Self {
            graph: plan.compile_graph_autograd_wgpu(device.runtime().clone())?,
            parameters: start..end,
            shapes,
            bound_revision: revision,
        })
    }

    fn forward(
        &mut self,
        parameters: &ResidentParameterSnapshot,
        input: &ResidentTensor,
    ) -> Result<GraphForward, InferenceError> {
        if self.bound_revision != parameters.revision() {
            let values = parameters.values()[self.parameters.clone()]
                .iter()
                .zip(&self.shapes)
                .map(|(value, shape)| value.reshape(shape))
                .collect::<Result<Vec<_>, _>>()?;
            self.graph.set_parameter_tensors(&values)?;
            self.bound_revision = parameters.revision();
        }
        self.graph.set_input_tensor(input)?;
        Ok(self.graph.forward()?)
    }
}

enum Node {
    Conv {
        spec: ResidentConvolutionSpec,
        parameter: usize,
    },
    Block {
        spec: ResidentConvolutionSpec,
        parameter: usize,
        tail: Box<GraphPart>,
    },
}

enum Tape {
    Conv(ResidentTensor),
    Block {
        input: ResidentTensor,
        tail: GraphForward,
    },
}

/// Owning prediction and same-model/version forward token. Only the latest
/// successful forward can reuse the compiled intermediate tape for backward.
pub struct ResidentConvNeXtForward {
    parameters: ResidentParameterSnapshot,
    submission: u64,
    nodes: Vec<Tape>,
    final_norm: GraphForward,
    feature_shape: Vec<usize>,
}

impl ResidentConvNeXtForward {
    pub fn prediction(&self) -> &ResidentTensor {
        self.final_norm.prediction()
    }
    pub fn parameter_revision(&self) -> u64 {
        self.parameters.revision()
    }
}

/// Exact derivatives from this model's forward, in the source Module parameter order.
pub struct ResidentConvNeXtGradients {
    input: ResidentTensor,
    parameters: Vec<ResidentTensor>,
    bound: ResidentParameterGradients,
}

impl ResidentConvNeXtGradients {
    pub fn input_gradient(&self) -> &ResidentTensor {
        &self.input
    }
    pub fn parameter_gradients(&self) -> &[ResidentTensor] {
        &self.parameters
    }
}

/// Model-owned resident forward, VJP and all-parameter plain SGD for a fixed batch shape.
/// Compilation creates fresh state; an explicit checkpoint restores plain-SGD
/// weights and the attempted-update clock, not ModuleTrainer policy or tapes.
pub struct ResidentConvNeXtBackbone {
    config: ConvNeXtConfig,
    device: TensorDevice,
    parameters: ResidentParameters,
    names: Vec<String>,
    shapes: Vec<[usize; 2]>,
    input_shape: [usize; 4],
    output_shape: [usize; 2],
    nodes: Vec<Node>,
    final_norm: GraphPart,
    forwards: u64,
    latest: Option<u64>,
}

impl ConvNeXtBackbone {
    /// Snapshot this model's weights once. Later host mutations never overwrite
    /// GPU updates. Pending gradients/attached optimizer tapes must be resolved
    /// explicitly before compilation; they are not migrated or silently reset.
    pub fn compile_resident_training(
        &self,
        device: TensorDevice,
        batch: usize,
    ) -> Result<ResidentConvNeXtBackbone, InferenceError> {
        self.compile_resident_at_revision(device, batch, 0)
    }

    fn compile_resident_at_revision(
        &self,
        device: TensorDevice,
        batch: usize,
        revision: u64,
    ) -> Result<ResidentConvNeXtBackbone, InferenceError> {
        if batch == 0 {
            return Err(InferenceError::InvalidLayout);
        }
        let mut host_values = Vec::new();
        let mut names = Vec::new();
        let mut shapes = Vec::new();
        let mut attached = false;
        self.visit_parameters(&mut |p| {
            attached |= p.gradient().is_some() || p.hypergrad().is_some() || p.realgrad().is_some();
            let (rows, cols) = p.value().shape();
            shapes.push([rows, cols]);
            names.push(p.name().to_owned());
            host_values.push(p.value().to_layout(Layout::RowMajor)?.into_snapshot());
            Ok(())
        })?;
        if attached {
            return Err(InferenceError::ModuleUpdate(
                "resolve attached optimizer state before resident compilation",
            ));
        }
        let input_shape = self.stem.resident_spec().input_shape(batch);
        NdLayout::contiguous(&input_shape)?;
        let values = host_values
            .iter()
            .zip(&shapes)
            .map(|(value, shape)| device.upload(shape, value.data()))
            .collect::<Result<Vec<_>, _>>()?;
        let parameters = ResidentParameters::from_restored_values(values, revision)?;
        let mut nodes = vec![Node::Conv {
            spec: self.stem.resident_spec(),
            parameter: 0,
        }];
        let mut offset = 2;
        for stage in &self.stages {
            for block in &stage.blocks {
                let rows = batch
                    .checked_mul(block.hw.0)
                    .and_then(|n| n.checked_mul(block.hw.1))
                    .ok_or(TrainingError::Overflow)?;
                let tail = GraphPart::compile(
                    &device,
                    &[rows, block.channels],
                    block.resident_tail_operations()?,
                    offset + 2,
                    revision,
                )?;
                if tail.parameters.len() != 6 {
                    return Err(InferenceError::Shape(offset));
                }
                nodes.push(Node::Block {
                    spec: block.depthwise.resident_spec(),
                    parameter: offset,
                    tail: Box::new(tail),
                });
                offset += 8;
            }
            if let Some(downsample) = &stage.downsample {
                nodes.push(Node::Conv {
                    spec: downsample.resident_spec(),
                    parameter: offset,
                });
                offset += 2;
            }
        }
        let output_shape = [batch, self.final_norm.features()];
        let mut operations = Vec::new();
        self.final_norm.append_inference_ops(&mut operations)?;
        let final_norm = GraphPart::compile(&device, &output_shape, operations, offset, revision)?;
        if final_norm.parameters.end != shapes.len() {
            return Err(InferenceError::Shape(offset));
        }
        Ok(ResidentConvNeXtBackbone {
            config: self.config.clone(),
            device,
            parameters,
            names,
            shapes,
            input_shape,
            output_shape,
            nodes,
            final_norm,
            forwards: 0,
            latest: None,
        })
    }
}

fn weights(
    parameters: &ResidentParameterSnapshot,
    index: usize,
    spec: &ResidentConvolutionSpec,
) -> Result<(ResidentTensor, ResidentTensor), InferenceError> {
    Ok((
        parameters.values()[index].reshape(spec.weight_shape())?,
        parameters.values()[index + 1].reshape(&spec.bias_shape())?,
    ))
}

impl ResidentConvNeXtBackbone {
    pub fn input_shape(&self) -> &[usize; 4] {
        &self.input_shape
    }
    pub fn output_shape(&self) -> &[usize; 2] {
        &self.output_shape
    }
    pub fn parameter_names(&self) -> &[String] {
        &self.names
    }
    pub fn parameter_snapshot(&self) -> ResidentParameterSnapshot {
        self.parameters.snapshot()
    }

    pub fn forward(
        &mut self,
        input: &ResidentTensor,
    ) -> Result<ResidentConvNeXtForward, InferenceError> {
        if input.layout().shape() != self.input_shape {
            return Err(InferenceError::InvalidLayout);
        }
        if !input
            .device()
            .runtime()
            .context()
            .shares_handles_with(self.device.runtime().context())
        {
            return Err(st_backend_wgpu::resident_tensor::TensorError::DeviceMismatch.into());
        }
        let submission = self
            .forwards
            .checked_add(1)
            .ok_or(TrainingError::Overflow)?;
        self.latest = None;
        let parameters = self.parameters.snapshot();
        let mut value = input.clone();
        let mut nodes = Vec::with_capacity(self.nodes.len());
        for node in &mut self.nodes {
            match node {
                Node::Conv { spec, parameter } => {
                    let (weight, bias) = weights(&parameters, *parameter, spec)?;
                    let next = spec.forward(&value, &weight, &bias)?;
                    nodes.push(Tape::Conv(value));
                    value = next;
                }
                Node::Block {
                    spec,
                    parameter,
                    tail,
                } => {
                    let (weight, bias) = weights(&parameters, *parameter, spec)?;
                    let dw = spec.forward(&value, &weight, &bias)?;
                    let shape = dw.layout().shape();
                    let (batch, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);
                    let tokens = dw
                        .permute(&[0, 2, 3, 1])?
                        .contiguous()?
                        .reshape(&[dw.layout().len() / channels, channels])?;
                    let forward = tail.forward(&parameters, &tokens)?;
                    let next = forward
                        .prediction()
                        .reshape(&[batch, height, width, channels])?
                        .permute(&[0, 3, 1, 2])?
                        .add(&value)?;
                    nodes.push(Tape::Block {
                        input: value,
                        tail: forward,
                    });
                    value = next;
                }
            }
        }
        let feature_shape = value.layout().shape().to_vec();
        let features = value.contiguous()?.reshape(&self.output_shape)?;
        let final_norm = self.final_norm.forward(&parameters, &features)?;
        self.forwards = submission;
        self.latest = Some(submission);
        Ok(ResidentConvNeXtForward {
            parameters,
            submission,
            nodes,
            final_norm,
            feature_shape,
        })
    }

    pub fn backward(
        &mut self,
        forward: &ResidentConvNeXtForward,
        cotangent: &ResidentTensor,
    ) -> Result<ResidentConvNeXtGradients, InferenceError> {
        if self.latest != Some(forward.submission)
            || !self.parameters.is_current(&forward.parameters)
        {
            return Err(TrainingError::StaleForward.into());
        }
        let final_vjp = self
            .final_norm
            .graph
            .backward(&forward.final_norm, cotangent)?;
        let mut gradients = vec![None; self.shapes.len()];
        let assign = |destinations: &mut [Option<ResidentTensor>],
                      start: usize,
                      values: &[ResidentTensor]|
         -> Result<(), InferenceError> {
            for (i, value) in values.iter().enumerate() {
                destinations[start + i] = Some(value.reshape(&self.shapes[start + i])?);
            }
            Ok(())
        };
        assign(
            &mut gradients,
            self.final_norm.parameters.start,
            final_vjp.parameter_gradients(),
        )?;
        let mut gradient = final_vjp.input_gradient().reshape(&forward.feature_shape)?;
        for (node, tape) in self.nodes.iter_mut().zip(&forward.nodes).rev() {
            match (node, tape) {
                (Node::Conv { spec, parameter }, Tape::Conv(input)) => {
                    let (weight, _) = weights(&forward.parameters, *parameter, spec)?;
                    let vjp = spec.vjp(input, &weight, &gradient)?;
                    assign(&mut gradients, *parameter, &vjp[1..])?;
                    gradient = vjp[0].clone();
                }
                (
                    Node::Block {
                        spec,
                        parameter,
                        tail,
                    },
                    Tape::Block { input, tail: tape },
                ) => {
                    let shape = input.layout().shape();
                    let (batch, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);
                    let seed = gradient
                        .permute(&[0, 2, 3, 1])?
                        .contiguous()?
                        .reshape(tail.graph.output_layout().shape())?;
                    let vjp = tail.graph.backward(tape, &seed)?;
                    assign(
                        &mut gradients,
                        tail.parameters.start,
                        vjp.parameter_gradients(),
                    )?;
                    let grad_dw = vjp
                        .input_gradient()
                        .reshape(&[batch, height, width, channels])?
                        .permute(&[0, 3, 1, 2])?;
                    let (weight, _) = weights(&forward.parameters, *parameter, spec)?;
                    let dw = spec.vjp(input, &weight, &grad_dw)?;
                    assign(&mut gradients, *parameter, &dw[1..])?;
                    gradient = dw[0].add(&gradient)?;
                }
                _ => {
                    return Err(InferenceError::ModuleUpdate(
                        "resident tape topology differs",
                    ))
                }
            }
        }
        let parameters = gradients
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or(InferenceError::Shape(self.shapes.len()))?;
        let bound = forward.parameters.bind_gradients(parameters.clone())?;
        Ok(ResidentConvNeXtGradients {
            input: gradient,
            parameters,
            bound,
        })
    }

    /// Submit one checked update across stem, every block/downsample and final norm.
    /// Numerical acceptance is proven only by reading the returned receipt.
    pub fn sgd(
        &mut self,
        gradients: &ResidentConvNeXtGradients,
        rate: f32,
    ) -> Result<ResidentParameterUpdate, InferenceError> {
        let update = self.parameters.sgd(&gradients.bound, rate)?;
        self.latest = None;
        Ok(update)
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    mod checkpoint_checks {
        use crate as st_vision;
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/examples/support/convnext_checkpoint_checks.rs"
        ));
    }

    #[test]
    fn resident_convnext_checkpoint_resumes_with_fresh_identity() {
        if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
            return;
        }
        let (runtime, _) = st_backend_wgpu::runtime::ensure_default_runtime_blocking(
            "vision.convnext.checkpoint.test",
        )
        .unwrap();
        assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
        let device = st_backend_wgpu::resident_tensor::TensorDevice::new(runtime).unwrap();
        let imported = std::env::var("SPIRALTORCH_CONVNEXT_CHECKPOINT_IMPORT")
            .ok()
            .map(|path| std::fs::read_to_string(path).unwrap());
        let mut report =
            pollster::block_on(checkpoint_checks::run(&device, imported.as_deref())).unwrap();
        report.as_object_mut().unwrap().remove("checkpoint_json");
        println!("{report}");
    }

    mod checks {
        use crate as st_vision;
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/examples/support/convnext_learning_checks.rs"
        ));
    }

    #[test]
    fn resident_convnext_learning_matches_cpu_and_preserves_ownership() {
        if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
            return;
        }
        let (runtime, _) = st_backend_wgpu::runtime::ensure_default_runtime_blocking(
            "vision.convnext.learning.test",
        )
        .unwrap();
        assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
        let device = st_backend_wgpu::resident_tensor::TensorDevice::new(runtime).unwrap();
        let report = pollster::block_on(checks::run(&device)).unwrap();
        println!("{report}");
    }
}
