//! Explicit inference lowering from existing Modules, not a second model API.
//!
//! Plans own immutable parameter snapshots. Rebuild after parameter updates.
//! Dense-only plans retain their specialized fast path; rich graphs also own
//! Scaler/ReLU pointwise parameters and an explicit gradient policy.
//! unsupported modules fail before GPU allocation, never fall back to CPU.

use crate::{module::Module, Tensor, TensorError};
pub use st_kernel_contracts::graph::{
    GraphDefinition, GraphGradientPolicy, GraphParameter, GraphStage, ParameterRole,
};
use st_tensor::{Layout, NdLayout, NdLayoutError};
use thiserror::Error;

mod graph;
mod portable;
pub use portable::{DEFAULT_MAX_PLAN_JSON_BYTES, GRAPH_PLAN_SCHEMA, INFERENCE_PLAN_SCHEMA};

/// Modules must emit operations equivalent to their ordinary forward semantics.
#[derive(Clone, Debug)]
pub enum InferenceOp {
    Linear { weight: Tensor, bias: Tensor },
    Gelu,
    Relu,
    Scale { gain: Tensor },
}

#[derive(Debug, Error)]
pub enum InferenceError {
    #[error("module has no resident inference lowering: {0}")]
    UnsupportedModule(&'static str),
    #[error("inference requires a nonempty contiguous last-axis input at offset zero")]
    InvalidLayout,
    #[error("inference plan contains no operations")]
    EmptyPlan,
    #[error("this is a rich graph; use graph_definition/with_graph_values/compile_graph_wgpu/compile_graph_training_wgpu, not the dense-only API")]
    RequiresGraph,
    #[error(transparent)]
    Graph(#[from] st_kernel_contracts::graph::GraphError),
    #[error("GELU must directly follow an unfused Linear in this lowering")]
    UnsupportedGelu,
    #[error("linear parameter or input dimensions differ at stage {0}")]
    Shape(usize),
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Layout(#[from] NdLayoutError),
    #[error("unsupported inference plan schema: {0}")]
    Schema(String),
    #[error("inference plan JSON has {actual} bytes, exceeding the limit {limit}")]
    JsonLimit { actual: usize, limit: usize },
    #[error(
        "portable inference plans require u32-addressable input, parameter, and stage buffers"
    )]
    PortableAddressSpace,
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[cfg(feature = "wgpu")]
    #[error(transparent)]
    Gpu(#[from] st_backend_wgpu::resident_dense::DenseError),
    #[cfg(feature = "wgpu")]
    #[error(transparent)]
    GraphGpu(#[from] st_backend_wgpu::resident_graph::GraphInferenceError),
    #[cfg(feature = "wgpu")]
    #[error(transparent)]
    Training(#[from] st_backend_wgpu::resident_training::TrainingError),
}

#[derive(Clone, Debug)]
struct FrozenLinear {
    weight: Tensor,
    bias: Tensor,
    gelu: bool,
}

#[derive(Clone, Debug)]
pub struct InferencePlan {
    input: NdLayout,
    output: NdLayout,
    stages: Vec<FrozenLinear>,
    source_operations: usize,
    graph: Option<GraphDefinition>,
}

impl InferencePlan {
    /// Flatten leading axes for Linear, preserving their logical N-D shape.
    pub fn from_module(
        module: &(impl Module + ?Sized),
        input: NdLayout,
    ) -> Result<Self, InferenceError> {
        if input.rank() == 0 || input.is_empty() || !input.is_contiguous() || input.offset() != 0 {
            return Err(InferenceError::InvalidLayout);
        }
        let operations = module.inference_ops()?;
        Self::from_operations(input, operations)
    }

    fn from_operations(
        input: NdLayout,
        operations: Vec<InferenceOp>,
    ) -> Result<Self, InferenceError> {
        if input.rank() == 0 || input.is_empty() || !input.is_contiguous() || input.offset() != 0 {
            return Err(InferenceError::InvalidLayout);
        }
        let source_operations = operations.len();
        let rich = operations.iter().enumerate().any(|(i, op)| match op {
            InferenceOp::Scale { .. } | InferenceOp::Relu => true,
            InferenceOp::Gelu => i == 0 || !matches!(operations[i - 1], InferenceOp::Linear { .. }),
            _ => false,
        });
        if rich {
            return Self::lower_graph(input, operations);
        }
        let mut stages: Vec<FrozenLinear> = Vec::new();
        let mut width = *input.shape().last().unwrap();
        for operation in operations {
            match operation {
                InferenceOp::Linear { weight, bias } => {
                    let (inner, cols) = weight.shape();
                    if inner != width || cols == 0 || bias.shape() != (1, cols) {
                        return Err(InferenceError::Shape(stages.len()));
                    }
                    for value in weight.data().iter().chain(bias.data()) {
                        if !value.is_finite() {
                            return Err(TensorError::NonFiniteValue {
                                label: "inference_parameters",
                                value: *value,
                            }
                            .into());
                        }
                    }
                    stages.push(FrozenLinear {
                        weight: weight.to_layout(Layout::RowMajor)?.into_snapshot(),
                        bias: bias.to_layout(Layout::RowMajor)?.into_snapshot(),
                        gelu: false,
                    });
                    width = cols;
                }
                InferenceOp::Gelu => {
                    let stage = stages
                        .last_mut()
                        .filter(|stage| !stage.gelu)
                        .ok_or(InferenceError::UnsupportedGelu)?;
                    stage.gelu = true;
                }
                InferenceOp::Scale { .. } | InferenceOp::Relu => {
                    unreachable!("rich operations were lowered above")
                }
            }
        }
        if stages.is_empty() {
            return Err(InferenceError::EmptyPlan);
        }
        let mut shape = input.shape().to_vec();
        *shape.last_mut().unwrap() = width;
        Ok(Self {
            input,
            output: NdLayout::contiguous(&shape)?,
            stages,
            source_operations,
            graph: None,
        })
    }

    pub fn input_layout(&self) -> &NdLayout {
        &self.input
    }
    pub fn output_layout(&self) -> &NdLayout {
        &self.output
    }
    pub fn stage_count(&self) -> usize {
        self.graph
            .as_ref()
            .map_or(self.stages.len(), |g| g.stages().len())
    }
    pub fn source_operation_count(&self) -> usize {
        self.source_operations
    }

    /// Linear weight/bias snapshots only. For all roles (including gains), use
    /// graph_definition().parameters(); this legacy iterator never includes gains.
    pub fn parameter_snapshots(&self) -> impl Iterator<Item = (&Tensor, &Tensor)> {
        self.stages.iter().map(|stage| (&stage.weight, &stage.bias))
    }

    /// Replace values without changing the graph or the original frozen plan.
    pub fn with_parameters(
        &self,
        parameters: Vec<(Tensor, Tensor)>,
    ) -> Result<Self, InferenceError> {
        self.require_dense()?;
        if parameters.len() != self.stages.len() {
            return Err(InferenceError::Shape(parameters.len()));
        }
        let mut operations = Vec::new();
        for (index, ((weight, bias), original)) in
            parameters.into_iter().zip(&self.stages).enumerate()
        {
            if weight.shape() != original.weight.shape() || bias.shape() != original.bias.shape() {
                return Err(InferenceError::Shape(index));
            }
            operations.push(InferenceOp::Linear { weight, bias });
            if original.gelu {
                operations.push(InferenceOp::Gelu);
            }
        }
        Self::from_operations(self.input.clone(), operations)
    }

    /// Import a resident parameter readback without changing its source graph.
    #[cfg(feature = "wgpu")]
    pub fn with_dense_parameters(
        &self,
        layers: Vec<st_backend_wgpu::resident_dense::DenseLayer>,
    ) -> Result<Self, InferenceError> {
        self.require_dense()?;
        use st_backend_wgpu::resident_dense::DenseActivation;
        if layers.len() != self.stages.len() {
            return Err(InferenceError::Shape(layers.len()));
        }
        let mut parameters = Vec::with_capacity(layers.len());
        for (index, (layer, original)) in layers.into_iter().zip(&self.stages).enumerate() {
            if (layer.inner, layer.cols) != original.weight.shape()
                || (layer.activation == DenseActivation::Gelu) != original.gelu
            {
                return Err(InferenceError::Shape(index));
            }
            parameters.push((
                Tensor::from_vec(layer.inner, layer.cols, layer.weights)?,
                Tensor::from_vec(1, layer.cols, layer.bias)?,
            ));
        }
        self.with_parameters(parameters)
    }

    #[cfg(feature = "wgpu")]
    pub fn compile_wgpu(
        &self,
        runtime: st_backend_wgpu::runtime::WgpuRuntime,
    ) -> Result<st_backend_wgpu::resident_dense::ResidentDense, InferenceError> {
        self.compile_wgpu_with_options(
            runtime,
            Default::default(),
            st_backend_wgpu::resident_matmul::MatmulKernel::Scalar,
            st_backend_wgpu::resident_matmul::MatmulAccumulation::Sequential,
        )
    }

    /// Explicit kernel choices use the same checks at every lowered stage.
    #[cfg(feature = "wgpu")]
    pub fn compile_wgpu_with_options(
        &self,
        runtime: st_backend_wgpu::runtime::WgpuRuntime,
        tile: st_backend_wgpu::resident_matmul::MatmulTile,
        kernel: st_backend_wgpu::resident_matmul::MatmulKernel,
        accumulation: st_backend_wgpu::resident_matmul::MatmulAccumulation,
    ) -> Result<st_backend_wgpu::resident_dense::ResidentDense, InferenceError> {
        self.require_dense()?;
        Ok(st_backend_wgpu::resident_dense::ResidentDense::new(
            runtime,
            self.input.clone(),
            &self.dense_layers(),
            tile,
            kernel,
            accumulation,
        )?)
    }

    /// Explicit plain SGD with mean-MSE and exact VJPs. Host modules/tapes are not mutated.
    #[cfg(feature = "wgpu")]
    pub fn compile_training_wgpu(
        &self,
        runtime: st_backend_wgpu::runtime::WgpuRuntime,
    ) -> Result<st_backend_wgpu::resident_training::ResidentDenseTraining, InferenceError> {
        self.compile_training_wgpu_with_options(
            runtime,
            Default::default(),
            st_backend_wgpu::resident_matmul::MatmulKernel::Scalar,
            Default::default(),
        )
    }

    #[cfg(feature = "wgpu")]
    pub fn compile_training_wgpu_with_options(
        &self,
        runtime: st_backend_wgpu::runtime::WgpuRuntime,
        tile: st_backend_wgpu::resident_matmul::MatmulTile,
        kernel: st_backend_wgpu::resident_matmul::MatmulKernel,
        accumulation: st_backend_wgpu::resident_matmul::MatmulAccumulation,
    ) -> Result<st_backend_wgpu::resident_training::ResidentDenseTraining, InferenceError> {
        self.require_dense()?;
        Ok(
            st_backend_wgpu::resident_training::ResidentDenseTraining::new(
                runtime,
                self.input.clone(),
                &self.dense_layers(),
                tile,
                kernel,
                accumulation,
            )?,
        )
    }

    #[cfg(feature = "wgpu")]
    fn dense_layers(&self) -> Vec<st_backend_wgpu::resident_dense::DenseLayer> {
        use st_backend_wgpu::resident_dense::{DenseActivation, DenseLayer};
        self.stages
            .iter()
            .map(|stage| DenseLayer {
                inner: stage.weight.shape().0,
                cols: stage.weight.shape().1,
                weights: stage.weight.data().to_vec(),
                bias: stage.bias.data().to_vec(),
                activation: if stage.gelu {
                    DenseActivation::Gelu
                } else {
                    DenseActivation::None
                },
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        layers::{Gelu, Relu},
        Linear, Sequential,
    };

    #[test]
    fn existing_modules_lower_with_nd_shape_and_fusion() {
        let mut block = Sequential::new();
        block.push(Linear::new("up", 4, 7).unwrap());
        block.push(Gelu::new());
        block.push(Linear::new("down", 7, 3).unwrap());
        let mut nested = Sequential::new();
        nested.push(block);
        let module: &dyn Module = &nested;
        let plan =
            InferencePlan::from_module(module, NdLayout::contiguous(&[2, 5, 4]).unwrap()).unwrap();
        assert_eq!(plan.output_layout().shape(), &[2, 5, 3]);
        assert_eq!(plan.source_operation_count(), 3);
        assert_eq!(plan.stage_count(), 2);
        assert!(plan.stages[0].gelu);
        assert!(!plan.stages[1].gelu);
    }

    #[test]
    fn unsupported_layers_and_shapes_fail_closed() {
        let shape = NdLayout::contiguous(&[2, 4]).unwrap();
        assert!(!InferencePlan::from_module(&Relu::new(), shape.clone())
            .unwrap()
            .is_dense());
        assert!(!InferencePlan::from_module(&Gelu::new(), shape.clone())
            .unwrap()
            .is_dense());
        assert!(matches!(
            InferencePlan::from_module(&Sequential::new(), shape.clone()),
            Err(InferenceError::EmptyPlan)
        ));
        let linear = Linear::new("linear", 3, 4).unwrap();
        assert!(matches!(
            InferencePlan::from_module(&linear, shape.clone()),
            Err(InferenceError::Shape(0))
        ));
        assert!(matches!(
            InferencePlan::from_module(&linear, shape.permute(&[1, 0]).unwrap()),
            Err(InferenceError::InvalidLayout)
        ));
        for dims in [&[][..], &[0, 3][..]] {
            assert!(
                InferencePlan::from_module(&linear, NdLayout::contiguous(dims).unwrap()).is_err()
            );
        }
    }

    #[test]
    fn parameter_updates_require_a_new_snapshot() {
        let mut linear = Linear::new("linear", 2, 2).unwrap();
        let shape = NdLayout::contiguous(&[2]).unwrap();
        let plan = InferencePlan::from_module(&linear, shape.clone()).unwrap();
        let before = plan.stages[0].weight.data().to_vec();
        linear
            .visit_parameters_mut(&mut |parameter| {
                parameter.value_mut().data_mut().fill(42.0);
                Ok(())
            })
            .unwrap();
        assert_eq!(plan.stages[0].weight.data(), before);
        let refreshed = InferencePlan::from_module(&linear, shape).unwrap();
        assert_eq!(refreshed.stages[0].weight.data(), &[42.0; 4]);
    }

    #[test]
    fn exported_training_parameters_preserve_graph_and_validate_before_replacement() {
        let mut model = Sequential::new();
        model.push(Linear::new("linear", 2, 3).unwrap());
        model.push(Gelu::new());
        let plan =
            InferencePlan::from_module(&model, NdLayout::contiguous(&[2, 5, 2]).unwrap()).unwrap();
        let original = plan.stages[0].weight.data().to_vec();
        let parameters = || {
            vec![(
                Tensor::from_vec(2, 3, vec![0.5; 6]).unwrap(),
                Tensor::from_vec(1, 3, vec![0.25; 3]).unwrap(),
            )]
        };
        let updated = plan.with_parameters(parameters()).unwrap();
        assert_eq!(updated.stages[0].weight.data(), &[0.5; 6]);
        assert_eq!(plan.stages[0].weight.data(), original);
        assert_eq!(updated.output_layout().shape(), &[2, 5, 3]);
        assert_eq!(updated.source_operation_count(), 2);
        assert!(updated.stages[0].gelu);
        assert!(plan.with_parameters(vec![]).is_err());
        assert!(plan
            .with_parameters(vec![(
                Tensor::zeros(3, 2).unwrap(),
                Tensor::zeros(1, 3).unwrap()
            )])
            .is_err());
        let mut invalid = parameters();
        invalid[0].0.data_mut()[0] = f32::NAN;
        assert!(plan.with_parameters(invalid).is_err());
        assert_eq!(plan.stages[0].weight.data(), original);
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn dense_parameter_export_rejects_graph_and_value_corruption() {
        use st_backend_wgpu::resident_dense::{DenseActivation, DenseLayer};
        let mut model = Sequential::new();
        model.push(Linear::new("up", 2, 3).unwrap());
        model.push(Gelu::new());
        model.push(Linear::new("down", 3, 1).unwrap());
        let plan =
            InferencePlan::from_module(&model, NdLayout::contiguous(&[2, 5, 2]).unwrap()).unwrap();
        let original = plan.to_json().unwrap();
        let layers = || {
            vec![
                DenseLayer {
                    inner: 2,
                    cols: 3,
                    weights: vec![0.5; 6],
                    bias: vec![0.25; 3],
                    activation: DenseActivation::Gelu,
                },
                DenseLayer {
                    inner: 3,
                    cols: 1,
                    weights: vec![-0.125; 3],
                    bias: vec![0.0],
                    activation: DenseActivation::None,
                },
            ]
        };
        let updated = plan.with_dense_parameters(layers()).unwrap();
        assert_eq!(updated.input_layout(), plan.input_layout());
        assert_eq!(updated.output_layout(), plan.output_layout());
        assert_eq!(updated.source_operation_count(), 3);
        assert!(updated.stages[0].gelu && !updated.stages[1].gelu);
        assert_eq!(updated.stages[0].weight.data(), &[0.5; 6]);
        for kind in 0..8 {
            let mut invalid = layers();
            match kind {
                0 => {
                    invalid.pop();
                }
                1 => invalid[0].inner = 3,
                2 => invalid[1].cols = 2,
                3 => invalid[0].activation = DenseActivation::None,
                4 => {
                    invalid[0].weights.pop();
                }
                5 => {
                    invalid[1].bias.clear();
                }
                6 => invalid[0].weights[0] = f32::NAN,
                _ => invalid[1].bias[0] = f32::INFINITY,
            }
            assert!(plan.with_dense_parameters(invalid).is_err(), "case {kind}");
            assert_eq!(plan.to_json().unwrap(), original);
        }
    }
}
