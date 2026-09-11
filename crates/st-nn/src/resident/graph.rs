use super::*;
use st_kernel_contracts::{
    elementwise::ElementwiseOp,
    pointwise::{PointwiseChain, PointwiseStep},
};

impl InferencePlan {
    /// Compile a frozen graph with separate forward and exact, arbitrary-seed
    /// backward. No MSE, optimizer, ModuleTrainer policy or CPU fallback is added.
    #[cfg(feature = "wgpu")]
    pub fn compile_graph_autograd_wgpu(
        &self,
        runtime: st_backend_wgpu::runtime::WgpuRuntime,
    ) -> Result<st_backend_wgpu::resident_training::graph::ResidentGraphAutograd, InferenceError>
    {
        self.compile_graph_autograd_wgpu_with_options(
            runtime,
            Default::default(),
            st_backend_wgpu::resident_matmul::MatmulKernel::Scalar,
            Default::default(),
        )
    }

    #[cfg(feature = "wgpu")]
    pub fn compile_graph_autograd_wgpu_with_options(
        &self,
        runtime: st_backend_wgpu::runtime::WgpuRuntime,
        tile: st_backend_wgpu::resident_matmul::MatmulTile,
        kernel: st_backend_wgpu::resident_matmul::MatmulKernel,
        accumulation: st_backend_wgpu::resident_matmul::MatmulAccumulation,
    ) -> Result<st_backend_wgpu::resident_training::graph::ResidentGraphAutograd, InferenceError>
    {
        Ok(
            st_backend_wgpu::resident_training::graph::ResidentGraphAutograd::new(
                runtime,
                self.graph_definition()?,
                tile,
                kernel,
                accumulation,
            )?,
        )
    }

    /// Opt in to checked forward/VJP fusion without changing this frozen plan.
    /// Up to three inputs fit the portable graph-training binding floor (eight
    /// storage bindings including VJP scratch/guards). Noncomposable residuals
    /// and resource budgets retain their stage boundary. No CPU fallback occurs.
    /// Diagnostics on the returned plan use its fused stage numbering; parameter
    /// IDs/roles/shapes and the explicit gradient policy keep their meaning.
    pub fn fuse_pointwise(&self) -> Result<Self, InferenceError> {
        match &self.graph {
            Some(graph) => Self::from_graph_definition(graph.fuse_pointwise(3)?),
            None => Ok(self.clone()),
        }
    }

    pub fn is_dense(&self) -> bool {
        self.graph.is_none()
    }
    pub(super) fn require_dense(&self) -> Result<(), InferenceError> {
        if self.is_dense() {
            Ok(())
        } else {
            Err(InferenceError::RequiresGraph)
        }
    }

    pub(super) fn lower_graph(
        input: NdLayout,
        operations: Vec<InferenceOp>,
    ) -> Result<Self, InferenceError> {
        let mut parameters = Vec::new();
        let mut stages = Vec::new();
        let mut width = *input.shape().last().ok_or(InferenceError::InvalidLayout)?;
        for op in operations {
            match op {
                InferenceOp::Linear { weight, bias } => {
                    let (k, n) = weight.shape();
                    if bias.shape() != (1, n) {
                        return Err(InferenceError::Shape(stages.len()));
                    }
                    let weight = weight.to_layout(Layout::RowMajor)?;
                    let bias = bias.to_layout(Layout::RowMajor)?;
                    let id = parameters.len();
                    parameters.push(GraphParameter {
                        role: ParameterRole::Weight,
                        shape: vec![k, n],
                        values: weight.data().to_vec(),
                    });
                    parameters.push(GraphParameter {
                        role: ParameterRole::Bias,
                        shape: vec![n],
                        values: bias.data().to_vec(),
                    });
                    stages.push(GraphStage::Linear {
                        weight: id,
                        bias: id + 1,
                        gelu: false,
                    });
                    width = n;
                }
                InferenceOp::Scale { gain } => {
                    if gain.shape() != (1, width) {
                        return Err(InferenceError::Shape(stages.len()));
                    }
                    let gain = gain.to_layout(Layout::RowMajor)?;
                    let id = parameters.len();
                    parameters.push(GraphParameter {
                        role: ParameterRole::Gain,
                        shape: vec![gain.shape().1],
                        values: gain.data().to_vec(),
                    });
                    stages.push(GraphStage::Pointwise {
                        chain: PointwiseChain::new(
                            2,
                            vec![PointwiseStep {
                                op: ElementwiseOp::Multiply,
                                rhs: Some(1),
                            }],
                        )
                        .map_err(st_kernel_contracts::graph::GraphError::from)?,
                        parameters: vec![id],
                    });
                }
                InferenceOp::Gelu | InferenceOp::Relu => {
                    let gelu = matches!(op, InferenceOp::Gelu);
                    if gelu {
                        if let Some(GraphStage::Linear { gelu: fused, .. }) = stages.last_mut() {
                            if !*fused {
                                *fused = true;
                                continue;
                            }
                        }
                    }
                    stages.push(GraphStage::Pointwise {
                        chain: PointwiseChain::new(
                            1,
                            vec![PointwiseStep {
                                op: if gelu {
                                    ElementwiseOp::Gelu
                                } else {
                                    ElementwiseOp::Relu
                                },
                                rhs: None,
                            }],
                        )
                        .map_err(st_kernel_contracts::graph::GraphError::from)?,
                        parameters: vec![],
                    });
                }
            }
        }
        Self::from_graph_definition(GraphDefinition::new(input, stages, parameters)?)
    }

    /// Shared Rust-owned graph contract for any client. Dense local plans remain
    /// unrestricted until this explicit portable/GPU-addressable conversion.
    pub fn graph_definition(&self) -> Result<GraphDefinition, InferenceError> {
        if let Some(graph) = &self.graph {
            return Ok(graph.clone());
        }
        let mut parameters = Vec::new();
        let mut stages = Vec::new();
        for stage in &self.stages {
            let id = parameters.len();
            let (k, n) = stage.weight.shape();
            parameters.push(GraphParameter {
                role: ParameterRole::Weight,
                shape: vec![k, n],
                values: stage.weight.data().to_vec(),
            });
            parameters.push(GraphParameter {
                role: ParameterRole::Bias,
                shape: vec![n],
                values: stage.bias.data().to_vec(),
            });
            stages.push(GraphStage::Linear {
                weight: id,
                bias: id + 1,
                gelu: stage.gelu,
            });
        }
        Ok(GraphDefinition::new(
            self.input.clone(),
            stages,
            parameters,
        )?)
    }

    pub fn from_graph_definition(graph: GraphDefinition) -> Result<Self, InferenceError> {
        let mut stages = Vec::new();
        let mut source_operations = 0;
        for stage in graph.stages() {
            match stage {
                GraphStage::Linear { weight, bias, gelu } => {
                    let w = &graph.parameters()[*weight];
                    let b = &graph.parameters()[*bias];
                    stages.push(FrozenLinear {
                        weight: Tensor::from_vec(w.shape[0], w.shape[1], w.values.clone())?
                            .into_snapshot(),
                        bias: Tensor::from_vec(1, b.shape[0], b.values.clone())?.into_snapshot(),
                        gelu: *gelu,
                    });
                    source_operations += 1 + usize::from(*gelu);
                }
                GraphStage::Pointwise { chain, .. } => {
                    source_operations += chain.steps().len();
                }
            }
        }
        // Keep imported graph IDs/order even for dense-only v2 snapshots.
        Ok(Self {
            input: graph.input_layout().clone(),
            output: graph.output_layout().clone(),
            stages,
            source_operations,
            graph: Some(graph),
        })
    }

    /// Replaces every parameter role in stable slot order; topology is immutable.
    pub fn with_graph_values(&self, values: Vec<Vec<f32>>) -> Result<Self, InferenceError> {
        Self::from_graph_definition(self.graph_definition()?.with_values(values)?)
    }

    /// Forward-only compilation of either portable plan version, with resident
    /// N-D input/output and no training tape or optimizer allocation.
    #[cfg(feature = "wgpu")]
    pub fn compile_graph_wgpu(
        &self,
        runtime: st_backend_wgpu::runtime::WgpuRuntime,
    ) -> Result<st_backend_wgpu::resident_graph::ResidentGraph, InferenceError> {
        self.compile_graph_wgpu_with_options(
            runtime,
            Default::default(),
            st_backend_wgpu::resident_matmul::MatmulKernel::Scalar,
            Default::default(),
        )
    }

    #[cfg(feature = "wgpu")]
    pub fn compile_graph_wgpu_with_options(
        &self,
        runtime: st_backend_wgpu::runtime::WgpuRuntime,
        tile: st_backend_wgpu::resident_matmul::MatmulTile,
        kernel: st_backend_wgpu::resident_matmul::MatmulKernel,
        accumulation: st_backend_wgpu::resident_matmul::MatmulAccumulation,
    ) -> Result<st_backend_wgpu::resident_graph::ResidentGraph, InferenceError> {
        Ok(st_backend_wgpu::resident_graph::ResidentGraph::new(
            runtime,
            self.graph_definition()?,
            tile,
            kernel,
            accumulation,
        )?)
    }

    /// Opt-in general graph training. Scaler policy must be chosen explicitly;
    /// ordinary Module::backward and the specialized dense path are unchanged.
    #[cfg(feature = "wgpu")]
    pub fn compile_graph_training_wgpu(
        &self,
        runtime: st_backend_wgpu::runtime::WgpuRuntime,
        policy: GraphGradientPolicy,
    ) -> Result<st_backend_wgpu::resident_training::graph::ResidentGraphTraining, InferenceError>
    {
        self.compile_graph_training_wgpu_with_options(
            runtime,
            policy,
            Default::default(),
            st_backend_wgpu::resident_matmul::MatmulKernel::Scalar,
            Default::default(),
        )
    }

    #[cfg(feature = "wgpu")]
    pub fn compile_graph_training_wgpu_with_options(
        &self,
        runtime: st_backend_wgpu::runtime::WgpuRuntime,
        policy: GraphGradientPolicy,
        tile: st_backend_wgpu::resident_matmul::MatmulTile,
        kernel: st_backend_wgpu::resident_matmul::MatmulKernel,
        accumulation: st_backend_wgpu::resident_matmul::MatmulAccumulation,
    ) -> Result<st_backend_wgpu::resident_training::graph::ResidentGraphTraining, InferenceError>
    {
        Ok(
            st_backend_wgpu::resident_training::graph::ResidentGraphTraining::new(
                runtime,
                self.graph_definition()?,
                policy,
                tile,
                kernel,
                accumulation,
            )?,
        )
    }

    /// Diagnostic-only compilation on a private timestamp-capable device. Does
    /// not replace the default runtime or change the ordinary training schedule.
    #[cfg(feature = "wgpu")]
    pub async fn profile_graph_training_wgpu(
        &self,
        policy: GraphGradientPolicy,
        tile: st_backend_wgpu::resident_matmul::MatmulTile,
        kernel: st_backend_wgpu::resident_matmul::MatmulKernel,
        accumulation: st_backend_wgpu::resident_matmul::MatmulAccumulation,
    ) -> Result<st_backend_wgpu::resident_training::graph::ProfiledGraphTraining, InferenceError>
    {
        Ok(
            st_backend_wgpu::resident_training::graph::ProfiledGraphTraining::request(
                self.graph_definition()?,
                policy,
                tile,
                kernel,
                accumulation,
            )
            .await?,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        layers::{Gelu, Relu, Scaler},
        Linear, Sequential,
    };
    #[test]
    fn pointwise_fusion_is_explicit_portable_and_keeps_dense_fast_path() {
        let mut model = Sequential::new();
        model.push(Scaler::new("first", 4).unwrap());
        model.push(Relu::new());
        model.push(Scaler::new("second", 4).unwrap());
        model.push(Gelu::new());
        model.push(Linear::new("linear", 4, 3).unwrap());
        model.push(Gelu::new());
        model.push(Relu::new());
        model.push(Scaler::new("output", 3).unwrap());
        let plan =
            InferencePlan::from_module(&model, NdLayout::contiguous(&[2, 5, 4]).unwrap()).unwrap();
        let original = plan.to_json().unwrap();
        let fused = plan.fuse_pointwise().unwrap();
        assert_eq!((plan.stage_count(), fused.stage_count()), (7, 3));
        assert_eq!(
            fused.source_operation_count(),
            plan.source_operation_count()
        );
        assert_eq!(plan.to_json().unwrap(), original);
        let graph = fused.graph_definition().unwrap();
        assert_eq!(graph.parameter_owners(), &[0, 0, 1, 1, 2]);
        assert_eq!(fused.output_layout(), plan.output_layout());
        let portable = fused.to_json().unwrap();
        assert_eq!(
            InferencePlan::from_json(&portable)
                .unwrap()
                .to_json()
                .unwrap(),
            portable
        );
        assert_eq!(fused.fuse_pointwise().unwrap().to_json().unwrap(), portable);
        let dense = InferencePlan::from_module(
            &Linear::new("dense", 4, 3).unwrap(),
            NdLayout::contiguous(&[4]).unwrap(),
        )
        .unwrap();
        assert!(dense.fuse_pointwise().unwrap().is_dense());
        assert_eq!(
            dense.fuse_pointwise().unwrap().to_json().unwrap(),
            dense.to_json().unwrap()
        );
    }

    #[test]
    fn existing_scaler_relu_lower_with_owned_parameters_and_no_silent_dense_downgrade() {
        assert!(InferencePlan::from_module(
            &Scaler::new("wrong", 1).unwrap(),
            NdLayout::contiguous(&[2, 4]).unwrap()
        )
        .is_err());
        let mut model = Sequential::new();
        model.push(Scaler::new("scale", 4).unwrap());
        model.push(Linear::new("linear", 4, 3).unwrap());
        model.push(Gelu::new());
        model.push(Relu::new());
        let plan =
            InferencePlan::from_module(&model, NdLayout::contiguous(&[2, 5, 4]).unwrap()).unwrap();
        assert_eq!(plan.stage_count(), 3);
        assert_eq!(plan.source_operation_count(), 4);
        assert_eq!(plan.output_layout().shape(), &[2, 5, 3]);
        let graph = plan.graph_definition().unwrap();
        assert_eq!(
            graph
                .parameters()
                .iter()
                .map(|p| p.role)
                .collect::<Vec<_>>(),
            vec![
                ParameterRole::Gain,
                ParameterRole::Weight,
                ParameterRole::Bias
            ]
        );
        assert!(matches!(
            plan.with_parameters(vec![]),
            Err(InferenceError::RequiresGraph)
        ));
        let values = graph
            .parameters()
            .iter()
            .map(|p| vec![0.25; p.values.len()])
            .collect();
        let updated = plan
            .with_graph_values(values)
            .unwrap()
            .graph_definition()
            .unwrap();
        assert_eq!(updated.parameter_owners(), graph.parameter_owners());
        assert_eq!(updated.parameters()[0].values, vec![0.25; 4]);
        assert_eq!(plan.parameter_snapshots().count(), 1);
    }
}
