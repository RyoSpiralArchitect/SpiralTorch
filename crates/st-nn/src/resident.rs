//! Explicit inference lowering from existing Modules, not a second model API.
//!
//! Plans own immutable parameter snapshots. Rebuild after parameter updates.
//! This first lowering supports Linear chains with optional following GELU;
//! unsupported modules fail before GPU allocation, never fall back to CPU.

use crate::{module::Module, Tensor, TensorError};
use st_tensor::{Layout, NdLayout, NdLayoutError};
use thiserror::Error;

/// Modules must emit operations equivalent to their ordinary forward semantics.
#[derive(Clone, Debug)]
pub enum InferenceOp {
    Linear { weight: Tensor, bias: Tensor },
    Gelu,
}

#[derive(Debug, Error)]
pub enum InferenceError {
    #[error("module has no resident inference lowering: {0}")]
    UnsupportedModule(&'static str),
    #[error("inference requires a nonempty contiguous last-axis input at offset zero")]
    InvalidLayout,
    #[error("inference plan contains no linear operation")]
    EmptyPlan,
    #[error("GELU must directly follow an unfused Linear in this lowering")]
    UnsupportedGelu,
    #[error("linear parameter or input dimensions differ at stage {0}")]
    Shape(usize),
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Layout(#[from] NdLayoutError),
    #[cfg(feature = "wgpu")]
    #[error(transparent)]
    Gpu(#[from] st_backend_wgpu::resident_dense::DenseError),
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
        let source_operations = operations.len();
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
        })
    }

    pub fn input_layout(&self) -> &NdLayout {
        &self.input
    }
    pub fn output_layout(&self) -> &NdLayout {
        &self.output
    }
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }
    pub fn source_operation_count(&self) -> usize {
        self.source_operations
    }

    pub fn parameter_snapshots(&self) -> impl Iterator<Item = (&Tensor, &Tensor)> {
        self.stages.iter().map(|stage| (&stage.weight, &stage.bias))
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
        use st_backend_wgpu::resident_dense::{DenseActivation, DenseLayer, ResidentDense};
        let layers: Vec<_> = self
            .stages
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
            .collect();
        Ok(ResidentDense::new(
            runtime,
            self.input.clone(),
            &layers,
            tile,
            kernel,
            accumulation,
        )?)
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
        assert!(matches!(
            InferencePlan::from_module(&Relu::new(), shape.clone()),
            Err(InferenceError::UnsupportedModule(_))
        ));
        assert!(matches!(
            InferencePlan::from_module(&Gelu::new(), shape.clone()),
            Err(InferenceError::UnsupportedGelu)
        ));
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
}
