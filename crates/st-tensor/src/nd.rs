//! Immutable N-D tensors. Host storage reuses protected Tensor snapshots;
//! WGPU storage shares the backend's native/browser resident contract.

use crate::{Layout, NdLayout, NdLayoutError, Tensor, TensorError};
#[cfg(feature = "wgpu_dense")]
pub use st_backend_wgpu::resident_tensor::TensorDevice as WgpuTensorDevice;
#[cfg(feature = "wgpu_dense")]
use st_backend_wgpu::resident_tensor::{ResidentTensor, TensorError as DeviceError};
pub use st_kernel_contracts::pointwise::{
    PointwiseChain, PointwiseError, PointwiseExecution, PointwiseStep,
};
use st_kernel_contracts::{elementwise::ElementwiseOp, layout::broadcast_shape};
use thiserror::Error;

mod vjp;
pub use vjp::NdPointwiseVjpPlan;

#[derive(Debug, Error)]
pub enum NdTensorError {
    #[error(transparent)]
    Pointwise(#[from] PointwiseError),
    #[error(transparent)]
    Layout(#[from] NdLayoutError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[cfg(feature = "wgpu_dense")]
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error("data length does not match N-D shape")]
    Length,
    #[error("non-finite N-D input, intermediate or output")]
    NonFinite,
    #[error("operation requires tensors on the same device; transfer explicitly")]
    DeviceMismatch,
    #[error("legacy host layout cannot be represented by an N-D strided view")]
    HostLayout,
    #[error("legacy Tensor conversion requires rank two")]
    Rank,
}

#[derive(Clone, Debug)]
enum Storage {
    Host {
        tensor: Tensor,
        layout: NdLayout,
    },
    #[cfg(feature = "wgpu_dense")]
    Wgpu(ResidentTensor),
}

#[derive(Clone, Debug)]
pub struct NdTensor {
    storage: Storage,
}

/// A shape-preserving chain prepared for one device and exact input layouts.
/// GPU execution modes vary scheduling only; CPU uses the same checked steps.
/// New inputs can be supplied without recompiling, and outputs remain immutable.
#[derive(Debug)]
pub struct NdPointwisePlan {
    chain: PointwiseChain,
    layouts: Vec<NdLayout>,
    #[cfg(feature = "wgpu_dense")]
    gpu: Option<st_backend_wgpu::resident_tensor::pointwise::PointwisePlan>,
}

impl NdPointwisePlan {
    /// Add reverse-mode kernels explicitly; forward-only preparation stays unchanged.
    pub fn into_vjp(self) -> Result<NdPointwiseVjpPlan, NdTensorError> {
        NdPointwiseVjpPlan::from_forward(self)
    }
    pub fn new(chain: PointwiseChain, inputs: &[&NdTensor]) -> Result<Self, NdTensorError> {
        let layouts: Vec<_> = inputs.iter().map(|t| t.layout().clone()).collect();
        chain.validate_layouts(&layouts)?;
        #[cfg(feature = "wgpu_dense")]
        let gpu = if let Some(first) = inputs[0].as_wgpu() {
            for input in inputs {
                let input = input.as_wgpu().ok_or(NdTensorError::DeviceMismatch)?;
                if !first
                    .device()
                    .runtime()
                    .context()
                    .shares_handles_with(input.device().runtime().context())
                {
                    return Err(NdTensorError::DeviceMismatch);
                }
            }
            Some(
                st_backend_wgpu::resident_tensor::pointwise::PointwisePlan::new(
                    first.device().clone(),
                    chain.clone(),
                    layouts.clone(),
                )?,
            )
        } else {
            if inputs.iter().any(|t| t.is_wgpu()) {
                return Err(NdTensorError::DeviceMismatch);
            }
            None
        };
        Ok(Self {
            chain,
            layouts,
            #[cfg(feature = "wgpu_dense")]
            gpu,
        })
    }

    pub fn run(
        &self,
        inputs: &[&NdTensor],
        execution: PointwiseExecution,
    ) -> Result<NdTensor, NdTensorError> {
        if inputs.len() != self.layouts.len() {
            return Err(PointwiseError::Operands.into());
        }
        for (input, layout) in inputs.iter().zip(&self.layouts) {
            if input.layout() != layout {
                return Err(PointwiseError::LayoutMismatch.into());
            }
        }
        #[cfg(feature = "wgpu_dense")]
        if let Some(gpu) = &self.gpu {
            let inputs: Result<Vec<_>, _> = inputs
                .iter()
                .map(|t| t.as_wgpu().ok_or(NdTensorError::DeviceMismatch))
                .collect();
            return Ok(NdTensor::from_wgpu(gpu.run(&inputs?, execution)?));
        }
        let _ = execution;
        if inputs.iter().any(|t| t.is_wgpu()) {
            return Err(NdTensorError::DeviceMismatch);
        }
        let mut current = inputs[0].clone();
        for step in self.chain.steps() {
            current = current.apply(step.op, step.rhs.map(|i| inputs[i]))?;
        }
        Ok(current)
    }
}

impl NdTensor {
    pub fn from_vec(shape: &[usize], data: Vec<f32>) -> Result<Self, NdTensorError> {
        let layout = NdLayout::contiguous(shape)?;
        if data.len() != layout.len() {
            return Err(NdTensorError::Length);
        }
        if !data.iter().all(|v| v.is_finite()) {
            return Err(NdTensorError::NonFinite);
        }
        Ok(Self {
            storage: Storage::Host {
                tensor: Tensor::from_vec(1, data.len(), data)?.into_snapshot(),
                layout,
            },
        })
    }

    /// Freeze borrowed host storage. Existing snapshots share without copying;
    /// mutable/foreign aliases are isolated by the existing snapshot contract.
    pub fn from_tensor(tensor: &Tensor) -> Result<Self, NdTensorError> {
        Self::from_owned_tensor(tensor.clone())
    }

    /// Consuming uniquely owned native storage avoids the snapshot copy.
    pub fn from_owned_tensor(tensor: Tensor) -> Result<Self, NdTensorError> {
        let (rows, cols) = tensor.shape();
        let layout = match tensor.layout() {
            Layout::RowMajor => NdLayout::contiguous(&[rows, cols])?,
            Layout::ColMajor => NdLayout::contiguous(&[cols, rows])?.permute(&[1, 0])?,
            _ => return Err(NdTensorError::HostLayout),
        };
        let tensor = tensor.into_snapshot();
        if !tensor.data().iter().all(|v| v.is_finite()) {
            return Err(NdTensorError::NonFinite);
        }
        Ok(Self {
            storage: Storage::Host { tensor, layout },
        })
    }

    pub fn layout(&self) -> &NdLayout {
        match &self.storage {
            Storage::Host { layout, .. } => layout,
            #[cfg(feature = "wgpu_dense")]
            Storage::Wgpu(tensor) => tensor.layout(),
        }
    }
    pub fn shape(&self) -> &[usize] {
        self.layout().shape()
    }
    pub fn len(&self) -> usize {
        self.layout().len()
    }
    pub fn is_empty(&self) -> bool {
        self.layout().is_empty()
    }
    pub fn is_wgpu(&self) -> bool {
        match self.storage {
            Storage::Host { .. } => false,
            #[cfg(feature = "wgpu_dense")]
            Storage::Wgpu(_) => true,
        }
    }

    pub fn reshape(&self, shape: &[usize]) -> Result<Self, NdTensorError> {
        match &self.storage {
            Storage::Host { tensor, layout } => Ok(Self {
                storage: Storage::Host {
                    tensor: tensor.clone(),
                    layout: layout.reshape(shape)?,
                },
            }),
            #[cfg(feature = "wgpu_dense")]
            Storage::Wgpu(t) => Ok(Self::from_wgpu(t.reshape(shape)?)),
        }
    }
    pub fn permute(&self, axes: &[usize]) -> Result<Self, NdTensorError> {
        match &self.storage {
            Storage::Host { tensor, layout } => Ok(Self {
                storage: Storage::Host {
                    tensor: tensor.clone(),
                    layout: layout.permute(axes)?,
                },
            }),
            #[cfg(feature = "wgpu_dense")]
            Storage::Wgpu(t) => Ok(Self::from_wgpu(t.permute(axes)?)),
        }
    }
    pub fn narrow(&self, axis: usize, start: usize, length: usize) -> Result<Self, NdTensorError> {
        match &self.storage {
            Storage::Host { tensor, layout } => Ok(Self {
                storage: Storage::Host {
                    tensor: tensor.clone(),
                    layout: layout.narrow(axis, start, length)?,
                },
            }),
            #[cfg(feature = "wgpu_dense")]
            Storage::Wgpu(t) => Ok(Self::from_wgpu(t.narrow(axis, start, length)?)),
        }
    }
    pub fn broadcast_to(&self, shape: &[usize]) -> Result<Self, NdTensorError> {
        match &self.storage {
            Storage::Host { tensor, layout } => Ok(Self {
                storage: Storage::Host {
                    tensor: tensor.clone(),
                    layout: layout.broadcast_to(shape)?,
                },
            }),
            #[cfg(feature = "wgpu_dense")]
            Storage::Wgpu(t) => Ok(Self::from_wgpu(t.broadcast_to(shape)?)),
        }
    }

    fn apply(&self, op: ElementwiseOp, rhs: Option<&Self>) -> Result<Self, NdTensorError> {
        let other = rhs.unwrap_or(self);
        match (&self.storage, &other.storage) {
            (
                Storage::Host {
                    tensor: a,
                    layout: al,
                },
                Storage::Host {
                    tensor: b,
                    layout: bl,
                },
            ) => {
                let shape = broadcast_shape(al.shape(), bl.shape())?;
                let al = al.broadcast_to(&shape)?;
                let bl = bl.broadcast_to(&shape)?;
                let data: Result<Vec<_>, _> = (0..al.len())
                    .map(|i| {
                        op.apply(
                            a.data()[al.storage_index(i).unwrap()],
                            b.data()[bl.storage_index(i).unwrap()],
                        )
                        .ok_or(NdTensorError::NonFinite)
                    })
                    .collect();
                Self::from_vec(&shape, data?)
            }
            #[cfg(feature = "wgpu_dense")]
            (Storage::Wgpu(a), Storage::Wgpu(b)) => {
                Ok(Self::from_wgpu(a.apply(op, rhs.map(|_| b))?))
            }
            #[cfg(feature = "wgpu_dense")]
            _ => Err(NdTensorError::DeviceMismatch),
        }
    }
    pub fn add(&self, rhs: &Self) -> Result<Self, NdTensorError> {
        self.apply(ElementwiseOp::Add, Some(rhs))
    }
    pub fn mul(&self, rhs: &Self) -> Result<Self, NdTensorError> {
        self.apply(ElementwiseOp::Multiply, Some(rhs))
    }
    pub fn relu(&self) -> Result<Self, NdTensorError> {
        self.apply(ElementwiseOp::Relu, None)
    }
    pub fn gelu(&self) -> Result<Self, NdTensorError> {
        self.apply(ElementwiseOp::Gelu, None)
    }
    pub fn contiguous(&self) -> Result<Self, NdTensorError> {
        if self.layout().is_contiguous() && self.layout().offset() == 0 {
            Ok(self.clone())
        } else {
            self.apply(ElementwiseOp::Identity, None)
        }
    }

    #[cfg(feature = "wgpu_dense")]
    pub fn from_wgpu(tensor: ResidentTensor) -> Self {
        Self {
            storage: Storage::Wgpu(tensor),
        }
    }
    #[cfg(feature = "wgpu_dense")]
    pub fn as_wgpu(&self) -> Option<&ResidentTensor> {
        if let Storage::Wgpu(tensor) = &self.storage {
            Some(tensor)
        } else {
            None
        }
    }
    #[cfg(feature = "wgpu_dense")]
    pub fn to_wgpu(&self, device: &WgpuTensorDevice) -> Result<Self, NdTensorError> {
        match &self.storage {
            Storage::Host { tensor, layout } => {
                let values: Vec<_> = (0..layout.len())
                    .map(|i| tensor.data()[layout.storage_index(i).unwrap()])
                    .collect();
                Ok(Self::from_wgpu(device.upload(layout.shape(), &values)?))
            }
            Storage::Wgpu(t) => {
                if !t
                    .device()
                    .runtime()
                    .context()
                    .shares_handles_with(device.runtime().context())
                {
                    return Err(NdTensorError::DeviceMismatch);
                }
                Ok(self.clone())
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_values(&self) -> Result<Vec<f32>, NdTensorError> {
        match &self.storage {
            Storage::Host { tensor, layout } => Ok((0..layout.len())
                .map(|i| tensor.data()[layout.storage_index(i).unwrap()])
                .collect()),
            #[cfg(feature = "wgpu_dense")]
            Storage::Wgpu(t) => Ok(t.snapshot()?.read()?),
        }
    }
    #[cfg(target_arch = "wasm32")]
    pub async fn read_values_async(&self) -> Result<Vec<f32>, NdTensorError> {
        match &self.storage {
            Storage::Host { tensor, layout } => Ok((0..layout.len())
                .map(|i| tensor.data()[layout.storage_index(i).unwrap()])
                .collect()),
            #[cfg(feature = "wgpu_dense")]
            Storage::Wgpu(t) => Ok(t.snapshot()?.read_async().await?),
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn to_tensor(&self) -> Result<Tensor, NdTensorError> {
        if self.shape().len() != 2 {
            return Err(NdTensorError::Rank);
        }
        Ok(Tensor::from_vec(
            self.shape()[0],
            self.shape()[1],
            self.read_values()?,
        )?)
    }
}

impl Tensor {
    pub fn to_nd(&self) -> Result<NdTensor, NdTensorError> {
        NdTensor::from_tensor(self)
    }
    pub fn into_nd(self) -> Result<NdTensor, NdTensorError> {
        NdTensor::from_owned_tensor(self)
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn prepared_pointwise_reuses_layouts_without_retaining_input_values() {
        let first = NdTensor::from_vec(&[2], vec![-2., 3.]).unwrap();
        let rhs = NdTensor::from_vec(&[], vec![0.5]).unwrap();
        let chain = PointwiseChain::new(
            2,
            vec![
                PointwiseStep {
                    op: ElementwiseOp::Multiply,
                    rhs: Some(1),
                },
                PointwiseStep {
                    op: ElementwiseOp::Relu,
                    rhs: None,
                },
                PointwiseStep {
                    op: ElementwiseOp::Add,
                    rhs: Some(0),
                },
            ],
        )
        .unwrap();
        let plan = NdPointwisePlan::new(chain.clone(), &[&first, &rhs]).unwrap();
        for mode in [
            PointwiseExecution::Sequential,
            PointwiseExecution::Batched,
            PointwiseExecution::Fused,
        ] {
            assert_eq!(
                plan.run(&[&first, &rhs], mode)
                    .unwrap()
                    .read_values()
                    .unwrap(),
                vec![-2., 4.5]
            );
            let second = NdTensor::from_vec(&[2], vec![4., -6.]).unwrap();
            assert_eq!(
                plan.run(&[&second, &rhs], mode)
                    .unwrap()
                    .read_values()
                    .unwrap(),
                vec![6., -6.]
            );
            assert!(plan.run(&[&rhs, &first], mode).is_err());
            assert!(plan.run(&[&first], mode).is_err());
        }
        assert!(NdPointwisePlan::new(chain, &[&rhs, &first]).is_err());
        let huge = NdTensor::from_vec(&[2], vec![-f32::MAX; 2]).unwrap();
        let two = NdTensor::from_vec(&[], vec![2.]).unwrap();
        assert!(plan.run(&[&huge, &two], PointwiseExecution::Fused).is_err());
    }

    fn host(tensor: &NdTensor) -> &Tensor {
        match &tensor.storage {
            Storage::Host { tensor, .. } => tensor,
            #[cfg(feature = "wgpu_dense")]
            Storage::Wgpu(_) => panic!("host tensor"),
        }
    }

    #[test]
    fn host_views_broadcasts_empty_and_scalar_agree() {
        let root = NdTensor::from_vec(&[2, 3, 4], (0..24).map(|i| i as f32).collect()).unwrap();
        let view = root.permute(&[1, 0, 2]).unwrap().narrow(0, 1, 2).unwrap();
        let bias = NdTensor::from_vec(&[4], vec![0., 1., 2., 3.]).unwrap();
        let output = view.add(&bias).unwrap();
        assert_eq!(output.shape(), &[2, 2, 4]);
        assert_eq!(
            output.read_values().unwrap(),
            vec![4., 6., 8., 10., 16., 18., 20., 22., 8., 10., 12., 14., 20., 22., 24., 26.]
        );
        assert!(view.reshape(&[4, 4]).is_err());
        assert_eq!(
            view.contiguous()
                .unwrap()
                .reshape(&[4, 4])
                .unwrap()
                .to_tensor()
                .unwrap()
                .shape(),
            (4, 4)
        );
        let empty = NdTensor::from_vec(&[0, 4], vec![]).unwrap();
        assert!(empty.add(&bias).unwrap().read_values().unwrap().is_empty());
        let scalar = NdTensor::from_vec(&[], vec![2.]).unwrap();
        assert_eq!(
            scalar.broadcast_to(&[2, 1]).unwrap().read_values().unwrap(),
            [2., 2.]
        );
        assert!(root
            .add(&NdTensor::from_vec(&[5], vec![0.; 5]).unwrap())
            .is_err());
        assert!(NdTensor::from_vec(&[1], vec![f32::INFINITY]).is_err());
        assert!(NdTensor::from_vec(&[1], vec![f32::MAX])
            .unwrap()
            .mul(&scalar)
            .is_err());
    }

    #[test]
    fn protected_host_imports_reuse_unique_storage_and_detach_mutable_aliases() {
        let owned = Tensor::from_vec(2, 2, vec![1., 2., 3., 4.]).unwrap();
        let pointer = owned.data().as_ptr();
        let nd = owned.into_nd().unwrap();
        let tensor = host(&nd);
        assert!(tensor.is_snapshot());
        assert_eq!(tensor.data().as_ptr(), pointer);
        let shared = NdTensor::from_tensor(tensor).unwrap();
        let shared_tensor = host(&shared);
        assert_eq!(shared_tensor.data().as_ptr(), pointer);
        let mut source = Tensor::from_vec(2, 2, vec![1., 2., 3., 4.]).unwrap();
        let imported = source.to_nd().unwrap();
        source.data_mut().fill(99.);
        assert_eq!(imported.read_values().unwrap(), [1., 2., 3., 4.]);
        let col = Tensor::from_vec(2, 2, vec![1., 2., 3., 4.])
            .unwrap()
            .to_layout(Layout::ColMajor)
            .unwrap();
        let nd = col.into_nd().unwrap();
        assert_eq!(nd.layout().strides(), &[1, 2]);
        assert_eq!(nd.to_tensor().unwrap().data(), &[1., 2., 3., 4.]);
    }

    #[test]
    fn preexisting_writable_dlpack_alias_cannot_mutate_nd_storage() {
        use crate::dlpack::{DlpackCopyPolicy, DlpackExportOptions, DlpackProtocol};
        let source = Tensor::from_vec(1, 2, vec![1., 2.]).unwrap();
        let pointer = source.data().as_ptr().cast_mut();
        let exported = source
            .export_dlpack(DlpackExportOptions {
                protocol: DlpackProtocol::Versioned,
                copy: DlpackCopyPolicy::Never,
            })
            .unwrap();
        let nd = source.into_nd().unwrap();
        assert_ne!(host(&nd).data().as_ptr(), pointer.cast_const());
        // The writable export keeps the old allocation alive; no Rust slice of
        // that allocation remains. Model a sequential external producer write.
        unsafe {
            pointer.write(99.);
        }
        assert_eq!(nd.read_values().unwrap(), [1., 2.]);
        drop(exported);
    }
}
