//! Reusable VJP over logical input slots. This is not an implicit autograd tape:
//! callers supply the immutable forward inputs and an explicit output cotangent.
use super::*;
use st_kernel_contracts::pointwise::BroadcastAdjoint;

#[derive(Debug)]
pub struct NdPointwiseVjpPlan {
    forward: NdPointwisePlan,
    #[cfg(feature = "wgpu_dense")]
    gpu: Option<st_backend_wgpu::resident_tensor::pointwise::vjp::PointwiseVjpPlan>,
}

impl NdPointwiseVjpPlan {
    pub fn new(chain: PointwiseChain, inputs: &[&NdTensor]) -> Result<Self, NdTensorError> {
        NdPointwisePlan::new(chain, inputs)?.into_vjp()
    }
    pub(super) fn from_forward(forward: NdPointwisePlan) -> Result<Self, NdTensorError> {
        #[cfg(feature = "wgpu_dense")]
        let (forward, gpu) = {
            let mut forward = forward;
            let gpu = forward
                .gpu
                .take()
                .map(st_backend_wgpu::resident_tensor::pointwise::vjp::PointwiseVjpPlan::new)
                .transpose()?;
            (forward, gpu)
        };
        Ok(Self {
            forward,
            #[cfg(feature = "wgpu_dense")]
            gpu,
        })
    }
    pub fn forward(
        &self,
        inputs: &[&NdTensor],
        execution: PointwiseExecution,
    ) -> Result<NdTensor, NdTensorError> {
        #[cfg(feature = "wgpu_dense")]
        if let Some(gpu) = &self.gpu {
            let inputs: Result<Vec<_>, _> = inputs
                .iter()
                .map(|t| t.as_wgpu().ok_or(NdTensorError::DeviceMismatch))
                .collect();
            return Ok(NdTensor::from_wgpu(gpu.forward().run(&inputs?, execution)?));
        }
        self.forward.run(inputs, execution)
    }
    /// Return exact vector-Jacobian products in each original input's logical
    /// shape. Broadcast axes are summed, never implicitly averaged. Distinct
    /// slots remain distinct even when the caller passes the same tensor twice.
    pub fn vjp(
        &self,
        inputs: &[&NdTensor],
        cotangent: &NdTensor,
    ) -> Result<Vec<NdTensor>, NdTensorError> {
        if inputs.len() != self.forward.layouts.len() {
            return Err(PointwiseError::Operands.into());
        }
        for (input, layout) in inputs.iter().zip(&self.forward.layouts) {
            if input.layout() != layout {
                return Err(PointwiseError::LayoutMismatch.into());
            }
        }
        let shape = self.forward.layouts[0].shape();
        if cotangent.shape() != shape {
            return Err(PointwiseError::LayoutMismatch.into());
        }
        #[cfg(feature = "wgpu_dense")]
        if let Some(gpu) = &self.gpu {
            let inputs: Result<Vec<_>, _> = inputs
                .iter()
                .map(|t| t.as_wgpu().ok_or(NdTensorError::DeviceMismatch))
                .collect();
            let seed = cotangent.as_wgpu().ok_or(NdTensorError::DeviceMismatch)?;
            return Ok(gpu
                .run(&inputs?, seed)?
                .into_iter()
                .map(NdTensor::from_wgpu)
                .collect());
        }
        if inputs.iter().any(|t| t.is_wgpu()) || cotangent.is_wgpu() {
            return Err(NdTensorError::DeviceMismatch);
        }
        let n = self.forward.layouts[0].len();
        let count = inputs.len();
        n.checked_mul(count).ok_or(NdLayoutError::Overflow)?;
        let views: Result<Vec<_>, _> = inputs
            .iter()
            .map(|input| input.broadcast_to(shape))
            .collect();
        let views = views?;
        let host = |input: &NdTensor, index: usize| -> f32 {
            match &input.storage {
                Storage::Host { tensor, layout } => {
                    tensor.data()[layout.storage_index(index).unwrap()]
                }
                #[cfg(feature = "wgpu_dense")]
                Storage::Wgpu(_) => unreachable!("device checked before evaluation"),
            }
        };
        let mut contributions: Vec<_> = (0..count).map(|_| Vec::with_capacity(n)).collect();
        let mut values = vec![0.; count];
        for i in 0..n {
            for (slot, input) in views.iter().enumerate() {
                values[slot] = host(input, i);
            }
            let gradients = self.forward.chain.vjp_scalar(&values, host(cotangent, i))?;
            for (slot, gradient) in contributions.iter_mut().zip(gradients) {
                slot.push(gradient);
            }
        }
        inputs
            .iter()
            .zip(contributions)
            .map(|(input, values)| {
                let map = BroadcastAdjoint::new(shape, input.shape())?;
                NdTensor::from_vec(input.shape(), map.reduce(&values)?)
            })
            .collect()
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    #[test]
    fn host_vjp_unbroadcasts_views_and_accumulates_residuals() {
        let x = NdTensor::from_vec(&[2, 3], vec![1., 2., 3., 4., 5., 6.])
            .unwrap()
            .permute(&[1, 0])
            .unwrap();
        let gain = NdTensor::from_vec(&[2], vec![2., 3.]).unwrap();
        let seed = NdTensor::from_vec(&[3, 2], vec![0.5; 6]).unwrap();
        let plan = NdPointwiseVjpPlan::new(
            PointwiseChain::new(
                2,
                vec![
                    PointwiseStep {
                        op: ElementwiseOp::Multiply,
                        rhs: Some(1),
                    },
                    PointwiseStep {
                        op: ElementwiseOp::Add,
                        rhs: Some(0),
                    },
                    PointwiseStep {
                        op: ElementwiseOp::Relu,
                        rhs: None,
                    },
                ],
            )
            .unwrap(),
            &[&x, &gain],
        )
        .unwrap();
        let gradients = plan.vjp(&[&x, &gain], &seed).unwrap();
        assert_eq!(
            gradients[0].read_values().unwrap(),
            vec![1.5, 2., 1.5, 2., 1.5, 2.]
        );
        assert_eq!(gradients[1].read_values().unwrap(), vec![3., 7.5]);
        assert!(plan.vjp(&[&x, &gain], &gain).is_err());
        assert!(plan.vjp(&[&x], &seed).is_err());
    }
}
