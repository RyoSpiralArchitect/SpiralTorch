//! Channel-preserving spatial averaging for flattened NCHW Modules.
use crate::module::{Module, Parameter};
use crate::{PureResult, Tensor, TensorError};

#[cfg(test)]
mod tests;

#[cfg(feature = "wgpu")]
use crate::resident::InferenceError;
#[cfg(feature = "wgpu")]
use st_backend_wgpu::resident_tensor::{ResidentTensor, TensorDevice};

#[derive(Debug)]
pub struct GlobalAveragePool2d {
    channels: usize,
    hw: (usize, usize),
    area: usize,
    #[cfg(feature = "wgpu")]
    resident: std::cell::RefCell<Option<ResidentGlobalAveragePool2d>>,
}

impl GlobalAveragePool2d {
    pub fn input_chw(&self) -> [usize; 3] {
        [self.channels, self.hw.0, self.hw.1]
    }
    pub fn new(channels: usize, hw: (usize, usize)) -> PureResult<Self> {
        let area = hw.0.checked_mul(hw.1).ok_or(TensorError::InvalidValue {
            label: "global_pool_size",
        })?;
        if channels == 0 || area == 0 || channels.checked_mul(area).is_none() {
            return Err(TensorError::InvalidValue {
                label: "global_pool_size",
            });
        }
        Ok(Self {
            channels,
            hw,
            area,
            #[cfg(feature = "wgpu")]
            resident: Default::default(),
        })
    }

    fn validate(&self, input: &Tensor) -> PureResult<()> {
        if input.shape().1 != self.channels * self.area {
            return Err(TensorError::ShapeMismatch {
                left: input.shape(),
                right: (input.shape().0, self.channels * self.area),
            });
        }
        if let Some(&value) = input.data().iter().find(|v| !v.is_finite()) {
            return Err(TensorError::NonFiniteValue {
                label: "global_pool_input",
                value,
            });
        }
        Ok(())
    }

    #[cfg(feature = "wgpu")]
    pub fn compile_resident(
        &self,
        device: TensorDevice,
    ) -> Result<ResidentGlobalAveragePool2d, InferenceError> {
        crate::resident::require_uncommitted_route()?;
        let weights = device.upload(
            &[self.channels, self.hw.0, self.hw.1],
            &vec![1.0 / self.area as f32; self.channels * self.area],
        )?;
        let bias = device.upload(&[self.channels], &vec![0.0; self.channels])?;
        let zero = device.upload(&[1], &[0.0])?;
        let scale = device.upload(&[1], &[1.0 / self.area as f32])?;
        Ok(ResidentGlobalAveragePool2d {
            channels: self.channels,
            hw: self.hw,
            weights,
            bias,
            zero,
            scale,
            backward: Default::default(),
        })
    }
}

impl Module for GlobalAveragePool2d {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        self.validate(input)?;
        let input = input.to_layout(st_tensor::Layout::RowMajor)?;
        let scale = 1.0 / self.area as f32;
        let mut values = Vec::with_capacity(input.shape().0 * self.channels);
        for spatial in input.data().chunks_exact(self.area) {
            let mut mean = 0.0f32;
            for value in spatial {
                // Scale before summing, matching the resident fixed spatial filter.
                mean += value * scale;
                if !mean.is_finite() {
                    return Err(TensorError::NonFiniteValue {
                        label: "global_pool_sum",
                        value: mean,
                    });
                }
            }
            values.push(mean);
        }
        Tensor::from_vec(input.shape().0, self.channels, values)
    }

    fn backward(&mut self, input: &Tensor, gradient: &Tensor) -> PureResult<Tensor> {
        self.validate(input)?;
        if gradient.shape() != (input.shape().0, self.channels) {
            return Err(TensorError::ShapeMismatch {
                left: gradient.shape(),
                right: (input.shape().0, self.channels),
            });
        }
        let gradient = gradient.to_layout(st_tensor::Layout::RowMajor)?;
        let mut values = Vec::with_capacity(input.data().len());
        for &value in gradient.data() {
            if !value.is_finite() {
                return Err(TensorError::NonFiniteValue {
                    label: "global_pool_gradient",
                    value,
                });
            }
            values.resize(values.len() + self.area, value * (1.0 / self.area as f32));
        }
        Tensor::from_vec(input.shape().0, self.channels * self.area, values)
    }

    fn visit_parameters(&self, _: &mut dyn FnMut(&Parameter) -> PureResult<()>) -> PureResult<()> {
        Ok(())
    }
    fn visit_parameters_mut(
        &mut self,
        _: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        Ok(())
    }

    #[cfg(feature = "wgpu")]
    fn forward_resident(&self, input: &ResidentTensor) -> Result<ResidentTensor, InferenceError> {
        let mut cached = self.resident.borrow_mut();
        let compatible = cached.as_ref().is_some_and(|p| {
            p.weights
                .device()
                .runtime()
                .context()
                .shares_handles_with(input.device().runtime().context())
        });
        if !compatible {
            *cached = Some(self.compile_resident(input.device().clone())?);
        }
        cached.as_ref().unwrap().forward(input)
    }

    #[cfg(feature = "wgpu")]
    fn clear_resident_forward_cache(&self) {
        self.resident.borrow_mut().take();
    }
}

/// Parameter-free pooling using guarded depthwise forward and a fused broadcast VJP.
/// The fixed filter is compiled once; no values are mapped to the host.
#[cfg(feature = "wgpu")]
#[derive(Debug)]
pub struct ResidentGlobalAveragePool2d {
    channels: usize,
    hw: (usize, usize),
    weights: ResidentTensor,
    bias: ResidentTensor,
    zero: ResidentTensor,
    scale: ResidentTensor,
    backward: std::cell::RefCell<Option<PoolBackwardPlan>>,
}

#[cfg(feature = "wgpu")]
#[derive(Debug)]
struct PoolBackwardPlan {
    layouts: Vec<st_tensor::NdLayout>,
    plan: st_backend_wgpu::resident_tensor::pointwise::PointwisePlan,
}

#[cfg(feature = "wgpu")]
impl ResidentGlobalAveragePool2d {
    fn validate(&self, input: &ResidentTensor) -> Result<usize, InferenceError> {
        crate::resident::require_uncommitted_route()?;
        let shape = input.layout().shape();
        if shape.len() != 4 || shape[0] == 0 || shape[1..] != [self.channels, self.hw.0, self.hw.1]
        {
            return Err(InferenceError::InvalidLayout);
        }
        Ok(shape[0])
    }
    pub fn forward(&self, input: &ResidentTensor) -> Result<ResidentTensor, InferenceError> {
        let batch = self.validate(input)?;
        Ok(input
            .depthwise_conv2d(&self.weights, &self.bias, (1, 1), (0, 0), (1, 1))?
            .reshape(&[batch, self.channels])?)
    }
    pub fn backward(
        &self,
        input: &ResidentTensor,
        cotangent: &ResidentTensor,
    ) -> Result<ResidentTensor, InferenceError> {
        let batch = self.validate(input)?;
        if cotangent.layout().shape() != [batch, self.channels] {
            return Err(InferenceError::InvalidLayout);
        }
        let seed = cotangent
            .contiguous()?
            .reshape(&[batch, self.channels, 1, 1])?;
        use st_backend_wgpu::resident_tensor::{
            pointwise::PointwisePlan, TensorError as DeviceError,
        };
        use st_kernel_contracts::{
            elementwise::ElementwiseOp,
            pointwise::{PointwiseChain, PointwiseExecution, PointwiseStep},
        };
        let inputs = [input, &self.zero, &seed, &self.scale];
        let layouts: Vec<_> = inputs.iter().map(|v| v.layout().clone()).collect();
        let mut plan = self.backward.borrow_mut();
        if plan.as_ref().is_none_or(|saved| saved.layouts != layouts) {
            // (input * 0 + cotangent) / area preserves input validity without
            // computing meaningless (and potentially overflowing) filter VJPs.
            let chain = PointwiseChain::new(
                4,
                vec![
                    PointwiseStep {
                        op: ElementwiseOp::Multiply,
                        rhs: Some(1),
                    },
                    PointwiseStep {
                        op: ElementwiseOp::Add,
                        rhs: Some(2),
                    },
                    PointwiseStep {
                        op: ElementwiseOp::Multiply,
                        rhs: Some(3),
                    },
                ],
            )
            .map_err(DeviceError::from)?;
            let compiled =
                PointwisePlan::new(self.weights.device().clone(), chain, layouts.clone())?;
            *plan = Some(PoolBackwardPlan {
                layouts,
                plan: compiled,
            });
        }
        Ok(plan
            .as_ref()
            .unwrap()
            .plan
            .run(&inputs, PointwiseExecution::Fused)?)
    }
}
