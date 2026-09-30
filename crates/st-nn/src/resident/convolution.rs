//! Stateless convolution geometry; values are supplied by a resident parameter owner.
use super::*;
use st_backend_wgpu::resident_tensor::{ResidentTensor, TensorError as DeviceTensorError};

#[derive(Clone, Debug)]
pub struct ResidentConvolutionSpec {
    pub(crate) input_chw: [usize; 3],
    pub(crate) weight_shape: Vec<usize>,
    pub(crate) output_channels: usize,
    pub(crate) stride: (usize, usize),
    pub(crate) padding: (usize, usize),
    pub(crate) dilation: (usize, usize),
    pub(crate) depthwise: bool,
}

impl ResidentConvolutionSpec {
    pub fn input_shape(&self, batch: usize) -> [usize; 4] {
        [
            batch,
            self.input_chw[0],
            self.input_chw[1],
            self.input_chw[2],
        ]
    }

    pub fn weight_shape(&self) -> &[usize] {
        &self.weight_shape
    }
    pub fn bias_shape(&self) -> [usize; 1] {
        [self.output_channels]
    }

    fn validate(
        &self,
        input: &ResidentTensor,
        weight: &ResidentTensor,
    ) -> Result<(), InferenceError> {
        require_uncommitted_route()?;
        let shape = input.layout().shape();
        if shape.len() != 4
            || shape[1..] != self.input_chw
            || weight.layout().shape() != self.weight_shape
        {
            return Err(
                DeviceTensorError::ConvolutionShape("module convolution geometry differs").into(),
            );
        }
        Ok(())
    }

    pub fn forward(
        &self,
        input: &ResidentTensor,
        weight: &ResidentTensor,
        bias: &ResidentTensor,
    ) -> Result<ResidentTensor, InferenceError> {
        self.validate(input, weight)?;
        if bias.layout().shape() != self.bias_shape() {
            return Err(
                DeviceTensorError::ConvolutionShape("module convolution bias differs").into(),
            );
        }
        Ok(if self.depthwise {
            input.depthwise_conv2d(weight, bias, self.stride, self.padding, self.dilation)?
        } else {
            input.conv2d(weight, bias, self.stride, self.padding, self.dilation)?
        })
    }

    pub fn vjp(
        &self,
        input: &ResidentTensor,
        weight: &ResidentTensor,
        cotangent: &ResidentTensor,
    ) -> Result<[ResidentTensor; 3], InferenceError> {
        self.validate(input, weight)?;
        Ok(if self.depthwise {
            input.depthwise_conv2d_vjp(
                weight,
                cotangent,
                self.stride,
                self.padding,
                self.dilation,
            )?
        } else {
            input.conv2d_vjp(weight, cotangent, self.stride, self.padding, self.dilation)?
        })
    }
}
