// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::path::Path;

use st_nn::io;
use st_nn::layers::conv::{Conv2d, DepthwiseConv2d};
use st_nn::layers::gelu::Gelu;
use st_nn::layers::linear::Linear;
use st_nn::layers::normalization::LayerNorm;
use st_nn::module::{Module, Parameter};
use st_nn::PureResult;
use st_tensor::{Tensor, TensorError};

use crate::models::resnet::conv_output_hw;

fn conv_to_tokens(input: &Tensor, channels: usize, hw: (usize, usize)) -> PureResult<Tensor> {
    let (batch, cols) = input.shape();
    let expected = channels * hw.0 * hw.1;
    if cols != expected {
        return Err(TensorError::ShapeMismatch {
            left: (batch, cols),
            right: (batch, expected),
        });
    }
    let tokens_per_batch = hw.0 * hw.1;
    let mut data = Vec::with_capacity(batch * tokens_per_batch * channels);
    for b in 0..batch {
        let row = &input.data()[b * cols..(b + 1) * cols];
        for token in 0..tokens_per_batch {
            for c in 0..channels {
                let offset = c * tokens_per_batch + token;
                data.push(row[offset]);
            }
        }
    }
    Tensor::from_vec(batch * tokens_per_batch, channels, data)
}

fn tokens_to_conv(
    tokens: &Tensor,
    batch: usize,
    channels: usize,
    hw: (usize, usize),
) -> PureResult<Tensor> {
    let tokens_per_batch = hw.0 * hw.1;
    if tokens.shape().0 != batch * tokens_per_batch || tokens.shape().1 != channels {
        return Err(TensorError::ShapeMismatch {
            left: tokens.shape(),
            right: (batch * tokens_per_batch, channels),
        });
    }
    let mut data = vec![0.0f32; batch * channels * tokens_per_batch];
    for b in 0..batch {
        for token in 0..tokens_per_batch {
            for c in 0..channels {
                let src = (b * tokens_per_batch + token) * channels + c;
                let dst = b * channels * tokens_per_batch + c * tokens_per_batch + token;
                data[dst] = tokens.data()[src];
            }
        }
    }
    Tensor::from_vec(batch, channels * tokens_per_batch, data)
}

#[derive(Clone, Debug)]
pub struct ConvNeXtConfig {
    pub input_channels: usize,
    pub input_hw: (usize, usize),
    pub stage_dims: Vec<usize>,
    pub stage_depths: Vec<usize>,
    pub patch_size: (usize, usize),
    pub curvature: f32,
    pub epsilon: f32,
}

impl Default for ConvNeXtConfig {
    fn default() -> Self {
        Self {
            input_channels: 3,
            input_hw: (224, 224),
            stage_dims: vec![96, 192, 384, 768],
            stage_depths: vec![3, 3, 9, 3],
            patch_size: (4, 4),
            curvature: -1.0,
            epsilon: 1e-6,
        }
    }
}

/// One ConvNeXt residual block with host training and an explicit resident forward.
#[derive(Debug)]
pub struct ConvNeXtBlock {
    depthwise: DepthwiseConv2d,
    norm: LayerNorm,
    mlp1: Linear,
    activation: Gelu,
    mlp2: Linear,
    channels: usize,
    hw: (usize, usize),
}

impl ConvNeXtBlock {
    /// Builds a block over NCHW features with the given spatial dimensions.
    pub fn new(
        name: &str,
        channels: usize,
        input_hw: (usize, usize),
        curvature: f32,
        epsilon: f32,
    ) -> PureResult<Self> {
        let depthwise = DepthwiseConv2d::new(
            format!("{name}.dw"),
            channels,
            (7, 7),
            (1, 1),
            (3, 3),
            (1, 1),
            input_hw,
        )?;
        let norm = LayerNorm::new(format!("{name}.ln"), channels, curvature, epsilon)?;
        let mlp1 = Linear::new(format!("{name}.fc1"), channels, channels * 4)?;
        let activation = Gelu::new();
        let mlp2 = Linear::new(format!("{name}.fc2"), channels * 4, channels)?;
        Ok(Self {
            depthwise,
            norm,
            mlp1,
            activation,
            mlp2,
            channels,
            hw: input_hw,
        })
    }
}

impl Module for ConvNeXtBlock {
    #[cfg(feature = "wgpu")]
    fn forward_resident(
        &self,
        input: &st_backend_wgpu::resident_tensor::ResidentTensor,
    ) -> Result<st_backend_wgpu::resident_tensor::ResidentTensor, st_nn::resident::InferenceError>
    {
        if matches!(input.layout().shape(), [0, _, _, _]) {
            return Err(
                st_backend_wgpu::resident_tensor::TensorError::ConvolutionShape(
                    "empty ConvNeXt batch",
                )
                .into(),
            );
        }
        let dw = self.depthwise.forward_resident(input)?;
        let shape = dw.layout().shape();
        let (batch, height, width) = (shape[0], shape[2], shape[3]);
        let tokens = dw
            .permute(&[0, 2, 3, 1])?
            .contiguous()?
            .reshape(&[dw.layout().len() / self.channels, self.channels])?;
        let normed = self.norm.forward_resident(&tokens)?;
        let hidden = self.mlp1.forward_resident(&normed)?;
        let activated = self.activation.forward_resident(&hidden)?;
        let projected = self.mlp2.forward_resident(&activated)?;
        let conv_layout = projected
            .reshape(&[batch, height, width, self.channels])?
            .permute(&[0, 3, 1, 2])?;
        Ok(conv_layout.add(input)?)
    }

    #[cfg(feature = "wgpu")]
    fn forward_resident_snapshot(
        &self,
        input: &st_backend_wgpu::resident_tensor::ResidentTensor,
    ) -> Result<st_backend_wgpu::resident_tensor::TensorReadback, st_nn::resident::InferenceError>
    {
        Ok(self.forward_resident(input)?.snapshot()?)
    }

    #[cfg(feature = "wgpu")]
    fn clear_resident_forward_cache(&self) {
        self.depthwise.clear_resident_forward_cache();
        self.norm.clear_resident_forward_cache();
        self.mlp1.clear_resident_forward_cache();
        self.mlp2.clear_resident_forward_cache();
    }

    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let dw = self.depthwise.forward(input)?;
        let tokens = conv_to_tokens(&dw, self.channels, self.hw)?;
        let normed = self.norm.forward(&tokens)?;
        let hidden = self.mlp1.forward(&normed)?;
        let activated = self.activation.forward(&hidden)?;
        let projected = self.mlp2.forward(&activated)?;
        let conv_layout = tokens_to_conv(&projected, input.shape().0, self.channels, self.hw)?;
        conv_layout.add(input)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        if grad_output.shape() != input.shape() {
            return Err(TensorError::ShapeMismatch {
                left: grad_output.shape(),
                right: input.shape(),
            });
        }
        let dw = self.depthwise.forward(input)?;
        let tokens = conv_to_tokens(&dw, self.channels, self.hw)?;
        let normed = self.norm.forward(&tokens)?;
        let hidden = self.mlp1.forward(&normed)?;
        let activated = self.activation.forward(&hidden)?;

        let grad_projected = conv_to_tokens(grad_output, self.channels, self.hw)?;
        let grad_activated = self.mlp2.backward(&activated, &grad_projected)?;
        let grad_hidden = self.activation.backward(&hidden, &grad_activated)?;
        let grad_normed = self.mlp1.backward(&normed, &grad_hidden)?;
        let grad_tokens = self.norm.backward(&tokens, &grad_normed)?;
        let grad_dw = tokens_to_conv(&grad_tokens, input.shape().0, self.channels, self.hw)?;
        let grad_main = self.depthwise.backward(input, &grad_dw)?;
        grad_main.add(grad_output)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.depthwise.visit_parameters(visitor)?;
        self.norm.visit_parameters(visitor)?;
        self.mlp1.visit_parameters(visitor)?;
        self.mlp2.visit_parameters(visitor)
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.depthwise.visit_parameters_mut(visitor)?;
        self.norm.visit_parameters_mut(visitor)?;
        self.mlp1.visit_parameters_mut(visitor)?;
        self.mlp2.visit_parameters_mut(visitor)
    }
}

#[derive(Debug)]
struct ConvNeXtStage {
    blocks: Vec<ConvNeXtBlock>,
    downsample: Option<Conv2d>,
}

impl ConvNeXtStage {
    fn new(
        name: &str,
        channels: usize,
        depth: usize,
        input_hw: (usize, usize),
        next_channels: Option<usize>,
        curvature: f32,
        epsilon: f32,
    ) -> PureResult<(Self, (usize, usize), usize)> {
        let mut blocks = Vec::with_capacity(depth);
        for idx in 0..depth {
            blocks.push(ConvNeXtBlock::new(
                &format!("{name}.block{idx}"),
                channels,
                input_hw,
                curvature,
                epsilon,
            )?);
        }
        let (downsample, next_hw, next_channels) = if let Some(next) = next_channels {
            let conv = Conv2d::new(
                format!("{name}.downsample"),
                channels,
                next,
                (2, 2),
                (2, 2),
                (0, 0),
                (1, 1),
                input_hw,
            )?;
            let hw = conv_output_hw(input_hw, (2, 2), (2, 2), (0, 0), (1, 1))?;
            (Some(conv), hw, next)
        } else {
            (None, input_hw, channels)
        };
        Ok((Self { blocks, downsample }, next_hw, next_channels))
    }
}

impl Module for ConvNeXtStage {
    #[cfg(feature = "wgpu")]
    fn forward_resident(
        &self,
        input: &st_backend_wgpu::resident_tensor::ResidentTensor,
    ) -> Result<st_backend_wgpu::resident_tensor::ResidentTensor, st_nn::resident::InferenceError>
    {
        let mut activ = input.clone();
        for block in &self.blocks {
            activ = block.forward_resident(&activ)?;
        }
        if let Some(down) = &self.downsample {
            activ = down.forward_resident(&activ)?;
        }
        Ok(activ)
    }

    #[cfg(feature = "wgpu")]
    fn forward_resident_snapshot(
        &self,
        input: &st_backend_wgpu::resident_tensor::ResidentTensor,
    ) -> Result<st_backend_wgpu::resident_tensor::TensorReadback, st_nn::resident::InferenceError>
    {
        Ok(self.forward_resident(input)?.snapshot()?)
    }

    #[cfg(feature = "wgpu")]
    fn clear_resident_forward_cache(&self) {
        for block in &self.blocks {
            block.clear_resident_forward_cache();
        }
        if let Some(down) = &self.downsample {
            down.clear_resident_forward_cache();
        }
    }

    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let mut activ = input.clone();
        for block in &self.blocks {
            activ = block.forward(&activ)?;
        }
        if let Some(down) = &self.downsample {
            activ = down.forward(&activ)?;
        }
        Ok(activ)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        let mut block_inputs = Vec::with_capacity(self.blocks.len());
        let mut activ = input.clone();
        for block in &self.blocks {
            block_inputs.push(activ.clone());
            activ = block.forward(&activ)?;
        }
        let mut grad = if let Some(down) = &mut self.downsample {
            down.backward(&activ, grad_output)?
        } else {
            grad_output.clone()
        };
        for (block, block_input) in self
            .blocks
            .iter_mut()
            .rev()
            .zip(block_inputs.into_iter().rev())
        {
            grad = block.backward(&block_input, &grad)?;
        }
        Ok(grad)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        for block in &self.blocks {
            block.visit_parameters(visitor)?;
        }
        if let Some(down) = &self.downsample {
            down.visit_parameters(visitor)?;
        }
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        for block in &mut self.blocks {
            block.visit_parameters_mut(visitor)?;
        }
        if let Some(down) = &mut self.downsample {
            down.visit_parameters_mut(visitor)?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct ConvNeXtBackbone {
    stem: Conv2d,
    stages: Vec<ConvNeXtStage>,
    final_norm: LayerNorm,
    output_channels: usize,
    output_hw: (usize, usize),
}

impl ConvNeXtBackbone {
    pub fn new(config: ConvNeXtConfig) -> PureResult<Self> {
        if config.stage_dims.is_empty() {
            return Err(TensorError::InvalidValue {
                label: "convnext_stage_dims",
            });
        }
        if config.stage_dims.len() != config.stage_depths.len() {
            return Err(TensorError::InvalidDimensions {
                rows: config.stage_dims.len(),
                cols: config.stage_depths.len(),
            });
        }
        if config.curvature >= 0.0 || !config.curvature.is_finite() {
            return Err(TensorError::NonHyperbolicCurvature {
                curvature: config.curvature,
            });
        }
        if config.epsilon <= 0.0 || !config.epsilon.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "convnext_layernorm_epsilon",
                value: config.epsilon,
            });
        }
        let stem = Conv2d::new(
            "convnext.stem",
            config.input_channels,
            config.stage_dims[0],
            config.patch_size,
            config.patch_size,
            (0, 0),
            (1, 1),
            config.input_hw,
        )?;
        let mut current_hw = conv_output_hw(
            config.input_hw,
            config.patch_size,
            config.patch_size,
            (0, 0),
            (1, 1),
        )?;
        let mut stages = Vec::with_capacity(config.stage_dims.len());
        let mut current_channels = config.stage_dims[0];
        for (idx, (&channels, &depth)) in config
            .stage_dims
            .iter()
            .zip(config.stage_depths.iter())
            .enumerate()
        {
            let next_channels = config.stage_dims.get(idx + 1).copied();
            let (stage, next_hw, next_ch) = ConvNeXtStage::new(
                &format!("convnext.stage{idx}"),
                channels,
                depth,
                current_hw,
                next_channels,
                config.curvature,
                config.epsilon,
            )?;
            current_hw = next_hw;
            current_channels = next_ch;
            stages.push(stage);
        }
        let final_norm = LayerNorm::new(
            "convnext.final_norm",
            current_channels * current_hw.0 * current_hw.1,
            config.curvature,
            config.epsilon,
        )?;
        Ok(Self {
            stem,
            stages,
            final_norm,
            output_channels: current_channels,
            output_hw: current_hw,
        })
    }

    pub fn load_weights_json<P: AsRef<Path>>(&mut self, path: P) -> PureResult<()> {
        io::load_json(self, path)
    }

    pub fn load_weights_bincode<P: AsRef<Path>>(&mut self, path: P) -> PureResult<()> {
        io::load_bincode(self, path)
    }

    pub fn output_shape(&self) -> (usize, (usize, usize)) {
        (self.output_channels, self.output_hw)
    }
}

impl Module for ConvNeXtBackbone {
    #[cfg(feature = "wgpu")]
    fn forward_resident(
        &self,
        input: &st_backend_wgpu::resident_tensor::ResidentTensor,
    ) -> Result<st_backend_wgpu::resident_tensor::ResidentTensor, st_nn::resident::InferenceError>
    {
        if matches!(input.layout().shape(), [0, _, _, _]) {
            return Err(
                st_backend_wgpu::resident_tensor::TensorError::ConvolutionShape(
                    "empty ConvNeXt batch",
                )
                .into(),
            );
        }
        let mut activ = self.stem.forward_resident(input)?;
        for stage in &self.stages {
            activ = stage.forward_resident(&activ)?;
        }
        let [batch, channels, height, width] = activ.layout().shape() else {
            return Err(
                st_backend_wgpu::resident_tensor::TensorError::ConvolutionShape(
                    "unexpected ConvNeXt stage output",
                )
                .into(),
            );
        };
        if (*channels, *height, *width)
            != (self.output_channels, self.output_hw.0, self.output_hw.1)
        {
            return Err(
                st_backend_wgpu::resident_tensor::TensorError::ConvolutionShape(
                    "unexpected ConvNeXt stage output",
                )
                .into(),
            );
        }
        let flattened = self.output_channels * self.output_hw.0 * self.output_hw.1;
        let features = activ.contiguous()?.reshape(&[*batch, flattened])?;
        self.final_norm.forward_resident(&features)
    }

    #[cfg(feature = "wgpu")]
    fn forward_resident_snapshot(
        &self,
        input: &st_backend_wgpu::resident_tensor::ResidentTensor,
    ) -> Result<st_backend_wgpu::resident_tensor::TensorReadback, st_nn::resident::InferenceError>
    {
        Ok(self.forward_resident(input)?.snapshot()?)
    }

    #[cfg(feature = "wgpu")]
    fn clear_resident_forward_cache(&self) {
        self.stem.clear_resident_forward_cache();
        for stage in &self.stages {
            stage.clear_resident_forward_cache();
        }
        self.final_norm.clear_resident_forward_cache();
    }

    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let mut activ = self.stem.forward(input)?;
        for stage in &self.stages {
            activ = stage.forward(&activ)?;
        }
        self.final_norm.forward(&activ)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        let stem_output = self.stem.forward(input)?;
        let mut stage_inputs = Vec::with_capacity(self.stages.len());
        let mut activ = stem_output;
        for stage in &self.stages {
            stage_inputs.push(activ.clone());
            activ = stage.forward(&activ)?;
        }
        let mut grad = self.final_norm.backward(&activ, grad_output)?;
        for (stage, stage_input) in self
            .stages
            .iter_mut()
            .rev()
            .zip(stage_inputs.into_iter().rev())
        {
            grad = stage.backward(&stage_input, &grad)?;
        }
        self.stem.backward(input, &grad)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.stem.visit_parameters(visitor)?;
        for stage in &self.stages {
            stage.visit_parameters(visitor)?;
        }
        self.final_norm.visit_parameters(visitor)
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.stem.visit_parameters_mut(visitor)?;
        for stage in &mut self.stages {
            stage.visit_parameters_mut(visitor)?;
        }
        self.final_norm.visit_parameters_mut(visitor)
    }
}

#[cfg(all(test, feature = "wgpu", not(target_arch = "wasm32")))]
mod resident_tests {
    use super::*;
    use st_backend_wgpu::resident_tensor::{ResidentTensor, TensorDevice};
    use st_backend_wgpu::runtime::ensure_default_runtime_blocking;
    use st_core::backend::device_caps::DeviceCaps;
    use st_nn::execution::{push_backend_policy, BackendPolicy};

    #[test]
    fn resident_block_matches_host_and_refreshes_changed_parameters() {
        let runtime = match ensure_default_runtime_blocking("vision.convnext_block_resident") {
            Ok((runtime, _)) => runtime,
            Err(error)
                if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() == Ok("1") =>
            {
                panic!("ConvNeXt resident block requires a live WGPU adapter: {error}");
            }
            Err(_) => return,
        };
        let device = TensorDevice::new(runtime).unwrap();
        let mut block = ConvNeXtBlock::new("vision.test", 3, (4, 5), -1.0, 1e-6).unwrap();
        let host_input = Tensor::from_fn(2, 3 * 4 * 5, |row, col| {
            ((row * 37 + col * 13) % 83) as f32 / 83.0 - 0.5
        })
        .unwrap();
        let input = device.upload(&[2, 3, 4, 5], host_input.data()).unwrap();
        let compare = |block: &ConvNeXtBlock, input: &ResidentTensor| {
            let expected = {
                let _policy =
                    push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
                block.forward(&host_input).unwrap()
            };
            let resident = block.forward_resident(input).unwrap();
            assert_eq!(resident.layout().shape(), &[2, 3, 4, 5]);
            let actual = resident.snapshot().unwrap().read().unwrap();
            for (index, (&reference, &value)) in expected.data().iter().zip(&actual).enumerate() {
                assert!(
                    (reference - value).abs() <= 1e-3 * (1.0 + reference.abs()),
                    "ConvNeXt block value {index}: cpu={reference}, resident={value}"
                );
            }
        };
        compare(&block, &input);
        compare(&block, &input);
        assert_eq!(block.norm.resident_forward_stats().unwrap().cache_hits, 1);
        assert_eq!(block.mlp1.resident_forward_stats().unwrap().cache_hits, 1);

        let mut transposed = vec![0.0; host_input.data().len()];
        for batch in 0..2 {
            for channel in 0..3 {
                for y in 0..4 {
                    for x in 0..5 {
                        transposed[((batch * 3 + channel) * 5 + x) * 4 + y] =
                            host_input.data()[((batch * 3 + channel) * 4 + y) * 5 + x];
                    }
                }
            }
        }
        let strided = device
            .upload(&[2, 3, 5, 4], &transposed)
            .unwrap()
            .permute(&[0, 1, 3, 2])
            .unwrap();
        assert!(!strided.layout().is_contiguous());
        compare(&block, &strided);

        block
            .visit_parameters_mut(&mut |parameter| {
                if parameter.name().contains(".dw::weight")
                    || parameter.name().contains(".fc1::weight")
                {
                    parameter.value_mut().data_mut()[0] += 0.2;
                }
                Ok(())
            })
            .unwrap();
        compare(&block, &input);
        assert_eq!(block.mlp1.resident_forward_stats().unwrap().compilations, 2);
        block.clear_resident_forward_cache();
        compare(&block, &input);
        let empty = device.upload(&[0, 3, 4, 5], &[]).unwrap();
        assert!(matches!(
            block.forward_resident(&empty),
            Err(st_nn::resident::InferenceError::ResidentTensor(
                st_backend_wgpu::resident_tensor::TensorError::ConvolutionShape(_)
            ))
        ));
    }

    #[test]
    fn resident_backbone_matches_host_through_stem_stages_and_final_norm() {
        let runtime = match ensure_default_runtime_blocking("vision.convnext_backbone_resident") {
            Ok((runtime, _)) => runtime,
            Err(error)
                if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() == Ok("1") =>
            {
                panic!("ConvNeXt resident backbone requires a live WGPU adapter: {error}");
            }
            Err(_) => return,
        };
        let device = TensorDevice::new(runtime).unwrap();
        let mut backbone = ConvNeXtBackbone::new(ConvNeXtConfig {
            input_channels: 2,
            input_hw: (8, 8),
            stage_dims: vec![3, 4],
            stage_depths: vec![1, 1],
            patch_size: (2, 2),
            curvature: -1.0,
            epsilon: 1e-6,
        })
        .unwrap();
        let host_input = Tensor::from_fn(2, 2 * 8 * 8, |row, col| {
            ((row * 41 + col * 17) % 101) as f32 / 101.0 - 0.5
        })
        .unwrap();
        let input = device.upload(&[2, 2, 8, 8], host_input.data()).unwrap();
        let compare = |backbone: &ConvNeXtBackbone, input: &ResidentTensor| {
            let expected = {
                let _policy =
                    push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
                backbone.forward(&host_input).unwrap()
            };
            let resident = backbone.forward_resident(input).unwrap();
            assert_eq!(resident.layout().shape(), &[2, 4 * 2 * 2]);
            let actual = resident.snapshot().unwrap().read().unwrap();
            for (index, (&reference, &value)) in expected.data().iter().zip(&actual).enumerate() {
                assert!(
                    (reference - value).abs() <= 2e-3 * (1.0 + reference.abs()),
                    "ConvNeXt backbone value {index}: cpu={reference}, resident={value}"
                );
            }
            actual
        };
        let baseline = compare(&backbone, &input);
        compare(&backbone, &input);
        assert_eq!(
            backbone
                .final_norm
                .resident_forward_stats()
                .unwrap()
                .cache_hits,
            1
        );
        assert_eq!(
            backbone.stages[0].blocks[0]
                .norm
                .resident_forward_stats()
                .unwrap()
                .cache_hits,
            1
        );

        let mut transposed = vec![0.0; host_input.data().len()];
        for batch in 0..2 {
            for channel in 0..2 {
                for y in 0..8 {
                    for x in 0..8 {
                        transposed[((batch * 2 + channel) * 8 + x) * 8 + y] =
                            host_input.data()[((batch * 2 + channel) * 8 + y) * 8 + x];
                    }
                }
            }
        }
        let strided = device
            .upload(&[2, 2, 8, 8], &transposed)
            .unwrap()
            .permute(&[0, 1, 3, 2])
            .unwrap();
        assert!(!strided.layout().is_contiguous());
        compare(&backbone, &strided);

        backbone
            .visit_parameters_mut(&mut |parameter| {
                if parameter.name().contains("stem::weight")
                    || parameter.name().contains("downsample::weight")
                    || parameter.name().contains("fc1::weight")
                {
                    parameter.value_mut().data_mut()[0] += 0.25;
                }
                if parameter.name().contains("final_norm::bias") {
                    parameter.value_mut().data_mut()[0] += 0.1;
                }
                Ok(())
            })
            .unwrap();
        let changed = compare(&backbone, &input);
        assert!(baseline
            .iter()
            .zip(&changed)
            .any(|(a, b)| (a - b).abs() > 1e-4));
        backbone.clear_resident_forward_cache();
        compare(&backbone, &input);
        let empty = device.upload(&[0, 2, 8, 8], &[]).unwrap();
        assert!(matches!(
            backbone.forward_resident(&empty),
            Err(st_nn::resident::InferenceError::ResidentTensor(
                st_backend_wgpu::resident_tensor::TensorError::ConvolutionShape(_)
            ))
        ));
    }
}
