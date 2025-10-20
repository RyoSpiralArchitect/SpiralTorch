// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::models::utils::{conv_output_hw, pool_output_hw};
use st_nn::io;
use st_nn::layers::activation::Relu;
use st_nn::layers::conv::{AvgPool2d, Conv2d, MaxPool2d};
use st_nn::layers::normalization::LayerNorm;
use st_nn::layers::sequential::Sequential;
use st_nn::module::{Module, Parameter};
use st_nn::PureResult;
use st_tensor::{Tensor, TensorError};

#[derive(Clone, Debug)]
pub struct ResNetConfig {
    pub input_channels: usize,
    pub input_hw: (usize, usize),
    pub stage_channels: Vec<usize>,
    pub block_depths: Vec<usize>,
    pub stem_kernel: (usize, usize),
    pub stem_stride: (usize, usize),
    pub stem_padding: (usize, usize),
    pub use_max_pool: bool,
    pub global_pool: bool,
    pub curvature: f32,
    pub epsilon: f32,
}

impl Default for ResNetConfig {
    fn default() -> Self {
        Self {
            input_channels: 3,
            input_hw: (224, 224),
            stage_channels: vec![64, 128, 256, 512],
            block_depths: vec![2, 2, 2, 2],
            stem_kernel: (7, 7),
            stem_stride: (2, 2),
            stem_padding: (3, 3),
            use_max_pool: true,
            global_pool: true,
            curvature: -1.0,
            epsilon: 1.0e-5,
        }
    }
}

#[derive(Debug)]
struct ResidualBlock {
    main: Sequential,
    skip: Option<Sequential>,
    activation: Relu,
    output_hw: (usize, usize),
    out_channels: usize,
}

impl ResidualBlock {
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: &str,
        in_channels: usize,
        out_channels: usize,
        stride: (usize, usize),
        input_hw: (usize, usize),
        curvature: f32,
        epsilon: f32,
    ) -> PureResult<Self> {
        if out_channels == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: 1,
                cols: out_channels,
            });
        }
        if curvature >= 0.0 || !curvature.is_finite() {
            return Err(TensorError::NonHyperbolicCurvature { curvature });
        }
        if epsilon <= 0.0 || !epsilon.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "layernorm_epsilon",
                value: epsilon,
            });
        }
        let mut main = Sequential::new();
        let conv1_hw = conv_output_hw(input_hw, (3, 3), stride, (1, 1), (1, 1))?;
        main.push(Conv2d::new(
            format!("{name}.conv1"),
            in_channels,
            out_channels,
            (3, 3),
            stride,
            (1, 1),
            (1, 1),
            input_hw,
        )?);
        main.push(LayerNorm::new(
            format!("{name}.ln1"),
            out_channels * conv1_hw.0 * conv1_hw.1,
            curvature,
            epsilon,
        )?);
        main.push(Relu::new());
        let conv2_hw = conv_output_hw(conv1_hw, (3, 3), (1, 1), (1, 1), (1, 1))?;
        main.push(Conv2d::new(
            format!("{name}.conv2"),
            out_channels,
            out_channels,
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            conv1_hw,
        )?);
        main.push(LayerNorm::new(
            format!("{name}.ln2"),
            out_channels * conv2_hw.0 * conv2_hw.1,
            curvature,
            epsilon,
        )?);
        let activation = Relu::new();
        let skip = if stride != (1, 1) || in_channels != out_channels {
            let mut seq = Sequential::new();
            seq.push(Conv2d::new(
                format!("{name}.downsample"),
                in_channels,
                out_channels,
                (1, 1),
                stride,
                (0, 0),
                (1, 1),
                input_hw,
            )?);
            seq.push(LayerNorm::new(
                format!("{name}.down_ln"),
                out_channels * conv2_hw.0 * conv2_hw.1,
                curvature,
                epsilon,
            )?);
            Some(seq)
        } else {
            None
        };
        Ok(Self {
            main,
            skip,
            activation,
            output_hw: conv2_hw,
            out_channels,
        })
    }

    fn output_hw(&self) -> (usize, usize) {
        self.output_hw
    }

    fn output_channels(&self) -> usize {
        self.out_channels
    }
}

impl Module for ResidualBlock {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let main = self.main.forward(input)?;
        let skip = if let Some(skip) = &self.skip {
            skip.forward(input)?
        } else {
            input.clone()
        };
        let summed = main.add(&skip)?;
        self.activation.forward(&summed)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        let main_out = self.main.forward(input)?;
        let skip_out = if let Some(skip) = &self.skip {
            skip.forward(input)?
        } else {
            input.clone()
        };
        let summed = main_out.add(&skip_out)?;
        let grad = self.activation.backward(&summed, grad_output)?;
        let grad_skip = if let Some(skip) = &mut self.skip {
            skip.backward(input, &grad)?
        } else {
            grad.clone()
        };
        let grad_main = self.main.backward(input, &grad)?;
        grad_main.add(&grad_skip)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.main.visit_parameters(visitor)?;
        if let Some(skip) = &self.skip {
            skip.visit_parameters(visitor)?;
        }
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.main.visit_parameters_mut(visitor)?;
        if let Some(skip) = &mut self.skip {
            skip.visit_parameters_mut(visitor)?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct ResNetBackbone {
    stem: Sequential,
    stem_pool: Option<MaxPool2d>,
    blocks: Vec<ResidualBlock>,
    global_pool: Option<AvgPool2d>,
    stage_shapes: Vec<(usize, (usize, usize))>,
    output_channels: usize,
    output_hw: (usize, usize),
}

impl ResNetBackbone {
    pub fn new(config: ResNetConfig) -> PureResult<Self> {
        if config.stage_channels.is_empty() {
            return Err(TensorError::EmptyInput("resnet_stage_channels"));
        }
        if config.stage_channels.len() != config.block_depths.len() {
            return Err(TensorError::InvalidDimensions {
                rows: config.stage_channels.len(),
                cols: config.block_depths.len(),
            });
        }
        if config.input_channels == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: 1,
                cols: config.input_channels,
            });
        }
        if config.curvature >= 0.0 || !config.curvature.is_finite() {
            return Err(TensorError::NonHyperbolicCurvature {
                curvature: config.curvature,
            });
        }
        if config.epsilon <= 0.0 || !config.epsilon.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "resnet_layernorm_epsilon",
                value: config.epsilon,
            });
        }

        let mut stem = Sequential::new();
        let stem_hw = conv_output_hw(
            config.input_hw,
            config.stem_kernel,
            config.stem_stride,
            config.stem_padding,
            (1, 1),
        )?;
        let stem_channels = config.stage_channels[0];
        stem.push(Conv2d::new(
            "resnet.stem",
            config.input_channels,
            stem_channels,
            config.stem_kernel,
            config.stem_stride,
            config.stem_padding,
            (1, 1),
            config.input_hw,
        )?);
        stem.push(LayerNorm::new(
            "resnet.stem_ln",
            stem_channels * stem_hw.0 * stem_hw.1,
            config.curvature,
            config.epsilon,
        )?);
        stem.push(Relu::new());

        let mut stem_pool = None;
        let mut current_hw = stem_hw;
        if config.use_max_pool {
            let pool = MaxPool2d::new(stem_channels, (3, 3), (2, 2), (1, 1), current_hw)?;
            current_hw = pool_output_hw(current_hw, (3, 3), (2, 2), (1, 1))?;
            stem_pool = Some(pool);
        }

        let mut blocks = Vec::new();
        let mut stage_shapes = Vec::new();
        let mut in_channels = stem_channels;
        for (stage_idx, (&channels, &depth)) in config
            .stage_channels
            .iter()
            .zip(config.block_depths.iter())
            .enumerate()
        {
            for block_idx in 0..depth {
                let stride = if stage_idx > 0 && block_idx == 0 {
                    (2, 2)
                } else {
                    (1, 1)
                };
                let block = ResidualBlock::new(
                    &format!("resnet.stage{stage_idx}.block{block_idx}"),
                    in_channels,
                    channels,
                    stride,
                    current_hw,
                    config.curvature,
                    config.epsilon,
                )?;
                current_hw = block.output_hw();
                in_channels = block.output_channels();
                blocks.push(block);
            }
            stage_shapes.push((in_channels, current_hw));
        }

        let mut global_pool = None;
        let mut output_hw = current_hw;
        if config.global_pool {
            let pool = AvgPool2d::new(in_channels, current_hw, (1, 1), (0, 0), current_hw)?;
            output_hw = (1, 1);
            global_pool = Some(pool);
        }

        Ok(Self {
            stem,
            stem_pool,
            blocks,
            global_pool,
            stage_shapes,
            output_channels: in_channels,
            output_hw,
        })
    }

    pub fn output_channels(&self) -> usize {
        self.output_channels
    }

    pub fn output_hw(&self) -> (usize, usize) {
        self.output_hw
    }

    pub fn output_features(&self) -> usize {
        self.output_channels * self.output_hw.0 * self.output_hw.1
    }

    pub fn stage_shapes(&self) -> &[(usize, (usize, usize))] {
        &self.stage_shapes
    }

    pub fn load_weights_json<P: AsRef<std::path::Path>>(&mut self, path: P) -> PureResult<()> {
        io::load_json(self, path)
    }

    pub fn load_weights_bincode<P: AsRef<std::path::Path>>(&mut self, path: P) -> PureResult<()> {
        io::load_bincode(self, path)
    }
}

impl Module for ResNetBackbone {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let stem_out = self.stem.forward(input)?;
        let mut activ = if let Some(pool) = &self.stem_pool {
            pool.forward(&stem_out)?
        } else {
            stem_out
        };
        for block in &self.blocks {
            activ = block.forward(&activ)?;
        }
        if let Some(pool) = &self.global_pool {
            activ = pool.forward(&activ)?;
        }
        Ok(activ)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        let stem_out = self.stem.forward(input)?;
        let mut activations = Vec::with_capacity(self.blocks.len());
        let mut current = if let Some(pool) = &self.stem_pool {
            pool.forward(&stem_out)?
        } else {
            stem_out.clone()
        };
        for block in &self.blocks {
            activations.push(current.clone());
            current = block.forward(&current)?;
        }
        let mut grad = if let Some(pool) = &mut self.global_pool {
            pool.backward(&current, grad_output)?
        } else {
            grad_output.clone()
        };
        for (block, activ) in self.blocks.iter_mut().zip(activations.iter()).rev() {
            grad = block.backward(activ, &grad)?;
        }
        if let Some(pool) = &mut self.stem_pool {
            grad = pool.backward(&stem_out, &grad)?;
        }
        self.stem.backward(input, &grad)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.stem.visit_parameters(visitor)?;
        if let Some(pool) = &self.stem_pool {
            pool.visit_parameters(visitor)?;
        }
        for block in &self.blocks {
            block.visit_parameters(visitor)?;
        }
        if let Some(pool) = &self.global_pool {
            pool.visit_parameters(visitor)?;
        }
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.stem.visit_parameters_mut(visitor)?;
        if let Some(pool) = &mut self.stem_pool {
            pool.visit_parameters_mut(visitor)?;
        }
        for block in &mut self.blocks {
            block.visit_parameters_mut(visitor)?;
        }
        if let Some(pool) = &mut self.global_pool {
            pool.visit_parameters_mut(visitor)?;
        }
        Ok(())
    }
}
