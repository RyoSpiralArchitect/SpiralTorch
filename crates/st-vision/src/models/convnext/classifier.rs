//! A trainable classifier over the same ConvNeXt backbone exposed to inference.
use super::*;
use rand::{Rng, SeedableRng};
use st_nn::layers::global_average_pool::GlobalAveragePool2d;
mod checkpoint;
pub use checkpoint::ConvNeXtClassifierCheckpoint;
#[cfg(test)]
mod tests;

#[derive(Debug)]
pub struct ConvNeXtClassifier {
    pub(super) backbone: ConvNeXtBackbone,
    pub(super) pool: GlobalAveragePool2d,
    pub(super) head: Linear,
    classes: usize,
}

fn initialize(
    module: &mut impl Module,
    rng: &mut rand::rngs::StdRng,
    fan_in: usize,
    fan_out: usize,
) -> PureResult<()> {
    let bound = (6.0 / (fan_in as f64 + fan_out as f64)).sqrt() as f32;
    module.visit_parameters_mut(&mut |p| {
        if p.name().ends_with("::weight") {
            for value in p.value_mut().data_mut() {
                *value = rng.gen_range(-bound..bound);
            }
        }
        Ok(())
    })
}

impl ConvNeXtBackbone {
    /// Size-scaled, seeded learned weights, without changing process RNG state.
    /// Affine normalization keeps its ordinary unit gain and zero bias.
    pub fn new_seeded(config: ConvNeXtConfig, seed: u64) -> PureResult<Self> {
        let mut model = Self::new(config)?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let c = &model.config;
        let area = c.patch_size.0 * c.patch_size.1;
        initialize(
            &mut model.stem,
            &mut rng,
            c.input_channels * area,
            c.stage_dims[0] * area,
        )?;
        for (i, stage) in model.stages.iter_mut().enumerate() {
            let channels = c.stage_dims[i];
            for block in &mut stage.blocks {
                initialize(&mut block.depthwise, &mut rng, 49, 49)?;
                initialize(&mut block.mlp1, &mut rng, channels, 4 * channels)?;
                initialize(&mut block.mlp2, &mut rng, 4 * channels, channels)?;
            }
            if let Some(downsample) = &mut stage.downsample {
                initialize(downsample, &mut rng, channels * 4, c.stage_dims[i + 1] * 4)?;
            }
        }
        Ok(model)
    }
}

impl ConvNeXtClassifier {
    pub fn new(config: ConvNeXtConfig, classes: usize, seed: u64) -> PureResult<Self> {
        if classes == 0 {
            return Err(TensorError::InvalidValue {
                label: "convnext_classes",
            });
        }
        // Validate the head dimensions before allocating the backbone.
        config.parameter_budget()?;
        let channels = *config.stage_dims.last().unwrap();
        let head = Linear::new_xavier(
            "convnext.classifier",
            channels,
            classes,
            seed ^ 0x434c_4153_5349_4659,
        )?;
        let backbone = ConvNeXtBackbone::new_seeded(config, seed)?;
        let (channels, hw) = backbone.output_shape();
        Ok(Self {
            backbone,
            pool: GlobalAveragePool2d::new(channels, hw)?,
            head,
            classes,
        })
    }

    pub fn config(&self) -> &ConvNeXtConfig {
        self.backbone.config()
    }
    pub fn num_classes(&self) -> usize {
        self.classes
    }
    pub fn backbone(&self) -> &ConvNeXtBackbone {
        &self.backbone
    }

    pub fn features(&self, input: &Tensor) -> PureResult<Tensor> {
        self.pool.forward(&self.backbone.forward(input)?)
    }

    pub fn stem_features(&self, input: &Tensor) -> PureResult<Tensor> {
        self.backbone.stem.forward(input)
    }
}

impl Module for ConvNeXtClassifier {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        self.head.forward(&self.features(input)?)
    }

    fn backward(&mut self, input: &Tensor, gradient: &Tensor) -> PureResult<Tensor> {
        let features = self.backbone.forward(input)?;
        let pooled = self.pool.forward(&features)?;
        let seed = self.head.backward(&pooled, gradient)?;
        let seed = self.pool.backward(&features, &seed)?;
        self.backbone.backward(input, &seed)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.backbone.visit_parameters(visitor)?;
        self.head.visit_parameters(visitor)
    }
    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.backbone.visit_parameters_mut(visitor)?;
        self.head.visit_parameters_mut(visitor)
    }

    #[cfg(feature = "wgpu")]
    fn forward_resident(&self, input: &ResidentTensor) -> Result<ResidentTensor, InferenceError> {
        let features = self.backbone.forward_resident(input)?;
        let batch = input.layout().shape()[0];
        let (channels, (height, width)) = self.backbone.output_shape();
        let pooled = self
            .pool
            .forward_resident(&features.reshape(&[batch, channels, height, width])?)?;
        self.head.forward_resident(&pooled)
    }

    #[cfg(feature = "wgpu")]
    fn clear_resident_forward_cache(&self) {
        self.backbone.clear_resident_forward_cache();
        self.pool.clear_resident_forward_cache();
        self.head.clear_resident_forward_cache();
    }
}
