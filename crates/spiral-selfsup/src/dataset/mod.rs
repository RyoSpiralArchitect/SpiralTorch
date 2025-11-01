use crate::contrastive::{info_nce_loss_tensor, TensorInfoNCEResult};
use crate::{ObjectiveError, Result};
use rand::{rngs::StdRng, Rng};
use serde::Deserialize;
use spiral_config::determinism;
use st_tensor::Tensor;
use st_vision::datasets::MultiViewDatasetAdapter;
use st_vision::transforms::{ImageTensor, TransformOperation, TransformPipeline};
use std::fs;
use std::path::Path;

/// Pair of augmented views sampled from a [`MultiViewDatasetAdapter`].
#[derive(Clone, Debug, PartialEq)]
pub struct ViewPair {
    pub anchor: ImageTensor,
    pub positive: ImageTensor,
}

fn checked_spatial_volume(height: usize, width: usize) -> Result<usize> {
    height.checked_mul(width).ok_or_else(|| {
        ObjectiveError::InvalidArgument(format!(
            "image dimensions overflow when computing volume: {height}x{width}"
        ))
    })
}

fn colors_to_image(colors: &Tensor, height: usize, width: usize) -> Result<ImageTensor> {
    if height == 0 || width == 0 {
        return Err(ObjectiveError::InvalidArgument(
            "image dimensions must be > 0".to_string(),
        ));
    }
    let (rows, cols) = colors.shape();
    let spatial = checked_spatial_volume(height, width)?;
    if cols != 3 {
        return Err(ObjectiveError::Shape(format!(
            "expected RGB colors with 3 columns, found {cols}"
        )));
    }
    if rows != spatial {
        return Err(ObjectiveError::Shape(format!(
            "color rows ({rows}) do not match target image size {height}x{width}"
        )));
    }
    let mut data = vec![0.0f32; cols * spatial];
    let source = colors.data();
    for (index, pixel) in source.chunks_exact(cols).enumerate() {
        let y = index / width;
        let x = index % width;
        for channel in 0..cols {
            let offset = ((channel * height) + y) * width + x;
            data[offset] = pixel[channel];
        }
    }
    ImageTensor::new(3, height, width, data).map_err(ObjectiveError::from)
}

fn clone_operations(pipeline: Option<&TransformPipeline>) -> Vec<TransformOperation> {
    pipeline
        .map(|p| p.operations().iter().cloned().collect())
        .unwrap_or_default()
}

fn reseeded_pipeline(ops: &[TransformOperation], seed: u64) -> TransformPipeline {
    let mut pipeline = TransformPipeline::with_seed(seed);
    for op in ops {
        pipeline.add(op.clone());
    }
    pipeline
}

/// Utility capable of sampling augmented view pairs from a multi-view dataset.
#[derive(Debug)]
pub struct ViewPairSampler<'a> {
    dataset: &'a MultiViewDatasetAdapter,
    image_height: usize,
    image_width: usize,
    rng: StdRng,
    operations: Vec<TransformOperation>,
}

impl<'a> ViewPairSampler<'a> {
    pub fn new(
        dataset: &'a MultiViewDatasetAdapter,
        image_height: usize,
        image_width: usize,
        seed: Option<u64>,
        pipeline: Option<&TransformPipeline>,
    ) -> Result<Self> {
        if dataset.num_frames() == 0 {
            return Err(ObjectiveError::InvalidArgument(
                "dataset must contain at least one frame".to_string(),
            ));
        }
        if image_height == 0 || image_width == 0 {
            return Err(ObjectiveError::InvalidArgument(
                "image dimensions must be > 0".to_string(),
            ));
        }
        let label = format!(
            "spiral-selfsup/view_pair_sampler:{}:{}:{}",
            dataset.num_frames(),
            image_height,
            image_width
        );
        let rng = determinism::rng_from_optional(seed, &label);
        Ok(Self {
            dataset,
            image_height,
            image_width,
            rng,
            operations: clone_operations(pipeline),
        })
    }

    fn apply_pipeline(&mut self, image: &mut ImageTensor) -> Result<()> {
        if self.operations.is_empty() {
            return Ok(());
        }
        let seed = self.rng.gen();
        let mut pipeline = reseeded_pipeline(&self.operations, seed);
        pipeline.apply(image).map_err(ObjectiveError::from)
    }

    fn sample_frame_index(&mut self) -> usize {
        let frame_count = self.dataset.num_frames();
        if frame_count <= 1 {
            0
        } else {
            self.rng.gen_range(0..frame_count)
        }
    }

    pub fn sample_pair(&mut self) -> Result<ViewPair> {
        let frame_index = self.sample_frame_index();
        let frame = &self.dataset.frames()[frame_index];
        let base = colors_to_image(&frame.colors, self.image_height, self.image_width)?;
        let mut anchor = base.clone();
        let mut positive = base;
        self.apply_pipeline(&mut anchor)?;
        self.apply_pipeline(&mut positive)?;
        Ok(ViewPair { anchor, positive })
    }
}

/// Configuration used by [`InfoNCEBatcher`] to control sampling and loss hyperparameters.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct InfoNCEBatchConfig {
    pub batch_size: usize,
    pub image_height: usize,
    pub image_width: usize,
    pub temperature: f32,
    #[serde(default = "InfoNCEBatchConfig::default_normalize")]
    pub normalize: bool,
    #[serde(default)]
    pub seed: Option<u64>,
}

impl InfoNCEBatchConfig {
    fn default_normalize() -> bool {
        true
    }

    fn validate(&self) -> Result<()> {
        if self.batch_size == 0 {
            return Err(ObjectiveError::InvalidArgument(
                "batch_size must be > 0".to_string(),
            ));
        }
        if self.image_height == 0 || self.image_width == 0 {
            return Err(ObjectiveError::InvalidArgument(
                "image dimensions must be > 0".to_string(),
            ));
        }
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            return Err(ObjectiveError::InvalidArgument(format!(
                "temperature must be > 0, got {}",
                self.temperature
            )));
        }
        Ok(())
    }

    pub fn from_toml_str(config: &str) -> Result<Self> {
        let parsed: InfoNCEBatchConfig = toml::from_str(config).map_err(|err| {
            ObjectiveError::InvalidArgument(format!("failed to parse InfoNCE batch config: {err}"))
        })?;
        parsed.validate()?;
        Ok(parsed)
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let data = fs::read_to_string(&path).map_err(|err| {
            ObjectiveError::InvalidArgument(format!(
                "failed to read InfoNCE batch config {}: {err}",
                path.as_ref().display()
            ))
        })?;
        Self::from_toml_str(&data)
    }
}

/// Tensor batches ready to be consumed by [`info_nce_loss_tensor`].
#[derive(Debug, Clone, PartialEq)]
pub struct InfoNCEBatch {
    pub anchors: Tensor,
    pub positives: Tensor,
}

/// Helper that bridges dataset sampling, augmentation, batching and loss computation.
#[derive(Debug)]
pub struct InfoNCEBatcher<'a> {
    sampler: ViewPairSampler<'a>,
    config: InfoNCEBatchConfig,
}

impl<'a> InfoNCEBatcher<'a> {
    pub fn new(
        dataset: &'a MultiViewDatasetAdapter,
        pipeline: Option<&TransformPipeline>,
        config: InfoNCEBatchConfig,
    ) -> Result<Self> {
        config.validate()?;
        let sampler = ViewPairSampler::new(
            dataset,
            config.image_height,
            config.image_width,
            config.seed,
            pipeline,
        )?;
        Ok(Self { sampler, config })
    }

    pub fn sample_batch(&mut self) -> Result<InfoNCEBatch> {
        let mut anchors_data = Vec::new();
        let mut positives_data = Vec::new();
        let mut feature_dim: Option<usize> = None;
        for _ in 0..self.config.batch_size {
            let pair = self.sampler.sample_pair()?;
            let anchor_len = pair.anchor.as_slice().len();
            let positive_len = pair.positive.as_slice().len();
            if anchor_len != positive_len {
                return Err(ObjectiveError::Shape(format!(
                    "anchor/positive feature mismatch (anchor={}, positive={})",
                    anchor_len, positive_len
                )));
            }
            if let Some(dim) = feature_dim {
                if dim != anchor_len {
                    return Err(ObjectiveError::Shape(format!(
                        "inconsistent feature dims across batch: expected {dim}, got {anchor_len}"
                    )));
                }
            } else {
                feature_dim = Some(anchor_len);
            }
            anchors_data.extend_from_slice(pair.anchor.as_slice());
            positives_data.extend_from_slice(pair.positive.as_slice());
        }
        let feature_dim = feature_dim
            .ok_or_else(|| ObjectiveError::InvalidArgument("batch_size must be > 0".to_string()))?;
        let anchors = Tensor::from_vec(self.config.batch_size, feature_dim, anchors_data)
            .map_err(ObjectiveError::from)?;
        let positives = Tensor::from_vec(self.config.batch_size, feature_dim, positives_data)
            .map_err(ObjectiveError::from)?;
        Ok(InfoNCEBatch { anchors, positives })
    }

    pub fn next_loss(&mut self) -> Result<TensorInfoNCEResult> {
        let batch = self.sample_batch()?;
        info_nce_loss_tensor(
            &batch.anchors,
            &batch.positives,
            self.config.temperature,
            self.config.normalize,
        )
    }
}
