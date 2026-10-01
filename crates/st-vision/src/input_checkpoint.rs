//! Portable input state, independent of model weights and optimizer clocks.
use super::*;
use serde::{Deserialize, Serialize};

const MAX_JSON_BYTES: usize = 32 * 1024 * 1024;
const MAX_SAMPLES: usize = 4_000_000;
const LOADER_SCHEMA: &str = "spiraltorch.vision.input_checkpoint.v1";
const TRANSFORM_SCHEMA: &str = "spiraltorch.vision.transform_checkpoint.v1";
const RNG_SCHEMA: &str = "rand_0.8_chacha12_v1";
const CONSUMPTION: &str = "consume_on_successful_submission";

fn invalid(label: &'static str) -> TensorError {
    TensorError::InvalidValue { label }
}

fn serialization(error: serde_json::Error) -> TensorError {
    TensorError::SerializationError {
        message: error.to_string(),
    }
}

fn digest_valid(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RngCheckpoint {
    algorithm: String,
    seed: [u8; 32],
    // Decimal strings survive JSON clients that represent all numbers as f64.
    stream: String,
    word_position: String,
}

impl RngCheckpoint {
    fn capture(rng: &ChaCha12Rng) -> Self {
        Self {
            algorithm: RNG_SCHEMA.into(),
            seed: rng.get_seed(),
            stream: rng.get_stream().to_string(),
            word_position: rng.get_word_pos().to_string(),
        }
    }

    fn restore(&self) -> PureResult<ChaCha12Rng> {
        let bad = || invalid("vision_input_rng_state");
        let stream = self.stream.parse::<u64>().map_err(|_| bad())?;
        let position = self.word_position.parse::<u128>().map_err(|_| bad())?;
        if self.algorithm != RNG_SCHEMA
            || position >= 1_u128 << 68
            || stream.to_string() != self.stream
            || position.to_string() != self.word_position
        {
            return Err(bad());
        }
        let mut rng = ChaCha12Rng::from_seed(self.seed);
        rng.set_stream(stream);
        rng.set_word_pos(position);
        Ok(rng)
    }
}

// Float bit patterns preserve signed zero and reject altered configurations
// without depending on a Python/JavaScript float round trip.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
enum TransformConfig {
    Normalize { means: Vec<u32>, stds: Vec<u32> },
    Resize { height: usize, width: usize },
    CenterCrop { height: usize, width: usize },
    RandomHorizontalFlip { probability: u32 },
    ColorJitter { factors: [u32; 4] },
}

impl TransformConfig {
    fn capture(op: &TransformOperation) -> Self {
        match op {
            TransformOperation::Normalize(op) => Self::Normalize {
                means: op.means.iter().map(|x| x.to_bits()).collect(),
                stds: op.stds.iter().map(|x| x.to_bits()).collect(),
            },
            TransformOperation::Resize(op) => Self::Resize {
                height: op.height,
                width: op.width,
            },
            TransformOperation::CenterCrop(op) => Self::CenterCrop {
                height: op.height,
                width: op.width,
            },
            TransformOperation::RandomHorizontalFlip(op) => Self::RandomHorizontalFlip {
                probability: op.probability.to_bits(),
            },
            TransformOperation::ColorJitter(op) => Self::ColorJitter {
                factors: [op.brightness, op.contrast, op.saturation, op.hue].map(f32::to_bits),
            },
        }
    }

    fn validate(&self) -> PureResult<()> {
        match self {
            Self::Normalize { means, stds } => {
                Normalize::new(
                    means.iter().copied().map(f32::from_bits).collect(),
                    stds.iter().copied().map(f32::from_bits).collect(),
                )?;
            }
            Self::Resize { height, width } | Self::CenterCrop { height, width } => {
                if *height > u32::MAX as usize || *width > u32::MAX as usize {
                    return Err(invalid("vision_input_transform_dimensions"));
                }
                Resize::new(*height, *width)?;
            }
            Self::RandomHorizontalFlip { probability } => {
                RandomHorizontalFlip::new(f32::from_bits(*probability))?;
            }
            Self::ColorJitter { factors } => {
                let f = factors.map(f32::from_bits);
                if f.iter().any(|v| !v.is_finite()) {
                    return Err(invalid("vision_input_color_jitter"));
                }
                ColorJitter::new(f[0], f[1], f[2], f[3])?;
            }
        }
        Ok(())
    }
}

/// Transform configuration and RNG only; GPU resources remain caller-owned.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransformPipelineCheckpoint {
    schema: String,
    operations: Vec<TransformConfig>,
    rng: RngCheckpoint,
}

impl TransformPipelineCheckpoint {
    fn validate(&self) -> PureResult<()> {
        if self.schema != TRANSFORM_SCHEMA || self.operations.len() > 1024 {
            return Err(invalid("vision_transform_checkpoint"));
        }
        for operation in &self.operations {
            operation.validate()?;
        }
        self.rng.restore()?;
        Ok(())
    }
}

impl TransformPipeline {
    pub fn checkpoint(&self) -> PureResult<TransformPipelineCheckpoint> {
        let state = TransformPipelineCheckpoint {
            schema: TRANSFORM_SCHEMA.into(),
            operations: self.ops.iter().map(TransformConfig::capture).collect(),
            rng: RngCheckpoint::capture(&self.rng),
        };
        state.validate()?;
        Ok(state)
    }

    /// Restore only into the same transform configuration, retaining the local
    /// dispatcher/caches. Validation failure changes neither RNG nor resources.
    pub fn restore_checkpoint(&mut self, state: &TransformPipelineCheckpoint) -> PureResult<()> {
        state.validate()?;
        if self
            .ops
            .iter()
            .map(TransformConfig::capture)
            .collect::<Vec<_>>()
            != state.operations
        {
            return Err(invalid("vision_transform_checkpoint_configuration"));
        }
        self.rng = state.rng.restore()?;
        Ok(())
    }
}

/// Input cursor/order, both RNGs and transform configuration. No model,
/// schedule, optimizer, prefetched batch or in-flight GPU acceptance is stored.
/// The dataset SHA256 is supplied by the caller; this API never scans a dataset
/// implicitly and cannot authenticate a falsely supplied content identifier.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DataLoaderCheckpoint {
    schema: String,
    dataset_sha256: String,
    dataset_len: usize,
    batch_size: usize,
    position: usize,
    order: Vec<usize>,
    shuffle: bool,
    shuffle_rng: RngCheckpoint,
    pipeline: Option<TransformPipelineCheckpoint>,
    consumption: String,
}

impl DataLoaderCheckpoint {
    fn validate(&self) -> PureResult<()> {
        if self.schema != LOADER_SCHEMA
            || self.consumption != CONSUMPTION
            || !digest_valid(&self.dataset_sha256)
            || self.dataset_len > MAX_SAMPLES
            || self.batch_size == 0
            || self.batch_size > u32::MAX as usize
            || self.position > self.dataset_len
            || self.order.len() != self.dataset_len
            || (self.position != self.dataset_len && !self.position.is_multiple_of(self.batch_size))
        {
            return Err(invalid("vision_input_checkpoint"));
        }
        let mut seen = vec![false; self.dataset_len];
        for (position, &index) in self.order.iter().enumerate() {
            if index >= seen.len() || seen[index] || (!self.shuffle && position != index) {
                return Err(invalid("vision_input_checkpoint_order"));
            }
            seen[index] = true;
        }
        self.shuffle_rng.restore()?;
        if let Some(pipeline) = &self.pipeline {
            pipeline.validate()?;
        }
        Ok(())
    }

    pub fn position(&self) -> usize {
        self.position
    }
    pub fn dataset_len(&self) -> usize {
        self.dataset_len
    }
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
}

macro_rules! json_checkpoint {
    ($kind:ty) => {
        impl $kind {
            pub fn to_json(&self) -> PureResult<String> {
                self.validate()?;
                let json = serde_json::to_string(self).map_err(serialization)?;
                if json.len() > MAX_JSON_BYTES {
                    return Err(invalid("vision_input_checkpoint_size"));
                }
                Ok(json)
            }
            pub fn from_json(json: &str) -> PureResult<Self> {
                if json.len() > MAX_JSON_BYTES {
                    return Err(invalid("vision_input_checkpoint_size"));
                }
                let state: Self = serde_json::from_str(json).map_err(serialization)?;
                state.validate()?;
                Ok(state)
            }
        }
    };
}
json_checkpoint!(TransformPipelineCheckpoint);
json_checkpoint!(DataLoaderCheckpoint);

impl<D: VisionDataset> DataLoader<D> {
    pub fn checkpoint(&self, dataset_sha256: &str) -> PureResult<DataLoaderCheckpoint> {
        let state = DataLoaderCheckpoint {
            schema: LOADER_SCHEMA.into(),
            dataset_sha256: dataset_sha256.into(),
            dataset_len: self.dataset.len(),
            batch_size: self.batch_size,
            position: self.position,
            order: self.order.clone(),
            shuffle: self.shuffle,
            shuffle_rng: RngCheckpoint::capture(&self.shuffle_rng),
            pipeline: self
                .pipeline
                .as_ref()
                .map(TransformPipeline::checkpoint)
                .transpose()?,
            consumption: CONSUMPTION.into(),
        };
        state.validate()?;
        Ok(state)
    }

    /// Restore the input stream into an explicitly supplied dataset and matching
    /// batch/transform configuration. Device dispatchers are retained locally.
    /// Checkpoint at a settled training boundary: submission consumes input even
    /// if a later GPU update is rejected, just as `next_resident_batch` specifies.
    pub fn restore_checkpoint(
        &mut self,
        dataset_sha256: &str,
        state: &DataLoaderCheckpoint,
    ) -> PureResult<()> {
        state.validate()?;
        if state.dataset_sha256 != dataset_sha256
            || state.dataset_len != self.dataset.len()
            || state.batch_size != self.batch_size
        {
            return Err(invalid("vision_input_checkpoint_dataset_or_batch"));
        }
        let rng = state.shuffle_rng.restore()?;
        let mut pipeline = self.pipeline.clone();
        match (&mut pipeline, &state.pipeline) {
            (Some(target), Some(source)) => target.restore_checkpoint(source)?,
            (None, None) => {}
            _ => return Err(invalid("vision_input_checkpoint_pipeline")),
        }
        let order = state.order.clone();
        self.order = order;
        self.position = state.position;
        self.shuffle = state.shuffle;
        self.shuffle_rng = rng;
        self.pipeline = pipeline;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
