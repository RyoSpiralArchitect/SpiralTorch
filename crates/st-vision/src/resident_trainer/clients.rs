//! Shared construction for native and browser clients. Input ownership and
//! device attachment are established here, not independently in each binding.
use super::*;
use crate::{TransformDispatcher, TransformPipeline};

const MAX_CONFIG_BYTES: usize = 64 * 1024;

/// Initial settings only. Resume uses the checkpoint's clocks, order and RNGs.
/// Seeds are canonical decimal strings in JSON, preserving every u64 in JS.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResidentVisionTrainerConfig {
    pub model: ConvNeXtConfig,
    pub num_classes: usize,
    pub batch_size: usize,
    #[serde(with = "seed_string")]
    pub model_seed: u64,
    #[serde(with = "seed_string")]
    pub shuffle_seed: u64,
    pub shuffle: bool,
    pub learning_rate: ResidentLearningRate,
}

impl Default for ResidentVisionTrainerConfig {
    fn default() -> Self {
        Self {
            model: ConvNeXtConfig::default(),
            num_classes: 10,
            batch_size: 16,
            model_seed: 0,
            shuffle_seed: 0,
            shuffle: true,
            learning_rate: ResidentLearningRate::Constant { rate: 0.01 },
        }
    }
}

impl ResidentVisionTrainerConfig {
    pub fn to_json(&self) -> Result<String, InferenceError> {
        self.learning_rate.validate(0)?;
        let json = serde_json::to_string(self)?;
        if json.len() > MAX_CONFIG_BYTES {
            return Err(invalid("vision trainer configuration size limit"));
        }
        Ok(json)
    }

    pub fn from_json(json: &str) -> Result<Self, InferenceError> {
        if json.len() > MAX_CONFIG_BYTES {
            return Err(invalid("vision trainer configuration size limit"));
        }
        let config: Self = serde_json::from_str(json)?;
        config.learning_rate.validate(0)?;
        Ok(config)
    }
}

mod seed_string {
    use serde::{de::Error, Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(value: &u64, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&value.to_string())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<u64, D::Error> {
        let text = String::deserialize(deserializer)?;
        let value = text.parse::<u64>().map_err(D::Error::custom)?;
        if value.to_string() != text {
            return Err(D::Error::custom(
                "seed must be a canonical decimal u64 string",
            ));
        }
        Ok(value)
    }
}

fn attach_pipeline<D: VisionDataset>(
    loader: DataLoader<D>,
    device: &TensorDevice,
    pipeline: Option<TransformPipeline>,
) -> Result<DataLoader<D>, InferenceError> {
    match pipeline {
        Some(mut pipeline) => {
            let dispatcher = TransformDispatcher::from_runtime(device.runtime()).map_err(|e| {
                st_tensor::TensorError::BackendFailure {
                    backend: "wgpu",
                    message: e.to_string(),
                }
            })?;
            pipeline.set_gpu_dispatcher(dispatcher);
            Ok(loader.with_pipeline(pipeline))
        }
        None => Ok(loader),
    }
}

impl<D: VisionDataset> ResidentVisionTrainer<D> {
    /// Own a fresh input stream. A supplied pipeline is a template: clients
    /// clone it before this call, and its dispatcher is bound to this device.
    pub fn from_dataset(
        config: &ResidentVisionTrainerConfig,
        device: TensorDevice,
        dataset: Arc<D>,
        pipeline: Option<TransformPipeline>,
        dataset_sha256: &str,
    ) -> Result<Self, InferenceError> {
        config.learning_rate.validate(0)?;
        let mut loader = DataLoader::new(dataset, config.batch_size, Some(config.shuffle_seed))?;
        loader.enable_shuffle(config.shuffle);
        let loader = attach_pipeline(loader, &device, pipeline)?;
        // Validate the input before allocating the model's parameters.
        let state = ResidentVisionTrainingState {
            epoch: 0,
            accepted_updates: 0,
            rejected_updates: 0,
            learning_rate: config.learning_rate.clone(),
        };
        state.validate(0, &loader.checkpoint(dataset_sha256)?)?;
        let model =
            ConvNeXtClassifier::new(config.model.clone(), config.num_classes, config.model_seed)?;
        Self::new(
            &model,
            device,
            loader,
            dataset_sha256,
            config.learning_rate.clone(),
        )
    }

    /// Recreate the input owner on a local device. The supplied transform
    /// configuration and dataset identity must match the checkpoint exactly.
    pub fn from_dataset_checkpoint(
        device: TensorDevice,
        dataset: Arc<D>,
        pipeline: Option<TransformPipeline>,
        dataset_sha256: &str,
        checkpoint: &VisionTrainingCheckpoint,
    ) -> Result<Self, InferenceError> {
        checkpoint.validate()?;
        let loader = DataLoader::new(dataset, checkpoint.input.batch_size(), Some(0))?;
        let loader = attach_pipeline(loader, &device, pipeline)?;
        Self::from_checkpoint(device, loader, dataset_sha256, checkpoint)
    }
}
