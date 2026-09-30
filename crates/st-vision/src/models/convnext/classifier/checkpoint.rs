use super::*;
use crate::models::convnext::checkpoint::{
    restore_parameters, StoredParameter, MAX_JSON_BYTES, MAX_PARAMETER_VALUES,
};
use serde::{Deserialize, Serialize};

const SCHEMA: &str = "spiraltorch.convnext.classifier_plain_sgd_checkpoint.v1";

fn invalid() -> TensorError {
    TensorError::InvalidValue {
        label: "convnext_classifier_checkpoint",
    }
}
fn serialization(error: impl ToString) -> TensorError {
    TensorError::SerializationError {
        message: error.to_string(),
    }
}

/// One backbone and global-average-pool/Linear classifier at a single update clock.
/// The nested backbone payload is not a separately updated parameter owner.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConvNeXtClassifierCheckpoint {
    schema: String,
    backbone: ConvNeXtTrainingCheckpoint,
    classes: usize,
    head: Vec<StoredParameter>,
}

impl ConvNeXtClassifierCheckpoint {
    #[cfg(any(feature = "wgpu", test))]
    pub(in crate::models::convnext) fn new(
        config: ConvNeXtConfig,
        batch: usize,
        revision: u64,
        classes: usize,
        mut parameters: Vec<StoredParameter>,
    ) -> PureResult<Self> {
        let head = parameters.split_off(parameters.len().checked_sub(2).ok_or_else(invalid)?);
        let checkpoint = Self {
            schema: SCHEMA.into(),
            backbone: ConvNeXtTrainingCheckpoint::new(config, batch, revision, parameters)?,
            classes,
            head,
        };
        checkpoint.validate()?;
        Ok(checkpoint)
    }
    pub fn config(&self) -> &ConvNeXtConfig {
        self.backbone.config()
    }
    pub fn batch_size(&self) -> usize {
        self.backbone.batch_size()
    }
    pub fn attempted_updates(&self) -> u64 {
        self.backbone.attempted_updates()
    }
    pub fn num_classes(&self) -> usize {
        self.classes
    }

    fn validate(&self) -> PureResult<()> {
        self.backbone.validate()?;
        if self.schema != SCHEMA || self.classes == 0 || self.head.len() != 2 {
            return Err(invalid());
        }
        let channels = *self.config().stage_dims.last().unwrap();
        let head_values = channels
            .checked_mul(self.classes)
            .and_then(|n| n.checked_add(self.classes))
            .ok_or_else(invalid)?;
        let (_, backbone_values) = self.config().parameter_budget()?;
        if backbone_values
            .checked_add(head_values)
            .filter(|&n| n <= MAX_PARAMETER_VALUES)
            .is_none()
        {
            return Err(invalid());
        }
        for (parameter, (name, shape)) in self.head.iter().zip([
            ("convnext.classifier::weight", [channels, self.classes]),
            ("convnext.classifier::bias", [1, self.classes]),
        ]) {
            if parameter.name != name
                || parameter.shape != shape
                || parameter.values.len() != shape[0] * shape[1]
                || !parameter.values.iter().all(|v| v.is_finite())
            {
                return Err(invalid());
            }
        }
        Ok(())
    }

    pub fn to_json(&self) -> PureResult<String> {
        self.validate()?;
        let json = serde_json::to_string(self).map_err(serialization)?;
        if json.len() > MAX_JSON_BYTES {
            return Err(invalid());
        }
        Ok(json)
    }
    pub fn from_json(json: &str) -> PureResult<Self> {
        if json.len() > MAX_JSON_BYTES {
            return Err(invalid());
        }
        let checkpoint: Self = serde_json::from_str(json).map_err(serialization)?;
        checkpoint.validate()?;
        Ok(checkpoint)
    }
    pub fn restore_host(&self, target: &mut ConvNeXtClassifier) -> PureResult<()> {
        self.validate()?;
        if target.config() != self.config() || target.num_classes() != self.classes {
            return Err(invalid());
        }
        let parameters = self
            .backbone
            .parameters()
            .iter()
            .chain(&self.head)
            .collect::<Vec<_>>();
        restore_parameters(&parameters, target)
    }
    pub fn to_host(&self) -> PureResult<ConvNeXtClassifier> {
        self.validate()?;
        let mut model = ConvNeXtClassifier::new(self.config().clone(), self.classes, 0)?;
        self.restore_host(&mut model)?;
        Ok(model)
    }
}
