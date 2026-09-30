//! Validated, device-independent plain-SGD model state, not a DataLoader checkpoint.
use super::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

const SCHEMA: &str = "spiraltorch.convnext.plain_sgd_checkpoint.v1";
pub(super) const MAX_JSON_BYTES: usize = 512 * 1024 * 1024;
pub(super) const MAX_PARAMETER_VALUES: usize = 64 * 1024 * 1024;
const MAX_PARAMETER_TENSORS: usize = 65_536;

fn invalid(label: &'static str) -> TensorError {
    TensorError::InvalidValue { label }
}

fn serialization(error: impl ToString) -> TensorError {
    TensorError::SerializationError {
        message: error.to_string(),
    }
}

// Check the architecture's allocation budget before constructing a model from
// external metadata. The same preflight also protects the host constructor.
impl ConvNeXtConfig {
    pub(super) fn parameter_budget(&self) -> PureResult<(usize, usize)> {
        if self.stage_dims.is_empty() {
            return Err(invalid("convnext_stage_dims"));
        }
        if self.stage_dims.len() != self.stage_depths.len() {
            return Err(TensorError::InvalidDimensions {
                rows: self.stage_dims.len(),
                cols: self.stage_depths.len(),
            });
        }
        if self.curvature >= 0.0 || !self.curvature.is_finite() {
            return Err(TensorError::NonHyperbolicCurvature {
                curvature: self.curvature,
            });
        }
        if self.epsilon <= 0.0 || !self.epsilon.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "convnext_layernorm_epsilon",
                value: self.epsilon,
            });
        }
        if [
            self.input_channels,
            self.input_hw.0,
            self.input_hw.1,
            self.patch_size.0,
            self.patch_size.1,
        ]
        .contains(&0)
            || self.stage_dims.contains(&0)
        {
            return Err(invalid("convnext_dimensions"));
        }
        let mul = |a: usize, b: usize| {
            a.checked_mul(b)
                .ok_or_else(|| invalid("convnext_size_overflow"))
        };
        let add = |a: usize, b: usize| {
            a.checked_add(b)
                .ok_or_else(|| invalid("convnext_size_overflow"))
        };
        mul(self.input_channels, mul(self.input_hw.0, self.input_hw.1)?)?;
        let mut height = self.input_hw.0 / self.patch_size.0;
        let mut width = self.input_hw.1 / self.patch_size.1;
        let first = self.stage_dims[0];
        let mut values = add(
            mul(
                mul(first, self.input_channels)?,
                mul(self.patch_size.0, self.patch_size.1)?,
            )?,
            first,
        )?;
        let mut tensors = 4usize; // stem weight/bias and final norm gain/bias
        for (stage, (&channels, &depth)) in
            self.stage_dims.iter().zip(&self.stage_depths).enumerate()
        {
            if height == 0 || width == 0 {
                return Err(invalid("convnext_spatial_extent"));
            }
            // Two 4C MLP matrices, 7x7 depthwise weights, and five affine/bias vectors.
            let block_values = add(mul(8, mul(channels, channels)?)?, mul(57, channels)?)?;
            values = add(values, mul(depth, block_values)?)?;
            tensors = add(tensors, mul(8, depth)?)?;
            if let Some(&next) = self.stage_dims.get(stage + 1) {
                values = add(values, add(mul(4, mul(channels, next)?)?, next)?)?;
                tensors = add(tensors, 2)?;
                height /= 2;
                width /= 2;
            }
        }
        let features = mul(*self.stage_dims.last().unwrap(), mul(height, width)?)?;
        values = add(values, mul(2, features)?)?;
        Ok((tensors, values))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct StoredParameter {
    pub(super) name: String,
    pub(super) shape: [usize; 2],
    pub(super) values: Vec<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Optimizer {
    PlainSgd,
}

/// Exact model weights/configuration and attempted-update clock for plain SGD.
/// Loss/rate schedules, data cursors, RNGs and in-flight derivatives are external.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConvNeXtTrainingCheckpoint {
    schema: String,
    optimizer: Optimizer,
    config: ConvNeXtConfig,
    batch: usize,
    attempted_updates: u64,
    parameters: Vec<StoredParameter>,
}

impl ConvNeXtTrainingCheckpoint {
    pub(super) fn new(
        config: ConvNeXtConfig,
        batch: usize,
        attempted_updates: u64,
        parameters: Vec<StoredParameter>,
    ) -> PureResult<Self> {
        let checkpoint = Self {
            schema: SCHEMA.into(),
            optimizer: Optimizer::PlainSgd,
            config,
            batch,
            attempted_updates,
            parameters,
        };
        checkpoint.validate()?;
        Ok(checkpoint)
    }

    pub fn config(&self) -> &ConvNeXtConfig {
        &self.config
    }
    pub fn batch_size(&self) -> usize {
        self.batch
    }
    pub fn attempted_updates(&self) -> u64 {
        self.attempted_updates
    }

    pub(super) fn parameters(&self) -> &[StoredParameter] {
        &self.parameters
    }

    pub(super) fn validate(&self) -> PureResult<()> {
        if self.schema != SCHEMA || self.batch == 0 || self.attempted_updates == u64::MAX {
            return Err(invalid("convnext_checkpoint_header"));
        }
        let (count, expected_values) = self.config.parameter_budget()?;
        if expected_values > MAX_PARAMETER_VALUES
            || count > MAX_PARAMETER_TENSORS
            || count != self.parameters.len()
        {
            return Err(invalid("convnext_checkpoint_budget"));
        }
        self.batch
            .checked_mul(self.config.input_channels)
            .and_then(|n| n.checked_mul(self.config.input_hw.0))
            .and_then(|n| n.checked_mul(self.config.input_hw.1))
            .ok_or_else(|| invalid("convnext_checkpoint_batch_overflow"))?;
        let mut names = HashSet::new();
        let mut total = 0usize;
        for parameter in &self.parameters {
            let elements = parameter.shape[0]
                .checked_mul(parameter.shape[1])
                .ok_or_else(|| invalid("convnext_checkpoint_shape_overflow"))?;
            if elements == 0
                || elements != parameter.values.len()
                || parameter.name.is_empty()
                || !names.insert(&parameter.name)
                || !parameter.values.iter().all(|v| v.is_finite())
            {
                return Err(invalid("convnext_checkpoint_parameter"));
            }
            total = total
                .checked_add(elements)
                .ok_or_else(|| invalid("convnext_checkpoint_size_overflow"))?;
        }
        if total != expected_values {
            return Err(invalid("convnext_checkpoint_size"));
        }
        Ok(())
    }

    pub fn to_json(&self) -> PureResult<String> {
        self.validate()?;
        let json = serde_json::to_string(self).map_err(serialization)?;
        if json.len() > MAX_JSON_BYTES {
            return Err(invalid("convnext_checkpoint_json_size"));
        }
        Ok(json)
    }

    pub fn from_json(json: &str) -> PureResult<Self> {
        if json.len() > MAX_JSON_BYTES {
            return Err(invalid("convnext_checkpoint_json_size"));
        }
        let checkpoint: Self = serde_json::from_str(json).map_err(serialization)?;
        checkpoint.validate()?;
        Ok(checkpoint)
    }

    /// Validate every name, shape and optimizer attachment before mutating any
    /// target parameter. This is explicit host handoff, not live weight aliasing.
    pub fn restore_host(&self, target: &mut ConvNeXtBackbone) -> PureResult<()> {
        self.validate()?;
        if target.config() != &self.config {
            return Err(invalid("convnext_checkpoint_config"));
        }
        restore_parameters(&self.parameters.iter().collect::<Vec<_>>(), target)
    }

    /// Reconstruct a fresh host model only after validating its allocation budget.
    pub fn to_host(&self) -> PureResult<ConvNeXtBackbone> {
        self.validate()?;
        let mut model = ConvNeXtBackbone::new(self.config.clone())?;
        self.restore_host(&mut model)?;
        Ok(model)
    }
}

pub(super) fn restore_parameters(
    parameters: &[&StoredParameter],
    target: &mut impl Module,
) -> PureResult<()> {
    let mut index = 0usize;
    target.visit_parameters(&mut |p| {
        let source = parameters
            .get(index)
            .ok_or_else(|| invalid("convnext_checkpoint_count"))?;
        if source.name != p.name() || source.shape != [p.value().shape().0, p.value().shape().1] {
            return Err(invalid("convnext_checkpoint_layout"));
        }
        if p.gradient().is_some() || p.hypergrad().is_some() || p.realgrad().is_some() {
            return Err(invalid("convnext_checkpoint_attached_optimizer"));
        }
        index += 1;
        Ok(())
    })?;
    if index != parameters.len() {
        return Err(invalid("convnext_checkpoint_count"));
    }
    let state = parameters
        .iter()
        .map(|p| {
            Ok((
                p.name.clone(),
                Tensor::from_vec(p.shape[0], p.shape[1], p.values.clone())?,
            ))
        })
        .collect::<PureResult<HashMap<_, _>>>()?;
    target.load_state_dict(&state)
}

#[cfg(test)]
mod tests;
