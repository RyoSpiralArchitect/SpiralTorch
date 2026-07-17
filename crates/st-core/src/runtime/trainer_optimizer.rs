//! Canonical validation contract for trainer optimizer controls.
//!
//! Trainer hosts may choose how to orchestrate training, but Rust owns the
//! admissible curvature, learning-rate, and gradient-guard semantics.

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const TRAINER_OPTIMIZER_CONFIG_CONTRACT_VERSION: &str =
    "spiraltorch.trainer_optimizer_config.v1";
pub const TRAINER_OPTIMIZER_CONFIG_KIND: &str = "spiraltorch.trainer_optimizer_config";
pub const TRAINER_OPTIMIZER_CONFIG_SEMANTIC_OWNER: &str = "st-core::runtime::trainer_optimizer";
pub const TRAINER_OPTIMIZER_CONFIG_SEMANTIC_BACKEND: &str = "rust";
pub const TRAINER_OPTIMIZER_CONFIG_VALIDATION_RULE: &str =
    "curvature<0;hyper_learning_rate>0;fallback_learning_rate>0;real_learning_rate=null|>0;grad_clip_max_norm=null|>0;all_values_finite;invalid_updates_rejected";

#[derive(Clone, Copy, Debug, Error, PartialEq)]
pub enum TrainerOptimizerConfigError {
    #[error("curvature must be negative and finite, got {value}")]
    InvalidCurvature { value: f32 },
    #[error("{field} must be positive and finite, got {value}")]
    InvalidPositiveFinite { field: &'static str, value: f32 },
}

impl TrainerOptimizerConfigError {
    /// Preserves established tensor error categories at `PureResult` boundaries.
    pub fn into_tensor_error(self) -> st_tensor::TensorError {
        match self {
            Self::InvalidCurvature { value } => {
                st_tensor::TensorError::NonHyperbolicCurvature { curvature: value }
            }
            Self::InvalidPositiveFinite {
                field: "hyper_learning_rate" | "fallback_learning_rate" | "real_learning_rate",
                value,
            } => st_tensor::TensorError::NonPositiveLearningRate { rate: value },
            Self::InvalidPositiveFinite { field, value } => st_tensor::TensorError::Generic(
                format!("{field} must be positive and finite, got {value}"),
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrainerOptimizerConfig {
    pub curvature: f32,
    pub hyper_learning_rate: f32,
    pub fallback_learning_rate: f32,
    pub real_learning_rate: Option<f32>,
    pub grad_clip_max_norm: Option<f32>,
}

impl TrainerOptimizerConfig {
    pub fn try_new(
        curvature: f32,
        hyper_learning_rate: f32,
        fallback_learning_rate: f32,
    ) -> Result<Self, TrainerOptimizerConfigError> {
        let config = Self {
            curvature,
            hyper_learning_rate,
            fallback_learning_rate,
            real_learning_rate: None,
            grad_clip_max_norm: None,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), TrainerOptimizerConfigError> {
        if !self.curvature.is_finite() || self.curvature >= 0.0 {
            return Err(TrainerOptimizerConfigError::InvalidCurvature {
                value: self.curvature,
            });
        }
        validate_positive_finite("hyper_learning_rate", self.hyper_learning_rate)?;
        validate_positive_finite("fallback_learning_rate", self.fallback_learning_rate)?;
        if let Some(value) = self.real_learning_rate {
            validate_positive_finite("real_learning_rate", value)?;
        }
        if let Some(value) = self.grad_clip_max_norm {
            validate_positive_finite("grad_clip_max_norm", value)?;
        }
        Ok(())
    }

    pub fn with_real_learning_rate(
        mut self,
        value: Option<f32>,
    ) -> Result<Self, TrainerOptimizerConfigError> {
        self.real_learning_rate = value;
        self.validate()?;
        Ok(self)
    }

    pub fn with_grad_clip_max_norm(
        mut self,
        value: Option<f32>,
    ) -> Result<Self, TrainerOptimizerConfigError> {
        self.grad_clip_max_norm = value;
        self.validate()?;
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct TrainerOptimizerConfigContract {
    pub kind: &'static str,
    pub contract_version: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub validation_rule: &'static str,
    pub realgrad_enabled: bool,
    pub gradient_clip_enabled: bool,
    pub config: TrainerOptimizerConfig,
}

pub fn evaluate_trainer_optimizer_config(
    config: TrainerOptimizerConfig,
) -> Result<TrainerOptimizerConfigContract, TrainerOptimizerConfigError> {
    config.validate()?;
    Ok(TrainerOptimizerConfigContract {
        kind: TRAINER_OPTIMIZER_CONFIG_KIND,
        contract_version: TRAINER_OPTIMIZER_CONFIG_CONTRACT_VERSION,
        semantic_owner: TRAINER_OPTIMIZER_CONFIG_SEMANTIC_OWNER,
        semantic_backend: TRAINER_OPTIMIZER_CONFIG_SEMANTIC_BACKEND,
        validation_rule: TRAINER_OPTIMIZER_CONFIG_VALIDATION_RULE,
        realgrad_enabled: config.real_learning_rate.is_some(),
        gradient_clip_enabled: config.grad_clip_max_norm.is_some(),
        config,
    })
}

fn validate_positive_finite(
    field: &'static str,
    value: f32,
) -> Result<(), TrainerOptimizerConfigError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(TrainerOptimizerConfigError::InvalidPositiveFinite { field, value });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn valid_config_emits_a_versioned_rust_contract() {
        let config = TrainerOptimizerConfig::try_new(-1.0, 0.02, 0.01)
            .unwrap()
            .with_real_learning_rate(Some(0.005))
            .unwrap()
            .with_grad_clip_max_norm(Some(1.5))
            .unwrap();

        let contract = evaluate_trainer_optimizer_config(config).unwrap();

        assert_eq!(contract.kind, TRAINER_OPTIMIZER_CONFIG_KIND);
        assert_eq!(
            contract.contract_version,
            TRAINER_OPTIMIZER_CONFIG_CONTRACT_VERSION
        );
        assert_eq!(
            contract.semantic_owner,
            TRAINER_OPTIMIZER_CONFIG_SEMANTIC_OWNER
        );
        assert_eq!(
            contract.semantic_backend,
            TRAINER_OPTIMIZER_CONFIG_SEMANTIC_BACKEND
        );
        assert!(contract.realgrad_enabled);
        assert!(contract.gradient_clip_enabled);
        assert_eq!(contract.config, config);
    }

    #[test]
    fn curvature_and_learning_rates_fail_closed() {
        assert!(matches!(
            TrainerOptimizerConfig::try_new(0.0, 0.02, 0.01),
            Err(TrainerOptimizerConfigError::InvalidCurvature { .. })
        ));
        assert!(matches!(
            TrainerOptimizerConfig::try_new(f32::NAN, 0.02, 0.01),
            Err(TrainerOptimizerConfigError::InvalidCurvature { .. })
        ));
        assert!(matches!(
            TrainerOptimizerConfig::try_new(-1.0, 0.0, 0.01),
            Err(TrainerOptimizerConfigError::InvalidPositiveFinite {
                field: "hyper_learning_rate",
                ..
            })
        ));
        assert!(matches!(
            TrainerOptimizerConfig::try_new(-1.0, 0.02, f32::INFINITY),
            Err(TrainerOptimizerConfigError::InvalidPositiveFinite {
                field: "fallback_learning_rate",
                ..
            })
        ));
    }

    #[test]
    fn optional_control_updates_reject_noise_without_changing_the_source() {
        let config = TrainerOptimizerConfig::try_new(-1.0, 0.02, 0.01)
            .unwrap()
            .with_real_learning_rate(Some(0.005))
            .unwrap()
            .with_grad_clip_max_norm(Some(1.5))
            .unwrap();

        assert!(config.with_real_learning_rate(Some(f32::NAN)).is_err());
        assert!(config.with_grad_clip_max_norm(Some(0.0)).is_err());
        assert_eq!(config.real_learning_rate, Some(0.005));
        assert_eq!(config.grad_clip_max_norm, Some(1.5));

        let invalid_base = TrainerOptimizerConfig {
            curvature: 0.5,
            ..config
        };
        assert!(matches!(
            invalid_base.with_real_learning_rate(None),
            Err(TrainerOptimizerConfigError::InvalidCurvature { .. })
        ));
    }

    #[test]
    fn serde_ingress_rejects_unknown_fields() {
        let error = serde_json::from_value::<TrainerOptimizerConfig>(json!({
            "curvature": -1.0,
            "hyper_learning_rate": 0.02,
            "fallback_learning_rate": 0.01,
            "real_learning_rate": null,
            "grad_clip_max_norm": null,
            "commander": "python"
        }))
        .unwrap_err();

        assert!(error.to_string().contains("unknown field"));
    }

    #[test]
    fn optimizer_errors_preserve_existing_tensor_error_categories() {
        let curvature =
            TrainerOptimizerConfigError::InvalidCurvature { value: 0.5 }.into_tensor_error();
        assert!(matches!(
            curvature,
            st_tensor::TensorError::NonHyperbolicCurvature { curvature: 0.5 }
        ));

        let rate = TrainerOptimizerConfigError::InvalidPositiveFinite {
            field: "hyper_learning_rate",
            value: 0.0,
        }
        .into_tensor_error();
        assert!(matches!(
            rate,
            st_tensor::TensorError::NonPositiveLearningRate { rate: 0.0 }
        ));

        let clip = TrainerOptimizerConfigError::InvalidPositiveFinite {
            field: "grad_clip_max_norm",
            value: 0.0,
        }
        .into_tensor_error();
        assert!(
            matches!(clip, st_tensor::TensorError::Generic(message) if message.contains("grad_clip_max_norm"))
        );
    }
}
