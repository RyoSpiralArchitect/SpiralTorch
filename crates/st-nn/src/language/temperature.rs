// SPDX-License-Identifier: AGPL-3.0-or-later
// (c) 2025 Ryo SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch - Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL Section 13.

use st_core::inference::temperature_control::{
    apply_temperature_control, temperature_distribution_entropy, TemperatureControlConfig,
    TemperatureControlError, TemperatureControlFeedback, TemperatureControlPayload,
    TemperatureControlRequest, TemperatureControlState,
};
use st_core::telemetry::hub::SoftlogicZFeedback;
use st_tensor::TensorError;

/// Stateful st-nn adapter over the canonical st-core temperature transition.
#[derive(Clone, Debug)]
pub struct TemperatureController {
    config: TemperatureControlConfig,
    state: TemperatureControlState,
}

impl TemperatureController {
    pub fn new(
        value: f32,
        target_entropy: f32,
        eta: f32,
        min: f32,
        max: f32,
    ) -> Result<Self, TemperatureControlError> {
        let config = TemperatureControlConfig::new(
            f64::from(target_entropy),
            f64::from(eta),
            f64::from(min),
            f64::from(max),
        )?;
        let state = TemperatureControlState::new(f64::from(value))?;
        state.validate_for(&config)?;
        Ok(Self { config, state })
    }

    pub fn with_feedback(
        mut self,
        kappa: f32,
        relax: f32,
    ) -> Result<Self, TemperatureControlError> {
        self.config = self
            .config
            .with_feedback(f64::from(kappa), f64::from(relax))?;
        self.state.validate_for(&self.config)?;
        Ok(self)
    }

    pub fn with_scale_gain(mut self, gain: f32) -> Result<Self, TemperatureControlError> {
        self.config = self.config.with_scale_gain(f64::from(gain))?;
        self.state.validate_for(&self.config)?;
        Ok(self)
    }

    pub fn with_gradient_decay(mut self, decay: f32) -> Result<Self, TemperatureControlError> {
        self.config = self.config.with_gradient_decay(f64::from(decay))?;
        self.state.validate_for(&self.config)?;
        Ok(self)
    }

    pub fn value(&self) -> f32 {
        self.state.temperature() as f32
    }

    pub fn config(&self) -> TemperatureControlConfig {
        self.config
    }

    pub fn state(&self) -> TemperatureControlState {
        self.state
    }

    pub fn update_detailed(
        &mut self,
        distribution: &[f32],
        z_feedback: Option<&SoftlogicZFeedback>,
    ) -> Result<TemperatureControlPayload, TemperatureControlError> {
        self.transition(distribution, z_feedback, None)
    }

    pub fn update(
        &mut self,
        distribution: &[f32],
        z_feedback: Option<&SoftlogicZFeedback>,
    ) -> Result<f32, TemperatureControlError> {
        self.update_detailed(distribution, z_feedback)
            .map(|payload| payload.temperature as f32)
    }

    pub fn observe_grad(
        &mut self,
        pressure: f32,
        entropy_bias: f32,
    ) -> Result<(), TemperatureControlError> {
        let next = self
            .state
            .with_gradient_observation(f64::from(pressure), f64::from(entropy_bias))?;
        next.validate_for(&self.config)?;
        self.state = next;
        Ok(())
    }

    pub fn update_with_gradient_detailed(
        &mut self,
        distribution: &[f32],
        heat: f32,
    ) -> Result<TemperatureControlPayload, TemperatureControlError> {
        self.transition(distribution, None, Some(f64::from(heat)))
    }

    pub fn update_with_gradient(
        &mut self,
        distribution: &[f32],
        heat: f32,
    ) -> Result<f32, TemperatureControlError> {
        self.update_with_gradient_detailed(distribution, heat)
            .map(|payload| payload.temperature as f32)
    }

    fn transition(
        &mut self,
        distribution: &[f32],
        z_feedback: Option<&SoftlogicZFeedback>,
        gradient_heat: Option<f64>,
    ) -> Result<TemperatureControlPayload, TemperatureControlError> {
        let feedback = z_feedback.map(|feedback| TemperatureControlFeedback {
            psi_total: f64::from(feedback.psi_total),
            band_energy: [
                f64::from(feedback.band_energy.0),
                f64::from(feedback.band_energy.1),
                f64::from(feedback.band_energy.2),
            ],
            drift: f64::from(feedback.drift),
            z_signal: f64::from(feedback.z_signal),
            scale_log_radius: feedback.scale.map(|scale| f64::from(scale.log_radius)),
        });
        let request = TemperatureControlRequest {
            probabilities: distribution.iter().copied().map(f64::from).collect(),
            config: self.config,
            state: self.state,
            feedback,
            gradient_heat,
        };
        let payload = apply_temperature_control(request)?;
        self.state = payload.next_state;
        Ok(payload)
    }
}

pub fn entropy(distribution: &[f32]) -> Result<f32, TemperatureControlError> {
    let probabilities = distribution
        .iter()
        .copied()
        .map(f64::from)
        .collect::<Vec<_>>();
    temperature_distribution_entropy(&probabilities).map(|value| value as f32)
}

pub(super) fn temperature_error_to_tensor(error: TemperatureControlError) -> TensorError {
    match error {
        TemperatureControlError::EmptyDistribution => {
            TensorError::EmptyInput("temperature control distribution")
        }
        TemperatureControlError::NonFinite { field, value }
        | TemperatureControlError::NonFiniteDerived { field, value } => {
            TensorError::NonFiniteValue {
                label: field,
                value: value as f32,
            }
        }
        error => TensorError::Generic(format!("temperature control failed: {error}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use st_core::inference::temperature_control::{
        ZSPACE_TEMPERATURE_CONTROL_CONTRACT_VERSION, ZSPACE_TEMPERATURE_CONTROL_SEMANTIC_OWNER,
    };
    use st_core::theory::zpulse::ZScale;

    #[test]
    fn controller_tracks_gradient_pressure() {
        let mut controller =
            TemperatureController::new(1.0, 0.8, 0.4, 0.4, 2.0).expect("valid controller");
        let baseline = controller
            .update(&[0.6, 0.4], None)
            .expect("valid baseline update");
        controller
            .observe_grad(32.0, 0.15)
            .expect("valid gradient observation");
        let warmed = controller
            .update_with_gradient(&[0.6, 0.4], 1.5)
            .expect("valid gradient update");
        assert!(warmed >= baseline);
        controller
            .observe_grad(0.0, 0.95)
            .expect("valid gradient observation");
        let cooled = controller
            .update_with_gradient(&[0.6, 0.4], 1.5)
            .expect("valid gradient update");
        assert!(cooled <= warmed);
        assert!(cooled >= baseline);
    }

    #[test]
    fn controller_responds_to_scale_feedback() {
        let mut controller = TemperatureController::new(1.0, 0.8, 0.2, 0.3, 2.0)
            .expect("valid controller")
            .with_feedback(0.4, 0.2)
            .expect("valid feedback")
            .with_scale_gain(0.6)
            .expect("valid scale gain");
        let distribution = [0.6, 0.4];
        let baseline = controller
            .update(&distribution, None)
            .expect("valid baseline update");
        let micro_feedback = SoftlogicZFeedback {
            psi_total: 0.0,
            weighted_loss: 0.0,
            band_energy: (0.0, 0.0, 0.0),
            drift: 0.0,
            z_signal: 0.0,
            scale: ZScale::new(0.25),
            events: Vec::new(),
            attributions: Vec::new(),
            elliptic: None,
        };
        let warmed = controller
            .update(&distribution, Some(&micro_feedback))
            .expect("valid micro-scale update");
        assert!(warmed > baseline);

        let macro_feedback = SoftlogicZFeedback {
            psi_total: 0.0,
            weighted_loss: 0.0,
            band_energy: (0.0, 0.0, 0.0),
            drift: 0.0,
            z_signal: 0.0,
            scale: ZScale::new(8.0),
            events: Vec::new(),
            attributions: Vec::new(),
            elliptic: None,
        };
        let cooled = controller
            .update(&distribution, Some(&macro_feedback))
            .expect("valid macro-scale update");
        assert!(cooled < warmed);
    }

    #[test]
    fn controller_commits_only_successful_transitions() {
        let mut controller =
            TemperatureController::new(1.0, 0.8, 0.2, 0.3, 2.0).expect("valid controller");
        let before = controller.state();
        let invalid_feedback = SoftlogicZFeedback {
            drift: f32::NAN,
            ..SoftlogicZFeedback::default()
        };

        let error = controller
            .update(&[0.6, 0.4], Some(&invalid_feedback))
            .expect_err("non-finite telemetry must fail");

        assert!(matches!(
            error,
            TemperatureControlError::NonFinite { field: "drift", .. }
        ));
        assert_eq!(controller.state(), before);
    }

    #[test]
    fn detailed_transition_exposes_the_core_contract() {
        let mut controller =
            TemperatureController::new(1.0, 0.8, 0.2, 0.3, 2.0).expect("valid controller");

        let payload = controller
            .update_detailed(&[0.6, 0.4], None)
            .expect("valid transition");

        assert_eq!(
            payload.contract_version,
            ZSPACE_TEMPERATURE_CONTROL_CONTRACT_VERSION
        );
        assert_eq!(
            payload.semantic_owner,
            ZSPACE_TEMPERATURE_CONTROL_SEMANTIC_OWNER
        );
        assert_eq!(controller.state(), payload.next_state);
    }

    #[test]
    fn invalid_constructor_values_are_not_silently_clamped() {
        let error = TemperatureController::new(1.0, 0.8, 0.2, 2.0, 0.3)
            .expect_err("inverted bounds must fail");

        assert!(matches!(
            error,
            TemperatureControlError::InvalidTemperatureBounds { .. }
        ));
    }

    #[test]
    fn entropy_rejects_non_distributions() {
        let error = entropy(&[0.8, 0.8]).expect_err("invalid mass must fail");

        assert!(matches!(
            error,
            TemperatureControlError::InvalidProbabilityMass { .. }
        ));
    }
}
