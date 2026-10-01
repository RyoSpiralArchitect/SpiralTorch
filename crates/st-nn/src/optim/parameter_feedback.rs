//! Checkpointable owner adapter for the existing Rust loss-feedback gate.
//! A preview advances no live state; the trainer commits it at settlement.
use super::{checked_scaled_learning_rate, PureResult, TensorError};
use serde::{Deserialize, Serialize};
use st_core::runtime::zspace_optimizer_feedback::{
    control_zspace_optimizer_feedback, initialize_zspace_optimizer_feedback,
    observe_zspace_optimizer_feedback, restore_zspace_optimizer_feedback,
    ZSpaceOptimizerFeedbackConfig, ZSpaceOptimizerFeedbackControlRequest,
    ZSpaceOptimizerFeedbackObservation, ZSpaceOptimizerFeedbackObserveRequest,
    ZSpaceOptimizerFeedbackRestoreRequest, ZSpaceOptimizerFeedbackState,
};

fn error(error: impl std::fmt::Display) -> TensorError {
    TensorError::Generic(error.to_string())
}

// Reserve headroom for relative-loss subtraction and its f64 EMA before an
// f32 parameter update can commit. The standalone core still accepts f64 losses.
const MIN_F32_LOSS_FLOOR: f64 = (f32::MAX as f64 * 8.0) / f64::MAX;

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ZSpaceParameterFeedbackState {
    config: ZSpaceOptimizerFeedbackConfig,
    state: ZSpaceOptimizerFeedbackState,
}

impl ZSpaceParameterFeedbackState {
    pub fn new(config: ZSpaceOptimizerFeedbackConfig) -> PureResult<Self> {
        let initialized = initialize_zspace_optimizer_feedback(config).map_err(error)?;
        let owner = Self {
            config: initialized.config,
            state: initialized.state,
        };
        owner.validate()?;
        Ok(owner)
    }

    pub fn config(&self) -> &ZSpaceOptimizerFeedbackConfig {
        &self.config
    }

    pub fn state(&self) -> &ZSpaceOptimizerFeedbackState {
        &self.state
    }

    pub fn validate(&self) -> PureResult<()> {
        restore_zspace_optimizer_feedback(ZSpaceOptimizerFeedbackRestoreRequest {
            config: self.config.clone(),
            state: self.state.clone(),
        })
        .map_err(error)?;
        if self.config.loss_floor < MIN_F32_LOSS_FLOOR
            || self
                .state
                .last_loss
                .is_some_and(|value| value.abs() > f64::from(f32::MAX))
            || self
                .state
                .loss_ema
                .is_some_and(|value| value.abs() > 2.0 * f64::from(f32::MAX))
            || self
                .state
                .relative_loss_delta_ema
                .is_some_and(|value| value.abs() > f64::MAX / 2.0)
            || self.state.loss_window.as_ref().is_some_and(|window| {
                [window.mean, window.previous_mean]
                    .into_iter()
                    .flatten()
                    .any(|value| value.abs() > 2.0 * f64::from(f32::MAX))
            })
        {
            return Err(error("feedback exceeds the finite f32 loss domain"));
        }
        // This adapter observes completed attempts, never a synthetic step zero.
        if self.state.observation_count > self.state.control_step
            || self.state.last_observation_step == Some(0)
        {
            return Err(error(
                "feedback observation is not a completed update attempt",
            ));
        }
        Ok(())
    }

    /// Gate an absolute proposal back toward identity, then scale the nominal
    /// rate once. The returned state belongs to the pending update attempt.
    pub fn preview(&self, nominal_rate: f32, proposed_scale: f32) -> PureResult<(Self, f32)> {
        self.validate()?;
        let report = control_zspace_optimizer_feedback(ZSpaceOptimizerFeedbackControlRequest {
            config: self.config.clone(),
            state: self.state.clone(),
            target_step: self.state.control_step + 1,
            proposed_learning_rate_scale: f64::from(proposed_scale),
        })
        .map_err(error)?;
        let rate = if nominal_rate == 0.0 {
            nominal_rate
        } else {
            checked_scaled_learning_rate(nominal_rate, report.applied_learning_rate_scale as f32)?
        };
        Ok((
            Self {
                config: self.config.clone(),
                state: report.state_after,
            },
            rate,
        ))
    }

    /// Observe the frozen pre-update loss of an accepted attempt. Rejected
    /// attempts commit only the preview, allowing the core staleness rule to act.
    pub fn observe(&self, loss: f32, learning_rate: f32, epoch: u64) -> PureResult<Self> {
        self.validate()?;
        if self.state.control_step == 0 {
            return Err(error("feedback requires a completed update attempt"));
        }
        let report = observe_zspace_optimizer_feedback(ZSpaceOptimizerFeedbackObserveRequest {
            config: self.config.clone(),
            state: self.state.clone(),
            observation: ZSpaceOptimizerFeedbackObservation {
                step: self.state.control_step,
                max_steps: None,
                epoch: Some(epoch as f64),
                loss: f64::from(loss),
                grad_norm: None,
                learning_rate: Some(f64::from(learning_rate)),
            },
        })
        .map_err(error)?;
        Ok(Self {
            config: self.config.clone(),
            state: report.state_after,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsafe_loss_floor_is_rejected_before_parameter_updates() {
        let config = ZSpaceOptimizerFeedbackConfig {
            loss_floor: 1e-310,
            ..Default::default()
        };
        assert!(initialize_zspace_optimizer_feedback(config.clone()).is_ok());
        assert!(ZSpaceParameterFeedbackState::new(config).is_err());
    }

    #[test]
    fn admitted_floor_keeps_extreme_f32_loss_transitions_finite() {
        let mut state = ZSpaceParameterFeedbackState::new(ZSpaceOptimizerFeedbackConfig {
            loss_floor: MIN_F32_LOSS_FLOOR,
            relative_delta_ema_alpha: 1.0,
            ..Default::default()
        })
        .unwrap();
        for loss in [0.0, f32::MAX, 0.0, -f32::MAX, 0.0, f32::MAX] {
            let (pending, rate) = state.preview(0.01, 0.5).unwrap();
            state = pending.observe(loss, rate, 0).unwrap();
            state.validate().unwrap();
        }
    }

    #[test]
    fn restored_out_of_domain_history_is_rejected_before_preview() {
        let mut state = ZSpaceParameterFeedbackState::new(Default::default()).unwrap();
        let (pending, rate) = state.preview(0.01, 0.5).unwrap();
        state = pending.observe(1.0, rate, 0).unwrap();
        let (pending, rate) = state.preview(0.01, 0.5).unwrap();
        state = pending.observe(0.5, rate, 0).unwrap();
        for field in ["last_loss", "loss_ema", "relative_loss_delta_ema"] {
            let mut value = serde_json::to_value(&state).unwrap();
            value["state"][field] = serde_json::json!(f64::MAX);
            let restored: ZSpaceParameterFeedbackState = serde_json::from_value(value).unwrap();
            assert!(restored.preview(0.01, 0.5).is_err(), "{field}");
        }
    }

    #[test]
    fn window_means_remain_finite_and_restore_rejects_out_of_domain_means() {
        let mut state = ZSpaceParameterFeedbackState::new(ZSpaceOptimizerFeedbackConfig {
            loss_window_observations: 2,
            loss_floor: MIN_F32_LOSS_FLOOR,
            relative_delta_ema_alpha: 1.0,
            ..Default::default()
        })
        .unwrap();
        for loss in [0.0, 0.0, f32::MAX, f32::MAX, -f32::MAX, -f32::MAX] {
            let (pending, rate) = state.preview(0.01, 0.5).unwrap();
            state = pending.observe(loss, rate, 0).unwrap();
            state.validate().unwrap();
        }
        let mut encoded = serde_json::to_value(&state).unwrap();
        encoded["state"]["loss_window"]["previous_mean"] = serde_json::json!(f64::MAX);
        let restored: ZSpaceParameterFeedbackState = serde_json::from_value(encoded).unwrap();
        assert!(restored.preview(0.01, 0.5).is_err());

        let mut state = ZSpaceParameterFeedbackState::new(ZSpaceOptimizerFeedbackConfig {
            loss_window_observations: 3,
            ..Default::default()
        })
        .unwrap();
        for loss in [1.0, 2.0] {
            let (pending, rate) = state.preview(0.01, 0.5).unwrap();
            state = pending.observe(loss, rate, 0).unwrap();
        }
        let mut encoded = serde_json::to_value(&state).unwrap();
        encoded["state"]["loss_window"]["mean"] = serde_json::json!(f64::MAX);
        let restored: ZSpaceParameterFeedbackState = serde_json::from_value(encoded).unwrap();
        assert!(restored.preview(0.01, 0.5).is_err());
    }

    #[test]
    fn core_gate_changes_rates_and_missing_observations_revert_to_identity() {
        let mut state = ZSpaceParameterFeedbackState::new(ZSpaceOptimizerFeedbackConfig {
            recovery_rate: 0.5,
            relative_delta_ema_alpha: 1.,
            ..Default::default()
        })
        .unwrap();
        assert!(state.observe(1., 0.1, 0).is_err());
        for (loss, rate) in [(1., 0.1), (0.9, 0.1), (0.8, 0.1), (0.7, 0.075)] {
            let (pending, actual) = state.preview(0.1, 0.5).unwrap();
            assert_eq!(actual, rate);
            state = pending.observe(loss, actual, 0).unwrap();
            assert!(state.observe(loss, actual, 0).is_err());
        }
        assert_eq!(state.preview(0.1, 0.5).unwrap().1, 0.05);
        let restored: ZSpaceParameterFeedbackState =
            serde_json::from_str(&serde_json::to_string(&state).unwrap()).unwrap();
        assert_eq!(state, restored);
        let (rejected, _) = state.preview(0.1, 0.5).unwrap();
        assert_eq!(rejected.state.last_loss, state.state.last_loss);
        assert_eq!(rejected.preview(0.1, 0.5).unwrap().1, 0.1);
        let (pending, _) = state.preview(0.1, 0.5).unwrap();
        let halted = pending.observe(2., 0.05, 0).unwrap();
        assert!(halted.state.halted);
        assert_eq!(halted.preview(0.1, 0.5).unwrap().1, 0.1);
    }

    #[test]
    fn invalid_inputs_and_corrupt_checkpoints_do_not_mutate_the_owner() {
        let state = ZSpaceParameterFeedbackState::new(Default::default()).unwrap();
        for rate in [-1., f32::NAN, f32::INFINITY] {
            assert!(state.preview(rate, 0.5).is_err());
        }
        for proposal in [0., 2., f32::NAN] {
            assert!(state.preview(0.1, proposal).is_err());
        }
        assert_eq!(
            state.preview(-0., 0.5).unwrap().1.to_bits(),
            (-0_f32).to_bits()
        );
        assert_eq!(state.state.control_step, 0);
        let (pending, _) = state.preview(0.1, 0.5).unwrap();
        assert!(pending.observe(f32::NAN, 0.1, 0).is_err());
        let mut bad = serde_json::to_value(&pending).unwrap();
        bad["state"]["gate"] = serde_json::json!(1.0);
        let bad: ZSpaceParameterFeedbackState = serde_json::from_value(bad).unwrap();
        assert!(bad.validate().is_err());
    }
}
