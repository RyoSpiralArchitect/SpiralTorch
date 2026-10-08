use super::{checked_increment, ema_update, require_finite, ZSpaceOptimizerFeedbackError};
use serde::{Deserialize, Serialize};

/// Equal-weight loss aggregation; does not infer sample identity or an epoch.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ZSpaceOptimizerFeedbackLossWindow {
    pub observations_per_window: u64,
    pub observations: u64,
    pub completed_windows: u64,
    pub mean: Option<f64>,
    pub previous_mean: Option<f64>,
}

impl ZSpaceOptimizerFeedbackLossWindow {
    pub(super) fn new(observations_per_window: u64) -> Self {
        Self {
            observations_per_window,
            observations: 0,
            completed_windows: 0,
            mean: None,
            previous_mean: None,
        }
    }

    pub(super) fn validate(
        &self,
        size: u64,
        total: u64,
        last_loss: Option<f64>,
    ) -> Result<(), ZSpaceOptimizerFeedbackError> {
        if self.observations_per_window != size
            || self.observations != total % size
            || self.completed_windows != total / size
            || self.mean.is_some() != (self.observations > 0)
            || self.previous_mean.is_some() != (self.completed_windows > 0)
            || (self.observations == 1 && self.mean != last_loss)
        {
            return Err(ZSpaceOptimizerFeedbackError::InvalidState {
                field: "state.loss_window",
                message: "window size, counts and means must match accepted observations",
            });
        }
        for value in [self.mean, self.previous_mean].into_iter().flatten() {
            require_finite("state.loss_window.mean", value)?;
        }
        Ok(())
    }

    pub(super) fn observe(
        &mut self,
        loss: f64,
    ) -> Result<Option<(f64, Option<f64>)>, ZSpaceOptimizerFeedbackError> {
        let count = checked_increment("state.loss_window.observations", self.observations)?;
        let mean = ema_update(
            self.mean,
            loss,
            1.0 / count as f64,
            "state.loss_window.mean",
        )?;
        if count == self.observations_per_window {
            let previous = self.previous_mean;
            self.completed_windows = checked_increment(
                "state.loss_window.completed_windows",
                self.completed_windows,
            )?;
            self.previous_mean = Some(mean);
            self.observations = 0;
            self.mean = None;
            Ok(Some((mean, previous)))
        } else {
            self.observations = count;
            self.mean = Some(mean);
            Ok(None)
        }
    }
}
