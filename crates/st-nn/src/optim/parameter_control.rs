//! Parameter-side control state shared with resident clients. Latent optimizer
//! state remains owned by its producer; this stores only the applied projection.
use super::{
    checked_scaled_learning_rate, plan_zspace_parameter_control, PureResult, TensorError,
    ZSpaceParameterControl, ZSpaceParameterControlReceipt,
};
use serde::{Deserialize, Serialize};
use st_core::runtime::zspace_optimizer::{
    ZSPACE_META_OPTIMIZER_MAX_SAFE_STEP, ZSPACE_PARAMETER_CONTROL_MAX_LEARNING_RATE_SCALE,
    ZSPACE_PARAMETER_CONTROL_MIN_LEARNING_RATE_SCALE,
};

#[derive(Clone, Copy, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ZSpaceParameterControlState {
    absolute_learning_rate_scale: f32,
    source_meta_step: Option<u64>,
}

impl Default for ZSpaceParameterControlState {
    fn default() -> Self {
        Self {
            absolute_learning_rate_scale: 1.0,
            source_meta_step: None,
        }
    }
}

impl ZSpaceParameterControlState {
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }

    pub fn absolute_learning_rate_scale(&self) -> f32 {
        self.absolute_learning_rate_scale
    }

    /// Control-message clock, distinct from accepted model-update count.
    pub fn source_meta_step(&self) -> Option<u64> {
        self.source_meta_step
    }

    pub fn validate(&self) -> PureResult<()> {
        let scale = self.absolute_learning_rate_scale;
        if !scale.is_finite()
            || !(ZSPACE_PARAMETER_CONTROL_MIN_LEARNING_RATE_SCALE as f32
                ..=ZSPACE_PARAMETER_CONTROL_MAX_LEARNING_RATE_SCALE as f32)
                .contains(&scale)
            || match self.source_meta_step {
                Some(step) => step == 0 || step > ZSPACE_META_OPTIMIZER_MAX_SAFE_STEP,
                None => scale != 1.0,
            }
        {
            return Err(TensorError::Generic(
                "invalid Z-space parameter control checkpoint state".into(),
            ));
        }
        Ok(())
    }

    /// Prepare without mutating the owner. Commit the returned state only after
    /// validating the owner's affected rates; no event is emitted by preview.
    pub fn preview(
        &self,
        control: &ZSpaceParameterControl,
    ) -> PureResult<(Self, ZSpaceParameterControlReceipt)> {
        self.validate()?;
        let receipt = plan_zspace_parameter_control(
            self.absolute_learning_rate_scale,
            self.source_meta_step,
            control,
        )?
        .receipt();
        let next = Self {
            absolute_learning_rate_scale: receipt.absolute_learning_rate_scale,
            source_meta_step: Some(receipt.source_meta_step),
        };
        next.validate()?;
        Ok((next, receipt))
    }

    /// Scale the nominal scheduler rate exactly once. A zero-rate SGD probe
    /// remains zero; positive rates may not silently underflow or overflow.
    pub fn effective_learning_rate(&self, nominal_rate: f32) -> PureResult<f32> {
        self.validate()?;
        if nominal_rate == 0.0 {
            return Ok(nominal_rate);
        }
        checked_scaled_learning_rate(nominal_rate, self.absolute_learning_rate_scale)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use st_core::runtime::zspace_optimizer::{
        transition_zspace_meta_optimizer, zspace_parameter_control_from_report,
        ZSpaceMetaObservation, ZSpaceMetaOptimizerConfig, ZSpaceMetaOptimizerState,
        ZSpaceMetaOptimizerStepRequest,
    };
    use std::collections::BTreeMap;

    fn control(step: u64, scale: f64) -> ZSpaceParameterControl {
        let mut state = ZSpaceMetaOptimizerState::zeros(2);
        state.step = step - 1;
        let report = transition_zspace_meta_optimizer(ZSpaceMetaOptimizerStepRequest {
            config: ZSpaceMetaOptimizerConfig {
                dimension: 2,
                topos_control_gain: 1.,
                ..Default::default()
            },
            state,
            observation: ZSpaceMetaObservation {
                gradient: vec![0.1, -0.2],
                telemetry: BTreeMap::from([(
                    "topos.training_hints.learning_rate_scale".into(),
                    scale,
                )]),
                ..Default::default()
            },
        })
        .unwrap();
        zspace_parameter_control_from_report(&report).unwrap()
    }

    #[test]
    fn shared_preview_matches_host_plan_and_never_compounds() {
        let state = ZSpaceParameterControlState::default();
        let first = control(1, 0.5);
        let (next, receipt) = state.preview(&first).unwrap();
        assert!(state.is_default());
        assert_eq!(
            receipt,
            plan_zspace_parameter_control(1., None, &first)
                .unwrap()
                .receipt()
        );
        assert_eq!(next.effective_learning_rate(0.02).unwrap(), 0.01);
        let (replayed, receipt) = next.preview(&first).unwrap();
        assert_eq!(next, replayed);
        assert!(!receipt.changed);
        assert_eq!(replayed.effective_learning_rate(0.04).unwrap(), 0.02);
        let (reset, _) = next.preview(&control(2, 1.)).unwrap();
        assert_eq!(reset.effective_learning_rate(0.04).unwrap(), 0.04);
        assert!(!reset.is_default());
        assert!(reset.preview(&first).is_err());
        assert!(reset.preview(&control(2, 0.5)).is_err());
        let restored: ZSpaceParameterControlState =
            serde_json::from_str(&serde_json::to_string(&reset).unwrap()).unwrap();
        assert_eq!(reset, restored);
        assert!(restored.preview(&first).is_err());
    }

    #[test]
    fn scaled_rates_keep_zero_probes_and_reject_lost_positive_updates() {
        let (half, _) = ZSpaceParameterControlState::default()
            .preview(&control(1, 0.5))
            .unwrap();
        assert_eq!(
            half.effective_learning_rate(-0.).unwrap().to_bits(),
            (-0_f32).to_bits()
        );
        for rate in [-1., f32::NAN, f32::INFINITY, f32::from_bits(1)] {
            assert!(half.effective_learning_rate(rate).is_err());
        }
        let (larger, _) = half.preview(&control(2, 1.25)).unwrap();
        assert!(larger.effective_learning_rate(f32::MAX).is_err());
    }

    #[test]
    fn deserialized_state_cannot_skip_validation() {
        for (scale, step) in [
            (0.5, None),
            (0., Some(1)),
            (2., Some(1)),
            (0.5, Some(0)),
            (0.5, Some(ZSPACE_META_OPTIMIZER_MAX_SAFE_STEP + 1)),
        ] {
            let state: ZSpaceParameterControlState = serde_json::from_value(json!({
                "absolute_learning_rate_scale": scale, "source_meta_step": step
            }))
            .unwrap();
            assert!(state.validate().is_err());
            assert!(state.effective_learning_rate(0.).is_err());
            assert!(state.preview(&control(1, 0.5)).is_err());
        }
    }
}
