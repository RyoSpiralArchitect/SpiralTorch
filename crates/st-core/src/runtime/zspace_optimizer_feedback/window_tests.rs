use super::*;

fn config(size: u64) -> ZSpaceOptimizerFeedbackConfig {
    ZSpaceOptimizerFeedbackConfig {
        loss_window_observations: size,
        ..Default::default()
    }
}

fn advance(
    config: &ZSpaceOptimizerFeedbackConfig,
    state: ZSpaceOptimizerFeedbackState,
    loss: f64,
) -> ZSpaceOptimizerFeedbackObservationReport {
    let step = state.control_step + 1;
    let controlled = control_zspace_optimizer_feedback(ZSpaceOptimizerFeedbackControlRequest {
        config: config.clone(),
        state,
        target_step: step,
        proposed_learning_rate_scale: 0.5,
    })
    .unwrap();
    observe_zspace_optimizer_feedback(ZSpaceOptimizerFeedbackObserveRequest {
        config: config.clone(),
        state: controlled.state_after,
        observation: ZSpaceOptimizerFeedbackObservation {
            step,
            max_steps: None,
            epoch: None,
            loss,
            grad_norm: None,
            learning_rate: None,
        },
    })
    .unwrap()
}

#[test]
fn default_config_and_state_keep_the_legacy_encoding() {
    let configured = config(1);
    let encoded = serde_json::to_string(&configured).unwrap();
    assert!(!encoded.contains("loss_window"));
    assert_eq!(
        serde_json::from_str::<ZSpaceOptimizerFeedbackConfig>(&encoded).unwrap(),
        configured
    );
    let mut state = initialize_zspace_optimizer_feedback(configured.clone())
        .unwrap()
        .state;
    for loss in [2.0, 1.5, 1.6, 1.2] {
        state = advance(&configured, state, loss).state_after;
        let json = serde_json::to_string(&state).unwrap();
        assert!(!json.contains("loss_window"));
        assert_eq!(
            serde_json::from_str::<ZSpaceOptimizerFeedbackState>(&json).unwrap(),
            state
        );
    }
    assert!(initialize_zspace_optimizer_feedback(config(0)).is_err());
    assert!(
        initialize_zspace_optimizer_feedback(config(ZSPACE_META_OPTIMIZER_MAX_SAFE_STEP + 1))
            .is_err()
    );
}

#[test]
fn equal_population_windows_do_not_treat_batch_composition_as_regression() {
    let losses = [1.0, 3.0, 3.0, 1.0, 1.0, 3.0, 3.0, 1.0];
    let mut legacy = initialize_zspace_optimizer_feedback(config(1))
        .unwrap()
        .state;
    let mut windowed = initialize_zspace_optimizer_feedback(config(2))
        .unwrap()
        .state;
    let mut legacy_halts = 0;
    for (index, loss) in losses.into_iter().enumerate() {
        let old = advance(&config(1), legacy, loss);
        legacy_halts += usize::from(old.action == ZSpaceOptimizerFeedbackObservationAction::Halt);
        legacy = old.state_after;
        let next = advance(&config(2), windowed, loss);
        if index % 2 == 0 {
            assert_eq!(
                next.action,
                ZSpaceOptimizerFeedbackObservationAction::AwaitWindow
            );
            assert_eq!(next.relative_loss_delta, None);
        } else if index > 1 {
            assert_eq!(next.relative_loss_delta, Some(0.0));
        }
        assert!(!next.state_after.halted);
        assert_eq!(next.state_after.gate, 0.0);
        windowed = next.state_after;
    }
    assert!(legacy_halts > 0);
    assert_eq!(windowed.observation_count, 8);
    let window = windowed.loss_window.unwrap();
    assert_eq!(window.completed_windows, 4);
    assert_eq!(window.previous_mean, Some(2.0));
    assert_eq!(window.mean, None);
}

#[test]
fn real_regression_is_detected_at_the_window_boundary_not_before_it() {
    let configured = ZSpaceOptimizerFeedbackConfig {
        relative_delta_ema_alpha: 1.0,
        ..config(2)
    };
    let mut state = initialize_zspace_optimizer_feedback(configured.clone())
        .unwrap()
        .state;
    for loss in [4.0, 4.0, 2.0, 2.0] {
        state = advance(&configured, state, loss).state_after;
    }
    assert!(state.gate > 0.0);
    let gate = state.gate;
    let pending = advance(&configured, state, 8.0);
    assert_eq!(
        pending.action,
        ZSpaceOptimizerFeedbackObservationAction::AwaitWindow
    );
    assert!(!pending.state_after.halted);
    assert_eq!(pending.state_after.gate, gate);
    let complete = advance(&configured, pending.state_after, 8.0);
    assert_eq!(
        complete.action,
        ZSpaceOptimizerFeedbackObservationAction::Halt
    );
    assert_eq!(complete.relative_loss_delta, Some(3.0));
    assert_eq!(complete.state_after.gate, 0.0);
}

#[test]
fn partial_window_preserves_warmup_and_staleness_semantics() {
    let configured = ZSpaceOptimizerFeedbackConfig {
        warmup_observations: 4,
        ..config(2)
    };
    let mut state = initialize_zspace_optimizer_feedback(configured.clone())
        .unwrap()
        .state;
    for loss in [1.0, 1.0, 4.0, 4.0, 4.0] {
        state = advance(&configured, state, loss).state_after;
    }
    let controlled = control_zspace_optimizer_feedback(ZSpaceOptimizerFeedbackControlRequest {
        config: configured.clone(),
        state,
        target_step: 6,
        proposed_learning_rate_scale: 0.5,
    })
    .unwrap();
    assert_eq!(
        controlled.disposition,
        ZSpaceOptimizerFeedbackControlDisposition::Warmup
    );

    let configured = config(2);
    let mut state = initialize_zspace_optimizer_feedback(configured.clone())
        .unwrap()
        .state;
    for loss in [4.0, 4.0, 2.0, 2.0] {
        state = advance(&configured, state, loss).state_after;
    }
    let original = state.clone();
    state = control_zspace_optimizer_feedback(ZSpaceOptimizerFeedbackControlRequest {
        config: configured.clone(),
        state,
        target_step: 5,
        proposed_learning_rate_scale: 0.5,
    })
    .unwrap()
    .state_after;
    let stale = control_zspace_optimizer_feedback(ZSpaceOptimizerFeedbackControlRequest {
        config: configured,
        state,
        target_step: 6,
        proposed_learning_rate_scale: 0.5,
    })
    .unwrap();
    assert_eq!(
        stale.disposition,
        ZSpaceOptimizerFeedbackControlDisposition::Stale
    );
    assert_eq!(stale.applied_learning_rate_scale, 1.0);
    assert_eq!(stale.state_after.loss_window, original.loss_window);
    assert_eq!(
        stale.state_after.observation_count,
        original.observation_count
    );
    assert_eq!(stale.state_after.gate, original.gate);
}

#[test]
fn full_pass_window_roundtrips_and_resumes_from_observation_37() {
    let configured = config(80);
    let mut state = initialize_zspace_optimizer_feedback(configured.clone())
        .unwrap()
        .state;
    let mut restored = None;
    for step in 1..=400 {
        let loss = f64::from(0.001 + ((step * 37) % 1001) as f32 / 517.);
        let next = advance(&configured, state, loss);
        if let Some(previous) = restored.take() {
            assert_eq!(advance(&configured, previous, loss), next, "step {step}");
        }
        state = next.state_after;
        let json = serde_json::to_string(&state).unwrap();
        let decoded: ZSpaceOptimizerFeedbackState = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, state);
        let checked = restore_zspace_optimizer_feedback(ZSpaceOptimizerFeedbackRestoreRequest {
            config: configured.clone(),
            state: decoded,
        })
        .unwrap();
        if step >= 37 {
            restored = Some(checked.state);
        }
        if step == 37 {
            assert_eq!(state.loss_window.as_ref().unwrap().observations, 37);
        }
    }
    assert_eq!(state.loss_window.unwrap().completed_windows, 5);
}

#[test]
fn corrupted_partial_windows_and_silent_reconfiguration_are_rejected() {
    let configured = config(2);
    let state = advance(
        &configured,
        initialize_zspace_optimizer_feedback(configured.clone())
            .unwrap()
            .state,
        1.0,
    )
    .state_after;
    for (field, value) in [
        ("observations_per_window", serde_json::json!(4)),
        ("observations", serde_json::json!(0)),
        ("completed_windows", serde_json::json!(1)),
        ("mean", serde_json::json!(2.0)),
        ("previous_mean", serde_json::json!(1.0)),
    ] {
        let mut encoded = serde_json::to_value(&state).unwrap();
        encoded["loss_window"][field] = value;
        let decoded = serde_json::from_value(encoded).unwrap();
        assert!(
            restore_zspace_optimizer_feedback(ZSpaceOptimizerFeedbackRestoreRequest {
                config: configured.clone(),
                state: decoded,
            })
            .is_err(),
            "{field}"
        );
    }
    for configured in [config(1), config(4)] {
        assert!(
            restore_zspace_optimizer_feedback(ZSpaceOptimizerFeedbackRestoreRequest {
                config: configured,
                state: state.clone(),
            })
            .is_err()
        );
    }
}
