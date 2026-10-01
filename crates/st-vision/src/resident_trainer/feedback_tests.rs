use super::*;
use st_core::runtime::zspace_optimizer_feedback::{
    control_zspace_optimizer_feedback, initialize_zspace_optimizer_feedback,
    observe_zspace_optimizer_feedback, ZSpaceOptimizerFeedbackControlRequest,
    ZSpaceOptimizerFeedbackObservation, ZSpaceOptimizerFeedbackObserveRequest,
};

fn config() -> ZSpaceOptimizerFeedbackConfig {
    ZSpaceOptimizerFeedbackConfig {
        warmup_observations: 1,
        recovery_rate: 0.5,
        relative_delta_ema_alpha: 1.,
        recovery_threshold: 0.,
        ..Default::default()
    }
}

fn controlled(
    device: &TensorDevice,
    scheduled: bool,
) -> ResidentVisionTrainer<TensorVisionDataset> {
    let mut trainer = ResidentVisionTrainer::new(
        &model(),
        device.clone(),
        loader(device, 4, 20),
        DATA_ID,
        rate(scheduled),
    )
    .unwrap();
    trainer.enable_zspace_optimizer_feedback(config()).unwrap();
    trainer
        .apply_zspace_meta_optimizer_report(&control_tests::report(1, 0.5))
        .unwrap();
    trainer
}

#[test]
fn feedback_configuration_is_opt_in_and_validated_before_execution() {
    let legacy = ResidentVisionTrainerConfig::default().to_json().unwrap();
    assert!(!legacy.contains("optimizer_feedback"));
    let mut value: Value = serde_json::from_str(&legacy).unwrap();
    value["optimizer_feedback"] = json!({});
    assert_eq!(
        ResidentVisionTrainerConfig::from_json(&value.to_string())
            .unwrap()
            .optimizer_feedback,
        Some(ZSpaceOptimizerFeedbackConfig::default())
    );
    value["optimizer_feedback"] = json!({"maximum_gate": 2.});
    assert!(ResidentVisionTrainerConfig::from_json(&value.to_string()).is_err());
    value["optimizer_feedback"] = json!({"unknown": true});
    assert!(ResidentVisionTrainerConfig::from_json(&value.to_string()).is_err());
}

#[test]
fn actual_losses_drive_the_core_gate_and_all_weights_match_explicit_rates() {
    let Some(device) = device() else { return };
    let mut trainer = controlled(&device, false);
    let mut reference = ResidentVisionTrainer::new(
        &model(),
        device.clone(),
        loader(&device, 4, 20),
        DATA_ID,
        rate(false),
    )
    .unwrap();
    let mut feedback = initialize_zspace_optimizer_feedback(config())
        .unwrap()
        .state;
    let mut changed = 0;
    let mut rejected = 0;
    for step in 1..=30 {
        let control = control_zspace_optimizer_feedback(ZSpaceOptimizerFeedbackControlRequest {
            config: config(),
            state: feedback.clone(),
            target_step: step,
            proposed_learning_rate_scale: 0.5,
        })
        .unwrap();
        let expected_rate = 0.001 * control.applied_learning_rate_scale as f32;
        reference.state.learning_rate = ResidentLearningRate::Constant {
            rate: expected_rate,
        };
        let before = trainer.state.clone();
        let submitted = trainer.submit_next().unwrap();
        assert_eq!(submitted.learning_rate, expected_rate);
        changed += usize::from(expected_rate != 0.001);
        assert_eq!(trainer.state.optimizer_feedback, before.optimizer_feedback);
        // A missing observation cannot partially commit the accepted update.
        assert!(trainer
            .finish_settlement(Ok(submitted.attempted_revision), None)
            .is_err());
        assert!(trainer.has_pending_update());
        assert_eq!(trainer.state.optimizer_feedback, before.optimizer_feedback);
        let ref_submitted = reference.submit_next().unwrap();
        assert_eq!(submitted.labels, ref_submitted.labels);
        let outcome = trainer.settle().unwrap();
        assert_eq!(outcome, reference.settle().unwrap());
        feedback = control.state_after;
        if outcome.accepted {
            let loss = submitted.loss.snapshot().unwrap().read().unwrap()[0];
            let observed =
                observe_zspace_optimizer_feedback(ZSpaceOptimizerFeedbackObserveRequest {
                    config: config(),
                    state: feedback,
                    observation: ZSpaceOptimizerFeedbackObservation {
                        step,
                        max_steps: None,
                        epoch: Some(submitted.epoch as f64),
                        loss: f64::from(loss),
                        grad_norm: None,
                        learning_rate: Some(f64::from(expected_rate)),
                    },
                })
                .unwrap();
            feedback = observed.state_after;
        } else {
            rejected += 1;
            assert_eq!(trainer.state.learning_rate, before.learning_rate);
            assert_eq!(
                feedback.last_loss,
                before.optimizer_feedback.unwrap().state().last_loss
            );
        }
        assert_eq!(
            trainer.state.optimizer_feedback().unwrap().state(),
            &feedback
        );
        assert_eq!(
            checkpoint(&trainer).model.to_json().unwrap(),
            checkpoint(&reference).model.to_json().unwrap()
        );
    }
    assert!(
        changed > 0,
        "real losses must actually change the applied rate"
    );
    assert!(rejected > 0);
    assert_eq!(feedback.observation_count, trainer.state.accepted_updates);
    assert_eq!(feedback.control_step, 30);
}

#[test]
fn feedback_restart_preserves_loss_history_rates_and_every_weight() {
    let Some(device) = device() else { return };
    for scheduled in [false, true] {
        let mut trainer = controlled(&device, scheduled);
        let initial = checkpoint(&trainer);
        assert_eq!(initial.schema, FEEDBACK_SCHEMA);
        assert!(trainer.enable_zspace_optimizer_feedback(config()).is_err());
        steps(&mut trainer, 37);
        let saved = checkpoint(&trainer).to_json().unwrap();
        let saved = VisionTrainingCheckpoint::from_json(&saved).unwrap();
        let mut restored = ResidentVisionTrainer::from_checkpoint(
            device.clone(),
            loader(&device, 4, 20),
            DATA_ID,
            &saved,
        )
        .unwrap();
        assert_eq!(steps(&mut trainer, 63), steps(&mut restored, 63));
        let current = checkpoint(&trainer);
        assert_eq!(
            current.to_json().unwrap(),
            checkpoint(&restored).to_json().unwrap()
        );
        assert_eq!(
            current
                .trainer
                .optimizer_feedback()
                .unwrap()
                .state()
                .control_step,
            100
        );
        assert!(trainer.enable_zspace_optimizer_feedback(config()).is_err());

        let mut bad_clock = current.clone();
        let mut value =
            serde_json::to_value(bad_clock.trainer.optimizer_feedback.as_ref().unwrap()).unwrap();
        value["state"]["control_step"] = json!(101);
        bad_clock.trainer.optimizer_feedback = Some(serde_json::from_value(value).unwrap());
        bad_clock.trainer_sha256 = digest(&bad_clock.trainer).unwrap();
        let mut bad_config = current.clone();
        let mut value =
            serde_json::to_value(bad_config.trainer.optimizer_feedback.as_ref().unwrap()).unwrap();
        value["config"]["maximum_gate"] = json!(0.);
        bad_config.trainer.optimizer_feedback = Some(serde_json::from_value(value).unwrap());
        bad_config.trainer_sha256 = digest(&bad_config.trainer).unwrap();
        let mut downgraded = current.clone();
        downgraded.schema = CONTROLLED_SCHEMA.into();
        for corrupt in [bad_clock, bad_config, downgraded] {
            assert!(trainer.restore_checkpoint(&corrupt).is_err());
            assert_eq!(
                checkpoint(&trainer).to_json().unwrap(),
                current.to_json().unwrap()
            );
        }
        trainer.restore_checkpoint(&initial).unwrap();
        assert_eq!(
            checkpoint(&trainer).to_json().unwrap(),
            initial.to_json().unwrap()
        );
    }
}

#[test]
fn shared_client_configuration_connects_feedback_without_client_policy() {
    let Some(device) = device() else { return };
    let input = loader(&device, 4, 20);
    let config = ResidentVisionTrainerConfig {
        model: model().config().clone(),
        num_classes: 2,
        batch_size: 2,
        model_seed: 43,
        shuffle_seed: 17,
        shuffle: true,
        learning_rate: rate(false),
        optimizer_feedback: Some(config()),
    };
    let mut trainer = ResidentVisionTrainer::from_dataset(
        &config,
        device.clone(),
        Arc::clone(&input.dataset),
        input.pipeline.clone(),
        DATA_ID,
    )
    .unwrap();
    trainer
        .apply_zspace_meta_optimizer_report(&control_tests::report(1, 0.5))
        .unwrap();
    let mut reference = controlled(&device, false);
    assert_eq!(steps(&mut trainer, 12), steps(&mut reference, 12));
    assert_eq!(
        checkpoint(&trainer).to_json().unwrap(),
        checkpoint(&reference).to_json().unwrap()
    );

    let mut malformed = ResidentVisionTrainer::new(
        &model(),
        device.clone(),
        loader(&device, 5, 20),
        DATA_ID,
        rate(false),
    )
    .unwrap();
    malformed
        .enable_zspace_optimizer_feedback(config.optimizer_feedback.unwrap())
        .unwrap();
    let before = checkpoint(&malformed).to_json().unwrap();
    assert!(malformed.submit_next().is_err());
    assert!(!malformed.has_pending_update());
    assert_eq!(checkpoint(&malformed).to_json().unwrap(), before);
}
