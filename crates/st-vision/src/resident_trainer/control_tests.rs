use super::*;
use st_core::runtime::zspace_optimizer::{
    transition_zspace_meta_optimizer, ZSpaceMetaObservation, ZSpaceMetaOptimizerConfig,
    ZSpaceMetaOptimizerState, ZSpaceMetaOptimizerStepRequest,
};
use std::collections::BTreeMap;

pub(super) fn report(step: u64, scale: f64) -> ZSpaceMetaOptimizerStepReport {
    let mut state = ZSpaceMetaOptimizerState::zeros(2);
    state.step = step - 1;
    transition_zspace_meta_optimizer(ZSpaceMetaOptimizerStepRequest {
        config: ZSpaceMetaOptimizerConfig {
            dimension: 2,
            topos_control_gain: 1.,
            ..Default::default()
        },
        state,
        observation: ZSpaceMetaObservation {
            gradient: vec![0.1, -0.2],
            telemetry: BTreeMap::from([("topos.training_hints.learning_rate_scale".into(), scale)]),
            ..Default::default()
        },
    })
    .unwrap()
}

fn model_json(trainer: &ResidentVisionTrainer<TensorVisionDataset>) -> Value {
    serde_json::to_value(checkpoint(trainer).model()).unwrap()
}

#[test]
fn plain_state_bytes_are_unchanged_and_control_clock_requires_v2() {
    let legacy = r#"{"epoch":0,"accepted_updates":0,"rejected_updates":0,"learning_rate":{"kind":"constant","rate":0.125}}"#;
    let mut state: ResidentVisionTrainingState = serde_json::from_str(legacy).unwrap();
    assert_eq!(serde_json::to_string(&state).unwrap(), legacy);
    assert_eq!(state.schema(), SCHEMA);
    state.parameter_control = state
        .parameter_control
        .preview(&zspace_parameter_control_from_report(&report(1, 1.)).unwrap())
        .unwrap()
        .0;
    assert_eq!(state.schema(), CONTROLLED_SCHEMA);
    assert_eq!(state.parameter_control.absolute_learning_rate_scale(), 1.);
    assert_eq!(state.parameter_control.source_meta_step(), Some(1));
}

#[test]
fn actual_controlled_sgd_matches_explicit_rate_without_compounding() {
    let Some(device) = device() else { return };
    let make = |rate| {
        ResidentVisionTrainer::new(
            &model(),
            device.clone(),
            loader(&device, 4, 20),
            DATA_ID,
            ResidentLearningRate::Constant { rate },
        )
        .unwrap()
    };
    let mut controlled = make(0.001);
    let mut reference = make(0.0005);
    let mut unscaled = make(0.001);
    let control = report(1, 0.5);
    let receipt = controlled
        .apply_zspace_meta_optimizer_report(&control)
        .unwrap();
    assert!(receipt.changed);
    let saved = checkpoint(&controlled).to_json().unwrap();
    assert!(
        !controlled
            .apply_zspace_meta_optimizer_report(&control)
            .unwrap()
            .changed
    );
    assert_eq!(checkpoint(&controlled).to_json().unwrap(), saved);
    assert_eq!(steps(&mut controlled, 14), steps(&mut reference, 14));
    steps(&mut unscaled, 14);
    assert_eq!(model_json(&controlled), model_json(&reference));
    assert_ne!(model_json(&controlled), model_json(&unscaled));

    let before = checkpoint(&controlled).to_json().unwrap();
    let mut tampered = serde_json::to_value(&control).unwrap();
    tampered["topos_control"]["learning_rate_scale"] = json!(0.75);
    for bad in [
        "{}".into(),
        "{".into(),
        tampered.to_string(),
        serde_json::to_string(&report(1, 0.75)).unwrap(),
    ] {
        assert!(controlled
            .apply_zspace_meta_optimizer_report_json(&bad)
            .is_err());
        assert_eq!(checkpoint(&controlled).to_json().unwrap(), before);
    }
    controlled.submit_next().unwrap();
    let pending_state = controlled.state().clone();
    assert!(controlled
        .apply_zspace_meta_optimizer_report(&report(2, 1.))
        .is_err());
    assert_eq!(*controlled.state(), pending_state);
    controlled.settle().unwrap();
    controlled
        .apply_zspace_meta_optimizer_report(&report(2, 1.))
        .unwrap();
    let before = checkpoint(&controlled).to_json().unwrap();
    assert!(controlled
        .apply_zspace_meta_optimizer_report(&control)
        .is_err());
    assert_eq!(checkpoint(&controlled).to_json().unwrap(), before);
    assert_eq!(checkpoint(&controlled).schema, CONTROLLED_SCHEMA);
}

#[test]
fn controlled_restart_rejection_and_invalid_restore_are_atomic() {
    let Some(device) = device() else { return };
    let mut input = loader(&device, 4, 20);
    input.enable_shuffle(false);
    let mut trainer =
        ResidentVisionTrainer::new(&model(), device.clone(), input, DATA_ID, rate(true)).unwrap();
    let original = checkpoint(&trainer);
    trainer
        .apply_zspace_meta_optimizer_report(&report(1, 0.5))
        .unwrap();
    steps(&mut trainer, 3);
    let before = checkpoint(&trainer);
    let submitted = trainer.submit_next().unwrap();
    assert_eq!(submitted.labels, vec![Some("6".into()), Some("7".into())]);
    assert!(!trainer.settle().unwrap().accepted);
    let saved = checkpoint(&trainer);
    assert_eq!(saved.trainer.learning_rate, before.trainer.learning_rate);
    assert_eq!(
        saved.trainer.parameter_control,
        before.trainer.parameter_control
    );
    let a = serde_json::to_value(&saved.model).unwrap();
    let b = serde_json::to_value(&before.model).unwrap();
    assert_eq!(a["backbone"]["parameters"], b["backbone"]["parameters"]);
    assert_eq!(a["head"], b["head"]);
    assert_eq!(saved.input.position(), 8);
    let mut restored = ResidentVisionTrainer::from_checkpoint(
        device.clone(),
        loader(&device, 4, 20),
        DATA_ID,
        &saved,
    )
    .unwrap();
    assert_eq!(
        checkpoint(&restored).to_json().unwrap(),
        saved.to_json().unwrap()
    );
    assert_eq!(steps(&mut trainer, 23), steps(&mut restored, 23));
    assert_eq!(
        checkpoint(&trainer).to_json().unwrap(),
        checkpoint(&restored).to_json().unwrap()
    );

    let current = checkpoint(&trainer);
    let mut downgraded = current.clone();
    downgraded.schema = SCHEMA.into();
    let mut corrupted = current.clone();
    corrupted.trainer.parameter_control = ZSpaceParameterControlState::default();
    let mut invalid_control = current.clone();
    invalid_control.trainer.parameter_control = serde_json::from_value(json!({
        "absolute_learning_rate_scale": 0.5, "source_meta_step": null,
    }))
    .unwrap();
    invalid_control.trainer_sha256 = digest(&invalid_control.trainer).unwrap();
    for bad in [downgraded, corrupted, invalid_control] {
        assert!(trainer.restore_checkpoint(&bad).is_err());
        assert_eq!(
            checkpoint(&trainer).to_json().unwrap(),
            current.to_json().unwrap()
        );
    }
    assert!(restored
        .apply_zspace_meta_optimizer_report(&report(1, 0.75))
        .is_err());
    trainer.restore_checkpoint(&original).unwrap();
    assert_eq!(
        checkpoint(&trainer).to_json().unwrap(),
        original.to_json().unwrap()
    );
}

#[test]
fn unrepresentable_rate_rejects_control_without_consuming_a_batch() {
    let Some(device) = device() else { return };
    for (nominal, scale) in [(f32::MAX, 1.25), (f32::from_bits(1), 0.5)] {
        let mut trainer = ResidentVisionTrainer::new(
            &model(),
            device.clone(),
            loader(&device, 4, 20),
            DATA_ID,
            ResidentLearningRate::Constant { rate: nominal },
        )
        .unwrap();
        let saved = checkpoint(&trainer).to_json().unwrap();
        assert!(trainer
            .apply_zspace_meta_optimizer_report(&report(1, scale))
            .is_err());
        assert_eq!(checkpoint(&trainer).to_json().unwrap(), saved);
        assert!(!trainer.has_pending_update());
    }
}
