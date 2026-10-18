use super::*;
use crate::geometry_runtime::{AdamWConfig, AdamWStep, GlobalNormClip, NativeGeometryAdamW};
use sha2::{Digest, Sha256};

fn config() -> CausalLmTrainingConfig {
    CausalLmTrainingConfig {
        context_length: 4,
        microbatch_windows: 2,
        accumulation_steps: 2,
        epochs: 2,
        learning_rate: 0.001,
        weight_decay: 0.01,
        label_smoothing: 0.1,
        repetition: None,
    }
}

fn documents() -> Vec<Vec<usize>> {
    vec![
        vec![1, 2, 3],
        vec![4, 5, 6, 7, 8, 9, 10, 0, 1],
        vec![2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2],
    ]
}

fn create(config: CausalLmTrainingConfig) -> NativeCausalLmTrainer {
    create_with_readout(config, CausalLmReadoutAccumulator::F32)
}

fn create_with_readout(
    config: CausalLmTrainingConfig,
    accumulator: CausalLmReadoutAccumulator,
) -> NativeCausalLmTrainer {
    let base = load().with_readout_accumulator(accumulator);
    let artifact = trainable(&base);
    let optimizer = NativeGeometryAdamW::new(
        &artifact,
        AdamWConfig::new(0.9, 0.999, 1e-8).unwrap(),
        Some(GlobalNormClip::new(0.1).unwrap()),
    )
    .unwrap();
    NativeCausalLmTrainer::new(
        base.with_geometry(artifact).unwrap(),
        optimizer,
        documents(),
        None,
        config,
    )
    .unwrap()
}

#[test]
fn training_resume_requires_same_readout_policy_and_preserves_default_bytes() {
    for accumulator in [
        CausalLmReadoutAccumulator::F32,
        CausalLmReadoutAccumulator::F64,
    ] {
        let mut trainer = create_with_readout(config(), accumulator);
        trainer.step().unwrap();
        let payload = trainer.to_json().unwrap();
        let id = trainer.checkpoint_id().unwrap();
        assert_eq!(
            payload.contains("readout_accumulator"),
            accumulator == CausalLmReadoutAccumulator::F64
        );
        let base = load().with_readout_accumulator(accumulator);
        let other =
            load().with_readout_accumulator(if accumulator == CausalLmReadoutAccumulator::F32 {
                CausalLmReadoutAccumulator::F64
            } else {
                CausalLmReadoutAccumulator::F32
            });
        assert!(NativeCausalLmTrainer::from_json(
            other.clone(),
            documents(),
            None,
            payload.as_bytes(),
            &id
        )
        .is_err());
        assert!(NativeCausalLm::from_training_checkpoint(other, payload.as_bytes(), &id).is_err());
        let mut restored = NativeCausalLmTrainer::from_json(
            base.clone(),
            documents(),
            None,
            payload.as_bytes(),
            &id,
        )
        .unwrap();
        assert_eq!(restored.to_json().unwrap(), payload);
        assert_eq!(
            NativeCausalLm::from_training_checkpoint(base, payload.as_bytes(), &id)
                .unwrap()
                .readout_accumulator(),
            accumulator
        );
        while !trainer.exhausted() {
            let a = trainer.step().unwrap().unwrap();
            let b = restored.step().unwrap().unwrap();
            assert_eq!(
                serde_json::to_string(&a).unwrap(),
                serde_json::to_string(&b).unwrap()
            );
            assert_eq!(trainer.to_json().unwrap(), restored.to_json().unwrap());
        }
    }
}

#[test]
fn ordered_groups_match_manual_native_accumulation_and_keep_partial_epoch_group() {
    let mut trainer = create(config());
    let mut model = trainer.model().clone();
    let mut optimizer = trainer.optimizer().clone();
    let docs = documents();
    assert_eq!(
        (
            trainer.plan().windows().len(),
            trainer.updates_per_epoch(),
            trainer.total_updates()
        ),
        (5, 2, 4)
    );
    for update in 0..4 {
        let start = if update % 2 == 0 { 0 } else { 4 };
        let end = if start == 0 { 4 } else { 5 };
        let mut accumulator = model
            .geometry_accumulator_with_objective(CausalLmObjective {
                label_smoothing: 0.1,
                repetition: None,
            })
            .unwrap();
        for windows in trainer.plan().windows()[start..end].chunks(2) {
            let ids: Vec<_> = windows
                .iter()
                .flat_map(|w| {
                    docs[w.document as usize][w.start as usize..(w.start + w.length) as usize]
                        .iter()
                        .copied()
                })
                .collect();
            accumulator
                .accumulate_loss(&model, &ids, [windows.len(), 4])
                .unwrap();
        }
        (model, optimizer) = accumulator
            .step_adamw(&model, &optimizer, AdamWStep::new(0.001, 0.01).unwrap())
            .unwrap();
        let report = trainer.step().unwrap().unwrap();
        assert_eq!(
            (report.epoch, report.window_start, report.window_end),
            (update / 2, start, end)
        );
        assert_eq!(report.loss, accumulator.loss().unwrap());
        assert_eq!(report.target_tokens, if start == 0 { 12 } else { 3 });
        assert_eq!(
            trainer.model().geometry_artifact().unwrap().to_json(),
            model.geometry_artifact().unwrap().to_json()
        );
        assert_eq!(
            trainer.optimizer().to_json().unwrap(),
            optimizer.to_json().unwrap()
        );
    }
    assert!(trainer.exhausted());
    let saved = trainer.to_json().unwrap();
    assert!(trainer.step().unwrap().is_none());
    assert_eq!(trainer.to_json().unwrap(), saved);
}

#[test]
fn checkpoint_restores_exact_cursor_and_remaining_updates() {
    let mut trainer = create(config());
    trainer.step().unwrap();
    let json = trainer.to_json().unwrap();
    let mut restored = NativeCausalLmTrainer::from_json(
        load(),
        documents(),
        None,
        json.as_bytes(),
        &trainer.checkpoint_id().unwrap(),
    )
    .unwrap();
    assert_eq!(restored.to_json().unwrap(), json);
    while !trainer.exhausted() {
        let a = trainer.step().unwrap().unwrap();
        let b = restored.step().unwrap().unwrap();
        assert_eq!(
            serde_json::to_string(&a).unwrap(),
            serde_json::to_string(&b).unwrap()
        );
        assert_eq!(trainer.to_json().unwrap(), restored.to_json().unwrap());
    }
}

#[test]
fn checkpoints_bind_dropped_tail_content_labels_recipe_and_clocks() {
    let mut trainer = create(config());
    trainer.step().unwrap();
    let json = trainer.to_json().unwrap();
    let id = trainer.checkpoint_id().unwrap();
    let mut docs = documents();
    docs[0][0] = 2;
    assert!(NativeCausalLmTrainer::from_json(load(), docs, None, json.as_bytes(), &id).is_err());
    let mut docs = documents();
    docs[1][8] = 2;
    assert!(NativeCausalLmTrainer::from_json(load(), docs, None, json.as_bytes(), &id).is_err());
    assert!(
        NativeCausalLmTrainer::from_json(load(), documents(), None, json.as_bytes(), "wrong")
            .is_err()
    );
    let record: serde_json::Value = serde_json::from_str(&json).unwrap();
    for (key, value) in [
        ("updates", json!(99)),
        ("updates", json!(0)),
        ("initial_optimizer_revision", json!(u64::MAX)),
    ] {
        let mut changed = record.clone();
        changed[key] = value;
        let bytes = serde_json::to_vec(&changed).unwrap();
        let id = format!("sha256:{:x}", Sha256::digest(&bytes));
        assert!(NativeCausalLmTrainer::from_json(load(), documents(), None, &bytes, &id).is_err());
    }
    let mut changed = record;
    changed["config"]["microbatch_windows"] = json!(0);
    let bytes = serde_json::to_vec(&changed).unwrap();
    let id = format!("sha256:{:x}", Sha256::digest(&bytes));
    assert!(NativeCausalLmTrainer::from_json(load(), documents(), None, &bytes, &id).is_err());
}

#[test]
fn checkpoint_rejects_reset_placement_clock_after_uniform_updates() {
    let mut trainer = create(config());
    trainer.step().unwrap();
    let mut record: serde_json::Value = serde_json::from_str(&trainer.to_json().unwrap()).unwrap();
    let mut optimizer: serde_json::Value =
        serde_json::from_str(record["optimizer_json"].as_str().unwrap()).unwrap();
    optimizer["slots"][0]["steps"] = json!(0);
    for field in ["first", "second"] {
        for value in optimizer["slots"][0][field].as_array_mut().unwrap() {
            *value = json!(0.);
        }
    }
    let payload = serde_json::to_string(&optimizer).unwrap();
    record["optimizer_id"] = json!(format!("sha256:{:x}", Sha256::digest(payload.as_bytes())));
    record["optimizer_json"] = json!(payload);
    let bytes = serde_json::to_vec(&record).unwrap();
    let id = format!("sha256:{:x}", Sha256::digest(&bytes));
    assert!(NativeCausalLmTrainer::from_json(load(), documents(), None, &bytes, &id).is_err());
}

#[test]
fn continuation_preserves_independently_aged_initial_placements() {
    let original = create(config());
    let mut saved: serde_json::Value =
        serde_json::from_str(&original.optimizer().to_json().unwrap()).unwrap();
    saved["revision"] = json!(2);
    for (slot, clock) in saved["slots"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .zip([2, 1, 0])
    {
        slot["steps"] = json!(clock);
    }
    let bytes = serde_json::to_vec(&saved).unwrap();
    let id = format!("sha256:{:x}", Sha256::digest(&bytes));
    let optimizer =
        NativeGeometryAdamW::from_json(&bytes, &id, original.model().geometry_artifact().unwrap())
            .unwrap();
    let mut trainer = NativeCausalLmTrainer::new(
        original.model().clone(),
        optimizer,
        documents(),
        None,
        config(),
    )
    .unwrap();
    trainer.step().unwrap();
    assert_eq!(
        trainer
            .optimizer()
            .state()
            .slots
            .iter()
            .map(|s| s.steps)
            .collect::<Vec<_>>(),
        [3, 2, 1]
    );
    let mut restored = NativeCausalLmTrainer::from_json(
        load(),
        documents(),
        None,
        trainer.to_json().unwrap().as_bytes(),
        &trainer.checkpoint_id().unwrap(),
    )
    .unwrap();
    trainer.step().unwrap();
    restored.step().unwrap();
    assert_eq!(trainer.to_json().unwrap(), restored.to_json().unwrap());
}

#[test]
fn invalid_inputs_and_failed_numerics_leave_sources_and_cursor_unchanged() {
    let trainer = create(config());
    let before = trainer.to_json().unwrap();
    let mut docs = documents();
    docs[0][0] = 11;
    assert!(NativeCausalLmTrainer::new(
        trainer.model().clone(),
        trainer.optimizer().clone(),
        docs,
        None,
        config()
    )
    .is_err());
    let labels = documents().iter().map(|d| vec![-100; d.len()]).collect();
    assert!(NativeCausalLmTrainer::new(
        trainer.model().clone(),
        trainer.optimizer().clone(),
        documents(),
        Some(labels),
        config()
    )
    .is_err());
    let huge = [NativeCausalLmTrainer::MAX_TOKENS + 1];
    assert!(NativeCausalLmTrainer::document_plan(trainer.model(), config(), &huge, false).is_err());
    assert_eq!(trainer.to_json().unwrap(), before);
    let mut bad = config();
    bad.learning_rate = f32::MAX;
    bad.weight_decay = f32::MAX;
    let mut failed = create(bad);
    let before = failed.to_json().unwrap();
    assert!(failed.step().is_err());
    assert_eq!(failed.to_json().unwrap(), before);
}

#[test]
fn labels_keep_selected_target_weighting_and_zero_rate_keeps_optimizer_clock() {
    let original = create(config());
    let docs = documents();
    let mut labels: Vec<Vec<i64>> = docs
        .iter()
        .map(|d| d.iter().map(|&x| x as i64).collect())
        .collect();
    labels[1][1] = -100;
    labels[2][1] = -100;
    let mut recipe = config();
    recipe.learning_rate = 0.;
    let mut trainer = NativeCausalLmTrainer::new(
        original.model().clone(),
        original.optimizer().clone(),
        docs,
        Some(labels.clone()),
        recipe,
    )
    .unwrap();
    let id = trainer.model().geometry_artifact_id().unwrap().to_string();
    assert_eq!(trainer.step().unwrap().unwrap().target_tokens, 10);
    assert_eq!(trainer.optimizer().revision(), 0);
    assert_eq!(trainer.model().geometry_artifact_id(), Some(id.as_str()));
    let restored = NativeCausalLmTrainer::from_json(
        load(),
        documents(),
        Some(labels.clone()),
        trainer.to_json().unwrap().as_bytes(),
        &trainer.checkpoint_id().unwrap(),
    )
    .unwrap();
    assert_eq!(restored.completed_updates(), 1);
    labels[1][1] = 5;
    assert!(NativeCausalLmTrainer::from_json(
        load(),
        documents(),
        Some(labels),
        trainer.to_json().unwrap().as_bytes(),
        &trainer.checkpoint_id().unwrap()
    )
    .is_err());
}
