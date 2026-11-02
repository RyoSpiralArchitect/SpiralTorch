use super::*;
use sha2::{Digest, Sha256};

fn session_config(mode: CausalLmAdaptationMode) -> CausalLmTrainingSessionConfig {
    CausalLmTrainingSessionConfig {
        context_length: 4,
        epochs: 2,
        learning_rate: 0.2,
        mode,
    }
}
fn documents() -> Vec<Vec<usize>> {
    vec![
        vec![1, 2, 3],
        vec![1, 2, 3, 4, 5, 6, 7, 8],
        vec![3, 4, 5, 6],
    ]
}
fn geometry(base: &NativeCausalLm) -> GeometryAdapterArtifact {
    let mut record = artifact(base, "transformer.h.0.attn", 0.2).record().clone();
    record.training = true;
    record.placements[0].training = true;
    for tensor in &mut record.placements[0].tensors {
        tensor.requires_grad = tensor.name == "gate";
    }
    GeometryAdapterArtifact::create(record).unwrap()
}
fn adapted(base: &NativeCausalLm, mode: CausalLmAdaptationMode) -> NativeCausalLm {
    match mode {
        CausalLmAdaptationMode::Lora => base.with_lora(0, 2, 3.).unwrap(),
        CausalLmAdaptationMode::Geometry => base.with_geometry(geometry(base)).unwrap(),
        CausalLmAdaptationMode::Joint => base
            .with_geometry(geometry(base))
            .unwrap()
            .with_lora(0, 2, 3.)
            .unwrap(),
    }
}
fn id(json: &str) -> String {
    format!("sha256:{:x}", Sha256::digest(json.as_bytes()))
}

#[test]
fn session_inference_binds_both_owners_cursor_and_recorded_corpus_without_resuming() {
    let base = load();
    for mode in [
        CausalLmAdaptationMode::Lora,
        CausalLmAdaptationMode::Geometry,
        CausalLmAdaptationMode::Joint,
    ] {
        let model = adapted(&base, mode).with_readout_accumulator(CausalLmReadoutAccumulator::F64);
        let mut session =
            NativeCausalLmTrainingSession::new(model, documents(), session_config(mode)).unwrap();
        let step = session.step().unwrap().unwrap();
        let (json, pin) = session.checkpoint_with_id().unwrap();
        assert_eq!(json, session.to_json().unwrap());
        assert_eq!(pin, session.checkpoint_id().unwrap());
        let lora = step.post_lora_artifact_id.as_deref();
        let geom = step.post_geometry_artifact_id.as_deref();
        let restore = |base,
                       json: &str,
                       pin: &str,
                       u,
                       data: &str,
                       l: Option<&str>,
                       g: Option<&str>| {
            NativeCausalLm::from_session_checkpoint_at(base, json.as_bytes(), pin, u, data, l, g)
        };
        let data = session.data_id();
        let restored = restore(base.clone(), &json, &pin, 1, data, lora, geom).unwrap();
        assert_eq!(
            restored.readout_accumulator(),
            CausalLmReadoutAccumulator::F64
        );
        assert_eq!(
            restored.logits(&[1, 2, 3], [1, 3], false).unwrap(),
            session.model().logits(&[1, 2, 3], [1, 3], false).unwrap()
        );
        assert!(restore(base.clone(), &json, &pin, 2, data, lora, geom).is_err());
        assert!(restore(base.clone(), &json, &pin, 1, &id("other data"), lora, geom).is_err());
        assert!(restore(base.clone(), &json, &id("other bytes"), 1, data, lora, geom).is_err());
        assert!(restore(base.clone(), &json, &pin, 1, data, Some("wrong"), geom).is_err());
        assert!(restore(base.clone(), &json, &pin, 1, data, lora, Some("wrong")).is_err());
        assert!(restore(
            adapted(&base, CausalLmAdaptationMode::Lora),
            &json,
            &pin,
            1,
            data,
            lora,
            geom
        )
        .is_err());
        let mut invalid: serde_json::Value = serde_json::from_str(&json).unwrap();
        invalid["updates"] = serde_json::json!(u64::MAX);
        let forged = invalid.to_string();
        assert!(restore(
            base.clone(),
            &forged,
            &id(&forged),
            u64::MAX,
            data,
            lora,
            geom
        )
        .is_err());
    }
}

#[test]
fn session_save_resume_owns_policy_cursor_and_both_groups_exactly() {
    for readout in [
        CausalLmReadoutAccumulator::F32,
        CausalLmReadoutAccumulator::F64,
    ] {
        for mode in [
            CausalLmAdaptationMode::Lora,
            CausalLmAdaptationMode::Geometry,
            CausalLmAdaptationMode::Joint,
        ] {
            let base = load().with_readout_accumulator(readout);
            let mut whole = NativeCausalLmTrainingSession::new(
                adapted(&base, mode),
                documents(),
                session_config(mode),
            )
            .unwrap();
            assert_eq!(whole.updates_per_epoch(), 3);
            assert_eq!(whole.total_updates(), 6);
            assert_eq!(whole.next_window(), 0);
            whole.step().unwrap();
            let json = whole.to_json().unwrap();
            // Unlike adapter-only artifacts, this continuation artifact owns readout.
            let mut resumed = NativeCausalLmTrainingSession::from_json(
                load(),
                documents(),
                json.as_bytes(),
                &whole.checkpoint_id().unwrap(),
            )
            .unwrap();
            assert_eq!(resumed.model().readout_accumulator(), readout);
            assert_eq!(resumed.to_json().unwrap(), json);
            while !whole.exhausted() {
                let a = whole.step().unwrap().unwrap();
                let b = resumed.step().unwrap().unwrap();
                assert_eq!(
                    serde_json::to_string(&a).unwrap(),
                    serde_json::to_string(&b).unwrap()
                );
                assert_eq!(whole.to_json().unwrap(), resumed.to_json().unwrap());
                assert_eq!(
                    whole.model().logits(&[1, 2, 3], [1, 3], false).unwrap(),
                    resumed.model().logits(&[1, 2, 3], [1, 3], false).unwrap()
                );
            }
            assert!(whole.step().unwrap().is_none());
            assert!(resumed.step().unwrap().is_none());
            assert_eq!(whole.epoch(), 2);
            assert_eq!(whole.next_window(), 3);
            let finished = whole.to_json().unwrap();
            let restored = NativeCausalLmTrainingSession::from_json(
                load(),
                documents(),
                finished.as_bytes(),
                &whole.checkpoint_id().unwrap(),
            )
            .unwrap();
            assert!(restored.exhausted());
            let generation =
                CausalLmGenerationConfig::from_json(br#"{"max_new_tokens":2}"#).unwrap();
            let mut original_generator =
                NativeCausalLmGenerator::new(whole.model().clone(), vec![1, 2], generation.clone())
                    .unwrap();
            let mut restored_generator =
                NativeCausalLmGenerator::new(restored.model().clone(), vec![1, 2], generation)
                    .unwrap();
            while !original_generator.finished() {
                original_generator.step().unwrap();
                restored_generator.step().unwrap();
                assert_eq!(
                    original_generator.token_ids(),
                    restored_generator.token_ids()
                );
            }
        }
    }
}

#[test]
fn session_windows_and_updates_equal_manual_native_operations() {
    let mode = CausalLmAdaptationMode::Joint;
    let base = load();
    let mut session =
        NativeCausalLmTrainingSession::new(adapted(&base, mode), documents(), session_config(mode))
            .unwrap();
    let mut manual = session.model().clone();
    let docs = documents();
    for update in 0..session.total_updates() {
        let w = &session.plan().windows()[update as usize % 3];
        let tokens = &docs[w.document as usize][w.start as usize..(w.start + w.length) as usize];
        let gradients = manual.adaptation_loss_gradients(tokens, [1, 4]).unwrap();
        manual = manual.step_adaptation_sgd(&gradients, mode, 0.2).unwrap();
        let report = session.step().unwrap().unwrap();
        assert_eq!(report.update, update + 1);
        assert_eq!(report.epoch, (update / 3) as u32);
        assert_eq!(report.target_tokens, 3);
        assert_eq!(report.loss, gradients.loss());
        assert_eq!(
            session.model().lora_to_json().unwrap(),
            manual.lora_to_json().unwrap()
        );
        assert_eq!(
            session.model().geometry_artifact().unwrap().to_json(),
            manual.geometry_artifact().unwrap().to_json()
        );
    }
}

#[test]
fn session_group_selection_keeps_other_group_present_and_unchanged() {
    let base = load();
    for mode in [
        CausalLmAdaptationMode::Lora,
        CausalLmAdaptationMode::Geometry,
    ] {
        let model = adapted(&base, CausalLmAdaptationMode::Joint);
        let lora = model.lora_to_json().unwrap();
        let geometry = model.geometry_artifact().unwrap().to_json().to_owned();
        let mut session =
            NativeCausalLmTrainingSession::new(model, documents(), session_config(mode)).unwrap();
        let report = session.step().unwrap().unwrap();
        assert!(report.pre_lora_artifact_id.is_some());
        assert!(report.pre_geometry_artifact_id.is_some());
        match mode {
            CausalLmAdaptationMode::Lora => {
                assert_eq!(
                    session.model().geometry_artifact().unwrap().to_json(),
                    geometry
                );
                assert_ne!(session.model().lora_to_json().unwrap(), lora);
            }
            CausalLmAdaptationMode::Geometry => {
                assert_eq!(session.model().lora_to_json().unwrap(), lora);
                assert_ne!(
                    session.model().geometry_artifact().unwrap().to_json(),
                    geometry
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn session_restore_rejects_corruption_owners_and_impossible_cursor() {
    let mode = CausalLmAdaptationMode::Joint;
    let session = NativeCausalLmTrainingSession::new(
        adapted(&load(), mode),
        documents(),
        session_config(mode),
    )
    .unwrap();
    let saved = session.to_json().unwrap();
    let pinned = session.checkpoint_id().unwrap();
    let mut changed_docs = documents();
    changed_docs[0][0] = 2; // Even dropped tail contents are bound by the corpus identity.
    assert!(NativeCausalLmTrainingSession::from_json(
        load(),
        changed_docs,
        saved.as_bytes(),
        &pinned
    )
    .is_err());
    assert!(NativeCausalLmTrainingSession::prepare_restore(
        load(),
        saved.as_bytes(),
        "sha256:wrong"
    )
    .is_err());
    assert!(NativeCausalLmTrainingSession::prepare_restore(
        session.model().clone(),
        saved.as_bytes(),
        &pinned
    )
    .is_err());
    for (field, value) in [
        ("updates", json!(99)),
        ("base_model_identity", json!("wrong")),
        ("geometry_id", json!("wrong")),
        ("lora_id", json!("wrong")),
        ("lora_json", serde_json::Value::Null),
        ("schema", json!("future")),
    ] {
        let mut changed: serde_json::Value = serde_json::from_str(&saved).unwrap();
        changed[field] = value;
        let changed = changed.to_string();
        assert!(
            NativeCausalLmTrainingSession::from_json(
                load(),
                documents(),
                changed.as_bytes(),
                &id(&changed)
            )
            .is_err(),
            "accepted invalid {field}"
        );
    }
    let mut p = parameters();
    p.get_mut("wte.weight").unwrap().1[0] += 0.1;
    let other = NativeCausalLm::from_gpt2_json(
        &serde_json::to_vec(&config()).unwrap(),
        &archive(&p),
        Default::default(),
    )
    .unwrap();
    assert!(
        NativeCausalLmTrainingSession::prepare_restore(other, saved.as_bytes(), &pinned).is_err()
    );
}

#[test]
fn session_admission_bounds_and_step_failure_are_atomic() {
    let mode = CausalLmAdaptationMode::Lora;
    let model = adapted(&load(), mode);
    for cfg in [
        CausalLmTrainingSessionConfig {
            context_length: 1,
            ..session_config(mode)
        },
        CausalLmTrainingSessionConfig {
            epochs: 0,
            ..session_config(mode)
        },
        CausalLmTrainingSessionConfig {
            learning_rate: f32::NAN,
            ..session_config(mode)
        },
        CausalLmTrainingSessionConfig {
            mode: CausalLmAdaptationMode::Joint,
            ..session_config(mode)
        },
    ] {
        assert!(NativeCausalLmTrainingSession::new(model.clone(), documents(), cfg).is_err());
    }
    assert!(NativeCausalLmTrainingSession::document_plan(
        &model,
        session_config(mode),
        &[usize::MAX]
    )
    .is_err());
    assert!(NativeCausalLmTrainingSession::new(
        model.clone(),
        vec![vec![1, 2]],
        session_config(mode)
    )
    .is_err());
    assert!(NativeCausalLmTrainingSession::new(
        model.clone(),
        vec![vec![1, 2, 3, 11]],
        session_config(mode)
    )
    .is_err());
    let mut record: serde_json::Value =
        serde_json::from_str(&model.lora_to_json().unwrap()).unwrap();
    record["a"] = json!(vec![f32::MAX; 8]);
    record["b"] = json!(vec![f32::MAX; 8]);
    let dangerous = load().with_lora_json(&record.to_string()).unwrap();
    let mut session =
        NativeCausalLmTrainingSession::new(dangerous, documents(), session_config(mode)).unwrap();
    let before = session.to_json().unwrap();
    assert!(session.step().is_err());
    assert_eq!(session.to_json().unwrap(), before);
    assert_eq!(session.completed_updates(), 0);
}
