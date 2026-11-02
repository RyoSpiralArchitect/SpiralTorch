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

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn file_checkpoint_publishes_exact_v1_and_v2_bytes_without_overwrite() {
    let dir = tempfile::tempdir().unwrap();
    let base = load();
    let mode = CausalLmAdaptationMode::Joint;
    for batch in [1, 2] {
        let path = dir.path().join(format!("checkpoint-{batch}.json"));
        let mut s = NativeCausalLmTrainingSession::new_batched(
            adapted(&base, mode),
            documents(),
            session_config(mode),
            batch,
        )
        .unwrap();
        s.step().unwrap().unwrap();
        let (json, pin) = s.checkpoint_with_id().unwrap();
        assert_eq!(s.save_checkpoint_new(&path).unwrap(), pin);
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes, json.as_bytes());
        let mut restored =
            NativeCausalLmTrainingSession::from_json(base.clone(), documents(), &bytes, &pin)
                .unwrap();
        s.step().unwrap().unwrap();
        restored.step().unwrap().unwrap();
        assert_eq!(
            restored.checkpoint_id().unwrap(),
            s.checkpoint_id().unwrap()
        );
        assert!(s.save_checkpoint_new(&path).is_err());
        assert_eq!(std::fs::read(path).unwrap(), bytes);
        assert_eq!(restored.completed_updates(), s.completed_updates());
    }
}

#[test]
fn checkpointed_segments_match_each_report_and_resume_across_epoch_tails() {
    let base = load();
    let both = adapted(&base, CausalLmAdaptationMode::Joint);
    let dir = tempfile::tempdir().unwrap();
    for mode in [
        CausalLmAdaptationMode::Joint,
        CausalLmAdaptationMode::Lora,
        CausalLmAdaptationMode::Geometry,
    ] {
        for batch in [1, 2] {
            let mut s = NativeCausalLmTrainingSession::new_batched(
                both.clone(),
                documents(),
                session_config(mode),
                batch,
            )
            .unwrap();
            let mut reference = s.clone();
            let start = dir.path().join(format!("{mode:?}-{batch}-first"));
            let finish = dir.path().join(format!("{mode:?}-{batch}-last"));
            let stop = s.total_updates() - 1;
            let (steps, pin) = s.advance_to_checkpoint_new(stop, &start).unwrap();
            for step in steps {
                assert_eq!(
                    serde_json::to_string(&step).unwrap(),
                    serde_json::to_string(&reference.step().unwrap().unwrap()).unwrap()
                );
            }
            let (raw, expected) = reference.checkpoint_with_id().unwrap();
            assert_eq!(s.completed_updates(), stop);
            assert_eq!(pin, expected);
            assert_eq!(std::fs::read(&start).unwrap(), raw.as_bytes());
            let mut resumed = NativeCausalLmTrainingSession::from_json(
                base.clone(),
                documents(),
                raw.as_bytes(),
                &pin,
            )
            .unwrap();
            let (tail, final_pin) = resumed
                .advance_to_checkpoint_new(resumed.total_updates(), &finish)
                .unwrap();
            assert_eq!(tail.len(), 1);
            assert_eq!(
                serde_json::to_string(&tail[0]).unwrap(),
                serde_json::to_string(&reference.step().unwrap().unwrap()).unwrap()
            );
            assert_eq!(final_pin, reference.checkpoint_id().unwrap());
            assert!(resumed.exhausted());
            assert!(resumed
                .advance_to_checkpoint_new(stop, dir.path().join("no-work"))
                .is_err());
        }
    }
}

#[test]
fn segment_rejects_invalid_limits_or_used_path_without_advancing() {
    let mode = CausalLmAdaptationMode::Lora;
    let mut cfg = session_config(mode);
    cfg.epochs = 2000;
    let mut s =
        NativeCausalLmTrainingSession::new(adapted(&load(), mode), documents(), cfg).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("snapshot");
    let before = s.checkpoint_id().unwrap();
    for stop in [
        0,
        NativeCausalLmTrainingSession::MAX_CHECKPOINT_INTERVAL + 1,
        u64::MAX,
    ] {
        assert!(s.advance_to_checkpoint_new(stop, &path).is_err());
        assert!(!path.exists());
    }
    assert!(s
        .advance_to_checkpoint_new(1, dir.path().join("missing/next"))
        .is_err());
    std::fs::write(&path, b"retained").unwrap();
    let e = s.advance_to_checkpoint_new(1, &path).unwrap_err();
    assert_eq!(
        e.downcast_ref::<std::io::Error>().unwrap().kind(),
        std::io::ErrorKind::AlreadyExists
    );
    assert_eq!(std::fs::read(&path).unwrap(), b"retained");
    assert_eq!(s.checkpoint_id().unwrap(), before);
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
        let initial_lora = session.model().lora_artifact_id().map(str::to_owned);
        let initial_geometry = session.model().geometry_artifact_id().map(str::to_owned);
        let step = session.step().unwrap().unwrap();
        let (json, pin) = session.checkpoint_with_id().unwrap();
        assert_eq!(json, session.to_json().unwrap());
        assert_eq!(pin, session.checkpoint_id().unwrap());
        assert_eq!(pin, id(&json));
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
        if mode == CausalLmAdaptationMode::Joint {
            assert_ne!(initial_lora.as_deref(), lora);
            assert_ne!(initial_geometry.as_deref(), geom);
            assert!(restore(
                base.clone(),
                &json,
                &pin,
                1,
                data,
                initial_lora.as_deref(),
                geom
            )
            .is_err());
            assert!(restore(
                base.clone(),
                &json,
                &pin,
                1,
                data,
                lora,
                initial_geometry.as_deref()
            )
            .is_err());
        }
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
fn corpus_free_restore_rejects_intrinsic_geometry_work_that_could_not_have_trained() {
    let base = load();
    let mut recipe: serde_json::Value = serde_json::from_str(geometry(&base).to_json()).unwrap();
    recipe["placements"][0]["recipe"]["max_values"] = json!(12);
    let geometry =
        GeometryAdapterArtifact::create_json(&serde_json::to_vec(&recipe).unwrap()).unwrap();
    let mut session = NativeCausalLmTrainingSession::new_batched(
        base.with_geometry(geometry).unwrap(),
        vec![vec![1, 2, 3, 4]],
        session_config(CausalLmAdaptationMode::Geometry),
        2,
    )
    .unwrap();
    session.step().unwrap();
    let (json, pin) = session.checkpoint_with_id().unwrap();
    assert!(
        NativeCausalLmTrainingSession::prepare_restore(base.clone(), json.as_bytes(), &pin).is_ok()
    );
    let saved: serde_json::Value = serde_json::from_str(&json).unwrap();
    for (context, updates) in [(5, 1), (4, 3)] {
        let mut fake = saved.clone();
        fake["config"]["context_length"] = json!(context);
        fake["updates"] = json!(updates);
        let text = fake.to_string();
        assert!(NativeCausalLm::from_session_checkpoint_at(
            base.clone(),
            text.as_bytes(),
            &id(&text),
            updates,
            session.data_id(),
            None,
            session.model().geometry_artifact_id()
        )
        .is_err());
    }
}

#[test]
fn batched_session_updates_once_per_target_mean_and_keeps_epoch_tails() {
    let documents = vec![vec![1, 2, 3, 4], vec![8, 7, 6, 5], vec![5, 4, 3, 2]];
    for mode in [
        CausalLmAdaptationMode::Lora,
        CausalLmAdaptationMode::Geometry,
        CausalLmAdaptationMode::Joint,
    ] {
        let base = load();
        let mut manual = adapted(&base, mode);
        let cfg = session_config(mode);
        let mut session =
            NativeCausalLmTrainingSession::new_batched(manual.clone(), documents.clone(), cfg, 2)
                .unwrap();
        assert_eq!(
            (
                session.batch_windows(),
                session.updates_per_epoch(),
                session.total_updates()
            ),
            (2, 2, 4)
        );
        for (u, (start, end)) in [(0, 2), (2, 3), (0, 2), (2, 3)].into_iter().enumerate() {
            let ids: Vec<_> = documents[start..end].iter().flatten().copied().collect();
            let grads = manual
                .adaptation_loss_gradients(&ids, [end - start, 4])
                .unwrap();
            let separate: Vec<_> = documents[start..end]
                .iter()
                .map(|row| manual.adaptation_loss_gradients(row, [1, 4]).unwrap())
                .collect();
            let n = separate.len() as f32;
            assert!(
                (grads.loss() - separate.iter().map(|g| g.loss()).sum::<f32>() / n).abs() < 3e-6
            );
            for (path, values) in grads.parameters() {
                for (p, g) in values {
                    for (i, &v) in g.iter().enumerate() {
                        let mean = separate
                            .iter()
                            .map(|row| row.parameters()[path][p][i])
                            .sum::<f32>()
                            / n;
                        assert!(
                            (mean - v).abs() < 3e-5,
                            "{path}:{p}[{i}] mean={mean} actual={v}"
                        );
                    }
                }
            }
            manual = manual
                .step_adaptation_sgd(&grads, mode, cfg.learning_rate)
                .unwrap();
            let r = session.step().unwrap().unwrap();
            assert_eq!(
                (
                    r.update,
                    r.epoch,
                    r.window_start,
                    r.window_end,
                    r.target_tokens
                ),
                (u as u64 + 1, (u / 2) as u32, start, end, 3 * (end - start))
            );
            assert_eq!(r.loss, grads.loss());
            assert_eq!(
                session.model().lora_artifact_id(),
                manual.lora_artifact_id()
            );
            assert_eq!(
                session.model().geometry_artifact_id(),
                manual.geometry_artifact_id()
            );
            let (json, pin) = session.checkpoint_with_id().unwrap();
            let saved: serde_json::Value = serde_json::from_str(&json).unwrap();
            assert_eq!(
                saved["schema"],
                "spiraltorch.native_causal_lm_training_session.v2"
            );
            assert_eq!(saved["batch_windows"], 2);
            let inference = NativeCausalLm::from_session_checkpoint_at(
                base.clone(),
                json.as_bytes(),
                &pin,
                r.update,
                session.data_id(),
                r.post_lora_artifact_id.as_deref(),
                r.post_geometry_artifact_id.as_deref(),
            )
            .unwrap();
            assert_eq!(
                inference.logits(&[1, 2, 3], [1, 3], true).unwrap(),
                manual.logits(&[1, 2, 3], [1, 3], true).unwrap()
            );
            session = NativeCausalLmTrainingSession::from_json(
                base.clone(),
                documents.clone(),
                json.as_bytes(),
                &pin,
            )
            .unwrap();
            assert_eq!(session.to_json().unwrap(), json);
        }
        assert!(session.exhausted());
        assert_eq!(session.next_window(), 3);
        assert!(session.step().unwrap().is_none());
    }
}

#[test]
fn batch_admission_and_wire_versions_refuse_ambiguous_or_impossible_policies() {
    let base = load();
    let cfg = session_config(CausalLmAdaptationMode::Joint);
    let model = adapted(&base, cfg.mode);
    for count in [0, 1_048_577] {
        assert!(
            NativeCausalLmTrainingSession::new_batched(model.clone(), documents(), cfg, count)
                .is_err()
        );
    }
    let ordinary = NativeCausalLmTrainingSession::new(model.clone(), documents(), cfg).unwrap();
    let explicit =
        NativeCausalLmTrainingSession::new_batched(model.clone(), documents(), cfg, 1).unwrap();
    assert_eq!(ordinary.to_json().unwrap(), explicit.to_json().unwrap());
    assert!(!ordinary.to_json().unwrap().contains("batch_windows"));
    assert_eq!(
        ordinary.config_json().unwrap(),
        serde_json::to_string(&cfg).unwrap()
    );
    let broad = NativeCausalLmTrainingSession::new_batched(model, documents(), cfg, 100).unwrap();
    assert_eq!(broad.updates_per_epoch(), 1);
    let batch: serde_json::Value = serde_json::from_str(&broad.to_json().unwrap()).unwrap();
    for (schema, count) in [
        ("v1", Some(json!(2))),
        ("v1", Some(serde_json::Value::Null)),
        ("v2", None),
        ("v2", Some(json!(1))),
        ("v2", Some(json!(0))),
        ("v2", Some(json!(1_048_577))),
    ] {
        let mut fake = batch.clone();
        fake["schema"] = json!(format!(
            "spiraltorch.native_causal_lm_training_session.{schema}"
        ));
        fake.as_object_mut().unwrap().remove("batch_windows");
        if let Some(n) = count {
            fake["batch_windows"] = n;
        }
        let payload = fake.to_string();
        assert!(NativeCausalLmTrainingSession::prepare_restore(
            base.clone(),
            payload.as_bytes(),
            &id(&payload)
        )
        .is_err());
    }
    let mut too_far = batch;
    too_far["updates"] = json!(3);
    let payload = too_far.to_string();
    assert!(NativeCausalLmTrainingSession::from_json(
        base,
        documents(),
        payload.as_bytes(),
        &id(&payload)
    )
    .is_err());
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
    let dir = tempfile::tempdir().unwrap();
    let failed = dir.path().join("failed-gradient");
    assert!(session.advance_to_checkpoint_new(1, &failed).is_err());
    assert!(!failed.exists());
    assert_eq!(session.to_json().unwrap(), before);
    assert_eq!(session.completed_updates(), 0);
}
