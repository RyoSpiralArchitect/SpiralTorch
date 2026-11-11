use super::*;
use sha2::{Digest, Sha256};
use st_kernel_contracts::adamw::{AdamWConfig, AdamWStep};

fn cfg() -> CausalLmQkvTrainingConfig {
    CausalLmQkvTrainingConfig {
        context_length: 4,
        batch_windows: 3,
        max_rows: 2,
        epochs: 2,
        learning_rate: 0.03,
        weight_decay: 0.02,
        label_smoothing: 0.1,
    }
}
fn corpus() -> (Vec<Vec<usize>>, Vec<Vec<i64>>) {
    (
        vec![
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            vec![2, 4, 6, 8, 10],
            vec![1],
            vec![],
        ],
        vec![
            vec![-100, -100, -100, -100, 5, 6, 7, 8, 9, 10],
            vec![-100, 4, 6, 8, 10],
            vec![-100],
            vec![],
        ],
    )
}
fn optimizer(model: &NativeCausalLm) -> NativeCausalLmQkvAdamW {
    NativeCausalLmQkvAdamW::new(model, 0, AdamWConfig::default()).unwrap()
}
fn seal(bytes: &[u8], change: impl FnOnce(&mut serde_json::Value)) -> (Vec<u8>, String) {
    let n = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
    let mut h: serde_json::Value = serde_json::from_slice(&bytes[4..4 + n]).unwrap();
    change(&mut h);
    let h = serde_json::to_vec(&h).unwrap();
    let mut out = (h.len() as u32).to_le_bytes().to_vec();
    out.extend_from_slice(&h);
    out.extend_from_slice(&bytes[4 + n..]);
    let pin = format!("sha256:{:x}", Sha256::digest(&out));
    (out, pin)
}

#[test]
fn qkv_evaluation_binds_trained_projection_and_survives_restore() {
    let model = load();
    let (docs, labels) = corpus();
    let config = CausalLmEvaluationConfig {
        context_length: 4,
        microbatch_windows: 2,
    };
    let mut trainer =
        NativeCausalLmQkvTrainer::new(optimizer(&model), docs.clone(), Some(labels.clone()), cfg())
            .unwrap();
    let initial = trainer
        .optimizer()
        .evaluate_documents(&docs, Some(&labels), config)
        .unwrap();
    let baseline = model
        .evaluate_documents(&docs, Some(&labels), config)
        .unwrap();
    assert_eq!(initial.evaluation.nll.to_bits(), baseline.nll.to_bits());
    assert_eq!(initial.evaluation.data_id, baseline.data_id);
    assert_eq!(initial.serving_id, trainer.optimizer().serving_id());
    assert_eq!(initial.steps, 0);

    trainer.step().unwrap().unwrap();
    let (bytes, pin) = trainer.optimizer().checkpoint().unwrap();
    let updated = trainer
        .optimizer()
        .evaluate_documents(&docs, Some(&labels), config)
        .unwrap();
    assert_ne!(updated.serving_id, initial.serving_id);
    assert_eq!(updated.steps, 1);
    assert_eq!(updated.serving_id, trainer.optimizer().serving_id());
    let restored = NativeCausalLmQkvAdamW::restore(&model, &bytes, &pin).unwrap();
    let replay = restored
        .evaluate_documents(&docs, Some(&labels), config)
        .unwrap();
    assert_eq!(
        serde_json::to_value(updated).unwrap(),
        serde_json::to_value(replay).unwrap()
    );
    assert_eq!(trainer.optimizer().checkpoint().unwrap(), (bytes, pin));
    assert!(restored
        .evaluate_documents(&docs, Some(&labels[..1]), config)
        .is_err());
}

#[test]
fn qkv_session_matches_explicit_target_covering_windows_and_all_restart_boundaries() {
    let base = load();
    let model = base
        .with_lora(1, 2, 1.)
        .unwrap()
        .with_geometry(artifact(&base, "transformer.h.1.attn", 0.2))
        .unwrap();
    let mut opt = optimizer(&model);
    opt.step_windows(
        &[1, 2, 3],
        [1, 3],
        None,
        0.,
        AdamWStep::new(0.01, 0.).unwrap(),
    )
    .unwrap();
    let mut manual = opt.clone();
    let (docs, labels) = corpus();
    let mut trainer =
        NativeCausalLmQkvTrainer::new(opt, docs.clone(), Some(labels.clone()), cfg()).unwrap();
    assert_eq!((trainer.active_windows(), trainer.total_updates()), (4, 4));
    let mut targets = 0;
    for update in 0..4 {
        let (raw, pin) = trainer.checkpoint().unwrap();
        let mut restored = NativeCausalLmQkvTrainer::prepare_resume(&model, &raw, &pin)
            .unwrap()
            .resume(docs.clone(), Some(labels.clone()))
            .unwrap();
        let (qs, ys): (Vec<&[usize]>, Vec<&[i64]>) = if update % 2 == 0 {
            (
                vec![&docs[0][3..7], &docs[0][6..10], &docs[1][0..4]],
                vec![&labels[0][3..7], &labels[0][6..10], &labels[1][0..4]],
            )
        } else {
            (vec![&docs[1][3..5]], vec![&labels[1][3..5]])
        };
        let expected = manual
            .step_documents(&qs, Some(&ys), 2, 0.1, AdamWStep::new(0.03, 0.02).unwrap())
            .unwrap();
        let a = trainer.step().unwrap().unwrap();
        let b = restored.step().unwrap().unwrap();
        assert_eq!((a.update, a.epoch), (update + 1, (update / 2) as u32));
        assert_eq!(
            (a.window_start, a.window_end),
            if update % 2 == 0 { (0, 3) } else { (3, 4) }
        );
        assert_eq!(
            serde_json::to_value(&a).unwrap(),
            serde_json::to_value(b).unwrap()
        );
        assert_eq!(
            serde_json::to_value(&a.optimizer).unwrap(),
            serde_json::to_value(expected).unwrap()
        );
        assert_eq!(
            trainer.optimizer().checkpoint().unwrap(),
            manual.checkpoint().unwrap()
        );
        assert_eq!(
            trainer.checkpoint().unwrap(),
            restored.checkpoint().unwrap()
        );
        targets += a.optimizer.target_tokens;
    }
    assert_eq!(targets, 20);
    let end = trainer.checkpoint().unwrap();
    assert!(trainer.step().unwrap().is_none());
    assert_eq!(end, trainer.checkpoint().unwrap());
    let mut restored = NativeCausalLmQkvTrainer::prepare_resume(&model, &end.0, &end.1)
        .unwrap()
        .resume(docs, Some(labels))
        .unwrap();
    assert!(restored.step().unwrap().is_none());
    assert_eq!(restored.checkpoint().unwrap(), end);
}

#[test]
fn qkv_session_restore_pins_even_ignored_data_masks_order_and_arithmetic() {
    let model = load();
    let (docs, labels) = corpus();
    let trainer =
        NativeCausalLmQkvTrainer::new(optimizer(&model), docs.clone(), Some(labels.clone()), cfg())
            .unwrap();
    let (raw, pin) = trainer.checkpoint().unwrap();
    for kind in 0..3 {
        let mut ds = docs.clone();
        let mut ls = labels.clone();
        match kind {
            0 => ds[0][0] = 2,
            1 => ls[0][0] = 1,
            _ => {
                ds.swap(0, 1);
                ls.swap(0, 1);
            }
        }
        assert!(NativeCausalLmQkvTrainer::prepare_resume(&model, &raw, &pin)
            .unwrap()
            .resume(ds, Some(ls))
            .is_err());
    }
    assert!(NativeCausalLmQkvTrainer::prepare_resume(&model, &raw, &pin)
        .unwrap()
        .resume(docs.clone(), None)
        .is_err());
    assert!(NativeCausalLmQkvTrainer::prepare_resume(
        &model
            .clone()
            .with_readout_accumulator(CausalLmReadoutAccumulator::F64),
        &raw,
        &pin
    )
    .is_err());
    let mut corrupt = raw.clone();
    *corrupt.last_mut().unwrap() ^= 1;
    assert!(NativeCausalLmQkvTrainer::prepare_resume(&model, &corrupt, &pin).is_err());
    for (key, value) in [
        ("updates", json!(1)),
        ("updates", json!(u64::MAX)),
        ("initial_steps", json!(1)),
        ("schema", json!("other")),
        ("unexpected", json!(true)),
        ("optimizer_id", json!("sha256:bad")),
    ] {
        let (bad, p) = seal(&raw, |h| h[key] = value);
        assert!(NativeCausalLmQkvTrainer::prepare_resume(&model, &bad, &p).is_err());
    }
    let (bad, p) = seal(&raw, |h| h["config"]["learning_rate"] = json!(0));
    assert!(NativeCausalLmQkvTrainer::prepare_resume(&model, &bad, &p).is_err());
}

#[test]
fn qkv_session_rejects_empty_targets_budgets_and_exhausted_plans_without_training() {
    let model = load();
    let opt = optimizer(&model);
    for lengths in [
        vec![],
        vec![1],
        vec![usize::MAX],
        vec![NativeCausalLmQkvTrainer::MAX_TOKENS + 1],
    ] {
        assert!(NativeCausalLmQkvTrainer::document_plan(&opt, cfg(), &lengths, true).is_err());
    }
    assert!(NativeCausalLmQkvTrainer::new(
        opt.clone(),
        vec![vec![1, 2, 3]],
        Some(vec![vec![-100; 3]]),
        cfg()
    )
    .is_err());
    assert!(
        NativeCausalLmQkvTrainer::new(opt.clone(), vec![vec![1, 2, 3], vec![99]], None, cfg())
            .is_err()
    );
    assert!(NativeCausalLmQkvTrainer::new(
        opt.clone(),
        vec![vec![1, 2]],
        Some(vec![vec![99, 2]]),
        cfg()
    )
    .is_err());
    let huge = CausalLmQkvTrainingConfig {
        batch_windows: 1,
        epochs: 1_048_576,
        ..cfg()
    };
    assert!(
        NativeCausalLmQkvTrainer::new(opt.clone(), vec![vec![1, 2]; 4096], None, huge).is_err()
    );
    let mut tight = model.clone();
    tight.limits.max_logit_elements = 11;
    let ys = Some(vec![vec![-100, -100, -100, 4], vec![-100, -100, -100, 5]]);
    assert!(NativeCausalLmQkvTrainer::new(
        optimizer(&tight),
        vec![vec![1, 2, 3, 4], vec![2, 3, 4, 5]],
        ys,
        CausalLmQkvTrainingConfig {
            batch_windows: 2,
            max_rows: 2,
            ..cfg()
        }
    )
    .is_err());
}

#[test]
fn qkv_session_failed_optimizer_candidate_preserves_cursor_weights_and_checkpoint() {
    let model = load();
    let mut trainer = NativeCausalLmQkvTrainer::new(
        optimizer(&model),
        vec![vec![1, 2, 3]],
        None,
        CausalLmQkvTrainingConfig {
            learning_rate: f32::MAX,
            weight_decay: f32::MAX,
            ..cfg()
        },
    )
    .unwrap();
    let before = trainer.checkpoint().unwrap();
    assert!(trainer.step().is_err());
    assert_eq!(trainer.updates(), 0);
    assert_eq!(trainer.optimizer().steps(), 0);
    assert_eq!(trainer.checkpoint().unwrap(), before);
}

#[test]
fn qkv_binary_file_segments_match_memory_and_never_clobber_existing_paths() {
    let model = load();
    let (docs, labels) = corpus();
    let mut a =
        NativeCausalLmQkvTrainer::new(optimizer(&model), docs.clone(), Some(labels.clone()), cfg())
            .unwrap();
    let mut direct = a.clone();
    let dir = tempfile::tempdir().unwrap();
    let initial = dir.path().join("initial.bin");
    let before = a.checkpoint().unwrap();
    assert_eq!(a.save_checkpoint_new(&initial).unwrap(), before.1);
    assert_eq!(std::fs::read(&initial).unwrap(), before.0);
    assert!(a.advance_to_checkpoint_new(2, &initial).is_err());
    assert_eq!(a.checkpoint().unwrap(), before);
    assert_eq!(std::fs::read(&initial).unwrap(), before.0);
    let path = dir.path().join("next.bin");
    let (reports, pin) = a.advance_to_checkpoint_new(2, &path).unwrap();
    for report in reports {
        assert_eq!(
            serde_json::to_value(report).unwrap(),
            serde_json::to_value(direct.step().unwrap().unwrap()).unwrap()
        );
    }
    assert_eq!(a.checkpoint().unwrap(), direct.checkpoint().unwrap());
    assert_eq!(std::fs::read(&path).unwrap(), a.checkpoint().unwrap().0);
    let restored =
        NativeCausalLmQkvTrainer::prepare_resume(&model, &std::fs::read(&path).unwrap(), &pin)
            .unwrap()
            .resume(docs, Some(labels))
            .unwrap();
    assert_eq!(restored.checkpoint().unwrap(), a.checkpoint().unwrap());
    let projection = dir.path().join("qkv.bin");
    let saved = a.optimizer().checkpoint().unwrap();
    assert_eq!(
        a.optimizer().save_checkpoint_new(&projection).unwrap(),
        saved.1
    );
    assert_eq!(std::fs::read(&projection).unwrap(), saved.0);
    for stop in [0, 2, 5, 1027, u64::MAX] {
        let state = a.checkpoint().unwrap();
        assert!(a
            .advance_to_checkpoint_new(stop, dir.path().join("invalid.bin"))
            .is_err());
        assert_eq!(a.checkpoint().unwrap(), state);
        assert!(!dir.path().join("invalid.bin").exists());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::{symlink, PermissionsExt};
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        let link = dir.path().join("dangling");
        symlink(dir.path().join("missing"), &link).unwrap();
        assert!(a.advance_to_checkpoint_new(3, &link).is_err());
        assert!(a.optimizer().save_checkpoint_new(&link).is_err());
        assert!(!dir.path().join("missing").exists());
    }
}

#[test]
fn qkv_shared_publication_failure_after_work_keeps_live_session_unchanged() {
    let model = load();
    let (docs, labels) = corpus();
    let mut a =
        NativeCausalLmQkvTrainer::new(optimizer(&model), docs, Some(labels), cfg()).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let occupied = dir.path().join("occupied");
    std::fs::write(&occupied, b"keep").unwrap();
    let before = a.checkpoint().unwrap();
    // Exercise the same transaction with a late real publication error, after
    // a candidate has updated, rather than only advisory path admission failure.
    let result: CausalLmResult<String> =
        crate::causal_lm::checkpoint::transaction(&mut a, |next| {
            next.step()?.unwrap();
            assert_eq!(next.updates(), 1);
            crate::causal_lm::checkpoint::save(&occupied, next.checkpoint()?)
        });
    assert!(result.is_err());
    assert_eq!(std::fs::read(occupied).unwrap(), b"keep");
    assert_eq!(a.checkpoint().unwrap(), before);
    let mut bad = NativeCausalLmQkvTrainer::new(
        optimizer(&model),
        vec![vec![1, 2, 3]],
        None,
        CausalLmQkvTrainingConfig {
            learning_rate: f32::MAX,
            weight_decay: f32::MAX,
            ..cfg()
        },
    )
    .unwrap();
    let initial = bad.checkpoint().unwrap();
    let path = dir.path().join("no-result.bin");
    assert!(bad.advance_to_checkpoint_new(1, &path).is_err());
    assert!(!path.exists());
    assert_eq!(bad.checkpoint().unwrap(), initial);
}
