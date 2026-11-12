use super::*;
use st_kernel_contracts::adamw::{AdamWConfig, AdamWStep};
use st_kernel_contracts::gradient_clip::GlobalNormClip;

fn adapted(lag: usize) -> NativeCausalLm {
    let base = load();
    let base = if lag == 0 {
        base
    } else {
        base.with_key_lag(CausalLmKeyLag { layer: 0, lag }).unwrap()
    };
    base.with_lora(1, 2, 1.)
        .unwrap()
        .with_geometry(artifact(&base, "transformer.h.1.attn", 0.2))
        .unwrap()
}

#[test]
fn causal_qkv_matches_selected_token_prefix_derivatives_not_sequence_means() {
    let ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    // One target in the first sequence, three in the second.
    let labels = [-100, -100, -100, -100, 5, -100, -100, 8, 9, 10];
    for lag in [0, 1] {
        for layer in [0, 1] {
            let model = adapted(lag);
            let trainer =
                NativeCausalLmQkvAdamW::new(&model, layer, AdamWConfig::default()).unwrap();
            let state = trainer.checkpoint().unwrap();
            for epsilon in [0., 0.1] {
                let (loss, gradient, count) = trainer
                    .causal_loss_gradients(&ids, [2, 5], Some(&labels), epsilon)
                    .unwrap();
                assert_eq!(count, 4);
                let expected_loss = model
                    .next_token_loss_with_objective(
                        &ids,
                        [2, 5],
                        Some(&labels),
                        CausalLmObjective {
                            label_smoothing: epsilon,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                assert_eq!(loss.to_bits(), expected_loss.to_bits());
                let mut sum = vec![0f64; 48];
                let mut losses = Vec::new();
                for (row, targets) in ids.chunks_exact(5).zip(labels.chunks_exact(5)) {
                    for pos in 1..5 {
                        if targets[pos] == -100 {
                            continue;
                        }
                        let (l, g) = trainer
                            .answer_sequence_gradients_with_label_smoothing(
                                &[&row[..pos]],
                                &[targets[pos] as usize],
                                1,
                                epsilon,
                            )
                            .unwrap();
                        losses.push(f64::from(l));
                        for (v, &x) in sum.iter_mut().zip(g.data()) {
                            *v += f64::from(x) / 4.;
                        }
                    }
                }
                assert!((f64::from(loss) - losses.iter().sum::<f64>() / 4.).abs() < 1e-6);
                assert!(
                    (losses[0] / 2. + losses[1..].iter().sum::<f64>() / 6. - f64::from(loss)).abs()
                        > 1e-6
                );
                assert!(gradient
                    .data()
                    .iter()
                    .zip(sum)
                    .all(|(&a, b)| (f64::from(a) - b).abs() < 2e-6));
                assert!(gradient.data().iter().any(|v| v.abs() > 1e-6));
            }
            assert_eq!(trainer.checkpoint().unwrap(), state);
        }
    }
}

#[test]
fn causal_qkv_sparse_admission_masks_loss_without_masking_context_or_leaking_future() {
    let mut model = adapted(1);
    model.limits.max_logit_elements = 11;
    let mut trainer = NativeCausalLmQkvAdamW::new(&model, 0, AdamWConfig::default()).unwrap();
    let ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let mut labels = [-100; 10];
    labels[2] = 3;
    assert!(trainer.loss_input_count([2, 5], false).is_err());
    assert_eq!(trainer.loss_input_count([2, 5], true).unwrap(), 10);
    let (loss, gradient, count) = trainer
        .causal_loss_gradients(&ids, [2, 5], Some(&labels), 0.)
        .unwrap();
    assert_eq!(count, 1);
    let mut future = ids;
    future[3] = 10;
    future[4] = 0;
    let (future_loss, future_gradient, _) = trainer
        .causal_loss_gradients(&future, [2, 5], Some(&labels), 0.)
        .unwrap();
    assert_eq!(loss.to_bits(), future_loss.to_bits());
    assert_eq!(gradient, future_gradient);
    let mut context = ids;
    context[0] = 9;
    assert_ne!(
        trainer
            .causal_loss_gradients(&context, [2, 5], Some(&labels), 0.)
            .unwrap()
            .1,
        gradient
    );
    let before = trainer.checkpoint().unwrap();
    for bad in [vec![-100; 10], vec![-100; 9], vec![99; 10], vec![1; 10]] {
        assert!(trainer
            .step_windows(
                &ids,
                [2, 5],
                Some(&bad),
                0.,
                AdamWStep::new(0.01, 0.).unwrap()
            )
            .is_err());
        assert_eq!(trainer.checkpoint().unwrap(), before);
    }
    assert!(trainer.loss_input_count([1, 1], true).is_err());
    assert!(trainer.loss_input_count([1, 10], true).is_err());
}

#[test]
fn causal_qkv_derivative_agrees_with_physical_qkv_weight_edits() {
    let ids = [1, 2, 3, 4, 5];
    let labels = [-100, -100, 3, 4, 5];
    for lag in [0, 1] {
        let make = |p: &BTreeMap<String, (Vec<usize>, Vec<f32>)>| {
            let base = NativeCausalLm::from_gpt2_json(
                &serde_json::to_vec(&config()).unwrap(),
                &archive(p),
                Default::default(),
            )
            .unwrap();
            if lag == 0 {
                base
            } else {
                base.with_key_lag(CausalLmKeyLag { layer: 0, lag }).unwrap()
            }
        };
        for layer in [0, 1] {
            let trainer =
                NativeCausalLmQkvAdamW::new(&make(&parameters()), layer, AdamWConfig::default())
                    .unwrap();
            let (_, gradient, _) = trainer
                .causal_loss_gradients(&ids, [1, 5], Some(&labels), 0.1)
                .unwrap();
            for output in [0, 3, 4, 7, 8, 11] {
                for input in [0, 2] {
                    let loss = |sign: f32| {
                        let mut p = parameters();
                        p.get_mut(&format!("h.{layer}.attn.c_attn.weight"))
                            .unwrap()
                            .1[input * 12 + output] += sign * 0.002;
                        f64::from(
                            make(&p)
                                .next_token_loss_with_objective(
                                    &ids,
                                    [1, 5],
                                    Some(&labels),
                                    CausalLmObjective {
                                        label_smoothing: 0.1,
                                        ..Default::default()
                                    },
                                )
                                .unwrap(),
                        )
                    };
                    let numerical = (loss(1.) - loss(-1.)) / 0.004;
                    assert!(
                        (numerical - f64::from(gradient.data()[output * 4 + input])).abs() < 1e-4
                    );
                }
            }
        }
    }
}

#[test]
fn causal_qkv_single_target_agrees_with_answer_update_and_full_response_resumes() {
    let model = adapted(1);
    let mut answers = NativeCausalLmQkvAdamW::new_with_gradient_clip(
        &model,
        0,
        AdamWConfig::default(),
        Some(GlobalNormClip::new(0.001).unwrap()),
    )
    .unwrap();
    let mut windows = answers.clone();
    let step = AdamWStep::new(0.03, 0.02).unwrap();
    answers
        .step_answers(&[1, 2, 3, 5, 6, 7], [2, 3], &[4, 8], step)
        .unwrap();
    let ids = [1, 2, 3, 4, 5, 6, 7, 8];
    let labels = [-100, -100, -100, 4, -100, -100, -100, 8];
    let report = windows
        .step_windows(&ids, [2, 4], Some(&labels), 0., step)
        .unwrap();
    assert_eq!(report.target_tokens, 2);
    assert_eq!(windows.checkpoint().unwrap(), answers.checkpoint().unwrap());
    let (raw, pin) = windows.checkpoint().unwrap();
    let mut restored = NativeCausalLmQkvAdamW::restore(&model, &raw, &pin).unwrap();
    let a = windows.step_windows(&ids, [2, 4], None, 0.1, step).unwrap();
    let b = restored
        .step_windows(&ids, [2, 4], None, 0.1, step)
        .unwrap();
    assert_eq!(a.target_tokens, 6);
    assert!(a.gradient_clip.is_some());
    assert_eq!(
        serde_json::to_value(a).unwrap(),
        serde_json::to_value(b).unwrap()
    );
    assert_eq!(
        windows.checkpoint().unwrap(),
        restored.checkpoint().unwrap()
    );
    let before = windows.checkpoint().unwrap();
    windows
        .step_windows(&ids, [2, 4], None, 0.1, AdamWStep::new(0., 0.).unwrap())
        .unwrap();
    assert_eq!(windows.checkpoint().unwrap(), before);
}

#[test]
fn causal_qkv_rivals_preserve_ce_control_and_resume_exactly() {
    let model = adapted(1);
    let ids = [1, 2, 3, 4, 5, 6, 7, 8];
    let labels = [-100, -100, 3, 4, -100, -100, 7, 8];
    let step = AdamWStep::new(0.03, 0.01).unwrap();
    let mut ce = NativeCausalLmQkvAdamW::new(&model, 0, AdamWConfig::default()).unwrap();
    let mut control = ce.clone();
    let mut rival = ce.clone();
    let off = RivalCrossEntropy { rivals: 2, mix: 0. };
    let on = RivalCrossEntropy {
        rivals: 2,
        mix: 0.5,
    };
    let (ce_loss, ce_gradient, count) = ce
        .causal_loss_gradients(&ids, [2, 4], Some(&labels), 0.)
        .unwrap();
    let (off_report, off_gradient) = control
        .rival_loss_gradients(&ids, [2, 4], Some(&labels), off)
        .unwrap();
    let (on_report, on_gradient) = rival
        .rival_loss_gradients(&ids, [2, 4], Some(&labels), on)
        .unwrap();
    assert_eq!(off_report.target_tokens, count);
    assert_eq!(off_report.loss.to_bits(), ce_loss.to_bits());
    assert_eq!(off_gradient, ce_gradient);
    assert_eq!(on_report.target_tokens, count);
    assert_ne!(on_gradient, ce_gradient);

    ce.step_windows(&ids, [2, 4], Some(&labels), 0., step)
        .unwrap();
    let control_report = control
        .step_rival_windows(&ids, [2, 4], Some(&labels), off, step)
        .unwrap();
    assert_eq!(control_report.competition.loss.to_bits(), ce_loss.to_bits());
    assert_eq!(ce.checkpoint().unwrap(), control.checkpoint().unwrap());
    let first = rival
        .step_rival_windows(&ids, [2, 4], Some(&labels), on, step)
        .unwrap();
    assert_eq!(first.competition.target_tokens, count);
    assert_ne!(rival.checkpoint().unwrap(), control.checkpoint().unwrap());
    let (raw, pin) = rival.checkpoint().unwrap();
    let mut restored = NativeCausalLmQkvAdamW::restore(&model, &raw, &pin).unwrap();
    let next = rival
        .step_rival_windows(&ids, [2, 4], Some(&labels), on, step)
        .unwrap();
    let replay = restored
        .step_rival_windows(&ids, [2, 4], Some(&labels), on, step)
        .unwrap();
    assert_eq!(
        serde_json::to_value(next).unwrap(),
        serde_json::to_value(replay).unwrap()
    );
    assert_eq!(rival.checkpoint().unwrap(), restored.checkpoint().unwrap());

    let before = rival.checkpoint().unwrap();
    for invalid in [
        RivalCrossEntropy {
            rivals: 0,
            mix: 0.5,
        },
        RivalCrossEntropy {
            rivals: 2,
            mix: f64::NAN,
        },
    ] {
        assert!(rival.rival_loss_input_count([2, 4], true, invalid).is_err());
        assert!(rival
            .step_rival_windows(&ids, [2, 4], Some(&labels), invalid, step)
            .is_err());
        assert_eq!(rival.checkpoint().unwrap(), before);
    }
}

#[test]
fn causal_qkv_rival_derivative_matches_physical_weight_edits() {
    let ids = [1, 2, 3, 4, 5];
    let labels = [-100, -100, 3, 4, 5];
    let recipe = RivalCrossEntropy {
        rivals: 2,
        mix: 0.5,
    };
    let make = |p: &BTreeMap<String, (Vec<usize>, Vec<f32>)>| {
        NativeCausalLm::from_gpt2_json(
            &serde_json::to_vec(&config()).unwrap(),
            &archive(p),
            Default::default(),
        )
        .unwrap()
    };
    let trainer =
        NativeCausalLmQkvAdamW::new(&make(&parameters()), 0, AdamWConfig::default()).unwrap();
    let (report, gradient) = trainer
        .rival_loss_gradients(&ids, [1, 5], Some(&labels), recipe)
        .unwrap();
    assert_eq!(report.target_tokens, 3);
    for (output, input) in [(0, 0), (3, 2), (7, 0), (11, 2)] {
        let loss = |sign: f32| {
            let mut p = parameters();
            p.get_mut("h.0.attn.c_attn.weight").unwrap().1[input * 12 + output] += sign * 0.002;
            let model = make(&p);
            let prepared = model
                .prepare_objective(&ids, [1, 5], Some(&labels), CausalLmObjective::default())
                .unwrap();
            let hidden = model
                .decode_hidden(&prepared.input, prepared.shape)
                .unwrap();
            let logits = model.causal_logits(hidden, &prepared).unwrap();
            f64::from(
                logits
                    .rival_cross_entropy_mean_gradient(&prepared.labels, recipe)
                    .unwrap()
                    .0
                    .loss,
            )
        };
        let numerical = (loss(1.) - loss(-1.)) / 0.004;
        assert!(
            (numerical - f64::from(gradient.data()[output * 4 + input])).abs() < 2e-4,
            "QKV rival derivative mismatch at ({output}, {input})"
        );
    }
}
