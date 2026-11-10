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
