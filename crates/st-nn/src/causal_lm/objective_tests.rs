use super::*;

fn explicit_loss(model: &NativeCausalLm, windows: &[usize], labels: &[i64], smoothing: f64) -> f64 {
    let input: Vec<_> = windows
        .chunks_exact(4)
        .flat_map(|w| w[..3].iter().copied())
        .collect();
    let targets: Vec<_> = labels
        .chunks_exact(4)
        .flat_map(|w| w[1..].iter().copied())
        .collect();
    let logits = model.logits(&input, [2, 3], false).unwrap();
    let mut total = 0.;
    let mut count = 0;
    for (row, label) in logits.data().chunks_exact(11).zip(targets) {
        if label == -100 {
            continue;
        }
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let log_z = row
            .iter()
            .map(|&v| (f64::from(v) - max).exp())
            .sum::<f64>()
            .ln()
            + max;
        total += log_z
            - (1. - smoothing) * f64::from(row[label as usize])
            - smoothing * row.iter().map(|&v| f64::from(v)).sum::<f64>() / 11.;
        count += 1;
    }
    total / count as f64
}

#[test]
fn masked_smoothed_loss_and_pullback_match_explicit_objective() {
    let base = load();
    let artifact = trainable(&base);
    let model = base.with_geometry(artifact.clone()).unwrap();
    let windows = [1, 2, 3, 4, 5, 6, 7, 8];
    // Different labels are intentional; prompts remain context, not padding.
    let labels = [-100, -100, 8, -100, 4, -100, 3, 1];
    for smoothing in [0., 0.2, 1.] {
        let objective = CausalLmObjective {
            label_smoothing: smoothing,
        };
        let gradient = model
            .geometry_loss_gradients_with_objective(&windows, [2, 4], Some(&labels), objective)
            .unwrap();
        assert_eq!(gradient.target_tokens(), 3);
        assert_eq!(gradient.objective(), objective);
        assert_eq!(
            gradient.loss(),
            model
                .next_token_loss_with_objective(&windows, [2, 4], Some(&labels), objective)
                .unwrap()
        );
        assert!(
            (f64::from(gradient.loss()) - explicit_loss(&model, &windows, &labels, smoothing))
                .abs()
                < 2e-6
        );
        let epsilon = 0.002;
        for (path, parameters) in gradient.parameters() {
            for (name, values) in parameters {
                for (index, &expected) in values.iter().enumerate() {
                    let loss = |sign: f32| {
                        let mut record = artifact.record().clone();
                        record
                            .placements
                            .iter_mut()
                            .find(|p| p.path == *path)
                            .unwrap()
                            .tensors
                            .iter_mut()
                            .find(|p| p.name == *name)
                            .unwrap()
                            .values[index] += sign * epsilon;
                        let changed = base
                            .with_geometry(GeometryAdapterArtifact::create(record).unwrap())
                            .unwrap();
                        explicit_loss(&changed, &windows, &labels, smoothing)
                    };
                    let numerical = (loss(1.) - loss(-1.)) / (2. * f64::from(epsilon));
                    assert!(
                        (numerical - f64::from(expected)).abs() < 8e-5,
                        "{path}/{name}[{index}]: {numerical} vs {expected}"
                    );
                }
            }
        }
    }
}

#[test]
fn shifted_selected_targets_control_microbatch_weights_not_window_sizes() {
    let base = load();
    let model = base.with_geometry(trainable(&base)).unwrap();
    let windows = [1, 2, 3, 4, 5, 6, 7, 8];
    let labels = [8, -100, -100, 3, -100, -100, 1, 2];
    let objective = CausalLmObjective {
        label_smoothing: 0.2,
    };
    let full = model
        .geometry_loss_gradients_with_objective(&windows, [2, 4], Some(&labels), objective)
        .unwrap();
    let mut mean = model
        .geometry_accumulator_with_objective(objective)
        .unwrap();
    for row in 0..2 {
        mean.accumulate_loss_with_labels(
            &model,
            &windows[row * 4..row * 4 + 4],
            [1, 4],
            Some(&labels[row * 4..row * 4 + 4]),
        )
        .unwrap();
    }
    assert_eq!((mean.microbatches(), mean.target_tokens()), (2, 3));
    assert!((mean.loss().unwrap() - f64::from(full.loss())).abs() < 2e-6);
    for (path, parameters) in full.parameters() {
        for (name, values) in parameters {
            for (&a, &b) in values
                .iter()
                .zip(&mean.parameter_gradients().unwrap()[path][name])
            {
                assert!((a - b).abs() < 1e-7);
            }
        }
    }
    let before = format!("{mean:?}");
    assert!(mean
        .add(&model.geometry_loss_gradients(&windows, [2, 4]).unwrap())
        .is_err());
    assert!(mean
        .accumulate_loss_with_labels(&model, &windows, [2, 4], Some(&[-100; 8]))
        .is_err());
    assert_eq!(format!("{mean:?}"), before);
}

#[test]
fn ignored_row_is_not_counted_and_default_path_is_unchanged() {
    let base = load();
    let model = base.with_geometry(trainable(&base)).unwrap();
    let windows = [1, 2, 3, 4, 5, 6, 7, 8];
    let labels: Vec<_> = windows.iter().map(|&v| v as i64).collect();
    let plain = model.geometry_loss_gradients(&windows, [2, 4]).unwrap();
    let same = model
        .geometry_loss_gradients_with_objective(&windows, [2, 4], Some(&labels), Default::default())
        .unwrap();
    assert_eq!(plain.loss(), same.loss());
    assert_eq!(plain.parameters(), same.parameters());
    assert_eq!(same.target_tokens(), 6);
    let mut masked = labels.clone();
    masked[..4].fill(-100);
    let selected = model
        .geometry_loss_gradients_with_objective(&windows, [2, 4], Some(&masked), Default::default())
        .unwrap();
    let single = model
        .geometry_loss_gradients(&windows[4..], [1, 4])
        .unwrap();
    assert_eq!(selected.target_tokens(), single.target_tokens());
    assert_eq!(selected.loss(), single.loss());
    for (path, parameters) in selected.parameters() {
        for (name, values) in parameters {
            for (&a, &b) in values.iter().zip(&single.parameters()[path][name]) {
                assert!((a - b).abs() < 1e-7);
            }
        }
    }
}

#[test]
fn invalid_labels_and_objectives_fail_before_pullback() {
    let base = load();
    let model = base.with_geometry(trainable(&base)).unwrap();
    let windows = [1, 2, 3];
    for labels in [
        vec![-100; 3],
        vec![1, -100, -100],
        vec![1, 2],
        vec![1, -99, 3],
        vec![11, 2, 3],
        vec![1, 2, i64::MAX],
    ] {
        assert!(model
            .geometry_loss_gradients_with_objective(
                &windows,
                [1, 3],
                Some(&labels),
                Default::default()
            )
            .is_err());
        assert!(model
            .next_token_loss_with_objective(&windows, [1, 3], Some(&labels), Default::default())
            .is_err());
    }
    for label_smoothing in [-0.1, 1.1, f64::NAN, f64::INFINITY] {
        let objective = CausalLmObjective { label_smoothing };
        assert!(model
            .geometry_loss_gradients_with_objective(&windows, [1, 3], None, objective)
            .is_err());
        assert!(model
            .next_token_loss_with_objective(&windows, [1, 3], None, objective)
            .is_err());
        assert!(model
            .geometry_accumulator_with_objective(objective)
            .is_err());
    }
    assert!(model.backward_head.get().is_none());
}

#[test]
fn selected_readout_admits_only_actual_logits_without_losing_context_rows() {
    let base = load();
    let model = base.with_geometry(trainable(&base)).unwrap();
    let ids = [1, 2, 3, 4, 5, 6, 7, 8];
    let labels = [-100, -100, 8, -100, 4, -100, 3, 1];
    let objective = CausalLmObjective {
        label_smoothing: 0.2,
    };
    let prepared = model
        .prepare_objective(&ids, [2, 4], Some(&labels), objective)
        .unwrap();
    assert_eq!(prepared.input, [1, 2, 3, 5, 6, 7]);
    assert_eq!(prepared.shape, [2, 3]);
    assert_eq!(
        prepared.selected_rows.as_deref(),
        Some([1, 4, 5].as_slice())
    );
    assert_eq!(prepared.labels, [8, 3, 1]);
    assert!(model
        .prepare_objective(&ids, [2, 4], None, objective)
        .unwrap()
        .selected_rows
        .is_none());
    let expected = model
        .geometry_loss_gradients_with_objective(&ids, [2, 4], Some(&labels), objective)
        .unwrap();
    let mut limited = model.clone();
    limited.limits.max_logit_elements = 33;
    assert!(limited.geometry_gradient_input_count([2, 4]).is_err());
    assert!(limited.geometry_objective_input_count([2, 4], true).is_ok());
    let actual = limited
        .geometry_loss_gradients_with_objective(&ids, [2, 4], Some(&labels), objective)
        .unwrap();
    assert_eq!(expected.loss(), actual.loss());
    assert_eq!(expected.parameters(), actual.parameters());
    assert_eq!(
        limited
            .next_token_loss_with_objective(&ids, [2, 4], Some(&labels), objective)
            .unwrap(),
        actual.loss()
    );
    let mut group = limited
        .geometry_accumulator_with_objective(objective)
        .unwrap();
    group
        .accumulate_loss_with_labels(&limited, &ids, [2, 4], Some(&labels))
        .unwrap();
    let before = format!("{group:?}");
    assert!(group
        .accumulate_loss_with_labels(&limited, &ids, [2, 4], None)
        .is_err());
    assert!(group
        .accumulate_loss_with_labels(&limited, &ids, [2, 4], Some(&[1; 8]))
        .is_err());
    assert_eq!(before, format!("{group:?}"));
    limited.limits.max_logit_elements = 32;
    assert!(limited
        .next_token_loss_with_objective(&ids, [2, 4], Some(&labels), objective)
        .is_err());
    limited.limits.max_logit_elements = 10;
    assert!(limited.loss_objective_input_count([2, 4], true).is_err());
}

#[test]
fn unused_vocabulary_projection_cannot_overflow_a_selected_loss() {
    let mut weights = parameters();
    for (name, (_, values)) in &mut weights {
        values.fill(if name.contains("ln_") && name.ends_with("weight") {
            1.
        } else {
            0.
        });
    }
    weights.get_mut("wte.weight").unwrap().1[4..8].copy_from_slice(&[1., 1., -1., -1.]);
    let mut head = vec![0.; 44];
    head[..2].fill(f32::MAX);
    weights.insert("lm_head.weight".into(), (vec![11, 4], head));
    let mut c = config();
    c["tie_word_embeddings"] = json!(false);
    let model = NativeCausalLm::from_gpt2_json(
        &serde_json::to_vec(&c).unwrap(),
        &archive(&weights),
        Default::default(),
    )
    .unwrap();
    assert!(model.next_token_loss(&[1, 2, 3], [1, 3]).is_err());
    let selected = model
        .next_token_loss_with_objective(
            &[1, 2, 3],
            [1, 3],
            Some(&[-100, -100, 3]),
            Default::default(),
        )
        .unwrap();
    assert!((selected - 11f32.ln()).abs() < 1e-6);
}
