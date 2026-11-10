use super::*;

#[path = "line_menu_decoder_tests.rs"]
mod line_menu_decoder;

#[path = "projection_probe_tests.rs"]
mod projection_probe;

fn geometry(base: &NativeCausalLm, layer: usize, gate: f32) -> GeometryAdapterArtifact {
    let mut record = artifact(base, &format!("transformer.h.{layer}.attn"), gate)
        .record()
        .clone();
    record.training = true;
    record.placements[0].training = true;
    for tensor in &mut record.placements[0].tensors {
        tensor.requires_grad = true;
    }
    GeometryAdapterArtifact::create(record).unwrap()
}

fn model(layer: usize, policy: CausalLmReadoutAccumulator) -> NativeCausalLm {
    let base = load().with_readout_accumulator(policy);
    let mut lora: serde_json::Value = serde_json::from_str(
        &base
            .with_lora(layer, 2, 3.)
            .unwrap()
            .lora_to_json()
            .unwrap(),
    )
    .unwrap();
    lora["a"] = json!([0.2, -0.1, 0.3, 0.15, -0.2, 0.1, 0.05, 0.3]);
    lora["b"] = json!([0.1, -0.2, 0.15, 0.05, -0.1, 0.2, 0.1, -0.15]);
    base.with_geometry(geometry(&base, layer, 0.2))
        .unwrap()
        .with_lora_json(&lora.to_string())
        .unwrap()
}

#[test]
fn anchored_projection_enters_real_decoder_and_sealed_paired_sgd_without_moving_lora() {
    use crate::geometry_artifact::{
        GeometryAdapterRecipe, GeometryAdapterTensorRole, WaveStageRecipe,
    };
    let base = model(1, CausalLmReadoutAccumulator::F32WideBackward);
    let path = "transformer.h.1.attn";
    let mut r = base.geometry_artifact().unwrap().record().clone();
    r.placements[0].recipe = GeometryAdapterRecipe::WaveStage {
        features: 4,
        strength: 0.5,
        max_values: 1024,
        stage: WaveStageRecipe::Projection {
            curvature: -1.,
            log_radius: 0.,
        },
    };
    let raw = GeometryAdapterArtifact::create(r).unwrap();
    let centered = raw.anchor_wave_projection(path).unwrap();
    let m = base.with_geometry(raw).unwrap();
    let c = base.with_geometry(centered.clone()).unwrap();
    let ids = [1, 2, 3, 5];
    let shape = [1, 4];
    let mode = CausalLmAdaptationMode::Geometry;
    let g = m.adaptation_loss_gradients_for(&ids, shape, mode).unwrap();
    let h = c.adaptation_loss_gradients_for(&ids, shape, mode).unwrap();
    assert_eq!(
        m.logits(&ids, shape, false).unwrap().data(),
        c.logits(&ids, shape, false).unwrap().data()
    );
    assert_eq!(g.parameters(), h.parameters());
    assert_eq!(g.loss(), h.loss());
    assert!(c
        .step_paired_geometry_sgd(&g, &h, 0.1, CausalLmPairWeighting::Equal, None)
        .is_err());
    let (next, step) = c
        .step_paired_geometry_sgd(&h, &h, 0.1, CausalLmPairWeighting::CommonDescent, None)
        .unwrap();
    assert_eq!(step.trainable_values, 8);
    assert!(step.applied.both_descend && step.applied.l2 > 0.);
    assert_eq!(next.lora_to_json().unwrap(), c.lora_to_json().unwrap());
    let changed = next.geometry_artifact().unwrap();
    for (a, b) in changed.record().placements[0]
        .tensors
        .iter()
        .zip(&centered.record().placements[0].tensors)
    {
        if a.role == GeometryAdapterTensorRole::Buffer {
            assert_eq!(a.values, b.values);
        }
    }
    let restored = GeometryAdapterArtifact::from_json(
        changed.to_json().as_bytes(),
        changed.artifact_id(),
        c.base_model_identity(),
    )
    .unwrap();
    assert_eq!(
        c.with_geometry(restored)
            .unwrap()
            .logits(&ids, shape, true)
            .unwrap()
            .data(),
        next.logits(&ids, shape, true).unwrap().data()
    );
    assert_ne!(
        c.logits(&ids, shape, true).unwrap().data(),
        next.logits(&ids, shape, true).unwrap().data()
    );
}

#[test]
fn paired_competition_recomputes_both_trajectories_and_matches_manual_rounded_updates() {
    let recipe = RivalCrossEntropy {
        rivals: 3,
        mix: 0.5,
    };
    let mode = CausalLmAdaptationMode::Geometry;
    for (layer, policy) in [
        (0, CausalLmReadoutAccumulator::F32),
        (1, CausalLmReadoutAccumulator::F64),
    ] {
        let mut ce = model(layer, policy);
        let mut mixed = ce.clone();
        let original_lora = ce.lora_to_json().unwrap();
        for i in 0..4 {
            let ids = [1 + i, 2, 4, 3, 8 - i, 6, 5, 7];
            let mask = [-100, -100, -100, 3, -100, -100, 5, -100];
            let labels = (i % 2 == 0).then_some(mask.as_slice());
            let rate = 0.75;
            let a = ce.geometry_artifact().unwrap();
            let b = mixed.geometry_artifact().unwrap();
            let ce_grad = ce
                .adaptation_loss_gradients_with_labels(&ids, [2, 4], labels, Some(mode))
                .unwrap();
            let cg = mixed
                .geometry_competition_gradients(&ids, [2, 4], labels, recipe)
                .unwrap();
            let reference = ce.step_adaptation_sgd(&ce_grad, mode, rate).unwrap();
            let proposal = mixed.step_adaptation_sgd(&cg, mode, rate).unwrap();
            let distance = a
                .trainable_delta_l2(reference.geometry_artifact().unwrap())
                .unwrap();
            let (expected, receipt) = b
                .match_trainable_delta(proposal.geometry_artifact().unwrap(), distance)
                .unwrap();
            let (single, single_receipt) = mixed
                .step_geometry_sgd_at_distance(&cg, rate, distance)
                .unwrap();
            let (c, m, r) = ce
                .geometry_competition_pair(&mixed, recipe, rate)
                .unwrap()
                .step(&ids, [2, 4], labels)
                .unwrap();
            assert_eq!(
                c.geometry_artifact().unwrap().to_json(),
                reference.geometry_artifact().unwrap().to_json()
            );
            assert_eq!(m.geometry_artifact().unwrap().to_json(), expected.to_json());
            assert_eq!(single.geometry_artifact_id(), m.geometry_artifact_id());
            assert_eq!(
                serde_json::to_value(&r.mixed_step).unwrap(),
                serde_json::to_value(&receipt).unwrap()
            );
            assert_eq!(
                serde_json::to_value(single_receipt).unwrap(),
                serde_json::to_value(&receipt).unwrap()
            );
            assert_eq!(
                serde_json::to_value(r.competition).unwrap(),
                serde_json::to_value(cg.competition().unwrap()).unwrap()
            );
            assert_eq!(
                (
                    r.reference_loss,
                    r.target_tokens,
                    r.shared_prefix_layers,
                    r.window_shape
                ),
                (
                    ce_grad.loss(),
                    if labels.is_some() { 2 } else { 6 },
                    layer,
                    [2, 4]
                )
            );
            assert_eq!(r.reference_pre_geometry_id, a.artifact_id());
            assert_eq!(
                r.reference_post_geometry_id,
                c.geometry_artifact_id().unwrap()
            );
            assert_eq!(r.lora_artifact_id.as_deref(), ce.lora_artifact_id());
            assert_eq!(r.readout_accumulator, policy);
            assert_eq!(r.mixed_step.requested_l2, distance);
            assert!(distance > 0. && (r.mixed_step.realized_l2 / distance - 1.).abs() <= 1e-4);
            assert_eq!(r.pre_arm_separation_l2, a.trainable_delta_l2(b).unwrap());
            assert_eq!(c.lora_to_json().unwrap(), original_lora);
            assert_eq!(m.lora_to_json().unwrap(), original_lora);
            if i > 0 {
                assert!(r.pre_arm_separation_l2 > 0.);
            }
            ce = c;
            mixed = m;
        }
    }
}

#[test]
fn paired_and_archived_distance_steps_reject_incompatible_owners_and_budgets() {
    let model = model(1, CausalLmReadoutAccumulator::F32);
    let light = load()
        .with_geometry(model.geometry_artifact().unwrap().clone())
        .unwrap();
    let recipe = RivalCrossEntropy {
        rivals: 1,
        mix: 0.5,
    };
    let lr = 0.5;
    let ids = [2, 5, 3];
    let grad = model
        .geometry_competition_gradients(&ids, [1, 3], None, recipe)
        .unwrap();
    let origin = model.geometry_artifact_id();
    let (unchanged, z) = model.step_geometry_sgd_at_distance(&grad, lr, 0.).unwrap();
    assert_eq!(unchanged.geometry_artifact_id(), origin);
    assert_eq!(z.realized_l2, 0.);
    for distance in [-1., f64::NAN, f64::INFINITY] {
        assert!(model
            .step_geometry_sgd_at_distance(&grad, lr, distance)
            .is_err());
    }
    let wrong_readout = model.with_readout_accumulator(CausalLmReadoutAccumulator::F64);
    for other in [&wrong_readout, &light] {
        assert!(model.geometry_competition_pair(other, recipe, lr).is_err());
        assert!(other.step_geometry_sgd_at_distance(&grad, lr, 0.).is_err());
    }
    assert!(model.geometry_competition_pair(&model, recipe, 0.).is_err());
    assert!(model
        .geometry_competition_pair(
            &model,
            RivalCrossEntropy {
                rivals: 0,
                mix: 0.5
            },
            lr
        )
        .is_err());
    let mut rec = model.geometry_artifact().unwrap().record().clone();
    rec.placements[0].tensors[0].requires_grad = false;
    let fixed = model
        .with_geometry(GeometryAdapterArtifact::create(rec.clone()).unwrap())
        .unwrap();
    assert!(model.geometry_competition_pair(&fixed, recipe, lr).is_err());
    rec.placements[0].tensors[0].values[0] += 0.5;
    let changed = model
        .with_geometry(GeometryAdapterArtifact::create(rec).unwrap())
        .unwrap();
    assert!(fixed
        .geometry_competition_pair(&changed, recipe, lr)
        .is_err());
    let mut tight = model.clone();
    tight.limits.max_logit_elements = 11;
    let pair = || model.geometry_competition_pair(&tight, recipe, lr).unwrap();
    assert!(pair().input_count([1, 3], false).is_err());
    assert!(pair().step(&ids, [1, 3], Some(&[-100, 5, 3])).is_err());
    assert!(pair()
        .step(&ids, [1, 3], Some(&[-100, -100, -100]))
        .is_err());
    assert_eq!(model.geometry_artifact_id(), origin);
    assert_eq!(tight.geometry_artifact_id(), origin);
    let next = model
        .step_adaptation_sgd(&grad, CausalLmAdaptationMode::Geometry, lr)
        .unwrap();
    assert!(next.step_geometry_sgd_at_distance(&grad, lr, 0.).is_err());
}

#[test]
fn paired_geometry_models_without_lora_have_an_exact_plain_ce_control() {
    let base = load();
    let x = base.with_geometry(geometry(&base, 1, 0.2)).unwrap();
    let ids = [2, 5, 3];
    for recipe in [
        RivalCrossEntropy { rivals: 2, mix: 0. },
        RivalCrossEntropy {
            rivals: 10,
            mix: 0.7,
        },
    ] {
        let (c, m, r) = x
            .geometry_competition_pair(&x, recipe, 0.75)
            .unwrap()
            .step(&ids, [1, 3], None)
            .unwrap();
        assert_eq!(
            c.geometry_artifact().unwrap().to_json(),
            m.geometry_artifact().unwrap().to_json()
        );
        assert!(r.lora_artifact_id.is_none());
        assert_eq!(r.shared_prefix_layers, 1);
        assert_eq!(r.mixed_step.requested_l2, r.mixed_step.direction_l2);
        assert_eq!(r.mixed_step.relative_vector_error, 0.);
    }
}

#[test]
fn paired_zero_signal_preserves_mixed_owner_and_cannot_invent_a_positive_step() {
    let mut config = config();
    config["tie_word_embeddings"] = json!(false);
    let mut params = parameters();
    params.insert("lm_head.weight".into(), (vec![11, 4], vec![0.; 44]));
    let base = NativeCausalLm::from_gpt2_json(
        &serde_json::to_vec(&config).unwrap(),
        &archive(&params),
        Default::default(),
    )
    .unwrap();
    let x = base.with_geometry(geometry(&base, 1, 0.2)).unwrap();
    let ids = [1, 2, 3];
    let recipe = RivalCrossEntropy {
        rivals: 2,
        mix: 0.5,
    };
    let g = x
        .geometry_competition_gradients(&ids, [1, 3], None, recipe)
        .unwrap();
    assert!(x.step_geometry_sgd_at_distance(&g, 0.5, 1e-4).is_err());
    let (_, m, r) = x
        .geometry_competition_pair(&x, recipe, 0.5)
        .unwrap()
        .step(&ids, [1, 3], None)
        .unwrap();
    assert_eq!(
        (
            r.mixed_step.direction_l2,
            r.mixed_step.requested_l2,
            r.mixed_step.realized_l2
        ),
        (0., 0., 0.)
    );
    assert_eq!(
        m.geometry_artifact().unwrap().to_json(),
        x.geometry_artifact().unwrap().to_json()
    );
}

#[test]
fn coverage_guard_reads_the_live_ce_snapshot_without_changing_its_update() {
    let ce = model(1, CausalLmReadoutAccumulator::F32);
    let mixed = ce.with_geometry(geometry(&ce, 1, -0.8)).unwrap();
    let recipe = RivalCrossEntropy {
        rivals: 2,
        mix: 0.5,
    };
    let left = ce.logits(&[2, 5], [1, 2], true).unwrap();
    let right = mixed.logits(&[2, 5], [1, 2], true).unwrap();
    let mut activated = 0;
    let mut changed = 0;
    for t in 0..11 {
        let labels = [-100, -100, t as i64];
        let ref_ = left.rival_coverage_reference(&[t as i64], 2).unwrap();
        let (oracle, _) = right
            .guarded_rival_cross_entropy_mean_gradient(
                &[t as i64],
                recipe,
                &ref_,
                RivalCoverageGuard::new(0.5).unwrap(),
            )
            .unwrap();
        let pair = || ce.geometry_competition_pair(&mixed, recipe, 0.75).unwrap();
        let ids = [2, 5, t];
        let (c0, m0, baseline) = pair().step(&ids, [1, 3], Some(&labels)).unwrap();
        let (c1, m1, guarded) = pair()
            .with_coverage_guard(0.5)
            .unwrap()
            .step(&ids, [1, 3], Some(&labels))
            .unwrap();
        assert_eq!(
            serde_json::to_value(guarded.competition).unwrap(),
            serde_json::to_value(oracle).unwrap()
        );
        assert_eq!(c0.geometry_artifact_id(), c1.geometry_artifact_id());
        assert_eq!(baseline.reference_loss, guarded.reference_loss);
        assert_eq!(
            baseline.mixed_step.requested_l2,
            guarded.mixed_step.requested_l2
        );
        assert_eq!(m1.lora_artifact_id(), ce.lora_artifact_id());
        if guarded.competition.coverage_guard.unwrap().active_targets == 0 {
            assert_eq!(m0.geometry_artifact_id(), m1.geometry_artifact_id());
        } else {
            activated += 1;
            changed += usize::from(m0.geometry_artifact_id() != m1.geometry_artifact_id());
        }
    }
    assert!(activated > 0 && activated < 11 && changed > 0);
}

// Independent oracle: one forward per prefix, no shared shift, gather or loss.
fn objective(model: &NativeCausalLm, ids: &[usize], width: usize, labels: Option<&[i64]>) -> f64 {
    let mut sum = 0.;
    let mut count = 0;
    for (i, window) in ids.chunks_exact(width).enumerate() {
        for t in 1..width {
            let label = labels.map_or(window[t] as i64, |v| v[i * width + t]);
            if label == -100 {
                continue;
            }
            let logits = model.logits(&window[..t], [1, t], true).unwrap();
            let row = logits.data();
            let m = f64::from(row.iter().copied().fold(f32::NEG_INFINITY, f32::max));
            sum += m - f64::from(row[label as usize])
                + row
                    .iter()
                    .map(|&v| (f64::from(v) - m).exp())
                    .sum::<f64>()
                    .ln();
            count += 1;
        }
    }
    sum / f64::from(count)
}

#[test]
fn composed_gradients_match_finite_differences_in_both_groups() {
    check_composed_gradients(None);
}

#[test]
fn masked_composed_gradients_match_independent_prefix_differences() {
    check_composed_gradients(Some(&[3, -100, 7, -100, -100, 2, -100, 8]));
}

fn check_composed_gradients(labels: Option<&[i64]>) {
    let ids = [1, 2, 3, 4, 5, 6, 7, 8];
    for policy in [
        CausalLmReadoutAccumulator::F32,
        CausalLmReadoutAccumulator::F64,
    ] {
        for layer in 0..2 {
            let model = model(layer, policy);
            let gradients = model
                .adaptation_loss_gradients_with_labels(&ids, [2, 4], labels, None)
                .unwrap();
            assert_eq!(gradients.parameters().len(), 2);
            assert_eq!(
                gradients.target_tokens(),
                if labels.is_some() { 3 } else { 6 }
            );
            assert!(
                (f64::from(gradients.loss()) - objective(&model, &ids, 4, labels)).abs() < 1e-6
            );
            assert_eq!(
                gradients.loss(),
                model
                    .next_token_loss_with_objective(&ids, [2, 4], labels, Default::default())
                    .unwrap()
            );
            for (path, parameters) in gradients.parameters() {
                for (name, values) in parameters {
                    for (index, &expected) in values.iter().enumerate() {
                        let eps = 0.002;
                        let perturb = |sign: f32| {
                            if path.ends_with("c_proj") {
                                let mut record: serde_json::Value =
                                    serde_json::from_str(&model.lora_to_json().unwrap()).unwrap();
                                record[name][index] = json!(
                                    record[name][index].as_f64().unwrap() as f32 + sign * eps
                                );
                                model.with_lora_json(&record.to_string()).unwrap()
                            } else {
                                let mut record =
                                    model.geometry_artifact().unwrap().record().clone();
                                record.placements[0]
                                    .tensors
                                    .iter_mut()
                                    .find(|t| t.name == *name)
                                    .unwrap()
                                    .values[index] += sign * eps;
                                model
                                    .with_geometry(GeometryAdapterArtifact::create(record).unwrap())
                                    .unwrap()
                            }
                        };
                        let actual = (objective(&perturb(1.), &ids, 4, labels)
                            - objective(&perturb(-1.), &ids, 4, labels))
                            / (2. * eps as f64);
                        assert!(
                            (actual - f64::from(expected)).abs() < 1e-4,
                            "{policy:?} {path}/{name}[{index}]: {actual} != {expected}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn geometry_rival_objective_matches_prefix_differences_and_freezes_lora() {
    let model = model(0, CausalLmReadoutAccumulator::F32);
    let ids = [1, 2, 3, 4, 5, 6, 7, 8];
    let labels = [-100, -100, -100, 4, -100, -100, 7, -100];
    let recipe = RivalCrossEntropy {
        rivals: 3,
        mix: 0.5,
    };
    let loss = |model: &NativeCausalLm| {
        let mut total = 0.;
        for (prefix, target) in [(&ids[..3], 4usize), (&ids[4..6], 7)] {
            let logits = model.logits(prefix, [1, prefix.len()], true).unwrap();
            let x = logits.data();
            let mut wrong: Vec<_> = (0..x.len())
                .filter(|&i| i != target)
                .map(|i| f64::from(x[i]) - f64::from(x[target]))
                .collect();
            wrong.sort_by(|a, b| b.total_cmp(a));
            total += 0.5 * wrong.iter().map(|x| x.exp()).sum::<f64>().ln_1p();
            total += 0.5 * wrong[..3].iter().map(|x| x.exp()).sum::<f64>().ln_1p();
        }
        total / 2.
    };
    let grads = model
        .geometry_competition_gradients(&ids, [2, 4], Some(&labels), recipe)
        .unwrap();
    let r = grads.competition().unwrap();
    assert_eq!((r.rivals, r.target_tokens), (3, 2));
    assert!((f64::from(grads.loss()) - loss(&model)).abs() < 2e-6);
    assert_eq!(grads.parameters().len(), 1);
    for (name, values) in &grads.parameters()["transformer.h.0.attn"] {
        for (j, &grad) in values.iter().enumerate() {
            let eval = |offset: f32| {
                let mut state = model.geometry_artifact().unwrap().record().clone();
                state.placements[0]
                    .tensors
                    .iter_mut()
                    .find(|t| t.name == *name)
                    .unwrap()
                    .values[j] += offset;
                loss(
                    &model
                        .with_geometry(GeometryAdapterArtifact::create(state).unwrap())
                        .unwrap(),
                )
            };
            let fd = (eval(0.002) - eval(-0.002)) / 0.004;
            assert!(
                (fd - f64::from(grad)).abs() < 1.5e-4,
                "{name}[{j}]: {grad} != {fd}"
            );
        }
    }
    let mode = CausalLmAdaptationMode::Geometry;
    let ce = model
        .adaptation_loss_gradients_with_labels(&ids, [2, 4], Some(&labels), Some(mode))
        .unwrap();
    let zero = model
        .geometry_competition_gradients(
            &ids,
            [2, 4],
            Some(&labels),
            RivalCrossEntropy { mix: 0., ..recipe },
        )
        .unwrap();
    assert!(ce.competition().is_none());
    assert_eq!(
        (zero.loss(), zero.parameters()),
        (ce.loss(), ce.parameters())
    );
    assert!(model
        .step_adaptation_sgd(&grads, CausalLmAdaptationMode::Joint, 0.1)
        .is_err());
    let next = model.step_adaptation_sgd(&grads, mode, 0.5).unwrap();
    assert_eq!(next.base_model_identity(), model.base_model_identity());
    assert_eq!(next.lora_to_json().unwrap(), model.lora_to_json().unwrap());
    assert_ne!(next.geometry_artifact_id(), model.geometry_artifact_id());
    assert!(loss(&next) < loss(&model));
    assert!(next.step_adaptation_sgd(&grads, mode, 0.1).is_err());
}

#[test]
fn ordinary_training_uses_the_wide_head_probe_without_changing_forward_or_owners() {
    let original = model(1, CausalLmReadoutAccumulator::F32);
    let split = original.with_readout_accumulator(CausalLmReadoutAccumulator::F32WideBackward);
    let all_wide = original.with_readout_accumulator(CausalLmReadoutAccumulator::F64);
    let prompt = [4usize, 8, 2, 1];
    let target = 2;
    let rival = 7;
    assert_eq!(
        split.logits(&prompt, [1, 4], true).unwrap().data(),
        original.logits(&prompt, [1, 4], true).unwrap().data()
    );
    let probe = original
        .last_attention_mixture_probe(&prompt, target, rival, 0.5)
        .unwrap();
    let doc = [4, 8, 2, 1, target];
    let labels = [-100, -100, -100, -100, target as i64];
    let mode = CausalLmAdaptationMode::Geometry;
    for arm in probe.objectives {
        let grad = |m: &NativeCausalLm| {
            if arm.objective == "ce" {
                m.adaptation_loss_gradients_with_labels(&doc, [1, 5], Some(&labels), Some(mode))
                    .unwrap()
            } else {
                m.geometry_declared_rival_gradients(
                    &doc,
                    [1, 5],
                    Some(&labels),
                    &[DeclaredRivalRow {
                        position: 4,
                        target,
                        rivals: vec![rival],
                    }],
                    if arm.objective == "mixed" { 0.5 } else { 1. },
                )
                .unwrap()
            }
        };
        let g = grad(&split);
        assert_eq!(g.loss(), arm.loss);
        assert_eq!(g.parameters()[&probe.site], arm.paths[1].parameters);
        assert_eq!(
            g.readout_accumulator(),
            CausalLmReadoutAccumulator::F32WideBackward
        );
        assert!(g.alignment_with(&grad(&original)).is_err());
        for foreign in [&original, &all_wide] {
            assert!(foreign.step_adaptation_sgd(&g, mode, 0.1).is_err());
        }
        let next = split.step_adaptation_sgd(&g, mode, 0.1).unwrap();
        assert_eq!(next.readout_accumulator(), split.readout_accumulator());
        assert_eq!(next.lora_to_json().unwrap(), split.lora_to_json().unwrap());
    }
}

#[test]
fn fixed_primal_mixture_probe_matches_ordinary_gradients_and_independent_head_dot() {
    let m = model(1, CausalLmReadoutAccumulator::F32);
    let (target, rival) = (2usize, 7usize);
    for prompt in [&[4usize][..], &[4, 8, 2, 1][..]] {
        let n = prompt.len();
        let mut ids = prompt.to_vec();
        ids.push(target);
        let mut labels = vec![-100; n];
        labels.push(target as i64);
        let shape = [1, n + 1];
        let probe = m
            .last_attention_mixture_probe(prompt, target, rival, 0.5)
            .unwrap();
        assert_eq!(probe.logits, m.logits(prompt, [1, n], true).unwrap().data());
        assert_eq!(probe.objectives.len(), 3);
        for arm in probe.objectives {
            let expected = if arm.objective == "ce" {
                m.adaptation_loss_gradients_with_labels(
                    &ids,
                    shape,
                    Some(&labels),
                    Some(CausalLmAdaptationMode::Geometry),
                )
                .unwrap()
            } else {
                m.geometry_declared_rival_gradients(
                    &ids,
                    shape,
                    Some(&labels),
                    &[DeclaredRivalRow {
                        position: n,
                        target,
                        rivals: vec![rival],
                    }],
                    if arm.objective == "mixed" { 0.5 } else { 1. },
                )
                .unwrap()
            };
            assert_eq!(expected.loss(), arm.loss);
            assert_eq!(arm.paths.len(), 2);
            assert_eq!(arm.paths[0].parameters, expected.parameters()[&probe.site]);
            for path in &arm.paths {
                assert_eq!(
                    path.boundaries.iter().map(|x| x.0).collect::<Vec<_>>(),
                    vec![
                        "head_input",
                        "final_norm_input",
                        "mlp_contract_input",
                        "gelu_input",
                        "mlp_expand_input",
                        "mlp_norm_input",
                        "attention_output_upstream"
                    ]
                );
                assert_eq!(path.boundaries[2].1.len(), m.config.intermediate);
                assert_eq!(path.boundaries[6].1.len(), m.config.width);
            }
            let wide = &arm.paths[1];
            assert_eq!(wide.head_backward, CausalLmReadoutAccumulator::F64);
            for (channel, &got) in wide.boundaries[0].1.iter().enumerate() {
                let truth = arm
                    .d_logits
                    .iter()
                    .enumerate()
                    .map(|(i, &g)| {
                        f64::from(g) * f64::from(m.token.data()[i * m.config.width + channel])
                    })
                    .sum::<f64>();
                assert_eq!(got, truth as f32);
            }
        }
    }
    let zero = m
        .last_attention_mixture_probe(&[1, 2, 3], target, rival, 0.)
        .unwrap();
    assert_eq!(zero.objectives[0].d_logits, zero.objectives[2].d_logits);
    for i in 0..2 {
        assert_eq!(
            zero.objectives[0].paths[i].boundaries,
            zero.objectives[2].paths[i].boundaries
        );
        assert_eq!(
            zero.objectives[0].paths[i].parameters,
            zero.objectives[2].paths[i].parameters
        );
    }
    assert!(m.mixture_probe_input_count(0, target, rival, 0.5).is_err());
    assert!(m.mixture_probe_input_count(2, target, target, 0.5).is_err());
    assert!(m
        .mixture_probe_input_count(2, target, rival, f64::NAN)
        .is_err());
    assert!(m
        .last_attention_mixture_probe(&[1000], target, rival, 0.5)
        .is_err());
    assert!(m
        .with_readout_accumulator(CausalLmReadoutAccumulator::F64)
        .mixture_probe_input_count(2, target, rival, 0.5)
        .is_err());
    assert!(model(0, CausalLmReadoutAccumulator::F32)
        .mixture_probe_input_count(2, target, rival, 0.5)
        .is_err());
}

#[test]
fn gradient_alignment_allows_different_inputs_not_different_weights_or_coordinates() {
    let model = model(0, CausalLmReadoutAccumulator::F32);
    let mode = Some(CausalLmAdaptationMode::Geometry);
    let grad = |m: &NativeCausalLm, ids: &[usize], mode| {
        m.adaptation_loss_gradients_with_labels(ids, [1, ids.len()], None, mode)
            .unwrap()
    };
    let a = grad(&model, &[1, 2, 3], mode);
    let b = grad(&model, &[5, 6], mode);
    let report = a.alignment_with(&b).unwrap();
    assert_eq!(
        (report.left_target_tokens, report.right_target_tokens),
        (2, 1)
    );
    assert_eq!(
        report.geometry_artifact_id.as_deref(),
        model.geometry_artifact_id()
    );
    let mut pairs = vec![];
    for (site, group) in a.parameters() {
        for (name, grad) in group {
            assert_eq!(report.parameters[site][name].elements, grad.len());
            pairs.extend(
                grad.iter()
                    .zip(&b.parameters()[site][name])
                    .map(|(&a, &b)| (f64::from(a), f64::from(b))),
            );
        }
    }
    let expected = pairs.iter().map(|(x, y)| x * y).sum::<f64>();
    assert_eq!(report.total.elements, pairs.len());
    assert!((expected - report.total.dot).abs() < 1e-12);
    let rev = b.alignment_with(&a).unwrap().total;
    assert_eq!(
        (rev.dot, rev.left_loss_slope),
        (report.total.dot, report.total.right_loss_slope)
    );
    let next = model
        .step_adaptation_sgd(&a, CausalLmAdaptationMode::Geometry, 0.1)
        .unwrap();
    assert!(a.alignment_with(&grad(&next, &[1, 2, 3], mode)).is_err());
    assert!(a
        .alignment_with(&grad(
            &model,
            &[1, 2, 3],
            Some(CausalLmAdaptationMode::Joint)
        ))
        .is_err());
    let wide = model.with_readout_accumulator(CausalLmReadoutAccumulator::F64);
    assert!(a.alignment_with(&grad(&wide, &[1, 2, 3], mode)).is_err());
}

#[test]
fn declared_rivals_select_unshifted_positions_and_update_only_geometry() {
    let model = model(0, CausalLmReadoutAccumulator::F32);
    let ids = [1, 2, 3, 4, 5, 6, 7, 8];
    let labels = [-100, -100, -100, 4, -100, -100, 7, -100];
    let entry = |position, target, rivals: &[usize]| DeclaredRivalRow {
        position,
        target,
        rivals: rivals.to_vec(),
    };
    let plan = [entry(6, 7, &[1, 4])];
    let mode = CausalLmAdaptationMode::Geometry;
    let loss = |model: &NativeCausalLm| {
        let plain = model.next_token_score(&ids[..3], 4).unwrap().nll;
        let all = model.next_token_score(&ids[4..6], 7).unwrap().nll;
        let logits = model.logits(&ids[4..6], [1, 2], true).unwrap();
        let row = logits.data();
        let restricted = [1, 4]
            .iter()
            .map(|&j| (f64::from(row[j]) - f64::from(row[7])).exp())
            .sum::<f64>()
            .ln_1p();
        (plain + 0.5 * (all + restricted)) / 2.
    };
    let g = model
        .geometry_declared_rival_gradients(&ids, [2, 4], Some(&labels), &plan, 0.5)
        .unwrap();
    let r = g.declared_rival_report().unwrap();
    assert!(g.competition().is_none() && (f64::from(g.loss()) - loss(&model)).abs() < 2e-6);
    assert_eq!(
        (r.target_tokens, r.contrasted_tokens, r.rival_ids),
        (2, 1, 2)
    );
    let ce = model
        .adaptation_loss_gradients_with_labels(&ids, [2, 4], Some(&labels), Some(mode))
        .unwrap();
    for (set, mix) in [(&plan[..], 0.), (&[][..], 1.)] {
        let zero = model
            .geometry_declared_rival_gradients(&ids, [2, 4], Some(&labels), set, mix)
            .unwrap();
        assert_eq!(
            (ce.loss(), ce.parameters()),
            (zero.loss(), zero.parameters())
        );
    }
    let next = model.step_adaptation_sgd(&g, mode, 0.5).unwrap();
    assert!(loss(&next) < loss(&model));
    assert_eq!(next.lora_to_json().unwrap(), model.lora_to_json().unwrap());
    assert_eq!(next.base_model_identity(), model.base_model_identity());
    assert!(model
        .step_adaptation_sgd(&g, CausalLmAdaptationMode::Joint, 0.1)
        .is_err());
    for pos in [0, 2, 4, 5, 7, 8] {
        assert!(model
            .geometry_declared_rival_gradients(
                &ids,
                [2, 4],
                Some(&labels),
                &[entry(pos, 7, &[1])],
                0.5
            )
            .is_err());
    }
    assert!(model
        .geometry_declared_rival_gradients(&ids, [2, 4], Some(&labels), &[entry(6, 6, &[1])], 0.5)
        .is_err());
}

#[test]
fn selective_updates_preserve_frozen_groups_and_seal_both_owners() {
    let model = model(0, CausalLmReadoutAccumulator::F32);
    let grads = model
        .adaptation_loss_gradients(&[1, 2, 3, 4], [1, 4])
        .unwrap();
    let lora = model
        .step_adaptation_sgd(&grads, CausalLmAdaptationMode::Lora, 0.1)
        .unwrap();
    let geometry = model
        .step_adaptation_sgd(&grads, CausalLmAdaptationMode::Geometry, 0.1)
        .unwrap();
    let both = model
        .step_adaptation_sgd(&grads, CausalLmAdaptationMode::Joint, 0.1)
        .unwrap();
    assert_eq!(lora.geometry_artifact_id(), model.geometry_artifact_id());
    assert_eq!(geometry.lora_artifact_id(), model.lora_artifact_id());
    assert_ne!(lora.lora_artifact_id(), model.lora_artifact_id());
    assert_ne!(
        geometry.geometry_artifact_id(),
        model.geometry_artifact_id()
    );
    assert_eq!(both.lora_artifact_id(), lora.lora_artifact_id());
    assert_eq!(both.geometry_artifact_id(), geometry.geometry_artifact_id());
    assert_eq!(both.base_model_identity(), model.base_model_identity());
    // Changing even the group frozen in the requested update invalidates the
    // gradient, since that group changed the original forward function.
    assert!(lora
        .step_adaptation_sgd(&grads, CausalLmAdaptationMode::Geometry, 0.1)
        .is_err());
    assert!(geometry
        .step_adaptation_sgd(&grads, CausalLmAdaptationMode::Lora, 0.1)
        .is_err());
    assert!(model
        .with_readout_accumulator(CausalLmReadoutAccumulator::F64)
        .step_adaptation_sgd(&grads, CausalLmAdaptationMode::Joint, 0.1)
        .is_err());
    assert!(model.lora_loss_gradients(&[1, 2, 3], [1, 3]).is_err());
    assert!(model.geometry_loss_gradients(&[1, 2, 3], [1, 3]).is_err());
    for rate in [0., -1., f32::NAN, f32::INFINITY] {
        assert!(model
            .step_adaptation_sgd(&grads, CausalLmAdaptationMode::Joint, rate)
            .is_err());
    }
    assert_eq!(model.lora_artifact_id(), grads.lora_artifact_id());
    assert_eq!(model.geometry_artifact_id(), grads.geometry_artifact_id());
}

#[test]
fn selected_backward_matches_full_snapshots_and_rejects_an_uncomputed_group() {
    use CausalLmAdaptationMode::{Geometry, Joint, Lora};
    for p in [
        CausalLmReadoutAccumulator::F32,
        CausalLmReadoutAccumulator::F64,
    ] {
        let m = model(0, p);
        let ids = [1, 2, 3, 4, 4, 3, 2, 1];
        let all = m.adaptation_loss_gradients(&ids, [2, 4]).unwrap();
        for (mode, other) in [(Lora, Geometry), (Geometry, Lora), (Joint, Joint)] {
            let g = m.adaptation_loss_gradients_for(&ids, [2, 4], mode).unwrap();
            assert_eq!(
                (g.target_tokens(), g.loss().to_bits()),
                (all.target_tokens(), all.loss().to_bits())
            );
            assert_eq!(g.lora_artifact_id(), all.lora_artifact_id());
            assert_eq!(g.geometry_artifact_id(), all.geometry_artifact_id());
            assert_eq!(g.parameters().len(), if mode == Joint { 2 } else { 1 });
            for (path, v) in g.parameters() {
                assert_eq!(v, &all.parameters()[path]);
            }
            let same = m.step_adaptation_sgd(&all, mode, 0.1).unwrap();
            let next = m.step_adaptation_sgd(&g, mode, 0.1).unwrap();
            assert_eq!(next.geometry_artifact_id(), same.geometry_artifact_id());
            assert_eq!(next.lora_artifact_id(), same.lora_artifact_id());
            if mode != Joint {
                assert!(m.step_adaptation_sgd(&g, other, 0.1).is_err());
                assert!(m.step_adaptation_sgd(&g, Joint, 0.1).is_err());
                let changed_other = m.step_adaptation_sgd(&all, other, 0.1).unwrap();
                assert!(changed_other.step_adaptation_sgd(&g, mode, 0.1).is_err());
            }
        }
    }
}

#[test]
fn composition_admission_zero_delta_and_single_group_compatibility() {
    let base = load();
    let geo = geometry(&base, 0, 0.);
    let lora = base.with_lora(0, 2, 2.).unwrap();
    let combined = lora.with_geometry(geo.clone()).unwrap();
    let reverse = base
        .with_geometry(geo)
        .unwrap()
        .with_lora(0, 2, 2.)
        .unwrap();
    assert_eq!(
        combined.logits(&[1, 2, 3], [1, 3], false).unwrap(),
        base.logits(&[1, 2, 3], [1, 3], false).unwrap()
    );
    assert_eq!(
        combined.logits(&[1, 2, 3], [1, 3], false).unwrap(),
        reverse.logits(&[1, 2, 3], [1, 3], false).unwrap()
    );
    assert!(lora.with_geometry(geometry(&base, 1, 0.2)).is_err());
    assert!(base
        .with_geometry(artifact(&base, "transformer.h.0.mlp", 0.2))
        .unwrap()
        .with_lora(0, 2, 2.)
        .is_err());
    let grads = lora.adaptation_loss_gradients(&[1, 2, 3], [1, 3]).unwrap();
    let old = lora.lora_loss_gradients(&[1, 2, 3], [1, 3]).unwrap();
    assert_eq!(
        lora.step_adaptation_sgd(&grads, CausalLmAdaptationMode::Lora, 0.1)
            .unwrap()
            .lora_to_json()
            .unwrap(),
        lora.step_lora_sgd(&old, 0.1)
            .unwrap()
            .lora_to_json()
            .unwrap()
    );
    assert!(lora
        .step_adaptation_sgd(&grads, CausalLmAdaptationMode::Joint, 0.1)
        .is_err());
    let geo = base.with_geometry(geometry(&base, 0, 0.2)).unwrap();
    let grads = geo.adaptation_loss_gradients(&[1, 2, 3], [1, 3]).unwrap();
    let old = geo.geometry_loss_gradients(&[1, 2, 3], [1, 3]).unwrap();
    assert_eq!(
        geo.step_adaptation_sgd(&grads, CausalLmAdaptationMode::Geometry, 0.1)
            .unwrap()
            .geometry_artifact_id(),
        geo.step_geometry_sgd(&old, 0.1)
            .unwrap()
            .geometry_artifact_id()
    );
}
