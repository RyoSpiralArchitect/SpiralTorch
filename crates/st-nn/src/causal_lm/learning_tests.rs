use super::*;

fn trainable(model: &NativeCausalLm) -> GeometryAdapterArtifact {
    let mut record = artifact(model, "transformer.h.0.attn", 0.2)
        .record()
        .clone();
    record.training = true;
    record.placements = [
        "transformer.h.0.attn",
        "transformer.h.0.mlp",
        "transformer.h.1",
    ]
    .iter()
    .map(|path| artifact(model, path, 0.2).record().placements[0].clone())
    .collect();
    for site in &mut record.placements {
        site.training = true;
        for parameter in &mut site.tensors {
            parameter.requires_grad = parameter.name == "gate";
        }
    }
    GeometryAdapterArtifact::create(record).unwrap()
}

fn objective(model: &NativeCausalLm, windows: &[usize], shape: [usize; 2]) -> f64 {
    let (input, labels, shape) = model.shift_windows(windows, shape).unwrap();
    let logits = model.logits(&input, shape, false).unwrap();
    logits
        .data()
        .chunks_exact(11)
        .zip(&labels)
        .map(|(row, &label)| {
            let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            maximum - f64::from(row[label as usize])
                + row
                    .iter()
                    .map(|&v| (f64::from(v) - maximum).exp())
                    .sum::<f64>()
                    .ln()
        })
        .sum::<f64>()
        / labels.len() as f64
}

#[test]
fn joint_decoder_pullback_matches_every_geometry_parameter_difference() {
    let base = load();
    let artifact = trainable(&base);
    let model = base.with_geometry(artifact.clone()).unwrap();
    let windows = [1, 2, 3, 4, 5, 6, 7, 8];
    let gradients = model.geometry_loss_gradients(&windows, [2, 4]).unwrap();
    assert_eq!(
        gradients.loss(),
        model.next_token_loss(&windows, [2, 4]).unwrap()
    );
    assert_eq!(gradients.parameters().len(), 3);
    let epsilon = 0.002;
    for (path, params) in gradients.parameters() {
        for (name, values) in params {
            for (i, &expected) in values.iter().enumerate() {
                let perturbed = |sign: f32| {
                    let mut record = artifact.record().clone();
                    let site = record
                        .placements
                        .iter_mut()
                        .find(|p| p.path == *path)
                        .unwrap();
                    site.tensors
                        .iter_mut()
                        .find(|t| t.name == *name)
                        .unwrap()
                        .values[i] += sign * epsilon;
                    base.with_geometry(GeometryAdapterArtifact::create(record).unwrap())
                        .unwrap()
                };
                let numerical = (objective(&perturbed(1.), &windows, [2, 4])
                    - objective(&perturbed(-1.), &windows, [2, 4]))
                    / (2. * f64::from(epsilon));
                assert!(
                    (numerical - f64::from(expected)).abs() < 8e-5,
                    "{path}/{name}[{i}]: {numerical} vs {expected}"
                );
            }
        }
    }
    let a = model
        .geometry_loss_gradients(&windows[..4], [1, 4])
        .unwrap();
    let b = model
        .geometry_loss_gradients(&windows[4..], [1, 4])
        .unwrap();
    for (path, params) in gradients.parameters() {
        for (name, values) in params {
            for (i, &v) in values.iter().enumerate() {
                let mean = (a.parameters()[path][name][i] + b.parameters()[path][name][i]) / 2.;
                assert!((v - mean).abs() < 1e-7);
            }
        }
    }
}

#[test]
fn joint_sgd_freezes_base_and_flagged_parameters_and_resumes_exactly() {
    let base = load();
    let initial = base.with_geometry(trainable(&base)).unwrap();
    let windows = [1, 2, 3, 4, 5, 6, 7, 8];
    let first = initial.geometry_loss_gradients(&windows, [2, 4]).unwrap();
    let mut model = initial.step_geometry_sgd(&first, 0.5).unwrap();
    assert!(model.step_geometry_sgd(&first, 0.5).is_err());
    assert_eq!(base.token.data(), model.token.data());
    assert_eq!(base.head.as_slice(), model.head.as_slice());
    assert_eq!(model.base_model_identity(), base.base_model_identity());
    assert!(base.geometry_artifact().is_none());
    assert_eq!(
        initial.geometry_artifact_id(),
        Some(first.geometry_artifact_id())
    );
    for _ in 0..19 {
        let gradient = model.geometry_loss_gradients(&windows, [2, 4]).unwrap();
        model = model.step_geometry_sgd(&gradient, 0.5).unwrap();
    }
    assert!(model.next_token_loss(&windows, [2, 4]).unwrap() < first.loss() - 1e-4);
    for (before, after) in initial
        .geometry_artifact()
        .unwrap()
        .record()
        .placements
        .iter()
        .zip(&model.geometry_artifact().unwrap().record().placements)
    {
        for p in &before.tensors {
            let q = after.tensors.iter().find(|q| q.name == p.name).unwrap();
            if !p.requires_grad {
                assert_eq!(p.values, q.values);
            }
        }
    }
    let saved = model.geometry_artifact().unwrap();
    let restored = GeometryAdapterArtifact::from_json(
        saved.to_json().as_bytes(),
        saved.artifact_id(),
        base.base_model_identity(),
    )
    .unwrap();
    let restored = base.with_geometry(restored).unwrap();
    let a = model.geometry_loss_gradients(&windows, [2, 4]).unwrap();
    let b = restored.geometry_loss_gradients(&windows, [2, 4]).unwrap();
    assert_eq!(a.loss(), b.loss());
    assert_eq!(a.parameters(), b.parameters());
    assert_eq!(
        model
            .step_geometry_sgd(&a, 0.5)
            .unwrap()
            .geometry_artifact()
            .unwrap()
            .to_json(),
        restored
            .step_geometry_sgd(&b, 0.5)
            .unwrap()
            .geometry_artifact()
            .unwrap()
            .to_json()
    );
}

#[test]
fn training_admission_and_invalid_updates_fail_without_changes() {
    let base = load();
    assert!(base.geometry_gradient_input_count([1, 2]).is_err());
    let frozen = base
        .with_geometry(artifact(&base, "transformer.h.0", 0.1))
        .unwrap();
    assert!(frozen.geometry_gradient_input_count([1, 2]).is_err());
    let model = NativeCausalLm::from_gpt2_json(
        &serde_json::to_vec(&config()).unwrap(),
        &archive(&parameters()),
        CausalLmLimits {
            max_training_checkpoint_elements: 23,
            ..Default::default()
        },
    )
    .unwrap();
    let model = model.with_geometry(trainable(&model)).unwrap();
    assert!(model.geometry_gradient_input_count([1, 3]).is_err()); // 3 checkpoints * 2 rows * 4 features
    assert!(model.geometry_gradient_input_count([1, 2]).is_ok());
    let gradient = model.geometry_loss_gradients(&[1, 2], [1, 2]).unwrap();
    let before = model.geometry_artifact().unwrap().to_json();
    for rate in [0., -0.1, f32::NAN, f32::INFINITY] {
        assert!(model.step_geometry_sgd(&gradient, rate).is_err());
        assert_eq!(model.geometry_artifact().unwrap().to_json(), before);
    }
    assert!(model.geometry_loss_gradients(&[1, 11], [1, 2]).is_err());
    let mut p = parameters();
    p.get_mut("wte.weight").unwrap().1[0] += 0.1;
    let other = NativeCausalLm::from_gpt2_json(
        &serde_json::to_vec(&config()).unwrap(),
        &archive(&p),
        Default::default(),
    )
    .unwrap();
    let other = other.with_geometry(trainable(&other)).unwrap();
    assert!(other.step_geometry_sgd(&gradient, 0.1).is_err());
}

#[test]
fn native_adamw_learns_and_resumes_without_changing_the_base() {
    use crate::geometry_runtime::{AdamWStep, GlobalNormClip, NativeGeometryAdamW};
    let base = load();
    let mut model = base.with_geometry(trainable(&base)).unwrap();
    let original = model.geometry_artifact().unwrap().clone();
    let mut optimizer = NativeGeometryAdamW::new(
        &original,
        Default::default(),
        Some(GlobalNormClip::new(0.1).unwrap()),
    )
    .unwrap();
    let windows = [1, 2, 3, 4, 5, 6, 7, 8];
    let first = model.geometry_loss_gradients(&windows, [2, 4]).unwrap();
    let step = AdamWStep::new(0.05, 0.01).unwrap();
    for _ in 0..20 {
        let gradient = model.geometry_loss_gradients(&windows, [2, 4]).unwrap();
        (model, optimizer) = model
            .step_geometry_adamw(&gradient, &optimizer, step)
            .unwrap();
    }
    assert!(model.next_token_loss(&windows, [2, 4]).unwrap() < first.loss() - 1e-4);
    assert_eq!(base.token.data(), model.token.data());
    assert_eq!(base.head.as_slice(), model.head.as_slice());
    assert!(model.step_geometry_adamw(&first, &optimizer, step).is_err());
    assert_eq!(optimizer.revision(), 20);
    for (before, after) in original
        .record()
        .placements
        .iter()
        .zip(&model.geometry_artifact().unwrap().record().placements)
    {
        for (p, q) in before.tensors.iter().zip(&after.tensors) {
            if !p.requires_grad {
                assert_eq!(p.values, q.values);
            }
        }
    }
    let saved = model.geometry_artifact().unwrap();
    let restored = base
        .with_geometry(
            GeometryAdapterArtifact::from_json(
                saved.to_json().as_bytes(),
                saved.artifact_id(),
                base.base_model_identity(),
            )
            .unwrap(),
        )
        .unwrap();
    let restored_state = NativeGeometryAdamW::from_json(
        optimizer.to_json().unwrap().as_bytes(),
        &optimizer.checkpoint_id().unwrap(),
        restored.geometry_artifact().unwrap(),
    )
    .unwrap();
    let gradients = model.geometry_loss_gradients(&windows, [2, 4]).unwrap();
    let a = model
        .step_geometry_adamw(&gradients, &optimizer, step)
        .unwrap();
    let b = restored
        .step_geometry_adamw(&gradients, &restored_state, step)
        .unwrap();
    assert_eq!(
        a.0.geometry_artifact().unwrap().to_json(),
        b.0.geometry_artifact().unwrap().to_json()
    );
    assert_eq!(a.1.to_json().unwrap(), b.1.to_json().unwrap());
}

fn overflowing_frozen_base_fixture() -> NativeCausalLm {
    let mut p = parameters();
    for (name, (_, values)) in &mut p {
        values.fill(if name.contains("ln_") && name.ends_with("weight") {
            1.
        } else {
            0.
        });
    }
    p.get_mut("h.1.mlp.c_proj.weight").unwrap().1[0] = f32::MAX;
    let mut head = vec![0.; 44];
    head[0] = 1.;
    head[1] = -1.;
    p.insert("lm_head.weight".into(), (vec![11, 4], head));
    let mut c = config();
    c["tie_word_embeddings"] = json!(false);
    NativeCausalLm::from_gpt2_json(
        &serde_json::to_vec(&c).unwrap(),
        &archive(&p),
        Default::default(),
    )
    .unwrap()
}

#[test]
fn valid_geometry_gradients_do_not_differentiate_unused_overflowing_base_paths() {
    let base = overflowing_frozen_base_fixture();
    let mut record = artifact(&base, "transformer.h.1", 0.).record().clone();
    for parameter in &mut record.placements[0].tensors {
        parameter.requires_grad = true;
    }
    let model = base
        .with_geometry(GeometryAdapterArtifact::create(record).unwrap())
        .unwrap();
    let gradient = model.geometry_loss_gradients(&[0, 0], [1, 2]).unwrap();
    assert!((gradient.loss() - 11f32.ln()).abs() < 1e-6);
    assert_eq!(gradient.parameters()["transformer.h.1"]["gate"], [0.; 4]);
    assert!(gradient.parameters()["transformer.h.1"]["bias"]
        .iter()
        .all(|v| v.is_finite()));
    assert!(gradient.parameters()["transformer.h.1"]["bias"]
        .iter()
        .any(|v| v.abs() > 1.));
}

#[test]
fn first_geometry_site_skips_its_unused_overflowing_input_gradient() {
    let base = overflowing_frozen_base_fixture();
    let mut record = artifact(&base, "transformer.h.1", 0.).record().clone();
    for parameter in &mut record.placements[0].tensors {
        parameter.requires_grad = true;
        if parameter.name == "gate" {
            parameter.values[0] = f32::MAX;
        }
    }
    let model = base
        .with_geometry(GeometryAdapterArtifact::create(record).unwrap())
        .unwrap();
    assert!((model.next_token_loss(&[0, 0], [1, 2]).unwrap() - 11f32.ln()).abs() < 1e-6);
    let gradient = model.geometry_loss_gradients(&[0, 0], [1, 2]).unwrap();
    assert_eq!(gradient.parameters()["transformer.h.1"]["gate"], [0.; 4]);
    assert!(gradient.parameters()["transformer.h.1"]["bias"]
        .iter()
        .all(|v| v.is_finite()));
    assert!(gradient.parameters()["transformer.h.1"]["bias"]
        .iter()
        .any(|v| v.abs() > 1.));
}

#[test]
fn learning_stops_at_the_first_site_including_frozen_placements() {
    for suffix in ["", ".mlp", ".attn"] {
        let base = load();
        let path = format!("transformer.h.1{suffix}");
        let mut record = artifact(&base, &path, 0.1).record().clone();
        for t in &mut record.placements[0].tensors {
            t.requires_grad = true;
        }
        let model = base
            .with_geometry(GeometryAdapterArtifact::create(record).unwrap())
            .unwrap();
        model.geometry_loss_gradients(&[1, 2, 3], [1, 3]).unwrap();
        for dense in [
            &model.blocks[0].qkv,
            &model.blocks[0].output,
            &model.blocks[0].expand,
            &model.blocks[0].contract,
        ] {
            assert!(dense.backward_weight.get().is_none());
        }
        assert!(model.blocks[1].qkv.backward_weight.get().is_none());
        assert!(model.blocks[1].output.backward_weight.get().is_none());
        assert_eq!(
            model.blocks[1].expand.backward_weight.get().is_some(),
            suffix == ".attn"
        );
        assert_eq!(
            model.blocks[1].contract.backward_weight.get().is_some(),
            suffix == ".attn"
        );
    }
    let base = load();
    let mut record = trainable(&base).record().clone();
    for p in &mut record.placements {
        for t in &mut p.tensors {
            t.requires_grad = p.path == "transformer.h.1";
        }
    }
    let model = base
        .with_geometry(GeometryAdapterArtifact::create(record).unwrap())
        .unwrap();
    let gradients = model.geometry_loss_gradients(&[1, 2, 3], [1, 3]).unwrap();
    assert_eq!(gradients.parameters().len(), 3);
    assert!(gradients.parameters()["transformer.h.0.attn"]["gate"]
        .iter()
        .any(|v| *v != 0.));
}

#[test]
fn cold_parallel_pullbacks_share_transposes_and_return_identical_gradients() {
    let base = load();
    let model = base.with_geometry(trainable(&base)).unwrap();
    let clone = model.clone();
    let barrier = std::sync::Barrier::new(3);
    std::thread::scope(|scope| {
        let a = scope.spawn(|| {
            barrier.wait();
            model.geometry_loss_gradients(&[1, 2, 3], [1, 3]).unwrap()
        });
        let b = scope.spawn(|| {
            barrier.wait();
            clone.geometry_loss_gradients(&[1, 2, 3], [1, 3]).unwrap()
        });
        barrier.wait();
        let (a, b) = (a.join().unwrap(), b.join().unwrap());
        assert_eq!(a.loss(), b.loss());
        assert_eq!(a.parameters(), b.parameters());
    });
    assert!(Arc::ptr_eq(&model.backward_head, &clone.backward_head));
    assert!(model.backward_head.get().unwrap().is_ok());
}
