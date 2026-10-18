use super::{
    tests::elliptic_artifact, GeometryAdapterArtifact, GeometryAdapterRecipe, NativeGeometryRuntime,
};

fn artifact(gated: bool, strength: f64) -> GeometryAdapterArtifact {
    let mut record = elliptic_artifact(gated, strength).record().clone();
    let (features, warp) = match &record.placements[0].recipe {
        GeometryAdapterRecipe::Elliptic { features, warp, .. }
        | GeometryAdapterRecipe::EllipticAnchored { features, warp, .. } => {
            (*features, warp.clone())
        }
        _ => unreachable!(),
    };
    record.placements[0].recipe = if gated {
        GeometryAdapterRecipe::EllipticGated {
            features,
            strength,
            warp,
        }
    } else {
        GeometryAdapterRecipe::EllipticCausal {
            features,
            strength,
            warp,
        }
    };
    GeometryAdapterArtifact::create(record).unwrap()
}

fn input() -> Vec<f32> {
    (0..18)
        .map(|i| ((i * 7) % 19) as f32 * 0.05 - 0.3)
        .collect()
}

#[test]
fn full_sequence_vjp_matches_finite_differences_including_the_context_gate() {
    let x = input();
    let seed: Vec<_> = (0..18).map(|i| (i % 5) as f32 * 0.2 - 0.3).collect();
    for gated in [false, true] {
        let runtime = NativeGeometryRuntime::new(artifact(gated, 0.4));
        let batch = runtime.forward_sequence("site", &x, [2, 3]).unwrap();
        assert_eq!(batch.sequence_shape(), Some([2, 3, 3]));
        assert_eq!(batch.output().shape(), (6, 3));
        let gradient = batch.vjp(&seed).unwrap();
        let objective = |runtime: &NativeGeometryRuntime, values: &[f32]| {
            runtime
                .forward_sequence("site", values, [2, 3])
                .unwrap()
                .output()
                .data()
                .iter()
                .zip(&seed)
                .map(|(v, d)| v * d)
                .sum::<f32>()
        };
        let epsilon = 0.002;
        for index in 0..x.len() {
            let (mut plus, mut minus) = (x.clone(), x.clone());
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let numerical =
                (objective(&runtime, &plus) - objective(&runtime, &minus)) / (2. * epsilon);
            assert!(
                (numerical - gradient.input.data()[index]).abs() < 3e-4,
                "input {index}"
            );
        }
        for (name, values) in &gradient.parameters {
            for (index, expected) in values.iter().enumerate() {
                let perturbed = |sign: f32| {
                    let mut record = runtime.artifact().record().clone();
                    record.placements[0]
                        .tensors
                        .iter_mut()
                        .find(|t| t.name == *name)
                        .unwrap()
                        .values[index] += sign * epsilon;
                    NativeGeometryRuntime::new(GeometryAdapterArtifact::create(record).unwrap())
                };
                let numerical = (objective(&perturbed(1.), &x) - objective(&perturbed(-1.), &x))
                    / (2. * epsilon);
                assert!(
                    (numerical - expected).abs() < 3e-4,
                    "{name}[{index}]: {numerical} vs {expected}"
                );
            }
        }
        assert_eq!(gradient.parameters.contains_key("raw_mix"), gated);
    }
}

#[test]
fn causal_outputs_and_input_vjps_preserve_batch_and_future_boundaries() {
    let x = input();
    for gated in [false, true] {
        let runtime = NativeGeometryRuntime::new(artifact(gated, 0.4));
        let batch = runtime.forward_sequence("site", &x, [2, 3]).unwrap();
        for index in 0..2 {
            let separate = runtime
                .forward_sequence("site", &x[index * 9..(index + 1) * 9], [1, 3])
                .unwrap();
            assert_eq!(
                separate.output().data(),
                &batch.output().data()[index * 9..(index + 1) * 9]
            );
        }
        let mut changed = x.clone();
        changed[6..].iter_mut().for_each(|v| *v += 3.);
        assert_eq!(
            &runtime
                .forward_sequence("site", &changed, [2, 3])
                .unwrap()
                .output()
                .data()[..6],
            &batch.output().data()[..6]
        );
        let mut first_only = vec![0.; 18];
        first_only[..3].copy_from_slice(&[0.2, -0.4, 0.1]);
        let gradient = batch.vjp(&first_only).unwrap();
        assert!(gradient.input.data()[3..].iter().all(|v| *v == 0.));
        let mut last_only = vec![0.; 18];
        last_only[6..9].copy_from_slice(&[0.2, -0.4, 0.1]);
        let gradient = batch.vjp(&last_only).unwrap();
        assert!(gradient.input.data()[..6].iter().any(|v| v.abs() > 1e-8));
        assert!(gradient.input.data()[9..].iter().all(|v| *v == 0.));
    }
}

#[test]
fn zero_context_gate_is_pointwise_but_still_has_a_learning_signal() {
    let mut record = artifact(true, 0.4).record().clone();
    record.placements[0]
        .tensors
        .iter_mut()
        .find(|t| t.name == "raw_mix")
        .unwrap()
        .values[0] = 0.;
    let gated = NativeGeometryRuntime::new(GeometryAdapterArtifact::create(record).unwrap());
    let pointwise = NativeGeometryRuntime::new(elliptic_artifact(false, 0.4));
    let x = input();
    let sequence = gated.forward_sequence("site", &x, [2, 3]).unwrap();
    let rows = pointwise.forward("site", &x, 6).unwrap();
    assert_eq!(sequence.output().data(), rows.output().data());
    let seed: Vec<_> = (0..18).map(|i| i as f32 * 0.1 - 0.3).collect();
    let gradient = sequence.vjp(&seed).unwrap();
    let reference = rows.vjp(&seed).unwrap();
    assert_eq!(gradient.input.data(), reference.input.data());
    for (name, values) in reference.parameters {
        assert_eq!(gradient.parameters[&name], values);
    }
    assert!(gradient.parameters["raw_mix"][0].abs() > 1e-8);
}

#[test]
fn sequence_learning_restores_exactly_and_invalid_updates_are_atomic() {
    let x = input();
    let target = vec![0.1; 18];
    for gated in [false, true] {
        let mut runtime = NativeGeometryRuntime::new(artifact(gated, 0.4));
        let first = runtime
            .train_sequence_mse("site", &x, &target, [2, 3], 0.2)
            .unwrap();
        for _ in 0..39 {
            runtime
                .train_sequence_mse("site", &x, &target, [2, 3], 0.2)
                .unwrap();
        }
        let saved = runtime.artifact();
        let loaded = GeometryAdapterArtifact::from_json(
            saved.to_json().as_bytes(),
            saved.artifact_id(),
            &saved.record().base_model_identity,
        )
        .unwrap();
        let mut restored = NativeGeometryRuntime::new(loaded);
        let loss = runtime
            .train_sequence_mse("site", &x, &target, [2, 3], 0.2)
            .unwrap();
        assert!(loss < first);
        assert_eq!(
            loss,
            restored
                .train_sequence_mse("site", &x, &target, [2, 3], 0.2)
                .unwrap()
        );
        assert_eq!(
            runtime.artifact().artifact_id(),
            restored.artifact().artifact_id()
        );
        let id = runtime.artifact().artifact_id().to_string();
        assert!(runtime
            .train_sequence_mse("site", &x, &[f32::NAN; 18], [2, 3], 0.2)
            .is_err());
        assert!(runtime
            .train_sequence_mse("site", &x, &[0.; 19], [2, 3], 0.2)
            .is_err());
        assert_eq!(runtime.artifact().artifact_id(), id);
    }
}

#[test]
fn explicit_sequence_admission_precedes_input_copy_even_at_zero_strength() {
    for strength in [0., 0.5] {
        let runtime = NativeGeometryRuntime::new(artifact(true, strength));
        for shape in [[0, 1], [1, 0], [usize::MAX, 2], [65_537, 1], [1, 1025]] {
            assert!(runtime.sequence_dimensions("site", shape).is_err());
            assert!(runtime.forward_sequence("site", &[], shape).is_err());
        }
        assert!(runtime.forward("site", &input(), 6).is_err());
        assert_eq!(
            runtime.sequence_dimensions("site", [2, 3]).unwrap(),
            (3, 18)
        );
    }
    let identity = NativeGeometryRuntime::new(artifact(true, 0.))
        .forward_sequence("site", &[f32::MAX; 18], [2, 3])
        .unwrap();
    assert_eq!(identity.output().data(), &[f32::MAX; 18]);
    assert_eq!(identity.sequence_shape(), Some([2, 3, 3]));
    let pointwise = NativeGeometryRuntime::new(elliptic_artifact(false, 0.5));
    let flat = pointwise.forward("site", &input(), 6).unwrap();
    let sequence = pointwise
        .forward_sequence("site", &input(), [2, 3])
        .unwrap();
    assert_eq!(flat.sequence_shape(), None);
    assert_eq!(sequence.sequence_shape(), Some([2, 3, 3]));
    assert_eq!(flat.output().data(), sequence.output().data());
    assert_eq!(
        flat.vjp(&[1.; 18]).unwrap().parameters,
        sequence.vjp(&[1.; 18]).unwrap().parameters
    );
}
