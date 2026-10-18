use super::{GeometryAdapterArtifact, GeometryAdapterRecipe, NativeGeometryRuntime};
use crate::geometry_artifact::{
    FractionalRecipe, GeometryAdapterTensor, GeometryAdapterTensorRole, ToposRecipe,
};

fn artifact(kind: usize) -> GeometryAdapterArtifact {
    let mut record = super::tests::artifact(false, None, 0.4).record().clone();
    let p = &mut record.placements[0];
    p.tensors.retain(|t| t.name == "gate");
    let kernel = FractionalRecipe {
        kernel_len: 5,
        step: 0.8,
        max_values: 96,
        max_products: 480,
    };
    p.recipe = match kind {
        0 => GeometryAdapterRecipe::Topos {
            features: 3,
            strength: 0.4,
            kernel: ToposRecipe {
                coupling: 0.2,
                iterations: 4,
                saturation: 1.,
                porosity: 0.05,
                max_values: 96,
            },
        },
        1 => GeometryAdapterRecipe::Fractional {
            features: 3,
            strength: 0.4,
            kernel,
        },
        2 => GeometryAdapterRecipe::History {
            features: 3,
            strength: 0.4,
            kernel,
        },
        3 => GeometryAdapterRecipe::HistoryL2 {
            features: 3,
            strength: 0.4,
            kernel,
            gain: 0.8,
        },
        4 => GeometryAdapterRecipe::HistoryGain {
            features: 3,
            strength: 0.4,
            kernel,
            lag_window: Some([1, 4]),
        },
        5 => GeometryAdapterRecipe::HistoryAngle {
            features: 3,
            strength: 0.4,
            kernel,
            lag_window: Some([1, 4]),
        },
        _ => unreachable!(),
    };
    let mut parameter = |name: &str, shape, values| {
        p.tensors.push(GeometryAdapterTensor {
            name: name.into(),
            shape,
            values,
            requires_grad: true,
            role: GeometryAdapterTensorRole::Parameter,
        })
    };
    if kind > 0 {
        parameter(
            if kind == 5 {
                "history_angle"
            } else {
                "log_alpha"
            },
            vec![],
            vec![-0.2],
        );
    }
    if kind > 1 {
        parameter("local_gate", vec![3], vec![0.2, -0.1, 0.15]);
    }
    if kind > 3 {
        parameter("log_gain", vec![], vec![0.1]);
    }
    GeometryAdapterArtifact::create(record).unwrap()
}

fn input() -> Vec<f32> {
    (0..18).map(|i| (i * 7 % 19) as f32 * 0.05 - 0.3).collect()
}

#[test]
fn composite_input_and_all_parameter_vjps_match_finite_differences() {
    let x = input();
    let seed: Vec<_> = (0..18).map(|i| (i % 5) as f32 * 0.2 - 0.3).collect();
    for kind in 0..6 {
        let runtime = NativeGeometryRuntime::new(artifact(kind));
        let batch = runtime.forward_sequence("site", &x, [2, 3]).unwrap();
        let gradient = batch.vjp(&seed).unwrap();
        assert_eq!(
            gradient.parameters.len(),
            runtime.artifact().record().placements[0].tensors.len()
        );
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
        let epsilon = 0.001;
        for index in 0..x.len() {
            let (mut plus, mut minus) = (x.clone(), x.clone());
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let numerical =
                (objective(&runtime, &plus) - objective(&runtime, &minus)) / (2. * epsilon);
            assert!(
                (numerical - gradient.input.data()[index]).abs() < 2e-4,
                "kind {kind} input {index}"
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
                    (numerical - expected).abs() < 2e-4,
                    "kind {kind} {name}[{index}]: {numerical} vs {expected}"
                );
            }
        }
    }
}

#[test]
fn fractional_contexts_keep_past_gradients_and_exclude_future_and_other_batches() {
    let x = input();
    for kind in 1..6 {
        let runtime = NativeGeometryRuntime::new(artifact(kind));
        assert!(runtime.forward("site", &x, 6).is_err());
        let batch = runtime.forward_sequence("site", &x, [2, 3]).unwrap();
        for b in 0..2 {
            assert_eq!(
                runtime
                    .forward_sequence("site", &x[b * 9..(b + 1) * 9], [1, 3])
                    .unwrap()
                    .output()
                    .data(),
                &batch.output().data()[b * 9..(b + 1) * 9]
            );
        }
        let mut changed = x.clone();
        changed[6..].iter_mut().for_each(|v| *v += 2.);
        assert_eq!(
            &batch.output().data()[..6],
            &runtime
                .forward_sequence("site", &changed, [2, 3])
                .unwrap()
                .output()
                .data()[..6]
        );
        let mut seed = vec![0.; 18];
        seed[..3].copy_from_slice(&[0.2, -0.4, 0.1]);
        assert!(batch.vjp(&seed).unwrap().input.data()[3..]
            .iter()
            .all(|v| *v == 0.));
        seed[..3].fill(0.);
        seed[6..9].copy_from_slice(&[0.2, -0.4, 0.1]);
        let gradient = batch.vjp(&seed).unwrap();
        assert!(gradient.input.data()[..6].iter().any(|v| v.abs() > 1e-8));
        assert!(gradient.input.data()[9..].iter().all(|v| *v == 0.));
    }
}

#[test]
fn zero_feature_gates_are_identity_but_learn_and_topos_rows_remain_independent() {
    let x = input();
    for kind in 0..6 {
        let mut record = artifact(kind).record().clone();
        for t in &mut record.placements[0].tensors {
            if t.name.ends_with("gate") {
                t.values.fill(0.);
            }
        }
        let runtime = NativeGeometryRuntime::new(GeometryAdapterArtifact::create(record).unwrap());
        let batch = runtime.forward_sequence("site", &x, [2, 3]).unwrap();
        assert_eq!(batch.output().data(), x);
        let gradient = batch.vjp(&x).unwrap();
        assert_eq!(gradient.input.data(), x);
        assert!(gradient.parameters["gate"].iter().any(|v| v.abs() > 1e-8));
        for (name, values) in gradient.parameters {
            if !name.ends_with("gate") {
                assert!(values.iter().all(|v| *v == 0.));
            }
        }
    }
    let runtime = NativeGeometryRuntime::new(artifact(0));
    let flat = runtime.forward("site", &x, 6).unwrap();
    let sequence = runtime.forward_sequence("site", &x, [2, 3]).unwrap();
    assert_eq!(flat.output().data(), sequence.output().data());
    assert_eq!(
        flat.vjp(&x).unwrap().parameters,
        sequence.vjp(&x).unwrap().parameters
    );
    let mut seed = vec![0.; 18];
    seed[..3].copy_from_slice(&[0.2, -0.4, 0.1]);
    assert!(flat.vjp(&seed).unwrap().input.data()[3..]
        .iter()
        .all(|v| *v == 0.));
    let empty = runtime.forward("site", &[], 0).unwrap();
    assert!(empty.output().data().is_empty());
    assert_eq!(empty.vjp(&[]).unwrap().parameters["gate"], vec![0.; 3]);
}

#[test]
fn empty_history_windows_leave_only_the_local_gate_and_no_order_gain_derivative() {
    for kind in 4..6 {
        let mut record = artifact(kind).record().clone();
        match &mut record.placements[0].recipe {
            GeometryAdapterRecipe::HistoryGain { lag_window, .. }
            | GeometryAdapterRecipe::HistoryAngle { lag_window, .. } => *lag_window = Some([2, 2]),
            _ => unreachable!(),
        }
        let runtime = NativeGeometryRuntime::new(GeometryAdapterArtifact::create(record).unwrap());
        let x = input();
        let batch = runtime.forward_sequence("site", &x, [2, 3]).unwrap();
        let gradient = batch.vjp(&x).unwrap();
        assert!(gradient.parameters["local_gate"].iter().any(|v| *v != 0.));
        for (name, values) in gradient.parameters {
            if name != "local_gate" {
                assert!(values.iter().all(|v| *v == 0.));
            }
        }
        for (i, (&value, &source)) in batch.output().data().iter().zip(&x).enumerate() {
            let gate = [0.2f32, -0.1, 0.15][i % 3].tanh();
            assert!((value - (source + 0.4 * gate * source)).abs() < 1e-7);
        }
    }
}

#[test]
fn native_memory_learning_reloads_exactly_and_invalid_order_updates_are_atomic() {
    let x = input();
    let target = vec![0.1; 18];
    for kind in 0..6 {
        let mut runtime = NativeGeometryRuntime::new(artifact(kind));
        let first = runtime
            .train_sequence_mse("site", &x, &target, [2, 3], 0.2)
            .unwrap();
        for _ in 0..39 {
            runtime
                .train_sequence_mse("site", &x, &target, [2, 3], 0.2)
                .unwrap();
        }
        let saved = runtime.artifact();
        let mut restored = NativeGeometryRuntime::new(
            GeometryAdapterArtifact::from_json(
                saved.to_json().as_bytes(),
                saved.artifact_id(),
                &saved.record().base_model_identity,
            )
            .unwrap(),
        );
        let last = runtime
            .train_sequence_mse("site", &x, &target, [2, 3], 0.2)
            .unwrap();
        assert!(last < first, "kind {kind}");
        assert_eq!(
            last,
            restored
                .train_sequence_mse("site", &x, &target, [2, 3], 0.2)
                .unwrap()
        );
        assert_eq!(
            runtime.artifact().artifact_id(),
            restored.artifact().artifact_id()
        );
    }
    let mut runtime = NativeGeometryRuntime::new(artifact(5));
    let batch = runtime.forward_sequence("site", &x, [2, 3]).unwrap();
    assert!(batch.vjp(&x).unwrap().parameters["history_angle"][0].abs() > 1e-8);
    let id = runtime.artifact().artifact_id().to_string();
    assert!(runtime.step_sgd(&batch, &x, 1e10).is_err());
    assert_eq!(runtime.artifact().artifact_id(), id);
    let mut record = runtime.artifact().record().clone();
    for t in &mut record.placements[0].tensors {
        if !t.name.ends_with("gate") {
            t.requires_grad = false;
        }
    }
    let before = record.placements[0].tensors.clone();
    let mut frozen = NativeGeometryRuntime::new(GeometryAdapterArtifact::create(record).unwrap());
    frozen
        .train_sequence_mse("site", &x, &target, [2, 3], 0.2)
        .unwrap();
    for (old, new) in before
        .iter()
        .zip(&frozen.artifact().record().placements[0].tensors)
    {
        if !old.requires_grad {
            assert_eq!(old.values, new.values);
        }
    }
}

#[test]
fn convolution_products_and_value_budgets_apply_before_copy_even_on_identity() {
    for strength in [0., 0.4] {
        let mut record = artifact(1).record().clone();
        if let GeometryAdapterRecipe::Fractional {
            kernel,
            strength: value,
            ..
        } = &mut record.placements[0].recipe
        {
            kernel.max_products = 90;
            *value = strength;
        }
        let runtime = NativeGeometryRuntime::new(GeometryAdapterArtifact::create(record).unwrap());
        assert_eq!(
            runtime.sequence_dimensions("site", [2, 3]).unwrap(),
            (3, 18)
        );
        for shape in [[0, 1], [1, 0], [2, 4], [usize::MAX, 2], [1, 33]] {
            assert!(runtime.sequence_dimensions("site", shape).is_err());
            assert!(runtime.forward_sequence("site", &[], shape).is_err());
        }
        if strength == 0. {
            let batch = runtime
                .forward_sequence("site", &[f32::MAX; 18], [2, 3])
                .unwrap();
            assert_eq!(batch.output().data(), &[f32::MAX; 18]);
            assert!(batch
                .vjp(&[1.; 18])
                .unwrap()
                .parameters
                .values()
                .flatten()
                .all(|v| *v == 0.));
        }
    }
}
