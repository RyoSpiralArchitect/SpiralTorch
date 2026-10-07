use super::*;

fn operator(coupling: f32, iterations: usize, porosity: f32) -> ToposResonatorOperator {
    ToposResonatorOperator::new(
        ToposResonatorConfig::new(coupling, iterations).unwrap(),
        OpenCartesianTopos::new(-1.0, 1e-6, 1.0, 4097, 8192)
            .unwrap()
            .with_porosity(porosity)
            .unwrap(),
    )
    .unwrap()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn shared_capture_matches_expanded_recurrence_and_tensor_reduction() {
    let pattern = [0.0, -0.0, f32::from_bits(1), -0.3, 0.5, 1.3, -2.0];
    for rows in [0, 1, 7, 257] {
        for features in [1, 3, 17] {
            for (coupling, iterations) in [(0.0, 1), (0.25, 5), (0.75, 64)] {
                for porosity in [0.0, f32::EPSILON, 0.3, 1.0] {
                    let op = operator(coupling, iterations, porosity);
                    let input: Vec<_> = (0..rows * features)
                        .map(|i| pattern[i % pattern.len()])
                        .collect();
                    let gate: Vec<_> = (0..features).map(|i| (i % 5) as f32 - 1.5).collect();
                    let expanded = gate.repeat(rows);
                    let expected = op.capture(&input, &expanded, rows, features).unwrap();
                    let batch = op
                        .capture_shared_rows(&input, &gate, rows, features)
                        .unwrap();
                    assert_eq!(batch.gate_layout(), ToposResonatorGateLayout::SharedRows);
                    assert_eq!(batch.gate().len(), features);
                    assert_eq!(bits(batch.output()), bits(expected.output()));
                    assert_eq!(
                        serde_json::to_string(batch.step()).unwrap(),
                        serde_json::to_string(expected.step()).unwrap()
                    );
                    assert_eq!(
                        op.forward_shared_rows(&input, &gate, rows, features)
                            .unwrap(),
                        *batch.step()
                    );
                    for sign in [1.0, -0.5] {
                        let dy: Vec<_> = (0..input.len())
                            .map(|i| sign * ((i % 11) as f32 / 7.0 - 0.5))
                            .collect();
                        let expected_vjp = expected.vjp(&dy).unwrap();
                        let (actual, audit) = batch.vjp_audited(&dy).unwrap();
                        let tensor =
                            st_tensor::Tensor::from_vec(rows, features, expected_vjp.grad_gate)
                                .unwrap();
                        assert_eq!(bits(&actual.grad_input), bits(&expected_vjp.grad_input));
                        assert_eq!(
                            bits(&actual.grad_gate),
                            bits(&tensor.try_sum_axis0().unwrap())
                        );
                        assert_eq!(audit.grad_gate_rms, root_mean_square(&actual.grad_gate));
                        assert_eq!(audit.rows, rows);
                        assert_eq!(audit.features, features);
                    }
                }
            }
        }
    }
    let op = operator(0.99, 4096, 0.3);
    let shared = op.capture_shared_rows(&[0.1; 6], &[0.2; 3], 2, 3).unwrap();
    let expanded = op.capture(&[0.1; 6], &[0.2; 6], 2, 3).unwrap();
    assert_eq!(bits(shared.output()), bits(expanded.output()));
    assert_eq!(
        shared.vjp(&[0.1; 6]).unwrap().grad_input,
        expanded.vjp(&[0.1; 6]).unwrap().grad_input
    );
}

#[test]
fn shared_capture_owns_only_feature_gate_and_survives_source_mutation() {
    let op = operator(0.25, 5, 0.3);
    let mut input = vec![0.2; 6];
    let mut gate = vec![0.5; 3];
    let borrowed = op.capture_shared_rows(&input, &gate, 2, 3).unwrap();
    let allocations = (
        input.as_ptr(),
        input.capacity(),
        gate.as_ptr(),
        gate.capacity(),
    );
    let owned = op
        .capture_shared_rows_owned(input.clone(), gate.clone(), 2, 3)
        .unwrap();
    let retained = op.capture_shared_rows_owned(input, gate, 2, 3).unwrap();
    assert_eq!(
        (
            retained.input.as_ptr(),
            retained.input.capacity(),
            retained.gate.as_ptr(),
            retained.gate.capacity()
        ),
        allocations
    );
    input = vec![0.2; 6];
    gate = vec![0.5; 3];
    let snapshot = op.capture_shared_rows(&input, &gate, 2, 3).unwrap();
    input.fill(f32::NAN);
    gate.fill(f32::NAN);
    drop(op);
    for batch in [borrowed, owned, retained, snapshot.clone()] {
        assert_eq!(batch.step(), snapshot.step());
        assert_eq!(
            batch.vjp(&[0.3; 6]).unwrap(),
            snapshot.vjp(&[0.3; 6]).unwrap()
        );
    }
}

#[test]
fn shared_sum_preserves_finite_cancellation_and_rejects_final_overflow() {
    let op = operator(0.0, 1, 0.0);
    let batch = op
        .capture_shared_rows(
            &[f32::MAX, f32::MAX, -f32::MAX, -f32::MAX, 1.0],
            &[0.0],
            5,
            1,
        )
        .unwrap();
    assert_eq!(batch.vjp(&[1.0; 5]).unwrap().grad_gate, [1.0]);
    assert!(matches!(
        batch.vjp(&[1.0, 1.0, 0.0, 0.0, 0.0]),
        Err(ToposResonatorError::NonFiniteDerived {
            field: "grad_gate_sum",
            ..
        })
    ));
    assert_eq!(batch.vjp(&[1.0; 5]).unwrap().grad_gate, [1.0]);
    // A wide sum cannot hide a nonfinite individual f32 contribution.
    assert!(batch.vjp(&[2.0, -2.0, 0.0, 0.0, 0.0]).is_err());
    let empty = op.capture_shared_rows(&[], &[0.3, -0.0], 0, 2).unwrap();
    assert_eq!(
        empty.vjp(&[]).unwrap(),
        ToposResonatorBackward {
            grad_input: vec![],
            grad_gate: vec![0.0; 2]
        }
    );
}

#[test]
fn shared_shape_finite_and_budget_guards_preserve_reusability() {
    let op = operator(0.25, 5, 0.3);
    for (input, gate, rows, features) in [
        (vec![], vec![], 0, 0),
        (vec![], vec![1.0], usize::MAX, 2),
        (vec![1.0; 2], vec![1.0; 2], 2, 1),
        (vec![], vec![1.0; 8193], 0, 8193),
        (vec![1.0; 8193], vec![1.0], 8193, 1),
        (vec![], vec![f32::NAN], 0, 1),
        (vec![f32::INFINITY], vec![1.0], 1, 1),
        (vec![f32::MAX], vec![2.0], 1, 1),
    ] {
        let expected = op
            .capture_shared_rows(&input, &gate, rows, features)
            .unwrap_err();
        let forward = op
            .forward_shared_rows(&input, &gate, rows, features)
            .unwrap_err();
        let owned = op
            .capture_shared_rows_owned(input, gate, rows, features)
            .unwrap_err();
        assert_eq!(format!("{expected:?}"), format!("{forward:?}"));
        assert_eq!(format!("{expected:?}"), format!("{owned:?}"));
    }
    let batch = op.capture_shared_rows(&[0.2; 6], &[0.5; 3], 2, 3).unwrap();
    for dy in [vec![], vec![f32::NAN; 6], vec![f32::INFINITY; 6]] {
        assert!(batch.vjp(&dy).is_err());
    }
    assert_eq!(batch.vjp(&[0.1; 6]).unwrap().grad_gate.len(), 3);
}
