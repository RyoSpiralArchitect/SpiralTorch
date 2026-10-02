use super::*;

#[test]
fn attention_vjp_matches_all_input_finite_differences_with_batches_heads_and_offsets() {
    for mask in [
        AttentionMask::None,
        AttentionMask::Causal { query_offset: 1 },
    ] {
        let spec =
            AttentionSpec::new(&[2, 2, 2, 2], &[2, 2, 3, 2], &[2, 2, 3, 2], 0.7, mask).unwrap();
        let make = |n: usize, shift: f32| {
            (0..n)
                .map(|i| ((i as f32 + shift) * 0.3).sin())
                .collect::<Vec<_>>()
        };
        let fields = [
            make(16, 0.),
            make(24, 1.),
            make(24, 2.),
            make(12, 3.),
            make(24, 4.),
        ];
        let upstream = make(16, 5.);
        let g = attention_vjp_reference(
            spec,
            &fields[0],
            &fields[1],
            &fields[2],
            Some(&fields[3]),
            Some(&fields[4]),
            &upstream,
        )
        .unwrap();
        let grads = [
            g.query,
            g.key,
            g.value,
            g.z_bias.unwrap(),
            g.pair_bias.unwrap(),
        ];
        let loss = |f: &[Vec<f32>; 5]| {
            attention_reference(spec, &f[0], &f[1], &f[2], Some(&f[3]), Some(&f[4]))
                .unwrap()
                .iter()
                .zip(&upstream)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum::<f64>()
        };
        for field in 0..5 {
            for i in 0..fields[field].len() {
                let mut plus = fields.clone();
                let mut minus = fields.clone();
                plus[field][i] += 0.001;
                minus[field][i] -= 0.001;
                let numeric =
                    (loss(&plus) - loss(&minus)) / f64::from(plus[field][i] - minus[field][i]);
                assert!(
                    (numeric - f64::from(grads[field][i])).abs() < 0.0003,
                    "field={field} index={i} numerical={numeric} analytic={}",
                    grads[field][i]
                );
            }
        }
    }
}

#[test]
fn attention_vjp_preserves_causality_and_has_no_cross_batch_gradients() {
    let shape = [2, 1, 3, 1];
    let spec = AttentionSpec::new(
        &shape,
        &shape,
        &shape,
        0.5,
        AttentionMask::Causal { query_offset: 0 },
    )
    .unwrap();
    let g = attention_vjp_reference(
        spec,
        &[1.; 6],
        &[2.; 6],
        &[3.; 6],
        Some(&[0.; 6]),
        Some(&[0.; 18]),
        &[1., 0., 0., 0., 0., 0.],
    )
    .unwrap();
    assert_eq!(g.query, [0.; 6]);
    assert_eq!(g.key, [0.; 6]);
    assert_eq!(g.value, [1., 0., 0., 0., 0., 0.]);
    assert_eq!(g.pair_bias.unwrap(), [0.; 18]);
    assert_eq!(g.z_bias.unwrap(), [0.; 6]);
}

#[test]
fn attention_vjp_rejects_invalid_upstream_and_handles_empty_without_key_scratch() {
    let shape = [1, 1, 1, 1];
    let spec = AttentionSpec::new(&shape, &shape, &shape, 1., AttentionMask::None).unwrap();
    assert!(matches!(
        attention_vjp_reference(spec, &[1.], &[1.], &[1.], None, None, &[]),
        Err(AttentionError::Length)
    ));
    assert!(matches!(
        attention_vjp_reference(spec, &[1.], &[1.], &[1.], None, None, &[f32::NAN]),
        Err(AttentionError::NonFinite)
    ));
    let empty = [0, 1, usize::MAX, 1];
    let spec = AttentionSpec::new(&empty, &empty, &empty, 1., AttentionMask::None).unwrap();
    let g = attention_vjp_reference(spec, &[], &[], &[], None, None, &[]).unwrap();
    assert!(g.query.is_empty() && g.key.is_empty() && g.value.is_empty());
    assert!(g.z_bias.is_none() && g.pair_bias.is_none());
}

#[test]
fn merged_output_shape_checks_width_even_with_an_empty_batch() {
    for shape in [[2, 3, 5, 7], [0, 3, 5, 7], [2, 0, 5, 7], [2, 3, 0, 7]] {
        let key = [shape[0], shape[1], 6, shape[3]];
        let spec = AttentionSpec::new(&shape, &key, &key, 1., AttentionMask::None).unwrap();
        assert_eq!(
            spec.merged_output_shape().unwrap(),
            [shape[0], shape[2], shape[1] * shape[3]]
        );
    }
    let shape = [0, usize::MAX, 1, 2];
    let spec = AttentionSpec::new(&shape, &shape, &shape, 1., AttentionMask::None).unwrap();
    assert_eq!(spec.merged_output_shape(), Err(AttentionError::Overflow));
}

#[test]
fn causal_prefill_and_cached_query_offsets_are_not_interchangeable() {
    let shape = [1, 1, 3, 1];
    let spec = AttentionSpec::new(
        &shape,
        &shape,
        &shape,
        1.,
        AttentionMask::Causal { query_offset: 0 },
    )
    .unwrap();
    let output = attention_reference(spec, &[0.; 3], &[0.; 3], &[2., 4., 6.], None, None).unwrap();
    assert_eq!(output, [2., 3., 4.]);
    let spec = AttentionSpec::new(
        &[1, 1, 1, 1],
        &shape,
        &shape,
        1.,
        AttentionMask::Causal { query_offset: 2 },
    )
    .unwrap();
    assert_eq!(
        attention_reference(spec, &[0.], &[0.; 3], &[2., 4., 6.], None, None).unwrap(),
        [4.]
    );
}

#[test]
fn biases_are_post_scale_and_cannot_unmask_future_keys() {
    let shape = [1, 1, 2, 1];
    let spec = AttentionSpec::new(
        &shape,
        &shape,
        &shape,
        0.,
        AttentionMask::Causal { query_offset: 0 },
    )
    .unwrap();
    let output = attention_reference(
        spec,
        &[3.; 2],
        &[4.; 2],
        &[2., 10.],
        Some(&[0., 3f32.ln()]),
        Some(&[0., 100., 0., 0.]),
    )
    .unwrap();
    assert!((output[0] - 2.).abs() < 1e-6);
    assert!((output[1] - 8.).abs() < 1e-6);
}

#[test]
fn empty_queries_and_batches_are_valid_but_keys_and_head_dimensions_are_not_empty() {
    for shape in [[0, 2, 4, 3], [2, 0, 4, 3], [2, 2, 0, 3]] {
        let key = [shape[0], shape[1], 4, 3];
        let spec = AttentionSpec::new(&shape, &key, &key, 1., AttentionMask::None).unwrap();
        let kv = vec![0.; product(&key).unwrap()];
        assert!(attention_reference(spec, &[], &kv, &kv, None, None)
            .unwrap()
            .is_empty());
    }
    for shape in [[1, 1, 0, 2], [1, 1, 2, 0]] {
        assert_eq!(
            AttentionSpec::new(&shape, &shape, &shape, 1., AttentionMask::None).unwrap_err(),
            AttentionError::Shape
        );
    }
}

#[test]
fn shape_offset_scalar_and_bias_errors_fail_before_execution() {
    let shape = [1, 1, 2, 3];
    let spec = AttentionSpec::new(&shape, &shape, &shape, 1., AttentionMask::None).unwrap();
    assert_eq!(
        spec.validate_bias_shapes(Some(&[2]), None),
        Err(AttentionError::BiasShape)
    );
    assert_eq!(
        spec.validate_bias_shapes(None, Some(&[1, 1, 2, 3])),
        Err(AttentionError::BiasShape)
    );
    for offset in [1, usize::MAX] {
        assert_eq!(
            AttentionSpec::new(
                &shape,
                &shape,
                &shape,
                1.,
                AttentionMask::Causal {
                    query_offset: offset
                }
            )
            .unwrap_err(),
            AttentionError::QueryOffset
        );
    }
    assert_eq!(
        AttentionSpec::new(&shape, &shape, &shape, f32::NAN, AttentionMask::None).unwrap_err(),
        AttentionError::Scale
    );
    assert_eq!(
        AttentionSpec::new(
            &[2, usize::MAX, 2, 1],
            &[2, usize::MAX, 2, 1],
            &[2, usize::MAX, 2, 1],
            1.,
            AttentionMask::None
        )
        .unwrap_err(),
        AttentionError::Overflow
    );
    assert_eq!(
        AttentionSpec::new(
            &shape,
            &[1, 2, 2, 3],
            &[1, 2, 2, 3],
            1.,
            AttentionMask::None
        )
        .unwrap_err(),
        AttentionError::Shape
    );
}

#[test]
fn invalid_data_and_overflow_fail_closed() {
    let shape = [1, 1, 1, 1];
    let spec = AttentionSpec::new(&shape, &shape, &shape, 1., AttentionMask::None).unwrap();
    assert_eq!(
        attention_reference(spec, &[], &[1.], &[1.], None, None),
        Err(AttentionError::Length)
    );
    assert_eq!(
        attention_reference(spec, &[f32::NAN], &[1.], &[1.], None, None),
        Err(AttentionError::NonFinite)
    );
    assert_eq!(
        attention_reference(spec, &[f32::MAX], &[2.], &[1.], None, None),
        Err(AttentionError::NonFinite)
    );
    assert_eq!(
        attention_reference(spec, &[0.], &[1.], &[1.], Some(&[f32::INFINITY]), None),
        Err(AttentionError::NonFinite)
    );
}
