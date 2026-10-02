use super::*;

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
