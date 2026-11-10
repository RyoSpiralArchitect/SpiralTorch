use super::*;

#[test]
fn lag_matches_materialized_keys_and_their_scatter_adjoint() {
    let spec = AttentionSpec::new(
        &[2, 2, 2, 2],
        &[2, 2, 4, 2],
        &[2, 2, 4, 2],
        0.7,
        AttentionMask::Causal { query_offset: 2 },
    )
    .unwrap();
    let vals = |n, phase: f32| {
        (0..n)
            .map(|i| ((i as f32) * 0.79 + phase).sin())
            .collect::<Vec<_>>()
    };
    let (q, k, v, z, p, up) = (
        vals(16, 0.1),
        vals(32, -0.2),
        vals(32, 1.2),
        vals(16, 0.4),
        vals(32, -1.2),
        vals(16, 0.8),
    );
    for lag in [0, 1, 2, 7] {
        let mut shifted = k.clone();
        for c in 0..4 {
            for j in 0usize..4 {
                let a = 8 * c + 2 * j;
                let b = 8 * c + 2 * j.saturating_sub(lag);
                shifted[a..a + 2].copy_from_slice(&k[b..b + 2]);
            }
        }
        assert_eq!(
            attention_key_lag_reference(spec, &q, &k, &v, Some(&z), Some(&p), lag).unwrap(),
            attention_reference(spec, &q, &shifted, &v, Some(&z), Some(&p)).unwrap()
        );
        let actual =
            attention_key_lag_vjp_reference(spec, &q, &k, &v, Some(&z), Some(&p), &up, lag)
                .unwrap();
        let refg =
            attention_vjp_reference(spec, &q, &shifted, &v, Some(&z), Some(&p), &up).unwrap();
        assert_eq!(actual.query, refg.query);
        assert_eq!(actual.value, refg.value);
        assert_eq!(actual.z_bias, refg.z_bias);
        assert_eq!(actual.pair_bias, refg.pair_bias);
        let mut scatter = vec![0f64; 32];
        for c in 0..4 {
            for j in 0usize..4 {
                for d in 0..2 {
                    scatter[8 * c + 2 * j.saturating_sub(lag) + d] +=
                        f64::from(refg.key[8 * c + 2 * j + d]);
                }
            }
        }
        for (&a, b) in actual.key.iter().zip(scatter) {
            assert!((f64::from(a) - b).abs() < 1e-6);
        }
        if lag == 1 {
            let loss = |at: f32| {
                let mut keys = k.clone();
                keys[0] += at;
                attention_key_lag_reference(spec, &q, &keys, &v, Some(&z), Some(&p), 1)
                    .unwrap()
                    .iter()
                    .zip(&up)
                    .map(|(x, y)| f64::from(*x) * f64::from(*y))
                    .sum::<f64>()
            };
            assert!(((loss(0.005) - loss(-0.005)) / 0.01 - f64::from(actual.key[0])).abs() < 6e-5);
        }
    }
    let noncausal = AttentionSpec::new(
        &[2, 2, 2, 2],
        &[2, 2, 4, 2],
        &[2, 2, 4, 2],
        0.7,
        AttentionMask::None,
    )
    .unwrap();
    assert_eq!(
        attention_key_lag_reference(noncausal, &q, &k, &v, None, None, 1).unwrap_err(),
        AttentionError::KeyLagMask
    );
    assert_eq!(
        attention_key_lag_vjp_reference(noncausal, &q, &k, &v, None, None, &up, 1).unwrap_err(),
        AttentionError::KeyLagMask
    );
}

#[test]
fn a_constructed_content_query_reads_the_successor_value_not_a_future_token() {
    let spec = AttentionSpec::new(
        &[1, 1, 8, 2],
        &[1, 1, 8, 2],
        &[1, 1, 8, 2],
        8.,
        AttentionMask::Causal { query_offset: 0 },
    )
    .unwrap();
    for first in 0..2 {
        for idle in 0..2 {
            for query in 0..2 {
                let mut q = [0.; 16];
                let mut k = [0.; 16];
                let mut v = [0.; 16];
                k[2 + first] = 1.;
                k[8 + 1 - first] = 1.;
                k[14 + query] = 1.;
                q[14 + query] = 1.;
                v[4] = if first == idle { 1. } else { -1. };
                v[10] = -v[4];
                let right = if query == idle { 1. } else { -1. };
                for lag in [0, 1, 2] {
                    let out =
                        attention_key_lag_reference(spec, &q, &k, &v, None, None, lag).unwrap();
                    if lag == 1 {
                        assert!(right * out[14] > 0.99);
                    } else {
                        assert!(out[14].abs() < 1e-6);
                    }
                    let mut future = v;
                    future[10] = 970.;
                    let changed =
                        attention_key_lag_reference(spec, &q, &k, &future, None, None, lag)
                            .unwrap();
                    assert_eq!(&out[..10], &changed[..10]);
                }
            }
        }
    }
}
