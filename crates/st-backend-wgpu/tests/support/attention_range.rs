//! Analytic range witnesses: an absolute-only tolerance would hide these bugs.
use super::{read, upload, Result};
use serde_json::{json, Value};
use st_backend_wgpu::resident_tensor::{attention::AttentionMask, TensorDevice};

fn record(cases: &mut Vec<Value>, name: String, actual: Vec<f32>, expected: Vec<f64>) {
    let passed = actual.len() == expected.len()
        && actual
            .iter()
            .zip(&expected)
            .all(|(&a, &e)| a.is_finite() && (f64::from(a) - e).abs() <= e.abs() * 4e-6);
    cases.push(json!({"name": name, "actual": actual, "expected": expected, "passed": passed}));
}

pub async fn run(device: &TensorDevice) -> Result<Value> {
    let mut cases = Vec::new();
    // exp(-110) underflows in f32, but its product with a finite value does not.
    for count in [2, 128, 131] {
        for reverse in [false, true] {
            for sign in [-1., 1.] {
                let small = if reverse { count - 1 } else { 0 };
                let mut pair = vec![0.; count];
                pair[small] = -110.;
                let mut values = vec![0.; count];
                values[small] = sign * 1e38;
                let q = upload(device, &[1, 1, 1, 1], &[0.], reverse)?;
                let k = upload(device, &[1, 1, count, 1], &vec![0.; count], reverse)?;
                let v = upload(device, &[1, 1, count, 1], &values, reverse)?;
                let pair = upload(device, &[1, 1, 1, count], &pair, reverse)?;
                let w = (-110f64).exp();
                let expected = f64::from(values[small]) * w / (count as f64 - 1. + w);
                for merged in [false, true] {
                    let result = if merged {
                        q.scaled_dot_attention_merged_heads(
                            &k,
                            &v,
                            1.,
                            AttentionMask::None,
                            None,
                            Some(&pair),
                        )?
                    } else {
                        q.scaled_dot_attention(&k, &v, 1., AttentionMask::None, None, Some(&pair))?
                    };
                    record(
                        &mut cases,
                        format!("forward_underflow_{count}_{reverse}_{sign}_{merged}"),
                        read(&result).await?,
                        vec![expected],
                    );
                }
            }
        }
    }
    // A common value offset must not erase the derivative of the dominant key.
    // Evaluate the two-key analytic expression, not 1 - a rounded softmax mean.
    for gap in [85., 120.] {
        for dominant in [0, 1] {
            let mut pair = vec![-gap; 2];
            pair[dominant] = 0.;
            let mut v = vec![0.; 2];
            v[dominant] = 1.;
            let u = 2f32.powi(120);
            let q = device.upload(&[1, 1, 1, 1], &[1.])?;
            let k = device.upload(&[1, 1, 2, 1], &[0., 0.])?;
            let v = device.upload(&[1, 1, 2, 1], &v)?;
            let p = device.upload(&[1, 1, 1, 2], &pair)?;
            let z = device.upload(&[1, 1, 2], &[0., 0.])?;
            let grad = q.scaled_dot_attention_vjp(
                &k,
                &v,
                &device.upload(&[1, 1, 1, 1], &[u])?,
                1.,
                AttentionMask::None,
                Some(&z),
                Some(&p),
            )?;
            let w = (-f64::from(gap)).exp();
            let a = f64::from(u) * w / (1. + w).powi(2);
            let mut expected = vec![-a; 2];
            expected[dominant] = a;
            for (name, tensor) in [
                ("key", grad.key),
                ("pair", grad.pair_bias.unwrap()),
                ("z", grad.z_bias.unwrap()),
            ] {
                record(
                    &mut cases,
                    format!("shifted_center_{gap}_{dominant}_{name}"),
                    read(&tensor).await?,
                    expected.clone(),
                );
            }
        }
    }
    // Summation must preserve a small residual between cancelling large terms.
    for width in [1, 65, 256] {
        for sign in [-1., 1.] {
            let u: Vec<_> = [2f32.powi(121), 1., -2f32.powi(121)]
                .into_iter()
                .flat_map(|x| std::iter::repeat_n(sign * x, width))
                .collect();
            let q = device.upload(&[1, 1, 3, width], &vec![0.; 3 * width])?;
            let kv = device.upload(&[1, 1, 1, width], &vec![0.; width])?;
            let grad = q.scaled_dot_attention_vjp(
                &kv,
                &kv,
                &device.upload(&[1, 1, 3, width], &u)?,
                1.,
                AttentionMask::None,
                None,
                None,
            )?;
            record(
                &mut cases,
                format!("value_cancellation_{width}_{sign}"),
                read(&grad.value).await?,
                vec![f64::from(sign); width],
            );
        }
    }
    // Shifting must not discard a small V between two cancelling large values.
    for sign in [-1., 1.] {
        let q = device.upload(&[1, 1, 1, 1], &[0.])?;
        let k = device.upload(&[1, 1, 3, 1], &[0.; 3])?;
        let v = device.upload(&[1, 1, 3, 1], &[2f32.powi(121), -2f32.powi(121), sign])?;
        let p = device.upload(&[1, 1, 1, 3], &[0.; 3])?;
        let u = device.upload(&[1, 1, 1, 1], &[1.])?;
        let output = q.scaled_dot_attention(&k, &v, 1., AttentionMask::None, None, Some(&p))?;
        record(
            &mut cases,
            format!("forward_cancellation_{sign}"),
            read(&output).await?,
            vec![f64::from(sign) / 3.],
        );
        let grad =
            q.scaled_dot_attention_vjp(&k, &v, &u, 1., AttentionMask::None, None, Some(&p))?;
        record(
            &mut cases,
            format!("shifted_value_tail_{sign}"),
            vec![read(&grad.pair_bias.unwrap()).await?[2]],
            vec![2. * f64::from(sign) / 9.],
        );
    }
    let passed = cases.iter().all(|case| case["passed"] == true);
    Ok(json!({"schema": "spiraltorch.attention_range.v1", "passed": passed, "cases": cases}))
}
