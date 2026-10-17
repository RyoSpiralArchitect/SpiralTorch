//! Algebraic shift directions, separate from trajectory agreement with float32 Torch.
use super::{compare, read, Result};
use serde_json::{json, Value};
use st_backend_wgpu::resident_tensor::{attention::AttentionMask, TensorDevice};
use st_kernel_contracts::{
    attention::{attention_reference, attention_vjp_reference, AttentionSpec},
    poincare::{PoincareBiasForward, PoincareBiasSpec},
};

fn conservation(label: &str, values: &[f32], expected: f64) -> Value {
    let sum: f64 = values.iter().map(|&v| f64::from(v)).sum();
    // One ULP per output allows its final f32 rounding, plus bounded Wide noise.
    let bound = values
        .iter()
        .map(|v| f64::from(v.abs().next_up()) - f64::from(v.abs()))
        .sum::<f64>()
        + 2e-12;
    json!({"direction": label, "sum": sum, "expected": expected, "bound": bound,
           "passed": values.iter().all(|v| v.is_finite()) && (sum - expected).abs() <= bound})
}

pub(super) async fn run(device: &TensorDevice) -> Result<Value> {
    let mut cases = Vec::new();
    let mut configs = Vec::new();
    for keys in [8, 16, 32, 64, 257] {
        for reverse in [false, true] {
            configs.push((
                format!("ordered_{keys}_{reverse}"),
                1,
                1,
                1,
                keys,
                1,
                None,
                reverse,
            ));
        }
    }
    for offset in [None, Some(0), Some(29)] {
        configs.push((
            format!("batched_bias_{offset:?}"),
            2,
            2,
            3,
            32,
            7,
            offset,
            false,
        ));
    }
    for (name, batch, heads, queries, keys, dim, offset, reverse) in configs {
        let contexts = batch * heads;
        let qs = [batch, heads, queries, dim];
        let ks = [batch, heads, keys, dim];
        let scalar = dim == 1;
        let q: Vec<_> = (0..contexts * queries * dim)
            .map(|i| {
                if scalar {
                    1.0
                } else {
                    ((i * 7 % 31) as f32 - 15.0) / 32.0
                }
            })
            .collect();
        let k: Vec<_> = (0..contexts * keys * dim)
            .map(|i| {
                if scalar {
                    (if reverse { keys - 1 - i } else { i }) as f32 / 32.0
                } else {
                    ((i * 11 % 37) as f32 - 18.0) / 32.0
                }
            })
            .collect();
        let v: Vec<_> = (0..k.len())
            .map(|i| {
                if scalar {
                    (if reverse { keys - 1 - i } else { i }) as f32 / 8.0
                } else {
                    ((i * 5 % 29) as f32 - 14.0) / 8.0
                }
            })
            .collect();
        let u: Vec<_> = (0..q.len())
            .map(|i| {
                if scalar {
                    1.0
                } else {
                    ((i * 3 % 17) as f32 - 8.0) / 16.0
                }
            })
            .collect();
        let z: Vec<_> = (0..contexts * keys)
            .map(|i| {
                if scalar {
                    0.0
                } else {
                    ((i * 3 % 17) as f32 - 8.0) / 16.0
                }
            })
            .collect();
        let geometry = if !scalar && offset.is_some() {
            let coordinates: Vec<_> = (0..batch * keys * 2)
                .map(|i| ((i * 7 % 23) as f32 - 11.0) / 64.0)
                .collect();
            Some(PoincareBiasForward::new(
                PoincareBiasSpec::new([batch, keys, 2], heads, -0.75)?,
                &coordinates,
                &[0.0, 0.5],
            )?)
        } else {
            None
        };
        let mut pair = vec![0.0; contexts * queries * keys];
        if let Some(geometry) = &geometry {
            for context in 0..contexts {
                for query in 0..queries {
                    let source = (context * keys + offset.unwrap() + query) * keys;
                    let target = (context * queries + query) * keys;
                    pair[target..target + keys]
                        .copy_from_slice(&geometry.scores()[source..source + keys]);
                }
            }
        } else if !scalar {
            for (i, bias) in pair.iter_mut().enumerate() {
                *bias = -((i % 7) as f32) / 16.0;
            }
        }
        let mask = offset.map_or(AttentionMask::None, |query_offset| AttentionMask::Causal {
            query_offset,
        });
        let scale = if scalar { 1.0 } else { 0.375 };
        let spec = AttentionSpec::new(&qs, &ks, &ks, scale, mask)?;
        let qg = device.upload(&qs, &q)?;
        let kg = device.upload(&ks, &k)?;
        let vg = device.upload(&ks, &v)?;
        let ug = device.upload(&qs, &u)?;
        let zg = device.upload(&spec.z_bias_shape(), &z)?;
        let pg = device.upload(&spec.pair_bias_shape(), &pair)?;
        let gradients =
            qg.scaled_dot_attention_vjp(&kg, &vg, &ug, scale, mask, Some(&zg), Some(&pg))?;
        let actual = [
            read(&gradients.query).await?,
            read(&gradients.key).await?,
            read(&gradients.value).await?,
            read(gradients.z_bias.as_ref().unwrap()).await?,
            read(gradients.pair_bias.as_ref().unwrap()).await?,
        ];
        let expected = attention_vjp_reference(spec, &q, &k, &v, Some(&z), Some(&pair), &u)?;
        let expected_pair = expected.pair_bias.as_ref().unwrap().clone();
        let mut errors = serde_json::Map::new();
        for ((label, expected), actual) in [
            ("query", expected.query),
            ("key", expected.key),
            ("value", expected.value),
            ("z_bias", expected.z_bias.unwrap()),
            ("pair_bias", expected.pair_bias.unwrap()),
        ]
        .into_iter()
        .zip(&actual)
        {
            errors.insert(label.into(), json!(compare(label, actual, &expected)?));
        }
        let output =
            read(&qg.scaled_dot_attention(&kg, &vg, scale, mask, Some(&zg), Some(&pg))?).await?;
        errors.insert(
            "forward".into(),
            json!(compare(
                "forward",
                &output,
                &attention_reference(spec, &q, &k, &v, Some(&z), Some(&pair))?
            )?),
        );
        let shifted: Vec<_> = k
            .iter()
            .enumerate()
            .map(|(i, value)| {
                value
                    + if scalar {
                        0.25
                    } else {
                        ((i % dim) as f32 - 3.0) / 8.0
                    }
            })
            .collect();
        let shifted = device.upload(&ks, &shifted)?;
        let shifted_output =
            read(&qg.scaled_dot_attention(&shifted, &vg, scale, mask, Some(&zg), Some(&pg))?)
                .await?;
        let shifted_gradients =
            qg.scaled_dot_attention_vjp(&shifted, &vg, &ug, scale, mask, Some(&zg), Some(&pg))?;
        let shift_forward = compare("key shift forward", &shifted_output, &output)?;
        let mut shift_errors = serde_json::Map::new();
        for ((label, tensor), expected) in [
            ("query", shifted_gradients.query),
            ("key", shifted_gradients.key),
            ("value", shifted_gradients.value),
            ("z_bias", shifted_gradients.z_bias.unwrap()),
            ("pair_bias", shifted_gradients.pair_bias.unwrap()),
        ]
        .into_iter()
        .zip(&actual)
        {
            shift_errors.insert(
                label.into(),
                json!(compare(label, &read(&tensor).await?, expected)?),
            );
        }
        let mut directions = Vec::new();
        for context in 0..contexts {
            for d in 0..dim {
                let rows = |values: &[f32]| {
                    (0..keys)
                        .map(|k| values[(context * keys + k) * dim + d])
                        .collect::<Vec<_>>()
                };
                directions.push(conservation("key", &rows(&actual[1]), 0.0));
                let expected: f64 = (0..queries)
                    .map(|q| f64::from(u[(context * queries + q) * dim + d]))
                    .sum();
                directions.push(conservation("value", &rows(&actual[2]), expected));
            }
            directions.push(conservation(
                "z_bias",
                &actual[3][context * keys..(context + 1) * keys],
                0.0,
            ));
            for query in 0..queries {
                let start = (context * queries + query) * keys;
                directions.push(conservation(
                    "pair_bias",
                    &actual[4][start..start + keys],
                    0.0,
                ));
            }
        }
        if let Some(geometry) = &geometry {
            let mut seed = vec![0.0; contexts * keys * keys];
            let mut reference_seed = seed.clone();
            for context in 0..contexts {
                for query in 0..queries {
                    let target = (context * keys + offset.unwrap() + query) * keys;
                    let source = (context * queries + query) * keys;
                    seed[target..target + keys].copy_from_slice(&actual[4][source..source + keys]);
                    reference_seed[target..target + keys]
                        .copy_from_slice(&expected_pair[source..source + keys]);
                }
            }
            let pullback = geometry.backward(&seed)?;
            let reference = geometry.backward(&reference_seed)?;
            for (label, actual, expected) in [
                (
                    "poincare_coordinates",
                    &pullback.coordinates,
                    &reference.coordinates,
                ),
                ("poincare_raw_gain", &pullback.raw_gain, &reference.raw_gain),
            ] {
                if !actual.iter().any(|&v| v != 0.0) || !expected.iter().any(|&v| v != 0.0) {
                    return Err("disconnected geometry pullback probe".into());
                }
                errors.insert(label.into(), json!(compare(label, actual, expected)?));
            }
        }
        let passed = directions.iter().all(|d| d["passed"] == true);
        cases.push(json!({"name": name, "passed": passed, "query_shape": qs, "key_shape": ks,
            "query_offset": offset, "poincare_pullback": geometry.is_some(), "max_abs_reference_error": errors,
            "key_shift_forward_error": shift_forward, "key_shift_gradient_errors": shift_errors,
            "directions": directions}));
    }
    Ok(
        json!({"passed": cases.iter().all(|c| c["passed"] == true), "cases": cases,
        "oracle": "algebraic shift conservation and shared f64-accumulating Rust VJP",
        "scope": "no gradient masking, no full-model or performance qualification"}),
    )
}
