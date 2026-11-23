use super::*;
use crate::geometry_artifact::PrefixRotation::{self, *};
use serde_json::json;

fn artifact(
    turn: PrefixRotation,
    gate: [f32; 3],
    bias: [f32; 3],
    strength: f64,
) -> GeometryAdapterArtifact {
    GeometryAdapterArtifact::create_json(&serde_json::to_vec(&json!({
        "schema": "spiraltorch.geometry_adapter_artifact.v1", "base_model_identity":format!("sha256:{}", "a".repeat(64)),
        "training":true,"policy":"full_prefix_f32_no_cache.v1",
        "placements":[{"path":"site","training":true,
            "recipe":{"schema":"spiraltorch.prefix_transport_adapter.v1","features":3,"strength":strength,"max_values":24,"rotation":turn},
            "tensors":[{"name":"gate","shape":[3],"role":"parameter","requires_grad":true,"values":gate},
                      {"name":"bias","shape":[3],"role":"parameter","requires_grad":true,"values":bias}]
        }]
    })).unwrap()).unwrap()
}

// Deliberately quadratic: each past row is rotated directly by its lag, not by a scan.
fn direct(x: &[f64], gate: &[f64], bias: &[f64], turn: PrefixRotation, steps: usize) -> Vec<f64> {
    let r = match turn {
        Identity => 0,
        QuarterTurn => 1,
        InverseQuarterTurn => 3,
    };
    let mut y = Vec::new();
    for doc in x.chunks_exact(3 * steps) {
        for t in 0..steps {
            let mut sum = [0.; 3];
            for (i, row) in doc.chunks_exact(3).take(t + 1).enumerate() {
                let (a, b) = (row[0], row[1]);
                let (a, b) = match ((t - i) * r) % 4 {
                    0 => (a, b),
                    1 => (-b, a),
                    2 => (-a, -b),
                    _ => (b, -a),
                };
                sum[0] += a;
                sum[1] += b;
                sum[2] += row[2];
            }
            for j in 0..3 {
                y.push(doc[3 * t + j] + 0.5 * (gate[j] * sum[j] / (t + 1) as f64 + bias[j]));
            }
        }
    }
    y
}

#[test]
fn prefix_transport_oracle_and_all_three_pullbacks_across_two_documents() {
    let x: Vec<f32> = (0..24).map(|i| ((i * 13 % 31) as f32 - 11.) / 8.).collect();
    let dy: Vec<f32> = (0..24).map(|i| ((i * 7 % 13) as f32 - 6.) / 8.).collect();
    let g = [0.25, -0.5, 0.75];
    let b = [-0.125, 0.25, 0.75];
    for turn in [Identity, QuarterTurn, InverseQuarterTurn] {
        let m = NativeGeometryRuntime::new(artifact(turn, g, b, 0.5));
        let saved = m.forward_sequence("site", &x, [2, 4]).unwrap();
        assert_eq!(saved.sequence_shape(), Some([2, 4, 3]));
        let vars: Vec<_> = x
            .iter()
            .chain(&g)
            .chain(&b)
            .map(|v| f64::from(*v))
            .collect();
        let reference = |v: &[f64]| direct(&v[..24], &v[24..27], &v[27..], turn, 4);
        for (a, e) in saved.output().data().iter().zip(reference(&vars)) {
            assert!((f64::from(*a) - e).abs() < 3e-7);
        }
        let v = saved.vjp(&dy).unwrap();
        assert_eq!(saved.vjp_input(&dy).unwrap().data(), v.input.data());
        assert_eq!(saved.vjp_parameters(&dy).unwrap(), v.parameters);
        let loss = |p: &[f64]| {
            reference(p)
                .iter()
                .zip(&dy)
                .map(|(a, b)| a * f64::from(*b))
                .sum::<f64>()
        };
        for (j, val) in v
            .input
            .data()
            .iter()
            .chain(&v.parameters["gate"])
            .chain(&v.parameters["bias"])
            .enumerate()
        {
            let mut plus = vars.clone();
            let mut minus = vars.clone();
            plus[j] += 0.001;
            minus[j] -= 0.001;
            let fd = (loss(&plus) - loss(&minus)) / 0.002;
            assert!(
                (fd - f64::from(*val)).abs() < 2e-6,
                "{turn:?}/{j}: {fd} vs {val}"
            );
        }
        assert!(m.forward("site", &x, 8).is_err());
        assert!(m.sequence_dimensions("site", [1, 9]).is_err());
        assert!(m.sequence_dimensions("site", [usize::MAX, 2]).is_err());
        assert!(m.forward_sequence("site", &x[..21], [2, 4]).is_err());
        assert!(saved.vjp_input(&[f32::NAN; 24]).is_err());
        assert_eq!(
            m.sequence_dependency("site").unwrap(),
            GeometrySequenceDependency::CausalPrefix
        );
    }
}

#[test]
fn prefix_transport_is_causal_per_document_and_identity_is_the_positionless_control() {
    let x = [2., 1., 0., -2., 3., 0., 0., 0., 0.];
    let swapped = [-2., 3., 0., 2., 1., 0., 0., 0., 0.];
    for turn in [Identity, QuarterTurn, InverseQuarterTurn] {
        let r = NativeGeometryRuntime::new(artifact(turn, [1.; 3], [0.; 3], 1.));
        let f = |s: &[f32]| {
            r.forward_sequence("site", s, [1, s.len() / 3])
                .unwrap()
                .output()
                .data()
                .to_vec()
        };
        assert_eq!(&f(&x)[..6], f(&x[..6]));
        assert_eq!(f(&x)[6..9] == f(&swapped)[6..9], turn == Identity);
        let mut s = None;
        for j in 0..3 {
            let (out, next) = r
                .extend_transport(
                    "site",
                    Tensor::from_vec(1, 3, x[3 * j..3 * j + 3].to_vec()).unwrap(),
                    s.as_ref(),
                    j + 1,
                )
                .unwrap();
            assert_eq!(out.data(), &f(&x)[3 * j..3 * j + 3]);
            assert_eq!(next.cache_values(), 6);
            s = Some(next);
        }
        let changed = [9.; 3];
        assert!(r
            .extend_transport(
                "site",
                Tensor::from_vec(1, 3, changed.to_vec()).unwrap(),
                s.as_ref(),
                5
            )
            .is_err());
    }
}

#[test]
fn prefix_transport_artifact_limits_and_zero_strength_agree_on_both_routes() {
    let a = artifact(QuarterTurn, [f32::MAX; 3], [0.; 3], 0.);
    let mut v: serde_json::Value = serde_json::from_str(a.to_json()).unwrap();
    for (key, value) in [
        ("rotation", json!("learned")),
        ("max_values", json!(2)),
        ("max_values", json!(1_048_577)),
    ] {
        let mut bad = v.clone();
        bad["placements"][0]["recipe"][key] = value;
        assert!(GeometryAdapterArtifact::create_json(&serde_json::to_vec(&bad).unwrap()).is_err());
    }
    let r = NativeGeometryRuntime::new(a);
    let x = [3.; 6];
    let data = Tensor::from_vec(2, 3, x.to_vec()).unwrap();
    let z = r.forward_sequence("site", &x, [1, 2]).unwrap();
    assert_eq!(z.output().data(), x);
    assert_eq!(z.vjp_input(&x).unwrap().data(), x);
    assert!(z
        .vjp_parameters(&x)
        .unwrap()
        .values()
        .flatten()
        .all(|v| *v == 0.));
    assert_eq!(
        r.extend_transport("site", data.clone(), None, 2)
            .unwrap()
            .0
            .data(),
        x
    );
    v["placements"][0]["recipe"]["strength"] = json!(1.);
    let r = NativeGeometryRuntime::new(
        GeometryAdapterArtifact::create_json(&serde_json::to_vec(&v).unwrap()).unwrap(),
    );
    assert!(r.forward_sequence("site", &x, [1, 2]).is_err());
    assert!(r.extend_transport("site", data, None, 2).is_err());
}

#[test]
fn prefix_transport_parameter_only_does_not_evaluate_overflowing_input_adjoint() {
    let rt = NativeGeometryRuntime::new(artifact(QuarterTurn, [1.; 3], [0.; 3], 1.));
    let b = rt.forward_sequence("site", &[0.; 3], [1, 1]).unwrap();
    assert!(b.vjp(&[f32::MAX; 3]).is_err());
    assert!(b.vjp_input(&[f32::MAX; 3]).is_err());
    assert_eq!(
        b.vjp_parameters(&[f32::MAX; 3]).unwrap()["bias"],
        [f32::MAX; 3]
    );
}
