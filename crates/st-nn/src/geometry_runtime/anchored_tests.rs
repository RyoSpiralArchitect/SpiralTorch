use super::*;
use serde_json::json;

fn projection(gate: [f32; 2], bias: [f32; 2], strength: f32) -> GeometryAdapterArtifact {
    let tensors = [("gate", gate), ("bias", bias)].map(|(name, v)| {
        json!({
            "name":name,"shape":[2],"values":v,"role":"parameter","requires_grad":true
        })
    });
    GeometryAdapterArtifact::create_json(&serde_json::to_vec(&json!({
        "schema":"spiraltorch.geometry_adapter_artifact.v1","base_model_identity":format!("sha256:{}","0".repeat(64)),
        "policy":"full_prefix_f32_no_cache.v1","training":true,"placements":[{"path":"site","training":true,
        "recipe":{"schema":"spiraltorch.wave_stage_adapter.v1","features":2,"strength":strength,"max_values":32,
                  "stage":{"kind":"projection","curvature":-1.,"log_radius":0.}},"tensors":tensors}]
    })).unwrap()).unwrap()
}

fn change(
    a: &GeometryAdapterArtifact,
    name: &str,
    index: usize,
    delta: f32,
) -> GeometryAdapterArtifact {
    let mut r = a.record().clone();
    r.placements[0]
        .tensors
        .iter_mut()
        .find(|t| t.name == name)
        .unwrap()
        .values[index] += delta;
    GeometryAdapterArtifact::create(r).unwrap()
}

#[test]
fn anchor_preserves_zero_offset_forward_and_all_training_routes() {
    let original = projection([1.4, -0.7], [0.12, -0.06], 0.75);
    let anchored = original.anchor_wave_projection("site").unwrap();
    assert_ne!(original.artifact_id(), anchored.artifact_id());
    assert!(anchored.anchor_wave_projection("site").is_err());
    assert!(original.anchor_wave_projection("missing").is_err());
    assert!(original.trainable_delta_l2(&anchored).is_err());
    let a = NativeGeometryRuntime::new(original);
    let b = NativeGeometryRuntime::new(anchored.clone());
    let x = [0., 0., 1.9, 1.5, -0.375, 0.7];
    let dy = [0.4, -0.8, -0.2, 0.16, 0.6, 0.25];
    let old = a.forward_sequence("site", &x, [1, 3]).unwrap();
    let new = b.forward_sequence("site", &x, [1, 3]).unwrap();
    let g = old.vjp(&dy).unwrap();
    let h = new.vjp(&dy).unwrap();
    assert_eq!(old.output().data(), new.output().data());
    assert_eq!(g.input.data(), h.input.data());
    assert_eq!(g.parameters, h.parameters);
    assert_eq!(new.vjp_input(&dy).unwrap().data(), h.input.data());
    assert_eq!(new.vjp_parameters(&dy).unwrap(), h.parameters);
    assert_eq!(h.parameters.len(), 2);
    assert!(b.forward("site", &[0.; 34], 17).is_err());
    assert!(new.vjp_input(&[f32::NAN; 6]).is_err());
    let rebuilt = GeometryAdapterArtifact::from_json(
        anchored.to_json().as_bytes(),
        anchored.artifact_id(),
        &anchored.record().base_model_identity,
    )
    .unwrap();
    assert_eq!(rebuilt.to_json(), anchored.to_json());
    let mut bad = anchored.record().clone();
    bad.placements[0]
        .tensors
        .iter_mut()
        .find(|t| t.name == "gate_anchor")
        .unwrap()
        .requires_grad = true;
    assert!(GeometryAdapterArtifact::create(bad).is_err());
    let foreign = change(&anchored, "gate_anchor", 0, 0.5);
    assert!(anchored.trainable_delta_l2(&foreign).is_err());
}

#[test]
fn sub_ulp_gate_offset_affects_residual_forward_and_matches_the_same_radius() {
    let raw = projection([16., -2.], [-16., 0.], 1.);
    let centered = raw.anchor_wave_projection("site").unwrap();
    let small = 2.0f32.powi(-22);
    assert_eq!(16. + small, 16.);
    let radius = f64::from(small);
    assert!(raw
        .match_trainable_delta(&change(&raw, "gate", 0, 1.), radius)
        .is_err());
    let (endpoint, step) = centered
        .match_trainable_delta(&change(&centered, "gate", 0, 1.), radius)
        .unwrap();
    assert_eq!((step.realized_l2, step.relative_vector_error), (radius, 0.));
    let apply = |a| {
        NativeGeometryRuntime::new(a)
            .forward("site", &[1., 0.], 1)
            .unwrap()
            .output()
            .data()
            .to_vec()
    };
    assert_eq!(apply(raw), [1., 0.]);
    assert_eq!(apply(centered), [1., 0.]);
    assert_eq!(apply(endpoint), [1. + small, 0.]);
}

#[test]
fn offset_and_input_vjps_match_finite_differences_away_from_the_anchor() {
    let centered = projection([0.8, -1.2], [0.05, -0.06], 0.6)
        .anchor_wave_projection("site")
        .unwrap();
    let moved = change(&change(&centered, "gate", 0, 0.07), "bias", 1, -0.13);
    let x = [0.25, -0.3, 0.6, 0.17];
    let dy = [-0.7, 0.99, 0.25, -0.38];
    let runtime = NativeGeometryRuntime::new(moved.clone());
    let batch = runtime.forward("site", &x, 2).unwrap();
    let grad = batch.vjp(&dy).unwrap();
    let loss = |a: &GeometryAdapterArtifact, x: &[f32]| {
        NativeGeometryRuntime::new(a.clone())
            .forward("site", x, 2)
            .unwrap()
            .output()
            .data()
            .iter()
            .zip(dy)
            .map(|(&v, g)| f64::from(v) * f64::from(g))
            .sum::<f64>()
    };
    let eps = 0.001f32;
    for k in ["gate", "bias"] {
        for j in 0..2 {
            let numeric = (loss(&change(&moved, k, j, eps), &x)
                - loss(&change(&moved, k, j, -eps), &x))
                / (2. * f64::from(eps));
            assert!((numeric - f64::from(grad.parameters[k][j])).abs() < 5e-5);
        }
    }
    for j in 0..4 {
        let mut plus = x;
        let mut minus = x;
        plus[j] += eps;
        minus[j] -= eps;
        let numeric = (loss(&moved, &plus) - loss(&moved, &minus)) / (2. * f64::from(eps));
        assert!((numeric - f64::from(grad.input.data()[j])).abs() < 5e-5);
    }
    assert_eq!(batch.vjp_input(&dy).unwrap().data(), grad.input.data());
    assert_eq!(batch.vjp_parameters(&dy).unwrap(), grad.parameters);
}
