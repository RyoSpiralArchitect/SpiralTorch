use super::*;
use crate::geometry_runtime::tests::artifact;
use serde_json::{json, Value};

fn bound(a: &GeometryAdapterArtifact, width: f32) -> Result<NativeGeometryAdamW> {
    NativeGeometryAdamW::new_with_log_radius_drift(
        a,
        Default::default(),
        Some(GlobalNormClip::new(0.5)?),
        width,
    )
}

fn gradients(path: &str, radius: f32) -> ParameterGradients {
    ParameterGradients::from([(
        path.into(),
        std::collections::BTreeMap::from([
            ("gate".into(), vec![0.1; 3]),
            ("bias".into(), vec![-0.2; 3]),
            ("log_radius".into(), vec![radius]),
        ]),
    )])
}

fn value<'a>(a: &'a GeometryAdapterArtifact, path: &str, name: &str) -> &'a [f32] {
    &a.record()
        .placements
        .iter()
        .find(|p| p.path == path)
        .unwrap()
        .tensors
        .iter()
        .find(|p| p.name == name)
        .unwrap()
        .values
}

#[test]
fn projects_only_radius_in_both_directions_and_keeps_real_adamw_moments() {
    let a = artifact(true, Some(true), 0.5);
    let narrow = 0.02_f32;
    let runtime = NativeGeometryRuntime::new(a.clone());
    let bounded = bound(&a, narrow).unwrap();
    let free = NativeGeometryAdamW::new(
        &a,
        Default::default(),
        Some(GlobalNormClip::new(0.5).unwrap()),
    )
    .unwrap();
    assert!(free.max_log_radius_drift().is_none());
    assert!(free.log_radius_projection_count().is_none());
    for g in [-1., 1.] {
        let step = AdamWStep::new(0.1, 0.01).unwrap();
        let grads = gradients("site", g);
        let (b, bs) = runtime
            .with_gradients_adamw(a.artifact_id(), &grads, &bounded, step)
            .unwrap();
        let (u, us) = runtime
            .with_gradients_adamw(a.artifact_id(), &grads, &free, step)
            .unwrap();
        let initial = f64::from(value(&a, "site", "log_radius")[0]);
        let shift = f64::from(value(b.artifact(), "site", "log_radius")[0]) - initial;
        assert!(shift * f64::from(g) < 0.);
        assert!(shift.abs() <= f64::from(narrow) && f64::from(narrow) - shift.abs() < 2e-8);
        assert_ne!(
            value(b.artifact(), "site", "log_radius"),
            value(u.artifact(), "site", "log_radius")
        );
        for key in ["gate", "bias"] {
            assert_eq!(
                value(b.artifact(), "site", key),
                value(u.artifact(), "site", key)
            );
        }
        assert_eq!(bs.state().slots, us.state().slots);
        assert_eq!(bs.log_radius_projection_count(), Some(1));
        assert_eq!(bounded.log_radius_projection_count(), Some(0));
        let wide = bound(&a, 1.).unwrap();
        let (same, state) = runtime
            .with_gradients_adamw(a.artifact_id(), &grads, &wide, step)
            .unwrap();
        assert_eq!(same.artifact().to_json(), u.artifact().to_json());
        assert_eq!(state.log_radius_projection_count(), Some(0));
    }
}

#[test]
fn resume_uses_initial_per_site_anchors_and_never_projects_absent_or_fixed_sites() {
    let mut initial = artifact(true, Some(true), 0.5).record().clone();
    let mut second = initial.placements[0].clone();
    second.path = "another".into();
    second
        .tensors
        .iter_mut()
        .find(|t| t.name == "log_radius")
        .unwrap()
        .values[0] = -0.45;
    initial.placements.push(second);
    let mut fixed = artifact(true, Some(false), 0.5).record().placements[0].clone();
    fixed.path = "fixed".into();
    initial.placements.push(fixed);
    let a = GeometryAdapterArtifact::create(initial).unwrap();
    let runtime = NativeGeometryRuntime::new(a.clone());
    let state = bound(&a, 0.02).unwrap();
    let g = gradients("site", -1.);
    let step = AdamWStep::new(0.1, 0.).unwrap();
    let (first, one) = runtime
        .with_gradients_adamw(a.artifact_id(), &g, &state, step)
        .unwrap();
    let bytes = one.to_json().unwrap();
    let resumed = NativeGeometryAdamW::from_json(
        bytes.as_bytes(),
        &one.checkpoint_id().unwrap(),
        first.artifact(),
    )
    .unwrap();
    let (second, two) = first
        .with_gradients_adamw(first.artifact().artifact_id(), &g, &one, step)
        .unwrap();
    let (copy, copy_state) = first
        .with_gradients_adamw(first.artifact().artifact_id(), &g, &resumed, step)
        .unwrap();
    assert_eq!(copy.artifact().to_json(), second.artifact().to_json());
    assert_eq!(copy_state.to_json().unwrap(), two.to_json().unwrap());
    assert_eq!(
        value(first.artifact(), "site", "log_radius"),
        value(second.artifact(), "site", "log_radius")
    );
    assert_eq!(two.log_radius_projection_count(), Some(2));
    for path in ["another", "fixed"] {
        assert_eq!(
            value(&a, path, "log_radius"),
            value(second.artifact(), path, "log_radius")
        );
    }
    let (third, three) = second
        .with_gradients_adamw(
            second.artifact().artifact_id(),
            &gradients("another", 1.),
            &two,
            step,
        )
        .unwrap();
    let drift =
        f64::from(value(third.artifact(), "another", "log_radius")[0]) - f64::from(-0.45f32);
    assert!(drift < -0.01999 && drift.abs() <= f64::from(0.02f32));
    assert_eq!(three.log_radius_projection_count(), Some(3));
    let (same, zero) = third
        .with_gradients_adamw(
            third.artifact().artifact_id(),
            &g,
            &three,
            AdamWStep::new(0., 1.).unwrap(),
        )
        .unwrap();
    assert_eq!(same.artifact().to_json(), third.artifact().to_json());
    assert_eq!(zero.to_json().unwrap(), three.to_json().unwrap());
}

#[test]
fn bound_is_opt_in_and_malformed_or_resealed_drift_records_are_rejected() {
    let a = artifact(true, Some(true), 0.5);
    for delta in [0., -0.01, f32::INFINITY, f32::NAN] {
        assert!(bound(&a, delta).is_err());
    }
    assert!(bound(&artifact(true, Some(false), 0.5), 0.02).is_err());
    assert!(bound(&artifact(false, None, 0.5), 0.02).is_err());
    let legacy = NativeGeometryAdamW::new(&a, Default::default(), None).unwrap();
    let legacy_raw = legacy.to_json().unwrap();
    assert!(!legacy_raw.contains("radius_drift"));
    assert_eq!(
        NativeGeometryAdamW::from_json(legacy_raw.as_bytes(), &legacy.checkpoint_id().unwrap(), &a)
            .unwrap()
            .to_json()
            .unwrap(),
        legacy_raw
    );
    let mut null: Value = serde_json::from_str(&legacy_raw).unwrap();
    null["radius_drift"] = Value::Null;
    let rejected = |record: &Value| {
        let bytes = serde_json::to_vec(record).unwrap();
        assert!(NativeGeometryAdamW::from_json(&bytes, &identity(&bytes), &a).is_err());
    };
    rejected(&null);
    let saved: Value = serde_json::from_str(&bound(&a, 0.02).unwrap().to_json().unwrap()).unwrap();
    assert_eq!(saved["schema"], RADIUS_SCHEMA);
    for (k, bad) in [
        ("placement", json!("elsewhere")),
        ("initial_log_radius", json!(99.)),
        ("initial_log_radius", json!(0.19)),
        ("projections", json!(1)),
    ] {
        let mut v = saved.clone();
        v["radius_drift"]["sites"][0][k] = bad;
        rejected(&v);
    }
    let mut old = saved.clone();
    old["schema"] = json!(SCHEMA);
    rejected(&old);
    let mut missing = saved.clone();
    missing.as_object_mut().unwrap().remove("radius_drift");
    rejected(&missing);
    let mut empty = saved.clone();
    empty["radius_drift"]["sites"] = json!([]);
    rejected(&empty);
    let mut duplicate = saved.clone();
    duplicate["radius_drift"]["sites"]
        .as_array_mut()
        .unwrap()
        .push(saved["radius_drift"]["sites"][0].clone());
    rejected(&duplicate);
}
