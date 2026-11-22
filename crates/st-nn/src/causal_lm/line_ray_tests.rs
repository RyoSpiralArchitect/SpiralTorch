use super::CausalLmAdaptationMode::{Geometry, Joint, Lora};
use super::*;
use std::collections::BTreeMap;

const D: f64 = 1. / 256.;
fn spec() -> CausalLmLineObjective {
    CausalLmLineObjective::Subset {
        accepted_indices: vec![0, 2],
    }
}
fn grad(m: &NativeCausalLm, mode: CausalLmAdaptationMode) -> NativeCausalLmAdaptationGradients {
    spec()
        .gradients_for(
            m,
            &[1, 3],
            &[vec![2], vec![2, 4], vec![3, 5]],
            0,
            Some(11),
            mode,
        )
        .unwrap()
}

type Blocks = BTreeMap<(String, String), Vec<f32>>;
fn weights(m: &NativeCausalLm) -> Blocks {
    let mut out = Blocks::new();
    if let Some(l) = &m.lora {
        let record: serde_json::Value = serde_json::from_str(&m.lora_to_json().unwrap()).unwrap();
        for key in ["a", "b"] {
            out.insert(
                (l.path(), key.into()),
                serde_json::from_value(record[key].clone()).unwrap(),
            );
        }
    }
    if let Some(g) = m.geometry_artifact() {
        for p in &g.record().placements {
            for t in &p.tensors {
                out.insert((p.path.clone(), t.name.clone()), t.values.clone());
            }
        }
    }
    out
}

#[test]
fn one_global_scale_matches_independent_rounded_weights_and_each_group_budget() {
    for precision in [
        CausalLmReadoutAccumulator::F32,
        CausalLmReadoutAccumulator::F32WideBackward,
        CausalLmReadoutAccumulator::F64,
    ] {
        let mut m = model(1, precision);
        // A nontrainable parameter is still part of the model, but not of the metric.
        let mut record = m.geometry_artifact().unwrap().record().clone();
        let frozen = record.placements[0].tensors[0].name.clone();
        record.placements[0].tensors[0].requires_grad = false;
        m = m
            .with_geometry(GeometryAdapterArtifact::create(record).unwrap())
            .unwrap();
        let all = grad(&m, Joint);
        let before = weights(&m);
        for mode in [Lora, Geometry, Joint] {
            let own = grad(&m, mode);
            let (next, r) = m
                .step_adaptation_gradient_at_distance(&all, mode, D)
                .unwrap();
            let (same, s) = m
                .step_adaptation_gradient_at_distance(&own, mode, D)
                .unwrap();
            assert_eq!(weights(&same), weights(&next));
            assert_eq!(
                serde_json::to_value(&r).unwrap(),
                serde_json::to_value(s).unwrap()
            );
            let norm = own
                .parameters()
                .iter()
                .flat_map(|(p, ps)| {
                    ps.iter()
                        .filter(|(n, _)| p.ends_with("c_proj") || **n != frozen)
                })
                .flat_map(|(_, xs)| xs)
                .map(|&x| f64::from(x).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!((norm / r.gradient_l2 - 1.).abs() < 2e-14);
            let after = weights(&next);
            let (mut squared, mut dot, mut err) = ([0.; 2], [0.; 2], 0.);
            let mut counts = [0; 2];
            for ((p, n), xs) in &before {
                let lo = p.ends_with("c_proj");
                let active = if lo {
                    mode != Geometry
                } else {
                    mode != Lora && *n != frozen
                };
                for (i, (&x, &y)) in xs.iter().zip(&after[&(p.clone(), n.clone())]).enumerate() {
                    if !active {
                        assert_eq!(x.to_bits(), y.to_bits());
                        continue;
                    }
                    let k = usize::from(!lo);
                    let g = f64::from(own.parameters()[p][n][i]);
                    let dx = f64::from(y) - f64::from(x);
                    assert_eq!(
                        y.to_bits(),
                        ((f64::from(x) + D * (-g / norm)) as f32).to_bits()
                    );
                    squared[k] += dx * dx;
                    dot[k] += g * dx;
                    counts[k] += 1;
                    err += (dx / D + g / norm).powi(2);
                }
            }
            for (k, report) in [&r.lora, &r.geometry].into_iter().enumerate() {
                assert_eq!(report.is_some(), counts[k] > 0);
                if let Some(group) = report {
                    assert_eq!(group.trainable_values, counts[k]);
                    assert!((group.realized_l2 - squared[k].sqrt()).abs() < 1e-15);
                    assert!((group.gradient_dot_step - dot[k]).abs() < 1e-15);
                }
            }
            assert!(r.gradient_dot_step < 0.);
            assert!((r.gradient_dot_step - dot.iter().sum::<f64>()).abs() < 1e-15);
            assert!((r.relative_vector_error - err.sqrt()).abs() < 1e-14);
            assert!(
                (r.realized_l2 / D - 1.).abs()
                    <= crate::geometry_artifact::MAX_RELATIVE_DELTA_ERROR
            );
            if mode == Geometry {
                let (legacy, old) = m.step_geometry_gradient_at_distance(&own, D).unwrap();
                assert_eq!(legacy.geometry_artifact_id(), next.geometry_artifact_id());
                assert_eq!(
                    (old.gradient_l2, old.realized_l2, old.relative_vector_error),
                    (r.gradient_l2, r.realized_l2, r.relative_vector_error)
                );
            }
        }
    }
}

#[test]
fn zero_norm_rounding_stale_arithmetic_and_missing_groups_never_advance_other_weights() {
    let m = model(1, CausalLmReadoutAccumulator::F32);
    let g = grad(&m, Joint);
    let original = weights(&m);
    for bad in [-1., f64::INFINITY, f64::NAN, f64::MAX, 1e-200] {
        assert!(m
            .step_adaptation_gradient_at_distance(&g, Joint, bad)
            .is_err());
        assert_eq!(weights(&m), original);
    }
    let (same, r) = m
        .step_adaptation_gradient_at_distance(&g, Joint, 0.)
        .unwrap();
    assert_eq!(weights(&same), original);
    assert_eq!(
        (r.realized_l2, r.relative_vector_error, r.gradient_dot_step),
        (0., 0., 0.)
    );
    let (other, _) = m
        .step_adaptation_gradient_at_distance(&g, Geometry, D)
        .unwrap();
    assert!(other
        .step_adaptation_gradient_at_distance(&g, Lora, 0.)
        .is_err());
    assert!(m
        .with_readout_accumulator(CausalLmReadoutAccumulator::F64)
        .step_adaptation_gradient_at_distance(&g, Joint, 0.)
        .is_err());
    assert!(m
        .step_adaptation_gradient_at_distance(&grad(&m, Lora), Joint, 0.)
        .is_err());
    assert!(m
        .step_adaptation_gradient_at_distance(&grad(&m, Geometry), Joint, 0.)
        .is_err());
    let mut record: serde_json::Value = serde_json::from_str(&m.lora_to_json().unwrap()).unwrap();
    for k in ["a", "b"] {
        for v in record[k].as_array_mut().unwrap() {
            *v = json!(0.);
        }
    }
    let zero = m.with_lora_json(&record.to_string()).unwrap();
    let joint = grad(&zero, Joint);
    assert!(zero
        .step_adaptation_gradient_at_distance(&joint, Lora, D)
        .is_err());
    let (_, none) = zero
        .step_adaptation_gradient_at_distance(&joint, Lora, 0.)
        .unwrap();
    assert_eq!(none.gradient_l2, 0.);
    let (n, r) = zero
        .step_adaptation_gradient_at_distance(&joint, Joint, D)
        .unwrap();
    assert_eq!(n.lora_artifact_id(), zero.lora_artifact_id());
    assert_eq!(r.lora.unwrap().realized_l2, 0.);
    assert!(r.geometry.unwrap().realized_l2 > 0.);
}

#[test]
fn criteria_apply_to_the_full_rounded_vector_not_to_each_group_independently() {
    use crate::geometry_artifact::{ParameterRay, RayMotion};
    let n = 1f64.hypot(1e-6);
    let ray = ParameterRay::new(D, n, 2).unwrap();
    let (mut a, mut b) = (0f32, 1024f32);
    let (mut ra, mut rb) = (RayMotion::default(), RayMotion::default());
    ray.apply(&mut a, 1., &mut ra).unwrap();
    ray.apply(&mut b, 1e-6, &mut rb).unwrap();
    assert_eq!(b, 1024.);
    assert!(rb.relative_error() > 0.);
    ray.admit_motion(
        ra.norm().hypot(rb.norm()),
        ra.relative_error().hypot(rb.relative_error()),
    )
    .unwrap();
    let tiny = f64::from(f32::from_bits(1));
    let ray = ParameterRay::new(D, tiny, 1).unwrap();
    let mut x = 0.;
    let mut actual = RayMotion::default();
    ray.apply(&mut x, tiny, &mut actual).unwrap();
    assert_eq!(x, -(D as f32));
    ray.admit_motion(actual.norm(), actual.relative_error())
        .unwrap();
}
