use super::*;
use crate::causal_lm::tests::{artifact, load};
use crate::causal_lm::CausalLmPairWeighting::{CommonDescent, Equal};

const SITE: &str = "transformer.h.1.attn";

#[test]
fn common_geometry_updates_group_means_with_unequal_members_and_actual_dots() {
    let m = model();
    let a = gradient(&m, [3., 4., 0., 0.], 1e20);
    let b = gradient(&m, [-1., -4., 0., 0.], -1e20);
    let c = gradient(&m, [0., 2., 0., 0.], 1e20);
    let d = gradient(&m, [0., 0., 4., 0.], -1e20);
    let (end, r) = m
        .step_common_geometry_sgd(&[&[&a, &b], &[&c], &[&d]], 0.125, Some(0.01))
        .unwrap();
    assert_eq!(
        (r.trainable_values, r.group_sizes.as_slice()),
        (4, [2, 1, 1].as_slice())
    );
    assert_eq!(r.group_gradient_l2, [1., 2., 4.]);
    assert!(r
        .unit_group_weights
        .iter()
        .all(|w| (*w - 1. / 3.).abs() < 1e-12));
    for (x, v) in r.raw_group_weights.iter().zip([4. / 7., 2. / 7., 1. / 7.]) {
        assert!((*x - v).abs() < 1e-12);
    }
    assert!(
        (r.predicted_min_unit_margin - 1. / 3f64.sqrt()).abs() < 1e-12
            && r.applied_min_unit_margin > 0.57
    );
    let val = |m: &NativeCausalLm| {
        m.geometry_artifact().unwrap().record().placements[0]
            .tensors
            .iter()
            .find(|t| t.name == "gate")
            .unwrap()
            .values
            .clone()
    };
    for (i, (&x, &y)) in val(&m).iter().zip(&val(&end)).take(3).enumerate() {
        assert_eq!(
            r.group_loss_dots[i],
            r.group_gradient_l2[i] * (f64::from(y) - f64::from(x))
        );
    }
    assert_eq!(m.lora_to_json().unwrap(), end.lora_to_json().unwrap());
    assert!((r.applied_l2 / 0.01 - 1.).abs() < 1e-4);
}

#[test]
fn common_geometry_refuses_zero_stale_joint_invisible_and_rounded_non_descent() {
    let m = model();
    let id = m.geometry_artifact_id().unwrap().to_owned();
    let a = gradient(&m, [1., 0., 0., 0.], 0.);
    let b = gradient(&m, [-1., 0., 0., 0.], 0.);
    let (other, _) = m.step_geometry_cohort_sgd(&[&a], 0.2, None).unwrap();
    let stale = gradient(&other, [1.; 4], 0.);
    let joint = m.adaptation_loss_gradients(&[1, 2, 3], [1, 3]).unwrap();
    assert!(m
        .step_common_geometry_sgd(&[&[&a], &[&b]], 0.2, None)
        .is_err());
    assert!(m
        .step_common_geometry_sgd(&[&[&a, &b], &[&a]], 0.2, None)
        .is_err());
    for bad in [&stale, &joint] {
        assert!(m
            .step_common_geometry_sgd(&[&[&a], &[bad]], 0.2, None)
            .is_err());
    }
    assert!(m
        .step_common_geometry_sgd(&[&[&a], &[&a]], 0.1, Some(0.))
        .is_err());
    assert!(m
        .step_common_geometry_sgd(&[&[&a], &[&a]], 1e-12, None)
        .is_err());
    // The x update disappears at 0.75, but y moves from zero: this is a
    // representable nonzero proposal that actually hurts the first group.
    let a = gradient(&m, [1., -0.5, 0., 0.], 0.);
    let b = gradient(&m, [-0.5, 1., 0., 0.], 0.);
    let (_, rounded) = m.step_geometry_cohort_sgd(&[&a, &b], 1e-8, None).unwrap();
    assert!(
        rounded.applied_l2 > 0.
            && rounded.member_loss_dots[0] > 0.
            && rounded.member_loss_dots[1] < 0.
    );
    assert!(m
        .step_common_geometry_sgd(&[&[&a], &[&b]], 1e-8, None)
        .is_err());
    assert_eq!(m.geometry_artifact_id(), Some(id.as_str()));
}

#[test]
fn guard_repairs_nine_objectives_and_preserves_an_already_safe_candidate() {
    let m = model();
    let a = gradient(&m, [1., 0., 0., 0.], 99.);
    let b = gradient(&m, [-1., 2., 0., 0.], 88.);
    let (proposal, _) = m.step_geometry_cohort_sgd(&[&a], 0.125, None).unwrap();
    let repeated = [&a, &a];
    let tail = [&b];
    let mut groups = vec![repeated.as_slice(); 8];
    groups.push(&tail);
    let (end, r) = m.guard_geometry_step(&proposal, &groups, 1e-5).unwrap();
    assert_eq!(r.worst_candidate_group, 8);
    assert!(r.candidate_loss_dots[8] > 0. && r.applied_loss_dots.iter().all(|&d| d < 0.));
    assert_eq!(r.group_sizes, [2, 2, 2, 2, 2, 2, 2, 2, 1]);
    assert_eq!(r.group_target_tokens[8], b.target_tokens());
    assert_eq!(r.candidate_l2, 0.125);
    assert!((r.applied_l2 / r.candidate_l2 - 1.).abs() < 1e-5);
    let [lo, hi] = r.feasible_mix.unwrap();
    assert_eq!(r.mix, (lo + hi) * 0.5);
    let y: f64 = 1. / 5f64.sqrt();
    assert!(
        (lo - (1e-5 + y) / (1. + y)).abs() < 1e-12 && (hi - (1. - 1e-5) / (1. + y)).abs() < 1e-12
    );
    assert!(r.applied_min_cosine > 0.1);
    let (keep, k) = m.guard_geometry_step(&end, &groups, 1e-5).unwrap();
    assert_eq!(keep.geometry_artifact_id(), end.geometry_artifact_id());
    assert_eq!(k.feasible_mix, None);
    assert_eq!(k.mix, 0.);
    assert_eq!(k.applied_loss_dots, r.applied_loss_dots);
    assert_eq!(keep.lora_to_json().unwrap(), m.lora_to_json().unwrap());
    assert_ne!(proposal.geometry_artifact_id(), end.geometry_artifact_id());
}

#[test]
fn guard_refuses_unusable_segments_stale_handles_and_unrepresentable_steps() {
    let m = model();
    let a = gradient(&m, [1., 0., 0., 0.], 0.);
    let y = gradient(&m, [0., 1., 0., 0.], 0.);
    let no_y = gradient(&m, [0., -1., 1., 0.], 0.);
    let zero = gradient(&m, [0.; 4], 1.);
    let (candidate, _) = m.step_geometry_cohort_sgd(&[&a], 0.1, None).unwrap();
    // A direction +y+2z in descent coordinates could work, but not this segment toward y.
    assert!(m
        .guard_geometry_step(&candidate, &[&[&y], &[&no_y]], 1e-5)
        .is_err());
    let stale = gradient(&candidate, [1.; 4], 0.);
    for bad in [&zero, &stale] {
        assert!(m.guard_geometry_step(&candidate, &[&[bad]], 1e-5).is_err());
    }
    assert!(m.guard_geometry_step(&m, &[&[&a]], 1e-5).is_err());
    for bad in [0., 1., f64::NAN, f64::INFINITY] {
        assert!(m.guard_geometry_step(&candidate, &[&[&a]], bad).is_err());
    }
    assert!(m.guard_geometry_step(&candidate, &[&[]], 1e-5).is_err());
    assert!(m
        .validate_geometry_step_guard(&candidate, &[256, 1], 1e-5)
        .is_err());
    let mut foreign = candidate.clone();
    foreign.lora = None;
    assert!(m.guard_geometry_step(&foreign, &[&[&a]], 1e-5).is_err());
    let mut r = candidate.geometry_artifact().unwrap().record().clone();
    r.placements[0]
        .tensors
        .iter_mut()
        .find(|t| t.name == "bias")
        .unwrap()
        .values[0] = 0.25;
    let foreign = m
        .with_geometry(GeometryAdapterArtifact::create(r).unwrap())
        .unwrap();
    assert!(m.guard_geometry_step(&foreign, &[&[&a]], 1e-5).is_err());
    // y can move by 1e-10 from zero, but the required x correction disappears at 0.75.
    let (tiny, _) = m.step_geometry_cohort_sgd(&[&y], 1e-10, None).unwrap();
    assert!(m.guard_geometry_step(&tiny, &[&[&a], &[&y]], 1e-5).is_err());
}

#[test]
fn primary_steering_uses_the_open_guard_segment_and_certifies_rounded_parameters() {
    let m = model();
    let x = gradient(&m, [1., 0., 0., 0.], 0.);
    let y = gradient(&m, [0., 1., 0., 0.], 0.);
    let p = gradient(&m, [3., -4., 0., 0.], 9.);
    let q = gradient(&m, [-1., 2., 0., 0.], -9.);
    let (anchor, _) = m.step_geometry_cohort_sgd(&[&x, &y], 0.125, None).unwrap();
    let gx = [&x];
    let gy = [&y, &y];
    let mut protected = vec![gx.as_slice(); 7];
    protected.extend([gy.as_slice(), gy.as_slice()]);
    let floor = 0.1;
    let (end, r) = m
        .steer_geometry_step(&anchor, &[&p, &q], &protected, floor)
        .unwrap();
    assert_eq!(r.limiting_protected_group, Some(7));
    assert_eq!(r.group_sizes, [2, 1, 1, 1, 1, 1, 1, 1, 2, 2]);
    assert_eq!(
        r.group_mean_losses[0],
        (f64::from(p.loss()) + f64::from(q.loss())) / 2.
    );
    assert_eq!(
        r.group_target_tokens[0],
        p.target_tokens() + q.target_tokens()
    );
    let s = 1. / 2f64.sqrt();
    assert!((r.direct_min_protected_cosine + s).abs() < 1e-12);
    assert!((r.mix_limit - (s - floor) / (2. * s)).abs() < 1e-12);
    assert_eq!(r.mix, r.mix_limit / 2.);
    let coords = |m: &NativeCausalLm| {
        m.geometry_artifact().unwrap().record().placements[0]
            .tensors
            .iter()
            .find(|t| t.name == "gate")
            .unwrap()
            .values
            .clone()
    };
    let before = coords(&m);
    let after = coords(&end);
    let dx = f64::from(after[0]) - f64::from(before[0]);
    let dy = f64::from(after[1]) - f64::from(before[1]);
    assert_eq!(r.applied_loss_dots[0], dx - dy);
    assert_eq!((r.applied_loss_dots[1], r.applied_loss_dots[9]), (dx, dy));
    assert!(r.applied_primary_cosine > r.anchor_primary_cosine && r.applied_loss_dots[0] < 0.);
    assert!(
        r.applied_min_protected_cosine > floor && (r.applied_l2 / r.anchor_l2 - 1.).abs() < 1e-4
    );
    assert_eq!(end.lora_to_json().unwrap(), m.lora_to_json().unwrap());
    assert_eq!(
        Some(r.anchor_geometry_id.as_str()),
        anchor.geometry_artifact_id()
    );
    assert_eq!(
        Some(r.post_geometry_id.as_str()),
        end.geometry_artifact_id()
    );
}

#[test]
fn primary_steering_takes_a_feasible_direct_direction_but_never_fakes_a_gain() {
    let m = model();
    let x = gradient(&m, [1., 0., 0., 0.], 0.);
    let y = gradient(&m, [0., 1., 0., 0.], 0.);
    let primary = gradient(&m, [1., 2., 0., 0.], 0.);
    let opposed = gradient(&m, [-1., 0., 0., 0.], 0.);
    let zero = gradient(&m, [0.; 4], 0.);
    let (a, _) = m.step_geometry_cohort_sgd(&[&x, &y], 0.125, None).unwrap();
    let (end, r) = m
        .steer_geometry_step(&a, &[&primary], &[&[&x], &[&y]], 0.1)
        .unwrap();
    assert_eq!(
        (r.mix, r.mix_limit, r.limiting_protected_group),
        (1., 1., None)
    );
    assert!(r.applied_primary_cosine > 1. - 1e-10 && r.applied_min_protected_cosine > 0.44);
    assert!(m
        .steer_geometry_step(&a, &[&opposed], &[&[&x]], 0.1)
        .is_err());
    let (axis, _) = m.step_geometry_cohort_sgd(&[&x], 0.125, None).unwrap();
    assert!(m.steer_geometry_step(&axis, &[&x], &[&[&x]], 0.1).is_err());
    assert!(m
        .steer_geometry_step(&axis, &[&primary], &[&[&y]], 0.1)
        .is_err());
    let foreign = gradient(&end, [1.; 4], 0.);
    let joint = m.adaptation_loss_gradients(&[1, 2, 3], [1, 3]).unwrap();
    for bad in [&zero, &foreign, &joint] {
        assert!(m.steer_geometry_step(&a, &[bad], &[&[&x]], 0.1).is_err());
        assert!(m
            .steer_geometry_step(&a, &[&primary], &[&[bad]], 0.1)
            .is_err());
    }
    let (tiny, _) = m.step_geometry_cohort_sgd(&[&y], 1e-10, None).unwrap();
    assert!(m.steer_geometry_step(&tiny, &[&x], &[&[&y]], 0.1).is_err());
    for sizes in [&[1][..], &[0, 1], &[1, 0], &[usize::MAX, 1], &[128, 129]] {
        assert!(m.validate_geometry_steering(&a, sizes, 0.1).is_err());
    }
    for floor in [0., 1., f64::NAN] {
        assert!(m.validate_geometry_steering(&a, &[1, 1], floor).is_err());
    }
    let mut other = a.clone();
    other.lora = None;
    assert!(m
        .steer_geometry_step(&other, &[&primary], &[&[&x]], 0.1)
        .is_err());
}

#[test]
fn cohort_mean_retains_small_votes_and_ignores_frozen_coordinates_and_target_counts() {
    let m = model();
    let mut a = gradient(&m, [1e20, 1., 2., 0.], 1e20);
    let mut b = gradient(&m, [1., 1., 2., -1.], -1e20);
    let mut c = gradient(&m, [-1e20, 1., 2., 1.], 0.);
    a.target_tokens = 1;
    b.target_tokens = 20;
    c.target_tokens = 3;
    a.loss = 2.;
    b.loss = 7.;
    c.loss = 3.;
    let ideal = gradient(&m, [1. / 3., 1., 2., 0.], 0.);
    let expected = m
        .step_adaptation_sgd(&ideal, CausalLmAdaptationMode::Geometry, 0.25)
        .unwrap();
    let (out, r) = m
        .step_geometry_cohort_sgd(&[&a, &b, &c], 0.25, None)
        .unwrap();
    assert_eq!(expected.geometry_artifact_id(), out.geometry_artifact_id());
    assert_eq!(
        (
            r.members,
            r.trainable_values,
            r.total_target_tokens,
            r.mean_input_loss
        ),
        (3, 4, 24, 4.)
    );
    assert_eq!(
        r.mean_gradient_l2,
        (f64::from(1f32 / 3.).powi(2) + 5.).sqrt()
    );
    assert_eq!(r.member_loss_dots.len(), 3);
    assert_eq!(out.lora_to_json().unwrap(), m.lora_to_json().unwrap());
    let (permuted, back) = m
        .step_geometry_cohort_sgd(&[&c, &a, &b], 0.25, None)
        .unwrap();
    assert_eq!(permuted.geometry_artifact_id(), out.geometry_artifact_id());
    assert_eq!(
        back.member_loss_dots,
        [
            r.member_loss_dots[2],
            r.member_loss_dots[0],
            r.member_loss_dots[1]
        ]
    );
}

#[test]
fn cohort_singleton_and_pair_match_existing_sgd_and_rounded_loss_dots() {
    let m = model();
    let a = gradient(&m, [0.5, 2., -1., 0.], 1e10);
    let b = gradient(&m, [-1., 0.25, 4., 1.], -1e10);
    let (single, s) = m.step_geometry_cohort_sgd(&[&a], 0.1, None).unwrap();
    assert_eq!(
        single.geometry_artifact_id(),
        m.step_adaptation_sgd(&a, CausalLmAdaptationMode::Geometry, 0.1)
            .unwrap()
            .geometry_artifact_id()
    );
    assert_eq!(s.member_loss_dots.len(), 1);
    for distance in [None, Some(0.), Some(0.005)] {
        let (paired, p) = m
            .step_paired_geometry_sgd(&a, &b, 0.25, Equal, distance)
            .unwrap();
        let (cohort, c) = m
            .step_geometry_cohort_sgd(&[&a, &b], 0.25, distance)
            .unwrap();
        assert_eq!(paired.geometry_artifact_id(), cohort.geometry_artifact_id());
        assert_eq!(c.member_loss_dots, p.applied.loss_dots);
        assert_eq!(c.applied_l2, p.applied.l2);
        assert_eq!(c.distance_match.is_some(), distance.is_some());
    }
}

#[test]
fn cohort_rejects_stale_malformed_or_unrepresentable_inputs_atomically() {
    let m = model();
    let a = gradient(&m, [1.; 4], 0.);
    let zero = gradient(&m, [0.; 4], 0.);
    let id = m.geometry_artifact_id().unwrap().to_owned();
    let (next, _) = m.step_geometry_cohort_sgd(&[&a], 0.1, None).unwrap();
    let stale = gradient(&next, [1.; 4], 0.);
    let joint = m.adaptation_loss_gradients(&[1, 2, 3], [1, 3]).unwrap();
    let mut nan = gradient(&m, [1.; 4], 0.);
    nan.parameters
        .get_mut(SITE)
        .unwrap()
        .get_mut("bias")
        .unwrap()[0] = f32::NAN;
    let mut overflow = gradient(&m, [1.; 4], 0.);
    overflow.target_tokens = usize::MAX;
    let other = m
        .clone()
        .with_readout_accumulator(CausalLmReadoutAccumulator::F64);
    let wrong = gradient(&other, [1.; 4], 0.);
    for bad in [&stale, &joint, &nan, &overflow, &wrong] {
        assert!(m.step_geometry_cohort_sgd(&[&a, bad], 0.1, None).is_err());
    }
    assert!(m.step_geometry_cohort_sgd(&[], 0.1, None).is_err());
    assert!(m
        .step_geometry_cohort_sgd(&vec![&a; MAX_GEOMETRY_COHORT_MEMBERS + 1], 0.1, None)
        .is_err());
    assert!(m
        .step_geometry_cohort_sgd(&[&zero], 0.1, Some(0.01))
        .is_err());
    assert!(m
        .step_geometry_cohort_sgd(&[&a], f32::INFINITY, None)
        .is_err());
    assert!(m
        .step_geometry_cohort_sgd(&[&a], 0.1, Some(f64::NAN))
        .is_err());
    assert_eq!(m.geometry_artifact_id(), Some(id.as_str()));
}

fn model() -> NativeCausalLm {
    let base = load().with_lora(1, 2, 3.).unwrap();
    let mut r = artifact(&base, SITE, 0.75).record().clone();
    r.training = true;
    r.placements[0].training = true;
    for t in &mut r.placements[0].tensors {
        t.requires_grad = t.name == "gate";
    }
    base.with_geometry(GeometryAdapterArtifact::create(r).unwrap())
        .unwrap()
}

fn gradient(
    m: &NativeCausalLm,
    gate: [f32; 4],
    frozen_bias: f32,
) -> NativeCausalLmAdaptationGradients {
    let mut g = m
        .adaptation_loss_gradients_for(&[1, 2, 3], [1, 3], CausalLmAdaptationMode::Geometry)
        .unwrap();
    let placement = g.parameters.get_mut(SITE).unwrap();
    placement.insert("gate".into(), gate.into());
    placement.insert("bias".into(), vec![frozen_bias; 4]);
    g
}

#[test]
fn conflicting_directions_need_not_reweight_a_common_mean() {
    let m = model();
    let a = gradient(&m, [2., 0., 0., 0.], 1e20);
    let b = gradient(&m, [-0.25, 2., 0., 0.], -1e19);
    let mean = gradient(&m, [0.875, 1., 0., 0.], 0.);
    let reference = m
        .step_adaptation_sgd(&mean, CausalLmAdaptationMode::Geometry, 0.25)
        .unwrap();
    for policy in [Equal, CommonDescent] {
        let (out, r) = m
            .step_paired_geometry_sgd(&a, &b, 0.25, policy, None)
            .unwrap();
        assert_eq!(out.geometry_artifact_id(), reference.geometry_artifact_id());
        assert_eq!(out.lora_to_json().unwrap(), m.lora_to_json().unwrap());
        assert_eq!(r.weights, [0.5, 0.5]);
        assert_eq!(r.trainable_values, 4);
        assert!(r.applied.both_descend && !r.reweighted);
        assert_eq!(r.applied.loss_dots, [-0.4375, -0.4453125]);
        assert_eq!(r.gradient_l2, [2., 0.25f64.hypot(2.)]);
    }
}

#[test]
fn balance_is_symmetric_and_measures_the_final_rounded_move_only_on_trainable_coordinates() {
    let m = model();
    let a = gradient(&m, [4., 0., 0., 0.], 1e20);
    let b = gradient(&m, [-1., 1., 0., 0.], -1e19);
    let (_, eq) = m
        .step_paired_geometry_sgd(&a, &b, 0.25, Equal, None)
        .unwrap();
    assert_eq!(eq.equal.loss_dots, [-1.5, 0.25]);
    let target = eq.applied.l2 * 1.5;
    let (out, r) = m
        .step_paired_geometry_sgd(&a, &b, 0.25, CommonDescent, Some(target))
        .unwrap();
    let (swap, s) = m
        .step_paired_geometry_sgd(&b, &a, 0.25, CommonDescent, Some(target))
        .unwrap();
    assert_eq!(out.geometry_artifact_id(), swap.geometry_artifact_id());
    assert!(r.reweighted && !r.equal.both_descend && r.applied.both_descend);
    assert_eq!(r.weights, [s.weights[1], s.weights[0]]);
    assert!((r.weights[0] - 2f64.sqrt() / (4. + 2f64.sqrt())).abs() < 1e-14);
    assert_eq!(
        r.applied.loss_dots,
        [s.applied.loss_dots[1], s.applied.loss_dots[0]]
    );
    let origin = m.geometry_artifact().unwrap();
    let end = out.geometry_artifact().unwrap();
    let mut differences = Vec::new();
    for (x, y) in origin.record().placements[0]
        .tensors
        .iter()
        .zip(&end.record().placements[0].tensors)
    {
        if x.requires_grad {
            differences.extend(
                x.values
                    .iter()
                    .zip(&y.values)
                    .map(|(&a, &b)| f64::from(b) - f64::from(a)),
            );
        } else {
            assert_eq!(x.values, y.values);
        }
    }
    assert_eq!(
        r.applied.loss_dots,
        [4. * differences[0], -differences[0] + differences[1]]
    );
    assert_eq!(r.applied.l2, origin.trainable_delta_l2(end).unwrap());
    assert_eq!(r.applied.l2, r.distance_match.unwrap().realized_l2);
    assert!((r.applied.l2 / target - 1.).abs() < 1e-4);
}

#[test]
fn degenerate_or_unrepresentable_common_steps_fail_without_changing_source() {
    let m = model();
    let before = m.geometry_artifact().unwrap().to_json().to_owned();
    let a = gradient(&m, [4., 0., 0., 0.], 0.);
    let b = gradient(&m, [-1., 0., 0., 0.], 0.);
    let zero = gradient(&m, [0.; 4], 0.);
    for (left, right, rate, target) in [
        (&a, &b, 0.25, None),
        (&a, &zero, 1., None),
        (&zero, &zero, 1., Some(0.1)),
        (&a, &a, 0.25, Some(0.)),
        (&a, &a, 1e-30, None),
    ] {
        assert!(m
            .step_paired_geometry_sgd(left, right, rate, CommonDescent, target)
            .is_err());
    }
    assert!(m
        .step_paired_geometry_sgd(&zero, &zero, 1., Equal, Some(0.1))
        .is_err());
    let (same, r) = m
        .step_paired_geometry_sgd(&a, &a, 1., Equal, Some(0.))
        .unwrap();
    assert_eq!(same.geometry_artifact_id(), m.geometry_artifact_id());
    assert!(!r.applied.both_descend);
    assert_eq!(m.geometry_artifact().unwrap().to_json(), before);
}
