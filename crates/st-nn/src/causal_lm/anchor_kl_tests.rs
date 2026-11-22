use super::*;
use CausalLmAdaptationMode::{Geometry, Joint, Lora};

const ANCHORS: &[&[usize]] = &[&[1, 3], &[2, 4, 5]];

#[path = "anchor_envelope_tests.rs"]
mod envelope;

#[path = "anchor_secant_tests.rs"]
mod secant;

#[path = "anchor_midpoint_tests.rs"]
mod midpoint;

fn independent(a: &NativeCausalLm, b: &NativeCausalLm) -> Vec<f64> {
    let logs = |m: &NativeCausalLm, p| {
        let row = m.next_token_logits(p).unwrap();
        let raw: Vec<f64> = row.data().iter().map(|&v| f64::from(v)).collect();
        let z = raw.iter().map(|v| v.exp()).sum::<f64>().ln();
        raw.into_iter().map(|x| x - z).collect::<Vec<_>>()
    };
    ANCHORS
        .iter()
        .map(|&p| {
            logs(a, p)
                .into_iter()
                .zip(logs(b, p))
                .map(|(x, y)| x.exp() * (x - y))
                .sum()
        })
        .collect()
}

#[test]
fn matched_candidates_are_measured_against_one_origin_and_one_sided_target() {
    let m = model(1, CausalLmReadoutAccumulator::F32WideBackward);
    let g = m.adaptation_loss_gradients(&[1, 3, 5], [1, 3]).unwrap();
    for mode in [Lora, Geometry, Joint] {
        let (known, _) = m
            .step_adaptation_gradient_at_distance(&g, mode, 1. / 64.)
            .unwrap();
        let expected = independent(&m, &known);
        let target = expected.iter().sum::<f64>() / 2. * 1.01;
        let policy = CausalLmAnchorKlPolicy {
            target_kl: target,
            max_parameter_l2: 1. / 16.,
        };
        let (end, ray, r) = m
            .step_adaptation_at_anchor_kl(&g, mode, ANCHORS, policy)
            .unwrap();
        let actual = independent(&m, &end);
        assert!(r.trials.len() > 1 && r.trials.len() <= CausalLmAnchorKlPolicy::MAX_TRIALS);
        assert_eq!(r.trials[0].requested_l2, policy.max_parameter_l2);
        assert_eq!(r.trials.last().unwrap().requested_l2, ray.requested_l2);
        assert_eq!(r.anchor_readouts, (r.trials.len() + 1) * ANCHORS.len());
        assert!(r.mean_kl <= target && r.mean_kl / target >= 0.98);
        for (a, b) in actual.iter().zip(&r.per_anchor_kl) {
            assert!((a - b).abs() < 1e-14);
        }
        for t in r.trials {
            let (c, _) = m
                .step_adaptation_gradient_at_distance(&g, mode, t.requested_l2)
                .unwrap();
            let reference = independent(&m, &c).iter().sum::<f64>() / 2.;
            assert!((t.mean_kl - reference).abs() < 1e-14);
        }
        assert_eq!(
            m.geometry_artifact_id() == end.geometry_artifact_id(),
            mode == Lora
        );
        assert_eq!(
            m.lora_artifact_id() == end.lora_artifact_id(),
            mode == Geometry
        );
    }
}

#[test]
fn plan_and_anchor_failures_are_atomic_and_resume_binds_the_anchor_selection() {
    type S = NativeCausalLmLineTrainingSession;
    let m = model(1, CausalLmReadoutAccumulator::F32);
    let mut p = json!({"schema":"spiraltorch.native_causal_lm_line_plan.v3","mode":"joint","epochs":2,
        "examples":[{"prompt_ids":[1,3],"candidates":[[1],[2]],"stop_id":0,"objective":{"kind":"set"}},
                    {"prompt_ids":[2,4,5],"candidates":[[1],[2]],"stop_id":0,"objective":{"kind":"set"}}],
        "update":{"kind":"anchor_forward_kl","target_kl":1e-5,"max_parameter_l2":0.125,"anchor_examples":[0,1]}});
    let start = |p: &serde_json::Value| S::new(m.clone(), p.to_string().as_bytes());
    for (key, value) in [
        ("target_kl", json!(0)),
        ("max_parameter_l2", json!(1e100)),
        ("anchor_examples", json!([])),
        ("anchor_examples", json!([1, 1])),
        ("anchor_examples", json!([2])),
        ("anchor_examples", json!(vec![0; 17])),
    ] {
        let mut invalid = p.clone();
        invalid["update"][key] = value;
        decoder::work_counts::take();
        assert!(start(&invalid).is_err());
        assert_eq!(decoder::work_counts::take(), [0; 3]);
    }
    let mut duplicate = p.clone();
    duplicate["examples"][1]["prompt_ids"] = json!([1, 3]);
    assert!(start(&duplicate).is_err());
    // Construct a feasible target for this test's (different) tiny decoder.
    let g = CausalLmLineObjective::Set {}
        .gradients_for(&m, ANCHORS[0], &[vec![1], vec![2]], 0, None, Joint)
        .unwrap();
    let (c, _) = m
        .step_adaptation_gradient_at_distance(&g, Joint, 1. / 32.)
        .unwrap();
    let budget = independent(&m, &c).iter().sum::<f64>() / 2. * 1.005;
    p["update"]["target_kl"] = json!(budget);
    p["update"]["max_parameter_l2"] = json!(1. / 32.);
    let initial = start(&p).unwrap();
    let (raw, pin) = initial.checkpoint_with_id().unwrap();
    let mut cold = S::restore(m.clone(), p.to_string().as_bytes(), raw.as_bytes(), &pin).unwrap();
    let mut warm = initial;
    let r = warm.step_adaptation().unwrap().unwrap();
    assert_eq!(r.anchor_kl.as_ref().unwrap().trials.len(), 1);
    assert_eq!(
        serde_json::to_value(&r).unwrap(),
        serde_json::to_value(cold.step_adaptation().unwrap().unwrap()).unwrap()
    );
    let cut = warm.checkpoint_with_id().unwrap();
    let restored = S::restore(
        m.clone(),
        p.to_string().as_bytes(),
        cut.0.as_bytes(),
        &cut.1,
    )
    .unwrap();
    assert_eq!(restored.checkpoint_with_id().unwrap(), cut);
    p["update"]["anchor_examples"] = json!([1, 0]);
    assert!(S::restore(
        m.clone(),
        p.to_string().as_bytes(),
        cut.0.as_bytes(),
        &cut.1
    )
    .is_err());
    p["update"]["target_kl"] = json!(1e9);
    let mut refused = start(&p).unwrap();
    let before = refused.checkpoint_with_id().unwrap();
    let failure = refused.step_adaptation().unwrap_err();
    let refusal = failure.downcast_ref::<CausalLmAnchorKlRefusal>().unwrap();
    assert_eq!(refusal.trials.len(), 1);
    assert!(refusal.trials[0].mean_kl < refusal.policy.target_kl * 0.98);
    assert_eq!(refused.checkpoint_with_id().unwrap(), before);
    assert_eq!(refused.completed_updates(), 0);
}

#[test]
fn no_match_at_the_trial_limit_is_not_committed_as_a_small_update() {
    let m = model(1, CausalLmReadoutAccumulator::F32);
    let g = m.adaptation_loss_gradients(&[1, 3, 5], [1, 3]).unwrap();
    // Even the twentieth halving is a representable, nontrivial parameter move.
    let policy = CausalLmAnchorKlPolicy {
        target_kl: 1e-30,
        max_parameter_l2: 16384.,
    };
    let id = m.geometry_artifact_id().unwrap().to_owned();
    let err = match m.step_adaptation_at_anchor_kl(&g, Geometry, ANCHORS, policy) {
        Err(err) => err,
        Ok(_) => panic!("an unmatched proposal was accepted"),
    };
    let r = err.downcast_ref::<CausalLmAnchorKlRefusal>().unwrap();
    assert_eq!(r.reason, "bounded bracket exhausted");
    assert_eq!(r.trials.len(), CausalLmAnchorKlPolicy::MAX_TRIALS);
    for (i, t) in r.trials.iter().enumerate() {
        assert_eq!(
            t.requested_l2,
            policy.max_parameter_l2 * 0.5_f64.powi(i as i32)
        );
        assert!(t.mean_kl > policy.target_kl);
    }
    assert_eq!(m.geometry_artifact_id(), Some(id.as_str()));
}

#[test]
fn stale_gradient_and_bad_bank_are_rejected_before_any_anchor_decode() {
    let m = model(1, CausalLmReadoutAccumulator::F32);
    let g = m.adaptation_loss_gradients(&[1, 3, 5], [1, 3]).unwrap();
    let (other, _) = m
        .step_adaptation_gradient_at_distance(&g, Joint, 1. / 16.)
        .unwrap();
    let policy = CausalLmAnchorKlPolicy {
        target_kl: 1e-5,
        max_parameter_l2: 1. / 16.,
    };
    let many = vec![0; CausalLmAnchorKlPolicy::MAX_ANCHOR_IDS + 1];
    for prompts in [
        ANCHORS,
        &[&[99][..]][..],
        &[&[][..]][..],
        &[many.as_slice()][..],
    ] {
        decoder::work_counts::take();
        assert!(other
            .step_adaptation_at_anchor_kl(&g, Joint, prompts, policy)
            .is_err());
        assert_eq!(decoder::work_counts::take(), [0; 3]);
    }
}
