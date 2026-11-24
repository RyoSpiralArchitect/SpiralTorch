use super::*;

#[test]
fn same_origin_matches_peak_or_mean_without_changing_the_ray() {
    let m = model(1, CausalLmReadoutAccumulator::F32WideBackward);
    let g = m.adaptation_loss_gradients(&[1, 3, 5], [1, 3]).unwrap();
    for mode in [Lora, Geometry, Joint] {
        let distance = 1. / 32.;
        let (expected, _) = m
            .step_adaptation_gradient_at_distance(&g, mode, distance)
            .unwrap();
        let row = independent(&m, &expected);
        let mean = row.iter().sum::<f64>() / row.len() as f64;
        let index = usize::from(row[1] > row[0]);
        let peak = row[index];
        for (rule, stat) in [
            (
                CausalLmAnchorKlCriterion::MeanWithCap {
                    max_anchor_kl: peak * 1.01,
                },
                mean,
            ),
            (CausalLmAnchorKlCriterion::Peak {}, peak),
        ] {
            let policy = CausalLmAnchorKlPolicy {
                target_kl: stat * 1.005,
                max_parameter_l2: distance,
            };
            let (end, ray, r) = m
                .step_adaptation_at_anchor_kl_with_criterion(&g, mode, ANCHORS, policy, rule)
                .unwrap();
            assert_eq!(
                (end.lora_artifact_id(), end.geometry_artifact_id()),
                (expected.lora_artifact_id(), expected.geometry_artifact_id())
            );
            assert_eq!(r.criterion, rule);
            assert_eq!(ray.requested_l2, distance);
            assert_eq!(r.trials.len(), 1);
            assert_eq!(r.anchor_readouts, 4);
            assert!((r.mean_kl - mean).abs() < 1e-14);
            let p = r.trials[0].peak.unwrap();
            assert_eq!(p.anchor_index, index);
            assert!((p.kl - peak).abs() < 1e-14);
        }
        let policy = CausalLmAnchorKlPolicy {
            target_kl: mean * 1.005,
            max_parameter_l2: distance,
        };
        let (_, _, legacy) = m
            .step_adaptation_at_anchor_kl(&g, mode, ANCHORS, policy)
            .unwrap();
        let r = serde_json::to_value(legacy).unwrap();
        assert!(r.get("criterion").is_none() && r["trials"][0].get("peak").is_none());
        // The old first choice is inside the mean band but breaches this stricter cap.
        assert!(peak > mean * 1.02);
        let cap = (mean + peak) * 0.5;
        let rule = CausalLmAnchorKlCriterion::MeanWithCap { max_anchor_kl: cap };
        let trials =
            match m.step_adaptation_at_anchor_kl_with_criterion(&g, mode, ANCHORS, policy, rule) {
                Ok((_, _, r)) => {
                    assert!(r.per_anchor_kl.iter().all(|&x| x <= cap));
                    assert!((0.98..=1.).contains(&(r.mean_kl / policy.target_kl)));
                    r.trials
                }
                Err(err) => err.downcast::<CausalLmAnchorKlRefusal>().unwrap().trials,
            };
        assert!(trials.len() > 1 && trials[0].peak.unwrap().kl > cap);
    }
}

#[test]
fn plan_admission_seals_criterion_and_attempt_only_catches_measurement_refusals() {
    type S = NativeCausalLmLineTrainingSession;
    let m = model(1, CausalLmReadoutAccumulator::F32);
    let mut p = json!({"schema":"spiraltorch.native_causal_lm_line_plan.v3","mode":"joint","epochs":1,
        "examples":[{"prompt_ids":[1,3],"candidates":[[1],[2]],"stop_id":0,"objective":{"kind":"set"}}],
        "update":{"kind":"anchor_forward_kl","target_kl":1e6,"max_parameter_l2":0.125,"anchor_examples":[0]}});
    let start = |p: &serde_json::Value| S::new(m.clone(), p.to_string().as_bytes());
    for bad in [
        json!(null),
        json!("peak"),
        json!({"kind":"maximum"}),
        json!({"kind":"peak","max_anchor_kl":2e6}),
        json!({"kind":"mean_with_cap"}),
        json!({"kind":"mean_with_cap","max_anchor_kl":0.5e6}),
    ] {
        p["update"]["criterion"] = bad;
        decoder::work_counts::take();
        assert!(start(&p).is_err());
        assert_eq!(decoder::work_counts::take(), [0; 3]);
    }
    p["update"]["criterion"] = json!({"kind":"peak"});
    let mut strict = start(&p).unwrap();
    let origin = strict.checkpoint_with_id().unwrap();
    let mut attempt = S::restore(
        m.clone(),
        p.to_string().as_bytes(),
        origin.0.as_bytes(),
        &origin.1,
    )
    .unwrap();
    let err = strict
        .step_adaptation()
        .unwrap_err()
        .downcast::<CausalLmAnchorKlRefusal>()
        .unwrap();
    match attempt.attempt_adaptation_step().unwrap().unwrap() {
        CausalLmLineAttempt::Unmatched {
            plan_id,
            next_update,
            example_index,
            refusal,
        } => {
            assert_eq!(plan_id, attempt.plan_id());
            assert_eq!(
                (next_update, example_index, refusal.anchor_readouts),
                (1, 0, 2)
            );
            assert_eq!(
                serde_json::to_value(&*err).unwrap(),
                serde_json::to_value(refusal).unwrap()
            );
        }
        _ => panic!("an unmatched trajectory advanced"),
    }
    assert_eq!(strict.checkpoint_with_id().unwrap(), origin);
    assert_eq!(attempt.checkpoint_with_id().unwrap(), origin);
    p["update"]["criterion"] = json!({"kind":"mean_with_cap","max_anchor_kl":2e6});
    assert!(S::restore(
        m.clone(),
        p.to_string().as_bytes(),
        origin.0.as_bytes(),
        &origin.1
    )
    .is_err());
    p["update"] = json!({"kind":"parameter_gradient_l2","target_l2":1e-200});
    let mut invalid = start(&p).unwrap();
    assert!(invalid.attempt_adaptation_step().is_err());
    assert_eq!(invalid.completed_updates(), 0);
    p["update"] = json!({"kind":"sgd","learning_rate":0.1});
    let mut ordinary = start(&p).unwrap();
    assert!(matches!(
        ordinary.attempt_adaptation_step().unwrap(),
        Some(CausalLmLineAttempt::Applied { .. })
    ));
    assert!(ordinary.attempt_adaptation_step().unwrap().is_none());
}
