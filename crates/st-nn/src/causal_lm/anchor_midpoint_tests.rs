use super::*;

fn rescue() -> CausalLmAnchorDirection {
    CausalLmAnchorDirection::RawThenSecantMidpoint {
        probe_l2: 1. / 256.,
        anchor: CausalLmSecantAnchor::Largest,
    }
}

#[test]
fn raw_success_has_identical_weights_receipt_and_actual_decoder_work() {
    assert!(
        std::mem::size_of::<CausalLmLineAttempt>() <= 64,
        "keep rare refusal records out of successful attempts"
    );
    let m = model(1, CausalLmReadoutAccumulator::F32WideBackward);
    let g = m.adaptation_loss_gradients(&[1, 3, 5], [1, 3]).unwrap();
    for mode in [Lora, Geometry, Joint] {
        let d = 1. / 32.;
        let expected = m
            .step_adaptation_gradient_at_distance(&g, mode, d)
            .unwrap()
            .0;
        let kl = independent(&m, &expected);
        let policy = CausalLmAnchorKlPolicy {
            max_parameter_l2: d,
            target_kl: 1.005 * kl.iter().sum::<f64>() / 2.,
        };
        decoder::work_counts::take();
        let (_, r, k) = m
            .step_adaptation_at_anchor_kl(&g, mode, ANCHORS, policy)
            .unwrap();
        let work = decoder::work_counts::take();
        let (new, r2, k2) = m
            .step_adaptation_at_anchor_kl_with_direction(
                &g,
                mode,
                ANCHORS,
                policy,
                Default::default(),
                rescue(),
            )
            .unwrap();
        assert_eq!(decoder::work_counts::take(), work);
        assert_eq!(work[2], 4);
        assert_eq!(new.geometry_artifact_id(), expected.geometry_artifact_id());
        assert_eq!(new.lora_artifact_id(), expected.lora_artifact_id());
        assert_eq!(
            serde_json::to_string(&r).unwrap(),
            serde_json::to_string(&r2).unwrap()
        );
        assert_eq!(
            serde_json::to_string(&k).unwrap(),
            serde_json::to_string(&k2).unwrap()
        );
        assert!(k2.direction.is_none() && k2.raw_miss.is_none());
    }
}

#[test]
fn second_bracket_uses_one_origin_bank_and_retains_first_miss() {
    let m = model(1, CausalLmReadoutAccumulator::F32);
    let g = m.adaptation_loss_gradients(&[1, 3, 5], [1, 3]).unwrap();
    let policy = CausalLmAnchorKlPolicy {
        max_parameter_l2: 1. / 32.,
        target_kl: 1e6,
    };
    let call = |d| {
        m.step_adaptation_at_anchor_kl_with_direction(
            &g,
            Joint,
            ANCHORS,
            policy,
            CausalLmAnchorKlCriterion::MeanWithCap { max_anchor_kl: 2e6 },
            d,
        )
        .err()
        .unwrap()
        .downcast::<CausalLmAnchorKlRefusal>()
        .unwrap()
    };
    let raw = call(Default::default());
    decoder::work_counts::take();
    let both = call(rescue());
    let count = decoder::work_counts::take();
    let first = both.raw_miss.unwrap();
    assert_eq!(first.reason, raw.reason);
    assert_eq!(
        serde_json::to_string(&first.trials).unwrap(),
        serde_json::to_string(&raw.trials).unwrap()
    );
    assert_eq!(first.anchor_readouts, 4);
    assert_eq!(both.trials.len(), 1);
    assert_eq!(both.anchor_readouts, 8); // origin, raw ceiling, probe, midpoint ceiling
    assert_eq!(count[2], both.anchor_readouts);
    assert!(count[1] > 0, "rescue did not pull back the anchor");
    let d = both.direction.unwrap();
    assert!(d.source_loss_dot_step.is_none());
    let mid = d.midpoint.unwrap();
    assert!((mid.source_cosine - mid.projected_cosine).abs() < 1e-6);
    assert!((mid.source_cosine - ((1. + d.retained_fraction) / 2.).sqrt()).abs() < 1e-6);
}

#[test]
fn raw_errors_are_not_a_license_for_another_search() {
    let m = model(1, CausalLmReadoutAccumulator::F32);
    let g = m
        .adaptation_loss_gradients_for(&[1, 3, 5], [1, 3], Geometry)
        .unwrap();
    let policy = CausalLmAnchorKlPolicy {
        max_parameter_l2: 0.125,
        target_kl: 1e6,
    };
    decoder::work_counts::take();
    let err = m
        .step_adaptation_at_anchor_kl_with_direction(
            &g,
            Joint,
            ANCHORS,
            policy,
            Default::default(),
            rescue(),
        )
        .err()
        .unwrap();
    assert!(err.downcast_ref::<CausalLmAnchorKlRefusal>().is_none());
    assert_eq!(decoder::work_counts::take(), [0; 3]);
}

#[test]
fn loss_prompt_requires_sealed_line_provenance_and_exact_bank_membership_before_decode() {
    let m = model(1, CausalLmReadoutAccumulator::F32);
    let direction = CausalLmAnchorDirection::RawThenSecantMidpoint {
        probe_l2: 1. / 256.,
        anchor: CausalLmSecantAnchor::LossPrompt,
    };
    let line = CausalLmLineObjective::Set {}
        .gradients_for(&m, ANCHORS[0], &[vec![1], vec![2]], 0, None, Joint)
        .unwrap();
    let other = m.adaptation_loss_gradients(&[1, 3, 5], [1, 3]).unwrap();
    assert!(other.line_prompt_id().is_none());
    let run = |g, bank: &[&[usize]]| {
        m.step_adaptation_at_anchor_kl_with_direction(
            g,
            Joint,
            bank,
            CausalLmAnchorKlPolicy {
                target_kl: 1e6,
                max_parameter_l2: 1. / 32.,
            },
            Default::default(),
            direction,
        )
    };
    let absent: &[&[usize]] = &[&[3, 1], ANCHORS[1]];
    for (g, bank, message) in [
        (&other, ANCHORS, "complete-line gradient"),
        (&line, absent, "absent from the anchor bank"),
    ] {
        decoder::work_counts::take();
        assert!(run(g, bank).err().unwrap().to_string().contains(message));
        assert_eq!(decoder::work_counts::take(), [0; 3]);
    }
    for (index, bank) in [(0, ANCHORS), (1, &[ANCHORS[1], ANCHORS[0]][..])] {
        let miss = run(&line, bank)
            .err()
            .unwrap()
            .downcast::<CausalLmAnchorKlRefusal>()
            .unwrap();
        let d = miss.direction.unwrap();
        assert_eq!(d.selected_anchor, index);
        assert_eq!(d.loss_prompt_id, line.line_prompt_id());
    }
}

#[test]
fn scheduled_prompt_coverage_is_by_tokens_not_example_index() {
    type S = NativeCausalLmLineTrainingSession;
    let m = model(1, CausalLmReadoutAccumulator::F32);
    let mut p = json!({"schema":"spiraltorch.native_causal_lm_line_plan.v3","mode":"joint","epochs":1,
        "examples":[{"prompt_ids":[1,3],"candidates":[[1],[2]],"stop_id":0,"objective":{"kind":"set"}},
                    {"prompt_ids":[2,4,5],"candidates":[[1],[2]],"stop_id":0,"objective":{"kind":"set"}},
                    {"prompt_ids":[1,3],"candidates":[[3],[4]],"stop_id":0,"objective":{"kind":"menu","gold_index":1,"mix":0.3}}],
        "update":{"kind":"anchor_forward_kl","target_kl":0.01,"max_parameter_l2":0.125,"anchor_examples":[1,0],
            "direction":{"kind":"raw_then_secant_midpoint","anchor":"loss_prompt","probe_l2":0.003}}});
    let start = |p: &serde_json::Value| S::new(m.clone(), p.to_string().as_bytes());
    let (cut, id) = start(&p).unwrap().checkpoint_with_id().unwrap();
    p["update"]["anchor_examples"] = json!([0, 1]);
    assert!(S::restore(m.clone(), p.to_string().as_bytes(), cut.as_bytes(), &id).is_err());
    p["examples"][2]["prompt_ids"] = json!([3, 1]);
    decoder::work_counts::take();
    assert!(start(&p)
        .unwrap_err()
        .to_string()
        .contains("every scheduled prompt"));
    assert_eq!(decoder::work_counts::take(), [0; 3]);
    p["update"]["direction"]["anchor"] = json!("successor");
    assert!(start(&p).is_ok());
}

#[test]
fn sealed_selection_works_on_both_policies_and_raw_matches_skip_the_probe() {
    let m = model(1, CausalLmReadoutAccumulator::F32WideBackward);
    let g = CausalLmLineObjective::Set {}
        .gradients_for(&m, ANCHORS[1], &[vec![1], vec![2]], 0, None, Joint)
        .unwrap();
    let d = 1. / 32.;
    let raw = m
        .step_adaptation_gradient_at_distance(&g, Joint, d)
        .unwrap()
        .0;
    let mean = independent(&m, &raw).iter().sum::<f64>() / 2.;
    let policy = CausalLmAnchorKlPolicy {
        target_kl: 1.005 * mean,
        max_parameter_l2: d,
    };
    decoder::work_counts::take();
    let (_, r, k) = m
        .step_adaptation_at_anchor_kl(&g, Joint, ANCHORS, policy)
        .unwrap();
    let counts = decoder::work_counts::take();
    let (new, r2, k2) = m
        .step_adaptation_at_anchor_kl_with_direction(
            &g,
            Joint,
            ANCHORS,
            policy,
            Default::default(),
            CausalLmAnchorDirection::RawThenSecantMidpoint {
                probe_l2: 1. / 256.,
                anchor: CausalLmSecantAnchor::LossPrompt,
            },
        )
        .unwrap();
    assert_eq!(decoder::work_counts::take(), counts);
    assert_eq!(new.lora_artifact_id(), raw.lora_artifact_id());
    assert_eq!(new.geometry_artifact_id(), raw.geometry_artifact_id());
    assert_eq!(
        serde_json::to_value((r, k)).unwrap(),
        serde_json::to_value((r2, k2)).unwrap()
    );
    let failure = m
        .step_adaptation_at_anchor_kl_with_direction(
            &g,
            Joint,
            ANCHORS,
            CausalLmAnchorKlPolicy {
                target_kl: 1e6,
                ..policy
            },
            Default::default(),
            CausalLmAnchorDirection::SecantProjection {
                probe_l2: 1. / 256.,
                anchor: CausalLmSecantAnchor::LossPrompt,
            },
        )
        .err()
        .unwrap()
        .downcast::<CausalLmAnchorKlRefusal>()
        .unwrap();
    assert!(failure.raw_miss.is_none());
    let d = failure.direction.unwrap();
    assert!(d.midpoint.is_none());
    assert_eq!(d.selected_anchor, 1);
    assert_eq!(d.loss_prompt_id, g.line_prompt_id());
}
