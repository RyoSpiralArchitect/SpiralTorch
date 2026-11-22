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
