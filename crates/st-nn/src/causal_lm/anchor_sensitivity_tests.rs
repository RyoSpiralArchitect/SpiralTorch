use super::*;

use CausalLmSensitivityBasis::{Coordinate, ParameterBlock};
use CausalLmSensitivitySelect::{Envelope, GradientNorm};

fn direction(
    basis: CausalLmSensitivityBasis,
    select: CausalLmSensitivitySelect,
) -> CausalLmAnchorDirection {
    CausalLmAnchorDirection::RawThenSensitivityRay {
        probe_l2: 1. / 256.,
        basis,
        select,
    }
}

#[test]
fn actual_masked_probes_match_independent_owner_rays_without_another_backward() {
    let m = model(1, CausalLmReadoutAccumulator::F32WideBackward);
    let g = m.adaptation_loss_gradients(&[1, 3, 5], [1, 3]).unwrap();
    let policy = CausalLmAnchorKlPolicy {
        target_kl: 1e6,
        max_parameter_l2: 1. / 32.,
    };
    for basis in [ParameterBlock, Coordinate] {
        for select in [Envelope, GradientNorm] {
            decoder::work_counts::take();
            let err = m
                .step_adaptation_at_anchor_kl_with_direction(
                    &g,
                    Geometry,
                    ANCHORS,
                    policy,
                    CausalLmAnchorKlCriterion::MeanWithCap { max_anchor_kl: 2e6 },
                    direction(basis, select),
                )
                .err()
                .unwrap()
                .downcast::<CausalLmAnchorKlRefusal>()
                .unwrap();
            let work = decoder::work_counts::take();
            assert_eq!(work[1], 0, "sensitivity must not repeat the loss backward");
            assert_eq!(work[2], err.anchor_readouts);
            let raw = err.raw_miss.unwrap();
            let report = err.sensitivity.unwrap();
            assert!(report.source_loss_dot_step.is_none() && err.direction.is_none());
            let nonzero = report.axes.iter().filter(|a| a.gradient_l2 != 0.).count();
            assert!(nonzero > 0 && report.selected_axis.is_some());
            assert_eq!(report.probe_readouts, nonzero * ANCHORS.len());
            assert_eq!(
                err.anchor_readouts,
                raw.anchor_readouts + report.probe_readouts + err.trials.len() * ANCHORS.len()
            );
            for a in report.axes {
                let mut manual = g.parameters().clone();
                for (path, blocks) in &mut manual {
                    for (name, values) in blocks {
                        for (j, v) in values.iter_mut().enumerate() {
                            if *path != a.path
                                || *name != a.tensor
                                || a.coordinate.is_some_and(|i| i != j)
                            {
                                *v = 0.;
                            }
                        }
                    }
                }
                if a.gradient_l2 == 0. {
                    assert!(a.probe_per_anchor_kl.is_none());
                    continue;
                }
                let (probe, ray) = m.step_parameter_ray(&manual, Geometry, 1. / 256.).unwrap();
                assert!((ray.gradient_l2 - a.gradient_l2).abs() < 1e-10);
                let expected = independent(&m, &probe);
                for (x, y) in expected.iter().zip(a.probe_per_anchor_kl.unwrap()) {
                    assert!((x - y).abs() < 1e-14);
                }
            }
        }
    }
}

#[test]
fn raw_success_and_admission_errors_never_decode_a_probe_bank() {
    let m = model(1, CausalLmReadoutAccumulator::F32);
    let g = m.adaptation_loss_gradients(&[1, 3, 5], [1, 3]).unwrap();
    let d = 1. / 32.;
    let (target, _) = m
        .step_adaptation_gradient_at_distance(&g, Geometry, d)
        .unwrap();
    let expected = independent(&m, &target);
    let t = 1.005 * expected.iter().sum::<f64>() / 2.;
    let policy = CausalLmAnchorKlPolicy {
        target_kl: t,
        max_parameter_l2: d,
    };
    let cap = CausalLmAnchorKlCriterion::MeanWithCap {
        max_anchor_kl: 2. * t,
    };
    let (_, ray, raw) = m
        .step_adaptation_at_anchor_kl_with_criterion(&g, Geometry, ANCHORS, policy, cap)
        .unwrap();
    decoder::work_counts::take();
    let (end, actual, report) = m
        .step_adaptation_at_anchor_kl_with_direction(
            &g,
            Geometry,
            ANCHORS,
            policy,
            cap,
            direction(Coordinate, Envelope),
        )
        .unwrap();
    assert_eq!(end.geometry_artifact_id(), target.geometry_artifact_id());
    assert_eq!(
        serde_json::to_value((ray, raw)).unwrap(),
        serde_json::to_value((actual, report)).unwrap()
    );
    assert_eq!(decoder::work_counts::take()[2], 4);
    for (dir, rule) in [
        (
            direction(ParameterBlock, Envelope),
            CausalLmAnchorKlCriterion::Peak {},
        ),
        (
            CausalLmAnchorDirection::RawThenSensitivityRay {
                probe_l2: 1.,
                basis: Coordinate,
                select: Envelope,
            },
            cap,
        ),
    ] {
        assert!(m
            .step_adaptation_at_anchor_kl_with_direction(&g, Geometry, ANCHORS, policy, rule, dir)
            .is_err());
        assert_eq!(decoder::work_counts::take(), [0; 3]);
    }
    for bad in [
        r#"{"kind":"raw_then_sensitivity_ray","basis":"coordinate","probe_l2":0.01}"#,
        r#"{"kind":"raw_then_sensitivity_ray","basis":"coordinate","select":"last","probe_l2":0.01}"#,
        r#"{"kind":"raw_then_sensitivity_ray","basis":"coordinate","select":"envelope","probe_l2":0.01,"retry":true}"#,
    ] {
        assert!(serde_json::from_str::<CausalLmAnchorDirection>(bad).is_err());
    }
}

#[test]
fn zero_gradient_is_skipped_but_frozen_blocks_still_validate() {
    let base = load().with_readout_accumulator(CausalLmReadoutAccumulator::F32);
    let mut rec = geometry(&base, 1, 0.2).record().clone();
    let frozen = rec.placements[0]
        .tensors
        .iter_mut()
        .find(|t| t.name == "bias")
        .unwrap();
    frozen.requires_grad = false;
    let m = base
        .with_geometry(GeometryAdapterArtifact::create(rec).unwrap())
        .unwrap()
        .with_lora(1, 2, 3.)
        .unwrap();
    let g = m.adaptation_loss_gradients(&[1, 3, 5], [1, 3]).unwrap();
    let policy = CausalLmAnchorKlPolicy {
        target_kl: 1e6,
        max_parameter_l2: 1. / 32.,
    };
    for mode in [Joint, Geometry] {
        let err = m
            .step_adaptation_at_anchor_kl_with_direction(
                &g,
                mode,
                ANCHORS,
                policy,
                CausalLmAnchorKlCriterion::MeanWithCap { max_anchor_kl: 2e6 },
                direction(ParameterBlock, GradientNorm),
            )
            .err()
            .unwrap()
            .downcast::<CausalLmAnchorKlRefusal>()
            .unwrap();
        let info = err.sensitivity.unwrap();
        assert!(!info.axes.iter().any(|a| a.tensor == "bias"));
        if mode == Joint {
            let a = info.axes.iter().find(|a| a.tensor == "a").unwrap();
            assert_eq!(a.gradient_l2, 0.);
            assert!(a.probe_per_anchor_kl.is_none() && a.prediction.is_none());
        }
        assert_eq!(
            info.probe_readouts,
            2 * info.axes.iter().filter(|a| a.gradient_l2 > 0.).count()
        );
    }
}
