use super::*;

fn probabilities(z: &[f32]) -> Vec<f64> {
    let v: Vec<_> = z.iter().map(|&x| f64::from(x).exp()).collect();
    let sum: f64 = v.iter().sum();
    v.into_iter().map(|p| p / sum).collect()
}

fn modify(m: &NativeCausalLm, name: &str, i: usize, dx: f32) -> NativeCausalLm {
    if name == "a" || name == "b" {
        let mut r: serde_json::Value = serde_json::from_str(&m.lora_to_json().unwrap()).unwrap();
        r[name][i] = json!(r[name][i].as_f64().unwrap() as f32 + dx);
        m.with_lora_json(&r.to_string()).unwrap()
    } else {
        let mut r = m.geometry_artifact().unwrap().record().clone();
        r.placements[0]
            .tensors
            .iter_mut()
            .find(|t| t.name == name)
            .unwrap()
            .values[i] += dx;
        m.with_geometry(GeometryAdapterArtifact::create(r).unwrap())
            .unwrap()
    }
}

#[test]
fn selected_prompt_secant_matches_independent_coordinate_differences() {
    let m = model(1, CausalLmReadoutAccumulator::F32WideBackward);
    let mut r = m.geometry_artifact().unwrap().record().clone();
    let frozen = r.placements[0].tensors[0].name.clone();
    r.placements[0].tensors[0].requires_grad = false;
    let m = m
        .with_geometry(GeometryAdapterArtifact::create(r).unwrap())
        .unwrap();
    let g = m.adaptation_loss_gradients(&[1, 3, 5], [1, 3]).unwrap();
    let d = 1. / 256.;
    let (probe, _) = m
        .step_adaptation_gradient_at_distance(&g, Joint, d)
        .unwrap();
    let kls = independent(&m, &probe);
    let peak = if kls[0] >= kls[1] { 0 } else { 1 };
    for which in [
        CausalLmSecantAnchor::Largest,
        CausalLmSecantAnchor::Successor,
    ] {
        let p = ANCHORS[if which == CausalLmSecantAnchor::Largest {
            peak
        } else {
            1 - peak
        }];
        let probs = |m: &NativeCausalLm| probabilities(m.next_token_logits(p).unwrap().data());
        let seed: Vec<f64> = probs(&probe)
            .into_iter()
            .zip(probs(&m))
            .map(|(b, a)| b - a)
            .collect();
        let objective = |m: NativeCausalLm| {
            m.next_token_logits(p)
                .unwrap()
                .data()
                .iter()
                .zip(&seed)
                .map(|(&x, s)| f64::from(x) * s)
                .sum::<f64>()
        };
        let (mut raw, mut u) = (Vec::new(), Vec::new());
        let eps = 1. / 1024.;
        for blocks in g.parameters().values() {
            for (name, values) in blocks {
                if name == &frozen {
                    continue;
                }
                for (i, &gi) in values.iter().enumerate() {
                    raw.push(f64::from(gi));
                    u.push(
                        (objective(modify(&m, name, i, eps))
                            - objective(modify(&m, name, i, -eps)))
                            / (2. * f64::from(eps)),
                    );
                }
            }
        }
        let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
        let scale = dot(&raw, &u) / dot(&u, &u);
        let h: Vec<_> = raw.iter().zip(&u).map(|(&g, &n)| g - scale * n).collect();
        let direction = CausalLmAnchorDirection::SecantProjection {
            probe_l2: d,
            anchor: which,
        };
        let call = |t| {
            m.step_adaptation_at_anchor_kl_with_direction(
                &g,
                Joint,
                ANCHORS,
                CausalLmAnchorKlPolicy {
                    target_kl: t,
                    max_parameter_l2: 1. / 32.,
                },
                Default::default(),
                direction,
            )
        };
        // Use a known reachable target for this unit test, not for the frozen study.
        let failure = call(1e3).err().unwrap();
        let stop = failure.downcast_ref::<CausalLmAnchorKlRefusal>().unwrap();
        assert!(stop
            .direction
            .as_ref()
            .unwrap()
            .source_loss_dot_step
            .is_none());
        let (end, ray, report) = call(stop.trials[0].mean_kl * 1.005).unwrap();
        let info = report.direction.unwrap();
        assert_eq!(info.peak_anchor, peak);
        assert_eq!(
            info.selected_anchor,
            if which == CausalLmSecantAnchor::Largest {
                peak
            } else {
                1 - peak
            }
        );
        assert_eq!(report.anchor_readouts, 6);
        assert!((info.normal_l2 / dot(&u, &u).sqrt() - 1.).abs() < 0.006);
        assert!((info.projected_l2 / dot(&h, &h).sqrt() - 1.).abs() < 0.006);
        assert!(info.projected_normal_cosine.abs() < 1e-6);
        assert_eq!(ray.method, "selected_parameter_secant_projected_ray_f32.v1");
        let weights = |m: &NativeCausalLm| {
            let mut w = std::collections::BTreeMap::new();
            for (n, v) in m.lora.as_ref().unwrap().factor_values() {
                w.insert((m.lora.as_ref().unwrap().path(), n.to_owned()), v.to_vec());
            }
            for p in &m.geometry_artifact().unwrap().record().placements {
                for t in &p.tensors {
                    w.insert((p.path.clone(), t.name.clone()), t.values.clone());
                }
            }
            w
        };
        let a = weights(&m);
        let b = weights(&end);
        let mut dx = Vec::new();
        for (key, block) in a {
            if key.1 == frozen {
                assert_eq!(block, b[&key]);
                continue;
            }
            dx.extend(
                b[&key]
                    .iter()
                    .zip(block)
                    .map(|(&y, x)| f64::from(y) - f64::from(x)),
            );
        }
        assert!(dot(&dx, &h) / (dot(&dx, &dx) * dot(&h, &h)).sqrt() < -0.999);
        assert!((dot(&raw, &dx) - info.source_loss_dot_step.unwrap()).abs() < 1e-12);
    }
}

#[test]
fn invalid_shaping_is_rejected_before_decoding() {
    let m = model(1, CausalLmReadoutAccumulator::F32);
    let g = m.adaptation_loss_gradients(&[1, 3, 5], [1, 3]).unwrap();
    let end = m
        .step_adaptation_gradient_at_distance(&g, Joint, 1. / 64.)
        .unwrap()
        .0;
    for (origin, anchors, d) in [
        (&m, &ANCHORS[..1], 0.01),
        (&m, ANCHORS, 0.),
        (&m, ANCHORS, 2.),
        (&m, ANCHORS, f64::NAN),
        (&end, ANCHORS, 0.01),
    ] {
        decoder::work_counts::take();
        assert!(origin
            .step_adaptation_at_anchor_kl_with_direction(
                &g,
                Joint,
                anchors,
                CausalLmAnchorKlPolicy {
                    target_kl: 0.01,
                    max_parameter_l2: 0.1
                },
                Default::default(),
                CausalLmAnchorDirection::SecantProjection {
                    probe_l2: d,
                    anchor: CausalLmSecantAnchor::Largest
                }
            )
            .is_err());
        assert_eq!(decoder::work_counts::take(), [0; 3]);
    }
}
