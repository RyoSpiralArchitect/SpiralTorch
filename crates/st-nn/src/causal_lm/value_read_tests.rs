use super::*;
use crate::geometry_artifact::{AttentionValueRoute as R, GeometryAdapterRecipe};

fn placed(base: &NativeCausalLm, r: R, g: f32) -> NativeCausalLm {
    let path = "transformer.h.1.mlp";
    let mut rec = artifact(base, path, g).record().clone();
    let p = &mut rec.placements[0];
    p.training = true;
    rec.training = true;
    p.recipe = GeometryAdapterRecipe::AttentionValueGate {
        features: 4,
        strength: 0.5,
        max_values: 64,
        route: r,
    };
    for t in &mut p.tensors {
        t.requires_grad = t.name == "gate";
        if t.requires_grad {
            t.values = vec![g, -g, 2. * g, -3. * g];
        } else {
            t.values.fill(0.);
        }
    }
    base.with_geometry(GeometryAdapterArtifact::create(rec).unwrap())
        .unwrap()
}
const ROUTES: [R; 3] = [R::Attention, R::Uniform, R::Fixed { position: 2 }];

#[test]
fn value_read_gate_uses_causal_per_head_values_and_no_extra_kv_words() {
    let base = load()
        .with_key_lag(CausalLmKeyLag { layer: 1, lag: 1 })
        .unwrap();
    let ids = [2, 4, 6, 5, 7];
    let cap = 2 * 2 * ids.len() * 4;
    for route in ROUTES {
        let m = placed(&base, route, 0.7);
        for n in 1..=ids.len() {
            let tr = m.last_mlp_value_read_signal(&ids[..n], 7, 8).unwrap();
            assert_eq!(tr.route, route);
            assert_eq!(tr.contrast, m.next_token_contrast(&ids[..n], 7, 8).unwrap());
            let w = 4;
            let dim = w / tr.heads;
            for h in 0..tr.heads {
                let mut p: Vec<f64> = (0..n)
                    .map(|t| {
                        let j = t.saturating_sub(1) * w + h * dim;
                        f64::from(
                            tr.scale
                                * (0..dim)
                                    .map(|d| tr.query[h * dim + d] * tr.keys[j + d])
                                    .sum::<f32>(),
                        )
                    })
                    .collect();
                let hi = p.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                for x in &mut p {
                    *x = (*x - hi).exp();
                }
                let z = p.iter().sum::<f64>();
                for d in h * dim..(h + 1) * dim {
                    let mu = (0..n)
                        .map(|t| {
                            f64::from(tr.values[t * w + d])
                                * match route {
                                    R::Attention => p[t] / z,
                                    R::Uniform => 1. / n as f64,
                                    R::Fixed { position } => {
                                        f64::from(t == (position as usize).min(n - 1))
                                    }
                                }
                        })
                        .sum::<f64>();
                    assert!((mu - f64::from(tr.source[d])).abs() < 2e-6);
                    assert!(
                        (tr.gate_gradient[d] - tr.bias_gradient[d] * tr.source[d]).abs() < 2e-6
                    );
                }
            }
        }
        let before = NativeCausalLmContext::new(m.clone(), ids[..1].to_vec(), cap).unwrap();
        let chunk = before.with_suffix(&ids[1..]).unwrap();
        let mut singles = before.clone();
        for &id in &ids[1..] {
            singles.append(id).unwrap();
        }
        close(
            chunk.logit_values(),
            m.next_token_logits(&ids).unwrap().data(),
        );
        close(chunk.logit_values(), singles.logit_values());
        assert_eq!(chunk.cache_values(), cap);
        assert_eq!(before.token_ids(), &ids[..1]);
        let packed = m
            .logits(&[ids, [8, 7, 6, 5, 4]].concat(), [2, 5], true)
            .unwrap();
        close(
            &packed.data()[..m.config.vocabulary],
            m.next_token_logits(&ids).unwrap().data(),
        );
        close(
            &packed.data()[m.config.vocabulary..],
            m.next_token_logits(&[8, 7, 6, 5, 4]).unwrap().data(),
        );
        let zero = placed(&base, route, 0.);
        assert_eq!(
            zero.next_token_logits(&ids).unwrap().data(),
            base.next_token_logits(&ids).unwrap().data()
        );
        assert_ne!(zero.geometry_artifact_id(), m.geometry_artifact_id());
    }
}

#[test]
fn value_read_gate_reaches_qkv_at_its_own_and_earlier_layers() {
    use CausalLmProjectionSite::{Key, Query, Value};
    let ids = [2, 4, 6, 5];
    for mode in ROUTES {
        let m = placed(&load(), mode, 1.5);
        for layer in [0, 1] {
            for site in [Query, Key, Value] {
                let spec = CausalLmProjectionSpec {
                    layer,
                    site,
                    basis: CausalLmProjectionBasis::Dense,
                };
                let g = m.projection_contrast_pullback(&ids, 7, 8, spec).unwrap();
                let norm = g.weight.iter().map(|x| x * x).sum::<f64>().sqrt();
                assert!(norm > 1e-8);
                let value = |step: f64| {
                    let weights: Vec<_> =
                        g.weight.iter().map(|v| (v * step / norm) as f32).collect();
                    m.projection_perturbed_contrast(&ids, 7, 8, spec, &[0.; 4], &weights)
                        .unwrap()
                        .target_minus_alternative
                };
                let h = 0.003;
                let actual = (value(h) - value(-h)) / 2.;
                assert!(
                    (actual - h * norm).abs() < 9e-7 + 0.02 * h * norm,
                    "{mode:?} {layer} {site:?}: {actual} vs {}",
                    h * norm
                );
            }
        }
    }
}

#[test]
fn value_read_gate_keeps_both_explicit_inputs_and_calibration_owners_separate() {
    let base = load();
    let path = "transformer.h.1.mlp";
    let ids = vec![2, 3, 4];
    let mut anchors = Vec::new();
    for route in ROUTES {
        let m = placed(&base, route, 1.);
        let rt = m.runtime_at(path).unwrap();
        let x = [1., 2., 3., 4.];
        let v = [2., -3., 4., 0.];
        let dy = [1., -2., 0.5, 3.];
        assert!(rt.forward_sequence(path, &x, [1, 1]).is_err());
        assert!(rt
            .forward_value_read_sequence(path, &x, &v[..3], [1, 1])
            .is_err());
        let b = rt
            .forward_value_read_sequence(path, &x, &v, [1, 1])
            .unwrap();
        assert_eq!(b.vjp_input(&dy).unwrap().data(), dy);
        assert_eq!(
            b.vjp_value_source(&dy).unwrap().data(),
            [0.5, 1., 0.5, -4.5]
        );
        assert_eq!(b.vjp_parameters(&dy).unwrap()["gate"], [1., 3., 1., 0.]);
        let metric = m
            .calibrate_prefix_residual_metric(std::slice::from_ref(&ids))
            .unwrap();
        let report = serde_json::to_value(metric).unwrap();
        anchors.push(report["id"].clone());
    }
    assert!(anchors[0] != anchors[1] && anchors[0] != anchors[2] && anchors[1] != anchors[2]);
}
