use super::*;
use crate::geometry_artifact::{
    GeometryAdapterRecipe,
    PrefixRotation::{self, *},
};

fn placed(
    base: &NativeCausalLm,
    path: &str,
    turn: PrefixRotation,
    gate: f32,
    max_values: usize,
) -> NativeCausalLm {
    let mut r = artifact(base, path, gate).record().clone();
    r.training = true;
    let p = &mut r.placements[0];
    p.training = true;
    p.recipe = GeometryAdapterRecipe::PrefixTransport {
        features: 4,
        strength: 0.5,
        max_values,
        rotation: turn,
    };
    for t in &mut p.tensors {
        t.requires_grad = true;
    }
    base.with_geometry(GeometryAdapterArtifact::create(r).unwrap())
        .unwrap()
}

#[test]
fn prefix_transport_learns_restores_and_scores_suffixes_without_growing_geometry_state() {
    let base = load();
    for site in [
        "transformer.h.0.attn",
        "transformer.h.1.mlp",
        "transformer.h.0",
    ] {
        for turn in [Identity, QuarterTurn, InverseQuarterTurn] {
            let m = placed(&base, site, turn, 0., 64);
            let prompt = [2, 4, 6];
            let shape = [1, 3];
            let score =
                |m: &NativeCausalLm| m.logits(&prompt, shape, true).unwrap().data().to_vec();
            assert_eq!(score(&base), score(&m));
            let g = m
                .geometry_token_menu_gradients(&prompt, &[7, 8], 0, 1.)
                .unwrap();
            let (learned, _) = m
                .step_geometry_cohort_gradient_at_distance(&[&g], 1. / 256.)
                .unwrap();
            assert_ne!(score(&learned), score(&m));
            assert!(learned
                .step_geometry_cohort_gradient_at_distance(&[&g], 1. / 256.)
                .is_err());
            let a = learned.geometry_artifact().unwrap();
            let restored = base
                .with_geometry(
                    GeometryAdapterArtifact::from_json(
                        a.to_json().as_bytes(),
                        a.artifact_id(),
                        base.base_model_identity(),
                    )
                    .unwrap(),
                )
                .unwrap();
            assert_eq!(score(&restored), score(&learned));
            let root =
                NativeCausalLmContext::new(restored.clone(), prompt.to_vec(), 2 * 2 * 6 * 4 + 8)
                    .unwrap();
            let exact = root.logit_values().to_vec();
            close(&exact, &score(&restored));
            assert_eq!(root.cache_values(), 2 * 2 * 3 * 4 + 8);
            for suffix in [[3, 1, 8], [7, 4, 2]] {
                let branch = root.with_suffix(&suffix).unwrap();
                let tokens = [prompt, suffix].concat();
                let full = restored.logits(&tokens, [1, 6], true).unwrap();
                close(branch.logit_values(), full.data());
                let mut single = root.clone();
                for id in suffix {
                    single.append(id).unwrap();
                }
                close(single.logit_values(), full.data());
                assert_eq!(branch.cache_values(), 2 * 2 * 6 * 4 + 8);
                assert_eq!(
                    branch.cache_values(),
                    restored
                        .context_cache_values(6, branch.cache_values())
                        .unwrap()
                );
                assert!(restored
                    .context_cache_values(6, branch.cache_values() - 1)
                    .is_err());
                assert!(branch.with_suffix(&[0]).is_err());
            }
            assert_eq!(root.token_ids(), prompt);
            assert_eq!(root.logit_values(), exact);
        }
    }
}

#[test]
fn prefix_signal_factors_real_decoder_gradients_without_changing_model() {
    let base = load();
    let site = "transformer.h.1.mlp";
    let ids = [2, 4, 6, 5, 7];
    for turn in [Identity, QuarterTurn, InverseQuarterTurn] {
        let m = placed(&base, site, turn, 0.25, 64);
        let before = m.geometry_artifact().unwrap().artifact_id().to_owned();
        let trace = m.last_mlp_prefix_margin_signal(&ids, 7, 8).unwrap();
        assert_eq!(trace.contrast, m.next_token_contrast(&ids, 7, 8).unwrap());
        assert_eq!(trace.geometry_artifact_id, before);
        assert_eq!(trace.shape, [5, 4]);
        let gs = [0, 1].map(|gold| {
            m.geometry_token_menu_gradients(&ids, &[7, 8], gold, 1.)
                .unwrap()
        });
        for j in 0..4 {
            let reconstructed = trace
                .source_features
                .chunks_exact(4)
                .map(|r| r[j])
                .sum::<f64>()
                * f64::from(trace.bias_gradient[j]);
            let ordinary =
                gs[1].parameters()[site]["gate"][j] - gs[0].parameters()[site]["gate"][j];
            assert!((reconstructed - f64::from(trace.gate_gradient[j])).abs() < 2e-6);
            assert!((ordinary - trace.gate_gradient[j]).abs() < 2e-6);
            assert_eq!(trace.bias_gradient[j], trace.upstream[j] * trace.strength);
        }
        let swapped = m.last_mlp_prefix_margin_signal(&ids, 8, 7).unwrap();
        assert_eq!(swapped.site_input, trace.site_input);
        close(
            &swapped.gate_gradient,
            &trace.gate_gradient.iter().map(|x| -x).collect::<Vec<_>>(),
        );
        assert_eq!(m.geometry_artifact().unwrap().artifact_id(), before);
        assert!(m.last_mlp_prefix_margin_signal(&ids, 7, 7).is_err());
        assert!(m
            .last_mlp_prefix_margin_signal(&[usize::MAX], 7, 8)
            .is_err());
        assert!(m.prefix_margin_signal_input_count(0, 7, 8).is_err());
        assert!(m
            .prefix_margin_signal_input_count(usize::MAX, 7, 8)
            .is_err());
    }
    assert!(base.prefix_margin_signal_input_count(1, 7, 8).is_err());
    assert!(placed(&base, "transformer.h.0", Identity, 0., 64)
        .prefix_margin_signal_input_count(1, 7, 8)
        .is_err());
    assert!(placed(&base, site, Identity, 0., 4)
        .prefix_margin_signal_input_count(2, 7, 8)
        .is_err());
    let one = placed(&base, site, Identity, 0., 4)
        .last_mlp_prefix_margin_signal(&ids[..1], 7, 8)
        .unwrap();
    assert_eq!(
        one.source_features,
        one.site_input
            .iter()
            .copied()
            .map(f64::from)
            .collect::<Vec<_>>()
    );
}

#[test]
fn prefix_transport_cache_still_respects_full_geometry_budget() {
    let m = placed(&load(), "transformer.h.0", QuarterTurn, 0.3, 12);
    let c = NativeCausalLmContext::new(m.clone(), vec![2, 5], 1024)
        .unwrap()
        .with_suffix(&[1])
        .unwrap();
    assert!(c.with_suffix(&[4]).is_err());
    assert!(m.forward(&[2, 5, 1, 4], [1, 4], true).is_err());
    assert_eq!(c.token_ids(), [2, 5, 1]);
}

#[test]
fn mixed_transport_preserves_calibration_norm_not_gradient_orientation_or_cache_owners() {
    use crate::geometry_artifact::PrefixChannelMix::*;
    let base = load();
    let site = "transformer.h.1.mlp";
    let ids = vec![2, 4, 6];
    let mut rec = placed(&base, site, QuarterTurn, 0., 64)
        .geometry_artifact()
        .unwrap()
        .record()
        .clone();
    for t in &mut rec.placements[0].tensors {
        t.values.fill(0.);
        t.requires_grad = t.name == "gate";
    }
    let legacy = base
        .with_geometry(GeometryAdapterArtifact::create(rec.clone()).unwrap())
        .unwrap();
    let prompts = vec![ids.clone(), vec![4, 2, 5]];
    let ref_metric = legacy.calibrate_prefix_residual_metric(&prompts).unwrap();
    let ref_trace = legacy.last_mlp_prefix_margin_signal(&ids, 7, 8).unwrap();
    let old = serde_json::to_value(&ref_metric).unwrap();
    for mix in [CayleyTriples, InverseCayleyTriples] {
        rec.placements[0].recipe = GeometryAdapterRecipe::MixedPrefixTransport {
            features: 4,
            strength: 0.5,
            max_values: 64,
            rotation: QuarterTurn,
            output_mix: mix,
        };
        let m = base
            .with_geometry(GeometryAdapterArtifact::create(rec.clone()).unwrap())
            .unwrap();
        let trace = m.last_mlp_prefix_margin_signal(&ids, 7, 8).unwrap();
        assert_eq!(trace.output_mix, Some(mix));
        assert_eq!(trace.contrast, ref_trace.contrast);
        assert_eq!(trace.upstream, ref_trace.upstream);
        let metric = m.calibrate_prefix_residual_metric(&prompts).unwrap();
        let now = serde_json::to_value(&metric).unwrap();
        assert_ne!(old["id"], now["id"]);
        assert_eq!(old["second_moments"], now["second_moments"]);
        let g = m
            .geometry_token_menu_gradients(&ids, &[7, 8], 0, 1.)
            .unwrap();
        assert!(m
            .step_prefix_residual_rms(&ref_metric, &[&g], PrefixMetricRule::Diagonal, 0.001, 1000.)
            .is_err());
        let (end, r) = m
            .step_prefix_residual_rms(&metric, &[&g], PrefixMetricRule::Diagonal, 0.001, 1000.)
            .unwrap();
        assert!((r.realized_rms - 0.001).abs() < 1e-7);
        let saved = end.geometry_artifact().unwrap();
        let restored = base
            .with_geometry(
                GeometryAdapterArtifact::from_json(
                    saved.to_json().as_bytes(),
                    saved.artifact_id(),
                    base.base_model_identity(),
                )
                .unwrap(),
            )
            .unwrap();
        let ctx = NativeCausalLmContext::new(restored, ids.clone(), 1024).unwrap();
        close(
            ctx.with_suffix(&[5, 1]).unwrap().logit_values(),
            end.logits(&[2, 4, 6, 5, 1], [1, 5], true).unwrap().data(),
        );
        assert!(end
            .step_prefix_residual_rms(&metric, &[&g], PrefixMetricRule::Diagonal, 0.001, 1000.)
            .is_err());
    }
}
