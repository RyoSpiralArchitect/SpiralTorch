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
