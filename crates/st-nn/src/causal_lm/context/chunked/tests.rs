use super::*;
use crate::causal_lm::tests::{artifact, load};
use crate::geometry_artifact::{
    AttentionValueRoute, FractionalRecipe, GeometryAdapterRecipe, GeometryAdapterTensor,
    GeometryAdapterTensorRole, PrefixRotation,
};
use st_tensor::{TensorError, Tile};

fn placed(model: &NativeCausalLm, path: &str, recipe: GeometryAdapterRecipe) -> NativeCausalLm {
    let mut record = artifact(model, path, 0.3).record().clone();
    let p = &mut record.placements[0];
    if matches!(recipe, GeometryAdapterRecipe::Fractional { .. }) {
        p.tensors.retain(|t| t.name == "gate");
        p.tensors.push(GeometryAdapterTensor {
            name: "log_alpha".into(),
            shape: vec![],
            values: vec![-0.2],
            requires_grad: false,
            role: GeometryAdapterTensorRole::Parameter,
        });
    }
    p.recipe = recipe;
    model
        .with_geometry(GeometryAdapterArtifact::create(record).unwrap())
        .unwrap()
}

fn assert_words(a: &[f32], b: &[f32], case: &str) {
    assert_eq!(a.len(), b.len(), "{case}");
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "{case} at {i}");
    }
}

fn assert_same_context(a: &NativeCausalLmContext, b: &NativeCausalLmContext, case: &str) {
    assert_eq!(a.token_ids(), b.token_ids(), "{case}");
    assert_eq!(a.base_model_identity(), b.base_model_identity(), "{case}");
    assert_eq!(a.geometry_artifact_id(), b.geometry_artifact_id(), "{case}");
    assert_eq!(a.lora_artifact_id(), b.lora_artifact_id(), "{case}");
    assert_eq!(a.cache_values(), b.cache_values(), "{case}");
    assert_words(a.logit_values(), b.logit_values(), case);
    assert_eq!(a.blocks.len(), b.blocks.len(), "{case}");
    for (a, b) in a.blocks.iter().zip(&b.blocks) {
        assert_words(&a.key, &b.key, case);
        assert_words(&a.value, &b.value, case);
        for (a, b) in a.history.iter().zip(&b.history) {
            match (a, b) {
                (None, None) => {}
                (Some(GeometryHistory::Replay(a)), Some(GeometryHistory::Replay(b))) => {
                    assert_eq!(a.shape(), b.shape(), "{case}");
                    assert_words(a.data(), b.data(), case);
                }
                (Some(GeometryHistory::Transport(a)), Some(GeometryHistory::Transport(b))) => {
                    assert_eq!(a.cache_values(), b.cache_values(), "{case}");
                }
                _ => panic!("different geometry history: {case}"),
            }
        }
    }
}

#[test]
fn chunked_schedules_preserve_outputs_cache_and_forks() {
    let ids = [2, 4, 1, 8, 1, 5];
    let base = load();
    let lag = base
        .with_key_lag(CausalLmKeyLag { layer: 1, lag: 2 })
        .unwrap();
    let lora = base.with_lora(1, 2, 4.).unwrap();
    let lora = lora
        .step_lora_sgd(
            &lora.lora_loss_gradients(&ids, [1, ids.len()]).unwrap(),
            0.3,
        )
        .unwrap();
    let transport = || GeometryAdapterRecipe::DifferencePrefixTransport {
        features: 4,
        strength: 0.5,
        max_values: 64,
        rotation: PrefixRotation::QuarterTurn,
    };
    let replay = || GeometryAdapterRecipe::Fractional {
        features: 4,
        strength: 0.5,
        kernel: FractionalRecipe {
            kernel_len: 4,
            step: 0.8,
            max_values: 64,
            max_products: 256,
        },
    };
    let value = || GeometryAdapterRecipe::AttentionValueGate {
        features: 4,
        strength: 0.5,
        max_values: 64,
        route: AttentionValueRoute::Fixed { position: 2 },
    };
    let affine = |path| base.with_geometry(artifact(&base, path, 0.3)).unwrap();
    let cases = [
        ("base", base.clone(), false),
        ("last key lag", lag.clone(), false),
        ("learned last LoRA", lora, false),
        ("early affine", affine("transformer.h.0.attn"), false),
        (
            "early transport",
            placed(&base, "transformer.h.0", transport()),
            false,
        ),
        (
            "early replay",
            placed(&base, "transformer.h.0.attn", replay()),
            false,
        ),
        (
            "early value read",
            placed(&base, "transformer.h.0.mlp", value()),
            false,
        ),
        ("last affine", affine("transformer.h.1.attn"), true),
        (
            "last transport",
            placed(&base, "transformer.h.1", transport()),
            true,
        ),
        (
            "last replay",
            placed(&base, "transformer.h.1.attn", replay()),
            true,
        ),
        (
            "last value read with lag",
            placed(&lag, "transformer.h.1.mlp", value()),
            true,
        ),
    ];
    for (name, model, fallback) in cases {
        let eager = NativeCausalLmContext::new(model.clone(), ids.to_vec(), 1024).unwrap();
        for cap in 1..=ids.len() {
            let case = format!("{name}, chunk cap {cap}");
            decoder::work_counts::take();
            decoder::work_counts::take_cache_rows();
            let reference = NativeCausalLmContext::new_chunked_prefill(
                model.clone(),
                ids.to_vec(),
                1024,
                cap,
                1024,
            )
            .unwrap();
            assert_eq!(decoder::work_counts::take(), [0, 0, 1], "{case}");
            assert_eq!(decoder::work_counts::take_cache_rows(), [12, 12], "{case}");
            let optimized = NativeCausalLmContext::new_chunked_prefill_last_output(
                model.clone(),
                ids.to_vec(),
                1024,
                cap,
                1024,
            )
            .unwrap();
            assert_eq!(decoder::work_counts::take(), [0, 0, 1], "{case}");
            assert_eq!(
                decoder::work_counts::take_cache_rows(),
                if fallback { [12, 12] } else { [7, 7] },
                "{case}"
            );
            assert_same_context(&reference, &optimized, &case);
            assert!(
                reference
                    .logit_values()
                    .iter()
                    .zip(eager.logit_values())
                    .all(|(&a, &b)| (a - b).abs() < 5e-6),
                "{case}: changed canonical prefill"
            );
            for suffix in [[0, 3], [6, 9]] {
                assert_same_context(
                    &reference.with_suffix(&suffix).unwrap(),
                    &optimized.with_suffix(&suffix).unwrap(),
                    &case,
                );
            }
            assert_same_context(&reference, &optimized, &case);
        }
    }
}

#[test]
fn chunked_planner_bounds_every_chunk_before_execution() {
    let mut model = load();
    model.limits.max_attention_pairs = 32;
    let ids = vec![1, 4, 2, 6];
    assert!(NativeCausalLmContext::new(model.clone(), ids.clone(), 64).is_err());
    assert_eq!(
        model.context_chunked_prefill_plan(4, 64, 4, 48).unwrap(),
        CausalLmChunkedPrefillPlan {
            chunk_steps: 2,
            chunks: 2,
            attention_pairs: 48,
        },
    );
    assert_eq!(
        model
            .context_chunked_prefill_plan(4, 64, 1, 40)
            .unwrap()
            .attention_pairs,
        40
    );
    assert_eq!(
        model
            .context_chunked_prefill_plan(3, 64, 2, 28)
            .unwrap()
            .attention_pairs,
        28
    );
    let context = NativeCausalLmContext::new_chunked_prefill_last_output(
        model.clone(),
        ids.clone(),
        64,
        4,
        48,
    )
    .unwrap();
    assert_eq!(context.cache_values(), 64);
    assert_words(
        context.logit_values(),
        load().next_token_logits(&ids).unwrap().data(),
        "over square cap",
    );

    decoder::work_counts::take();
    decoder::work_counts::take_cache_rows();
    for bad in [
        model.context_chunked_prefill_plan(0, 64, 4, 48),
        model.context_chunked_prefill_plan(4, 63, 4, 48),
        model.context_chunked_prefill_plan(4, 64, 0, 48),
        model.context_chunked_prefill_plan(4, 64, 4, 47),
        model.context_chunked_prefill_plan(usize::MAX, 64, 4, usize::MAX),
    ] {
        assert!(bad.is_err());
    }
    let make = NativeCausalLmContext::new_chunked_prefill_last_output;
    assert!(make(model.clone(), ids, 64, 4, 47).is_err());
    assert!(make(model, vec![1, 4, 2, 11], 64, 4, 48).is_err());
    assert_eq!(decoder::work_counts::take_cache_rows(), [0, 0]);
    assert_eq!(decoder::work_counts::take(), [0, 0, 0]);
}

#[test]
fn chunked_pruning_still_rejects_nonfinite_projected_queries() {
    let mut model = load();
    let last = model.blocks.last_mut().unwrap();
    last.first_norm.gamma = Tensor::from_vec(1, 4, vec![0.; 4]).unwrap();
    last.first_norm.beta = Tensor::from_vec(1, 4, vec![2.; 4]).unwrap();
    let mut weight = vec![0.; 48];
    weight[0] = f32::MAX;
    last.qkv.weight =
        PackedB::from_tensor(&Tensor::from_vec(4, 12, weight).unwrap(), Tile::col_major()).unwrap();
    for chunk in [1, 3] {
        for last_output in [false, true] {
            let error = NativeCausalLmContext::chunked_prefill(
                model.clone(),
                vec![1, 2, 3],
                64,
                chunk,
                64,
                last_output,
            )
            .unwrap_err();
            assert!(
                matches!(
                    error.downcast_ref::<TensorError>(),
                    Some(TensorError::NonFiniteValue {
                        label: "matmul_prepacked_bias_output",
                        value,
                    }) if !value.is_finite()
                ),
                "chunk {chunk}, last output {last_output}: {error}",
            );
        }
    }
}
