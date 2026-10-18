use super::*;
use crate::weights::WeightLoadLimits;
use safetensors::tensor::{serialize, Dtype, TensorView};
use serde_json::json;
use std::collections::BTreeMap;

#[path = "context_tests.rs"]
mod context;
#[path = "evaluation_tests.rs"]
mod evaluation;
#[path = "generation_tests.rs"]
mod generation;
#[path = "learning_tests.rs"]
mod learning;

fn config() -> serde_json::Value {
    json!({"model_type":"gpt2","vocab_size":11,"n_positions":8,"n_embd":4,"n_head":2,"n_layer":2})
}

fn parameters() -> BTreeMap<String, (Vec<usize>, Vec<f32>)> {
    let mut weights = BTreeMap::new();
    let mut add = |name: String, shape: Vec<usize>| {
        let count = shape.iter().product();
        let offset = weights.len();
        let values = (0..count)
            .map(|i| {
                if name.contains("ln_") && name.ends_with("weight") {
                    1.
                } else {
                    ((i * 7 + offset * 3) as f32 * 0.23).sin() * 0.15
                }
            })
            .collect();
        weights.insert(name, (shape, values));
    };
    add("wte.weight".into(), vec![11, 4]);
    add("wpe.weight".into(), vec![8, 4]);
    for i in 0..2 {
        for norm in ["ln_1", "ln_2"] {
            for suffix in ["weight", "bias"] {
                add(format!("h.{i}.{norm}.{suffix}"), vec![4]);
            }
        }
        for (name, input, output) in [
            ("attn.c_attn", 4, 12),
            ("attn.c_proj", 4, 4),
            ("mlp.c_fc", 4, 16),
            ("mlp.c_proj", 16, 4),
        ] {
            add(format!("h.{i}.{name}.weight"), vec![input, output]);
            add(format!("h.{i}.{name}.bias"), vec![output]);
        }
    }
    add("ln_f.weight".into(), vec![4]);
    add("ln_f.bias".into(), vec![4]);
    weights
}

fn archive(weights: &BTreeMap<String, (Vec<usize>, Vec<f32>)>) -> SafeTensorArchive {
    let bytes: BTreeMap<_, Vec<u8>> = weights
        .iter()
        .map(|(name, (_, values))| (name, values.iter().flat_map(|v| v.to_le_bytes()).collect()))
        .collect();
    let views: Vec<_> = weights
        .iter()
        .map(|(name, (shape, _))| {
            (
                name.as_str(),
                TensorView::new(Dtype::F32, shape.clone(), &bytes[name]).unwrap(),
            )
        })
        .collect();
    SafeTensorArchive::from_owned_bytes(
        serialize(views, None).unwrap(),
        WeightLoadLimits::default(),
    )
    .unwrap()
}

fn load() -> NativeCausalLm {
    NativeCausalLm::from_gpt2_json(
        &serde_json::to_vec(&config()).unwrap(),
        &archive(&parameters()),
        Default::default(),
    )
    .unwrap()
}

fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert!((a - b).abs() < 2e-6, "{a} {b}");
    }
}

#[test]
fn complete_decoder_preserves_batch_causality_and_last_projection() {
    let model = load();
    let all = model.forward(&[1, 2, 3, 4, 5, 6], [2, 3], false).unwrap();
    assert_eq!(all.shape(), [2, 3, 11]);
    let values = all.read_values().unwrap();
    for (index, ids) in [[1, 2, 3], [4, 5, 6]].iter().enumerate() {
        close(
            &values[index * 33..(index + 1) * 33],
            &model
                .forward(ids, [1, 3], false)
                .unwrap()
                .read_values()
                .unwrap(),
        );
    }
    let changed = model
        .forward(&[1, 2, 9, 4, 5, 10], [2, 3], false)
        .unwrap()
        .read_values()
        .unwrap();
    close(&values[..22], &changed[..22]);
    close(&values[33..55], &changed[33..55]);
    close(
        &values[..22],
        &model
            .forward(&[1, 2], [1, 2], false)
            .unwrap()
            .read_values()
            .unwrap(),
    );
    let last = model.forward(&[1, 2, 3, 4, 5, 6], [2, 3], true).unwrap();
    assert_eq!(last.shape(), [2, 1, 11]);
    close(
        &last.read_values().unwrap(),
        &[&values[22..33], &values[55..66]].concat(),
    );
    assert_eq!(
        model
            .next_token_ids(&[1, 2, 3, 4, 5, 6], [2, 3])
            .unwrap()
            .len(),
        2
    );
}

#[test]
fn loss_shifts_within_rows_and_supports_final_label_beyond_position_capacity() {
    let model = load();
    let loss = model.next_token_loss(&[1, 2, 3, 4, 5, 6], [2, 3]).unwrap();
    let logits = model.logits(&[1, 2, 4, 5], [2, 2], false).unwrap();
    let expected = logits
        .cross_entropy_with_logits(&[2, 3, 5, 6], Default::default())
        .unwrap()
        .data()[0];
    assert_eq!(loss, expected);
    assert!(model.next_token_loss(&[1; 9], [1, 9]).unwrap().is_finite());
    assert!(model.next_token_loss(&[1; 10], [1, 10]).is_err());
    assert!(model.next_token_loss(&[1, 11], [1, 2]).is_err());
    assert!(model.next_token_loss(&[1], [1, 1]).is_err());
}

#[test]
fn greedy_rejects_finite_weight_overflow_and_breaks_ties_at_lowest_id() {
    let mut c = config();
    c["tie_word_embeddings"] = json!(false);
    let mut p = parameters();
    p.get_mut("ln_f.weight").unwrap().1.fill(0.);
    p.get_mut("ln_f.bias").unwrap().1.fill(2.);
    p.insert("lm_head.weight".into(), (vec![11, 4], vec![f32::MAX; 44]));
    let model = NativeCausalLm::from_gpt2_json(
        &serde_json::to_vec(&c).unwrap(),
        &archive(&p),
        Default::default(),
    )
    .unwrap();
    for error in [
        model.forward(&[1, 2], [1, 2], false).unwrap_err(),
        model.next_token_ids(&[1, 2], [1, 2]).unwrap_err(),
        model.next_token_loss(&[1, 2], [1, 2]).unwrap_err(),
    ] {
        assert!(error.to_string().contains("matmul_prepacked_output"));
    }
    p.get_mut("lm_head.weight").unwrap().1.fill(0.);
    let model = NativeCausalLm::from_gpt2_json(
        &serde_json::to_vec(&c).unwrap(),
        &archive(&p),
        Default::default(),
    )
    .unwrap();
    assert_eq!(model.next_token_ids(&[1, 2], [2, 1]).unwrap(), [0, 0]);
}

#[test]
fn shape_token_work_and_logit_budgets_fail_before_execution() {
    let model = load();
    for shape in [[0, 1], [1, 0], [1, 9], [usize::MAX, 2]] {
        assert!(model.forward(&[], shape, false).is_err());
    }
    assert!(model.forward(&[11], [1, 1], false).is_err());
    assert!(model.forward(&[], [1, 1], false).is_err());
    let weights = archive(&parameters());
    for limits in [
        CausalLmLimits {
            max_parameter_elements: 1,
            ..Default::default()
        },
        CausalLmLimits {
            max_attention_pairs: 0,
            ..Default::default()
        },
    ] {
        assert!(NativeCausalLm::from_gpt2_json(
            &serde_json::to_vec(&config()).unwrap(),
            &weights,
            limits
        )
        .is_err());
    }
    let model = NativeCausalLm::from_gpt2_json(
        &serde_json::to_vec(&config()).unwrap(),
        &weights,
        CausalLmLimits {
            max_logit_elements: 11,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(model.forward(&[1, 2], [1, 2], false).is_err());
    assert!(model.forward(&[1, 2], [1, 2], true).is_ok());
    let model = NativeCausalLm::from_gpt2_json(
        &serde_json::to_vec(&config()).unwrap(),
        &weights,
        CausalLmLimits {
            max_attention_pairs: 15,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(model.forward(&[1, 2], [1, 2], true).is_err()); // 2 layers * 2 heads * 2 * 2
}

#[test]
fn rejects_semantic_mismatch_missing_extra_and_wrong_shapes() {
    let weights = archive(&parameters());
    for (key, value) in [
        ("model_type", json!("llama")),
        ("activation_function", json!("relu")),
        ("add_cross_attention", json!(true)),
        ("reorder_and_upcast_attn", json!(true)),
        ("pruned_heads", json!({"0":[0]})),
        ("n_head", json!(3)),
        ("n_layer", json!(0)),
        ("layer_norm_epsilon", json!(0)),
        ("n_inner", json!(usize::MAX)),
    ] {
        let mut c = config();
        c[key] = value;
        assert!(
            NativeCausalLm::from_gpt2_json(
                &serde_json::to_vec(&c).unwrap(),
                &weights,
                Default::default()
            )
            .is_err(),
            "{key}"
        );
    }
    for kind in ["missing", "extra", "shape", "nan", "ambiguous"] {
        let mut p = parameters();
        match kind {
            "missing" => {
                p.remove("h.1.ln_1.weight");
            }
            "extra" => {
                p.insert("lora.weight".into(), (vec![1], vec![0.]));
            }
            "shape" => {
                p.get_mut("h.0.attn.c_attn.weight").unwrap().0 = vec![12, 4];
            }
            "nan" => {
                p.get_mut("wte.weight").unwrap().1[0] = f32::NAN;
            }
            _ => {
                p.insert("transformer.wte.weight".into(), p["wte.weight"].clone());
            }
        }
        assert!(
            NativeCausalLm::from_gpt2_json(
                &serde_json::to_vec(&config()).unwrap(),
                &archive(&p),
                Default::default()
            )
            .is_err(),
            "{kind}"
        );
    }
}

#[test]
fn prefixed_weights_tied_and_untied_heads_use_explicit_contract() {
    let p = parameters();
    let prefixed = p
        .iter()
        .map(|(name, v)| (format!("transformer.{name}"), v.clone()))
        .collect();
    let a = load();
    let b = NativeCausalLm::from_gpt2_json(
        &serde_json::to_vec(&config()).unwrap(),
        &archive(&prefixed),
        Default::default(),
    )
    .unwrap();
    close(
        &a.logits(&[1, 2], [1, 2], false).unwrap().data(),
        &b.logits(&[1, 2], [1, 2], false).unwrap().data(),
    );
    let mut p = p;
    p.insert("lm_head.weight".into(), p["wte.weight"].clone());
    assert!(NativeCausalLm::from_gpt2_json(
        &serde_json::to_vec(&config()).unwrap(),
        &archive(&p),
        Default::default()
    )
    .is_ok());
    p.get_mut("lm_head.weight").unwrap().1[0] += 0.5;
    assert!(NativeCausalLm::from_gpt2_json(
        &serde_json::to_vec(&config()).unwrap(),
        &archive(&p),
        Default::default()
    )
    .is_err());
    let mut c = config();
    c["tie_word_embeddings"] = json!(false);
    assert!(NativeCausalLm::from_gpt2_json(
        &serde_json::to_vec(&c).unwrap(),
        &archive(&p),
        Default::default()
    )
    .is_ok());
}

#[test]
fn validates_redundant_legacy_causal_masks_instead_of_ignoring_them() {
    let mut p = parameters();
    let mask: Vec<_> = (0..64)
        .map(|i| if i % 8 <= i / 8 { 1. } else { 0. })
        .collect();
    p.insert("h.0.attn.bias".into(), (vec![1, 1, 8, 8], mask));
    assert!(NativeCausalLm::from_gpt2_json(
        &serde_json::to_vec(&config()).unwrap(),
        &archive(&p),
        Default::default()
    )
    .is_ok());
    p.get_mut("h.0.attn.bias").unwrap().1[1] = 1.;
    assert!(NativeCausalLm::from_gpt2_json(
        &serde_json::to_vec(&config()).unwrap(),
        &archive(&p),
        Default::default()
    )
    .is_err());
}

fn artifact(model: &NativeCausalLm, path: &str, gate: f32) -> GeometryAdapterArtifact {
    GeometryAdapterArtifact::create_json(&serde_json::to_vec(&json!({
        "schema":"spiraltorch.geometry_adapter_artifact.v1","base_model_identity":model.base_model_identity(),
        "policy":"full_prefix_f32_no_cache.v1","training":false,"placements":[{
            "path":path,"training":false,"recipe":{"schema":"spiraltorch.channel_affine_adapter.v1","features":4,"strength":0.5,"max_values":1024},
            "tensors":[{"name":"gate","shape":[4],"values":[gate,0.,-gate,gate],"role":"parameter","requires_grad":false},
                {"name":"bias","shape":[4],"values":[0.,0.,0.,0.],"role":"parameter","requires_grad":false}]
        }]
    })).unwrap()).unwrap()
}

#[test]
fn frozen_geometry_is_identity_bound_and_affects_downstream_logits() {
    let model = load();
    let reference = model.logits(&[1, 2, 3], [1, 3], false).unwrap();
    for path in [
        "transformer.h.0.attn",
        "transformer.h.1.mlp",
        "transformer.h.0",
    ] {
        let identity = model.with_geometry(artifact(&model, path, 0.)).unwrap();
        assert_eq!(
            reference.data(),
            identity.logits(&[1, 2, 3], [1, 3], false).unwrap().data()
        );
        let changed = model.with_geometry(artifact(&model, path, 0.5)).unwrap();
        assert!(reference
            .data()
            .iter()
            .zip(changed.logits(&[1, 2, 3], [1, 3], false).unwrap().data())
            .any(|(a, b)| (a - b).abs() > 1e-5));
        assert!(changed.geometry_artifact_id().is_some());
    }
    assert!(model.geometry_artifact_id().is_none());
    assert!(model
        .with_geometry(artifact(&model, "transformer.h.99.mlp", 0.1))
        .is_err());
    let mut p = parameters();
    p.get_mut("wte.weight").unwrap().1[0] += 0.1;
    let other = NativeCausalLm::from_gpt2_json(
        &serde_json::to_vec(&config()).unwrap(),
        &archive(&p),
        Default::default(),
    )
    .unwrap();
    assert!(other
        .with_geometry(artifact(&model, "transformer.h.0", 0.1))
        .is_err());
}
