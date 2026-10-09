use super::*;
use serde_json::json;

fn checkpoint() -> ByteDecoderCheckpoint {
    let plan = super::super::tests::plan(AttentionMask::Causal { query_offset: 0 }, 2).unwrap();
    let geometry = super::super::tests::geometry_for(&plan);
    plan.with_causal_geometry(geometry)
        .unwrap()
        .initial_checkpoint()
}

#[test]
fn flat_metric_checkpoint_is_explicit_and_keeps_the_same_parameter_owner() {
    let old = checkpoint();
    let mut flat = old.clone();
    flat.plan.geometry = flat
        .plan
        .geometry
        .take()
        .map(|g| g.with_pair_metric(ByteDecoderPairMetric::EuclideanChordSquared));
    let payload = flat.to_json().unwrap();
    let value: serde_json::Value = serde_json::from_str(&payload).unwrap();
    assert_eq!(value["schema"], BYTE_DECODER_METRIC_CHECKPOINT_SCHEMA);
    assert_eq!(
        value["model"]["geometry"]["pair_metric"],
        "euclidean_chord_squared.v1"
    );
    assert_eq!(old.plan.parameter_layout(), flat.plan.parameter_layout());
    let restored = ByteDecoderCheckpoint::from_json(&payload).unwrap();
    assert_eq!(restored.to_json().unwrap(), payload);
    assert_eq!(
        restored.plan.geometry.unwrap().pair_metric(),
        ByteDecoderPairMetric::EuclideanChordSquared
    );
    for variant in 0..5 {
        let mut bad = value.clone();
        match variant {
            0 => bad["schema"] = json!(BYTE_DECODER_CHECKPOINT_SCHEMA),
            1 => bad["model"]["geometry"]["pair_metric"] = json!("unknown"),
            2 => bad["model"]["geometry"]["pair_metric"] = serde_json::Value::Null,
            3 => {
                bad["model"]["geometry"]
                    .as_object_mut()
                    .unwrap()
                    .remove("pair_metric");
            }
            _ => bad["model"]["geometry"]["pair_metric"] = json!("poincare_squared.v1"),
        }
        assert!(ByteDecoderCheckpoint::from_json(&bad.to_string()).is_err());
    }
    let old_json = old.to_json().unwrap();
    assert!(!old_json.contains("pair_metric"));
    assert!(old_json.contains(BYTE_DECODER_CHECKPOINT_SCHEMA));
    assert_eq!(
        ByteDecoderCheckpoint::from_json(&old_json)
            .unwrap()
            .to_json()
            .unwrap(),
        old_json
    );
}

#[test]
fn portable_checkpoint_roundtrip_preserves_full_geometry_and_parameter_order() {
    for mut checkpoint in [
        checkpoint(),
        super::super::tests::plan(AttentionMask::Causal { query_offset: 0 }, 1)
            .unwrap()
            .initial_checkpoint(),
    ] {
        for revision in [0, 1, (1u64 << 53) + 1, u64::MAX] {
            checkpoint.attempted_revision = revision;
            let json = checkpoint.to_json().unwrap();
            let restored = ByteDecoderCheckpoint::from_json(&json).unwrap();
            assert_eq!(restored.attempted_revision(), revision);
            assert_eq!(restored.to_json().unwrap(), json);
            assert_eq!(
                restored.plan().parameter_layout(),
                checkpoint.plan().parameter_layout()
            );
            assert_eq!(
                restored.plan().input_layout(),
                checkpoint.plan().input_layout()
            );
            assert_eq!(
                restored.plan().position_capacity(),
                checkpoint.plan().position_capacity()
            );
        }
    }
}

#[test]
fn portable_checkpoint_preserves_f32_bits_without_a_json_value_intermediate() {
    let mut checkpoint = checkpoint();
    let values = [
        0.,
        -0.,
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::MIN_POSITIVE,
        f32::from_bits(0x3f7fffff),
        f32::MAX,
    ];
    checkpoint.plan.token.values[..values.len()].copy_from_slice(&values);
    let restored = ByteDecoderCheckpoint::from_json(&checkpoint.to_json().unwrap()).unwrap();
    assert_eq!(
        restored.plan.token.values[..values.len()]
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        values.map(f32::to_bits)
    );
}

#[test]
fn malformed_checkpoints_reject_topology_semantics_and_noncanonical_clocks() {
    let payload = checkpoint().to_json().unwrap();
    let valid: serde_json::Value = serde_json::from_str(&payload).unwrap();
    for (pointer, value) in [
        ("/schema", json!("unknown")),
        ("/update_rule", json!("adam")),
        ("/window_state", json!("streaming")),
        ("/attempted_revision", json!(1)),
        ("/attempted_revision", json!("01")),
        ("/attempted_revision", json!("+1")),
        ("/attempted_revision", json!("-1")),
        ("/attempted_revision", json!("18446744073709551616")),
        ("/model/token/shape", json!([255, 4])),
        ("/model/position/shape", json!([u32::MAX, 4])),
        ("/model/token/values", json!([0.])),
        ("/model/blocks", json!([])),
        ("/model/blocks/0/heads", json!(0)),
        ("/model/blocks/0/heads", json!(3)),
        ("/model/blocks/0/qkv/parameters/0/role", json!("gain")),
        ("/model/blocks/0/qkv/stages/0/gelu", json!(true)),
        ("/model/blocks/0/output/input_shape", json!([3, 2, 4])),
        ("/model/geometry/curvature", json!(0.)),
        ("/model/geometry/raw_decay", json!([0.])),
        ("/model/geometry/raw_phase", json!([])),
        ("/model/geometry/raw_gains", json!([[0., 0.]])),
        ("/model/geometry/projection/input_shape", json!([1, 6, 4])),
        ("/model/head/parameters/1/values", json!([0.])),
    ] {
        let mut bad = valid.clone();
        *bad.pointer_mut(pointer).unwrap() = value;
        assert!(
            ByteDecoderCheckpoint::from_json(&bad.to_string()).is_err(),
            "{pointer}"
        );
    }
    for pointer in [
        "",
        "/model",
        "/model/blocks/0",
        "/model/geometry",
        "/model/token",
        "/model/head",
        "/model/head/parameters/0",
    ] {
        let mut bad = valid.clone();
        bad.pointer_mut(pointer)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert("unknown".into(), json!(true));
        assert!(
            ByteDecoderCheckpoint::from_json(&bad.to_string()).is_err(),
            "{pointer}"
        );
    }
    assert!(ByteDecoderCheckpoint::from_json_with_limit(&payload, payload.len() - 1).is_err());
    assert!(ByteDecoderCheckpoint::from_json(&payload.replace("0.1", "1e100")).is_err());
    assert!(ByteDecoderCheckpoint::from_json(
        &payload.replace("\"schema\":", "\"schema\":\"extra\",\"schema\":")
    )
    .is_err());
}

#[test]
fn checkpoint_import_does_not_drop_richer_attention_projection_operations() {
    let mut record: serde_json::Value =
        serde_json::from_str(&checkpoint().to_json().unwrap()).unwrap();
    record["model"]["blocks"][0]["qkv"]["stages"]
        .as_array_mut()
        .unwrap()
        .push(json!({
            "kind":"pointwise", "parameters":[], "steps":[{"op":"relu", "rhs":null}]
        }));
    assert!(ByteDecoderCheckpoint::from_json(&record.to_string()).is_err());
}

#[test]
fn all_graph_ranks_are_rejected_before_generic_layout_expansion() {
    let valid: serde_json::Value = serde_json::from_str(&checkpoint().to_json().unwrap()).unwrap();
    for pointer in [
        "/model/blocks/0/pre",
        "/model/blocks/0/qkv",
        "/model/blocks/0/output",
        "/model/blocks/0/feed_forward",
        "/model/geometry/projection",
        "/model/head",
    ] {
        let mut bad = valid.clone();
        bad.pointer_mut(pointer).unwrap()["input_shape"] = json!([1, 4]);
        assert!(
            matches!(
                ByteDecoderCheckpoint::from_json(&bad.to_string()),
                Err(InferenceError::ByteDecoder(
                    "checkpoint graph rank must be three"
                ))
            ),
            "{pointer}"
        );
    }
    let mut amplified = valid;
    let pre = &mut amplified["model"]["blocks"][0]["pre"];
    pre["input_shape"] = json!(vec![1; 200_000]);
    pre["parameters"] = json!([]);
    pre["stages"] = json!(vec![
        json!({"kind":"pointwise", "parameters":[],
        "steps":[{"op":"identity", "rhs":null}]});
        4096
    ]);
    let payload = amplified.to_string();
    assert!(payload.len() < 1024 * 1024);
    assert!(matches!(
        ByteDecoderCheckpoint::from_json(&payload),
        Err(InferenceError::ByteDecoder(
            "checkpoint graph rank must be three"
        ))
    ));
}

#[cfg(feature = "wgpu")]
#[test]
fn live_checkpoint_template_has_no_host_weight_allocations() {
    let mut template = CheckpointTemplate::new(checkpoint().plan()).unwrap();
    assert_eq!(template.0.values_mut().count(), 18);
    assert!(template
        .0
        .values_mut()
        .all(|v| v.is_empty() && v.capacity() == 0));
}
