use super::*;
use serde_json::json;
use st_kernel_contracts::adamw::{AdamWConfig, AdamWSlot};

fn checkpoint() -> ByteDecoderCheckpoint {
    let plan = super::super::tests::plan(AttentionMask::Causal { query_offset: 0 }, 2).unwrap();
    let geometry = super::super::tests::query_geometry_for(&plan);
    let mut c = plan
        .with_causal_geometry(geometry)
        .unwrap()
        .initial_checkpoint();
    c.attempted_revision = 3;
    c.optimizer = Some(AdamWState {
        config: AdamWConfig::default(),
        slots: c
            .plan
            .parameter_shapes()
            .unwrap()
            .into_iter()
            .map(|shape| {
                let len = shape.iter().product();
                AdamWSlot {
                    shape,
                    first: vec![-0.0; len],
                    second: vec![0.001; len],
                    steps: 2,
                }
            })
            .collect(),
    });
    c
}

#[test]
fn adamw_checkpoint_preserves_full_topology_history_and_exact_float_bits() {
    let c = checkpoint();
    let text = c.to_json().unwrap();
    assert!(text.contains(BYTE_DECODER_ADAMW_CHECKPOINT_SCHEMA));
    assert!(text.contains("adamw_zero_rate_freeze.v1"));
    let decoded = ByteDecoderCheckpoint::from_json(&text).unwrap();
    assert_eq!(decoded.to_json().unwrap(), text);
    assert_eq!(
        decoded.adamw_state().unwrap().slots[0].first[0].to_bits(),
        (-0.0f32).to_bits()
    );
    assert!(c.escaped_json_size_bound().unwrap() >= serde_json::to_string(&text).unwrap().len());
    assert_eq!(decoded.topology_json().unwrap(), c.topology_json().unwrap());
}

#[test]
fn adamw_contextual_checkpoint_rebuilds_updated_router_and_frozen_query() {
    let mut checkpoint = checkpoint();
    let geometry = checkpoint
        .plan
        .parameter_layout()
        .geometry()
        .unwrap()
        .clone();
    let router = geometry.mixture_logits().unwrap().start;
    let query_slots = geometry.query_projections()[0].clone();
    let mut record = ModelRecord::from_plan(&checkpoint.plan).unwrap();
    let mut values = record.values_mut().collect::<Vec<_>>();
    // Distinct query values represent frozen initialization; the router value is updated.
    values[router][0] = 0.375;
    values[query_slots.start][0] = 0.125;
    values[query_slots.start + 1][0] = -0.25;
    drop(values);
    checkpoint.plan = record.into_plan().unwrap();

    let state = checkpoint.optimizer.as_mut().unwrap();
    state.slots[router].first[0] = 0.0625;
    state.slots[router].second[0] = 0.03125;
    for slot in &mut state.slots[query_slots.clone()] {
        slot.first.fill(0.);
        slot.second.fill(0.);
        slot.steps = 0;
    }

    let encoded = checkpoint.to_json().unwrap();
    let restored = ByteDecoderCheckpoint::from_json(&encoded).unwrap();
    let geometry = restored.plan.causal_geometry().unwrap();
    assert_eq!(
        geometry.mixture().unwrap().raw_logits()[0][0].to_bits(),
        0.375f32.to_bits()
    );
    let query = geometry.mixture().unwrap().query_projections()[0]
        .graph_definition()
        .unwrap();
    assert_eq!(
        query.parameters()[0].values[0].to_bits(),
        0.125f32.to_bits()
    );
    assert_eq!(
        query.parameters()[1].values[0].to_bits(),
        (-0.25f32).to_bits()
    );
    let state = restored.adamw_state().unwrap();
    assert_eq!(state.slots[router].first[0].to_bits(), 0.0625f32.to_bits());
    assert_eq!(
        state.slots[router].second[0].to_bits(),
        0.03125f32.to_bits()
    );
    assert!(state.slots[query_slots].iter().all(|slot| slot.steps == 0
        && slot.first.iter().all(|&v| v == 0.)
        && slot.second.iter().all(|&v| v == 0.)));
    assert_eq!(restored.to_json().unwrap(), encoded);
}

#[test]
fn adamw_checkpoint_rejects_missing_downgraded_and_malformed_state() {
    let original: serde_json::Value =
        serde_json::from_str(&checkpoint().to_json().unwrap()).unwrap();
    for kind in 0..16 {
        let mut bad = original.clone();
        match kind {
            0 => {
                bad.as_object_mut().unwrap().remove("optimizer");
            }
            1 => bad["optimizer"] = json!(null),
            2 => bad["schema"] = json!(BYTE_DECODER_QUERY_MIXTURE_CHECKPOINT_SCHEMA),
            3 => bad["update_rule"] = json!("stateless_sgd.v1"),
            4 => bad["optimizer"]["beta1"] = json!(1.0),
            5 => bad["optimizer"]["epsilon"] = json!(0.0),
            6 => bad["optimizer"]["slots"][0]["second"][0] = json!(-0.1),
            7 => bad["optimizer"]["slots"][0]["steps"] = json!(4),
            8 => bad["optimizer"]["slots"][0]["steps"] = json!(0),
            9 => bad["optimizer"]["slots"][0]["shape"] = json!([1]),
            10 => bad["optimizer"]["slots"][0]["first"] = json!([]),
            11 => bad["optimizer"]["slots"].as_array_mut().unwrap().clear(),
            12 => bad["optimizer"]["unknown"] = json!(true),
            13 => {
                bad.as_object_mut().unwrap().remove("model_schema");
            }
            14 => bad["model_schema"] = json!(null),
            _ => bad["model_schema"] = json!(BYTE_DECODER_CHECKPOINT_SCHEMA),
        }
        assert!(
            ByteDecoderCheckpoint::from_json(&bad.to_string()).is_err(),
            "variant {kind}"
        );
    }
    let old = checkpoint().plan.initial_checkpoint().to_json().unwrap();
    assert!(!old.contains("optimizer"));
    let mut bad: serde_json::Value = serde_json::from_str(&old).unwrap();
    bad["optimizer"] = json!(null);
    assert!(ByteDecoderCheckpoint::from_json(&bad.to_string()).is_err());
}

#[test]
fn adamw_geometry_discriminators_cannot_disappear_on_resume() {
    let p = super::super::tests::plan(AttentionMask::Causal { query_offset: 0 }, 2).unwrap();
    let g = super::super::tests::geometry_for(&p);
    for (geometry, omitted) in [
        (
            g.clone()
                .with_pair_metric(ByteDecoderPairMetric::EuclideanChordSquared),
            "pair_metric",
        ),
        (
            g.clone()
                .with_pair_metric(ByteDecoderPairMetric::CategoricalFisherRaoSquared),
            "pair_metric",
        ),
        (
            g.with_head_pair_metrics(&vec![
                vec![
                    ByteDecoderPairMetric::EuclideanChordSquared,
                    ByteDecoderPairMetric::PoincareSquared
                ];
                2
            ])
            .unwrap(),
            "head_pair_metrics",
        ),
    ] {
        let mut c = p
            .clone()
            .with_causal_geometry(geometry)
            .unwrap()
            .initial_checkpoint();
        c.optimizer = Some(AdamWState {
            config: AdamWConfig::default(),
            slots: c
                .plan
                .parameter_shapes()
                .unwrap()
                .into_iter()
                .map(|shape| {
                    let len = shape.iter().product();
                    AdamWSlot {
                        shape,
                        first: vec![0.0; len],
                        second: vec![0.0; len],
                        steps: 0,
                    }
                })
                .collect(),
        });
        let text = c.to_json().unwrap();
        assert_eq!(
            ByteDecoderCheckpoint::from_json(&text)
                .unwrap()
                .to_json()
                .unwrap(),
            text
        );
        let mut bad: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert!(bad["model"]["geometry"]
            .as_object_mut()
            .unwrap()
            .remove(omitted)
            .is_some());
        assert!(
            ByteDecoderCheckpoint::from_json(&bad.to_string()).is_err(),
            "missing {omitted}"
        );
    }
}
