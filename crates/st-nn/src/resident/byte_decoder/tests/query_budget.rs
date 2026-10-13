use super::*;
use crate::resident::portable::GraphRecord;
use st_kernel_contracts::{
    elementwise::ElementwiseOp,
    graph::GraphError,
    pointwise::{PointwiseChain, PointwiseStep},
};

fn geometry() -> (ByteDecoderPlan, ByteDecoderGeometryPlan) {
    let plan = plan(AttentionMask::Causal { query_offset: 0 }, 1).unwrap();
    let geometry = geometry_for(&plan)
        .with_mixture(
            &[
                ByteDecoderPairMetric::PoincareSquared,
                ByteDecoderPairMetric::EuclideanChordSquared,
            ],
            &[vec![0.; 4]],
            &[vec![1.; 4]],
        )
        .unwrap();
    (plan, geometry)
}

fn projection(stages: usize, slots: usize) -> InferencePlan {
    let parameters = vec![
        GraphParameter {
            role: ParameterRole::Bias,
            shape: vec![4],
            values: vec![0.; 4],
        };
        slots
    ];
    let mut next = 0;
    let stages = (0..stages)
        .map(|_| {
            let end = (next + 8).min(slots);
            let ids: Vec<_> = (next..end).collect();
            next = end;
            let steps = if ids.is_empty() {
                vec![PointwiseStep {
                    op: ElementwiseOp::Identity,
                    rhs: None,
                }]
            } else {
                (1..=ids.len())
                    .map(|slot| PointwiseStep {
                        op: ElementwiseOp::Add,
                        rhs: Some(slot),
                    })
                    .collect()
            };
            GraphStage::Pointwise {
                chain: PointwiseChain::new(ids.len() + 1, steps).unwrap(),
                parameters: ids,
            }
        })
        .collect();
    assert_eq!(next, slots);
    InferencePlan::from_graph_definition(
        GraphDefinition::new(
            NdLayout::contiguous(&[2, 3, 4]).unwrap(),
            stages,
            parameters,
        )
        .unwrap(),
    )
    .unwrap()
}

#[test]
fn query_router_reserves_its_shared_bias_stage() {
    let (_, geometry) = geometry();
    geometry
        .clone()
        .with_query_projections(&[projection(4095, 1)])
        .unwrap();
    assert!(matches!(
        geometry.with_query_projections(&[projection(4096, 1)]),
        Err(InferenceError::Graph(GraphError::Budget))
    ));
}

#[test]
fn query_router_reserves_its_shared_bias_parameter_slot() {
    let (_, geometry) = geometry();
    geometry
        .clone()
        .with_query_projections(&[projection(1024, 8191)])
        .unwrap();
    assert!(matches!(
        geometry.with_query_projections(&[projection(1024, 8192)]),
        Err(InferenceError::Graph(GraphError::Budget))
    ));
}

#[test]
fn checkpoint_import_applies_the_augmented_query_graph_budget() {
    let (plan, geometry) = geometry();
    let geometry = geometry
        .with_query_projections(&[projection(1, 1)])
        .unwrap();
    let checkpoint = plan
        .with_causal_geometry(geometry)
        .unwrap()
        .initial_checkpoint();
    let original: serde_json::Value = serde_json::from_str(&checkpoint.to_json().unwrap()).unwrap();
    for (stages, slots, accepted) in [
        (4095, 1, true),
        (4096, 1, false),
        (1024, 8191, true),
        (1024, 8192, false),
    ] {
        let mut record = original.clone();
        let graph = projection(stages, slots).graph_definition().unwrap();
        record["model"]["geometry"]["mixture"]["query_projections"][0] =
            serde_json::to_value(GraphRecord::from_graph(&graph).unwrap()).unwrap();
        let payload = record.to_string();
        let result = ByteDecoderCheckpoint::from_json(&payload);
        if accepted {
            let restored = result.unwrap();
            assert_eq!(
                restored
                    .plan()
                    .causal_geometry()
                    .unwrap()
                    .mixture()
                    .unwrap()
                    .query_projections()[0]
                    .graph_definition()
                    .unwrap()
                    .parameters()
                    .len(),
                slots
            );
            ByteDecoderCheckpoint::from_json(&restored.to_json().unwrap()).unwrap();
        } else {
            assert!(matches!(
                result,
                Err(InferenceError::Graph(GraphError::Budget))
            ));
        }
    }
}
