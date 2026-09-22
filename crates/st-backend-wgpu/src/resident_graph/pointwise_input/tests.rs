use super::*;
use crate::resident_graph::direct_io_tests::{definition, graph, read, runtime};
use st_kernel_contracts::{
    graph::{GraphParameter, ParameterRole},
    pointwise::{PointwiseChain, PointwiseError, PointwiseStep},
};

fn affine_definition(tail: bool) -> GraphDefinition {
    let first = GraphStage::Pointwise {
        chain: PointwiseChain::new(
            3,
            vec![
                PointwiseStep::named("multiply", Some(1)).unwrap(),
                PointwiseStep::named("add", Some(2)).unwrap(),
                PointwiseStep::named("add", Some(0)).unwrap(),
                PointwiseStep::named("relu", None).unwrap(),
            ],
        )
        .unwrap(),
        parameters: vec![0, 1],
    };
    let mut stages = vec![first];
    if tail {
        stages.push(GraphStage::Pointwise {
            chain: PointwiseChain::new(1, vec![PointwiseStep::named("relu", None).unwrap()])
                .unwrap(),
            parameters: vec![],
        });
    }
    GraphDefinition::new(
        NdLayout::contiguous(&[2, 3, 3]).unwrap(),
        stages,
        vec![
            GraphParameter {
                role: ParameterRole::Gain,
                shape: vec![3],
                values: vec![0.25, -0.5, 1.],
            },
            GraphParameter {
                role: ParameterRole::Bias,
                shape: vec![1],
                values: vec![-0.125],
            },
        ],
    )
    .unwrap()
}

fn expected(view: &ResidentTensor, storage: &[f32]) -> Vec<f32> {
    (0..view.layout().len())
        .map(|i| {
            let x = storage[view.layout().storage_index(i).unwrap()];
            ((x * [0.25, -0.5, 1.][i % 3] - 0.125) + x).max(0.)
        })
        .collect()
}

#[test]
fn input_views_reuse_one_program_and_preserve_parameter_broadcasts_and_residuals() {
    let Some(runtime) = runtime() else { return };
    let device = TensorDevice::new(runtime.clone()).unwrap();
    let values: Vec<_> = (0..72).map(|i| (i as f32 - 30.) / 32.).collect();
    let base = device.upload(&[2, 3, 12], &values).unwrap();
    let views = [
        base.narrow(2, 0, 3).unwrap(),
        base.narrow(2, 4, 3).unwrap(),
        base.reshape(&[3, 2, 12])
            .unwrap()
            .permute(&[1, 0, 2])
            .unwrap()
            .narrow(2, 1, 3)
            .unwrap(),
        base.reshape(&[3, 2, 12])
            .unwrap()
            .permute(&[1, 2, 0])
            .unwrap()
            .narrow(1, 0, 3)
            .unwrap(),
        base.narrow(2, 0, 1)
            .unwrap()
            .broadcast_to(&[2, 3, 3])
            .unwrap(),
        base.narrow(0, 0, 1)
            .unwrap()
            .narrow(1, 0, 1)
            .unwrap()
            .narrow(2, 3, 3)
            .unwrap()
            .broadcast_to(&[2, 3, 3])
            .unwrap(),
    ];
    for tail in [false, true] {
        let mut graph = graph(&runtime, affine_definition(tail));
        let mut held = Vec::new();
        for view in &views {
            let want = expected(view, &values);
            let output = graph.forward_tensor(view).unwrap();
            assert_eq!(read(&output), want);
            assert!(graph
                .input_source
                .as_ref()
                .unwrap()
                .shares_storage_with(view));
            let Node::Pointwise { plan: original, .. } = &graph.nodes[0] else {
                unreachable!()
            };
            let cached = graph.pointwise_input.as_ref().unwrap().clone();
            assert!(cached.shares_pipeline_with(original));
            assert!(matches!(
                original.with_input_layout(&NdLayout::contiguous(&[18]).unwrap()),
                Err(TensorError::Pointwise(PointwiseError::LayoutMismatch))
            ));
            assert_eq!(read(&graph.forward_tensor(view).unwrap()), want);
            assert!(runtime::Shared::ptr_eq(
                &cached,
                graph.pointwise_input.as_ref().unwrap()
            ));
            held.push((output, want));
        }
        for (view, (_, want)) in views.iter().zip(&held) {
            assert_eq!(read(&graph.forward_tensor_packed(view).unwrap()), *want);
            assert!(!graph
                .input_source
                .as_ref()
                .unwrap()
                .shares_storage_with(view));
            assert_eq!(read(&graph.forward_tensor(view).unwrap()), *want);
            for _ in 0..2 {
                graph.dispatch().unwrap();
                assert_eq!(graph.snapshot().unwrap().read().unwrap(), *want);
            }
        }
        drop(graph);
        for (output, want) in held {
            assert_eq!(read(&output), want);
        }
    }
}

#[test]
fn cached_metadata_is_immutable_across_same_storage_layout_changes_and_aborts() {
    let Some(runtime) = runtime() else { return };
    let mut graph = graph(&runtime, affine_definition(true));
    let device = graph.tensor_device().clone();
    let values: Vec<_> = (0..36).map(|i| i as f32 / 32.).collect();
    let base = device.upload(&[2, 3, 6], &values).unwrap();
    let a = base.narrow(2, 0, 3).unwrap();
    let b = base.narrow(2, 3, 3).unwrap();
    let c = base
        .reshape(&[3, 2, 6])
        .unwrap()
        .permute(&[1, 0, 2])
        .unwrap()
        .narrow(2, 0, 3)
        .unwrap();
    assert_eq!(a.layout().offset(), c.layout().offset());
    assert_ne!(a.layout().strides(), c.layout().strides());
    let output = graph.forward_tensor(&a).unwrap();
    let before = (graph.generation(), graph.submitted_dispatches());
    for view in [&b, &c, &b, &c] {
        let failed: Result<(), GraphInferenceError> = graph.forward_composed(
            |_| Ok((view.clone(), ())),
            |_, _, ()| Err(GraphInferenceError::Readback),
        );
        assert!(failed.is_err());
        assert_eq!((graph.generation(), graph.submitted_dispatches()), before);
        assert_eq!(
            graph.snapshot().unwrap().read().unwrap(),
            expected(&a, &values)
        );
    }
    for view in [&a, &c, &b, &c, &a] {
        assert_eq!(
            read(&graph.forward_tensor(view).unwrap()),
            expected(view, &values)
        );
    }
    let cached = graph.pointwise_input.as_ref().unwrap().clone();
    let bindings = graph.direct_stats.input_bindings;
    for _ in 0..8 {
        graph.forward_tensor(&a).unwrap();
    }
    assert_eq!(graph.direct_stats.input_bindings, bindings);
    assert!(runtime::Shared::ptr_eq(
        &cached,
        graph.pointwise_input.as_ref().unwrap()
    ));
    graph.dispatch().unwrap();
    assert_eq!(
        graph.snapshot().unwrap().read().unwrap(),
        expected(&a, &values)
    );
    drop(graph);
    assert_eq!(read(&output), expected(&a, &values));
}

#[test]
fn view_input_preserves_masked_inherited_errors_and_first_stage_guard_indices() {
    let Some(runtime) = runtime() else { return };
    let device = TensorDevice::new(runtime.clone()).unwrap();
    let huge = device.upload(&[2, 3, 4], &[-f32::MAX; 24]).unwrap();
    let two = device.upload(&[1], &[2.]).unwrap();
    let masked = huge
        .mul(&two)
        .unwrap()
        .relu()
        .unwrap()
        .narrow(2, 0, 3)
        .unwrap();
    let mut graph = graph(&runtime, definition("pp", false));
    let invalid = graph.forward_tensor(&masked).unwrap();
    assert!(matches!(
        graph.snapshot().unwrap().read(),
        Err(GraphInferenceError::NonFinite { stage: 2, .. })
    ));
    graph.dispatch().unwrap();
    assert!(matches!(
        graph.snapshot().unwrap().read(),
        Err(GraphInferenceError::NonFinite { stage: 2, .. })
    ));
    let safe = device
        .upload(&[2, 3, 4], &[1.; 24])
        .unwrap()
        .narrow(2, 0, 3)
        .unwrap();
    read(&graph.forward_tensor(&safe).unwrap());
    drop(graph);
    assert!(invalid.snapshot().unwrap().read().is_err());

    let mut def = affine_definition(true);
    let mut parameters = def.parameters().to_vec();
    parameters[0].values.fill(f32::MAX);
    def = GraphDefinition::new(
        def.input_layout().clone(),
        def.stages().to_vec(),
        parameters,
    )
    .unwrap();
    let mut graph = super::super::direct_io_tests::graph(&runtime, def);
    let negative = device
        .upload(&[2, 3, 4], &[-2.; 24])
        .unwrap()
        .narrow(2, 0, 3)
        .unwrap();
    let rejected = graph.forward_tensor(&negative).unwrap();
    assert!(matches!(
        graph.snapshot().unwrap().read(),
        Err(GraphInferenceError::NonFinite { stage: 0, .. })
    ));
    drop(graph);
    assert!(rejected.snapshot().unwrap().read().is_err());
}
