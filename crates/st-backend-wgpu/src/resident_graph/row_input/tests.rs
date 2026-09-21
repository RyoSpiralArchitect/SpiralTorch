#[cfg(not(target_arch = "wasm32"))]
mod gpu {
    use super::super::*;
    use crate::resident_graph::direct_io_tests::{definition, graph, read, runtime};
    use st_kernel_contracts::graph::{GraphParameter, ParameterRole};

    #[test]
    fn singleton_columns_and_rows_match_packing_across_tiles() {
        let Some(runtime) = runtime() else { return };
        let device = TensorDevice::new(runtime.clone()).unwrap();
        let base = device
            .upload(
                &[7, 5],
                &(0..35).map(|i| i as f32 / 16.).collect::<Vec<_>>(),
            )
            .unwrap();
        let column = base.narrow(1, 2, 1).unwrap();
        let row = base.narrow(0, 2, 1).unwrap();
        for view in [&column, &row] {
            let width = view.layout().shape()[1];
            let definition = GraphDefinition::new(
                NdLayout::contiguous(view.layout().shape()).unwrap(),
                vec![GraphStage::Linear {
                    weight: 0,
                    bias: 1,
                    gelu: true,
                }],
                vec![
                    GraphParameter {
                        role: ParameterRole::Weight,
                        shape: vec![width, 3],
                        values: (0..width * 3).map(|i| (i as f32 - 2.) / 8.).collect(),
                    },
                    GraphParameter {
                        role: ParameterRole::Bias,
                        shape: vec![3],
                        values: vec![-0.25, 0., 0.5],
                    },
                ],
            )
            .unwrap();
            for tile in [[8, 8, 16], [16, 16, 16], [16, 16, 64]] {
                for kernel in [MatmulKernel::Scalar, MatmulKernel::Register2x2] {
                    let mut graph = ResidentGraph::new(
                        runtime.clone(),
                        definition.clone(),
                        MatmulTile::new(tile[0], tile[1], tile[2]).unwrap(),
                        kernel,
                        MatmulAccumulation::Compensated,
                    )
                    .unwrap();
                    let expected = read(&graph.forward_tensor_packed(view).unwrap());
                    let actual = graph.forward_tensor(view).unwrap();
                    assert_eq!(read(&actual), expected);
                    assert!(graph
                        .input_source
                        .as_ref()
                        .unwrap()
                        .shares_storage_with(view));
                    graph.dispatch().unwrap();
                    assert_eq!(graph.snapshot().unwrap().read().unwrap(), expected);
                    assert_eq!(read(&actual), expected);
                }
            }
        }
    }

    #[test]
    fn row_views_match_packed_graphs_and_keep_original_storage_for_every_kernel() {
        let Some(runtime) = runtime() else { return };
        let device = TensorDevice::new(runtime.clone()).unwrap();
        let values: Vec<_> = (0..84).map(|i| (i as f32 - 40.) / 64.).collect();
        let base = device.upload(&[4, 3, 7], &values).unwrap();
        let prefix = base.narrow(0, 0, 2).unwrap();
        let views = [
            (prefix.narrow(2, 0, 3).unwrap(), true),
            (prefix.narrow(2, 2, 3).unwrap(), true),
            (
                base.reshape(&[2, 3, 14]).unwrap().narrow(2, 0, 3).unwrap(),
                true,
            ),
            (base.narrow(0, 1, 2).unwrap().narrow(2, 1, 3).unwrap(), true),
            (
                device
                    .upload(&[3], &[-0.25, 0.5, 1.])
                    .unwrap()
                    .broadcast_to(&[2, 3, 3])
                    .unwrap(),
                true,
            ),
            (
                device
                    .upload(&[3, 2, 3], &values[..18])
                    .unwrap()
                    .permute(&[1, 0, 2])
                    .unwrap(),
                false,
            ),
            (
                device
                    .upload(&[2, 3, 1], &values[..6])
                    .unwrap()
                    .broadcast_to(&[2, 3, 3])
                    .unwrap(),
                false,
            ),
        ];
        for kind in [MatmulKernel::Scalar, MatmulKernel::Register2x2] {
            for accumulation in [
                MatmulAccumulation::Sequential,
                MatmulAccumulation::Tiled,
                MatmulAccumulation::Compensated,
            ] {
                for stages in ["l", "ll", "lp", "pl", "lpl"] {
                    let mut graph = ResidentGraph::new(
                        runtime.clone(),
                        definition(stages, true),
                        MatmulTile::default(),
                        kind,
                        accumulation,
                    )
                    .unwrap();
                    for (view, regular) in &views {
                        let packed = graph.forward_tensor_packed(view).unwrap();
                        assert!(!graph
                            .input_source
                            .as_ref()
                            .unwrap()
                            .shares_storage_with(view));
                        let expected = read(&packed);
                        let held = graph.forward_tensor(view).unwrap();
                        assert_eq!(
                            read(&held),
                            expected,
                            "{stages}, {kind:?}, {accumulation:?}"
                        );
                        assert_eq!(
                            graph
                                .input_source
                                .as_ref()
                                .unwrap()
                                .shares_storage_with(view),
                            *regular && stages.starts_with('l')
                        );
                        for _ in 0..2 {
                            graph.dispatch().unwrap();
                            assert_eq!(graph.snapshot().unwrap().read().unwrap(), expected);
                        }
                        assert_eq!(read(&held), expected);
                        assert_eq!(read(&packed), expected);
                        assert_eq!(read(&graph.forward_tensor(view).unwrap()), expected);
                    }
                }
            }
        }
    }

    #[test]
    fn same_storage_different_addresses_and_aborted_compositions_do_not_poison_bindings() {
        let Some(runtime) = runtime() else { return };
        let mut graph = graph(&runtime, definition("ll", false));
        let device = graph.tensor_device().clone();
        let base = device
            .upload(
                &[2, 3, 7],
                &(0..42).map(|i| i as f32 / 32.).collect::<Vec<_>>(),
            )
            .unwrap();
        let first = base.narrow(2, 0, 3).unwrap();
        let second = base.narrow(2, 3, 3).unwrap();
        let expected_a = read(&graph.forward_tensor_packed(&first).unwrap());
        let expected_b = read(&graph.forward_tensor_packed(&second).unwrap());
        assert_ne!(expected_a, expected_b);
        let held = graph.forward_tensor(&first).unwrap();
        let before = (graph.generation(), graph.submitted_dispatches());
        for _ in 0..4 {
            let result: Result<(), GraphInferenceError> = graph.forward_composed(
                |_| Ok((second.clone(), ())),
                |_, _, ()| Err(GraphInferenceError::Readback),
            );
            assert!(result.is_err());
            assert_eq!((graph.generation(), graph.submitted_dispatches()), before);
            assert_eq!(graph.snapshot().unwrap().read().unwrap(), expected_a);
        }
        assert_eq!(read(&graph.forward_tensor(&first).unwrap()), expected_a);
        for _ in 0..4 {
            assert_eq!(read(&graph.forward_tensor(&second).unwrap()), expected_b);
            assert_eq!(read(&graph.forward_tensor(&first).unwrap()), expected_a);
        }
        let builds = graph.row_input.uniform_builds;
        let bindings = graph.direct_stats.input_bindings;
        for _ in 0..8 {
            assert_eq!(read(&graph.forward_tensor(&first).unwrap()), expected_a);
        }
        assert_eq!(graph.row_input.uniform_builds, builds);
        assert_eq!(graph.direct_stats.input_bindings, bindings);
        drop(graph);
        assert_eq!(read(&held), expected_a);
    }

    #[test]
    fn row_input_keeps_deferred_upstream_guards_after_dispatch_and_drop() {
        let Some(runtime) = runtime() else { return };
        let device = TensorDevice::new(runtime.clone()).unwrap();
        let huge = device.upload(&[2, 3, 4], &[f32::MAX; 24]).unwrap();
        let bad = huge.add(&huge).unwrap().narrow(2, 0, 3).unwrap();
        let mut graph = graph(&runtime, definition("lpl", true));
        let rejected = graph.forward_tensor(&bad).unwrap();
        assert!(graph
            .input_source
            .as_ref()
            .unwrap()
            .shares_storage_with(&bad));
        assert!(matches!(
            rejected.snapshot().unwrap().read(),
            Err(TensorError::NonFinite)
        ));
        graph.dispatch().unwrap();
        assert!(graph.snapshot().unwrap().read().is_err());
        let safe = device
            .upload(&[2, 3, 4], &[0.; 24])
            .unwrap()
            .narrow(2, 0, 3)
            .unwrap();
        for _ in 0..6 {
            read(&graph.forward_tensor(&safe).unwrap());
        }
        drop(graph);
        assert!(rejected.snapshot().unwrap().read().is_err());
    }
}
