use super::*;
use st_kernel_contracts::{
    elementwise::ElementwiseOp,
    graph::{GraphParameter, ParameterRole},
    pointwise::{PointwiseChain, PointwiseStep},
};

fn runtime() -> Option<WgpuRuntime> {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return None;
    }
    let (runtime, _) = runtime::ensure_default_runtime_blocking("graph.direct_io.tests").unwrap();
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
    Some(runtime)
}

fn pointwise() -> GraphStage {
    GraphStage::Pointwise {
        chain: PointwiseChain::new(1, vec![PointwiseStep::named("relu", None).unwrap()]).unwrap(),
        parameters: vec![],
    }
}

fn definition(kinds: &str, gelu: bool) -> GraphDefinition {
    let mut stages = Vec::new();
    let mut parameters = Vec::new();
    let mut width = 3;
    for kind in kinds.chars() {
        if kind == 'p' {
            stages.push(pointwise());
        } else {
            let cols = if width == 3 { 5 } else { 3 };
            let weight = parameters.len();
            parameters.push(GraphParameter {
                role: ParameterRole::Weight,
                shape: vec![width, cols],
                values: (0..width * cols)
                    .map(|i| ((i % 7) as f32 - 3.) / 16.)
                    .collect(),
            });
            parameters.push(GraphParameter {
                role: ParameterRole::Bias,
                shape: vec![cols],
                values: (0..cols).map(|i| (i as f32 - 1.) / 32.).collect(),
            });
            stages.push(GraphStage::Linear {
                weight,
                bias: weight + 1,
                gelu,
            });
            width = cols;
        }
    }
    GraphDefinition::new(
        NdLayout::contiguous(&[2, 3, 3]).unwrap(),
        stages,
        parameters,
    )
    .unwrap()
}

fn graph(runtime: &WgpuRuntime, definition: GraphDefinition) -> ResidentGraph {
    ResidentGraph::new(
        runtime.clone(),
        definition,
        MatmulTile::default(),
        MatmulKernel::Scalar,
        MatmulAccumulation::Sequential,
    )
    .unwrap()
}

fn read(tensor: &ResidentTensor) -> Vec<f32> {
    tensor.snapshot().unwrap().read().unwrap()
}

#[test]
fn boundaries_and_views_match_explicit_execution_for_every_dense_variant() {
    let Some(runtime) = runtime() else { return };
    let device = TensorDevice::new(runtime.clone()).unwrap();
    let values: Vec<_> = (0..36).map(|i| (i as f32 - 17.) / 16.).collect();
    let base = device.upload(&[4, 3, 3], &values).unwrap();
    let views = [
        device.upload(&[2, 3, 3], &values[..18]).unwrap(),
        base.narrow(0, 0, 2).unwrap(),
        base.narrow(0, 1, 2).unwrap(),
        device
            .upload(&[3, 2, 3], &values[..18])
            .unwrap()
            .permute(&[1, 0, 2])
            .unwrap(),
        device
            .upload(&[3], &[-0.25, 0.5, 1.])
            .unwrap()
            .broadcast_to(&[2, 3, 3])
            .unwrap(),
    ];
    for kernel in [MatmulKernel::Scalar, MatmulKernel::Register2x2] {
        for accumulation in [
            MatmulAccumulation::Sequential,
            MatmulAccumulation::Tiled,
            MatmulAccumulation::Compensated,
        ] {
            for (index, kinds) in ["l", "p", "ll", "lp", "pl", "pp", "lpl", "plp"]
                .iter()
                .enumerate()
            {
                let mut graph = ResidentGraph::new(
                    runtime.clone(),
                    definition(kinds, index % 2 == 0),
                    MatmulTile::default(),
                    kernel,
                    accumulation,
                )
                .unwrap();
                for view in &views {
                    let original = read(view);
                    graph.upload(&original).unwrap();
                    graph.dispatch().unwrap();
                    let expected = graph.snapshot().unwrap().read().unwrap();
                    let direct = graph.forward_tensor(view).unwrap();
                    assert_eq!(
                        read(&direct),
                        expected,
                        "{kinds}, {kernel:?}, {accumulation:?}, {:?}",
                        view.layout()
                    );
                    assert_eq!(graph.snapshot().unwrap().read().unwrap(), expected);
                    assert_eq!(read(&graph.output_tensor().unwrap()), expected);
                    assert_eq!(read(view), original);
                    if view.layout().is_contiguous() && view.layout().offset() == 0 {
                        assert!(
                            std::ptr::eq(
                                graph.input_source.as_ref().unwrap().values(),
                                view.values()
                            ),
                            "packed input must not be copied into scratch"
                        );
                    }
                    assert!(
                        std::ptr::eq(graph.output_tensor().unwrap().values(), direct.values()),
                        "direct output must not be recaptured by a values copy"
                    );
                }
            }
        }
    }
}

#[test]
fn direct_and_explicit_transitions_preserve_input_counters_and_owning_outputs() {
    let Some(runtime) = runtime() else { return };
    let mut graph = graph(&runtime, definition("pp", false));
    let device = graph.tensor_device().clone();
    let input = device.upload(&[2, 3, 3], &[-1.; 18]).unwrap();
    let first = graph.forward_tensor(&input).unwrap();
    let snapshot = graph.snapshot().unwrap();
    assert_eq!((graph.generation(), graph.submitted_dispatches()), (1, 1));
    assert_eq!((snapshot.generation(), snapshot.dispatch()), (1, 1));
    let next = device.upload(&[2, 3, 3], &[2.; 18]).unwrap();
    let second = graph.forward_tensor(&next).unwrap();
    for count in [3, 4] {
        assert_eq!(graph.dispatch().unwrap(), count);
        assert_eq!(graph.generation(), 2);
        assert_eq!(graph.snapshot().unwrap().read().unwrap(), vec![2.; 18]);
        assert!(!graph.input_direct);
    }
    graph.upload(&[3.; 18]).unwrap();
    assert!(matches!(
        graph.output_tensor(),
        Err(GraphInferenceError::StaleOutput)
    ));
    graph.dispatch().unwrap();
    assert_eq!(read(&graph.output_tensor().unwrap()), vec![3.; 18]);
    graph.forward_tensor(&input).unwrap();
    graph.set_input_tensor(&next).unwrap();
    assert!(matches!(
        graph.snapshot(),
        Err(GraphInferenceError::StaleOutput)
    ));
    graph.dispatch().unwrap();
    assert_eq!(read(&graph.output_tensor().unwrap()), vec![2.; 18]);
    let final_output = graph.forward_tensor(&first).unwrap();
    assert_eq!((graph.generation(), graph.submitted_dispatches()), (6, 8));
    drop(graph);
    assert_eq!(snapshot.read().unwrap(), vec![0.; 18]);
    assert_eq!(read(&first), vec![0.; 18]);
    assert_eq!(read(&second), vec![2.; 18]);
    assert_eq!(read(&final_output), vec![0.; 18]);
}

#[test]
fn guards_survive_masking_reuse_and_packed_invalid_input() {
    let Some(runtime) = runtime() else { return };
    for dense in [false, true] {
        let parameters = if dense {
            vec![
                GraphParameter {
                    role: ParameterRole::Weight,
                    shape: vec![1, 1],
                    values: vec![f32::MAX],
                },
                GraphParameter {
                    role: ParameterRole::Bias,
                    shape: vec![1],
                    values: vec![0.],
                },
            ]
        } else {
            vec![GraphParameter {
                role: ParameterRole::Gain,
                shape: vec![1],
                values: vec![f32::MAX],
            }]
        };
        let first = if dense {
            GraphStage::Linear {
                weight: 0,
                bias: 1,
                gelu: false,
            }
        } else {
            GraphStage::Pointwise {
                chain: PointwiseChain::new(
                    2,
                    vec![PointwiseStep::named("multiply", Some(1)).unwrap()],
                )
                .unwrap(),
                parameters: vec![0],
            }
        };
        let def = GraphDefinition::new(
            NdLayout::contiguous(&[2, 1]).unwrap(),
            vec![first, pointwise()],
            parameters,
        )
        .unwrap();
        let mut graph = graph(&runtime, def);
        let device = graph.tensor_device().clone();
        let bad = graph
            .forward_tensor(&device.upload(&[2, 1], &[-2.; 2]).unwrap())
            .unwrap();
        let snapshot = graph.snapshot().unwrap();
        let good = graph
            .forward_tensor(&device.upload(&[2, 1], &[0.; 2]).unwrap())
            .unwrap();
        assert_eq!(read(&good), vec![0.; 2]);
        assert!(bad.snapshot().unwrap().read().is_err());
        assert!(matches!(
            snapshot.read(),
            Err(GraphInferenceError::NonFinite { stage: 0, .. })
        ));
        let upstream = device.upload(&[2, 2], &[f32::MAX; 4]).unwrap();
        let upstream = upstream
            .apply(ElementwiseOp::Multiply, Some(&upstream))
            .unwrap()
            .narrow(1, 1, 1)
            .unwrap();
        let inherited = graph.forward_tensor(&upstream).unwrap();
        assert!(inherited.snapshot().unwrap().read().is_err());
        graph.dispatch().unwrap();
        assert!(graph.snapshot().unwrap().read().is_err());
        graph.forward_tensor(&good).unwrap();
        drop(graph);
        assert!(inherited.snapshot().unwrap().read().is_err());
        assert!(bad.snapshot().unwrap().read().is_err());
    }
}

#[test]
fn rejected_direct_calls_preserve_previous_state() {
    let Some(runtime) = runtime() else { return };
    let mut graph = graph(&runtime, definition("p", false));
    let input = graph.tensor_device().upload(&[2, 3, 3], &[1.; 18]).unwrap();
    graph.forward_tensor(&input).unwrap();
    assert!(matches!(
        graph.forward_tensor(&input.reshape(&[6, 3]).unwrap()),
        Err(GraphInferenceError::InputShape)
    ));
    let other = pollster::block_on(WgpuRuntime::request_headless("graph.direct_io.other")).unwrap();
    let other = TensorDevice::new(other).unwrap();
    assert!(graph
        .forward_tensor(&other.upload(&[2, 3, 3], &[2.; 18]).unwrap())
        .is_err());
    assert_eq!((graph.generation(), graph.submitted_dispatches()), (1, 1));
    assert_eq!(graph.snapshot().unwrap().read().unwrap(), vec![1.; 18]);
    graph.submitted_dispatches = u64::MAX;
    assert!(matches!(
        graph.forward_tensor(&input),
        Err(GraphInferenceError::CounterOverflow)
    ));
    assert_eq!(graph.generation(), 1);
    graph.submitted_dispatches = 1;
    graph.generation = u64::MAX;
    assert!(matches!(
        graph.forward_tensor(&input),
        Err(GraphInferenceError::CounterOverflow)
    ));
    assert_eq!(graph.submitted_dispatches(), 1);
    graph.generation = 1;
    assert_eq!(read(&graph.output_tensor().unwrap()), vec![1.; 18]);
    assert_eq!(graph.snapshot().unwrap().read().unwrap(), vec![1.; 18]);
}
