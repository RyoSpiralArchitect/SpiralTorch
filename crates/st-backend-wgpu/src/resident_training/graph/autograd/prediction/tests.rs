use super::*;
use st_kernel_contracts::pointwise::{PointwiseChain, PointwiseStep};

fn runtime() -> Option<WgpuRuntime> {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return None;
    }
    let (runtime, _) = runtime::ensure_default_runtime_blocking("graph.prediction.tests").unwrap();
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
    Some(runtime)
}

fn definition(rows: usize, kind: usize) -> GraphDefinition {
    let dense = GraphStage::Linear {
        weight: 0,
        bias: 1,
        gelu: kind == 1 || kind == 4,
    };
    let pointwise = |parameters, inputs, steps: &[(&str, Option<usize>)]| GraphStage::Pointwise {
        chain: PointwiseChain::new(
            inputs,
            steps
                .iter()
                .map(|(op, rhs)| PointwiseStep::named(op, *rhs).unwrap())
                .collect(),
        )
        .unwrap(),
        parameters,
    };
    let mut parameters = vec![
        GraphParameter {
            role: ParameterRole::Weight,
            shape: vec![2, 2],
            values: vec![2., 0., 0., 2.],
        },
        GraphParameter {
            role: ParameterRole::Bias,
            shape: vec![2],
            values: vec![0.; 2],
        },
    ];
    let stages = match kind {
        0 | 1 => vec![dense],
        2 => {
            parameters.clear();
            parameters.push(GraphParameter {
                role: ParameterRole::Gain,
                shape: vec![2],
                values: vec![2.; 2],
            });
            vec![pointwise(
                vec![0],
                2,
                &[("multiply", Some(1)), ("relu", None)],
            )]
        }
        3 => vec![dense, pointwise(vec![], 1, &[("relu", None)])],
        4 => vec![pointwise(vec![], 1, &[("add", Some(0))]), dense],
        _ => unreachable!(),
    };
    GraphDefinition::new(
        NdLayout::contiguous(&[1, rows, 2]).unwrap(),
        stages,
        parameters,
    )
    .unwrap()
}

fn tape(runtime: WgpuRuntime, rows: usize, kind: usize) -> ResidentGraphAutograd {
    ResidentGraphAutograd::new(
        runtime,
        definition(rows, kind),
        Default::default(),
        MatmulKernel::Register2x2,
        Default::default(),
    )
    .unwrap()
}

fn read(tensor: &ResidentTensor) -> Vec<f32> {
    tensor.snapshot().unwrap().read().unwrap()
}

// The unchanged MSE encoder's terminal scratch path is the forward oracle.
fn reference(g: &ResidentGraphAutograd) -> Vec<f32> {
    let graph = &g.graph;
    let context = g.tensor_device().runtime().context();
    let mut encoder = context.device().create_command_encoder(&Default::default());
    encoder.clear_buffer(&graph.validation, 0, None);
    encoder.clear_buffer(&graph.pointwise_flags, 0, None);
    graph.encode_forward(&mut encoder, &mut Default::default());
    let output = g
        .tensor_device()
        .capture_into(
            &mut encoder,
            g.output_layout(),
            graph.activations.last().unwrap(),
            &graph.validation,
        )
        .unwrap();
    context.queue().submit(Some(encoder.finish()));
    read(&output)
}

#[test]
fn owning_predictions_bypass_terminal_scratch_and_preserve_the_vjp_tape() {
    let Some(runtime) = runtime() else { return };
    for kind in 0..5 {
        for rows in [1, 3, 258] {
            let mut g = tape(runtime.clone(), rows, kind);
            let device = g.tensor_device().clone();
            let source = device
                .upload(
                    &[2, 1, rows + 1],
                    &[vec![-0.25; rows + 1], vec![0.5; rows + 1]].concat(),
                )
                .unwrap()
                .narrow(2, 1, rows)
                .unwrap()
                .permute(&[1, 2, 0])
                .unwrap();
            g.set_input_tensor(&source).unwrap();
            let expected = reference(&g);
            let scratch = g.graph.activations.last().unwrap();
            device.runtime().context().queue().write_buffer(
                scratch,
                0,
                bytemuck::cast_slice(&vec![f32::NAN; g.output_layout().len()]),
            );
            let first = g.forward().unwrap();
            assert_eq!(read(first.prediction()), expected);
            let seed = device
                .upload(
                    g.output_layout().shape(),
                    &vec![0.5; g.output_layout().len()],
                )
                .unwrap();
            let direct = g.backward(&first, &seed).unwrap();
            let actual: Vec<_> = std::iter::once(direct.input_gradient())
                .chain(direct.parameter_gradients())
                .map(read)
                .collect();
            // No terminal-value copy or producer touched the old destination.
            let context = device.runtime().context();
            let mut encoder = context.device().create_command_encoder(&Default::default());
            let untouched = device
                .capture_into(
                    &mut encoder,
                    g.output_layout(),
                    g.graph.activations.last().unwrap(),
                    &g.graph.validation,
                )
                .unwrap();
            context.queue().submit(Some(encoder.finish()));
            assert!(matches!(
                untouched.snapshot().unwrap().read(),
                Err(TensorError::NonFinite)
            ));
            assert_eq!(reference(&g), expected);
            let legacy_tape = g.backward(&first, &seed).unwrap();
            assert_eq!(
                actual,
                std::iter::once(legacy_tape.input_gradient())
                    .chain(legacy_tape.parameter_gradients())
                    .map(read)
                    .collect::<Vec<_>>()
            );
            assert_eq!(g.predictions.allocations, 1);
        }
    }
}

#[test]
fn prediction_views_pin_versions_and_pending_reads_survive_reuse() {
    let Some(runtime) = runtime() else { return };
    for kind in 0..5 {
        let mut g = tape(runtime.clone(), 3, kind);
        g.upload(&[0.5; 6]).unwrap();
        let expected = reference(&g);
        let first = g.forward().unwrap();
        let view = first.prediction().reshape(&[6]).unwrap();
        let pending = first.prediction().snapshot().unwrap();
        drop(first);
        g.upload(&[0.; 6]).unwrap();
        drop(g.forward().unwrap());
        let next = g.forward().unwrap();
        assert_eq!((g.predictions.allocations, g.predictions.reuses), (2, 1));
        assert_eq!(read(next.prediction()), vec![0.; 6]);
        drop(g);
        assert_eq!(pending.read().unwrap(), expected);
        assert_eq!(read(&view), expected);
        assert_eq!(read(next.prediction()), vec![0.; 6]);
    }
}

#[test]
fn prediction_guards_survive_masking_and_recycled_versions_recover() {
    let Some(runtime) = runtime() else { return };
    for kind in 0..5 {
        let mut g = tape(runtime.clone(), 3, kind);
        g.upload(&[f32::MAX; 6]).unwrap();
        let bad = g.forward().unwrap();
        let pending = bad.prediction().snapshot().unwrap();
        let zero = g.tensor_device().upload(&[], &[0.]).unwrap();
        let masked = bad.prediction().mul(&zero).unwrap();
        drop(bad);
        g.upload(&[0.; 6]).unwrap();
        let good = g.forward().unwrap();
        assert_eq!((g.predictions.allocations, g.predictions.reuses), (1, 1));
        assert_eq!(read(good.prediction()), vec![0.; 6]);
        let d = g.tensor_device();
        let huge = d.upload(&[1, 3, 2], &[f32::MAX; 6]).unwrap();
        let poisoned = huge
            .mul(&d.upload(&[], &[-2.]).unwrap())
            .unwrap()
            .relu()
            .unwrap();
        g.set_input_tensor(&poisoned).unwrap();
        let inherited = g.forward().unwrap();
        let seed = zero.broadcast_to(&[1, 3, 2]).unwrap();
        let gradients = g.backward(&inherited, &seed).unwrap();
        drop(g);
        assert!(matches!(pending.read(), Err(TensorError::NonFinite)));
        for tensor in [&masked, inherited.prediction(), gradients.input_gradient()]
            .into_iter()
            .chain(gradients.parameter_gradients())
        {
            assert!(matches!(
                tensor.snapshot().unwrap().read(),
                Err(TensorError::NonFinite)
            ));
        }
        assert_eq!(read(good.prediction()), vec![0.; 6]);
    }
}

#[test]
fn prediction_retention_is_bounded_and_input_aliases_do_not_recycle() {
    let Some(runtime) = runtime() else { return };
    let mut g = tape(runtime.clone(), 3, 0);
    g.upload(&[1.; 6]).unwrap();
    let held: Vec<_> = (0..6).map(|_| g.forward().unwrap()).collect();
    assert_eq!(g.predictions.versions.len(), 4);
    assert_eq!((g.predictions.allocations, g.predictions.reuses), (6, 0));
    for f in &held {
        assert_eq!(read(f.prediction()), vec![2.; 6]);
    }
    drop(held);
    let f = g.forward().unwrap();
    g.set_input_tensor(f.prediction()).unwrap();
    drop(f);
    let next = g.forward().unwrap();
    assert_eq!(read(next.prediction()), vec![4.; 6]);
    assert!(!next
        .prediction()
        .shares_storage_with(g.input_source.as_ref().unwrap()));
    let mut g = tape(runtime, 3, 0);
    g.predictions.limit = 0;
    g.upload(&[1.; 6]).unwrap();
    for _ in 0..8 {
        drop(g.forward().unwrap());
    }
    assert!(g.predictions.versions.is_empty());
    assert_eq!((g.predictions.allocations, g.predictions.reuses), (8, 0));
}
