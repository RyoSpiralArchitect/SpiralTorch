use super::*;
use st_kernel_contracts::topos_resonator::ToposResonatorKernel;

fn runtime() -> Option<WgpuRuntime> {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return None;
    }
    let (runtime, _) = runtime::ensure_default_runtime_blocking("topos.graph.capture").unwrap();
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
    Some(runtime)
}

fn graph(
    runtime: WgpuRuntime,
    rows: usize,
    gate: &[f32],
    kernel: ToposResonatorKernel,
) -> ResidentGraphAutograd {
    ResidentGraphAutograd::new(
        runtime,
        GraphDefinition::new(
            NdLayout::contiguous(&[1, rows, gate.len()]).unwrap(),
            vec![GraphStage::ToposResonator {
                gate: 0,
                kernel,
                max_volume: rows * gate.len(),
            }],
            vec![GraphParameter {
                role: ParameterRole::Gate,
                shape: vec![gate.len()],
                values: gate.to_vec(),
            }],
        )
        .unwrap(),
        Default::default(),
        MatmulKernel::Scalar,
        Default::default(),
    )
    .unwrap()
}

fn read(tensor: &ResidentTensor) -> Vec<f32> {
    tensor.snapshot().unwrap().read().unwrap()
}

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &b) in actual.iter().zip(expected) {
        assert!(a.is_finite() && b.is_finite());
        assert!((a - b).abs() <= 3e-5 + 5e-4 * b.abs(), "{a} != {b}");
    }
}

#[test]
fn topos_graph_capture_matches_recomputed_vjp_and_retains_only_latest_tape() {
    let Some(runtime) = runtime() else { return };
    let device = TensorDevice::new(runtime.clone()).unwrap();
    for rows in [1, 257] {
        for porosity in [0., 0.3, 1.] {
            for iterations in [1, 5, 64, 4096] {
                let kernel = ToposResonatorKernel::new(0.2, 1., porosity, iterations).unwrap();
                let gate = [0.8, -0.4, 1.1];
                let mut graph = graph(runtime.clone(), rows, &gate, kernel);
                let shape = graph.input_layout().shape().to_vec();
                let x: Vec<_> = (0..rows * 3).map(|i| (i % 17) as f32 / 4. - 2.).collect();
                let input = device.upload(&shape, &x).unwrap();
                let gate_tensor = device.upload(&[3], &gate).unwrap();
                let recomputed = PointwiseVjpPlan::new(
                    PointwisePlan::topos_graph(
                        device.clone(),
                        kernel,
                        vec![input.layout().clone(), gate_tensor.layout().clone()],
                        0,
                    )
                    .unwrap(),
                )
                .unwrap();
                graph.set_input_tensor(&input).unwrap();
                let forward = graph.forward().unwrap();
                let expected = read(
                    &recomputed
                        .forward()
                        .run(
                            &[&input, &gate_tensor],
                            st_kernel_contracts::pointwise::PointwiseExecution::Fused,
                        )
                        .unwrap(),
                );
                close(&read(forward.prediction()), &expected);
                let mut retained = Vec::new();
                for factor in [1., -0.3, 0.] {
                    let seed: Vec<_> = (0..x.len())
                        .map(|i| factor * ((i % 7) as f32 / 5. - 0.6))
                        .collect();
                    let seed = device.upload(&shape, &seed).unwrap();
                    let actual = graph.backward(&forward, &seed).unwrap();
                    let reference = recomputed.run(&[&input, &gate_tensor], &seed).unwrap();
                    let dx = read(&reference[0]);
                    let dg = read(&reference[1]);
                    close(&read(actual.input_gradient()), &dx);
                    close(&read(&actual.parameter_gradients()[0]), &dg);
                    retained.push((actual, dx, dg));
                }
                let seed = device.upload(&shape, &vec![1.; x.len()]).unwrap();
                let latest = graph.forward().unwrap();
                assert!(matches!(
                    graph.backward(&forward, &seed),
                    Err(TrainingError::StaleForward)
                ));
                graph.backward(&latest, &seed).unwrap();
                graph
                    .set_parameter_tensors(&[device.upload(&[3], &[0.; 3]).unwrap()])
                    .unwrap();
                assert!(matches!(
                    graph.backward(&latest, &seed),
                    Err(TrainingError::StaleForward)
                ));
                let zero = graph.forward().unwrap();
                close(&read(zero.prediction()), &vec![0.; x.len()]);
                graph.upload(&vec![0.; x.len()]).unwrap();
                assert!(matches!(
                    graph.backward(&zero, &seed),
                    Err(TrainingError::StaleForward)
                ));
                drop(graph);
                close(&read(forward.prediction()), &expected);
                for (gradient, dx, dg) in retained {
                    close(&read(gradient.input_gradient()), &dx);
                    close(&read(&gradient.parameter_gradients()[0]), &dg);
                }
            }
        }
    }
}

#[test]
fn topos_graph_capture_preserves_forward_errors_and_allows_seed_retry() {
    let Some(runtime) = runtime() else { return };
    let device = TensorDevice::new(runtime.clone()).unwrap();
    let kernel = ToposResonatorKernel::new(0.5, f32::MAX, 0., 1).unwrap();
    let mut graph = graph(runtime, 1, &[1.], kernel);
    let zero = device.upload(&[1, 1, 1], &[0.]).unwrap();
    graph.upload(&[f32::MAX * 0.8]).unwrap();
    let invalid = graph.forward().unwrap();
    assert!(invalid.prediction().snapshot().unwrap().read().is_err());
    let gradients = graph.backward(&invalid, &zero).unwrap();
    assert!(gradients
        .input_gradient()
        .snapshot()
        .unwrap()
        .read()
        .is_err());
    assert!(gradients.parameter_gradients()[0]
        .snapshot()
        .unwrap()
        .read()
        .is_err());
    graph.upload(&[0.25]).unwrap();
    let valid = graph.forward().unwrap();
    let large = device.upload(&[1, 1, 1], &[f32::MAX]).unwrap();
    let bad_seed = large
        .apply(
            st_kernel_contracts::elementwise::ElementwiseOp::Add,
            Some(&large),
        )
        .unwrap();
    let bad = graph.backward(&valid, &bad_seed).unwrap();
    assert!(bad.input_gradient().snapshot().unwrap().read().is_err());
    let seed = device.upload(&[1, 1, 1], &[2.]).unwrap();
    let good = graph.backward(&valid, &seed).unwrap();
    close(&read(good.input_gradient()), &[2.]);
    close(&read(&good.parameter_gradients()[0]), &[0.5]);
    assert!(gradients
        .input_gradient()
        .snapshot()
        .unwrap()
        .read()
        .is_err());
}
