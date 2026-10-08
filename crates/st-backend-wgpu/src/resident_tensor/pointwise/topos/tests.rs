use super::*;
#[cfg(not(target_arch = "wasm32"))]
use crate::resident_tensor::pointwise::vjp::PointwiseVjpPlan;
#[cfg(not(target_arch = "wasm32"))]
use st_kernel_contracts::pointwise::BroadcastAdjoint;

#[cfg(not(target_arch = "wasm32"))]
fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(a.is_finite() && b.is_finite());
        assert!((a - b).abs() <= 3e-5 + 5e-4 * b.abs(), "{i}: {a} != {b}");
    }
}

#[test]
fn topos_shaders_and_layout_contract_are_validated_without_gpu() {
    for iterations in [1, 5, 4096] {
        for porosity in [0., 0.3, 1.] {
            let program =
                Program::Topos(ToposResonatorKernel::new(0.2, 1., porosity, iterations).unwrap());
            for variant in [
                program.clone(),
                Program::ToposResidualGuard(
                    ToposResonatorKernel::new(0.2, 1., porosity, iterations).unwrap(),
                ),
            ] {
                for evaluation in [
                    Evaluation::Forward,
                    Evaluation::RecomputedVjp,
                    Evaluation::Capture,
                    Evaluation::CapturedVjp,
                ] {
                    let source = generated_source_for(&variant, evaluation);
                    let module = naga::front::wgsl::parse_str(&source).unwrap();
                    naga::valid::Validator::new(
                        naga::valid::ValidationFlags::all(),
                        naga::valid::Capabilities::all(),
                    )
                    .validate(&module)
                    .unwrap();
                    if evaluation == Evaluation::CapturedVjp {
                        assert!(source.contains("let sensitivity = saved_sensitivity[i]"));
                        assert!(!source.contains("step < iterations"));
                    } else {
                        assert!(source.contains("step < iterations"));
                    }
                }
            }
            for gate in [&[5][..], &[1, 5], &[2, 3, 5]] {
                assert!(program
                    .validate_layouts(&[
                        NdLayout::contiguous(&[2, 3, 5]).unwrap(),
                        NdLayout::contiguous(gate).unwrap()
                    ])
                    .is_ok());
            }
            for gate in [&[2, 1, 1][..], &[1], &[3, 5]] {
                assert!(program
                    .validate_layouts(&[
                        NdLayout::contiguous(&[2, 3, 5]).unwrap(),
                        NdLayout::contiguous(gate).unwrap()
                    ])
                    .is_err());
            }
            assert!(program.validate_layouts(&[]).is_err());
        }
    }
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn topos_resident_forward_and_shared_vjp_execute_on_gpu() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) = runtime::ensure_default_runtime_blocking("topos.resident.vjp").unwrap();
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
    let device = TensorDevice::new(runtime).unwrap();
    let mut cases = 0;
    for rows in [0, 1, 3, 257] {
        for porosity in [0., 0.3, 1.] {
            let kernel = ToposResonatorKernel::new(0.2, 1., porosity, 5).unwrap();
            let x: Vec<_> = (0..rows * 5).map(|i| (i % 17) as f32 / 4. - 2.).collect();
            let seed: Vec<_> = (0..x.len()).map(|i| (i % 7) as f32 / 5. - 0.6).collect();
            for shared in [false, true] {
                let g: Vec<_> = (0..if shared { 5 } else { x.len() })
                    .map(|i| (i % 5) as f32 / 3. - 0.7)
                    .collect();
                let gate_shape = if shared { vec![1, 5] } else { vec![rows, 5] };
                let input = device.upload(&[rows, 5], &x).unwrap();
                let gate = device.upload(&gate_shape, &g).unwrap();
                let dy = device.upload(&[rows, 5], &seed).unwrap();
                let forward = PointwisePlan::topos_resonator(
                    device.clone(),
                    kernel,
                    vec![input.layout().clone(), gate.layout().clone()],
                )
                .unwrap();
                let output: Vec<_> = x
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| {
                        kernel
                            .capture(x, g[if shared { i % 5 } else { i }])
                            .unwrap()
                            .0
                    })
                    .collect();
                for execution in [
                    PointwiseExecution::Sequential,
                    PointwiseExecution::Batched,
                    PointwiseExecution::Fused,
                ] {
                    close(
                        &forward
                            .run(&[&input, &gate], execution)
                            .unwrap()
                            .snapshot()
                            .unwrap()
                            .read()
                            .unwrap(),
                        &output,
                    );
                }
                let vjp = PointwiseVjpPlan::new(forward).unwrap();
                let captured = PointwiseVjpPlan::for_graph(
                    PointwisePlan::topos_resonator(
                        device.clone(),
                        kernel,
                        vec![input.layout().clone(), gate.layout().clone()],
                    )
                    .unwrap(),
                )
                .unwrap();
                for factor in [1., -0.3] {
                    let seed: Vec<_> = seed.iter().map(|v| factor * v).collect();
                    let dy = if factor == 1. {
                        dy.clone()
                    } else {
                        device.upload(&[rows, 5], &seed).unwrap()
                    };
                    let scalar: Vec<_> = x
                        .iter()
                        .enumerate()
                        .map(|(i, &x)| {
                            kernel
                                .vjp(x, g[if shared { i % 5 } else { i }], seed[i])
                                .unwrap()
                        })
                        .collect();
                    let dx: Vec<_> = scalar.iter().map(|v| v[0]).collect();
                    let contributions: Vec<_> = scalar.iter().map(|v| v[1]).collect();
                    let dg = BroadcastAdjoint::new(&[rows, 5], &gate_shape)
                        .unwrap()
                        .reduce(&contributions)
                        .unwrap();
                    let gradients = vjp.run(&[&input, &gate], &dy).unwrap();
                    close(&gradients[0].snapshot().unwrap().read().unwrap(), &dx);
                    close(&gradients[1].snapshot().unwrap().read().unwrap(), &dg);
                    assert_eq!(gradients[1].layout().shape(), gate_shape);
                    let reused = captured.run(&[&input, &gate], &dy).unwrap();
                    close(&reused[0].snapshot().unwrap().read().unwrap(), &dx);
                    close(&reused[1].snapshot().unwrap().read().unwrap(), &dg);
                }
                cases += 1;
            }
        }
    }
    eprintln!("Topos resident kernel executed {cases} CPU/GPU cases, three execution modes, two arbitrary VJPs and shared sums");
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn topos_resident_views_boundaries_and_failure_guards_execute_on_gpu() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) = runtime::ensure_default_runtime_blocking("topos.resident.guards").unwrap();
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
    let device = TensorDevice::new(runtime).unwrap();
    let kernel = ToposResonatorKernel::new(0., 1., 0.3, 1).unwrap();
    let input = device
        .upload(&[3, 2], &[-1., 2., -0., 0.3, 1., -2.])
        .unwrap()
        .permute(&[1, 0])
        .unwrap();
    let gate = device.upload(&[3], &[1.; 3]).unwrap();
    let plan = PointwiseVjpPlan::new(
        PointwisePlan::topos_resonator(
            device.clone(),
            kernel,
            vec![input.layout().clone(), gate.layout().clone()],
        )
        .unwrap(),
    )
    .unwrap();
    let dy = device
        .upload(&[3, 2], &[1.; 6])
        .unwrap()
        .permute(&[1, 0])
        .unwrap();
    let x = [-1., -0., 1., 2., 0.3, -2.];
    close(
        &plan
            .forward()
            .run(&[&input, &gate], PointwiseExecution::Fused)
            .unwrap()
            .snapshot()
            .unwrap()
            .read()
            .unwrap(),
        &x.map(|v| kernel.capture(v, 1.).unwrap().0),
    );
    let gradients = plan.run(&[&input, &gate], &dy).unwrap();
    close(
        &gradients[0].snapshot().unwrap().read().unwrap(),
        &x.map(|v| kernel.vjp(v, 1., 1.).unwrap()[0]),
    );
    let expected = BroadcastAdjoint::new(&[2, 3], &[3])
        .unwrap()
        .reduce(&x.map(|v| kernel.vjp(v, 1., 1.).unwrap()[1]))
        .unwrap();
    close(&gradients[1].snapshot().unwrap().read().unwrap(), &expected);
    let wrong = device.upload(&[6], &[1.; 6]).unwrap();
    assert!(plan.run(&[&input, &gate], &wrong).is_err());

    for (input_values, gate_value, coupling, saturation, iterations, seed_value) in [
        (vec![f32::MAX, 0.], 2., 0., 1., 1, 0.), // drive overflows before saturation
        (vec![f32::MAX, 0.], 1., 0.5, f32::MAX, 2, 0.), // recurrence overflow
        (vec![1., 1.], 1., 0., 1., 1, f32::MAX), // shared gradient sum overflows
    ] {
        let input = device.upload(&[2, 1], &input_values).unwrap();
        let gate = device.upload(&[1], &[gate_value]).unwrap();
        let seed = device.upload(&[2, 1], &[seed_value; 2]).unwrap();
        let kernel = ToposResonatorKernel::new(coupling, saturation, 0., iterations).unwrap();
        let plan = PointwiseVjpPlan::new(
            PointwisePlan::topos_resonator(
                device.clone(),
                kernel,
                vec![input.layout().clone(), gate.layout().clone()],
            )
            .unwrap(),
        )
        .unwrap();
        for gradient in plan.run(&[&input, &gate], &seed).unwrap() {
            assert!(
                gradient.snapshot().unwrap().read().is_err(),
                "failure must taint every returned gradient"
            );
        }
        if seed_value == 0. {
            assert!(plan
                .forward()
                .run(&[&input, &gate], PointwiseExecution::Fused)
                .unwrap()
                .snapshot()
                .unwrap()
                .read()
                .is_err());
        }
    }
    eprintln!("Topos resident views, saturation boundary, recurrence/drive and shared-sum overflow guards executed on GPU");
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn topos_resident_nd_and_inherited_guard_match_cpu_on_gpu() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) = runtime::ensure_default_runtime_blocking("topos.resident.nd").unwrap();
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
    let device = TensorDevice::new(runtime).unwrap();
    let x: Vec<_> = (0..30).map(|i| i as f32 / 7. - 2.).collect();
    let g = [0.7, -1.1, 0.5, 1.7, -0.3];
    let input = device.upload(&[2, 3, 5], &x).unwrap();
    let gate = device.upload(&[5], &g).unwrap();
    let seed = device.upload(&[2, 3, 5], &[1.; 30]).unwrap();
    let kernel = ToposResonatorKernel::new(0.9, 1., 0.3, 16).unwrap();
    let plan = PointwiseVjpPlan::new(
        PointwisePlan::topos_resonator(
            device.clone(),
            kernel,
            vec![input.layout().clone(), gate.layout().clone()],
        )
        .unwrap(),
    )
    .unwrap();
    let expected: Vec<_> = x
        .iter()
        .enumerate()
        .map(|(i, &x)| kernel.vjp(x, g[i % 5], 1.).unwrap())
        .collect();
    let gradients = plan.run(&[&input, &gate], &seed).unwrap();
    let contributions: Vec<_> = expected.iter().map(|v| v[1]).collect();
    let dg = BroadcastAdjoint::new(&[2, 3, 5], &[5])
        .unwrap()
        .reduce(&contributions)
        .unwrap();
    close(
        &gradients[0].snapshot().unwrap().read().unwrap(),
        &expected.iter().map(|v| v[0]).collect::<Vec<_>>(),
    );
    close(&gradients[1].snapshot().unwrap().read().unwrap(), &dg);
    let zero = device.upload(&[], &[0.]).unwrap();
    let poisoned = seed.apply(ElementwiseOp::Divide, Some(&zero)).unwrap();
    for gradient in plan.run(&[&input, &gate], &poisoned).unwrap() {
        assert!(gradient.snapshot().unwrap().read().is_err());
    }
    assert!(plan
        .forward()
        .run(&[&poisoned, &gate], PointwiseExecution::Fused)
        .unwrap()
        .snapshot()
        .unwrap()
        .read()
        .is_err());
    // A failed invocation cannot contaminate a fresh invocation of the same plan.
    let clean = plan.run(&[&input, &gate], &seed).unwrap();
    close(&clean[1].snapshot().unwrap().read().unwrap(), &dg);
    eprintln!(
        "Topos resident 3-D shared gate, inherited failures and clean plan retry executed on GPU"
    );
}
