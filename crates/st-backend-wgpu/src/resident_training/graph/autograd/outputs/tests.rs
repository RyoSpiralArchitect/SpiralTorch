use super::*;
use st_kernel_contracts::{
    elementwise::ElementwiseOp,
    pointwise::{PointwiseChain, PointwiseStep},
};

fn runtime() -> Option<WgpuRuntime> {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return None;
    }
    let (runtime, _) = runtime::ensure_default_runtime_blocking("graph.direct_vjp.tests").unwrap();
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
    Some(runtime)
}

fn graph(runtime: WgpuRuntime, rows: usize, dense: bool) -> ResidentGraphAutograd {
    let (stages, parameters) = if dense {
        (
            vec![GraphStage::Linear {
                weight: 0,
                bias: 1,
                gelu: false,
            }],
            vec![
                GraphParameter {
                    role: ParameterRole::Weight,
                    shape: vec![2, 3],
                    values: vec![1.; 6],
                },
                GraphParameter {
                    role: ParameterRole::Bias,
                    shape: vec![3],
                    values: vec![0.; 3],
                },
            ],
        )
    } else {
        (
            vec![GraphStage::Pointwise {
                chain: PointwiseChain::new(
                    2,
                    vec![PointwiseStep {
                        op: ElementwiseOp::Multiply,
                        rhs: Some(1),
                    }],
                )
                .unwrap(),
                parameters: vec![0],
            }],
            vec![GraphParameter {
                role: ParameterRole::Gain,
                shape: vec![2],
                values: vec![2.; 2],
            }],
        )
    };
    let definition = GraphDefinition::new(
        NdLayout::contiguous(&[1, rows, 2]).unwrap(),
        stages,
        parameters,
    )
    .unwrap();
    let mut g = ResidentGraphAutograd::new(
        runtime,
        definition,
        Default::default(),
        MatmulKernel::Scalar,
        Default::default(),
    )
    .unwrap();
    g.upload(&vec![1.; rows * 2]).unwrap();
    // The old terminal scratch must be neither an input nor the destination of
    // the owning VJP. Intermediate activation gradients still use shared scratch.
    let queue = g.tensor_device().runtime().context().queue();
    for buffer in std::iter::once(&g.graph.gradients[0]).chain(&g.graph.raw_gradients) {
        queue.write_buffer(
            buffer,
            0,
            bytemuck::cast_slice(&vec![f32::NAN; buffer.size() as usize / 4]),
        );
    }
    g
}

fn backward(g: &mut ResidentGraphAutograd, f: &GraphForward, seed: f32) -> GraphGradients {
    let seed = g
        .tensor_device()
        .upload(
            g.output_layout().shape(),
            &vec![seed; g.output_layout().len()],
        )
        .unwrap();
    g.backward(f, &seed).unwrap()
}

fn check(gradients: &GraphGradients, rows: usize, dense: bool, seed: f32) {
    assert_eq!(
        gradients.input.snapshot().unwrap().read().unwrap(),
        vec![if dense { 3. * seed } else { 2. * seed }; rows * 2]
    );
    for p in &gradients.parameters {
        assert_eq!(
            p.snapshot().unwrap().read().unwrap(),
            vec![rows as f32 * seed; p.layout().len()]
        );
    }
}

#[test]
fn direct_vjp_reuses_owning_destinations_not_terminal_scratch() {
    let Some(runtime) = runtime() else {
        return;
    };
    for dense in [true, false] {
        // Nonbroadcast, one-pass and two-pass pointwise reductions, and GEMM tails.
        for rows in [1, 3, 258] {
            let mut g = graph(runtime.clone(), rows, dense);
            let f = g.forward().unwrap();
            let first = backward(&mut g, &f, 1.);
            let pending = first.input.snapshot().unwrap();
            let held = first.parameters[0]
                .reshape(&[first.parameters[0].layout().len()])
                .unwrap();
            drop(first);
            let bad = backward(&mut g, &f, f32::MAX);
            let bad_reads: Vec<_> = std::iter::once(&bad.input)
                .chain(&bad.parameters)
                .map(|t| t.snapshot().unwrap())
                .collect();
            let bad_consumer = bad
                .input
                .mul(&g.tensor_device().upload(&[], &[0.]).unwrap())
                .unwrap()
                .snapshot()
                .unwrap();
            drop(bad);
            let zero = backward(&mut g, &f, 0.);
            check(&zero, rows, dense, 0.);
            drop(zero);
            let recovered = backward(&mut g, &f, 2.);
            assert_eq!((g.outputs.allocations, g.outputs.reuses), (2, 2));
            check(&recovered, rows, dense, 2.);
            for buffer in std::iter::once(&g.graph.gradients[0]).chain(&g.graph.raw_gradients) {
                let context = g.tensor_device().runtime().context();
                let mut encoder = context.device().create_command_encoder(&Default::default());
                let result = g
                    .tensor_device()
                    .capture_into(
                        &mut encoder,
                        &NdLayout::contiguous(&[buffer.size() as usize / 4]).unwrap(),
                        buffer,
                        &g.graph.validation,
                    )
                    .unwrap();
                context.queue().submit(Some(encoder.finish()));
                assert!(matches!(
                    result.snapshot().unwrap().read(),
                    Err(TensorError::NonFinite)
                ));
            }
            drop(g);
            assert_eq!(
                pending.read().unwrap(),
                vec![if dense { 3. } else { 2. }; rows * 2]
            );
            assert_eq!(
                held.snapshot().unwrap().read().unwrap(),
                vec![rows as f32; held.layout().len()]
            );
            check(&recovered, rows, dense, 2.);
            for read in bad_reads.into_iter().chain([bad_consumer]) {
                assert!(matches!(read.read(), Err(TensorError::NonFinite)));
            }
        }
    }
}

#[test]
fn direct_vjp_busy_versions_spill_and_zero_retention_never_reuses() {
    let Some(runtime) = runtime() else {
        return;
    };
    for dense in [true, false] {
        let mut g = graph(runtime.clone(), 258, dense);
        let f = g.forward().unwrap();
        let mut retained: Vec<_> = (1..=6).map(|i| backward(&mut g, &f, i as f32)).collect();
        assert_eq!(g.outputs.versions.len(), 4);
        assert_eq!((g.outputs.allocations, g.outputs.reuses), (6, 0));
        drop(retained.remove(0));
        let reused = backward(&mut g, &f, 7.);
        assert_eq!((g.outputs.allocations, g.outputs.reuses), (6, 1));
        drop(g);
        for (i, version) in retained.iter().enumerate() {
            check(version, 258, dense, (i + 2) as f32);
        }
        check(&reused, 258, dense, 7.);
        let mut g = graph(runtime.clone(), 3, dense);
        g.outputs.limit = 0;
        let f = g.forward().unwrap();
        for _ in 0..8 {
            drop(backward(&mut g, &f, 1.));
        }
        assert!(g.outputs.versions.is_empty());
        assert_eq!((g.outputs.allocations, g.outputs.reuses), (8, 0));
    }
}
