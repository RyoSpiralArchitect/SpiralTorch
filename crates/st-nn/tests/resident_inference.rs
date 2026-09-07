#![cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]

use st_backend_wgpu::{
    resident_dense::DenseError,
    resident_matmul::{MatmulAccumulation, MatmulKernel},
    runtime,
};
use st_nn::{layers::Gelu, module::Module, resident::InferencePlan, Linear, Sequential};
use st_tensor::{NdLayout, Tensor};

fn enabled() -> bool {
    std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() == Ok("1")
}

fn cpu_policy() -> st_nn::BackendPolicyGuard {
    st_nn::push_backend_policy(st_nn::BackendPolicy::from_device_caps(
        st_core::backend::device_caps::DeviceCaps::cpu(),
    ))
}

fn model(depth: usize, width: usize) -> Sequential {
    let mut model = Sequential::new();
    for i in 0..depth {
        model.push(Linear::new(format!("linear_{i}"), width, width).unwrap());
        model.push(Gelu::new());
    }
    let mut counter = 0usize;
    model
        .visit_parameters_mut(&mut |parameter| {
            for value in parameter.value_mut().data_mut() {
                *value = ((counter * 17 % 23) as f32 - 11.0) / 64.0;
                counter += 1;
            }
            Ok(())
        })
        .unwrap();
    model
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && (a - b).abs() <= 1e-5 + 1e-4 * b.abs(),
            "element {i}: gpu={a}, cpu={b}"
        );
    }
}

#[test]
fn existing_sequential_runs_32_ops_with_owned_nd_snapshots() {
    if !enabled() {
        return;
    }
    let _cpu = cpu_policy();
    let (runtime, _) = runtime::ensure_default_runtime_blocking("nn.resident.tests").unwrap();
    assert_ne!(format!("{:?}", runtime.adapter_info().device_type), "Cpu");
    let model = model(16, 7);
    let plan =
        InferencePlan::from_module(&model, NdLayout::contiguous(&[2, 3, 7]).unwrap()).unwrap();
    assert_eq!(plan.source_operation_count(), 32);
    assert_eq!(plan.stage_count(), 16);
    let input = Tensor::from_fn(6, 7, |r, c| (r as f32 - c as f32) / 8.0).unwrap();
    let expected = model.forward(&input).unwrap();
    let mut gpu = plan.compile_wgpu(runtime).unwrap();
    assert_eq!(gpu.output_layout().shape(), &[2, 3, 7]);
    assert!(matches!(gpu.dispatch(), Err(DenseError::MissingInput)));
    assert!(matches!(gpu.snapshot(), Err(DenseError::StaleOutput)));
    gpu.upload(input.data()).unwrap();
    assert!(matches!(gpu.snapshot(), Err(DenseError::StaleOutput)));
    gpu.dispatch().unwrap();
    let first = gpu.snapshot().unwrap();
    let generation = gpu.generation();
    assert!(gpu.upload(&[f32::NAN; 42]).is_err());
    assert!(gpu.upload(&[0.0; 41]).is_err());
    assert_eq!(gpu.generation(), generation);
    let unchanged = gpu.snapshot().unwrap();
    let next = Tensor::from_fn(6, 7, |r, c| (r + c) as f32 / 16.0).unwrap();
    let expected_next = model.forward(&next).unwrap();
    gpu.upload(next.data()).unwrap();
    gpu.dispatch().unwrap();
    let second = gpu.snapshot().unwrap();
    drop(gpu);
    assert_eq!(first.layout().shape(), &[2, 3, 7]);
    assert_eq!(first.generation(), generation);
    assert_close(&first.read().unwrap(), expected.data());
    assert_close(&unchanged.read().unwrap(), expected.data());
    assert_close(&second.read().unwrap(), expected_next.data());
}

#[test]
fn changing_width_preserves_leading_axes_and_vector_inputs() {
    if !enabled() {
        return;
    }
    let _cpu = cpu_policy();
    let (runtime, _) = runtime::ensure_default_runtime_blocking("nn.resident.widths").unwrap();
    let mut model = Sequential::new();
    model.push(Linear::new("up", 4, 7).unwrap());
    model.push(Gelu::new());
    model.push(Linear::new("down", 7, 3).unwrap());
    model
        .visit_parameters_mut(&mut |parameter| {
            for (i, value) in parameter.value_mut().data_mut().iter_mut().enumerate() {
                *value = (i as f32 % 13.0 - 6.0) / 16.0;
            }
            Ok(())
        })
        .unwrap();
    for shape in [vec![4], vec![2, 3, 4]] {
        let layout = NdLayout::contiguous(&shape).unwrap();
        let input = Tensor::from_fn(layout.len() / 4, 4, |r, c| (r + c) as f32 / 8.0).unwrap();
        let expected = model.forward(&input).unwrap();
        let plan = InferencePlan::from_module(&model, layout).unwrap();
        let mut expected_shape = shape;
        *expected_shape.last_mut().unwrap() = 3;
        for kernel in [MatmulKernel::Scalar, MatmulKernel::Register2x2] {
            for accumulation in [
                MatmulAccumulation::Sequential,
                MatmulAccumulation::Tiled,
                MatmulAccumulation::Compensated,
            ] {
                let mut gpu = plan
                    .compile_wgpu_with_options(
                        runtime.clone(),
                        Default::default(),
                        kernel,
                        accumulation,
                    )
                    .unwrap();
                assert_eq!(gpu.output_layout().shape(), expected_shape);
                gpu.upload(input.data()).unwrap();
                gpu.dispatch().unwrap();
                assert_close(&gpu.snapshot().unwrap().read().unwrap(), expected.data());
            }
        }
    }
}

#[test]
fn fused_gelu_cannot_hide_an_intermediate_failure() {
    if !enabled() {
        return;
    }
    let _cpu = cpu_policy();
    let (runtime, _) = runtime::ensure_default_runtime_blocking("nn.resident.guards").unwrap();
    let mut model = Sequential::new();
    model.push(Linear::new("guard", 1, 1).unwrap());
    model.push(Gelu::new());
    model
        .visit_parameters_mut(&mut |parameter| {
            let value = if parameter.name().ends_with("weight") {
                1.0
            } else {
                0.0
            };
            parameter.value_mut().data_mut().fill(value);
            Ok(())
        })
        .unwrap();
    let plan =
        InferencePlan::from_module(&model, NdLayout::contiguous(&[1, 1, 1]).unwrap()).unwrap();
    let mut gpu = plan.compile_wgpu(runtime).unwrap();
    for value in [1e20f32, -1e20, 1e13, -1e13] {
        assert!(model
            .forward(&Tensor::from_vec(1, 1, vec![value]).unwrap())
            .is_err());
        gpu.upload(&[value]).unwrap();
        gpu.dispatch().unwrap();
        let failure = gpu.snapshot().unwrap();
        gpu.upload(&[0.25]).unwrap();
        gpu.dispatch().unwrap();
        let success = gpu.snapshot().unwrap();
        assert!(matches!(
            failure.read(),
            Err(DenseError::NonFiniteIntermediate { stage: 0, .. })
        ));
        let expected = model
            .forward(&Tensor::from_vec(1, 1, vec![0.25]).unwrap())
            .unwrap();
        assert_close(&success.read().unwrap(), expected.data());
    }
}
