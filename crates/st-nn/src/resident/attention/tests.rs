use super::*;

fn parameters() -> [(Tensor, Tensor); 4] {
    [0., 0.4, 0.8, 1.2].map(|phase| {
        (
            Tensor::from_fn(6, 6, |r, c| ((r * 6 + c) as f32 * 0.17 + phase).sin() * 0.2).unwrap(),
            Tensor::from_fn(1, 6, |_, c| (c as f32 + phase).cos() * 0.05).unwrap(),
        )
    })
}

fn plan(parameters: &[(Tensor, Tensor); 4], mask: AttentionMask) -> AttentionInferencePlan {
    AttentionInferencePlan::from_parameters(
        NdLayout::contiguous(&[2, 4, 6]).unwrap(),
        2,
        mask,
        parameters.each_ref().map(|(w, b)| (w, b)),
    )
    .unwrap()
}

#[test]
fn plan_fuses_qkv_once_with_logical_layout_and_freezes_source_parameters() {
    let mut params = parameters();
    params[1].0 = params[1].0.to_layout(Layout::ColMajor).unwrap();
    let frozen = plan(&params, AttentionMask::Causal { query_offset: 0 });
    assert_eq!(frozen.qkv.output_layout().shape(), &[2, 4, 18]);
    assert_eq!(frozen.output_layout().shape(), &[2, 4, 6]);
    let expected_key = params[1].0.to_layout(Layout::RowMajor).unwrap();
    assert_eq!(
        &frozen.qkv.stages[0].weight.data()[6..12],
        &expected_key.data()[..6]
    );
    let old = frozen.qkv.stages[0].weight.data().to_vec();
    params[0].0.data_mut().fill(10.);
    assert_eq!(frozen.qkv.stages[0].weight.data(), old);
    assert_ne!(
        plan(&params, AttentionMask::None).qkv.stages[0]
            .weight
            .data(),
        old
    );
}

#[test]
fn invalid_shapes_heads_offsets_and_parameters_fail_without_a_device() {
    let params = parameters();
    let make = |shape: &[usize], heads, mask| {
        AttentionInferencePlan::from_parameters(
            NdLayout::contiguous(shape).unwrap(),
            heads,
            mask,
            params.each_ref().map(|(w, b)| (w, b)),
        )
    };
    for (shape, heads) in [
        (&[2, 4, 6][..], 0),
        (&[2, 4, 6], 4),
        (&[0, 4, 6], 2),
        (&[4, 6], 2),
        (&[2, 4, 5], 2),
    ] {
        assert!(make(shape, heads, AttentionMask::None).is_err());
    }
    assert!(make(&[2, 4, 6], 2, AttentionMask::Causal { query_offset: 1 }).is_err());
    let mut bad = parameters();
    bad[0].1.data_mut()[0] = f32::NAN;
    assert!(AttentionInferencePlan::from_parameters(
        NdLayout::contiguous(&[2, 4, 6]).unwrap(),
        2,
        AttentionMask::None,
        bad.each_ref().map(|(w, b)| (w, b))
    )
    .is_err());
}

#[test]
fn original_linears_lower_with_their_biases_not_a_new_parameter_set() {
    let linears = ["q", "k", "v", "o"].map(|name| Linear::new(name, 6, 6).unwrap());
    let plan = AttentionInferencePlan::from_linears(
        NdLayout::contiguous(&[1, 3, 6]).unwrap(),
        2,
        AttentionMask::None,
        linears.each_ref(),
    )
    .unwrap();
    assert_eq!(
        &plan.qkv.stages[0].weight.data()[..6],
        &linears[0].weight().value().data()[..6]
    );
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
mod gpu {
    use super::*;
    use crate::z_rba::{
        attention::{SimpleZFrame, ZIndex, ZMetricWeights, ZRBFAttention},
        ZTensor,
    };
    use st_backend_wgpu::{
        resident_matmul::{MatmulKernel, MatmulTile},
        resident_tensor::{ResidentTensor, TensorDevice, TensorError as GpuError},
        runtime,
    };

    fn device() -> Option<TensorDevice> {
        if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
            return None;
        }
        let (runtime, _) =
            runtime::ensure_default_runtime_blocking("nn.attention.chain.tests").unwrap();
        assert_ne!(format!("{:?}", runtime.adapter_info().device_type), "Cpu");
        Some(TensorDevice::new(runtime).unwrap())
    }
    fn read(t: &ResidentTensor) -> Vec<f32> {
        t.snapshot().unwrap().read().unwrap()
    }
    fn close(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for (&a, &b) in a.iter().zip(b) {
            assert!(
                a.is_finite() && (a - b).abs() <= 3e-6 + 3e-5 * b.abs(),
                "{a} != {b}"
            );
        }
    }

    #[test]
    fn zrba_geometry_and_fused_projection_chain_match_original_mean() {
        let Some(device) = device() else { return };
        let _lock = crate::test_global_state_lock();
        let _cpu = crate::execution::push_backend_policy(
            crate::execution::BackendPolicy::from_device_caps_with_config(
                st_core::backend::device_caps::DeviceCaps::cpu(),
                Default::default(),
            ),
        );
        let layer = ZRBFAttention::new(6, 2, ZMetricWeights::default(), true).unwrap();
        let indices: Vec<_> = (0..5)
            .map(|i| ZIndex {
                band: i % 3,
                sheet: i % 2,
                echo: i,
            })
            .collect();
        let frame = SimpleZFrame::new(3, 2, 8);
        let values: Vec<_> = (0..30).map(|i| (i as f32 * 0.37).sin()).collect();
        let ztensor = ZTensor::new(
            Tensor::from_vec(5, 6, values.clone()).unwrap(),
            Tensor::zeros(5, 6).unwrap(),
            indices.clone(),
        )
        .unwrap();
        let expected = layer.forward(&ztensor, &frame).unwrap().mean;
        let bias = layer.kernel_bias(&frame, &indices, &indices).unwrap();
        let mut compiled = layer
            .mean_inference_plan(
                NdLayout::contiguous(&[1, 5, 6]).unwrap(),
                AttentionMask::None,
            )
            .unwrap()
            .compile_wgpu(device.runtime().clone())
            .unwrap();
        let input = device.upload(&[1, 5, 6], &values).unwrap();
        let bias = device.upload(&[1, 2, 5, 5], bias.data()).unwrap();
        let result = compiled.forward(&input, None, Some(&bias)).unwrap();
        close(&read(&result), expected.data());
        let plain = compiled.forward(&input, None, None).unwrap();
        assert!(read(&plain)
            .iter()
            .zip(read(&result))
            .any(|(a, b)| (a - b).abs() > 1e-8));
        drop((input, bias, compiled, layer));
        close(&read(&result), expected.data());
    }

    #[test]
    fn composed_chain_retains_outputs_and_failed_guards_across_reuse() {
        let Some(device) = device() else { return };
        for (tile, kernel) in [
            (MatmulTile::default(), MatmulKernel::Scalar),
            (MatmulTile::default(), MatmulKernel::Register2x2),
            (
                MatmulTile::new(16, 16, 16).unwrap(),
                MatmulKernel::Register2x2,
            ),
        ] {
            check_guards_and_reuse(&device, tile, kernel);
        }
    }

    fn check_guards_and_reuse(device: &TensorDevice, tile: MatmulTile, kernel: MatmulKernel) {
        let params = parameters();
        let mut compiled = plan(&params, AttentionMask::Causal { query_offset: 0 })
            .compile_wgpu_with_options(device.runtime().clone(), tile, kernel, Default::default())
            .unwrap();
        let input = device
            .upload(&[4, 2, 6], &[0.2; 48])
            .unwrap()
            .permute(&[1, 0, 2])
            .unwrap();
        let first = compiled.forward(&input, None, None).unwrap();
        let expected = read(&first);
        let huge = device.upload(&[2, 4, 6], &[f32::MAX; 48]).unwrap();
        let bad = huge
            .mul(&device.upload(&[2, 4, 6], &[2.; 48]).unwrap())
            .unwrap();
        let failed = compiled.forward(&bad, None, None).unwrap();
        for _ in 0..6 {
            close(
                &read(&compiled.forward(&input, None, None).unwrap()),
                &expected,
            );
        }
        close(&read(&first), &expected);
        assert!(matches!(
            failed.snapshot().unwrap().read(),
            Err(GpuError::NonFinite)
        ));
        let bias = device.upload(&[2, 2, 4], &[0.; 16]).unwrap();
        let biased = compiled.forward(&input, Some(&bias), None).unwrap();
        close(&read(&biased), &expected);
        assert!(compiled
            .forward(
                &input.contiguous().unwrap().reshape(&[8, 6]).unwrap(),
                None,
                None
            )
            .is_err());
        let wrong_bias = device.upload(&[4], &[0.; 4]).unwrap();
        assert!(compiled.forward(&input, Some(&wrong_bias), None).is_err());
    }

    #[test]
    fn invalid_register_tile_is_rejected_without_scalar_fallback() {
        let Some(device) = device() else { return };
        assert!(plan(&parameters(), AttentionMask::None)
            .compile_wgpu_with_options(
                device.runtime().clone(),
                MatmulTile::new(7, 8, 16).unwrap(),
                MatmulKernel::Register2x2,
                Default::default(),
            )
            .is_err());
    }
}
