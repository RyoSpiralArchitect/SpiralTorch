#[cfg(feature = "wgpu_dense")]
#[test]
fn ready_dense_provider_is_reachable_through_tensor_layer_norm() {
    use st_tensor::execution_capability::{
        observe_tensor_execution_capability, TensorExecutionCapabilityStatus,
    };
    use st_tensor::{LayerNormBackend, Tensor, TensorExecutionBackend, TensorExecutionWorkload};

    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        eprintln!("set SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 to require live WGPU dispatch");
        return;
    }
    let capability = observe_tensor_execution_capability(
        TensorExecutionBackend::Wgpu,
        TensorExecutionWorkload::LayerNorm { rows: 1, cols: 4 },
    );
    println!(
        "wgpu_alias={} capability={capability:?}",
        cfg!(feature = "wgpu")
    );
    assert_eq!(capability.status, TensorExecutionCapabilityStatus::Ready);
    let input = Tensor::from_vec(1, 4, vec![10000.0, 10001.0, 10002.0, 10003.0]).unwrap();
    let gamma = Tensor::from_vec(1, 4, vec![1.0; 4]).unwrap();
    let beta = Tensor::zeros(1, 4).unwrap();
    input
        .layer_norm_affine_with_backend(&gamma, &beta, 1e-5, LayerNormBackend::GpuWgpu)
        .expect("a Ready dense provider must be reachable without requiring the wgpu alias");
}

#[cfg(not(feature = "wgpu_dense"))]
#[test]
fn fractional_or_cpu_only_build_does_not_claim_dense_dispatch() {
    use st_tensor::execution::{push_accelerator_fallback, AcceleratorFallback};
    use st_tensor::execution_capability::{
        observe_tensor_execution_capability, TensorExecutionCapabilityStatus,
    };
    use st_tensor::{Tensor, TensorExecutionBackend, TensorExecutionWorkload, TensorUtilBackend};

    let capability = observe_tensor_execution_capability(
        TensorExecutionBackend::Wgpu,
        TensorExecutionWorkload::LayerNorm { rows: 1, cols: 4 },
    );
    assert_eq!(capability.status, TensorExecutionCapabilityStatus::NotBuilt);
    assert_eq!(capability.ready_proof, None);
    let input = Tensor::from_vec(1, 2, vec![1.0, 2.0]).unwrap();
    let _strict = push_accelerator_fallback(AcceleratorFallback::Forbid);
    assert!(input
        .scale_with_backend(2.0, TensorUtilBackend::GpuWgpu)
        .is_err());
    assert_eq!(
        input
            .scale_with_backend(2.0, TensorUtilBackend::Cpu)
            .unwrap()
            .data(),
        &[2.0, 4.0]
    );
}

#[test]
fn requested_feature_isolation_is_preserved() {
    if cfg!(feature = "wgpu") {
        assert_ne!(
            std::env::var("SPIRALTORCH_EXPECT_NO_WGPU_ALIAS").as_deref(),
            Ok("1"),
            "a dependency enabled the compatibility alias"
        );
    }
}

#[cfg(feature = "wgpu_dense")]
mod dense {
    use st_tensor::execution::{push_accelerator_fallback, AcceleratorFallback};
    use st_tensor::execution_capability::{
        observe_tensor_execution_capability, TensorExecutionCapabilityStatus,
    };
    use st_tensor::{
        set_thread_meta_observer, AttentionBackend, HardmaxBackend, LayerNormBackend,
        MatmulBackend, PackedB, PureResult, SoftmaxBackend, Tensor, TensorExecutionBackend,
        TensorExecutionWorkload, TensorOpMetaObserver, TensorUtilBackend, TensorUtilOperation,
        Tile,
    };
    use std::sync::{Arc, Mutex};

    struct RestoreObserver(Option<TensorOpMetaObserver>);
    impl Drop for RestoreObserver {
        fn drop(&mut self) {
            set_thread_meta_observer(self.0.take());
        }
    }

    fn live() -> bool {
        if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
            eprintln!("set SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 to require live WGPU dispatch");
            return false;
        }
        assert!(
            st_tensor::wgpu_dense::is_available(),
            "live test requires a GPU"
        );
        true
    }

    fn close(actual: &Tensor, expected: &Tensor) {
        assert_eq!(actual.shape(), expected.shape());
        for (&actual, &expected) in actual.data().iter().zip(expected.data()) {
            assert!(
                actual.is_finite() && (actual - expected).abs() <= 3e-5 * (1.0 + expected.abs()),
                "{actual} != {expected}"
            );
        }
    }

    fn check(
        operation: &'static str,
        workload: TensorExecutionWorkload,
        expected: &Tensor,
        run: impl FnOnce() -> PureResult<Tensor>,
    ) {
        let capability =
            observe_tensor_execution_capability(TensorExecutionBackend::Wgpu, workload);
        assert_eq!(
            capability.status,
            TensorExecutionCapabilityStatus::Ready,
            "{operation}"
        );
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&events);
        let _observer = RestoreObserver(set_thread_meta_observer(Some(Arc::new(move |event| {
            if event.op_name == operation && event.data.get("execution_receipt").is_some() {
                captured
                    .lock()
                    .unwrap()
                    .push(event.data["execution_receipt"].clone());
            }
        }))));
        let _strict = push_accelerator_fallback(AcceleratorFallback::Forbid);
        close(
            &run().unwrap_or_else(|error| panic!("{operation}: {error}")),
            expected,
        );
        let receipts = events.lock().unwrap();
        assert_eq!(receipts.len(), 1, "{operation}: one completed dispatch");
        assert_eq!(receipts[0]["requested_backend"], "wgpu", "{operation}");
        assert_eq!(receipts[0]["executed_backend"], "wgpu", "{operation}");
        assert_eq!(receipts[0]["kernel_backend"], "wgpu_dense", "{operation}");
        assert_eq!(receipts[0]["route_status"], "direct", "{operation}");
    }

    #[test]
    fn granular_owner_exposes_typed_gpu_selectors() {
        assert_eq!(MatmulBackend::GpuWgpu.execution_id(), "wgpu");
        assert_eq!(SoftmaxBackend::GpuWgpu.execution_id(), "wgpu");
        assert_eq!(HardmaxBackend::GpuWgpu.to_string(), "wgpu");
    }

    #[test]
    fn ready_component_families_reach_gpu_without_the_alias() {
        if !live() {
            return;
        }
        let input = Tensor::from_vec(2, 4, vec![1.0, -2.0, 3.0, 0.5, -1.0, 2.0, 0.5, 1.0]).unwrap();
        let rhs = Tensor::from_vec(4, 3, (0..12).map(|i| i as f32 * 0.25 - 1.0).collect()).unwrap();
        let expected = input
            .matmul_with_backend(&rhs, MatmulBackend::CpuNaive)
            .unwrap();
        check(
            "matmul",
            TensorExecutionWorkload::DenseMatmul {
                rows: 2,
                inner: 4,
                cols: 3,
            },
            &expected,
            || input.matmul_with_backend(&rhs, MatmulBackend::GpuWgpu),
        );
        let packed = PackedB::from_tensor(&rhs, Tile::col_major()).unwrap();
        check(
            "matmul_prepacked",
            TensorExecutionWorkload::PrepackedMatmul {
                rows: 2,
                inner: 4,
                cols: 3,
                bias: false,
            },
            &expected,
            || input.matmul_prepacked_with_backend(&packed, MatmulBackend::GpuWgpu),
        );
        let gamma = Tensor::from_vec(1, 4, vec![1.0; 4]).unwrap();
        let beta = Tensor::zeros(1, 4).unwrap();
        let expected = input
            .layer_norm_affine_with_backend(&gamma, &beta, 1e-5, LayerNormBackend::Cpu)
            .unwrap();
        check(
            "layer_norm",
            TensorExecutionWorkload::LayerNorm { rows: 2, cols: 4 },
            &expected,
            || input.layer_norm_affine_with_backend(&gamma, &beta, 1e-5, LayerNormBackend::GpuWgpu),
        );
        let expected = input.row_softmax_with_backend(SoftmaxBackend::Cpu).unwrap();
        check(
            "row_softmax",
            TensorExecutionWorkload::Softmax { rows: 2, cols: 4 },
            &expected,
            || input.row_softmax_with_backend(SoftmaxBackend::GpuWgpu),
        );
        let attention = |backend| {
            input.scaled_dot_attention_with_backend(&input, &input, 1, 2, 0.5, None, None, backend)
        };
        let expected = attention(AttentionBackend::Cpu).unwrap();
        check(
            "scaled_dot_attention",
            TensorExecutionWorkload::Attention {
                contexts: 1,
                sequence: 2,
                head_dim: 4,
                z_bias: false,
                attn_bias: false,
            },
            &expected,
            || attention(AttentionBackend::GpuWgpu),
        );
        let utility = |operation| TensorExecutionWorkload::TensorUtil {
            operation,
            rows: 2,
            cols: 4,
        };
        let expected = input
            .scale_with_backend(0.25, TensorUtilBackend::Cpu)
            .unwrap();
        check(
            "scale",
            utility(TensorUtilOperation::Scale),
            &expected,
            || input.scale_with_backend(0.25, TensorUtilBackend::GpuWgpu),
        );
        let bias = [0.5, -0.25, 0.0, 1.0];
        let mut expected = input.clone();
        expected
            .add_row_inplace_with_backend(&bias, TensorUtilBackend::Cpu)
            .unwrap();
        check(
            "add_row_inplace",
            utility(TensorUtilOperation::AddRow),
            &expected,
            || {
                let mut output = input.clone();
                output.add_row_inplace_with_backend(&bias, TensorUtilBackend::GpuWgpu)?;
                Ok(output)
            },
        );
        let expected = Tensor::from_vec(
            1,
            4,
            input
                .try_sum_axis0_with_backend(TensorUtilBackend::Cpu)
                .unwrap(),
        )
        .unwrap();
        check(
            "sum_axis0",
            utility(TensorUtilOperation::SumAxis0),
            &expected,
            || {
                Tensor::from_vec(
                    1,
                    4,
                    input.try_sum_axis0_with_backend(TensorUtilBackend::GpuWgpu)?,
                )
            },
        );
        let expected = Tensor::from_vec(
            1,
            4,
            input
                .try_sum_axis0_scaled_with_backend(0.5, TensorUtilBackend::Cpu)
                .unwrap(),
        )
        .unwrap();
        check(
            "sum_axis0_scaled",
            utility(TensorUtilOperation::SumAxis0Scaled),
            &expected,
            || {
                Tensor::from_vec(
                    1,
                    4,
                    input.try_sum_axis0_scaled_with_backend(0.5, TensorUtilBackend::GpuWgpu)?,
                )
            },
        );
    }

    #[test]
    fn granular_hardmax_selector_reaches_the_existing_gpu_kernel() {
        if !live() {
            return;
        }
        let input = Tensor::from_vec(2, 3, vec![1.0, -2.0, 3.0, -1.0, 2.0, 0.5]).unwrap();
        let expected = input.row_hardmax_with_backend(HardmaxBackend::Cpu).unwrap();
        let backends = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&backends);
        let _observer = RestoreObserver(set_thread_meta_observer(Some(Arc::new(move |event| {
            if event.op_name == "row_hardmax" {
                captured.lock().unwrap().push(event.data["backend"].clone());
            }
        }))));
        let _strict = push_accelerator_fallback(AcceleratorFallback::Forbid);
        close(
            &input
                .row_hardmax_with_backend(HardmaxBackend::GpuWgpu)
                .unwrap(),
            &expected,
        );
        assert_eq!(
            *backends.lock().unwrap(),
            vec![serde_json::json!("wgpu_dense")]
        );
    }
}
