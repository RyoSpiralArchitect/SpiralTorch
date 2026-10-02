use super::*;

#[test]
fn guards_require_complete_finite_coverage() {
    assert!(validate_guards(&[vec![0], vec![0]], 2).is_ok());
    assert!(matches!(
        validate_guards(&[vec![0, 0]], 1),
        Err(TensorError::Readback)
    ));
    assert!(matches!(
        validate_guards(&[], 1),
        Err(TensorError::Readback)
    ));
    assert!(matches!(
        validate_guards(&[vec![INVALID_TENSOR_FLAG]], 1),
        Err(TensorError::NonFinite)
    ));
}

#[test]
fn capture_scope_detaches_after_error_and_unwinding() {
    let slot = ProfileSlot::default();
    let scope = CaptureScope::new(&slot).unwrap();
    assert!(CaptureScope::new(&slot).is_err());
    assert!(scope.finish().is_err());
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _scope = CaptureScope::new(&slot).unwrap();
        panic!("diagnostic operation failed");
    }));
    assert!(panicked.is_err());
    assert!(slot.0.lock().unwrap().is_none());
    drop(CaptureScope::new(&slot).unwrap());
}

#[test]
fn owning_capture_budget_is_bounded_and_released() {
    let slot = Shared::new(ProfileSlot::default());
    let mut held = Vec::new();
    for _ in 0..MAX_PENDING_CAPTURES {
        held.push(CapturePermit::acquire(slot.clone()).unwrap());
    }
    assert!(CapturePermit::acquire(slot.clone()).is_err());
    held.pop();
    held.push(CapturePermit::acquire(slot.clone()).unwrap());
    drop(held);
    assert_eq!(slot.1.load(Ordering::Acquire), 0);
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::*;

    fn device() -> Option<TensorDevice> {
        if std::env::var("SPIRALTORCH_RUN_WGPU_TIMESTAMP_TESTS").as_deref() != Ok("1") {
            return None;
        }
        let runtime = WgpuRuntime::request_profiled_headless_blocking("conv.profile.test").unwrap();
        assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
        Some(TensorDevice::new(runtime).unwrap())
    }

    fn tensors(device: &TensorDevice) -> [ResidentTensor; 3] {
        [
            device.upload(&[1, 1, 2, 2], &[1., 2., 3., 4.]).unwrap(),
            device.upload(&[1, 1, 1, 1], &[0.5]).unwrap(),
            device
                .upload(&[1, 1, 2, 2], &[0.25, 0.5, 0.75, 1.])
                .unwrap(),
        ]
    }

    fn run(tensors: &[ResidentTensor; 3]) -> Result<[ResidentTensor; 3], TensorError> {
        tensors[0].conv2d_vjp(&tensors[1], &tensors[2], (1, 1), (0, 0), (1, 1))
    }

    #[test]
    fn profiles_existing_passes_without_changing_values_and_owns_pending_data() {
        let Some(device) = device() else { return };
        let slot = Shared::downgrade(&device.runtime().context().tensor_profile);
        let graph_device = TensorDevice::new(device.runtime().clone()).unwrap();
        let tensors = tensors(&graph_device);
        let expected = run(&tensors)
            .unwrap()
            .map(|t| t.snapshot().unwrap().read().unwrap());
        let (actual, pending) = device.profile_convolution_vjps(|| run(&tensors)).unwrap();
        assert_eq!(
            actual.map(|t| t.snapshot().unwrap().read().unwrap()),
            expected
        );
        assert!(device.profile_slot().0.lock().unwrap().is_none());
        let (_, cancelled) = device.profile_convolution_vjps(|| run(&tensors)).unwrap();
        drop(cancelled);
        drop(tensors);
        drop(graph_device);
        drop(device);
        let profile = pending.read().unwrap();
        assert!(
            slot.upgrade().is_none(),
            "capture retained a context ownership cycle"
        );
        assert_eq!(profile.operations.len(), 1);
        let operation = &profile.operations[0];
        assert_eq!(operation.geometry.kind, "dense");
        assert_eq!(operation.geometry.input, [1, 1, 2, 2]);
        assert_eq!(operation.timestamps.passes.len(), 3);
        assert!(operation
            .timestamps
            .passes
            .iter()
            .all(|p| p.elapsed_ns >= 0. && p.elapsed_ns.is_finite()));
    }

    #[test]
    fn unsupported_and_nested_capture_do_not_run_their_closure() {
        let Some(device) = device() else { return };
        let ordinary = TensorDevice::new(
            pollster::block_on(WgpuRuntime::request_headless("conv.profile.ordinary")).unwrap(),
        )
        .unwrap();
        let absent = ordinary.profile_convolution_vjps(|| -> Result<(), TensorError> {
            panic!("unsupported closure ran")
        });
        assert!(matches!(
            absent,
            Err(TensorError::Runtime(
                WgpuRuntimeError::TimestampQueriesUnavailable
            ))
        ));
        let tensors = tensors(&device);
        let (_, pending) = device
            .profile_convolution_vjps(|| {
                let nested = device.profile_convolution_vjps(|| -> Result<(), TensorError> {
                    panic!("nested closure ran")
                });
                assert!(matches!(
                    nested,
                    Err(TensorError::Runtime(
                        WgpuRuntimeError::TimestampProfilingBusy
                    ))
                ));
                std::thread::scope(|scope| {
                    assert!(matches!(
                        scope.spawn(|| run(&tensors)).join().unwrap(),
                        Err(TensorError::Runtime(
                            WgpuRuntimeError::TimestampProfilingBusy
                        ))
                    ));
                });
                run(&tensors)
            })
            .unwrap();
        assert_eq!(pending.read().unwrap().operations.len(), 1);
    }

    #[test]
    fn numerical_guard_and_budget_fail_closed_without_poisoning_reuse() {
        let Some(device) = device() else { return };
        let mut invalid = tensors(&device);
        invalid[0] = device.upload(&[1, 1, 2, 2], &[f32::MAX; 4]).unwrap();
        let (_, pending) = device.profile_convolution_vjps(|| run(&invalid)).unwrap();
        assert!(matches!(pending.read(), Err(TensorError::NonFinite)));
        let tensors = tensors(&device);
        let too_many = device.profile_convolution_vjps(|| {
            for _ in 0..=MAX_OPERATIONS {
                run(&tensors)?;
            }
            Ok::<_, TensorError>(())
        });
        assert!(matches!(
            too_many,
            Err(TensorError::Limit("convolution profile operation budget"))
        ));
        assert!(device.profile_slot().0.lock().unwrap().is_none());
        let (_, pending) = device.profile_convolution_vjps(|| run(&tensors)).unwrap();
        assert_eq!(pending.read().unwrap().operations.len(), 1);
    }
}
