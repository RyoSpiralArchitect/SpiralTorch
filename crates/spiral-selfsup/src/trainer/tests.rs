use super::distributed::st_distributed::DistributedError;
use super::{CpuDevice, DistributedDevice, MetricReduce, TrainingDevice, TrainingDeviceError};
use st_core::distributed::{AccumulatorSynchronizer, AccumulatorSynchronizerCheckpoint};
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

#[cfg(feature = "wgpu")]
#[test]
fn distributed_mean_scaling_obeys_the_core_wgpu_policy() {
    use st_core::backend::device_caps::DeviceCaps;
    use st_core::backend::execution::{
        push_backend_policy, AcceleratorFallback, BackendPolicy, ExecutionConfig,
    };
    use std::sync::Mutex;

    let events = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&events);
    let previous = st_tensor::set_thread_meta_observer(Some(Arc::new(move |event| {
        captured
            .lock()
            .unwrap()
            .push((event.op_name, event.data.clone()));
    })));
    let policy = BackendPolicy::from_device_caps_with_config(
        DeviceCaps::wgpu(32, true, 256),
        ExecutionConfig::new(AcceleratorFallback::Allow, 1),
    );
    let device = DistributedDevice::new("policy-routed-mean", 0, 1).unwrap();
    let mut gradients = vec![1.5, -2.0, 0.25, 4.0];
    {
        let _guard = push_backend_policy(policy);
        device.synchronize_gradients(&mut gradients).unwrap();
    }
    st_tensor::set_thread_meta_observer(previous);

    assert_eq!(gradients, vec![1.5, -2.0, 0.25, 4.0]);
    let events = events.lock().unwrap();
    let scale = events
        .iter()
        .find(|(op_name, _)| *op_name == "scale")
        .expect("distributed mean scale metadata");
    assert_eq!(scale.1["requested_backend"], "wgpu");
    assert!(matches!(
        scale.1["backend"].as_str(),
        Some("cpu" | "wgpu_dense")
    ));
}

#[test]
fn distributed_device_all_reduce_averages_gradients() {
    let world = 2;
    let mut handles = Vec::new();
    for rank in 0..world {
        handles.push(std::thread::spawn(move || {
            let device = DistributedDevice::new("unit-test", rank, world).unwrap();
            let mut gradients = vec![rank as f32 + 1.0, 2.0];
            device
                .synchronize_gradients(&mut gradients)
                .expect("all-reduce should succeed");
            gradients
        }));
    }

    let mut results = Vec::new();
    for handle in handles {
        results.push(handle.join().unwrap());
    }

    for gradients in results {
        assert_eq!(gradients, vec![1.5, 2.0]);
    }
}

#[test]
fn distributed_device_implements_accumulator_synchronizer_bridge() {
    let world = 2;
    let mut handles = Vec::new();
    for rank in 0..world {
        handles.push(std::thread::spawn(move || {
            let device = DistributedDevice::new("accumulator-bridge-test", rank, world).unwrap();
            let mut gradients = vec![rank as f32 + 1.0, 4.0];
            AccumulatorSynchronizer::synchronize_accumulators(&device, &mut gradients)
                .expect("accumulator bridge should all-reduce");
            gradients
        }));
    }

    for handle in handles {
        assert_eq!(handle.join().unwrap(), vec![1.5, 4.0]);
    }
}

#[test]
fn cpu_accumulator_checkpoint_has_a_verifiable_provider_identity() {
    let device = CpuDevice::new();
    let encoded = serde_json::to_string(&AccumulatorSynchronizer::checkpoint(&device).unwrap())
        .expect("CPU checkpoint should serialize");
    let checkpoint: AccumulatorSynchronizerCheckpoint =
        serde_json::from_str(&encoded).expect("CPU checkpoint should deserialize");

    assert_eq!(checkpoint.provider, "spiral-selfsup.cpu_accumulator.v1");
    assert!(checkpoint.state.is_none());
    AccumulatorSynchronizer::validate_checkpoint(&device, &checkpoint).unwrap();
}

#[test]
fn distributed_accumulator_checkpoint_verifies_provider_state() {
    let checkpoint = {
        let device = DistributedDevice::new("checkpoint-provider", 0, 1)
            .unwrap()
            .with_collective_timeout(Duration::from_millis(37));
        AccumulatorSynchronizer::checkpoint(&device).unwrap()
    };
    assert_eq!(
        checkpoint.provider,
        "spiral-selfsup.distributed_accumulator.v1"
    );
    assert_eq!(
        checkpoint.state.as_ref().unwrap()["group_id"],
        "checkpoint-provider"
    );
    assert_eq!(
        checkpoint.state.as_ref().unwrap()["strategy"],
        "all_reduce_mean"
    );

    let restored = DistributedDevice::new("checkpoint-provider", 0, 1)
        .unwrap()
        .with_collective_timeout(Duration::from_millis(37));
    AccumulatorSynchronizer::validate_checkpoint(&restored, &checkpoint).unwrap();
    drop(restored);

    let wrong = DistributedDevice::new("checkpoint-provider-wrong", 0, 1)
        .unwrap()
        .with_collective_timeout(Duration::from_millis(37));
    assert!(
        AccumulatorSynchronizer::validate_checkpoint(&wrong, &checkpoint)
            .unwrap_err()
            .to_string()
            .contains("distributed checkpoint state mismatch")
    );
}

#[test]
fn distributed_device_aggregates_metrics() {
    let world = 3;
    let mut handles = Vec::new();
    for rank in 0..world {
        handles.push(std::thread::spawn(move || {
            let device = DistributedDevice::new("metrics-test", rank, world).unwrap();
            let mut metrics = vec![rank as f32];
            device
                .aggregate_metrics(&mut metrics, MetricReduce::Mean)
                .expect("metric aggregation should succeed");
            metrics
        }));
    }

    for handle in handles {
        let metrics = handle.join().unwrap();
        assert!((metrics[0] - 1.0).abs() < f32::EPSILON);
    }
}

#[test]
fn distributed_device_rejects_invalid_world_size() {
    let error =
        DistributedDevice::new("invalid-world", 0, 0).expect_err("zero-sized worlds should fail");
    assert!(matches!(
        error,
        TrainingDeviceError::Rendezvous(DistributedError::EmptyWorldSize(0))
    ));
}

#[test]
fn distributed_device_collective_timeout_is_configurable() {
    let device = DistributedDevice::new("configurable-timeout", 0, 2)
        .unwrap()
        .with_collective_timeout(Duration::from_millis(20));
    let mut gradients = [1.0, -1.0];
    let started = Instant::now();

    let error = device.synchronize_gradients(&mut gradients).unwrap_err();

    assert!(matches!(
        error,
        TrainingDeviceError::Rendezvous(DistributedError::CollectiveTimeout { .. })
    ));
    assert!(started.elapsed() < Duration::from_secs(1));
    assert_eq!(gradients, [1.0, -1.0]);
}

#[test]
fn distributed_device_detects_duplicate_rank() {
    let device = DistributedDevice::new("duplicate-rank", 0, 2).unwrap();
    let error = DistributedDevice::new("duplicate-rank", 0, 2)
        .expect_err("duplicate ranks should be rejected");
    assert!(matches!(
        error,
        TrainingDeviceError::Rendezvous(DistributedError::DuplicateRank { rank: 0 })
    ));
    drop(device);
}

#[test]
fn distributed_device_reports_buffer_mismatch() {
    let world = 2;
    let barrier = Arc::new(Barrier::new(world));
    let group = "buffer-mismatch".to_string();

    let mut handles = Vec::new();
    for (rank, len) in [(0usize, 2usize), (1usize, 3usize)] {
        let group_name = group.clone();
        let barrier = Arc::clone(&barrier);
        handles.push(std::thread::spawn(move || {
            let device = DistributedDevice::new(group_name, rank, world).unwrap();
            let mut gradients = vec![1.0f32; len];
            barrier.wait();
            device.synchronize_gradients(&mut gradients)
        }));
    }

    let mut mismatch_errors = 0;
    for handle in handles {
        let result = handle.join().unwrap();
        match result {
            Ok(_) => {}
            Err(TrainingDeviceError::Rendezvous(DistributedError::BufferLengthMismatch {
                ..
            })) => {
                mismatch_errors += 1;
            }
            Err(other) => panic!("unexpected synchronization error: {other:?}"),
        }
    }

    assert_eq!(
        mismatch_errors, 2,
        "both participants should observe the mismatch"
    );
}
