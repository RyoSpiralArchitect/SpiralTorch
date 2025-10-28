use super::{DistributedDevice, MetricReduce, TrainingDevice};

#[test]
fn distributed_device_all_reduce_averages_gradients() {
    let world = 2;
    let mut handles = Vec::new();
    for rank in 0..world {
        handles.push(std::thread::spawn(move || {
            let device = DistributedDevice::new("unit-test", rank, world);
            let mut gradients = vec![rank as f32 + 1.0, 2.0];
            device.synchronize_gradients(&mut gradients);
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
fn distributed_device_aggregates_metrics() {
    let world = 3;
    let mut handles = Vec::new();
    for rank in 0..world {
        handles.push(std::thread::spawn(move || {
            let device = DistributedDevice::new("metrics-test", rank, world);
            let mut metrics = vec![rank as f32];
            device.aggregate_metrics(&mut metrics, MetricReduce::Mean);
            metrics
        }));
    }

    for handle in handles {
        let metrics = handle.join().unwrap();
        assert!((metrics[0] - 1.0).abs() < f32::EPSILON);
    }
}
