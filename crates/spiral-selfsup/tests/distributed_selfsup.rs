use std::sync::atomic::{AtomicUsize, Ordering};

use spiral_selfsup::trainer::{DistributedDevice, MetricReduce, TrainingDevice};

static TEST_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn unique_group(prefix: &str) -> String {
    let id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    format!("{prefix}-{id}")
}

#[test]
#[ignore]
fn gradients_are_synchronized_via_all_reduce() {
    let world_size = 4;
    let group = unique_group("gradients");
    let mut handles = Vec::new();

    for rank in 0..world_size {
        let group_name = group.clone();
        handles.push(std::thread::spawn(move || {
            let device = DistributedDevice::new(group_name, rank, world_size);
            let mut gradients = vec![rank as f32 + 1.0, 0.25 * rank as f32];
            device.synchronize_gradients(&mut gradients);
            gradients
        }));
    }

    let expected = vec![
        (0..world_size).map(|r| r as f32 + 1.0).sum::<f32>() / world_size as f32,
        (0..world_size).map(|r| 0.25 * r as f32).sum::<f32>() / world_size as f32,
    ];

    for handle in handles {
        let gradients = handle.join().expect("worker thread panicked");
        assert!(gradients
            .iter()
            .zip(expected.iter())
            .all(|(lhs, rhs)| (lhs - rhs).abs() < 1e-6));
    }
}

#[test]
#[ignore]
fn metrics_are_aggregated_across_workers() {
    let world_size = 3;
    let group = unique_group("metrics");
    let mut handles = Vec::new();

    for rank in 0..world_size {
        let group_name = group.clone();
        handles.push(std::thread::spawn(move || {
            let device = DistributedDevice::new(group_name, rank, world_size);
            let mut metrics = vec![rank as f32, (rank as f32).powi(2)];
            device.aggregate_metrics(&mut metrics, MetricReduce::Mean);
            metrics
        }));
    }

    let expected = vec![
        (0..world_size).map(|r| r as f32).sum::<f32>() / world_size as f32,
        (0..world_size).map(|r| (r * r) as f32).sum::<f32>() / world_size as f32,
    ];

    for handle in handles {
        let metrics = handle.join().expect("worker thread panicked");
        assert!(metrics
            .iter()
            .zip(expected.iter())
            .all(|(lhs, rhs)| (lhs - rhs).abs() < 1e-6));
    }
}

#[test]
#[ignore]
fn sum_reduction_keeps_total_metric() {
    let world_size = 2;
    let group = unique_group("sum-metrics");
    let mut handles = Vec::new();

    for rank in 0..world_size {
        let group_name = group.clone();
        handles.push(std::thread::spawn(move || {
            let device = DistributedDevice::new(group_name, rank, world_size);
            let mut metrics = vec![rank as f32 + 2.0];
            device.aggregate_metrics(&mut metrics, MetricReduce::Sum);
            metrics
        }));
    }

    let expected_total = (0..world_size).map(|r| r as f32 + 2.0).sum::<f32>();

    for handle in handles {
        let metrics = handle.join().expect("worker thread panicked");
        assert!((metrics[0] - expected_total).abs() < 1e-6);
    }
}
