#![cfg(feature = "wgpu_dense")]

use st_tensor::{SoftmaxBackend, Tensor};

#[test]
fn finite_logits_below_old_sentinel_normalize_on_strict_wgpu() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    assert_eq!(std::env::var("SPIRALTORCH_STRICT_GPU").as_deref(), Ok("1"));
    for cols in [1, 3, 31, 256, 257, 1025] {
        let rows = 3;
        let data: Vec<_> = (0..rows * cols)
            .map(|i| match i / cols {
                0 => -f32::MAX,
                1 => -2e30,
                _ => {
                    if i % cols == cols - 1 {
                        -2e30
                    } else {
                        -3e30
                    }
                }
            })
            .collect();
        let input = Tensor::from_vec(rows, cols, data).unwrap();
        let actual = input
            .row_softmax_with_backend(SoftmaxBackend::GpuWgpu)
            .unwrap();
        let expected = input.row_softmax_with_backend(SoftmaxBackend::Cpu).unwrap();
        for (i, (&got, &want)) in actual.data().iter().zip(expected.data()).enumerate() {
            assert!(
                got.is_finite() && (got - want).abs() <= 2e-7 + 2e-6 * want.abs(),
                "cols={cols} index={i}: {got} != {want}"
            );
        }
        for row in actual.data().chunks(cols) {
            assert!((row.iter().map(|&x| f64::from(x)).sum::<f64>() - 1.).abs() <= 2e-6);
        }
    }
}

#[test]
fn batched_softmax_peak_pair_reaches_tensor_spiral_without_changing_cpu_blend() {
    use std::sync::{Arc, Mutex};
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    assert_eq!(std::env::var("SPIRALTORCH_STRICT_GPU").as_deref(), Ok("1"));
    let flags = Arc::new(Mutex::new(Vec::new()));
    let captured = flags.clone();
    let previous = st_tensor::set_thread_meta_observer(Some(Arc::new(
        move |event: &st_tensor::TensorOpMetaEvent| {
            if event.op_name == "row_softmax_hardmax_spiral" {
                if let Some(flag) = event.data.get("spiral_consensus_gpu") {
                    captured.lock().unwrap().push(flag.clone());
                }
            }
        },
    )));
    for cols in [1, 3, 31, 256, 257, 1025] {
        let rows = 3;
        let data: Vec<_> = (0..rows * cols)
            .map(|i| match i / cols {
                0 => -f32::MAX,
                1 => (i % 7) as f32 / 8.,
                _ => {
                    if i % cols == cols - 1 {
                        100.
                    } else {
                        -100.
                    }
                }
            })
            .collect();
        let input = Tensor::from_vec(rows, cols, data).unwrap();
        let (soft, mask) = input
            .row_softmax_hardmax_with_backend(SoftmaxBackend::GpuWgpu)
            .unwrap();
        let reference = input.row_softmax_with_backend(SoftmaxBackend::Cpu).unwrap();
        let triple = input
            .row_softmax_hardmax_spiral_with_backend(SoftmaxBackend::GpuWgpu)
            .unwrap();
        assert_eq!(mask.data(), triple.hardmax.data());
        assert_eq!(soft.data(), triple.softmax.data());
        assert_eq!(triple.spiral.shape(), (rows, cols));
        assert!(triple
            .spiral
            .data()
            .iter()
            .all(|v| v.is_finite() && *v >= 0.));
        for row in 0..rows {
            let offset = row * cols;
            let max = input.data()[offset..offset + cols]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            for col in 0..cols {
                let at = offset + col;
                let want = reference.data()[at];
                assert!((soft.data()[at] - want).abs() <= 2e-7 + 2e-6 * want.abs());
                assert_eq!(
                    mask.data()[at],
                    if input.data()[at] == max { 1. } else { 0. }
                );
            }
        }
        let (raw_soft, raw_mask, raw_spiral, raw_metrics) =
            st_tensor::backend::wgpu_dense::row_softmax_hardmax_spiral(
                input.data(),
                rows,
                cols,
                st_tensor::Layout::RowMajor,
            )
            .unwrap();
        assert_eq!(raw_mask, mask.data());
        for (got, want) in raw_soft
            .iter()
            .zip(soft.data())
            .chain(raw_spiral.iter().zip(triple.spiral.data()))
        {
            assert!(got.is_finite() && (got - want).abs() <= 2e-6 + 5e-6 * want.abs());
        }
        // These logits have no tiny positive tail: CPU/GPU entropy floors agree.
        // The ordinary Tensor route must retain its additional telemetry blend.
        assert!(
            (triple.metrics.spiral_coherence - (raw_metrics.spiral_coherence + 1.) * 0.5).abs()
                < 2e-6
        );
        assert!(
            (triple.metrics.average_enrichment - raw_metrics.average_enrichment * 1.25).abs()
                < 2e-6
        );
    }
    st_tensor::set_thread_meta_observer(previous);
    let flags = flags.lock().unwrap();
    assert_eq!(flags.len(), 6);
    assert!(flags.iter().all(|flag| flag == true));
}
