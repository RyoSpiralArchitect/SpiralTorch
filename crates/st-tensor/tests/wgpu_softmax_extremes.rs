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
