#![cfg(feature = "wgpu_dense")]

use st_tensor::{MatmulBackend, Tensor};

#[test]
fn default_f32_matmul_does_not_quantize_large_rhs() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    assert_ne!(
        std::env::var("SPIRALTORCH_WGPU_ALLOW_INT8").as_deref(),
        Ok("1")
    );
    for (m, k, n) in [(3, 63, 65), (5, 64, 64), (7, 65, 67), (17, 256, 64)] {
        let values = |count, multiplier| {
            (0..count)
                .map(|i| ((i * multiplier % 1009) as f32 - 504.0) / 1024.0)
                .collect()
        };
        let lhs = Tensor::from_vec(m, k, values(m * k, 37)).unwrap();
        let rhs = Tensor::from_vec(k, n, values(k * n, 59)).unwrap();
        let actual = lhs
            .matmul_with_backend(&rhs, MatmulBackend::GpuWgpu)
            .unwrap();
        for row in 0..m {
            for col in 0..n {
                let expected: f64 = (0..k)
                    .map(|i| lhs.data()[row * k + i] as f64 * rhs.data()[i * n + col] as f64)
                    .sum();
                let got = actual.data()[row * n + col] as f64;
                assert!(
                    (got - expected).abs() <= 1e-5 + 1e-4 * expected.abs(),
                    "{m}x{k}x{n} [{row},{col}]: {got} != {expected}"
                );
            }
        }
    }
}
