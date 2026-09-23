#![cfg(feature = "wgpu_dense")]
use st_tensor::{backend::wgpu_dense, Layout, Tensor, TensorUtilBackend};

fn enabled() -> bool {
    std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() == Ok("1")
}

fn derivative(x: f64) -> f64 {
    if x.abs() >= 10. {
        return if x > 0. { 1. } else { 0. };
    }
    let c = (2. / std::f64::consts::PI).sqrt();
    let t = (c * (x + 0.044715 * x * x * x)).tanh();
    0.5 * (1. + t) + 0.5 * x * (1. - t * t) * c * (1. + 3. * 0.044715 * x * x)
}

#[test]
fn plain_and_fused_gelu_preserve_tails_saturation_and_residual_contract() {
    if !enabled() {
        return;
    }
    assert_eq!(std::env::var("SPIRALTORCH_STRICT_GPU").as_deref(), Ok("1"));
    let values = [
        -f32::MAX,
        -1e20,
        -10.,
        -9.999,
        -5.,
        -1.,
        -0.,
        0.,
        0.8,
        1.,
        5.,
        9.999,
        10.,
        1e20,
        f32::MAX,
    ];
    for (rows, cols) in [(1, 1), (3, 5), (17, 31), (33, 65), (65, 257), (2, 1025)] {
        let z: Vec<_> = (0..rows * cols).map(|i| values[i % values.len()]).collect();
        let seed: Vec<_> = (0..rows * cols)
            .map(|i| (i % 31) as f32 / 16. - 1.)
            .collect();
        let residual: Vec<_> = (0..rows * cols).map(|i| (i % 7) as f32 / 8.).collect();
        let reference: Vec<_> = z
            .iter()
            .zip(&seed)
            .map(|(&x, &g)| derivative(f64::from(x)) * f64::from(g))
            .collect();
        let input = Tensor::from_vec(rows, cols, z.clone()).unwrap();
        let gradient = Tensor::from_vec(rows, cols, seed.clone()).unwrap();
        let plain = input
            .gelu_backward_with_backend(&gradient, TensorUtilBackend::GpuWgpu)
            .unwrap();
        for residual in [None, Some(residual.as_slice())] {
            let (gz, dr, db) =
                wgpu_dense::fused_gelu_backward(&z, &seed, residual, rows, cols).unwrap();
            for (i, &want) in reference.iter().enumerate() {
                for (route, actual) in [("fused", gz[i]), ("plain", plain.data()[i])] {
                    assert!(
                        actual.is_finite()
                            && (f64::from(actual) - want).abs() <= 2e-6 + 1e-5 * want.abs(),
                        "{route} {rows}x{cols} index={i} residual={} z={} seed={} actual={actual} expected={want}",
                        residual.is_some(), z[i], seed[i]
                    );
                }
                let want = want + f64::from(residual.map_or(0., |r| r[i]));
                assert!(
                    dr[i].is_finite()
                        && (f64::from(dr[i]) - want).abs() <= 2e-6 + 1e-5 * want.abs(),
                    "residual {rows}x{cols} index={i} accumulated={} z={} seed={} base={} gz={} actual={} expected={want}",
                    residual.is_some(), z[i], seed[i], residual.map_or(0., |r| r[i]), gz[i], dr[i]
                );
            }
            for c in 0..cols {
                let want: f64 = (0..rows).map(|r| reference[r * cols + c]).sum();
                assert!(
                    db[c].is_finite()
                        && (f64::from(db[c]) - want).abs()
                            <= 2e-6 * rows as f64 + 1e-5 * want.abs(),
                    "bias {rows}x{cols} col={c} residual={} actual={} expected={want}",
                    residual.is_some(),
                    db[c]
                );
            }
        }
    }
}

#[test]
fn plain_gelu_keeps_logical_layout_and_finite_policy() {
    if !enabled() {
        return;
    }
    let rows = 33;
    let cols = 195;
    let z = Tensor::from_vec(
        rows,
        cols,
        (0..rows * cols)
            .map(|i| (i % 257) as f32 / 32. - 4.)
            .collect(),
    )
    .unwrap();
    let g = Tensor::from_vec(
        rows,
        cols,
        (0..rows * cols)
            .map(|i| (i % 29) as f32 / 16. - 1.)
            .collect(),
    )
    .unwrap();
    let expected = z
        .gelu_backward_with_backend(&g, TensorUtilBackend::Cpu)
        .unwrap();
    for a in [
        Layout::RowMajor,
        Layout::ColMajor,
        Layout::Chimera {
            stripes: 3,
            tile: 65,
        },
    ] {
        for b in [Layout::RowMajor, Layout::ColMajor] {
            let actual = z
                .to_layout(a)
                .unwrap()
                .gelu_backward_with_backend(&g.to_layout(b).unwrap(), TensorUtilBackend::GpuWgpu)
                .unwrap();
            assert_eq!(actual.layout(), Layout::RowMajor);
            for (&a, &b) in actual.data().iter().zip(expected.data()) {
                assert!((a - b).abs() <= 2e-6 + 1e-5 * b.abs());
            }
        }
    }
    let tensor = |v| Tensor::from_vec(1, 1, vec![v]).unwrap();
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(tensor(invalid)
            .gelu_backward_with_backend(&tensor(1.), TensorUtilBackend::GpuWgpu)
            .is_err());
        assert!(tensor(1.)
            .gelu_backward_with_backend(&tensor(invalid), TensorUtilBackend::GpuWgpu)
            .is_err());
    }
    assert!(tensor(1.)
        .gelu_backward_with_backend(&tensor(f32::MAX), TensorUtilBackend::GpuWgpu)
        .is_err());
    let empty = Tensor::zeros(0, cols).unwrap();
    assert!(empty
        .gelu_backward_with_backend(&empty, TensorUtilBackend::GpuWgpu)
        .unwrap()
        .is_empty());
}

#[test]
fn gelu_shape_overflow_rejects_before_requesting_a_device() {
    for (r, c) in [(0, 4), (4, 0), (usize::MAX, 2)] {
        assert!(wgpu_dense::gelu_backward(&[], &[], r, c).is_err());
        assert!(wgpu_dense::fused_gelu_backward(&[], &[], None, r, c).is_err());
    }
}
