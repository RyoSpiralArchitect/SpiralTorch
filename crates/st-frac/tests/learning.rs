use st_frac::learning::{FractionalGlKernel, FractionalLearningError};
use st_frac::{fracdiff_gl_nd_config, FracdiffGlConfig, Pad};

fn dot(x: &[f32], y: &[f32]) -> f64 {
    x.iter()
        .zip(y)
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum()
}

#[test]
fn saved_forward_is_the_existing_nd_operator_and_joint_adjoint() {
    let kernel = FractionalGlKernel::new(5, 0.7, 100, 500).unwrap();
    let x: Vec<f32> = (0..24).map(|i| (i as f32 * 0.3).sin()).collect();
    let upstream: Vec<f32> = (0..24).map(|i| (i as f32 * 0.2).cos()).collect();
    let dx: Vec<f32> = x.iter().map(|v| v * 0.2).collect();
    for alpha in [0.45, 1.0, 1.5] {
        let saved = kernel.forward(&x, &[2, 4, 3], 1, alpha).unwrap();
        let reference = fracdiff_gl_nd_config(
            &ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 4, 3]), x.clone()).unwrap(),
            FracdiffGlConfig::new(alpha, 1, 5, Pad::Zero).with_step(0.7),
        )
        .unwrap();
        assert_eq!(saved.output(), &reference);
        let dy = saved.jvp(&dx, 0.3).unwrap();
        let pullback = saved.vjp(&upstream).unwrap();
        let left = dot(&dy, &upstream);
        let right = dot(&dx, &pullback.input) + 0.3 * f64::from(pullback.alpha);
        assert!((left - right).abs() < 2e-6 * (1.0 + left.abs()));
        let epsilon = 1e-3;
        let plus = kernel.forward(&x, &[2, 4, 3], 1, alpha + epsilon).unwrap();
        let minus = kernel.forward(&x, &[2, 4, 3], 1, alpha - epsilon).unwrap();
        let difference: Vec<f32> = plus
            .output()
            .iter()
            .zip(minus.output())
            .map(|(a, b)| (a - b) / (2. * epsilon))
            .collect();
        assert!((dot(&difference, &upstream) - f64::from(pullback.alpha)).abs() < 3e-3);
    }
}

#[test]
fn causal_prefix_and_batch_independence_hold_in_forward_and_backward() {
    let kernel = FractionalGlKernel::new(8, 1., 64, 512).unwrap();
    let mut x: Vec<f32> = (0..24).map(|v| v as f32 / 10.).collect();
    let original = kernel.forward(&x, &[2, 4, 3], 1, 0.6).unwrap();
    x[6..].fill(90.);
    let changed = kernel.forward(&x, &[2, 4, 3], 1, 0.6).unwrap();
    assert_eq!(
        &original.output().as_slice().unwrap()[..6],
        &changed.output().as_slice().unwrap()[..6]
    );
    let mut upstream = vec![0.; 24];
    upstream[3..6].fill(1.);
    let gradient = original.vjp(&upstream).unwrap();
    assert!(gradient.input[6..].iter().all(|v| *v == 0.));
    // Snapshots own their forward result even after caller data is overwritten.
    assert_eq!(original.vjp(&upstream).unwrap().alpha, gradient.alpha);
}

#[test]
fn validates_shapes_budgets_and_nonfinite_directions() {
    let kernel = FractionalGlKernel::new(4, 1., 10, 20).unwrap();
    assert!(FractionalGlKernel::new(0, 1., 10, 20).is_err());
    assert!(FractionalGlKernel::new(4, 0., 10, 20).is_err());
    assert!(FractionalGlKernel::new(4, 1., 0, 20).is_err());
    for shape in [vec![], vec![0], vec![6], vec![usize::MAX, 2], vec![1; 17]] {
        assert!(matches!(
            kernel.forward(&[1.; 4], &shape, 0, 0.5),
            Err(FractionalLearningError::Shape)
        ));
    }
    assert!(matches!(
        kernel.forward(&[1.; 6], &[6], 0, 0.5),
        Err(FractionalLearningError::Budget)
    ));
    for alpha in [0., -1., f32::NAN, f32::INFINITY] {
        assert!(kernel.forward(&[1.; 4], &[4], 0, alpha).is_err());
    }
    assert!(kernel.forward(&[f32::NAN; 4], &[4], 0, 0.5).is_err());
    let batch = kernel.forward(&[1.; 4], &[4], 0, 0.5).unwrap();
    assert!(batch.vjp(&[1.; 3]).is_err());
    assert!(batch.vjp(&[f32::NAN; 4]).is_err());
    assert!(batch.jvp(&[1.; 4], f32::INFINITY).is_err());
}
