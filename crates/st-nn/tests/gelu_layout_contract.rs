use st_core::backend::device_caps::DeviceCaps;
use st_nn::execution::{push_backend_policy, BackendPolicy};
use st_nn::{Gelu, Module};
use st_tensor::{Layout, Tensor, TensorUtilBackend};

fn layouts(cols: usize) -> [Layout; 3] {
    [
        Layout::RowMajor,
        Layout::ColMajor,
        Layout::Chimera {
            stripes: 3,
            tile: (cols / 3) as u32,
        },
    ]
}

fn reference(x: f64) -> (f64, f64) {
    let c = (2.0 / std::f64::consts::PI).sqrt();
    let t = (c * (x + 0.044715 * x * x * x)).tanh();
    (
        0.5 * x * (1.0 + t),
        0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * c * (1.0 + 3.0 * 0.044715 * x * x),
    )
}

fn check(tensor: &Tensor, expected: &[f64]) {
    assert_eq!(tensor.layout(), Layout::RowMajor);
    assert_eq!(tensor.len(), expected.len());
    for (index, (&a, &b)) in tensor.data().iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && (f64::from(a) - b).abs() <= 2e-6 * (1.0 + b.abs()),
            "element {index}: {a} != {b}"
        );
    }
}

#[test]
fn forward_preserves_logical_values_for_every_layout() {
    for (rows, cols) in [(2, 6), (33, 195)] {
        let values: Vec<_> = (0..rows * cols)
            .map(|i| (i % 131) as f32 / 16.0 - 4.0)
            .collect();
        let expected: Vec<_> = values.iter().map(|&x| reference(f64::from(x)).0).collect();
        let input = Tensor::from_vec(rows, cols, values).unwrap();
        for layout in layouts(cols) {
            let oriented = input.to_layout(layout).unwrap();
            let before = oriented.clone();
            check(&Gelu::new().forward(&oriented).unwrap(), &expected);
            assert_eq!(oriented, before);
        }
    }
}

#[test]
fn backward_pairs_logical_input_and_seed_across_mixed_layouts() {
    let _policy = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
    for (rows, cols) in [(2, 6), (33, 195)] {
        let x: Vec<_> = (0..rows * cols)
            .map(|i| (i % 131) as f32 / 16.0 - 4.0)
            .collect();
        let g: Vec<_> = (0..rows * cols)
            .map(|i| (i % 29) as f32 / 16.0 - 0.5)
            .collect();
        let expected: Vec<_> = x
            .iter()
            .zip(&g)
            .map(|(&x, &g)| reference(f64::from(x)).1 * f64::from(g))
            .collect();
        let input = Tensor::from_vec(rows, cols, x).unwrap();
        let seed = Tensor::from_vec(rows, cols, g).unwrap();
        for input_layout in layouts(cols) {
            for seed_layout in layouts(cols) {
                let x = input.to_layout(input_layout).unwrap();
                let g = seed.to_layout(seed_layout).unwrap();
                let before = (x.clone(), g.clone());
                check(
                    &x.gelu_backward_with_backend(&g, TensorUtilBackend::Cpu)
                        .unwrap(),
                    &expected,
                );
                check(&Gelu::new().backward(&x, &g).unwrap(), &expected);
                assert_eq!((x, g), before);
            }
        }
    }
}
