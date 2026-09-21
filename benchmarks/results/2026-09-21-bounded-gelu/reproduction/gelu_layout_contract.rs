use st_core::backend::device_caps::DeviceCaps;
use st_nn::execution::{push_backend_policy, BackendPolicy};
use st_nn::{Gelu, Module};
use st_tensor::{Layout, Tensor, TensorError, TensorUtilBackend};

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
            check(&oriented.try_gelu().unwrap(), &expected);
            assert_eq!(oriented, before);
        }
    }
}

#[test]
fn checked_forward_preserves_error_precedence_and_signed_zero() {
    for (values, expected_label) in [
        (vec![f32::MAX, f32::NAN], "gelu_input"),
        (vec![f32::MAX], "gelu_square"),
        (vec![1e14], "gelu_cubic"),
        (vec![f32::NEG_INFINITY], "gelu_input"),
    ] {
        let input = Tensor::from_vec(1, values.len(), values).unwrap();
        for result in [input.try_gelu(), Gelu::new().forward(&input)] {
            assert!(
                matches!(result, Err(TensorError::NonFiniteValue { label, .. }) if label == expected_label)
            );
        }
    }
    for (rows, cols) in [(0, 6), (3, 0)] {
        let input = Tensor::zeros(rows, cols)
            .unwrap()
            .to_layout(Layout::ColMajor)
            .unwrap();
        let output = input.try_gelu().unwrap();
        assert_eq!(output.shape(), (rows, cols));
        assert!(output.is_empty());
    }
    let input = Tensor::from_vec(1, 4, vec![-0.0, 0.0, -100.0, 100.0]).unwrap();
    let result = input.try_gelu().unwrap();
    assert_eq!(
        result
            .data()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        [-0.0f32, 0.0, -0.0, 100.0].map(f32::to_bits)
    );
}

// Snapshot the checked scalar evaluation order, independent of the implementation
// under test. Wide exponent coverage catches changes hidden by an f64 tolerance.
fn checked_scalar_bits(value: f32) -> u32 {
    const C: f32 = std::f32::consts::FRAC_2_SQRT_PI * std::f32::consts::FRAC_1_SQRT_2;
    let square = value * value;
    let cubic = square * value;
    let arg = value + 0.044715 * cubic;
    let inner = C * arg;
    let tanh = inner.tanh();
    let output = 0.5 * value * (1.0 + tanh);
    assert!([square, cubic, arg, inner, tanh, output]
        .iter()
        .all(|x| x.is_finite()));
    output.to_bits()
}

#[test]
fn checked_forward_matches_scalar_bits_across_exponents_and_bounds() {
    let bound = 1e12f32;
    let mut values = vec![
        -0.0,
        0.0,
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
    ];
    for bits in [bound.to_bits() - 1, bound.to_bits()] {
        values.extend([f32::from_bits(bits), -f32::from_bits(bits)]);
    }
    let mut state = 0x6a09_e667u32;
    while values.len() < 12_288 {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let value = f32::from_bits(state);
        if value.is_finite() && value.abs() <= bound {
            values.push(value);
        }
    }
    // Test the entirely bounded batch separately: an outlier must not hide a
    // broken fast lane by sending every element through the fallback.
    for outliers in [false, true] {
        let mut values = values.clone();
        if outliers {
            values.extend([
                f32::from_bits(bound.to_bits() + 1),
                -f32::from_bits(bound.to_bits() + 1),
                2e12,
                -2e12,
                5e12,
                -5e12,
            ]);
        }
        let expected: Vec<_> = values.iter().map(|&x| checked_scalar_bits(x)).collect();
        let input = Tensor::from_vec(values.len() / 6, 6, values).unwrap();
        for layout in layouts(6) {
            let input = input.to_layout(layout).unwrap();
            for output in [
                input.try_gelu().unwrap(),
                Gelu::new().forward(&input).unwrap(),
            ] {
                assert_eq!(output.layout(), Layout::RowMajor);
                assert_eq!(output.shape(), input.shape());
                let bits: Vec<_> = output.data().iter().map(|x| x.to_bits()).collect();
                assert_eq!(bits, expected, "layout={layout:?}, outliers={outliers}");
            }
        }
    }
}

#[test]
fn checked_forward_scans_past_large_finite_inputs_before_arithmetic() {
    for nonfinite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let input = Tensor::from_vec(2, 3, vec![1.0, f32::MAX, 2.0, 1e14, 0.0, nonfinite]).unwrap();
        for layout in layouts(3) {
            let input = input.to_layout(layout).unwrap();
            for result in [input.try_gelu(), Gelu::new().forward(&input)] {
                assert!(matches!(
                    result,
                    Err(TensorError::NonFiniteValue {
                        label: "gelu_input",
                        ..
                    })
                ));
            }
        }
    }
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
#[test]
fn strict_wgpu_backward_pairs_logical_layouts() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let x = Tensor::from_vec(
        2,
        6,
        vec![
            -1.0, 0.0, 1.0, -2.0, 2.0, 0.5, 0.25, -0.5, 3.0, -3.0, 0.75, -0.75,
        ],
    )
    .unwrap();
    let g = Tensor::from_vec(2, 6, (0..12).map(|i| (i as f32 - 5.0) / 8.0).collect()).unwrap();
    let expected: Vec<_> = x
        .data()
        .iter()
        .zip(g.data())
        .map(|(&x, &g)| reference(f64::from(x)).1 * f64::from(g))
        .collect();
    for input_layout in layouts(6) {
        for seed_layout in layouts(6) {
            let x = x.to_layout(input_layout).unwrap();
            let g = g.to_layout(seed_layout).unwrap();
            check(
                &x.gelu_backward_with_backend(&g, TensorUtilBackend::GpuWgpu)
                    .unwrap(),
                &expected,
            );
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
