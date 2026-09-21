use st_core::backend::device_caps::DeviceCaps;
use st_nn::execution::{push_backend_policy, BackendPolicy};
use st_nn::{Gelu, Module, Parameter, Sequential};
use st_tensor::{Layout, Tensor, TensorError, TensorUtilBackend};
use std::cell::RefCell;
use std::rc::Rc;

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

struct RetainInput(Rc<RefCell<Vec<Tensor>>>);

impl Module for RetainInput {
    fn forward(&self, input: &Tensor) -> st_tensor::PureResult<Tensor> {
        self.0.borrow_mut().push(input.clone());
        Ok(input.clone())
    }

    fn backward(&mut self, _input: &Tensor, seed: &Tensor) -> st_tensor::PureResult<Tensor> {
        Ok(seed.clone())
    }

    fn visit_parameters(
        &self,
        _visitor: &mut dyn FnMut(&Parameter) -> st_tensor::PureResult<()>,
    ) -> st_tensor::PureResult<()> {
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        _visitor: &mut dyn FnMut(&mut Parameter) -> st_tensor::PureResult<()>,
    ) -> st_tensor::PureResult<()> {
        Ok(())
    }
}

#[test]
fn sequential_keeps_custom_module_retained_inputs_intact() {
    let _policy = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
    let saved = Rc::new(RefCell::new(Vec::new()));
    let mut model = Sequential::new();
    model.push(RetainInput(saved.clone()));
    model.push(Gelu::new());
    model.push(RetainInput(saved.clone()));
    model.push(Gelu::new());
    let input = Tensor::from_vec(2, 6, (0..12).map(|i| i as f32 / 4.0 - 1.5).collect()).unwrap();
    let first = input.try_gelu().unwrap();
    let second = first.try_gelu().unwrap();
    assert_eq!(model.forward(&input).unwrap(), second);
    assert_eq!(&*saved.borrow(), &[input.clone(), first.clone()]);
    model
        .backward(&input, &Tensor::from_vec(2, 6, vec![1.0; 12]).unwrap())
        .unwrap();
    assert_eq!(
        &*saved.borrow(),
        &[input.clone(), first.clone(), input, first]
    );
}

#[test]
fn nested_sequential_forward_and_backward_match_independent_chain() {
    let _policy = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
    let values: Vec<_> = (0..18).map(|i| i as f32 / 4.0 - 2.0).collect();
    let seeds: Vec<_> = (0..18).map(|i| (i as f32 - 9.0) / 8.0).collect();
    let input = Tensor::from_vec(3, 6, values.clone()).unwrap();
    let seed = Tensor::from_vec(3, 6, seeds.clone()).unwrap();
    let (mut expected, mut gradient) = (Vec::new(), Vec::new());
    for (&x, &g) in values.iter().zip(&seeds) {
        let (mut x, mut dx) = (f64::from(x), f64::from(g));
        for _ in 0..4 {
            let (y, derivative) = reference(x);
            x = y;
            dx *= derivative;
        }
        expected.push(x);
        gradient.push(dx);
    }
    for layout in layouts(6) {
        let input = input.to_layout(layout).unwrap();
        let before = input.clone();
        let mut inner = Sequential::new();
        inner.push(Gelu::new());
        inner.push(Gelu::new());
        let mut model = Sequential::new();
        model.push(Gelu::new());
        model.push(inner);
        model.push(Gelu::new());
        check(&model.forward(&input).unwrap(), &expected);
        check(&model.backward(&input, &seed).unwrap(), &gradient);
        check(&model.forward(&input).unwrap(), &expected);
        assert_eq!(input, before);
    }
}

fn owned_fixture() -> Tensor {
    Tensor::from_vec(
        2,
        6,
        vec![
            -0.0, 0.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0, -3.0, 0.25, -0.25,
        ],
    )
    .unwrap()
}

fn bits(tensor: &Tensor) -> Vec<u32> {
    tensor.data().iter().map(|x| x.to_bits()).collect()
}

#[test]
fn owned_gelu_reuses_unique_storage_and_revokes_weak_stamps() {
    for via_module in [false, true] {
        let input = owned_fixture();
        let expected = input.try_gelu().unwrap();
        let pointer = input.data().as_ptr();
        let stamp = input.content_stamp().unwrap();
        let output = if via_module {
            Gelu::new().forward_owned(input)
        } else {
            input.try_into_gelu()
        }
        .unwrap();
        assert_eq!(output.data().as_ptr(), pointer);
        assert_eq!(bits(&output), bits(&expected));
        assert!(!stamp.matches(&output));
        assert!(output.content_stamp().unwrap().matches(&output));
    }
}

#[test]
fn owned_gelu_materializes_shared_snapshot_external_and_nonrow_inputs() {
    use st_tensor::dlpack::{DlpackCopyPolicy, DlpackExportOptions, DlpackProtocol};
    let options = DlpackExportOptions {
        protocol: DlpackProtocol::Versioned,
        copy: DlpackCopyPolicy::Never,
    };
    for kind in 0..8 {
        let original = owned_fixture();
        let expected = original.try_gelu().unwrap();
        let mut alias = None;
        let input = match kind {
            0 => {
                alias = Some(original.clone());
                original
            }
            1 => original.into_snapshot(),
            2 => {
                alias = Some(
                    Tensor::from_managed_dlpack(original.export_dlpack(options).unwrap()).unwrap(),
                );
                original
            }
            3 => {
                let foreign =
                    Tensor::from_managed_dlpack(original.export_dlpack(options).unwrap()).unwrap();
                alias = Some(original);
                foreign
            }
            4 => original.to_layout(Layout::ColMajor).unwrap(),
            5 => original
                .to_layout(Layout::Chimera {
                    stripes: 3,
                    tile: 2,
                })
                .unwrap(),
            6 => {
                drop(original.export_dlpack(options).unwrap());
                assert!(original.content_stamp().is_none());
                original
            }
            _ => {
                let snapshot = original.into_snapshot();
                let foreign =
                    Tensor::from_managed_dlpack(snapshot.export_dlpack(options).unwrap()).unwrap();
                alias = Some(snapshot);
                foreign
            }
        };
        let before = alias.as_ref().map(bits);
        let pointer = input.data().as_ptr();
        let output = input.try_into_gelu().unwrap();
        assert_ne!(output.data().as_ptr(), pointer, "kind={kind}");
        assert_eq!(bits(&output), bits(&expected), "kind={kind}");
        assert_eq!(output.layout(), Layout::RowMajor);
        assert!(!output.is_snapshot());
        assert!(output.content_stamp().is_some());
        assert_eq!(alias.as_ref().map(bits), before);
    }
}

#[test]
fn owned_gelu_preserves_outlier_errors_and_aliases_on_failure() {
    for (values, label) in [
        (vec![0.5, f32::MAX, f32::NAN], "gelu_input"),
        (vec![0.5, f32::MAX], "gelu_square"),
        (vec![0.5, 1e14], "gelu_cubic"),
    ] {
        for shared in [false, true] {
            let input = Tensor::from_vec(1, values.len(), values.clone()).unwrap();
            let alias = shared.then(|| input.clone());
            assert!(
                matches!(input.try_into_gelu(), Err(TensorError::NonFiniteValue { label: actual, .. }) if actual == label)
            );
            if let Some(alias) = alias {
                assert_eq!(
                    bits(&alias),
                    values.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                );
            }
        }
    }
    let input = Tensor::from_vec(1, 4, vec![-0.0, 0.0, 2e12, -5e12]).unwrap();
    let expected = input.try_gelu().unwrap();
    let pointer = input.data().as_ptr();
    let output = input.try_into_gelu().unwrap();
    assert_eq!(output.data().as_ptr(), pointer);
    assert_eq!(bits(&output), bits(&expected));
    for (rows, cols) in [(0, 6), (3, 0)] {
        let output = Tensor::zeros(rows, cols).unwrap().try_into_gelu().unwrap();
        assert_eq!(output.shape(), (rows, cols));
        assert!(output.is_empty());
    }
}

#[test]
fn owned_sequential_and_default_custom_module_preserve_contracts() {
    let saved = Rc::new(RefCell::new(Vec::new()));
    let input = owned_fixture();
    let before = input.clone();
    let intermediate = RetainInput(saved.clone()).forward_owned(input).unwrap();
    let expected = intermediate.try_gelu().unwrap();
    let mut model = Sequential::new();
    model.push(Sequential::new());
    model.push(Gelu::new());
    let output = model.forward_owned(intermediate).unwrap();
    assert_eq!(bits(&output), bits(&expected));
    assert_eq!(&*saved.borrow(), &[before]);

    let mut inner = Sequential::new();
    inner.push(Gelu::new());
    let mut model = Sequential::new();
    model.push(inner);
    model.push(Gelu::new());
    let input = owned_fixture();
    let expected = input.try_gelu().unwrap().try_gelu().unwrap();
    let pointer = input.data().as_ptr();
    let output = model.forward_owned(input).unwrap();
    assert_eq!(output.data().as_ptr(), pointer);
    assert_eq!(bits(&output), bits(&expected));
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
