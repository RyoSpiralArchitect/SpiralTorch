#![cfg(feature = "nerf")]

use st_core::backend::device_caps::DeviceCaps;
use st_nn::execution::{push_backend_policy, BackendPolicy};
use st_nn::Module;
use st_tensor::{Layout, Tensor, TensorError};
use st_vision::nerf::{NerfField, NerfFieldConfig, PositionalEncoding};

fn layouts(cols: usize) -> [Layout; 3] {
    [Layout::RowMajor, Layout::ColMajor,
     Layout::Chimera { stripes: 2, tile: (cols / 2) as u32 }]
}

fn fixture_field() -> NerfField {
    let mut field = NerfField::new(NerfFieldConfig {
        position_frequencies: 1, direction_frequencies: 1,
        hidden_layers: 1, hidden_width: 4, feature_dim: 3,
        color_layers: 1, color_hidden_width: 4, ..NerfFieldConfig::default()
    }).unwrap();
    field.visit_parameters_mut(&mut |p| {
        for (i, v) in p.value_mut().data_mut().iter_mut().enumerate() {
            *v = 0.01 + (i % 11) as f32 / 100.0;
        }
        Ok(())
    }).unwrap();
    field
}

fn parameters(field: &NerfField, gradients: bool) -> Vec<Vec<f32>> {
    let mut values = Vec::new();
    field.visit_parameters(&mut |p| {
        values.push(if gradients { p.gradient().unwrap().data().to_vec() }
                    else { p.value().data().to_vec() });
        Ok(())
    }).unwrap();
    values
}

#[test]
fn encoding_matches_logical_coordinates_and_f64_reference() {
    let input = Tensor::from_vec(3, 6, (0..18).map(|i| (i as f32 - 8.0) / 16.0).collect()).unwrap();
    for bands in [0, 1, 8] {
        for residual in [false, true] {
            let mut encoder = PositionalEncoding::new(6, bands).unwrap();
            if !residual { encoder = encoder.without_input(); }
            let mut expected = Vec::new();
            for row in input.data().chunks_exact(6) {
                if residual { expected.extend(row.iter().copied().map(f64::from)); }
                for band in 0..bands {
                    for &x in row {
                        let phase = f64::from(x) * 2f64.powi(band as i32);
                        expected.extend([phase.sin(), phase.cos()]);
                    }
                }
            }
            let reference = encoder.encode(&input).unwrap();
            for layout in layouts(6) {
                let input = input.to_layout(layout).unwrap();
                let before: Vec<_> = input.data().iter().map(|v|v.to_bits()).collect();
                let output = encoder.encode(&input).unwrap();
                assert_eq!(output.layout(), Layout::RowMajor);
                assert_eq!(output.shape(), (3, encoder.output_dims()));
                assert_eq!(output.data(), reference.data());
                assert_eq!(output.len(), expected.len());
                for (&a, &b) in output.data().iter().zip(&expected) {
                    assert!(a.is_finite() && (f64::from(a) - b).abs() <= 2e-6);
                }
                assert_eq!(input.data().iter().map(|v|v.to_bits()).collect::<Vec<_>>(), before);
            }
        }
    }
}

#[test]
fn encoding_rejects_nonfinite_frequencies_phases_and_dimensions() {
    assert!(PositionalEncoding::new(0, 4).is_err());
    assert!(PositionalEncoding::new(3, 129).is_err());
    assert!(PositionalEncoding::new(3, usize::MAX).is_err());
    assert!(PositionalEncoding::new(usize::MAX, 1).is_err());
    let last_finite_band = PositionalEncoding::new(1, 128).unwrap();
    assert!(last_finite_band.encode(&Tensor::from_vec(1, 1, vec![1.0]).unwrap()).unwrap().data().iter().all(|x|x.is_finite()));
    let encoder = PositionalEncoding::new(1, 2).unwrap();
    for value in [f32::MAX, -f32::MAX] {
        let input = Tensor::from_vec(1, 1, vec![value]).unwrap();
        assert!(matches!(encoder.encode(&input), Err(TensorError::NonFiniteValue {label:"nerf_phase",..})));
        assert_eq!(input.data(), &[value]);
    }
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let input = Tensor::from_vec(2, 1, vec![f32::MAX, value]).unwrap();
        assert!(matches!(encoder.encode(&input), Err(TensorError::NonFiniteValue {label:"nerf_input",..})));
    }
    let no_features = PositionalEncoding::new(1, 0).unwrap().without_input();
    assert!(no_features.encode(&Tensor::from_vec(1, 1, vec![f32::NAN]).unwrap()).is_err());
}

#[test]
fn empty_encodings_and_signed_zero_remain_well_defined() {
    let encoder = PositionalEncoding::new(3, 4).unwrap();
    assert_eq!(encoder.encode(&Tensor::zeros(0, 3).unwrap()).unwrap().shape(), (0, 27));
    let encoder = PositionalEncoding::new(2, 0).unwrap().without_input();
    assert_eq!(encoder.encode(&Tensor::zeros(3, 2).unwrap()).unwrap().shape(), (3, 0));
    let encoder = PositionalEncoding::new(2, 1).unwrap();
    let output = encoder.encode(&Tensor::from_vec(1, 2, vec![-0.0, 0.0]).unwrap()).unwrap();
    assert_eq!(output.data().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),
               [-0.0f32, 0.0, -0.0, 1.0, 0.0, 1.0].iter().map(|v|v.to_bits()).collect::<Vec<_>>());
}

#[test]
fn assembly_preserves_mixed_input_layouts_and_missing_directions() {
    let field = NerfField::new(NerfFieldConfig {
        position_dims: 6, direction_dims: 6, hidden_layers: 0, color_layers: 0,
        position_frequencies: 0, direction_frequencies: 0, feature_dim: 2,
        ..NerfFieldConfig::default()
    }).unwrap();
    let positions = Tensor::from_vec(2, 6, (0..12).map(|v|v as f32).collect()).unwrap();
    let directions = Tensor::from_vec(2, 6, (0..12).map(|v|100.0+v as f32).collect()).unwrap();
    let expected = field.assemble_input(&positions, Some(&directions)).unwrap();
    for pos_layout in layouts(6) { for dir_layout in layouts(6) {
        let p = positions.to_layout(pos_layout).unwrap();
        let d = directions.to_layout(dir_layout).unwrap();
        assert_eq!(field.assemble_input(&p, Some(&d)).unwrap().data(), expected.data());
        let missing = field.assemble_input(&p, None).unwrap();
        for (row, p) in missing.data().chunks_exact(12).zip(positions.data().chunks_exact(6)) {
            assert_eq!(&row[..6], p); assert_eq!(&row[6..], &[0.0;6]);
        }
    }}
}

#[test]
fn field_gradients_and_optimizer_steps_preserve_input_and_seed_layouts() {
    let _policy = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
    let input = Tensor::from_vec(2, 6, (0..12).map(|i| 0.05 + i as f32 / 100.0).collect()).unwrap();
    let seed = Tensor::from_vec(2, 4, vec![0.5, -0.25, 0.75, 0.125, -0.5, 0.25, -0.125, 1.0]).unwrap();
    let mut reference = fixture_field();
    let expected = reference.forward(&input).unwrap();
    reference.backward(&input, &seed).unwrap();
    let gradients = parameters(&reference, true);
    reference.apply_step(1e-3).unwrap();
    let updated = parameters(&reference, false);
    for layout in layouts(6) { for seed_layout in layouts(4) {
        let x = input.to_layout(layout).unwrap(); let g = seed.to_layout(seed_layout).unwrap();
        let before = (x.clone(), g.clone());
        let mut field = fixture_field();
        assert_eq!(field.forward(&x).unwrap().data(), expected.data());
        assert!(field.backward(&x, &g).unwrap().data().iter().all(|v|*v==0.0));
        assert_eq!(parameters(&field, true), gradients);
        field.apply_step(1e-3).unwrap();
        assert_eq!(parameters(&field, false), updated);
        assert_eq!((x,g), before);
    }}
}

#[test]
fn malformed_seeds_fail_before_parameter_gradients_change() {
    let _policy = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
    let mut field = fixture_field();
    let input = Tensor::from_vec(2, 6, vec![0.125; 12]).unwrap();
    field.backward(&input, &Tensor::from_vec(2, 4, vec![1.0; 8]).unwrap()).unwrap();
    let before = parameters(&field, true);
    for seed in [Tensor::zeros(1, 4).unwrap(), Tensor::zeros(2, 3).unwrap(),
                 Tensor::from_vec(2, 4, vec![f32::NAN;8]).unwrap(),
                 Tensor::from_vec(2, 4, vec![f32::INFINITY;8]).unwrap()] {
        assert!(field.backward(&input, &seed).is_err());
        assert_eq!(parameters(&field, true), before);
    }
}

#[test]
fn field_parameter_vjp_matches_finite_differences() {
    let _policy = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
    let mut field = fixture_field();
    let input = Tensor::from_vec(2, 6, (0..12).map(|i|0.05+i as f32/100.0).collect()).unwrap();
    let seed = Tensor::from_vec(2, 4, vec![0.5,-0.25,0.75,0.125,-0.5,0.25,-0.125,1.0]).unwrap();
    field.backward(&input, &seed).unwrap();
    let values = parameters(&field, false); let gradients = parameters(&field, true);
    for (index, values) in values.iter().enumerate() { for (element, &value) in values.iter().enumerate() {
        let mut losses = Vec::new();
        for offset in [1e-3, -1e-3] {
            let mut current = 0;
            field.visit_parameters_mut(&mut |p| {
                if current == index { p.value_mut().data_mut()[element] = value + offset; }
                current += 1; Ok(())
            }).unwrap();
            losses.push(field.forward(&input).unwrap().data().iter().zip(seed.data())
                .map(|(&a,&b)|f64::from(a)*f64::from(b)).sum::<f64>());
        }
        let mut current = 0;
        field.visit_parameters_mut(&mut |p| {
            if current == index { p.value_mut().data_mut()[element] = value; }
            current += 1; Ok(())
        }).unwrap();
        let expected = (losses[0]-losses[1])/0.002;
        assert!((f64::from(gradients[index][element])-expected).abs() <= 2e-4*(1.0+expected.abs()),
                "parameter {index} element {element}: {} != {expected}", gradients[index][element]);
    }}
}
