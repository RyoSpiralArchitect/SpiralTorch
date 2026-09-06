use st_tensor::{mean_tensors_scaled, Layout, Tensor};

#[test]
fn mixed_mean_rejects_nonfinite_values_in_tail_tiles_even_at_zero_scale() {
    for layout in [Layout::RowMajor, Layout::ColMajor] {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let good = Tensor::zeros(17, 65).unwrap();
            let bad = Tensor::from_fn(17, 65, |r, c| if r == 16 && c == 64 { value } else { 1.0 })
                .unwrap()
                .to_layout(layout)
                .unwrap();
            let before: Vec<_> = bad.data().iter().map(|x| x.to_bits()).collect();
            let partials = [good.clone().to_layout(Layout::ColMajor).unwrap(), good, bad];
            assert!(mean_tensors_scaled(&partials, 0.0).is_err());
            assert_eq!(
                partials[2]
                    .data()
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>(),
                before
            );
        }
    }
}

#[test]
fn mixed_mean_preserves_cancellation_signed_zero_and_output_range() {
    let partials: Vec<_> = [2.0f32.powi(60), 1.0, -2.0f32.powi(60)]
        .into_iter()
        .map(|value| {
            Tensor::from_fn(17, 65, |_, _| value)
                .unwrap()
                .to_layout(Layout::ColMajor)
                .unwrap()
        })
        .collect();
    let result = mean_tensors_scaled(&partials, -1.25).unwrap();
    assert!(result
        .data()
        .iter()
        .all(|value| value.to_bits() == (-0.0f32).to_bits()));
    let large = Tensor::from_fn(17, 65, |_, _| f32::MAX)
        .unwrap()
        .to_layout(Layout::ColMajor)
        .unwrap();
    assert!(mean_tensors_scaled(&[large.clone(), large.clone()], 2.0).is_err());
    assert!(mean_tensors_scaled(&[large], 1.0)
        .unwrap()
        .data()
        .iter()
        .all(|value| *value == f32::MAX));
}
