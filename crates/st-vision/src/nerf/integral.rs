use super::{checked_elements, row_major, validate_finite};
use crate::tensor_contract::checked_allocation;
use st_tensor::{PureResult, Tensor, TensorError};

pub(super) struct RayIntegral {
    pub colors: Tensor,
    pub loss: f32,
    pub avg_transmittance: f32,
    pub gradient: Option<Tensor>,
}

/// Black-background, piecewise-constant quadrature in ray-parameter units.
/// The optional MSE VJP uses the exact same forward pass as rendering. Only
/// one ray's weights/transmittance are cached, not the whole sample batch.
pub(super) fn integrate_rays(
    outputs: &Tensor,
    widths: &[f64],
    samples_per_ray: usize,
    targets: Option<&Tensor>,
) -> PureResult<RayIntegral> {
    let (samples, cols) = outputs.shape();
    if samples_per_ray == 0 || samples == 0 || cols != 4 || samples % samples_per_ray != 0 {
        return Err(TensorError::InvalidDimensions {
            rows: samples,
            cols,
        });
    }
    if widths.len() != samples {
        return Err(TensorError::DataLength {
            expected: samples,
            got: widths.len(),
        });
    }
    if widths.iter().any(|d| !d.is_finite() || *d < 0.0) {
        return Err(TensorError::InvalidValue {
            label: "ray_sample_width",
        });
    }
    let batch_size = samples / samples_per_ray;
    checked_elements(samples, 4)?;
    checked_allocation::<f64>(samples_per_ray, 1)?;
    let color_len = checked_elements(batch_size, 3)?;
    let outputs = row_major(outputs)?;
    validate_finite(outputs.data(), "nerf_outputs")?;
    let targets = targets.map(row_major).transpose()?;
    if let Some(targets) = &targets {
        if targets.shape() != (batch_size, 3) {
            return Err(TensorError::ShapeMismatch {
                left: (batch_size, 3),
                right: targets.shape(),
            });
        }
        validate_finite(targets.data(), "ray_colors")?;
    }
    let mut colors = Tensor::zeros(batch_size, 3)?;
    let mut gradient = targets
        .as_ref()
        .map(|_| Tensor::zeros(samples, 4))
        .transpose()?;
    let cache_len = if targets.is_some() {
        samples_per_ray
    } else {
        0
    };
    let mut weights = vec![0.0f64; cache_len];
    let mut trans_after = vec![0.0f64; cache_len];
    let mut loss = 0.0f64;
    let mut trans_sum = 0.0f64;
    let grad_scale = 2.0 / color_len as f64;
    for ray in 0..batch_size {
        let start = ray * samples_per_ray;
        let samples = &outputs.data()[start * 4..(start + samples_per_ray) * 4];
        let widths = &widths[start..start + samples_per_ray];
        let mut trans = 1.0f64;
        let mut color = [0.0f64; 3];
        for (i, rgba) in samples.as_chunks::<4>().0.iter().enumerate() {
            let tau = f64::from(rgba[0].max(0.0)) * widths[i];
            // expm1 retains thin-slab opacity; exp retains the derivative of
            // opaque slabs even when alpha rounds to one.
            let alpha = -(-tau).exp_m1();
            let weight = trans * alpha;
            trans *= (-tau).exp();
            for channel in 0..3 {
                color[channel] += weight * f64::from(rgba[channel + 1]);
            }
            if targets.is_some() {
                weights[i] = weight;
                trans_after[i] = trans;
            }
        }
        for (channel, &value) in color.iter().enumerate() {
            colors.data_mut()[ray * 3 + channel] = value as f32;
        }
        trans_sum += trans;
        if let (Some(targets), Some(gradient)) = (&targets, &mut gradient) {
            let mut grad_color = [0.0f64; 3];
            for channel in 0..3 {
                let diff = color[channel] - f64::from(targets.data()[ray * 3 + channel]);
                loss += diff * diff;
                grad_color[channel] = diff * grad_scale;
            }
            let grad = &mut gradient.data_mut()[start * 4..(start + samples_per_ray) * 4];
            let mut future_sum = 0.0f64;
            for i in (0..samples_per_ray).rev() {
                let off = i * 4;
                let mut dot = 0.0f64;
                for channel in 0..3 {
                    dot += grad_color[channel] * f64::from(samples[off + channel + 1]);
                    grad[off + channel + 1] = (grad_color[channel] * weights[i]) as f32;
                }
                grad[off] = if samples[off] > 0.0 {
                    (widths[i] * (trans_after[i] * dot - future_sum)) as f32
                } else {
                    0.0
                };
                future_sum += weights[i] * dot;
            }
        }
    }
    let loss = (loss / color_len as f64) as f32;
    validate_finite(colors.data(), "nerf_render_colors")?;
    validate_finite(&[loss], "nerf_loss")?;
    if let Some(gradient) = &gradient {
        validate_finite(gradient.data(), "nerf_output_gradient")?;
    }
    Ok(RayIntegral {
        colors,
        loss,
        avg_transmittance: (trans_sum / batch_size as f64) as f32,
        gradient,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use st_tensor::Layout;

    #[test]
    fn vjp_matches_finite_differences_for_multiple_rays_and_layouts() {
        let outputs = Tensor::from_vec(
            6,
            4,
            vec![
                0.7, 0.4, -0.2, 0.1, 1.2, 0.1, 0.3, 0.9, -0.3, 0.8, 0.2, 0.4, 2.0, 0.5, 0.1, 0.3,
                0.4, 0.2, 0.9, 0.1, 1.5, 0.3, 0.4, 0.2,
            ],
        )
        .unwrap();
        let widths = [0.2, 0.3, 0.4, 0.1, 0.25, 0.5];
        let targets = Tensor::from_vec(2, 3, vec![0.1, 0.2, 0.3, 0.4, 0.1, 0.2]).unwrap();
        let result = integrate_rays(&outputs, &widths, 3, Some(&targets)).unwrap();
        let gradient = result.gradient.unwrap();
        let render = integrate_rays(&outputs, &widths, 3, None).unwrap();
        assert_eq!(render.colors.data(), result.colors.data());
        for i in 0..outputs.len() {
            let mut plus = outputs.clone();
            let mut minus = outputs.clone();
            plus.data_mut()[i] += 1e-3;
            minus.data_mut()[i] -= 1e-3;
            let a = integrate_rays(&plus, &widths, 3, Some(&targets))
                .unwrap()
                .loss;
            let b = integrate_rays(&minus, &widths, 3, Some(&targets))
                .unwrap()
                .loss;
            let finite_difference = (a - b) / (plus.data()[i] - minus.data()[i]);
            assert!(
                (finite_difference - gradient.data()[i]).abs() < 3e-6,
                "index {i}: {finite_difference} != {}",
                gradient.data()[i]
            );
        }
        for layout in [
            Layout::ColMajor,
            Layout::Chimera {
                stripes: 2,
                tile: 2,
            },
        ] {
            let altered = integrate_rays(
                &outputs.to_layout(layout).unwrap(),
                &widths,
                3,
                Some(&targets.to_layout(Layout::ColMajor).unwrap()),
            )
            .unwrap();
            assert_eq!(altered.colors.data(), result.colors.data());
            assert_eq!(altered.loss, result.loss);
            assert_eq!(altered.gradient.unwrap().data(), gradient.data());
        }
    }

    #[test]
    fn transparent_and_opaque_slabs_have_stable_values_and_gradients() {
        let target = Tensor::zeros(1, 3).unwrap();
        for width in [0.0, 1e-12, 1.0, 20.0, f64::from(f32::MAX) * 2.0] {
            let outputs = Tensor::from_vec(1, 4, vec![1.0, 0.4, 0.2, 0.1]).unwrap();
            let result = integrate_rays(&outputs, &[width], 1, Some(&target)).unwrap();
            let alpha = -(-width).exp_m1();
            let expected_color = f64::from(0.4f32) * alpha;
            assert!(
                (f64::from(result.colors.data()[0]) - expected_color).abs()
                    <= 1e-7 * expected_color.abs()
            );
            let expected_density_gradient = (2.0 / 3.0)
                * [0.4f32, 0.2, 0.1]
                    .iter()
                    .map(|&c| f64::from(c).powi(2))
                    .sum::<f64>()
                * alpha
                * width
                * (-width).exp();
            assert!(
                (f64::from(result.gradient.unwrap().data()[0]) - expected_density_gradient).abs()
                    <= 1e-7 * expected_density_gradient.abs()
            );
        }
        for sigma in [-1.0, 0.0] {
            let outputs = Tensor::from_vec(1, 4, vec![sigma, 0.4, 0.2, 0.1]).unwrap();
            let result = integrate_rays(&outputs, &[1.0], 1, Some(&target)).unwrap();
            assert_eq!(result.colors.data(), &[0.0; 3]);
            assert_eq!(result.gradient.unwrap().data(), &[0.0; 4]);
            assert_eq!(result.avg_transmittance, 1.0);
        }
    }

    #[test]
    fn invalid_shapes_values_widths_and_unrepresentable_loss_are_errors() {
        let outputs = Tensor::from_vec(1, 4, vec![1.0, 0.4, 0.2, 0.1]).unwrap();
        for widths in [vec![], vec![-1.0], vec![f64::NAN], vec![f64::INFINITY]] {
            assert!(integrate_rays(&outputs, &widths, 1, None).is_err());
        }
        for samples in [0, 2] {
            assert!(integrate_rays(&outputs, &[1.0], samples, None).is_err());
        }
        assert!(integrate_rays(&outputs, &[1.0], 1, Some(&Tensor::zeros(2, 3).unwrap())).is_err());
        for i in 0..4 {
            let mut bad = outputs.clone();
            bad.data_mut()[i] = f32::NAN;
            assert!(integrate_rays(&bad, &[1.0], 1, None).is_err());
        }
        let huge_target = Tensor::from_vec(1, 3, vec![f32::MAX; 3]).unwrap();
        assert!(integrate_rays(&outputs, &[1.0], 1, Some(&huge_target)).is_err());
    }
}
