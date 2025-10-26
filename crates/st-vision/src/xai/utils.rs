use st_tensor::{PureResult, Tensor, TensorError};

pub fn ensure_same_shape(lhs: &Tensor, rhs: &Tensor, _label: &'static str) -> PureResult<()> {
    if lhs.shape() != rhs.shape() {
        return Err(TensorError::ShapeMismatch {
            left: lhs.shape(),
            right: rhs.shape(),
        });
    }
    Ok(())
}

pub fn elementwise_binary(
    lhs: &Tensor,
    rhs: &Tensor,
    mut op: impl FnMut(f32, f32) -> f32,
) -> PureResult<Tensor> {
    ensure_same_shape(lhs, rhs, "elementwise_binary")?;
    let (rows, cols) = lhs.shape();
    let mut data = Vec::with_capacity(rows * cols);
    for (&l, &r) in lhs.data().iter().zip(rhs.data().iter()) {
        data.push(op(l, r));
    }
    Tensor::from_vec(rows, cols, data)
}

pub fn elementwise_unary(tensor: &Tensor, mut op: impl FnMut(f32) -> f32) -> PureResult<Tensor> {
    let (rows, cols) = tensor.shape();
    let mut data = Vec::with_capacity(rows * cols);
    for &value in tensor.data() {
        data.push(op(value));
    }
    Tensor::from_vec(rows, cols, data)
}

pub fn add(lhs: &Tensor, rhs: &Tensor) -> PureResult<Tensor> {
    elementwise_binary(lhs, rhs, |l, r| l + r)
}

pub fn sub(lhs: &Tensor, rhs: &Tensor) -> PureResult<Tensor> {
    elementwise_binary(lhs, rhs, |l, r| l - r)
}

pub fn mul(lhs: &Tensor, rhs: &Tensor) -> PureResult<Tensor> {
    elementwise_binary(lhs, rhs, |l, r| l * r)
}

pub fn scale(tensor: &Tensor, factor: f32) -> PureResult<Tensor> {
    elementwise_unary(tensor, |value| value * factor)
}

pub fn clamp(tensor: &Tensor, min: f32, max: f32) -> PureResult<Tensor> {
    elementwise_unary(tensor, |value| value.clamp(min, max))
}

pub fn accumulate_inplace(target: &mut Tensor, update: &Tensor) -> PureResult<()> {
    ensure_same_shape(target, update, "accumulate_inplace")?;
    for (dst, &src) in target.data_mut().iter_mut().zip(update.data().iter()) {
        *dst += src;
    }
    Ok(())
}

pub fn normalise_tensor(tensor: &Tensor, epsilon: f32) -> PureResult<Tensor> {
    let mut values = tensor.data().to_vec();
    normalise_unit_interval(&mut values, epsilon);
    Tensor::from_vec(tensor.shape().0, tensor.shape().1, values)
}

pub fn threshold_mask(tensor: &Tensor, threshold: f32) -> PureResult<Tensor> {
    elementwise_unary(tensor, |value| if value >= threshold { 1.0 } else { 0.0 })
}

pub fn normalise_unit_interval(values: &mut [f32], epsilon: f32) {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &value in values.iter() {
        if value < min {
            min = value;
        }
        if value > max {
            max = value;
        }
    }
    let range = (max - min).abs().max(epsilon);
    if range <= epsilon {
        for value in values.iter_mut() {
            *value = 0.0;
        }
        return;
    }
    for value in values.iter_mut() {
        *value = (*value - min) / range;
    }
}

pub fn box_blur(tensor: &Tensor, kernel_size: usize) -> PureResult<Tensor> {
    if kernel_size == 0 || kernel_size % 2 == 0 {
        return Err(TensorError::InvalidValue {
            label: "box_blur_kernel",
        });
    }
    let (rows, cols) = tensor.shape();
    if rows == 0 || cols == 0 {
        return Tensor::from_vec(rows, cols, tensor.data().to_vec());
    }
    let radius = kernel_size / 2;
    let mut output = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let r_min = r.saturating_sub(radius);
            let r_max = (r + radius).min(rows - 1);
            let c_min = c.saturating_sub(radius);
            let c_max = (c + radius).min(cols - 1);
            let mut accum = 0.0f32;
            let mut count = 0usize;
            for rr in r_min..=r_max {
                for cc in c_min..=c_max {
                    accum += tensor.data()[rr * cols + cc];
                    count += 1;
                }
            }
            output[r * cols + c] = accum / count.max(1) as f32;
        }
    }
    Tensor::from_vec(rows, cols, output)
}

pub fn blend_heatmap(base: &Tensor, heatmap: &Tensor, alpha: f32) -> PureResult<Tensor> {
    ensure_same_shape(base, heatmap, "blend_heatmap")?;
    let alpha = alpha.clamp(0.0, 1.0);
    let beta = 1.0 - alpha;
    let (rows, cols) = base.shape();
    let mut output = Vec::with_capacity(rows * cols);
    for (&b, &h) in base.data().iter().zip(heatmap.data().iter()) {
        output.push(b * beta + h * alpha);
    }
    Tensor::from_vec(rows, cols, output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_blur_applies_local_average() {
        let tensor = Tensor::from_vec(2, 2, vec![0.0, 1.0, 1.0, 0.0]).unwrap();
        let blurred = box_blur(&tensor, 3).unwrap();
        let values = blurred.data();
        assert!(values[0] > 0.0 && values[3] > 0.0);
        assert!((values[1] - values[2]).abs() < 1e-6);
    }

    #[test]
    fn blend_heatmap_interpolates() {
        let base = Tensor::from_vec(1, 4, vec![0.0, 0.0, 0.0, 0.0]).unwrap();
        let heatmap = Tensor::from_vec(1, 4, vec![1.0, 0.5, 0.25, 0.0]).unwrap();
        let blended = blend_heatmap(&base, &heatmap, 0.5).unwrap();
        let expected = vec![0.5, 0.25, 0.125, 0.0];
        for (value, expected) in blended.data().iter().zip(expected.iter()) {
            assert!((value - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn normalise_and_threshold() {
        let tensor = Tensor::from_vec(1, 4, vec![0.0, 2.0, 4.0, 6.0]).unwrap();
        let normalised = normalise_tensor(&tensor, 1e-6).unwrap();
        let mask = threshold_mask(&normalised, 0.5).unwrap();
        assert!(normalised.data()[0] <= 0.01);
        assert_eq!(mask.data()[0], 0.0);
        assert_eq!(mask.data()[3], 1.0);
    }
}
