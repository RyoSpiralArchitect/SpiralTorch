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

pub fn accumulate_inplace(target: &mut Tensor, update: &Tensor) -> PureResult<()> {
    ensure_same_shape(target, update, "accumulate_inplace")?;
    for (dst, &src) in target.data_mut().iter_mut().zip(update.data().iter()) {
        *dst += src;
    }
    Ok(())
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
