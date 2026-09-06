use crate::{Layout, PureResult, Tensor, TensorError};

const MEAN_BLOCK: usize = 1024;

/// Elementwise arithmetic mean of equally shaped, finite tensors, then scaled.
///
/// Each element is summed in input order starting at positive zero in f64,
/// divided by the input count, multiplied by the f32 scale promoted to f64,
/// and rounded once to f32. This is a signed-vector reduction, not the
/// probability/KL `z_space_barycenter`. No rank kernel or GPU is dispatched.
/// Non-finite inputs, scales, and final outputs are rejected, even at scale zero.
/// An empty list is invalid; equally shaped zero-volume tensors are supported.
/// The result is row-major. Scratch accumulation is bounded to 1024 f64 values;
/// row/column-major inputs need no layout-conversion buffer.
pub fn mean_tensors_scaled(partials: &[Tensor], scale: f32) -> PureResult<Tensor> {
    let first = partials
        .first()
        .ok_or(TensorError::EmptyInput("mean_tensors_scaled"))?;
    let (rows, cols) = first.shape();
    for tensor in partials {
        if tensor.shape() != (rows, cols) {
            return Err(TensorError::ShapeMismatch {
                left: (rows, cols),
                right: tensor.shape(),
            });
        }
    }
    finite("mean_tensors_scale", scale)?;
    let logical = partials
        .iter()
        .map(|tensor| {
            let tensor = match tensor.layout() {
                Layout::RowMajor | Layout::ColMajor => tensor.clone(),
                _ => tensor.to_layout(Layout::RowMajor)?,
            };
            Ok(tensor)
        })
        .collect::<PureResult<Vec<_>>>()?;
    let len = first.len();
    let mut output = Vec::with_capacity(len);
    let mut scratch = vec![0.0f64; len.min(MEAN_BLOCK)];
    for start in (0..len).step_by(MEAN_BLOCK) {
        let count = (len - start).min(MEAN_BLOCK);
        let accum = &mut scratch[..count];
        accum.fill(0.0);
        // Block independent output coordinates, never the reduction dimension:
        // reassociating the partials would change cancellation-sensitive results.
        for tensor in &logical {
            let data = tensor.data();
            if tensor.layout() == Layout::ColMajor {
                let mut offset = 0;
                while offset < count {
                    let index = start + offset;
                    let row = index / cols;
                    let col = index % cols;
                    let width = (cols - col).min(count - offset);
                    for (c, dst) in accum[offset..offset + width].iter_mut().enumerate() {
                        accumulate(dst, data[(col + c) * rows + row])?;
                    }
                    offset += width;
                }
            } else {
                for (dst, &src) in accum.iter_mut().zip(&data[start..start + count]) {
                    accumulate(dst, src)?;
                }
            }
        }
        for &sum in accum.iter() {
            let value = (sum / partials.len() as f64 * f64::from(scale)) as f32;
            finite("mean_tensors_result_exceeds_finite_float32", value)?;
            output.push(value);
        }
    }
    Tensor::from_vec(rows, cols, output)
}

fn finite(label: &'static str, value: f32) -> PureResult<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(TensorError::NonFiniteValue { label, value })
    }
}

#[inline]
fn accumulate(dst: &mut f64, src: f32) -> PureResult<()> {
    if !src.is_finite() {
        return Err(TensorError::InvalidValue {
            label: "mean_tensors_partials_must_be_finite",
        });
    }
    *dst += f64::from(src);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn legacy(partials: &[Tensor], scale: f32) -> Tensor {
        let (rows, cols) = partials[0].shape();
        let mut accum = vec![0.0f64; rows * cols];
        for tensor in partials {
            let logical = tensor.to_layout(Layout::RowMajor).unwrap();
            for (dst, &src) in accum.iter_mut().zip(logical.data()) {
                *dst += f64::from(src);
            }
        }
        Tensor::from_vec(
            rows,
            cols,
            accum
                .into_iter()
                .map(|sum| (sum / partials.len() as f64 * f64::from(scale)) as f32)
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn mean_tensors_matches_ordered_reference_across_block_and_layout_boundaries() {
        for (rows, cols) in [(1, 1), (3, 341), (4, 256), (5, 205), (37, 71)] {
            for count in [1, 2, 3, 17] {
                let partials: Vec<_> = (0..count)
                    .map(|p| {
                        let tensor = Tensor::from_fn(rows, cols, |r, c| {
                            ((r * 7 + c * 31 + p * 127) % 1009) as f32 / 31.0 - 16.0
                        })
                        .unwrap();
                        if p % 2 == 1 {
                            tensor.to_layout(Layout::ColMajor).unwrap()
                        } else {
                            tensor
                        }
                    })
                    .collect();
                for scale in [0.0, -0.0, 1.0, -1.75, 1.99999] {
                    let expected = legacy(&partials, scale);
                    let actual = mean_tensors_scaled(&partials, scale).unwrap();
                    assert_eq!(actual.layout(), Layout::RowMajor);
                    assert_eq!(
                        actual
                            .data()
                            .iter()
                            .map(|x| x.to_bits())
                            .collect::<Vec<_>>(),
                        expected
                            .data()
                            .iter()
                            .map(|x| x.to_bits())
                            .collect::<Vec<_>>()
                    );
                }
            }
        }
    }

    #[test]
    fn mean_tensors_preserves_chimera_layout_semantics() {
        let original = Tensor::from_fn(7, 12, |r, c| (r * 12 + c) as f32 - 30.0).unwrap();
        let chimera = original
            .to_layout(Layout::Chimera {
                stripes: 3,
                tile: 4,
            })
            .unwrap();
        let partials = [original, chimera];
        assert_eq!(
            mean_tensors_scaled(&partials, 1.25).unwrap(),
            legacy(&partials, 1.25)
        );
    }

    #[test]
    fn mean_tensors_preserves_f64_order_and_finite_extremes() {
        let partials: Vec<_> = [
            vec![f32::MAX, 16777216.0, 2.0f32.powi(60), f32::from_bits(1)],
            vec![f32::MAX, 1.0, 1.0, f32::from_bits(1)],
            vec![-f32::MAX, -16777216.0, -2.0f32.powi(60), f32::from_bits(1)],
        ]
        .into_iter()
        .map(|data| Tensor::from_vec(1, 4, data).unwrap())
        .collect();
        let actual = mean_tensors_scaled(&partials, 1.0).unwrap();
        assert_eq!(actual, legacy(&partials, 1.0));
        assert_eq!(actual.data()[1], 1.0 / 3.0);
        assert_eq!(actual.data()[2], 0.0);
        assert_eq!(actual.data()[3].to_bits(), 1);
        let reordered = [
            partials[0].clone(),
            partials[2].clone(),
            partials[1].clone(),
        ];
        assert_eq!(
            mean_tensors_scaled(&reordered, 1.0).unwrap().data()[2],
            1.0 / 3.0
        );
        let large = Tensor::from_vec(1, 1, vec![f32::MAX]).unwrap();
        assert_eq!(
            mean_tensors_scaled(&[large.clone(), large.clone()], 1.0).unwrap(),
            large
        );
        assert!(mean_tensors_scaled(&[large], 2.0).is_err());
    }

    #[test]
    fn mean_tensors_rejects_invalid_inputs_without_mutation() {
        assert!(mean_tensors_scaled(&[], 1.0).is_err());
        let good = Tensor::from_vec(1, 2, vec![1.0, -2.0]).unwrap();
        let before = good.clone();
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(mean_tensors_scaled(std::slice::from_ref(&good), value).is_err());
            let bad = Tensor::from_vec(1, 2, vec![value, 0.0]).unwrap();
            assert!(mean_tensors_scaled(&[good.clone(), bad], 0.0).is_err());
        }
        assert!(mean_tensors_scaled(&[good.clone(), Tensor::zeros(2, 1).unwrap()], 1.0).is_err());
        assert_eq!(good, before);
        for (rows, cols) in [(0, 3), (3, 0), (0, 0)] {
            let empty = Tensor::zeros(rows, cols).unwrap();
            assert_eq!(
                mean_tensors_scaled(&[empty.clone(), empty.clone()], 1.0).unwrap(),
                empty
            );
        }
    }
}
