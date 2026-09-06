use st_tensor::{mean_tensors_scaled, PureResult, Tensor, TensorError};

#[cfg(target_arch = "wasm32")]
use crate::utils::{js_error, js_u32};
#[cfg(target_arch = "wasm32")]
use js_sys::Number;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

/// A reusable batch in WASM linear memory, not GPU-resident storage.
/// Construction copies partial-major row-major f32 data once. Repeated means
/// execute the same ordered-f64 Rust reducer used by GoldenRetriever.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct TensorMeanBatch {
    partials: Vec<Tensor>,
}

impl TensorMeanBatch {
    pub fn from_flat(
        rows: usize,
        cols: usize,
        partial_count: usize,
        data: &[f32],
    ) -> PureResult<Self> {
        if partial_count == 0 {
            return Err(TensorError::EmptyInput("mean_tensors_scaled"));
        }
        let volume = rows
            .checked_mul(cols)
            .ok_or(TensorError::InvalidDimensions { rows, cols })?;
        let expected = volume
            .checked_mul(partial_count)
            .ok_or(TensorError::InvalidValue {
                label: "tensor_mean_batch_volume_overflow",
            })?;
        if data.len() != expected {
            return Err(TensorError::DataLength {
                expected,
                got: data.len(),
            });
        }
        // Validate before allocating all partial tensors. Scale is checked by
        // the shared reducer on each call, including for zero-volume batches.
        if !data.iter().all(|value| value.is_finite()) {
            return Err(TensorError::InvalidValue {
                label: "mean_tensors_partials_must_be_finite",
            });
        }
        // Empty partials have no values to reduce; do not allocate a user-sized
        // list of empty tensors (the constructor is also callable from JS).
        let stored_count = if volume == 0 { 1 } else { partial_count };
        let partials = (0..stored_count)
            .map(|index| {
                Tensor::from_vec(
                    rows,
                    cols,
                    data[index * volume..(index + 1) * volume].to_vec(),
                )
            })
            .collect::<PureResult<Vec<_>>>()?;
        Ok(Self { partials })
    }

    pub fn mean_scaled(&self, scale: f32) -> PureResult<Tensor> {
        mean_tensors_scaled(&self.partials, scale)
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl TensorMeanBatch {
    #[wasm_bindgen(constructor)]
    pub fn new(
        rows: Number,
        cols: Number,
        partial_count: Number,
        data: &[f32],
    ) -> Result<TensorMeanBatch, JsValue> {
        let rows = js_u32(rows.as_ref(), "tensor mean rows")? as usize;
        let cols = js_u32(cols.as_ref(), "tensor mean cols")? as usize;
        let partial_count = js_u32(partial_count.as_ref(), "tensor mean partial_count")? as usize;
        Self::from_flat(rows, cols, partial_count, data).map_err(js_error)
    }

    /// Returns a new Float32Array. No input buffer or earlier result is mutated.
    #[wasm_bindgen(js_name = meanScaled)]
    pub fn mean_scaled_js(&self, scale: f32) -> Result<Vec<f32>, JsValue> {
        self.mean_scaled(scale)
            .map(|tensor| tensor.data().to_vec())
            .map_err(js_error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wasm_mean_reuses_inputs_and_preserves_ordered_f64() {
        let data = [16777216.0, f32::MAX, 1.0, f32::MAX, -16777216.0, -f32::MAX];
        let batch = TensorMeanBatch::from_flat(1, 2, 3, &data).unwrap();
        let result = batch.mean_scaled(1.0).unwrap();
        assert_eq!(result.data()[0], 1.0 / 3.0);
        assert_eq!(result.data()[1], (f64::from(f32::MAX) / 3.0) as f32);
        assert_eq!(batch.mean_scaled(1.0).unwrap(), result);
        assert!(batch.mean_scaled(f32::NAN).is_err());
        assert_eq!(batch.mean_scaled(1.0).unwrap(), result);
    }

    #[test]
    fn wasm_mean_checks_shape_count_overflow_and_nonfinite() {
        assert!(TensorMeanBatch::from_flat(1, 1, 0, &[]).is_err());
        assert!(TensorMeanBatch::from_flat(2, 3, 2, &[1.0; 6]).is_err());
        assert!(TensorMeanBatch::from_flat(usize::MAX, 2, 1, &[]).is_err());
        assert!(TensorMeanBatch::from_flat(1, 2, usize::MAX, &[]).is_err());
        assert!(TensorMeanBatch::from_flat(1, 1, 1, &[f32::NAN]).is_err());
        assert_eq!(
            TensorMeanBatch::from_flat(0, 3, 2, &[])
                .unwrap()
                .mean_scaled(1.0)
                .unwrap()
                .shape(),
            (0, 3)
        );
        assert!(TensorMeanBatch::from_flat(0, 3, usize::MAX, &[])
            .unwrap()
            .mean_scaled(1.0)
            .unwrap()
            .is_empty());
    }
}
