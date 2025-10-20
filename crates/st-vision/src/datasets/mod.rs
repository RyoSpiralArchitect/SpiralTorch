// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use rand::Rng;
use st_tensor::{PureResult, Tensor, TensorError};

/// Bundle describing rays originating from a single calibrated view.
#[derive(Clone, Debug)]
pub struct MultiViewFrame {
    pub origins: Tensor,
    pub directions: Tensor,
    pub colors: Tensor,
    pub bounds: Tensor,
}

impl MultiViewFrame {
    /// Validates and constructs a multi-view frame from the provided tensors.
    pub fn new(
        origins: Tensor,
        directions: Tensor,
        colors: Tensor,
        bounds: Tensor,
    ) -> PureResult<Self> {
        let (rows, ocols) = origins.shape();
        if rows == 0 {
            return Err(TensorError::EmptyInput("multiview_rays"));
        }
        let (drows, dcols) = directions.shape();
        let (crows, ccols) = colors.shape();
        let (brows, bcols) = bounds.shape();
        if rows != drows || rows != crows || rows != brows {
            return Err(TensorError::ShapeMismatch {
                left: (rows, ocols),
                right: (drows, dcols),
            });
        }
        if dcols != ocols || ccols != 3 || bcols != 2 {
            return Err(TensorError::InvalidDimensions {
                rows: rows.max(1),
                cols: dcols.max(ccols).max(bcols),
            });
        }
        Ok(Self {
            origins,
            directions,
            colors,
            bounds,
        })
    }

    /// Number of rays contained in the frame.
    pub fn len(&self) -> usize {
        self.origins.shape().0
    }

    /// Returns true when the frame does not hold any rays.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Random access dataset adapter capable of sampling ray batches.
#[derive(Clone, Debug)]
pub struct MultiViewDatasetAdapter {
    frames: Vec<MultiViewFrame>,
    origin_dims: usize,
}

impl MultiViewDatasetAdapter {
    /// Constructs an adapter from precomputed multi-view frames.
    pub fn new(frames: Vec<MultiViewFrame>) -> PureResult<Self> {
        if frames.is_empty() {
            return Err(TensorError::EmptyInput("multiview_frames"));
        }
        let origin_dims = frames[0].origins.shape().1;
        for frame in &frames {
            if frame.origins.shape().1 != origin_dims || frame.is_empty() {
                return Err(TensorError::InvalidDimensions {
                    rows: frame.origins.shape().0,
                    cols: frame.origins.shape().1,
                });
            }
        }
        Ok(Self {
            frames,
            origin_dims,
        })
    }

    /// Number of calibrated frames in the dataset.
    pub fn num_frames(&self) -> usize {
        self.frames.len()
    }

    /// Dimensionality of the ray origins/directions.
    pub fn origin_dims(&self) -> usize {
        self.origin_dims
    }

    /// Samples a batch of rays using replacement across all frames.
    pub fn sample_batch<R: Rng>(&self, rng: &mut R, batch_size: usize) -> PureResult<RayBatch> {
        if batch_size == 0 {
            return Err(TensorError::InvalidDimensions { rows: 0, cols: 0 });
        }
        let mut origin_buffer = Vec::with_capacity(batch_size * self.origin_dims);
        let mut dir_buffer = Vec::with_capacity(batch_size * self.origin_dims);
        let mut color_buffer = Vec::with_capacity(batch_size * 3);
        let mut bound_buffer = Vec::with_capacity(batch_size * 2);
        for _ in 0..batch_size {
            let frame_idx = rng.gen_range(0..self.frames.len());
            let frame = &self.frames[frame_idx];
            let ray_idx = rng.gen_range(0..frame.len());
            copy_row(
                frame.origins.data(),
                self.origin_dims,
                ray_idx,
                &mut origin_buffer,
            );
            copy_row(
                frame.directions.data(),
                self.origin_dims,
                ray_idx,
                &mut dir_buffer,
            );
            copy_row(frame.colors.data(), 3, ray_idx, &mut color_buffer);
            copy_row(frame.bounds.data(), 2, ray_idx, &mut bound_buffer);
        }
        Ok(RayBatch {
            origins: Tensor::from_vec(batch_size, self.origin_dims, origin_buffer)?,
            directions: Tensor::from_vec(batch_size, self.origin_dims, dir_buffer)?,
            colors: Tensor::from_vec(batch_size, 3, color_buffer)?,
            bounds: Tensor::from_vec(batch_size, 2, bound_buffer)?,
        })
    }

    /// Returns an immutable view over the underlying frames.
    pub fn frames(&self) -> &[MultiViewFrame] {
        &self.frames
    }
}

/// Set of rays returned by [`MultiViewDatasetAdapter::sample_batch`].
#[derive(Clone, Debug)]
pub struct RayBatch {
    pub origins: Tensor,
    pub directions: Tensor,
    pub colors: Tensor,
    pub bounds: Tensor,
}

impl RayBatch {
    /// Number of rays in the batch.
    pub fn len(&self) -> usize {
        self.origins.shape().0
    }

    /// Returns true when the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

fn copy_row(source: &[f32], cols: usize, row: usize, target: &mut Vec<f32>) {
    let offset = row * cols;
    target.extend_from_slice(&source[offset..offset + cols]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn adapter_samples_consistent_batches() {
        let frame = MultiViewFrame::new(
            Tensor::from_vec(2, 3, vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap(),
            Tensor::from_vec(2, 3, vec![0.0, 0.0, 1.0, 0.0, 1.0, 0.0]).unwrap(),
            Tensor::from_vec(2, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap(),
            Tensor::from_vec(2, 2, vec![0.0, 1.0, 0.2, 1.2]).unwrap(),
        )
        .unwrap();
        let dataset = MultiViewDatasetAdapter::new(vec![frame]).unwrap();
        let mut rng = StdRng::seed_from_u64(42);
        let batch = dataset.sample_batch(&mut rng, 4).unwrap();
        assert_eq!(batch.origins.shape(), (4, 3));
        assert_eq!(batch.directions.shape(), (4, 3));
        assert_eq!(batch.colors.shape(), (4, 3));
        assert_eq!(batch.bounds.shape(), (4, 2));
    }
}
