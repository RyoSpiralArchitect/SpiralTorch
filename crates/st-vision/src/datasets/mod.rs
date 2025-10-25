// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use rand::Rng;
use st_tensor::{PureResult, Tensor, TensorError};
use std::cmp::Ordering;

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
    frame_weights: Vec<f32>,
    cumulative_weights: Vec<f32>,
    total_weight: f32,
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
        let mut adapter = Self {
            frames,
            origin_dims,
            frame_weights: Vec::new(),
            cumulative_weights: Vec::new(),
            total_weight: 0.0,
        };
        adapter.rebuild_weights(vec![1.0; adapter.frames.len()])?;
        Ok(adapter)
    }

    fn rebuild_weights(&mut self, weights: Vec<f32>) -> PureResult<()> {
        if weights.len() != self.frames.len() {
            return Err(TensorError::DataLength {
                expected: self.frames.len(),
                got: weights.len(),
            });
        }
        if weights.is_empty() {
            return Err(TensorError::EmptyInput("frame_weights"));
        }
        let mut cumulative = Vec::with_capacity(weights.len());
        let mut total = 0.0f32;
        for weight in &weights {
            if !weight.is_finite() || *weight < 0.0 {
                return Err(TensorError::InvalidValue {
                    label: "frame_weight",
                });
            }
            total += *weight;
            cumulative.push(total);
        }
        if total <= 0.0 {
            return Err(TensorError::InvalidValue {
                label: "frame_weights_total",
            });
        }
        self.frame_weights = weights;
        self.cumulative_weights = cumulative;
        self.total_weight = total;
        Ok(())
    }

    /// Number of calibrated frames in the dataset.
    pub fn num_frames(&self) -> usize {
        self.frames.len()
    }

    /// Dimensionality of the ray origins/directions.
    pub fn origin_dims(&self) -> usize {
        self.origin_dims
    }

    /// Returns the currently configured sampling weights per frame.
    pub fn frame_weights(&self) -> &[f32] {
        &self.frame_weights
    }

    /// Updates the sampling weights used by stochastic queries.
    pub fn set_frame_weights(&mut self, weights: &[f32]) -> PureResult<()> {
        self.rebuild_weights(weights.to_vec())
    }

    fn sample_frame_index<R: Rng>(&self, rng: &mut R) -> usize {
        if self.frames.len() == 1 {
            return 0;
        }
        let sample = if self.total_weight.is_finite() {
            rng.gen::<f32>() * self.total_weight
        } else {
            rng.gen_range(0..self.frames.len()) as f32
        };
        match self
            .cumulative_weights
            .binary_search_by(|probe| probe.partial_cmp(&sample).unwrap_or(Ordering::Greater))
        {
            Ok(index) => index,
            Err(index) => index,
        }
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
            let frame_idx = self.sample_frame_index(rng);
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

    /// Samples a contiguous span of rays from a specific frame with optional stride.
    pub fn sample_contiguous_span(
        &self,
        frame_index: usize,
        start: usize,
        count: usize,
        stride: usize,
    ) -> PureResult<RayBatch> {
        if count == 0 {
            return Err(TensorError::InvalidDimensions { rows: 0, cols: 0 });
        }
        if stride == 0 {
            return Err(TensorError::InvalidValue {
                label: "ray_stride",
            });
        }
        let frame = self
            .frames
            .get(frame_index)
            .ok_or(TensorError::InvalidValue {
                label: "frame_index",
            })?;
        let mut origin_buffer = Vec::with_capacity(count * self.origin_dims);
        let mut dir_buffer = Vec::with_capacity(count * self.origin_dims);
        let mut color_buffer = Vec::with_capacity(count * 3);
        let mut bound_buffer = Vec::with_capacity(count * 2);
        for i in 0..count {
            let ray_index = start + i.saturating_mul(stride);
            if ray_index >= frame.len() {
                return Err(TensorError::InvalidValue { label: "ray_index" });
            }
            copy_row(
                frame.origins.data(),
                self.origin_dims,
                ray_index,
                &mut origin_buffer,
            );
            copy_row(
                frame.directions.data(),
                self.origin_dims,
                ray_index,
                &mut dir_buffer,
            );
            copy_row(frame.colors.data(), 3, ray_index, &mut color_buffer);
            copy_row(frame.bounds.data(), 2, ray_index, &mut bound_buffer);
        }
        Ok(RayBatch {
            origins: Tensor::from_vec(count, self.origin_dims, origin_buffer)?,
            directions: Tensor::from_vec(count, self.origin_dims, dir_buffer)?,
            colors: Tensor::from_vec(count, 3, color_buffer)?,
            bounds: Tensor::from_vec(count, 2, bound_buffer)?,
        })
    }

    /// Samples a random contiguous span using the configured frame weights.
    pub fn sample_random_span<R: Rng>(
        &self,
        rng: &mut R,
        span: usize,
        max_stride: usize,
    ) -> PureResult<RayBatch> {
        if span == 0 {
            return Err(TensorError::InvalidDimensions { rows: 0, cols: 0 });
        }
        if max_stride == 0 {
            return Err(TensorError::InvalidValue {
                label: "random_span_stride",
            });
        }
        let frame_index = self.sample_frame_index(rng);
        let frame = &self.frames[frame_index];
        if span > frame.len() {
            return Err(TensorError::InvalidDimensions {
                rows: span,
                cols: frame.len(),
            });
        }
        let stride = if span <= 1 {
            1
        } else {
            let max_possible_stride = (frame.len() - 1) / (span - 1);
            let stride_cap = max_stride.min(max_possible_stride.max(1));
            rng.gen_range(1..=stride_cap)
        };
        let required_extent = (span - 1).saturating_mul(stride).saturating_add(1);
        if required_extent > frame.len() {
            return Err(TensorError::InvalidDimensions {
                rows: required_extent,
                cols: frame.len(),
            });
        }
        let max_start = frame.len() - required_extent;
        let start = if max_start == 0 {
            0
        } else {
            rng.gen_range(0..=max_start)
        };
        self.sample_contiguous_span(frame_index, start, span, stride)
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

    /// Concatenates a list of ray batches into a single batch.
    pub fn stack(batches: &[RayBatch]) -> PureResult<RayBatch> {
        if batches.is_empty() {
            return Err(TensorError::EmptyInput("ray_batches"));
        }
        let expected_shape = batches[0].origins.shape();
        let origin_dims = expected_shape.1;
        let mut total = 0usize;
        for batch in batches {
            if batch.origins.shape().1 != origin_dims {
                return Err(TensorError::ShapeMismatch {
                    left: expected_shape,
                    right: batch.origins.shape(),
                });
            }
            total += batch.len();
        }
        let mut origins = Vec::with_capacity(total * origin_dims);
        let mut directions = Vec::with_capacity(total * origin_dims);
        let mut colors = Vec::with_capacity(total * 3);
        let mut bounds = Vec::with_capacity(total * 2);
        for batch in batches {
            origins.extend_from_slice(batch.origins.data());
            directions.extend_from_slice(batch.directions.data());
            colors.extend_from_slice(batch.colors.data());
            bounds.extend_from_slice(batch.bounds.data());
        }
        Ok(RayBatch {
            origins: Tensor::from_vec(total, origin_dims, origins)?,
            directions: Tensor::from_vec(total, origin_dims, directions)?,
            colors: Tensor::from_vec(total, 3, colors)?,
            bounds: Tensor::from_vec(total, 2, bounds)?,
        })
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

    #[test]
    fn adapter_samples_contiguous_span() {
        let frame = MultiViewFrame::new(
            Tensor::from_vec(
                4,
                3,
                vec![0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 3.0, 3.1, 3.2],
            )
            .unwrap(),
            Tensor::from_vec(
                4,
                3,
                vec![0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.5],
            )
            .unwrap(),
            Tensor::from_vec(
                4,
                3,
                vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5],
            )
            .unwrap(),
            Tensor::from_vec(4, 2, vec![0.0, 1.0, 0.1, 1.1, 0.2, 1.2, 0.3, 1.3]).unwrap(),
        )
        .unwrap();
        let dataset = MultiViewDatasetAdapter::new(vec![frame]).unwrap();
        let batch = dataset
            .sample_contiguous_span(0, 1, 2, 2)
            .expect("contiguous sampling should succeed");
        assert_eq!(batch.len(), 2);
        assert_eq!(batch.origins.data(), &[1.0, 1.1, 1.2, 3.0, 3.1, 3.2]);
        assert_eq!(batch.bounds.data(), &[0.1, 1.1, 0.3, 1.3]);
    }

    #[test]
    fn weighted_sampling_honours_zero_weights() {
        let frame_a = MultiViewFrame::new(
            Tensor::from_vec(1, 3, vec![0.0, 0.0, 0.0]).unwrap(),
            Tensor::from_vec(1, 3, vec![1.0, 0.0, 0.0]).unwrap(),
            Tensor::from_vec(1, 3, vec![1.0, 0.0, 0.0]).unwrap(),
            Tensor::from_vec(1, 2, vec![0.0, 1.0]).unwrap(),
        )
        .unwrap();
        let frame_b = MultiViewFrame::new(
            Tensor::from_vec(1, 3, vec![2.0, 2.0, 2.0]).unwrap(),
            Tensor::from_vec(1, 3, vec![0.0, 1.0, 0.0]).unwrap(),
            Tensor::from_vec(1, 3, vec![0.0, 1.0, 0.0]).unwrap(),
            Tensor::from_vec(1, 2, vec![0.5, 1.5]).unwrap(),
        )
        .unwrap();
        let mut dataset = MultiViewDatasetAdapter::new(vec![frame_a.clone(), frame_b]).unwrap();
        dataset
            .set_frame_weights(&[1.0, 0.0])
            .expect("weights should be accepted");
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..8 {
            let batch = dataset.sample_batch(&mut rng, 1).unwrap();
            assert_eq!(batch.origins.data(), frame_a.origins.data());
        }
    }

    #[test]
    fn random_span_respects_stride_constraints() {
        let frame = MultiViewFrame::new(
            Tensor::from_vec(
                5,
                3,
                vec![
                    0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 3.0, 3.1, 3.2, 4.0, 4.1, 4.2,
                ],
            )
            .unwrap(),
            Tensor::from_vec(
                5,
                3,
                vec![
                    0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0,
                ],
            )
            .unwrap(),
            Tensor::from_vec(
                5,
                3,
                vec![
                    1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 1.0, 0.5, 0.0,
                ],
            )
            .unwrap(),
            Tensor::from_vec(5, 2, vec![0.0, 1.0, 0.1, 1.1, 0.2, 1.2, 0.3, 1.3, 0.4, 1.4]).unwrap(),
        )
        .unwrap();
        let dataset = MultiViewDatasetAdapter::new(vec![frame]).unwrap();
        let mut rng = StdRng::seed_from_u64(27);
        let batch = dataset
            .sample_random_span(&mut rng, 3, 2)
            .expect("random span should succeed");
        assert_eq!(batch.len(), 3);
        let bounds = batch.bounds.data();
        let mut previous_start = None;
        for chunk in bounds.chunks_exact(2) {
            assert!(chunk[0] < chunk[1]);
            if let Some(prev) = previous_start {
                assert!(chunk[0] >= prev);
            }
            previous_start = Some(chunk[0]);
        }
    }

    #[test]
    fn stacking_batches_concatenates_inputs() {
        let frame = MultiViewFrame::new(
            Tensor::from_vec(2, 3, vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap(),
            Tensor::from_vec(2, 3, vec![0.0, 0.0, 1.0, 0.0, 1.0, 0.0]).unwrap(),
            Tensor::from_vec(2, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap(),
            Tensor::from_vec(2, 2, vec![0.0, 1.0, 0.2, 1.2]).unwrap(),
        )
        .unwrap();
        let dataset = MultiViewDatasetAdapter::new(vec![frame]).unwrap();
        let mut rng = StdRng::seed_from_u64(123);
        let batch_a = dataset.sample_batch(&mut rng, 2).unwrap();
        let batch_b = dataset.sample_batch(&mut rng, 1).unwrap();
        let stacked = RayBatch::stack(&[batch_a.clone(), batch_b.clone()]).unwrap();
        assert_eq!(stacked.len(), batch_a.len() + batch_b.len());
        assert_eq!(stacked.origins.shape().1, batch_a.origins.shape().1);
    }
}
