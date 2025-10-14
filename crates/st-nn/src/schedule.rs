use crate::plan::RankPlanner;
use crate::{PureResult, Tensor};
use st_core::ops::rank_entry::RankPlan;
use st_tensor::pure::TensorError;

/// Configuration used to derive the A/B/C roundtable schedule.
#[derive(Debug, Clone, Copy)]
pub struct RoundtableConfig {
    pub top_k: u32,
    pub mid_k: u32,
    pub bottom_k: u32,
    pub here_tolerance: f32,
}

impl Default for RoundtableConfig {
    fn default() -> Self {
        Self {
            top_k: 8,
            mid_k: 8,
            bottom_k: 8,
            here_tolerance: 1e-5,
        }
    }
}

impl RoundtableConfig {
    /// Overrides the TopK count.
    pub fn with_top_k(mut self, top_k: u32) -> Self {
        self.top_k = top_k;
        self
    }

    /// Overrides the MidK count.
    pub fn with_mid_k(mut self, mid_k: u32) -> Self {
        self.mid_k = mid_k;
        self
    }

    /// Overrides the BottomK count.
    pub fn with_bottom_k(mut self, bottom_k: u32) -> Self {
        self.bottom_k = bottom_k;
        self
    }

    /// Overrides the tolerance that decides which entries stay in-place.
    pub fn with_here_tolerance(mut self, tol: f32) -> Self {
        self.here_tolerance = tol.max(0.0);
        self
    }
}

/// Roundtable schedule that binds TopK (Above), MidK (Here), and BottomK
/// (Beneath) ranks to gradient bands.
#[derive(Debug, Clone)]
pub struct RoundtableSchedule {
    above: RankPlan,
    here: RankPlan,
    beneath: RankPlan,
    here_tolerance: f32,
}

impl RoundtableSchedule {
    /// Builds a schedule for the provided output shape.
    pub fn new(
        planner: &RankPlanner,
        rows: u32,
        cols: u32,
        config: RoundtableConfig,
    ) -> Self {
        let above = planner.topk(rows, cols, config.top_k);
        let here = planner.midk(rows, cols, config.mid_k);
        let beneath = planner.bottomk(rows, cols, config.bottom_k);
        Self {
            above,
            here,
            beneath,
            here_tolerance: config.here_tolerance,
        }
    }

    /// Returns the TopK plan (Above band).
    pub fn above(&self) -> &RankPlan {
        &self.above
    }

    /// Returns the MidK plan (Here band).
    pub fn here(&self) -> &RankPlan {
        &self.here
    }

    /// Returns the BottomK plan (Beneath band).
    pub fn beneath(&self) -> &RankPlan {
        &self.beneath
    }

    /// Splits a gradient tensor into Above/Here/Beneath bands.
    pub fn split(&self, gradient: &Tensor) -> PureResult<GradientBands> {
        let len = gradient.data().len();
        if len == 0 {
            return Err(TensorError::DataLength {
                expected: 1,
                got: 0,
            });
        }
        let mut indexed: Vec<(usize, f32)> = gradient
            .data()
            .iter()
            .enumerate()
            .map(|(idx, value)| (idx, value.abs()))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top = usize::min(self.above.k as usize, len);
        let bottom = usize::min(self.beneath.k as usize, len.saturating_sub(top));
        let leftover = len.saturating_sub(top + bottom);
        let mid = usize::min(self.here.k as usize, leftover);

        // Prepare assignment vector defaulting to Here.
        #[derive(Clone, Copy, PartialEq)]
        enum Band {
            Above,
            Here,
            Beneath,
        }
        let mut assignment = vec![Band::Here; len];

        for &(idx, _) in indexed.iter().take(top) {
            assignment[idx] = Band::Above;
        }

        for &(idx, _) in indexed.iter().rev().take(bottom) {
            assignment[idx] = Band::Beneath;
        }

        // Assign middle slice explicitly to Here respecting `mid`.
        if mid > 0 {
            let start = top;
            let end = len.saturating_sub(bottom);
            let mut assigned = 0usize;
            for &(idx, _) in indexed[start..end].iter() {
                assignment[idx] = Band::Here;
                assigned += 1;
                if assigned >= mid {
                    break;
                }
            }
        }

        // Apply tolerance: very small magnitudes stay in the Here band.
        for (idx, magnitude) in indexed.iter() {
            if *magnitude <= self.here_tolerance {
                assignment[*idx] = Band::Here;
            }
        }

        let (rows, cols) = gradient.shape();
        let mut above_data = vec![0.0f32; len];
        let mut here_data = vec![0.0f32; len];
        let mut beneath_data = vec![0.0f32; len];
        for (idx, value) in gradient.data().iter().enumerate() {
            match assignment[idx] {
                Band::Above => above_data[idx] = *value,
                Band::Here => here_data[idx] = *value,
                Band::Beneath => beneath_data[idx] = *value,
            }
        }

        Ok(GradientBands {
            above: Tensor::from_vec(rows, cols, above_data)?,
            here: Tensor::from_vec(rows, cols, here_data)?,
            beneath: Tensor::from_vec(rows, cols, beneath_data)?,
        })
    }
}

/// Gradient components that mirror the roundtable decisions.
#[derive(Debug, Clone)]
pub struct GradientBands {
    above: Tensor,
    here: Tensor,
    beneath: Tensor,
}

impl GradientBands {
    /// Returns the gradient for the Above (TopK/A) band.
    pub fn above(&self) -> &Tensor {
        &self.above
    }

    /// Returns the gradient for the Here (MidK/B) band.
    pub fn here(&self) -> &Tensor {
        &self.here
    }

    /// Returns the gradient for the Beneath (BottomK/C) band.
    pub fn beneath(&self) -> &Tensor {
        &self.beneath
    }

    /// Returns an iterator over every band.
    pub fn iter(&self) -> [&Tensor; 3] {
        [&self.above, &self.here, &self.beneath]
    }

    /// Combines all bands back into a single tensor.
    pub fn combine(&self) -> PureResult<Tensor> {
        let (rows, cols) = self.above.shape();
        let mut data = Vec::with_capacity(rows * cols);
        for idx in 0..(rows * cols) {
            let value = self.above.data()[idx] + self.here.data()[idx] + self.beneath.data()[idx];
            data.push(value);
        }
        Tensor::from_vec(rows, cols, data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_splits_gradients() {
        let planner = RankPlanner::new(st_core::backend::device_caps::DeviceCaps::wgpu(32, true, 256));
        let schedule = RoundtableSchedule::new(&planner, 1, 8, RoundtableConfig::default());
        let gradient = Tensor::from_vec(1, 8, vec![
            -0.1, 0.2, -0.05, 0.9, -1.2, 0.0, 0.3, -0.4,
        ])
        .unwrap();
        let bands = schedule.split(&gradient).unwrap();
        let recombined = bands.combine().unwrap();
        assert_eq!(gradient, recombined);
        let energy = bands.above().squared_l2_norm()
            + bands.here().squared_l2_norm()
            + bands.beneath().squared_l2_norm();
        assert!(energy > 0.0);
    }
}
