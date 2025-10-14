use crate::plan::RankPlanner;
use crate::{PureResult, Tensor};
use st_core::ops::rank_entry::RankPlan;
use st_core::ops::zspace_round::{self, RoundtableBand, RoundtableError};
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
    pub fn new(planner: &RankPlanner, rows: u32, cols: u32, config: RoundtableConfig) -> Self {
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
        let assignment = match zspace_round::classify_roundtable(
            gradient.data(),
            &self.above,
            &self.here,
            &self.beneath,
            self.here_tolerance,
        ) {
            Ok(assign) => assign,
            Err(RoundtableError::EmptyGradient) => {
                return Err(TensorError::DataLength {
                    expected: 1,
                    got: 0,
                })
            }
        };

        let (rows, cols) = gradient.shape();
        let mut above_data = vec![0.0f32; len];
        let mut here_data = vec![0.0f32; len];
        let mut beneath_data = vec![0.0f32; len];
        for (idx, value) in gradient.data().iter().enumerate() {
            match assignment.band(idx) {
                RoundtableBand::Above => above_data[idx] = *value,
                RoundtableBand::Here => here_data[idx] = *value,
                RoundtableBand::Beneath => beneath_data[idx] = *value,
            }
        }

        Ok(GradientBands {
            above: Tensor::from_vec(rows, cols, above_data)?,
            here: Tensor::from_vec(rows, cols, here_data)?,
            beneath: Tensor::from_vec(rows, cols, beneath_data)?,
        })
    }

    /// Returns the absolute sum of each band without allocating tensors.
    pub fn band_energy(&self, gradient: &Tensor) -> PureResult<BandEnergy> {
        let assignment = zspace_round::classify_roundtable(
            gradient.data(),
            &self.above,
            &self.here,
            &self.beneath,
            self.here_tolerance,
        )
        .map_err(|err| match err {
            RoundtableError::EmptyGradient => TensorError::DataLength {
                expected: 1,
                got: 0,
            },
        })?;
        let (above, here, beneath) = assignment.energy(gradient.data());
        Ok(BandEnergy {
            above,
            here,
            beneath,
            drift: 0.0,
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

    /// Applies per-band scaling factors in-place.
    pub fn scale_inplace(&mut self, above: f32, here: f32, beneath: f32) {
        for value in self.above.data_mut() {
            *value *= above;
        }
        for value in self.here.data_mut() {
            *value *= here;
        }
        for value in self.beneath.data_mut() {
            *value *= beneath;
        }
    }
}

/// Aggregate magnitude per roundtable band.
#[derive(Debug, Clone, Copy)]
pub struct BandEnergy {
    pub above: f32,
    pub here: f32,
    pub beneath: f32,
    pub drift: f32,
}

impl BandEnergy {
    /// Returns the L1 norm of the band magnitudes.
    pub fn l1(&self) -> f32 {
        self.above.abs() + self.here.abs() + self.beneath.abs()
    }

    /// Normalises the energy so the Above/Here/Beneath components sum to one.
    pub fn norm(self) -> Self {
        let sum = self.l1();
        if sum <= f32::EPSILON {
            return Self {
                above: 1.0 / 3.0,
                here: 1.0 / 3.0,
                beneath: 1.0 / 3.0,
                drift: self.drift,
            };
        }
        Self {
            above: (self.above / sum).clamp(0.0, 1.0),
            here: (self.here / sum).clamp(0.0, 1.0),
            beneath: (self.beneath / sum).clamp(0.0, 1.0),
            drift: self.drift,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_splits_gradients() {
        let planner = RankPlanner::new(st_core::backend::device_caps::DeviceCaps::wgpu(
            32, true, 256,
        ));
        let schedule = RoundtableSchedule::new(&planner, 1, 8, RoundtableConfig::default());
        let gradient =
            Tensor::from_vec(1, 8, vec![-0.1, 0.2, -0.05, 0.9, -1.2, 0.0, 0.3, -0.4]).unwrap();
        let bands = schedule.split(&gradient).unwrap();
        let recombined = bands.combine().unwrap();
        assert_eq!(gradient, recombined);
        let energy = bands.above().squared_l2_norm()
            + bands.here().squared_l2_norm()
            + bands.beneath().squared_l2_norm();
        assert!(energy > 0.0);
    }

    #[test]
    fn band_energy_matches_split() {
        let planner = RankPlanner::new(st_core::backend::device_caps::DeviceCaps::wgpu(
            32, true, 256,
        ));
        let schedule = RoundtableSchedule::new(&planner, 1, 6, RoundtableConfig::default());
        let gradient = Tensor::from_vec(1, 6, vec![0.5, -0.2, 0.1, -0.4, 0.9, 0.0]).unwrap();
        let energy = schedule.band_energy(&gradient).unwrap();
        let bands = schedule.split(&gradient).unwrap();
        let recon = bands.combine().unwrap();
        assert_eq!(gradient, recon);
        let above = bands.above().data().iter().map(|v| v.abs()).sum::<f32>();
        let here = bands.here().data().iter().map(|v| v.abs()).sum::<f32>();
        let beneath = bands.beneath().data().iter().map(|v| v.abs()).sum::<f32>();
        assert!((energy.above - above).abs() < 1e-6);
        assert!((energy.here - here).abs() < 1e-6);
        assert!((energy.beneath - beneath).abs() < 1e-6);
    }
}
