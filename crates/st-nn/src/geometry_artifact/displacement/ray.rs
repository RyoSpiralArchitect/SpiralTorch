//! One normalization before the single f32 rounding of each selected weight.
use super::{check_distance_range, check_realization, GeometryDeltaMatch, Result, Squares};

#[derive(Default)]
pub(crate) struct GradientSize {
    pub(crate) values: usize,
    sum: Squares,
}

impl GradientSize {
    pub(crate) fn include<T: Copy + Into<f64>>(&mut self, block: &[T]) {
        self.values += block.len();
        for &g in block {
            self.sum.add(g.into());
        }
    }
    pub(crate) fn norm(&self) -> f64 {
        self.sum.norm()
    }
}

#[derive(Default)]
pub(crate) struct RayMotion {
    actual: Squares,
    error: Squares,
    pub(crate) dot: f64,
}

impl RayMotion {
    pub(crate) fn norm(&self) -> f64 {
        self.actual.norm()
    }
    pub(crate) fn relative_error(&self) -> f64 {
        self.error.norm()
    }
}

pub(crate) struct ParameterRay {
    pub(crate) target: f64,
    norm: f64,
}

impl ParameterRay {
    pub(crate) fn admit_distance(target: f64, count: usize) -> Result<()> {
        GeometryDeltaMatch::validate_requested_l2(target)?;
        check_distance_range(target, count)
    }

    pub(crate) fn new(target: f64, norm: f64, count: usize) -> Result<Self> {
        Self::admit_distance(target, count)?;
        if !norm.is_finite() || (target > 0. && norm == 0.) {
            return Err("a positive parameter distance needs a nonzero finite gradient".into());
        }
        Ok(Self { target, norm })
    }

    pub(crate) fn apply(&self, at: &mut f32, g: f64, motion: &mut RayMotion) -> Result<()> {
        if g == 0. || self.target == 0. {
            return Ok(());
        }
        let origin = f64::from(*at);
        let unit = -g / self.norm;
        let rounded = (origin + self.target * unit) as f32;
        if !rounded.is_finite() {
            return Err("parameter displacement overflowed f32".into());
        }
        let applied = f64::from(rounded) - origin;
        motion.actual.add(applied);
        motion.error.add(applied / self.target - unit);
        motion.dot += g * applied;
        *at = rounded;
        Ok(())
    }

    // Validate the whole vector, not each block. A small block may legitimately round to zero.
    pub(crate) fn admit_motion(&self, realized: f64, error: f64) -> Result<()> {
        if self.target == 0. {
            return Ok(());
        }
        check_realization(self.target, realized, error)
    }
}
