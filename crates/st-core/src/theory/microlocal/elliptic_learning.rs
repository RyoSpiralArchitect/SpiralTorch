//! Validated learning snapshots of the existing elliptic map, not a second map.

use super::{EllipticDifferential, EllipticTelemetry, EllipticWarp};

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum EllipticLearningError {
    #[error("invalid elliptic learning configuration")]
    Configuration,
    #[error("elliptic input must contain complete 3D rows within max_rows")]
    Shape,
    #[error("elliptic row {row} has no finite differential in this chart")]
    InvalidRow { row: usize },
    #[error("elliptic VJP needs nine finite upstream values per row")]
    InvalidUpstream,
    #[error("elliptic JVP needs three finite tangent values per row and a finite gate tangent")]
    InvalidTangent,
    #[error("elliptic JVP result is not finite")]
    NonFiniteTangent,
    #[error("elliptic VJP result is not finite")]
    NonFiniteGradient,
    #[error("elliptic learning output is not finite")]
    NonFiniteOutput,
    #[error("elliptic causal attention exceeds the score-pair budget")]
    PairBudget,
    #[error(transparent)]
    Attention(#[from] st_kernel_contracts::attention::AttentionError),
}

/// An immutable forward snapshot. Later warp reconfiguration cannot alter its derivatives.
#[derive(Clone, Debug)]
pub struct EllipticLearningBatch {
    features: Vec<f32>,
    differentials: Vec<EllipticDifferential>,
    telemetry: Vec<EllipticTelemetry>,
}

impl EllipticLearningBatch {
    pub fn features(&self) -> &[f32] {
        &self.features
    }

    pub fn telemetry(&self) -> &[EllipticTelemetry] {
        &self.telemetry
    }

    /// Apply the saved differential to a 3D input direction per row. This is
    /// first-order forward differentiation, not a derivative of the VJP.
    pub fn jvp(&self, tangent: &[f32]) -> Result<Vec<f32>, EllipticLearningError> {
        if tangent.len() != self.differentials.len() * 3 || tangent.iter().any(|v| !v.is_finite()) {
            return Err(EllipticLearningError::InvalidTangent);
        }
        let mut output = Vec::with_capacity(self.features.len());
        for (differential, direction) in self.differentials.iter().zip(tangent.as_chunks::<3>().0) {
            for row in differential.jacobian() {
                let value = row
                    .iter()
                    .zip(direction)
                    .map(|(&j, &v)| f64::from(j) * f64::from(v))
                    .sum::<f64>() as f32;
                if !value.is_finite() {
                    return Err(EllipticLearningError::NonFiniteTangent);
                }
                output.push(value);
            }
        }
        Ok(output)
    }

    /// CPU f32 outputs with an f64 accumulation of the nine feature contributions.
    pub fn vjp(&self, upstream: &[f32]) -> Result<Vec<f32>, EllipticLearningError> {
        if upstream.len() != self.features.len() || !upstream.iter().all(|v| v.is_finite()) {
            return Err(EllipticLearningError::InvalidUpstream);
        }
        let mut gradients = Vec::with_capacity(self.differentials.len() * 3);
        for (differential, seed) in self.differentials.iter().zip(upstream.as_chunks::<9>().0) {
            for coordinate in 0..3 {
                let value = seed
                    .iter()
                    .zip(differential.jacobian())
                    .map(|(&weight, row)| f64::from(weight) * f64::from(row[coordinate]))
                    .sum::<f64>() as f32;
                if !value.is_finite() {
                    return Err(EllipticLearningError::NonFiniteGradient);
                }
                gradients.push(value);
            }
        }
        Ok(gradients)
    }
}

impl EllipticWarp {
    pub fn for_learning(
        radius: f32,
        sheets: usize,
        harmonics: usize,
    ) -> Result<Self, EllipticLearningError> {
        if !radius.is_finite()
            || radius < 1e-6
            || !(radius * std::f32::consts::PI).is_finite()
            || sheets == 0
            || sheets > (u32::MAX >> 1) as usize
            || harmonics == 0
        {
            return Err(EllipticLearningError::Configuration);
        }
        Ok(Self::new(radius)
            .with_sheet_count(sheets)
            .with_spin_harmonics(harmonics))
    }

    pub fn differentiate_batch(
        &self,
        orientations: &[f32],
        max_rows: usize,
    ) -> Result<EllipticLearningBatch, EllipticLearningError> {
        Self::for_learning(
            self.curvature_radius(),
            self.sheet_count(),
            self.spin_harmonics(),
        )?;
        if max_rows == 0 {
            return Err(EllipticLearningError::Configuration);
        }
        if !orientations.len().is_multiple_of(3) || orientations.len() / 3 > max_rows {
            return Err(EllipticLearningError::Shape);
        }
        let rows = orientations.len() / 3;
        let values = rows.checked_mul(9).ok_or(EllipticLearningError::Shape)?;
        let mut batch = EllipticLearningBatch {
            features: Vec::with_capacity(values),
            differentials: Vec::with_capacity(rows),
            telemetry: Vec::with_capacity(rows),
        };
        for (row, orientation) in orientations.as_chunks::<3>().0.iter().enumerate() {
            // Poles and the azimuth cut do not have a single-valued chart VJP.
            // Forward-only legacy telemetry may still describe those points.
            if !orientation.iter().all(|v| v.is_finite())
                || (orientation[0] <= 0.0 && orientation[1] == 0.0)
            {
                return Err(EllipticLearningError::InvalidRow { row });
            }
            let (telemetry, differential) = self
                .map_orientation_with_differential(orientation)
                .ok_or(EllipticLearningError::InvalidRow { row })?;
            // Extreme component ratios can round a non-pole input onto a pole
            // in the f32 normalized chart. Do not return its fallback zero VJP.
            if telemetry.rotor_field[..2].iter().all(|&value| value == 0.0) {
                return Err(EllipticLearningError::InvalidRow { row });
            }
            if !differential
                .features
                .iter()
                .chain(differential.jacobian.iter().flatten())
                .all(|v| v.is_finite())
            {
                return Err(EllipticLearningError::InvalidRow { row });
            }
            batch.features.extend_from_slice(&differential.features);
            batch.differentials.push(differential);
            batch.telemetry.push(telemetry);
        }
        Ok(batch)
    }
}
