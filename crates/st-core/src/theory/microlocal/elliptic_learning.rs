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
    #[error("elliptic chart step needs nonempty finite two-row proposals (at most 131072 values)")]
    InvalidProposal,
    #[error("elliptic chart metric needs nonempty rows and positive finite trace")]
    InvalidMetric,
    #[error("elliptic chart step cannot be represented as finite nonzero f32 values")]
    InvalidStep,
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

/// Explicit optimizer proposal, not a replacement for the map's derivatives.
#[derive(Clone, Debug)]
pub struct EllipticChartStep {
    pub values: Vec<f32>,
    /// Mean J_chart^T J_chart in row-major order, before relative damping.
    pub metric: [f64; 4],
    pub damped_condition: f64,
    pub proposal_l2: f64,
    pub step_l2: f64,
    pub cosine: Option<f64>,
}

impl EllipticLearningBatch {
    pub fn features(&self) -> &[f32] {
        &self.features
    }

    pub fn telemetry(&self) -> &[EllipticTelemetry] {
        &self.telemetry
    }

    /// Precondition a row-major [2, C] proposal using the mean feature pullback
    /// of coordinates 1 and 2 in the fixed-first-coordinate chart. Preserve the
    /// proposal's global L2 norm, up to f32 rounding. This omits input covariance
    /// and the downstream readout/loss; it is not a full parameter-space metric.
    pub fn chart_step(
        &self,
        proposal: &[f32],
        relative_damping: f32,
    ) -> Result<EllipticChartStep, EllipticLearningError> {
        if !(1e-6..=1.0).contains(&relative_damping) {
            return Err(EllipticLearningError::Configuration);
        }
        if proposal.is_empty()
            || proposal.len() > 131_072
            || !proposal.len().is_multiple_of(2)
            || proposal.iter().any(|v| !v.is_finite())
        {
            return Err(EllipticLearningError::InvalidProposal);
        }
        if self.differentials.is_empty() {
            return Err(EllipticLearningError::InvalidMetric);
        }
        let (mut a, mut b, mut c) = (0.0, 0.0, 0.0);
        for differential in &self.differentials {
            for row in differential.jacobian() {
                let (x, y) = (f64::from(row[1]), f64::from(row[2]));
                a += x * x;
                b += x * y;
                c += y * y;
            }
        }
        let count = self.differentials.len() as f64;
        let metric = [a / count, b / count, b / count, c / count];
        let scale = (a + c) * 0.5;
        if !scale.is_finite() || scale <= 0.0 {
            return Err(EllipticLearningError::InvalidMetric);
        }
        let damping = f64::from(relative_damping);
        let (a, b, c) = (a / scale + damping, b / scale, c / scale + damping);
        let largest = ((a + c) + (a - c).hypot(2.0 * b)) * 0.5;
        let determinant = a * c - b * b;
        if !determinant.is_finite() || determinant <= 0.0 {
            return Err(EllipticLearningError::InvalidMetric);
        }
        let norm = |values: &[f32]| {
            values
                .iter()
                .map(|&v| f64::from(v).powi(2))
                .sum::<f64>()
                .sqrt()
        };
        let proposal_l2 = norm(proposal);
        let mut values = proposal.to_vec();
        if proposal_l2 > 0.0 {
            let columns = proposal.len() / 2;
            let mut raw = vec![0.0; proposal.len()];
            for i in 0..columns {
                let (x, y) = (f64::from(proposal[i]), f64::from(proposal[columns + i]));
                // The positive inverse determinant cancels during normalization.
                raw[i] = c * x - b * y;
                raw[columns + i] = a * y - b * x;
            }
            let raw_l2 = raw.iter().map(|v| v * v).sum::<f64>().sqrt();
            if !raw_l2.is_finite() || raw_l2 == 0.0 {
                return Err(EllipticLearningError::InvalidStep);
            }
            for (out, raw) in values.iter_mut().zip(raw) {
                *out = (raw * (proposal_l2 / raw_l2)) as f32;
            }
        }
        let step_l2 = norm(&values);
        if values.iter().any(|v| !v.is_finite()) || (proposal_l2 > 0.0 && step_l2 == 0.0) {
            return Err(EllipticLearningError::InvalidStep);
        }
        let cosine = (proposal_l2 > 0.0).then(|| {
            (values
                .iter()
                .zip(proposal)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum::<f64>()
                / (proposal_l2 * step_l2))
                .clamp(-1.0, 1.0)
        });
        Ok(EllipticChartStep {
            values,
            metric,
            damped_condition: largest * largest / determinant,
            proposal_l2,
            step_l2,
            cosine,
        })
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
