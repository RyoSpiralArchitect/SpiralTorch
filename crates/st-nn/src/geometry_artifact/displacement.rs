//! Euclidean displacement in trainable parameter coordinates, not an intrinsic or output-space distance.
use super::*;

pub const MAX_RELATIVE_DELTA_ERROR: f64 = 1e-4;

#[derive(Clone, Debug, Serialize)]
pub struct GeometryDeltaMatch {
    pub origin_id: String,
    pub direction_id: String,
    pub result_id: String,
    pub trainable_values: usize,
    pub direction_l2: f64,
    pub requested_l2: f64,
    pub realized_l2: f64,
    pub relative_vector_error: f64,
}

impl GeometryDeltaMatch {
    pub(crate) fn validate_requested_l2(requested: f64) -> Result<()> {
        if !requested.is_finite() || requested < 0.0 {
            return Err("requested geometry distance must be finite and nonnegative".into());
        }
        Ok(())
    }
}

#[derive(Default)]
struct Squares {
    sum: f64,
    correction: f64,
}

impl Squares {
    fn add(&mut self, delta: f64) {
        let v = delta * delta;
        let next = self.sum + v;
        self.correction += if self.sum >= v {
            (self.sum - next) + v
        } else {
            (v - next) + self.sum
        };
        self.sum = next;
    }
    fn norm(&self) -> f64 {
        (self.sum + self.correction).sqrt()
    }
}

fn trainable(t: &GeometryAdapterTensor) -> bool {
    t.role == GeometryAdapterTensorRole::Parameter && t.requires_grad
}

impl GeometryAdapterArtifact {
    fn checked_delta(&self, endpoint: &Self) -> Result<(usize, f64)> {
        if self.record.training != endpoint.record.training
            || self.layout_id()? != endpoint.layout_id()?
        {
            return Err(
                "geometry displacement requires the same base, layout and training flags".into(),
            );
        }
        let mut n = 0;
        let mut squares = Squares::default();
        for (a, b) in self
            .record
            .placements
            .iter()
            .zip(&endpoint.record.placements)
        {
            if a.training != b.training {
                return Err("geometry placement training flag changed".into());
            }
            for (t, u) in a.tensors.iter().zip(&b.tensors) {
                if trainable(t) {
                    n += t.values.len();
                    for (&x, &y) in t.values.iter().zip(&u.values) {
                        squares.add(f64::from(y) - f64::from(x));
                    }
                } else if t
                    .values
                    .iter()
                    .zip(&u.values)
                    .any(|(x, y)| x.to_bits() != y.to_bits())
                {
                    return Err(
                        "geometry displacement cannot change frozen parameters or buffers".into(),
                    );
                }
            }
        }
        if n == 0 {
            return Err("geometry displacement requires trainable parameters".into());
        }
        Ok((n, squares.norm()))
    }

    /// Distance across all trainable tensors together; frozen values must match bit for bit.
    pub fn trainable_delta_l2(&self, endpoint: &Self) -> Result<f64> {
        Ok(self.checked_delta(endpoint)?.1)
    }

    /// Keep the direction from self to endpoint and replace its length in trainable f32 coordinates.
    /// Zero length and unchanged length preserve exact original artifact bytes. Extrapolation is
    /// allowed only if the result is a valid artifact and f32 can represent the requested vector.
    pub fn match_trainable_delta(
        &self,
        endpoint: &Self,
        requested_l2: f64,
    ) -> Result<(Self, GeometryDeltaMatch)> {
        GeometryDeltaMatch::validate_requested_l2(requested_l2)?;
        let (n, direction_l2) = self.checked_delta(endpoint)?;
        let (result, realized_l2, error) = if requested_l2 == 0.0 {
            (self.clone(), 0.0, 0.0)
        } else if direction_l2 == 0.0 {
            return Err("cannot assign a positive distance to a zero geometry direction".into());
        } else if requested_l2 == direction_l2 {
            (endpoint.clone(), direction_l2, 0.0)
        } else {
            // Reject impossible extrapolations before squaring residuals or casting to f32.
            if requested_l2 > 2.0 * f64::from(f32::MAX) * (n as f64).sqrt() {
                return Err("requested geometry displacement exceeds f32 range".into());
            }
            let scale = requested_l2 / direction_l2;
            let mut record = self.record.clone();
            let (mut actual, mut error) = (Squares::default(), Squares::default());
            for (a, b) in record
                .placements
                .iter_mut()
                .zip(&endpoint.record.placements)
            {
                for (t, u) in a.tensors.iter_mut().zip(&b.tensors) {
                    if !trainable(t) {
                        continue;
                    }
                    for (x, &y) in t.values.iter_mut().zip(&u.values) {
                        let origin = f64::from(*x);
                        let d = f64::from(y) - origin;
                        if d == 0.0 {
                            continue;
                        }
                        let ideal = d * scale;
                        let value = (origin + ideal) as f32;
                        if !value.is_finite() {
                            return Err("geometry displacement overflowed f32".into());
                        }
                        *x = value;
                        let applied = f64::from(value) - origin;
                        actual.add(applied);
                        // Normalize before squaring, including when the requested length is subnormal.
                        error.add(applied / requested_l2 - d / direction_l2);
                    }
                }
            }
            let realized = actual.norm();
            let relative = error.norm();
            if !relative.is_finite()
                || relative > MAX_RELATIVE_DELTA_ERROR
                || (realized / requested_l2 - 1.0).abs() > MAX_RELATIVE_DELTA_ERROR
            {
                return Err("requested geometry direction or distance cannot be represented accurately in f32".into());
            }
            (Self::create(record)?, realized, relative)
        };
        let report = GeometryDeltaMatch {
            origin_id: self.artifact_id().to_owned(),
            direction_id: endpoint.artifact_id().to_owned(),
            result_id: result.artifact_id().to_owned(),
            trainable_values: n,
            direction_l2,
            requested_l2,
            realized_l2,
            relative_vector_error: error,
        };
        Ok((result, report))
    }
}
