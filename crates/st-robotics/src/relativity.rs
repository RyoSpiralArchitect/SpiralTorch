use crate::error::RoboticsError;
use crate::geometry::{ZSpaceDynamics, ZSpaceGeometry};
use crate::gravity::GravityField;

const TOLERANCE: f32 = 1e-4;

/// Symmetry ansatz mirroring the general relativity helpers used in theory modules.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SymmetryAnsatz {
    /// Static, spherically symmetric configurations (Schwarzschild-like).
    StaticSpherical,
    /// Homogeneous and isotropic cosmological metrics (FRW-like).
    HomogeneousIsotropic,
    /// Custom ansatz described by free-form text.
    Custom(String),
}

impl SymmetryAnsatz {
    fn seed_matrix(&self) -> [[f32; 4]; 4] {
        match self {
            SymmetryAnsatz::StaticSpherical | SymmetryAnsatz::HomogeneousIsotropic => [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            SymmetryAnsatz::Custom(_) => [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }
}

/// Bridges symmetry ansätze and Lorentzian metrics into robotics-friendly dynamics.
#[derive(Debug, Clone, Copy)]
pub struct RelativityBridge;

impl RelativityBridge {
    fn validate_metric(components: [[f32; 4]; 4]) -> Result<[[f32; 4]; 4], RoboticsError> {
        for i in 0..4 {
            for j in 0..4 {
                let diff = (components[i][j] - components[j][i]).abs();
                if diff > TOLERANCE {
                    return Err(RoboticsError::RelativityBridge(format!(
                        "metric tensor must be symmetric at ({i},{j}); |Δ|={diff:.3e}"
                    )));
                }
            }
        }
        if components[0][0] >= -TOLERANCE {
            return Err(RoboticsError::RelativityBridge(
                "time-time component must be negative for Lorentzian signature".to_string(),
            ));
        }
        for spatial in 1..4 {
            if components[spatial][spatial] <= TOLERANCE {
                return Err(RoboticsError::RelativityBridge(format!(
                    "spatial diagonal entry ({spatial},{spatial}) must be positive"
                )));
            }
        }
        Ok(components)
    }

    fn geometry_from_validated(metric: [[f32; 4]; 4]) -> ZSpaceGeometry {
        let spatial: Vec<Vec<f32>> = (0..3)
            .map(|i| (0..3).map(|j| metric[i + 1][j + 1]).collect())
            .collect();
        let time_dilation = Self::time_dilation(&metric);
        ZSpaceGeometry::general_relativity(spatial, time_dilation)
    }

    fn time_dilation(metric: &[[f32; 4]; 4]) -> f32 {
        let g_tt = metric[0][0] as f64;
        let shift = ((metric[0][1] as f64).powi(2)
            + (metric[0][2] as f64).powi(2)
            + (metric[0][3] as f64).powi(2))
        .sqrt();
        let base = if g_tt < 0.0 { (-g_tt).sqrt() } else { 1.0 };
        (base / (1.0 + shift)).max(1e-6) as f32
    }

    /// Build a Z-space geometry directly from matrix components.
    pub fn geometry_from_components(
        components: [[f32; 4]; 4],
    ) -> Result<ZSpaceGeometry, RoboticsError> {
        let metric = Self::validate_metric(components)?;
        Ok(Self::geometry_from_validated(metric))
    }

    /// Couple geometry and gravity from matrix components.
    pub fn dynamics_from_components(
        components: [[f32; 4]; 4],
        gravity: Option<GravityField>,
    ) -> Result<ZSpaceDynamics, RoboticsError> {
        let metric = Self::validate_metric(components)?;
        let geometry = Self::geometry_from_validated(metric);
        Ok(ZSpaceDynamics::new(geometry, gravity))
    }

    /// Generate dynamics from a symmetry ansatz and uniform scale factor.
    pub fn dynamics_from_ansatz(
        ansatz: SymmetryAnsatz,
        scale: f64,
        gravity: Option<GravityField>,
    ) -> Result<ZSpaceDynamics, RoboticsError> {
        let mut matrix = ansatz.seed_matrix();
        for row in &mut matrix {
            for value in row {
                *value = (*value as f64 * scale) as f32;
            }
        }
        Self::dynamics_from_components(matrix, gravity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gravity::{GravityRegime, GravityWell};

    #[test]
    fn geometry_from_components_handles_minkowski() {
        let components = [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let geometry = RelativityBridge::geometry_from_components(components).unwrap();
        let norm = geometry.metric_norm(&[1.0, 0.0, 0.0]);
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn dynamics_from_ansatz_scales_geometry() {
        let dynamics =
            RelativityBridge::dynamics_from_ansatz(SymmetryAnsatz::StaticSpherical, 2.0, None)
                .unwrap();
        let norm = dynamics.geometry().metric_norm(&[1.0, 0.0, 0.0]);
        assert!(norm > 1.0);
    }

    #[test]
    fn dynamics_from_components_preserves_gravity() {
        let components = [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let mut gravity = GravityField::default();
        gravity.add_well("pose", GravityWell::new(10.0, GravityRegime::Newtonian));
        let dynamics =
            RelativityBridge::dynamics_from_components(components, Some(gravity.clone())).unwrap();
        assert!(dynamics.gravity().is_some());
        let geometry = dynamics.geometry();
        let norm = geometry.metric_norm(&[1.0, 0.0, 0.0]);
        assert!((norm - 1.0).abs() < 1e-6);
    }
}
