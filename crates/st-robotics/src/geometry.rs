use std::collections::HashMap;

/// Geometry choices for modelling Z-space trajectories.
#[derive(Debug, Clone)]
pub enum GeometryKind {
    /// Default Euclidean geometry.
    Euclidean,
    /// Constant-curvature geometry approximating non-Euclidean manifolds.
    NonEuclidean { curvature: f32 },
    /// General-relativistic geometry encoded via a spatial metric tensor.
    GeneralRelativity {
        metric: Vec<Vec<f32>>,
        time_dilation: f32,
    },
}

/// Metric helper for computing distances and norms inside Z-space.
#[derive(Debug, Clone)]
pub struct ZSpaceGeometry {
    kind: GeometryKind,
}

impl ZSpaceGeometry {
    pub fn euclidean() -> Self {
        Self {
            kind: GeometryKind::Euclidean,
        }
    }

    pub fn non_euclidean(curvature: f32) -> Self {
        Self {
            kind: GeometryKind::NonEuclidean { curvature },
        }
    }

    pub fn general_relativity(metric: Vec<Vec<f32>>, time_dilation: f32) -> Self {
        Self {
            kind: GeometryKind::GeneralRelativity {
                metric,
                time_dilation: time_dilation.max(1e-6),
            },
        }
    }

    pub fn kind(&self) -> &GeometryKind {
        &self.kind
    }

    /// Compute a geometry-aware norm of a state vector.
    pub fn metric_norm(&self, vector: &[f32]) -> f32 {
        if vector.is_empty() {
            return 0.0;
        }
        let euclidean = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
        match &self.kind {
            GeometryKind::Euclidean => euclidean,
            GeometryKind::NonEuclidean { curvature } => {
                let curvature = *curvature;
                if curvature.abs() < 1e-6 {
                    euclidean
                } else if curvature > 0.0 {
                    // Positive curvature expands perceived distance.
                    let adjustment = 1.0 + curvature * euclidean * euclidean / 6.0;
                    euclidean * adjustment
                } else {
                    // Negative curvature contracts distance growth.
                    let contraction = 1.0 + curvature.abs() * euclidean * euclidean / 6.0;
                    euclidean / contraction.max(1e-6)
                }
            }
            GeometryKind::GeneralRelativity {
                metric,
                time_dilation,
            } => {
                let mut accumulator = 0.0;
                for (i, value_i) in vector.iter().enumerate() {
                    let row = metric.get(i);
                    for (j, value_j) in vector.iter().enumerate() {
                        let coefficient = row
                            .and_then(|r| r.get(j))
                            .copied()
                            .unwrap_or(if i == j { 1.0 } else { 0.0 });
                        accumulator += value_i * coefficient * value_j;
                    }
                }
                let spatial = accumulator.abs().sqrt();
                spatial * time_dilation.max(1e-6)
            }
        }
    }
}

impl Default for ZSpaceGeometry {
    fn default() -> Self {
        Self::euclidean()
    }
}

/// Encodes the physics regime for a given gravity source.
#[derive(Debug, Clone)]
pub enum GravityRegime {
    Newtonian,
    Relativistic { speed_of_light: f32 },
}

impl GravityRegime {
    fn relativistic_factor(&self, radius: f32) -> f32 {
        match self {
            GravityRegime::Newtonian => 1.0,
            GravityRegime::Relativistic { speed_of_light } => {
                let c = (*speed_of_light).max(1e-6);
                1.0 / (1.0 + radius / c)
            }
        }
    }
}

/// Mass descriptor for a gravitational source in Z-space.
#[derive(Debug, Clone)]
pub struct GravityWell {
    pub mass: f32,
    pub regime: GravityRegime,
}

impl GravityWell {
    pub fn new(mass: f32, regime: GravityRegime) -> Self {
        Self { mass, regime }
    }
}

/// Aggregate gravitational field covering multiple channels.
#[derive(Debug, Clone)]
pub struct GravityField {
    constant: f32,
    wells: HashMap<String, GravityWell>,
}

impl GravityField {
    pub fn new(constant: f32) -> Self {
        Self {
            constant,
            wells: HashMap::new(),
        }
    }

    pub fn with_wells(constant: f32, wells: HashMap<String, GravityWell>) -> Self {
        Self { constant, wells }
    }

    pub fn constant(&self) -> f32 {
        self.constant
    }

    pub fn wells(&self) -> &HashMap<String, GravityWell> {
        &self.wells
    }

    pub fn wells_mut(&mut self) -> &mut HashMap<String, GravityWell> {
        &mut self.wells
    }

    pub fn add_well(&mut self, channel: impl Into<String>, well: GravityWell) {
        self.wells.insert(channel.into(), well);
    }

    pub fn potential(
        &self,
        channel: &str,
        geometry: &ZSpaceGeometry,
        vector: &[f32],
    ) -> Option<f32> {
        let well = self.wells.get(channel)?;
        let radius = geometry.metric_norm(vector);
        if radius <= 1e-6 {
            return Some(0.0);
        }
        let base = -self.constant * well.mass / radius;
        let factor = well.regime.relativistic_factor(radius);
        Some(base * factor)
    }
}

impl Default for GravityField {
    fn default() -> Self {
        const G: f32 = 6.67430e-11_f32;
        Self::new(G)
    }
}

/// Coupled geometry and gravity descriptors for Z-space dynamics.
#[derive(Debug, Clone)]
pub struct ZSpaceDynamics {
    geometry: ZSpaceGeometry,
    gravity: Option<GravityField>,
}

impl ZSpaceDynamics {
    pub fn new(geometry: ZSpaceGeometry, gravity: Option<GravityField>) -> Self {
        Self { geometry, gravity }
    }

    pub fn geometry(&self) -> &ZSpaceGeometry {
        &self.geometry
    }

    pub fn gravity(&self) -> Option<&GravityField> {
        self.gravity.as_ref()
    }
}

impl Default for ZSpaceDynamics {
    fn default() -> Self {
        Self {
            geometry: ZSpaceGeometry::default(),
            gravity: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn euclidean_norm_matches_expectation() {
        let geometry = ZSpaceGeometry::euclidean();
        let norm = geometry.metric_norm(&[3.0, 4.0]);
        assert!((norm - 5.0).abs() < 1e-6);
    }

    #[test]
    fn non_euclidean_expands_distance() {
        let geometry = ZSpaceGeometry::non_euclidean(0.5);
        let norm = geometry.metric_norm(&[1.0, 0.0]);
        assert!(norm > 1.0);
    }

    #[test]
    fn gravity_field_returns_potential() {
        let mut field = GravityField::default();
        field.add_well("pose", GravityWell::new(10.0, GravityRegime::Newtonian));
        let geometry = ZSpaceGeometry::euclidean();
        let potential = field
            .potential("pose", &geometry, &[2.0, 0.0, 0.0])
            .unwrap();
        assert!(potential.is_sign_negative());
    }
}
