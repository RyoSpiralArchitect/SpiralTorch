use super::{GeometryAdapterArtifact, GeometryAdapterTensorRole, Result};
use crate::geometry_artifact::GeometryAdapterRecipe;
use crate::WaveGateKernel;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RadiusDrift {
    max_delta: f32,
    sites: Vec<Anchor>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Anchor {
    placement: String,
    initial_log_radius: f32,
    projections: u32,
}

impl RadiusDrift {
    pub(super) fn new(artifact: &GeometryAdapterArtifact, max_delta: f32) -> Result<Self> {
        if !max_delta.is_finite() || max_delta <= 0.0 {
            return Err("max_log_radius_drift must be positive and finite".into());
        }
        let mut sites = Vec::new();
        for p in &artifact.record().placements {
            if matches!(
                p.recipe,
                GeometryAdapterRecipe::WaveRadius {
                    learnable_radius: true,
                    ..
                }
            ) {
                if let Some(t) = p.tensors.iter().find(|t| {
                    t.name == "log_radius"
                        && t.role == GeometryAdapterTensorRole::Parameter
                        && t.requires_grad
                }) {
                    sites.push(Anchor {
                        placement: p.path.clone(),
                        initial_log_radius: t.values[0],
                        projections: 0,
                    });
                }
            }
        }
        if sites.is_empty() {
            return Err("radius drift bound requires a trainable WaveGate log_radius".into());
        }
        sites.sort_by(|a, b| a.placement.cmp(&b.placement));
        Ok(Self { max_delta, sites })
    }

    pub(super) fn max_delta(&self) -> f32 {
        self.max_delta
    }

    pub(super) fn projections(&self) -> u64 {
        self.sites.iter().map(|s| u64::from(s.projections)).sum()
    }

    pub(super) fn validate(
        &self,
        artifact: &GeometryAdapterArtifact,
        keys: &[(String, String)],
        clocks: &[u32],
    ) -> Result<()> {
        let current = Self::new(artifact, self.max_delta)?;
        if current.sites.len() != self.sites.len() {
            return Err("radius drift anchor count differs from artifact".into());
        }
        let radius_clocks: std::collections::BTreeMap<_, _> = keys
            .iter()
            .zip(clocks)
            .filter(|((_, name), _)| name == "log_radius")
            .map(|((path, _), n)| (path.as_str(), *n))
            .collect();
        for (now, saved) in current.sites.iter().zip(&self.sites) {
            WaveGateKernel::validate_log_radius(saved.initial_log_radius)?;
            let Some(&clock) = radius_clocks.get(saved.placement.as_str()) else {
                return Err("radius drift anchor has no optimizer slot".into());
            };
            if saved.placement != now.placement
                || saved.projections > clock
                || (clock == 0
                    && saved.initial_log_radius.to_bits() != now.initial_log_radius.to_bits())
                || (f64::from(now.initial_log_radius) - f64::from(saved.initial_log_radius)).abs()
                    > f64::from(self.max_delta)
            {
                return Err(
                    "radius drift anchor, displacement or projection clock mismatch".into(),
                );
            }
        }
        Ok(())
    }

    pub(super) fn project(&mut self, path: &str, candidate: f32) -> Result<f32> {
        let i = self
            .sites
            .binary_search_by(|s| s.placement.as_str().cmp(path))
            .map_err(|_| "missing radius drift anchor")?;
        let site = &mut self.sites[i];
        let initial = f64::from(site.initial_log_radius);
        let delta = f64::from(candidate) - initial;
        let width = f64::from(self.max_delta);
        if delta.abs() <= width {
            return Ok(candidate);
        }
        // Project onto the representable points *inside* the log-space interval.
        let edge = initial + width.copysign(delta);
        let mut value = edge as f32;
        if delta > 0.0 && f64::from(value) > edge {
            value = value.next_down();
        }
        if delta < 0.0 && f64::from(value) < edge {
            value = value.next_up();
        }
        site.projections = site
            .projections
            .checked_add(1)
            .ok_or("radius projection count exhausted")?;
        Ok(value)
    }
}
