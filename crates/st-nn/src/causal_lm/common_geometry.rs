//! Strict first-order common descent across several sealed, equal-example geometry objectives.
use super::*;
use crate::geometry_artifact::{GeometryAdapterTensorRole, GeometryDeltaMatch};
use std::collections::BTreeMap;
mod guard;
pub(super) mod hull;
mod steering;
pub use guard::CausalLmGeometryStepGuard;
pub use steering::CausalLmGeometrySteeringStep;

pub const MAX_COMMON_GEOMETRY_GROUPS: usize = 8;

struct GroupMeans {
    vectors: Vec<Vec<f64>>,
    targets: Vec<usize>,
    losses: Vec<f64>,
    width: usize,
}

// Both endpoints have already passed layout and frozen-value checks, or were
// produced by an update on the origin's validated runtime.
fn displacement(origin: &GeometryAdapterArtifact, end: &GeometryAdapterArtifact) -> Vec<f64> {
    let mut delta = Vec::new();
    for (a, b) in origin
        .record()
        .placements
        .iter()
        .zip(&end.record().placements)
    {
        for (t, u) in a.tensors.iter().zip(&b.tensors) {
            if t.role == GeometryAdapterTensorRole::Parameter && t.requires_grad {
                delta.extend(
                    t.values
                        .iter()
                        .zip(&u.values)
                        .map(|(&x, &y)| f64::from(y) - f64::from(x)),
                );
            }
        }
    }
    delta
}

#[derive(Debug, Serialize)]
pub struct CausalLmCommonGeometryStep {
    pub base_model_identity: String,
    pub lora_artifact_id: Option<String>,
    pub pre_geometry_id: String,
    pub post_geometry_id: String,
    pub readout_accumulator: CausalLmReadoutAccumulator,
    pub learning_rate: f32,
    pub group_sizes: Vec<usize>,
    pub group_target_tokens: Vec<usize>,
    pub group_mean_losses: Vec<f64>,
    pub trainable_values: usize,
    pub group_gradient_l2: Vec<f64>,
    /// Convex weights on *unit* group means, not on raw gradients.
    pub unit_group_weights: Vec<f64>,
    /// Equivalent convex mixture on raw group means used for the SGD proposal.
    pub raw_group_weights: Vec<f64>,
    pub hull_norm: f64,
    pub predicted_min_unit_margin: f64,
    pub applied_l2: f64,
    pub applied_min_unit_margin: f64,
    pub group_loss_dots: Vec<f64>,
    pub distance_match: Option<GeometryDeltaMatch>,
}

impl NativeCausalLm {
    pub fn validate_common_geometry(
        &self,
        sizes: &[usize],
        rate: f32,
        target: Option<f64>,
    ) -> CausalLmResult<()> {
        if !(2..=MAX_COMMON_GEOMETRY_GROUPS).contains(&sizes.len()) || sizes.contains(&0) {
            return Err("expected 2 to 8 nonempty geometry groups".into());
        }
        let n = sizes
            .iter()
            .try_fold(0usize, |s, &v| s.checked_add(v))
            .ok_or("common geometry member count overflow")?;
        self.validate_geometry_cohort(n, rate, target)?;
        if target == Some(0.) {
            return Err("strict common descent needs a positive displacement".into());
        }
        Ok(())
    }

    fn geometry_group_means(
        &self,
        groups: &[&[&NativeCausalLmAdaptationGradients]],
    ) -> CausalLmResult<GroupMeans> {
        let origin = self.geometry_artifact().unwrap();
        let placements = &origin.record().placements;
        let mut targets = Vec::with_capacity(groups.len());
        let mut losses = Vec::with_capacity(groups.len());
        for &group in groups {
            let mut n = 0usize;
            let mut loss = sum::Sum::default();
            for &g in group {
                self.validate_full_geometry_gradient(g)?;
                n = n
                    .checked_add(g.target_tokens())
                    .ok_or("common geometry target count overflow")?;
                loss.add(f64::from(g.loss()));
            }
            targets.push(n);
            losses.push(loss.mean(group.len() as u64));
        }
        let width = placements
            .iter()
            .flat_map(|p| &p.tensors)
            .filter(|t| t.role == GeometryAdapterTensorRole::Parameter && t.requires_grad)
            .map(|t| t.values.len())
            .sum::<usize>();
        let mut means = Vec::with_capacity(groups.len());
        for &group in groups {
            let mut v = Vec::with_capacity(width);
            for p in placements {
                for t in &p.tensors {
                    if t.role != GeometryAdapterTensorRole::Parameter || !t.requires_grad {
                        continue;
                    }
                    let slices: Vec<_> = group
                        .iter()
                        .map(|g| g.parameters()[&p.path][&t.name].as_slice())
                        .collect();
                    for k in 0..t.values.len() {
                        v.push(
                            hull::sum(slices.iter().map(|s| f64::from(s[k]))) / group.len() as f64,
                        );
                    }
                }
            }
            means.push(v);
        }
        Ok(GroupMeans {
            vectors: means,
            targets,
            losses,
            width,
        })
    }

    /// Find a direction maximizing the worst normalized first-order decrease.
    /// All members must belong to this snapshot. Each group mean weights its handles equally.
    /// A refusal is not a proof that the nonlinear objectives cannot improve together.
    pub fn step_common_geometry_sgd(
        &self,
        groups: &[&[&NativeCausalLmAdaptationGradients]],
        rate: f32,
        target_l2: Option<f64>,
    ) -> CausalLmResult<(Self, CausalLmCommonGeometryStep)> {
        if groups.len() > MAX_COMMON_GEOMETRY_GROUPS {
            return Err("too many geometry groups".into());
        }
        let sizes: Vec<_> = groups.iter().map(|g| g.len()).collect();
        self.validate_common_geometry(&sizes, rate, target_l2)?;
        let GroupMeans {
            vectors: means,
            targets,
            losses,
            width,
        } = self.geometry_group_means(groups)?;
        let origin = self.geometry_artifact().unwrap();
        let placements = &origin.record().placements;
        let chosen = hull::solve(&means)?;
        let mut gradient = BTreeMap::new();
        let mut pos = 0;
        for p in placements {
            let mut params = BTreeMap::new();
            for t in p
                .tensors
                .iter()
                .filter(|t| t.role == GeometryAdapterTensorRole::Parameter)
            {
                let values = if t.requires_grad {
                    let start = pos;
                    pos += t.values.len();
                    chosen.vector[start..pos]
                        .iter()
                        .map(|&x| x as f32)
                        .collect()
                } else {
                    vec![0.; t.values.len()]
                };
                params.insert(t.name.clone(), values);
            }
            gradient.insert(p.path.clone(), params);
        }
        let (endpoint, distance_match) =
            self.geometry_cohort_proposal(&gradient, rate, target_l2)?;
        let delta = displacement(origin, endpoint.artifact());
        let l2 = hull::norm(&delta);
        if l2 == 0. {
            return Err("common geometry step is invisible at model precision".into());
        }
        let dots: Vec<_> = means.iter().map(|m| hull::dot(m, &delta)).collect();
        let margin = dots
            .iter()
            .zip(&chosen.norms)
            .map(|(&d, &n)| -d / (l2 * n))
            .fold(f64::INFINITY, f64::min);
        if !margin.is_finite() || margin <= 1e-12 {
            return Err(
                "rounded geometry proposal does not strictly descend for every group".into(),
            );
        }
        let r = CausalLmCommonGeometryStep {
            base_model_identity: self.base_id.clone(),
            lora_artifact_id: self.lora_artifact_id().map(str::to_owned),
            pre_geometry_id: origin.artifact_id().into(),
            post_geometry_id: endpoint.artifact().artifact_id().into(),
            readout_accumulator: self.readout_accumulator,
            learning_rate: rate,
            group_sizes: sizes,
            group_target_tokens: targets,
            group_mean_losses: losses,
            trainable_values: width,
            group_gradient_l2: chosen.norms,
            unit_group_weights: chosen.alpha,
            raw_group_weights: chosen.beta,
            hull_norm: chosen.hull_norm,
            predicted_min_unit_margin: chosen.margin,
            applied_l2: l2,
            applied_min_unit_margin: margin,
            group_loss_dots: dots,
            distance_match,
        };
        let mut next = self.clone();
        next.geometry = Some(Arc::new(endpoint));
        Ok((next, r))
    }
}
