//! Certify a candidate displacement, or repair it on one bounded segment without a hull search.
use super::*;
use crate::geometry_artifact::MAX_RELATIVE_DELTA_ERROR;

#[derive(Debug, Serialize)]
pub struct CausalLmGeometryStepGuard {
    pub base_model_identity: String,
    pub lora_artifact_id: Option<String>,
    pub readout_accumulator: CausalLmReadoutAccumulator,
    pub pre_geometry_id: String,
    pub candidate_geometry_id: String,
    pub post_geometry_id: String,
    pub group_sizes: Vec<usize>,
    pub group_target_tokens: Vec<usize>,
    pub group_mean_losses: Vec<f64>,
    pub group_gradient_l2: Vec<f64>,
    pub trainable_values: usize,
    pub min_cosine: f64,
    pub candidate_l2: f64,
    pub applied_l2: f64,
    pub relative_vector_error: f64,
    /// First minimum in input order, even on a pass-through.
    pub worst_candidate_group: usize,
    /// None preserves the original candidate exactly. Otherwise mix is the interval midpoint.
    pub feasible_mix: Option<[f64; 2]>,
    pub mix: f64,
    pub candidate_min_cosine: f64,
    pub applied_min_cosine: f64,
    pub candidate_loss_dots: Vec<f64>,
    pub applied_loss_dots: Vec<f64>,
}

impl NativeCausalLm {
    pub(super) fn geometry_along_descent(
        &self,
        length: f64,
        direction: &[f64],
    ) -> CausalLmResult<Self> {
        let mut record = self.geometry_artifact().unwrap().record().clone();
        let params = record
            .placements
            .iter_mut()
            .flat_map(|p| &mut p.tensors)
            .filter(|t| t.role == GeometryAdapterTensorRole::Parameter && t.requires_grad)
            .flat_map(|t| &mut t.values);
        for (x, &d) in params.zip(direction) {
            if d != 0. {
                *x = (f64::from(*x) - length * d) as f32;
                if !x.is_finite() {
                    return Err("geometry guard displacement overflowed f32".into());
                }
            }
        }
        self.with_geometry(GeometryAdapterArtifact::create(record)?)
    }

    pub fn validate_geometry_step_guard(
        &self,
        candidate: &Self,
        sizes: &[usize],
        min_cosine: f64,
    ) -> CausalLmResult<()> {
        if !(min_cosine > 0. && min_cosine < 1.) {
            return Err(
                "geometry guard needs a finite minimum cosine strictly between zero and one".into(),
            );
        }
        if sizes.is_empty() || sizes.len() > MAX_GEOMETRY_COHORT_MEMBERS || sizes.contains(&0) {
            return Err("geometry guard needs 1 to 256 nonempty groups".into());
        }
        let members = sizes
            .iter()
            .try_fold(0usize, |n, &s| n.checked_add(s))
            .ok_or("geometry guard member count overflow")?;
        self.validate_geometry_cohort_size(members)?;
        if self.base_id != candidate.base_id
            || self.lora_artifact_id() != candidate.lora_artifact_id()
            || self.readout_accumulator != candidate.readout_accumulator
            || candidate.geometry.is_none()
            || self.geometry_artifact_id() == candidate.geometry_artifact_id()
        {
            return Err("geometry guard needs a distinct candidate geometry on the same base, LoRA and readout".into());
        }
        Ok(())
    }

    /// Keep the candidate length. If necessary turn its direction toward the worst group's
    /// negative unit gradient. Verify every group's dot on the actual f32 displacement.
    /// No new forward/backward, line search, automatic retry or full-space optimality claim.
    pub fn guard_geometry_step(
        &self,
        candidate: &Self,
        groups: &[&[&NativeCausalLmAdaptationGradients]],
        min_cosine: f64,
    ) -> CausalLmResult<(Self, CausalLmGeometryStepGuard)> {
        if groups.len() > MAX_GEOMETRY_COHORT_MEMBERS {
            return Err("too many geometry guard groups".into());
        }
        let sizes: Vec<_> = groups.iter().map(|g| g.len()).collect();
        self.validate_geometry_step_guard(candidate, &sizes, min_cosine)?;
        let origin = self.geometry_artifact().unwrap();
        let proposed = candidate.geometry_artifact().unwrap();
        // Checks base/layout/training compatibility and every frozen parameter/buffer bit.
        let length = origin.trainable_delta_l2(proposed)?;
        if !(length > 0. && length.is_finite()) {
            return Err("geometry guard needs a nonzero finite candidate displacement".into());
        }
        let GroupMeans {
            vectors,
            targets,
            losses,
            width,
        } = self.geometry_group_means(groups)?;
        let norms: Vec<_> = vectors.iter().map(|g| hull::norm(g)).collect();
        if norms.iter().any(|n| !n.is_finite() || *n == 0.) {
            return Err("geometry guard requires nonzero finite group means".into());
        }
        let delta = displacement(origin, proposed);
        let dots = |d: &[f64]| vectors.iter().map(|g| hull::dot(g, d)).collect::<Vec<_>>();
        let slopes = |ds: &[f64], l: f64| {
            ds.iter()
                .zip(&norms)
                .map(|(&d, &n)| -d / (n * l))
                .collect::<Vec<_>>()
        };
        let input_dots = dots(&delta);
        let input = slopes(&input_dots, length);
        let worst = (1..groups.len()).fold(0, |w, i| if input[i] < input[w] { i } else { w });
        let mut direction: Vec<_> = delta.iter().map(|d| -d / length).collect();
        let mut interval = None;
        let mut mix = 0.;
        let mut next = self.clone();
        if input[worst] > min_cosine {
            next.geometry.clone_from(&candidate.geometry);
        } else {
            let toward: Vec<_> = vectors[worst].iter().map(|v| v / norms[worst]).collect();
            let (mut lo, mut hi) = (0f64, 1f64);
            // Each dot on the unnormalized segment is affine in mix. A positive lower
            // bound survives normalization, since a mixture of unit vectors has norm <= 1.
            for (i, g) in vectors.iter().enumerate() {
                let slope = hull::dot(g, &toward) / norms[i] - input[i];
                if slope > 0. {
                    lo = lo.max((min_cosine - input[i]) / slope);
                } else if slope < 0. {
                    hi = hi.min((min_cosine - input[i]) / slope);
                } else if input[i] < min_cosine {
                    return Err("no feasible interval on this geometry guard segment".into());
                }
            }
            if lo >= hi {
                return Err("no open feasible interval on this geometry guard segment".into());
            }
            interval = Some([lo, hi]);
            mix = (lo + hi) * 0.5;
            for (d, u) in direction.iter_mut().zip(toward) {
                *d = (1. - mix) * *d + mix * u;
            }
            let n = hull::norm(&direction);
            if !(n > 0. && n.is_finite()) {
                return Err("unresolved geometry guard direction".into());
            }
            direction.iter_mut().for_each(|d| *d /= n);
            next = self.geometry_along_descent(length, &direction)?;
        }
        let end = next.geometry_artifact().unwrap();
        let applied = displacement(origin, end);
        let (applied_l2, error) = checked_direction(&applied, length, &direction)?;
        let output_dots = dots(&applied);
        let min = slopes(&output_dots, applied_l2)
            .into_iter()
            .fold(f64::INFINITY, f64::min);
        if !min.is_finite() || min <= min_cosine {
            return Err("rounded geometry guard step misses a group's required cosine".into());
        }
        let report = CausalLmGeometryStepGuard {
            base_model_identity: self.base_id.clone(),
            lora_artifact_id: self.lora_artifact_id().map(str::to_owned),
            readout_accumulator: self.readout_accumulator,
            pre_geometry_id: origin.artifact_id().into(),
            candidate_geometry_id: proposed.artifact_id().into(),
            post_geometry_id: end.artifact_id().into(),
            group_sizes: sizes,
            group_target_tokens: targets,
            group_mean_losses: losses,
            group_gradient_l2: norms,
            trainable_values: width,
            min_cosine,
            candidate_l2: length,
            applied_l2,
            relative_vector_error: error,
            worst_candidate_group: worst,
            feasible_mix: interval,
            mix,
            candidate_min_cosine: input[worst],
            applied_min_cosine: min,
            candidate_loss_dots: input_dots,
            applied_loss_dots: output_dots,
        };
        Ok((next, report))
    }
}

pub(super) fn checked_direction(
    applied: &[f64],
    length: f64,
    descent: &[f64],
) -> CausalLmResult<(f64, f64)> {
    let applied_l2 = hull::norm(applied);
    let error = hull::sum(
        applied
            .iter()
            .zip(descent)
            .map(|(&v, &d)| (v / length + d).powi(2)),
    )
    .sqrt();
    if !error.is_finite()
        || error > MAX_RELATIVE_DELTA_ERROR
        || (applied_l2 / length - 1.).abs() > MAX_RELATIVE_DELTA_ERROR
    {
        return Err("geometry guard direction or length is not representable in f32".into());
    }
    Ok((applied_l2, error))
}
