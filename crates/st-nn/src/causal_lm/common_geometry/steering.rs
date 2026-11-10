//! Give one declared objective authority over a step with a known safe anchor.
use super::*;

const MIN_PRIMARY_COSINE_GAIN: f64 = 1e-12;

#[derive(Debug, Serialize)]
pub struct CausalLmGeometrySteeringStep {
    pub base_model_identity: String,
    pub lora_artifact_id: Option<String>,
    pub readout_accumulator: CausalLmReadoutAccumulator,
    pub pre_geometry_id: String,
    pub anchor_geometry_id: String,
    pub post_geometry_id: String,
    /// In every per-group vector the primary is first, then the protected groups in input order.
    pub group_sizes: Vec<usize>,
    pub group_target_tokens: Vec<usize>,
    pub group_mean_losses: Vec<f64>,
    pub group_gradient_l2: Vec<f64>,
    pub anchor_loss_dots: Vec<f64>,
    pub applied_loss_dots: Vec<f64>,
    pub trainable_values: usize,
    pub min_cosine: f64,
    pub anchor_l2: f64,
    pub applied_l2: f64,
    pub relative_vector_error: f64,
    /// Index into the protected groups only. None indicates a strictly feasible direct direction.
    pub limiting_protected_group: Option<usize>,
    pub mix_limit: f64,
    pub mix: f64,
    pub anchor_min_protected_cosine: f64,
    pub direct_min_protected_cosine: f64,
    pub applied_min_protected_cosine: f64,
    pub anchor_primary_cosine: f64,
    pub applied_primary_cosine: f64,
}

impl NativeCausalLm {
    /// Sizes are [primary, protected_0, ...]. All handles share a single admission budget.
    pub fn validate_geometry_steering(
        &self,
        anchor: &Self,
        sizes: &[usize],
        floor: f64,
    ) -> CausalLmResult<()> {
        if sizes.len() < 2 {
            return Err(
                "geometry steering needs a primary and at least one protected group".into(),
            );
        }
        self.validate_geometry_step_guard(anchor, sizes, floor)
    }

    /// Turn a strictly feasible anchor toward the primary mean at the same trainable-coordinate
    /// length. Certify both protected slopes and an actual primary improvement over the anchor
    /// after rounding. No forward, line search, hidden fallback or finite-loss guarantee.
    pub fn steer_geometry_step(
        &self,
        anchor: &Self,
        primary: &[&NativeCausalLmAdaptationGradients],
        protected: &[&[&NativeCausalLmAdaptationGradients]],
        min_cosine: f64,
    ) -> CausalLmResult<(Self, CausalLmGeometrySteeringStep)> {
        if protected.len() >= MAX_GEOMETRY_COHORT_MEMBERS {
            return Err("too many protected geometry steering groups".into());
        }
        let mut groups = vec![primary];
        groups.extend_from_slice(protected);
        let sizes: Vec<_> = groups.iter().map(|g| g.len()).collect();
        self.validate_geometry_steering(anchor, &sizes, min_cosine)?;
        let origin = self.geometry_artifact().unwrap();
        let safe = anchor.geometry_artifact().unwrap();
        // Validate layout, training mode and bit identity of every frozen value.
        let length = origin.trainable_delta_l2(safe)?;
        if !(length > 0. && length.is_finite()) {
            return Err("geometry steering requires a nonzero finite anchor displacement".into());
        }
        let GroupMeans {
            vectors,
            targets,
            losses,
            width,
        } = self.geometry_group_means(&groups)?;
        let norms: Vec<_> = vectors.iter().map(|g| hull::norm(g)).collect();
        if norms.iter().any(|n| !n.is_finite() || *n == 0.) {
            return Err("geometry steering requires nonzero finite group means".into());
        }
        let dots = |d: &[f64]| vectors.iter().map(|g| hull::dot(g, d)).collect::<Vec<_>>();
        let delta = displacement(origin, safe);
        let anchor_dots = dots(&delta);
        let anchor_cos: Vec<_> = anchor_dots
            .iter()
            .zip(&norms)
            .map(|(x, n)| -x / (length * n))
            .collect();
        let anchor_min = anchor_cos[1..]
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        if anchor_min <= min_cosine {
            return Err("geometry steering anchor misses a protected cosine floor".into());
        }
        let desired: Vec<_> = vectors[0].iter().map(|x| x / norms[0]).collect();
        let mut direct_min = f64::INFINITY;
        let mut limit = 1.;
        let mut limiting = None;
        for i in 1..vectors.len() {
            let b = hull::dot(&vectors[i], &desired) / norms[i];
            direct_min = direct_min.min(b);
            if b <= min_cosine {
                // Dot on the unnormalized segment is affine. Its norm is <= 1,
                // so normalization preserves any positive strict lower bound.
                let boundary = (anchor_cos[i] - min_cosine) / (anchor_cos[i] - b);
                if limiting.is_none() || boundary < limit {
                    limiting = Some(i - 1);
                    limit = boundary;
                }
            }
        }
        let mix = if limiting.is_some() { 0.5 * limit } else { 1. };
        let mut direction = desired;
        if limiting.is_some() {
            for (d, &x) in direction.iter_mut().zip(&delta) {
                *d = (1. - mix) * (-x / length) + mix * *d;
            }
            let n = hull::norm(&direction);
            if !(n > 0. && n.is_finite()) {
                return Err("unresolved primary geometry steering direction".into());
            }
            direction.iter_mut().for_each(|x| *x /= n);
        }
        let next = self.geometry_along_descent(length, &direction)?;
        let end = next.geometry_artifact().unwrap();
        let applied = displacement(origin, end);
        let (applied_l2, error) = guard::checked_direction(&applied, length, &direction)?;
        let applied_dots = dots(&applied);
        let applied_cos: Vec<_> = applied_dots
            .iter()
            .zip(&norms)
            .map(|(x, n)| -x / (applied_l2 * n))
            .collect();
        let applied_min = applied_cos[1..]
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        if !applied_min.is_finite() || applied_min <= min_cosine {
            return Err("rounded geometry steering step misses a protected cosine floor".into());
        }
        if !(applied_cos[0] > anchor_cos[0] + MIN_PRIMARY_COSINE_GAIN
            && applied_dots[0] < 0.
            && applied_dots[0] < anchor_dots[0])
        {
            return Err("rounded geometry steering step does not improve primary descent".into());
        }
        let r = CausalLmGeometrySteeringStep {
            base_model_identity: self.base_id.clone(),
            lora_artifact_id: self.lora_artifact_id().map(str::to_owned),
            readout_accumulator: self.readout_accumulator,
            pre_geometry_id: origin.artifact_id().into(),
            anchor_geometry_id: safe.artifact_id().into(),
            post_geometry_id: end.artifact_id().into(),
            group_sizes: sizes,
            group_target_tokens: targets,
            group_mean_losses: losses,
            group_gradient_l2: norms,
            anchor_loss_dots: anchor_dots,
            applied_loss_dots: applied_dots,
            trainable_values: width,
            min_cosine,
            anchor_l2: length,
            applied_l2,
            relative_vector_error: error,
            limiting_protected_group: limiting,
            mix_limit: limit,
            mix,
            anchor_min_protected_cosine: anchor_min,
            direct_min_protected_cosine: direct_min,
            applied_min_protected_cosine: applied_min,
            anchor_primary_cosine: anchor_cos[0],
            applied_primary_cosine: applied_cos[0],
        };
        Ok((next, r))
    }
}
