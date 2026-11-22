//! A single finite-checked geometry proposal, not a line search or a validation-set guarantee.
use super::*;

#[derive(Debug, Serialize)]
pub struct CausalLmDeclaredSetStep {
    pub target_l2: f64,
    pub before: DeclaredSetNll,
    /// Scores at the guarded proposal, including when it was not accepted.
    pub proposed: DeclaredSetNll,
    pub accepted: bool,
    /// Unlike the first-order guard's post_geometry_id, this is the returned model.
    pub result_geometry_id: String,
    pub proposal_guard: CausalLmGeometryStepGuard,
}

impl NativeCausalLm {
    /// Policy admission only. The gradient API additionally validates positions, IDs and labels.
    pub fn validate_declared_set_step(
        &self,
        plan: &[DeclaredRivalRow],
        distance: f64,
    ) -> CausalLmResult<()> {
        if !distance.is_finite() || distance <= 0. {
            return Err(
                "a finite-checked declared-set step needs a finite positive distance".into(),
            );
        }
        if !plan
            .iter()
            .any(|r| r.rivals.len() < self.config.vocabulary.saturating_sub(1))
        {
            return Err(
                "strict coverage improvement needs a declared proper subset of the vocabulary"
                    .into(),
            );
        }
        Ok(())
    }

    /// Propose a within-set gradient step, guard both components at the realized f32
    /// displacement, then evaluate the guarded model once on exactly the same supervision.
    /// Return the origin on a finite tie/regression. Invalid or infeasible directions remain
    /// errors: neither path tries a different distance or changes the immutable source.
    pub fn step_declared_set_geometry_at_distance(
        &self,
        windows: &[usize],
        shape: [usize; 2],
        labels: Option<&[i64]>,
        plan: &[DeclaredRivalRow],
        target_l2: f64,
    ) -> CausalLmResult<(Self, CausalLmDeclaredSetStep)> {
        self.validate_declared_set_step(plan, target_l2)?;
        let (guarded, proposal_guard, before) = {
            let (within, coverage, before) =
                self.geometry_declared_set_gradients(windows, shape, labels, plan)?;
            let (candidate, _) = self.step_geometry_gradient_at_distance(&within, target_l2)?;
            let (guarded, guard) =
                self.guard_geometry_step(&candidate, &[&[&within], &[&coverage]], 1e-5)?;
            (guarded, guard, before)
        };
        let proposed = guarded.declared_set_nll(windows, shape, labels, plan)?;
        let accepted =
            proposed.within_nll < before.within_nll && proposed.coverage_nll < before.coverage_nll;
        let result = if accepted { guarded } else { self.clone() };
        let report = CausalLmDeclaredSetStep {
            target_l2,
            before,
            proposed,
            accepted,
            result_geometry_id: result
                .geometry_artifact_id()
                .ok_or("missing result geometry")?
                .to_owned(),
            proposal_guard,
        };
        Ok((result, report))
    }
}
