use super::*;
use crate::geometry_runtime::ParameterGradients;

pub(super) struct AnchorBank<'a> {
    pub(super) prompts: &'a [&'a [usize]],
    pub(super) refs: Vec<Reference>,
    pub(super) readouts: usize,
}

impl AnchorBank<'_> {
    // Cache references, not candidate models. In particular a raw miss must
    // not make the midpoint pass decode the same immutable origin again.
    pub(super) fn observe(
        &mut self,
        origin: &NativeCausalLm,
        candidate: &NativeCausalLm,
        mut visit: impl FnMut(usize, f64, Tensor),
    ) -> CausalLmResult<()> {
        if self.refs.is_empty() {
            for p in self.prompts {
                self.refs
                    .push(Reference::new(origin.next_token_logits(p)?.data())?);
                self.readouts += 1;
            }
        }
        for (i, (p, r)) in self.prompts.iter().zip(&self.refs).enumerate() {
            let row = candidate.next_token_logits(p)?;
            let kl = r.forward_kl(row.data())?;
            self.readouts += 1;
            visit(i, kl, row);
        }
        Ok(())
    }
}

pub(super) struct RaySearch<'a> {
    model: &'a NativeCausalLm,
    source: &'a ParameterGradients,
    mode: CausalLmAdaptationMode,
    policy: CausalLmAnchorKlPolicy,
    criterion: CausalLmAnchorKlCriterion,
    pub(super) bank: AnchorBank<'a>,
}

pub(super) enum SearchRay<'a> {
    Raw,
    Secant(&'a ParameterGradients, CausalLmSecantReport),
    Sensitivity(&'a ParameterGradients, Box<CausalLmSensitivityReport>),
}

impl<'a> RaySearch<'a> {
    pub(super) fn new(
        model: &'a NativeCausalLm,
        source: &'a ParameterGradients,
        mode: CausalLmAdaptationMode,
        prompts: &'a [&'a [usize]],
        policy: CausalLmAnchorKlPolicy,
        criterion: CausalLmAnchorKlCriterion,
    ) -> Self {
        Self {
            model,
            source,
            mode,
            policy,
            criterion,
            bank: AnchorBank {
                prompts,
                refs: Vec::with_capacity(prompts.len()),
                readouts: 0,
            },
        }
    }

    pub(super) fn run(
        &mut self,
        ray: SearchRay<'_>,
        raw_miss: Option<Box<CausalLmRawRayMiss>>,
    ) -> CausalLmResult<(
        NativeCausalLm,
        CausalLmAdaptationRayStep,
        CausalLmAnchorKlMatch,
    )> {
        let (policy, criterion) = (self.policy, self.criterion);
        let (gradient, mut guidance, mut sensitivity) = match ray {
            SearchRay::Raw => (self.source, None, None),
            SearchRay::Secant(g, report) => (g, Some(report), None),
            SearchRay::Sensitivity(g, report) => (g, None, Some(report)),
        };
        let (mut lo, mut hi) = (0., policy.max_parameter_l2);
        let mut distance = hi;
        let mut trials = Vec::new();
        let mut reason = "bounded bracket exhausted";
        for i in 0..CausalLmAnchorKlPolicy::MAX_TRIALS {
            let (candidate, mut ray) = self
                .model
                .step_parameter_ray(gradient, self.mode, distance)?;
            let mut sum = sum::Sum::default();
            let mut per_anchor_kl = Vec::with_capacity(self.bank.prompts.len());
            let mut peak = CausalLmAnchorKlPeak {
                anchor_index: 0,
                kl: 0.,
            };
            self.bank
                .observe(self.model, &candidate, |anchor_index, kl, _| {
                    sum.add(kl);
                    if kl > peak.kl {
                        peak = CausalLmAnchorKlPeak { anchor_index, kl };
                    }
                    per_anchor_kl.push(kl);
                })?;
            let mean_kl = sum.mean(self.bank.prompts.len() as u64);
            trials.push(CausalLmAnchorKlTrial {
                requested_l2: distance,
                mean_kl,
                peak: (!criterion.is_mean()).then_some(peak),
            });
            match criterion.position(policy.target_kl, mean_kl, peak.kl) {
                Position::Matched => {
                    if let Some(info) = guidance.as_mut() {
                        info.source_loss_dot_step = Some(self.model.source_loss_motion(
                            &candidate,
                            self.mode,
                            self.source,
                        )?);
                        ray.method = if info.midpoint.is_some() {
                            "selected_parameter_secant_midpoint_ray_f32.v1"
                        } else {
                            "selected_parameter_secant_projected_ray_f32.v1"
                        };
                    }
                    if let Some(info) = sensitivity.as_mut() {
                        info.source_loss_dot_step = Some(self.model.source_loss_motion(
                            &candidate,
                            self.mode,
                            self.source,
                        )?);
                        ray.method = "selected_parameter_sensitivity_ray_f32.v1";
                    }
                    return Ok((
                        candidate,
                        ray,
                        CausalLmAnchorKlMatch {
                            method: "last_token_anchor_forward_kl.v1",
                            direction: guidance,
                            sensitivity,
                            raw_miss,
                            policy,
                            criterion,
                            mean_kl,
                            per_anchor_kl,
                            anchor_readouts: self.bank.readouts,
                            trials,
                        },
                    ));
                }
                Position::Below if i == 0 => {
                    reason = "ceiling below matching interval";
                    break;
                }
                Position::Below => lo = distance,
                Position::Above => hi = distance,
            }
            distance = lo + 0.5 * (hi - lo);
            if distance <= lo || distance >= hi {
                break;
            }
        }
        Err(CausalLmAnchorKlRefusal {
            reason,
            direction: guidance,
            sensitivity,
            raw_miss,
            policy,
            criterion,
            anchor_readouts: self.bank.readouts,
            trials,
        }
        .into())
    }
}
