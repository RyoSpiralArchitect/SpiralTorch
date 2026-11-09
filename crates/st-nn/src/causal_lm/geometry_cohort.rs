//! Equal-vote reduction of sealed geometry objectives, independent of their target counts.
use super::sum::Sum;
use super::*;
use crate::geometry_artifact::{GeometryAdapterTensorRole, GeometryDeltaMatch};
use std::collections::BTreeMap;

pub const MAX_GEOMETRY_COHORT_MEMBERS: usize = 256;
pub const MAX_GEOMETRY_COHORT_ELEMENTS: usize = 16_777_216;

#[derive(Debug, Serialize)]
pub struct CausalLmGeometryCohortStep {
    pub base_model_identity: String,
    pub lora_artifact_id: Option<String>,
    pub pre_geometry_id: String,
    pub post_geometry_id: String,
    pub readout_accumulator: CausalLmReadoutAccumulator,
    pub members: usize,
    /// Descriptive only: the arithmetic mean has one vote per input handle.
    pub total_target_tokens: usize,
    pub mean_input_loss: f64,
    pub learning_rate: f32,
    pub trainable_values: usize,
    /// Norm of the mean gradient after the single narrowing to f32.
    pub mean_gradient_l2: f64,
    pub applied_l2: f64,
    /// First-order predictions on the actual displacement in input order, not measured loss changes.
    pub member_loss_dots: Vec<f64>,
    pub distance_match: Option<GeometryDeltaMatch>,
}

impl NativeCausalLm {
    pub fn validate_geometry_cohort(
        &self,
        members: usize,
        learning_rate: f32,
        target_l2: Option<f64>,
    ) -> CausalLmResult<()> {
        if !learning_rate.is_finite() || learning_rate <= 0. {
            return Err("learning rate must be finite and positive".into());
        }
        if let Some(target) = target_l2 {
            GeometryDeltaMatch::validate_requested_l2(target)?;
        }
        self.validate_geometry_cohort_size(members)
    }

    pub(super) fn validate_geometry_cohort_size(&self, members: usize) -> CausalLmResult<()> {
        self.validate_adaptation_mode(CausalLmAdaptationMode::Geometry)?;
        if !(1..=MAX_GEOMETRY_COHORT_MEMBERS).contains(&members) {
            return Err("geometry cohort requires 1 to 256 input gradients".into());
        }
        let width = self
            .geometry_artifact()
            .unwrap()
            .record()
            .placements
            .iter()
            .flat_map(|p| &p.tensors)
            .filter(|t| t.role == GeometryAdapterTensorRole::Parameter)
            .try_fold(0usize, |n, t| n.checked_add(t.values.len()))
            .ok_or("geometry cohort width overflow")?;
        product(
            &[members, width],
            MAX_GEOMETRY_COHORT_ELEMENTS,
            "geometry cohort gradient elements",
        )?;
        Ok(())
    }

    pub(super) fn validate_geometry_gradient_owner(
        &self,
        g: &NativeCausalLmAdaptationGradients,
    ) -> CausalLmResult<()> {
        if g.base_model_identity() != self.base_model_identity()
            || g.lora_artifact_id() != self.lora_artifact_id()
            || g.geometry_artifact_id() != self.geometry_artifact_id()
            || g.readout_accumulator() != self.readout_accumulator()
        {
            return Err(
                "geometry gradients belong to a different snapshot or readout policy".into(),
            );
        }
        Ok(())
    }

    pub(super) fn validate_full_geometry_gradient(
        &self,
        g: &NativeCausalLmAdaptationGradients,
    ) -> CausalLmResult<()> {
        self.validate_geometry_gradient_owner(g)?;
        let runtime = self
            .geometry
            .as_ref()
            .ok_or("missing geometry for cohort")?;
        let origin = runtime.artifact();
        if g.parameters().len() != origin.record().placements.len() || !g.loss().is_finite() {
            return Err(
                "geometry cohort requires complete geometry-only gradients and finite losses"
                    .into(),
            );
        }
        runtime.validate_gradients(origin.artifact_id(), g.parameters())
    }

    pub(super) fn geometry_cohort_proposal(
        &self,
        mean: &crate::geometry_runtime::ParameterGradients,
        rate: f32,
        distance: Option<f64>,
    ) -> CausalLmResult<(NativeGeometryRuntime, Option<GeometryDeltaMatch>)> {
        let runtime = self
            .geometry
            .as_ref()
            .ok_or("missing geometry for cohort")?;
        let origin = runtime.artifact();
        let proposal = runtime.with_gradients_sgd(origin.artifact_id(), mean, rate)?;
        if let Some(target) = distance {
            let (end, r) = origin.match_trainable_delta(proposal.artifact(), target)?;
            Ok((NativeGeometryRuntime::new(end), Some(r)))
        } else {
            Ok((proposal, None))
        }
    }

    /// One SGD proposal from the equal arithmetic mean, not a sequence of steps.
    /// Repeated handles count repeatedly. No tokens are decoded and the origin is immutable.
    pub fn step_geometry_cohort_sgd(
        &self,
        gradients: &[&NativeCausalLmAdaptationGradients],
        learning_rate: f32,
        target_l2: Option<f64>,
    ) -> CausalLmResult<(Self, CausalLmGeometryCohortStep)> {
        self.validate_geometry_cohort(gradients.len(), learning_rate, target_l2)?;
        let runtime = self.geometry.as_ref().unwrap();
        let origin = runtime.artifact();
        let placements = &origin.record().placements;
        let mut targets = 0usize;
        let mut losses = Sum::default();
        for &g in gradients {
            self.validate_full_geometry_gradient(g)?;
            targets = targets
                .checked_add(g.target_tokens())
                .ok_or("geometry cohort target count overflow")?;
            losses.add(f64::from(g.loss()));
        }
        let mut mean = BTreeMap::new();
        let mut blocks = Vec::new();
        let mut norm = Sum::default();
        let mut count = 0;
        for (i, p) in placements.iter().enumerate() {
            let mut params = BTreeMap::new();
            for (j, t) in p
                .tensors
                .iter()
                .enumerate()
                .filter(|(_, t)| t.role == GeometryAdapterTensorRole::Parameter)
            {
                let mut v = vec![0.; t.values.len()];
                if t.requires_grad {
                    let slices: Vec<_> = gradients
                        .iter()
                        .map(|g| g.parameters()[&p.path][&t.name].as_slice())
                        .collect();
                    for (k, dst) in v.iter_mut().enumerate() {
                        let mut sum = Sum::default();
                        for g in &slices {
                            sum.add(f64::from(g[k]));
                        }
                        *dst = sum.mean(gradients.len() as u64) as f32;
                        norm.add(f64::from(*dst).powi(2));
                    }
                    count += v.len();
                    blocks.push((i, j, slices));
                }
                params.insert(t.name.clone(), v);
            }
            mean.insert(p.path.clone(), params);
        }
        let (endpoint, distance_match) =
            self.geometry_cohort_proposal(&mean, learning_rate, target_l2)?;
        let mut dots: Vec<_> = (0..gradients.len()).map(|_| Sum::default()).collect();
        let mut l2 = Sum::default();
        for (i, j, slices) in blocks {
            let before = &placements[i].tensors[j].values;
            let after = &endpoint.artifact().record().placements[i].tensors[j].values;
            for (k, (&a, &b)) in before.iter().zip(after).enumerate() {
                let d = f64::from(b) - f64::from(a);
                l2.add(d * d);
                for (s, dot) in slices.iter().zip(&mut dots) {
                    dot.add(f64::from(s[k]) * d);
                }
            }
        }
        let report = CausalLmGeometryCohortStep {
            base_model_identity: self.base_id.clone(),
            lora_artifact_id: self.lora_artifact_id().map(str::to_owned),
            pre_geometry_id: origin.artifact_id().into(),
            post_geometry_id: endpoint.artifact().artifact_id().into(),
            readout_accumulator: self.readout_accumulator,
            members: gradients.len(),
            total_target_tokens: targets,
            mean_input_loss: losses.mean(gradients.len() as u64),
            learning_rate,
            trainable_values: count,
            mean_gradient_l2: norm.value().sqrt(),
            applied_l2: l2.value().sqrt(),
            member_loss_dots: dots.iter().map(Sum::value).collect(),
            distance_match,
        };
        let mut next = self.clone();
        next.geometry = Some(Arc::new(endpoint));
        Ok((next, report))
    }
}
