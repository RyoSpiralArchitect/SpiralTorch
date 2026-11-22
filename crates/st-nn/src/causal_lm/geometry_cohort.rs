//! Equal-vote reduction of sealed geometry objectives, independent of their target counts.
use super::sum::Sum;
use super::*;
use crate::geometry_artifact::{
    GeometryAdapterTensorRole, GeometryDeltaMatch, GeometryGradientStep,
};
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

#[derive(Debug, Serialize)]
pub struct CausalLmGeometryCohortRayStep {
    pub base_model_identity: String,
    pub lora_artifact_id: Option<String>,
    pub readout_accumulator: CausalLmReadoutAccumulator,
    pub members: usize,
    pub total_target_tokens: usize,
    pub mean_input_loss: f64,
    pub step: GeometryGradientStep,
    /// Each input gradient dotted with the stored displacement. Individual signs may differ.
    pub member_loss_dots: Vec<f64>,
}

struct TrainableBlock<'a> {
    placement: usize,
    tensor: usize,
    gradients: Vec<&'a [f32]>,
}

struct CohortMean<'a, T> {
    parameters: BTreeMap<String, BTreeMap<String, Vec<T>>>,
    blocks: Vec<TrainableBlock<'a>>,
    members: usize,
    targets: usize,
    loss: f64,
}

impl<T> CohortMean<'_, T> {
    fn realized(
        &self,
        origin: &GeometryAdapterArtifact,
        end: &GeometryAdapterArtifact,
    ) -> (f64, Vec<f64>) {
        let mut dots: Vec<_> = (0..self.members).map(|_| Sum::default()).collect();
        let mut l2 = Sum::default();
        for block in &self.blocks {
            let before = &origin.record().placements[block.placement].tensors[block.tensor].values;
            let after = &end.record().placements[block.placement].tensors[block.tensor].values;
            for (k, (&a, &b)) in before.iter().zip(after).enumerate() {
                let d = f64::from(b) - f64::from(a);
                l2.add(d * d);
                for (g, dot) in block.gradients.iter().zip(&mut dots) {
                    dot.add(f64::from(g[k]) * d);
                }
            }
        }
        (l2.value().sqrt(), dots.iter().map(Sum::value).collect())
    }
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

    pub(super) fn validate_full_geometry_gradient(
        &self,
        g: &NativeCausalLmAdaptationGradients,
    ) -> CausalLmResult<()> {
        self.validate_adaptation_gradient_owner(g)?;
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

    fn reduce_geometry_cohort<'a, T: Default + Clone>(
        &self,
        gradients: &[&'a NativeCausalLmAdaptationGradients],
        mut store: impl FnMut(f64) -> T,
    ) -> CausalLmResult<CohortMean<'a, T>> {
        let placements = &self.geometry_artifact().unwrap().record().placements;
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
        for (i, p) in placements.iter().enumerate() {
            let mut params = BTreeMap::new();
            for (j, t) in p
                .tensors
                .iter()
                .enumerate()
                .filter(|(_, t)| t.role == GeometryAdapterTensorRole::Parameter)
            {
                let mut v = vec![T::default(); t.values.len()];
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
                        *dst = store(sum.mean(gradients.len() as u64));
                    }
                    blocks.push(TrainableBlock {
                        placement: i,
                        tensor: j,
                        gradients: slices,
                    });
                }
                params.insert(t.name.clone(), v);
            }
            mean.insert(p.path.clone(), params);
        }
        Ok(CohortMean {
            parameters: mean,
            blocks,
            members: gradients.len(),
            targets,
            loss: losses.mean(gradients.len() as u64),
        })
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
        let mut norm = Sum::default();
        let mut count = 0;
        let mean = self.reduce_geometry_cohort(gradients, |x| {
            let stored = x as f32;
            norm.add(f64::from(stored).powi(2));
            count += 1;
            stored
        })?;
        let origin = self.geometry_artifact().unwrap();
        let (endpoint, distance_match) =
            self.geometry_cohort_proposal(&mean.parameters, learning_rate, target_l2)?;
        let (applied_l2, member_loss_dots) = mean.realized(origin, endpoint.artifact());
        let report = CausalLmGeometryCohortStep {
            base_model_identity: self.base_id.clone(),
            lora_artifact_id: self.lora_artifact_id().map(str::to_owned),
            pre_geometry_id: origin.artifact_id().into(),
            post_geometry_id: endpoint.artifact().artifact_id().into(),
            readout_accumulator: self.readout_accumulator,
            members: gradients.len(),
            total_target_tokens: mean.targets,
            mean_input_loss: mean.loss,
            learning_rate,
            trainable_values: count,
            mean_gradient_l2: norm.value().sqrt(),
            applied_l2,
            member_loss_dots,
            distance_match,
        };
        let mut next = self.clone();
        next.geometry = Some(Arc::new(endpoint));
        Ok((next, report))
    }

    pub fn validate_geometry_cohort_ray(
        &self,
        members: usize,
        target_l2: f64,
    ) -> CausalLmResult<()> {
        GeometryDeltaMatch::validate_requested_l2(target_l2)?;
        self.validate_geometry_cohort_size(members)
    }

    /// Take one fixed-length step along the negative equal-vote mean, not a rounded SGD proposal.
    /// A tiny nonzero mean is normalized without a floor; a zero mean has no positive-length ray.
    /// This is average descent, not a guarantee that every member benefits.
    pub fn step_geometry_cohort_gradient_at_distance(
        &self,
        gradients: &[&NativeCausalLmAdaptationGradients],
        target_l2: f64,
    ) -> CausalLmResult<(Self, CausalLmGeometryCohortRayStep)> {
        self.validate_geometry_cohort_ray(gradients.len(), target_l2)?;
        let mean = self.reduce_geometry_cohort(gradients, std::convert::identity)?;
        let origin = self.geometry_artifact().unwrap();
        let (end, step) = origin.match_validated_cohort_ray(&mean.parameters, target_l2)?;
        let (_, member_loss_dots) = mean.realized(origin, &end);
        let next = if step.result_id == step.origin_id {
            self.clone()
        } else {
            self.with_geometry(end)?
        };
        let report = CausalLmGeometryCohortRayStep {
            base_model_identity: self.base_id.clone(),
            lora_artifact_id: self.lora_artifact_id().map(str::to_owned),
            readout_accumulator: self.readout_accumulator,
            members: mean.members,
            total_target_tokens: mean.targets,
            mean_input_loss: mean.loss,
            step,
            member_loss_dots,
        };
        Ok((next, report))
    }
}
