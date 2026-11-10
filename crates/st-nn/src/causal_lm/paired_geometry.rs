//! Pair two example means at the same snapshot. Only trainable geometry participates.
use super::sum::Sum;
use super::*;
use crate::geometry_artifact::{GeometryAdapterTensorRole, GeometryDeltaMatch};
use std::collections::BTreeMap;

type Gradients = BTreeMap<String, BTreeMap<String, Vec<f32>>>;
const RELATIVE_SLOPE_TOLERANCE: f64 = 1e-12;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CausalLmPairWeighting {
    Equal,
    CommonDescent,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct CausalLmPairDisplacement {
    pub l2: f64,
    /// First-order changes on the realized, rounded parameter displacement, not measured losses.
    pub loss_dots: [f64; 2],
    pub both_descend: bool,
}

#[derive(Debug, Serialize)]
pub struct CausalLmPairedGeometryStep {
    pub base_model_identity: String,
    pub lora_artifact_id: Option<String>,
    pub pre_geometry_id: String,
    pub post_geometry_id: String,
    pub readout_accumulator: CausalLmReadoutAccumulator,
    pub policy: CausalLmPairWeighting,
    pub learning_rate: f32,
    pub trainable_values: usize,
    pub gradient_l2: [f64; 2],
    pub weights: [f64; 2],
    pub reweighted: bool,
    pub equal: CausalLmPairDisplacement,
    pub applied: CausalLmPairDisplacement,
    pub distance_match: Option<GeometryDeltaMatch>,
}

struct TrainableBlock<'a> {
    index: (usize, usize),
    before: &'a [f32],
    gradients: [&'a [f32]; 2],
}

struct Pair<'a> {
    runtime: &'a NativeGeometryRuntime,
    maps: [&'a Gradients; 2],
    blocks: Vec<TrainableBlock<'a>>,
    count: usize,
    norms: [f64; 2],
}

impl<'a> Pair<'a> {
    fn new(runtime: &'a NativeGeometryRuntime, maps: [&'a Gradients; 2]) -> CausalLmResult<Self> {
        let artifact = runtime.artifact();
        let placements = &artifact.record().placements;
        for map in maps {
            if map.len() != placements.len() {
                return Err("paired update requires full geometry-only gradients".into());
            }
            runtime.validate_gradients(artifact.artifact_id(), map)?;
        }
        let mut blocks = Vec::new();
        let mut squares = [Sum::default(), Sum::default()];
        let mut count = 0;
        for (i, p) in placements.iter().enumerate() {
            for (j, t) in p.tensors.iter().enumerate() {
                if t.role != GeometryAdapterTensorRole::Parameter || !t.requires_grad {
                    continue;
                }
                let gradients = maps.map(|m| m[&p.path][&t.name].as_slice());
                for (side, g) in gradients.iter().enumerate() {
                    for &x in *g {
                        squares[side].add(f64::from(x).powi(2));
                    }
                }
                blocks.push(TrainableBlock {
                    index: (i, j),
                    before: &t.values,
                    gradients,
                });
                count += t.values.len();
            }
        }
        Ok(Self {
            runtime,
            maps,
            blocks,
            count,
            norms: squares.map(|s| s.value().sqrt()),
        })
    }

    fn displacement(&self, endpoint: &GeometryAdapterArtifact) -> CausalLmPairDisplacement {
        let mut squares = Sum::default();
        let mut dots = [Sum::default(), Sum::default()];
        for block in &self.blocks {
            let (i, j) = block.index;
            let end = &endpoint.record().placements[i].tensors[j].values;
            for (k, (&a, &b)) in block.before.iter().zip(end).enumerate() {
                let d = f64::from(b) - f64::from(a);
                squares.add(d * d);
                for (s, dot) in dots.iter_mut().enumerate() {
                    dot.add(f64::from(block.gradients[s][k]) * d);
                }
            }
        }
        let l2 = squares.value().sqrt();
        let loss_dots = dots.map(|s| s.value());
        let both_descend =
            (0..2).all(|s| loss_dots[s] < -RELATIVE_SLOPE_TOLERANCE * self.norms[s] * l2);
        CausalLmPairDisplacement {
            l2,
            loss_dots,
            both_descend,
        }
    }

    fn candidate(
        &self,
        weights: [f64; 2],
        rate: f32,
        target: Option<f64>,
    ) -> CausalLmResult<(
        NativeGeometryRuntime,
        CausalLmPairDisplacement,
        Option<GeometryDeltaMatch>,
    )> {
        let [left, right] = self.maps;
        let mut combined = BTreeMap::new();
        for (path, tensors) in left {
            let mut placement = BTreeMap::new();
            for (name, a) in tensors {
                let b = &right[path][name];
                let values: Vec<f32> = a
                    .iter()
                    .zip(b)
                    .map(|(&x, &y)| (weights[0] * f64::from(x) + weights[1] * f64::from(y)) as f32)
                    .collect();
                placement.insert(name.clone(), values);
            }
            combined.insert(path.clone(), placement);
        }
        let origin = self.runtime.artifact();
        let proposal = self
            .runtime
            .with_gradients_sgd(origin.artifact_id(), &combined, rate)?;
        let raw = self.displacement(proposal.artifact());
        if let Some(target) = target {
            // An invisible direction cannot be scaled. Let the policy inspect that no-op
            // before deciding whether a balanced direction is available.
            if raw.l2 != 0. || target == 0. {
                let (end, receipt) = origin.match_trainable_delta(proposal.artifact(), target)?;
                let actual = self.displacement(&end);
                return Ok((NativeGeometryRuntime::new(end), actual, Some(receipt)));
            }
        }
        Ok((proposal, raw, None))
    }
}

impl NativeCausalLm {
    /// Combine two sealed geometry-only example gradients. Equal is the ordinary pair mean.
    /// CommonDescent changes its weights only if the rounded mean fails either first-order test.
    /// No cursor or optimizer state advances and no post-update forward is implied.
    pub fn step_paired_geometry_sgd(
        &self,
        left: &NativeCausalLmAdaptationGradients,
        right: &NativeCausalLmAdaptationGradients,
        learning_rate: f32,
        policy: CausalLmPairWeighting,
        target_l2: Option<f64>,
    ) -> CausalLmResult<(Self, CausalLmPairedGeometryStep)> {
        self.validate_adaptation_mode(CausalLmAdaptationMode::Geometry)?;
        if !learning_rate.is_finite() || learning_rate <= 0. {
            return Err("learning rate must be finite and positive".into());
        }
        if let Some(target) = target_l2 {
            GeometryDeltaMatch::validate_requested_l2(target)?;
        }
        for g in [left, right] {
            self.validate_geometry_gradient_owner(g)?;
        }
        let runtime = self
            .geometry
            .as_ref()
            .ok_or("paired update requires attached geometry")?;
        let pair = Pair::new(runtime, [left.parameters(), right.parameters()])?;
        let mut weights = [0.5, 0.5];
        let (candidate, equal, distance) = pair.candidate(weights, learning_rate, target_l2)?;
        let reweighted = policy == CausalLmPairWeighting::CommonDescent && !equal.both_descend;
        let (candidate, applied, distance_match) = if reweighted {
            let [a, b] = pair.norms;
            if a == 0. || b == 0. || target_l2 == Some(0.) {
                return Err("no strict common descent for a zero gradient or zero distance".into());
            }
            // Both contributions now have equal norm; either dot strictly descends unless
            // the gradients are antiparallel. Check the actual f32 result, not that ideal.
            weights = [b / (a + b), a / (a + b)];
            let balanced = pair.candidate(weights, learning_rate, target_l2)?;
            if !balanced.1.both_descend {
                return Err("no representable strict common descent for this pair and step".into());
            }
            balanced
        } else {
            (candidate, equal, distance)
        };
        if target_l2.is_some() && distance_match.is_none() {
            return Err("cannot assign a positive distance to a zero geometry direction".into());
        }
        let report = CausalLmPairedGeometryStep {
            base_model_identity: self.base_id.clone(),
            lora_artifact_id: self.lora_artifact_id().map(str::to_owned),
            pre_geometry_id: runtime.artifact().artifact_id().to_owned(),
            post_geometry_id: candidate.artifact().artifact_id().to_owned(),
            readout_accumulator: self.readout_accumulator,
            policy,
            learning_rate,
            trainable_values: pair.count,
            gradient_l2: pair.norms,
            weights,
            reweighted,
            equal,
            applied,
            distance_match,
        };
        let mut next = self.clone();
        next.geometry = Some(Arc::new(candidate));
        Ok((next, report))
    }
}
