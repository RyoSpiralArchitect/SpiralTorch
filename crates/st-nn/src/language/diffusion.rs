// SPDX-License-Identifier: AGPL-3.0-or-later
// (c) 2025 Ryo SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch - Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL Section 13.

use super::info_geometry::{concept_diffusion_error_to_tensor, InformationGeometryMetric};
use super::maxwell::NarrativeHint;
use crate::PureResult;
use nalgebra::{DMatrix, DVector};
use st_core::inference::concept_diffusion::{
    apply_concept_diffusion, blend_concept_diffusion_observation,
    validate_concept_diffusion_conductivity, ConceptDiffusionConfig, ConceptDiffusionObservation,
    ConceptDiffusionPayload, ConceptDiffusionRequest,
};
use st_tensor::{emit_tensor_op, emit_tensor_op_meta, TensorError};
use std::collections::HashMap;

/// Stateful st-nn adapter over the canonical st-core graph heat flow.
#[derive(Clone, Debug)]
pub struct ConceptDiffusion {
    metric: InformationGeometryMetric,
    diffusion_tensor: Vec<Vec<f64>>,
    z_bias: Vec<f64>,
    state: Vec<f64>,
    config: ConceptDiffusionConfig,
}

impl ConceptDiffusion {
    pub fn new(metric: InformationGeometryMetric) -> PureResult<Self> {
        let dimension = metric.dimension();
        if dimension == 0 {
            return Err(TensorError::EmptyInput("concept diffusion atlas"));
        }
        Ok(Self {
            metric,
            diffusion_tensor: vec![vec![1.0; dimension]; dimension],
            z_bias: vec![0.0; dimension],
            state: vec![1.0 / dimension as f64; dimension],
            config: ConceptDiffusionConfig::default(),
        })
    }

    /// Applies a symmetric non-negative edge-conductivity tensor.
    pub fn with_diffusion_tensor(mut self, tensor: DMatrix<f64>) -> PureResult<Self> {
        let rows = (0..tensor.nrows())
            .map(|row| (0..tensor.ncols()).map(|col| tensor[(row, col)]).collect())
            .collect::<Vec<Vec<f64>>>();
        validate_concept_diffusion_conductivity(
            &rows,
            self.metric.dimension(),
            self.config.symmetry_tolerance(),
        )
        .map_err(concept_diffusion_error_to_tensor)?;
        self.diffusion_tensor = rows;
        Ok(self)
    }

    pub fn with_timestep(mut self, timestep: f64) -> PureResult<Self> {
        self.config = self
            .config
            .with_timestep(timestep)
            .map_err(concept_diffusion_error_to_tensor)?;
        Ok(self)
    }

    pub fn with_cfl_limit(mut self, cfl_limit: f64) -> PureResult<Self> {
        self.config = self
            .config
            .with_cfl_limit(cfl_limit)
            .map_err(concept_diffusion_error_to_tensor)?;
        Ok(self)
    }

    pub fn with_max_substeps(mut self, max_substeps: usize) -> PureResult<Self> {
        self.config = self
            .config
            .with_max_substeps(max_substeps)
            .map_err(concept_diffusion_error_to_tensor)?;
        Ok(self)
    }

    /// Replaces the labelled Z-bias and rejects unknown or non-finite tags.
    pub fn set_z_bias_map(&mut self, bias: HashMap<String, f64>) -> PureResult<()> {
        let mut next = vec![0.0; self.metric.dimension()];
        for (tag, value) in bias {
            if !value.is_finite() {
                return Err(TensorError::NonFiniteValue {
                    label: "concept diffusion Z-bias",
                    value: value as f32,
                });
            }
            let index = self.metric.index_of(&tag).ok_or_else(|| {
                TensorError::Generic(format!("unknown concept diffusion tag '{tag}'"))
            })?;
            next[index] = value;
        }
        self.z_bias = next;
        Ok(())
    }

    /// Atomically blends a narrative observation through the Rust core.
    pub fn observe(&mut self, hint: &NarrativeHint, weight: f64) -> PureResult<()> {
        let encoded = self.metric.encode(hint)?;
        let next = blend_concept_diffusion_observation(
            &self.state,
            &ConceptDiffusionObservation {
                probabilities: encoded.as_slice().to_vec(),
                weight,
            },
        )
        .map_err(concept_diffusion_error_to_tensor)?;
        self.state = next;
        Ok(())
    }

    /// Runs one canonical transition and returns the complete audit payload.
    pub fn step_detailed(&mut self) -> PureResult<ConceptDiffusionPayload> {
        let payload = apply_concept_diffusion(ConceptDiffusionRequest {
            tags: self.metric.tag_labels().to_vec(),
            state: self.state.clone(),
            affinity: self.metric.affinity_rows(),
            diffusion_tensor: Some(self.diffusion_tensor.clone()),
            z_bias: self.z_bias.clone(),
            observation: None,
            config: self.config,
        })
        .map_err(concept_diffusion_error_to_tensor)?;
        self.state = payload.next_state.clone();
        emit_diffusion_meta(&payload);
        Ok(payload)
    }

    pub fn step(&mut self) -> PureResult<DiffusionStep> {
        let payload = self.step_detailed()?;
        Ok(DiffusionStep {
            state: DVector::from_vec(payload.next_state),
            tags: payload.tags,
        })
    }

    pub fn state_map(&self) -> HashMap<String, f64> {
        self.metric
            .tag_labels()
            .iter()
            .cloned()
            .zip(self.state.iter().copied())
            .collect()
    }

    pub fn state(&self) -> &[f64] {
        &self.state
    }
}

fn emit_diffusion_meta(payload: &ConceptDiffusionPayload) {
    let values = payload.next_state.len();
    emit_tensor_op("concept_diffusion_step", &[values], &[values]);
    emit_tensor_op_meta("concept_diffusion_step", || {
        serde_json::json!({
            "backend": payload.execution_backend,
            "requested_backend": "auto",
            "kind": "language_concept_diffusion_step",
            "semantic_owner": payload.semantic_owner,
            "contract_version": payload.contract_version,
            "values": values,
            "edge_count": payload.effects.edge_count,
            "substeps": payload.effects.substeps,
            "substep_timestep": payload.effects.substep_timestep,
            "max_degree": payload.effects.max_degree,
            "z_bias_applied": payload.effects.z_bias_applied,
            "entropy_before": payload.effects.entropy_before,
            "entropy_after": payload.effects.entropy_after_diffusion,
            "dirichlet_energy_before": payload.effects.dirichlet_energy_before,
            "dirichlet_energy_after": payload.effects.dirichlet_energy_after,
            "input_probability_sum": payload.input_probability_sum,
            "output_probability_sum": payload.output_probability_sum,
            "state_dtype": "f64",
            "route_blocker": "f64_graph_state",
            "empty": values == 0,
        })
    });
}

#[derive(Clone, Debug, PartialEq)]
pub struct DiffusionStep {
    pub state: DVector<f64>,
    pub tags: Vec<String>,
}

impl DiffusionStep {
    pub fn as_pairs(&self) -> Vec<(String, f64)> {
        self.tags
            .iter()
            .cloned()
            .zip(self.state.iter().copied())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    fn observer_lock() -> std::sync::MutexGuard<'static, ()> {
        crate::test_global_state_lock()
    }

    fn hint(channel: &str, tags: &[&str], intensity: f32) -> NarrativeHint {
        NarrativeHint::new(
            channel,
            tags.iter().map(|tag| (*tag).to_owned()).collect(),
            intensity,
        )
    }

    fn diffusion() -> ConceptDiffusion {
        let hints = vec![
            hint("alpha", &["spiral", "torch"], 1.0),
            hint("beta", &["spiral", "narrative"], 0.8),
        ];
        let metric = InformationGeometryMetric::from_narratives(&hints).unwrap();
        let mut diffusion = ConceptDiffusion::new(metric).unwrap();
        diffusion.observe(&hints[0], 0.8).unwrap();
        diffusion
    }

    #[test]
    fn diffusion_updates_state_with_a_true_heat_flow() {
        let mut diffusion = diffusion();
        diffusion
            .set_z_bias_map(HashMap::from([("spiral".to_owned(), 0.2)]))
            .unwrap();
        let payload = diffusion.step_detailed().unwrap();

        assert_eq!(payload.tags.len(), payload.next_state.len());
        assert!((payload.next_state.iter().sum::<f64>() - 1.0).abs() < 1.0e-12);
        assert!(
            payload.effects.entropy_after_diffusion + 1.0e-10 >= payload.effects.entropy_after_bias
        );
        assert!(
            payload.effects.dirichlet_energy_after
                <= payload.effects.dirichlet_energy_before + 1.0e-10
        );
        assert_eq!(diffusion.state(), payload.next_state);
    }

    #[test]
    fn invalid_configuration_and_bias_fail_before_state_mutation() {
        let mut diffusion = diffusion();
        let before = diffusion.state().to_vec();

        assert!(diffusion.clone().with_timestep(f64::NAN).is_err());
        assert!(diffusion
            .set_z_bias_map(HashMap::from([("missing".to_owned(), 1.0)]))
            .is_err());
        assert_eq!(diffusion.state(), before);

        let asymmetric = DMatrix::from_row_slice(2, 2, &[1.0, 0.5, 0.2, 1.0]);
        let two_tag_metric = InformationGeometryMetric::from_narratives(&[
            hint("left", &["left"], 1.0),
            hint("right", &["right"], 1.0),
        ])
        .unwrap();
        assert!(ConceptDiffusion::new(two_tag_metric)
            .unwrap()
            .with_diffusion_tensor(asymmetric)
            .is_err());

        let mut budgeted = diffusion
            .with_timestep(10.0)
            .unwrap()
            .with_max_substeps(1)
            .unwrap();
        let before_step = budgeted.state().to_vec();
        assert!(budgeted.step_detailed().is_err());
        assert_eq!(budgeted.state(), before_step);
    }

    #[test]
    fn diffusion_step_emits_core_owned_backend_meta() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_thread_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let mut diffusion = diffusion();
        let payload = diffusion.step_detailed().unwrap();
        st_tensor::set_thread_meta_observer(previous);

        let events = events.lock().unwrap();
        let meta = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "concept_diffusion_step"
                    && data["kind"] == "language_concept_diffusion_step"
            })
            .expect("concept diffusion metadata event");
        assert_eq!(meta.1["backend"], "f64_cpu");
        assert_eq!(meta.1["semantic_owner"], payload.semantic_owner);
        assert_eq!(meta.1["contract_version"], payload.contract_version);
        assert_eq!(meta.1["values"], payload.next_state.len());
        assert_eq!(meta.1["substeps"], payload.effects.substeps);
        assert_eq!(meta.1["state_dtype"], "f64");
        assert_eq!(meta.1["route_blocker"], "f64_graph_state");
    }
}
