//! Thin bounded transport to the shared Rust adaptation/session contract.
use super::{
    checkpoint_error, dimension, documents::documents, error, limit, reject_bool, PyNativeCausalLm,
};
use pyo3::prelude::*;
use st_nn::causal_lm::{
    CausalLmAdaptationMode, CausalLmSessionWindowPolicy, CausalLmTrainingSessionConfig,
    CausalLmTrainingSessionStep, NativeCausalLmAdaptationGradients, NativeCausalLmTrainingSession,
};
use st_nn::language_windows::MAX_DOCUMENTS;
use std::collections::BTreeMap;

pub(super) fn mode(value: &str) -> PyResult<CausalLmAdaptationMode> {
    match value {
        "lora" => Ok(CausalLmAdaptationMode::Lora),
        "geometry" => Ok(CausalLmAdaptationMode::Geometry),
        "joint" => Ok(CausalLmAdaptationMode::Joint),
        _ => Err(error("adaptation mode must be lora, geometry or joint")),
    }
}
fn mode_name(value: CausalLmAdaptationMode) -> &'static str {
    match value {
        CausalLmAdaptationMode::Lora => "lora",
        CausalLmAdaptationMode::Geometry => "geometry",
        CausalLmAdaptationMode::Joint => "joint",
    }
}

#[pyclass(
    name = "NativeCausalLmAdaptationGradients",
    module = "spiraltorch",
    frozen
)]
pub struct PyNativeCausalLmAdaptationGradients {
    pub(super) inner: NativeCausalLmAdaptationGradients,
}
#[pymethods]
impl PyNativeCausalLmAdaptationGradients {
    #[getter]
    fn loss(&self) -> f32 {
        self.inner.loss()
    }
    #[getter]
    fn target_tokens(&self) -> usize {
        self.inner.target_tokens()
    }
    #[getter]
    fn window_shape(&self) -> Vec<usize> {
        self.inner.window_shape().to_vec()
    }
    #[getter]
    fn base_model_identity(&self) -> &str {
        self.inner.base_model_identity()
    }
    #[getter]
    fn lora_artifact_id(&self) -> Option<&str> {
        self.inner.lora_artifact_id()
    }
    #[getter]
    fn geometry_artifact_id(&self) -> Option<&str> {
        self.inner.geometry_artifact_id()
    }
    #[getter]
    fn readout_accumulator(&self) -> &'static str {
        self.inner.readout_accumulator().as_str()
    }
    fn parameter_gradients(&self) -> BTreeMap<String, BTreeMap<String, Vec<f32>>> {
        self.inner.parameters().clone()
    }
}

#[pyclass(
    name = "NativeCausalLmTrainingSessionStep",
    module = "spiraltorch",
    frozen
)]
pub struct PyNativeCausalLmTrainingSessionStep {
    inner: CausalLmTrainingSessionStep,
}
#[pymethods]
impl PyNativeCausalLmTrainingSessionStep {
    #[getter]
    fn update(&self) -> u64 {
        self.inner.update
    }
    #[getter]
    fn epoch(&self) -> u32 {
        self.inner.epoch
    }
    #[getter]
    fn window_start(&self) -> usize {
        self.inner.window_start
    }
    #[getter]
    fn window_end(&self) -> usize {
        self.inner.window_end
    }
    #[getter]
    fn target_tokens(&self) -> usize {
        self.inner.target_tokens
    }
    #[getter]
    fn loss(&self) -> f32 {
        self.inner.loss
    }
    #[getter]
    fn mode(&self) -> &'static str {
        mode_name(self.inner.mode)
    }
    #[getter]
    fn pre_lora_artifact_id(&self) -> Option<&str> {
        self.inner.pre_lora_artifact_id.as_deref()
    }
    #[getter]
    fn post_lora_artifact_id(&self) -> Option<&str> {
        self.inner.post_lora_artifact_id.as_deref()
    }
    #[getter]
    fn pre_geometry_artifact_id(&self) -> Option<&str> {
        self.inner.pre_geometry_artifact_id.as_deref()
    }
    #[getter]
    fn post_geometry_artifact_id(&self) -> Option<&str> {
        self.inner.post_geometry_artifact_id.as_deref()
    }
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(error)
    }
}

#[pyclass(name = "NativeCausalLmTrainingSession", module = "spiraltorch")]
pub struct PyNativeCausalLmTrainingSession {
    inner: NativeCausalLmTrainingSession,
}
#[pymethods]
impl PyNativeCausalLmTrainingSession {
    #[classattr]
    const MAX_CHECKPOINT_BYTES: usize = NativeCausalLmTrainingSession::MAX_PAYLOAD_BYTES;
    #[classattr]
    const MAX_CHECKPOINT_INTERVAL: u64 = NativeCausalLmTrainingSession::MAX_CHECKPOINT_INTERVAL;
    #[classattr]
    const MAX_DOCUMENTS: usize = MAX_DOCUMENTS;
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(model, token_documents, context_length, *, learning_rate, mode="joint", epochs=None, batch_windows=None, labels=None, window_policy="disjoint"))]
    fn new(
        py: Python<'_>,
        model: &PyNativeCausalLm,
        token_documents: &Bound<'_, PyAny>,
        context_length: &Bound<'_, PyAny>,
        learning_rate: &Bound<'_, PyAny>,
        mode: &str,
        epochs: Option<&Bound<'_, PyAny>>,
        batch_windows: Option<&Bound<'_, PyAny>>,
        labels: Option<&Bound<'_, PyAny>>,
        window_policy: &str,
    ) -> PyResult<Self> {
        reject_bool(learning_rate)?;
        let config = CausalLmTrainingSessionConfig {
            context_length: u32::try_from(dimension(context_length)?).map_err(error)?,
            learning_rate: learning_rate.extract::<f32>()?,
            mode: self::mode(mode)?,
            epochs: u32::try_from(limit(epochs, 1)?).map_err(error)?,
        };
        let batch = limit(batch_windows, 1)?;
        let policy = match window_policy {
            "disjoint" => CausalLmSessionWindowPolicy::Disjoint,
            "covering_targets" => CausalLmSessionWindowPolicy::CoveringTargets,
            _ => return Err(error("window_policy must be disjoint or covering_targets")),
        };
        NativeCausalLmTrainingSession::validate_policy(&model.inner, config, batch)
            .map_err(error)?;
        let (docs, labels) = documents(
            token_documents,
            labels,
            NativeCausalLmTrainingSession::MAX_TOKENS,
            |lengths, has_labels| {
                py.detach(|| {
                    NativeCausalLmTrainingSession::document_plan_with_policy(
                        &model.inner,
                        config,
                        lengths,
                        batch,
                        policy,
                        has_labels,
                    )
                    .map(|_| ())
                })
                .map_err(error)
            },
        )?;
        py.detach(|| {
            NativeCausalLmTrainingSession::new_with_policy(
                model.inner.clone(),
                docs,
                labels,
                config,
                batch,
                policy,
            )
        })
        .map(|inner| Self { inner })
        .map_err(error)
    }
    #[staticmethod]
    #[pyo3(signature=(base, token_documents, payload, expected_id, *, labels=None))]
    fn from_json(
        py: Python<'_>,
        base: &PyNativeCausalLm,
        token_documents: &Bound<'_, PyAny>,
        payload: &Bound<'_, PyAny>,
        expected_id: &Bound<'_, PyAny>,
        labels: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let expected_id = super::session_input::pin(expected_id)?;
        let payload = super::session_input::payload(payload)?;
        let bytes = payload.as_bytes();
        let resume = py
            .detach(|| {
                NativeCausalLmTrainingSession::prepare_restore(
                    base.inner.clone(),
                    bytes,
                    &expected_id,
                )
            })
            .map_err(error)?;
        if resume.has_labels() != labels.is_some() {
            return Err(error("session checkpoint label presence differs"));
        }
        let (docs, labels) = documents(
            token_documents,
            labels,
            NativeCausalLmTrainingSession::MAX_TOKENS,
            |lengths, has_labels| {
                py.detach(|| {
                    NativeCausalLmTrainingSession::document_plan_with_policy(
                        resume.model(),
                        resume.config(),
                        lengths,
                        resume.batch_windows(),
                        resume.window_policy(),
                        has_labels,
                    )
                    .map(|_| ())
                })
                .map_err(error)
            },
        )?;
        py.detach(|| resume.resume_with_labels(docs, labels))
            .map(|inner| Self { inner })
            .map_err(error)
    }
    fn step(&mut self, py: Python<'_>) -> PyResult<Option<PyNativeCausalLmTrainingSessionStep>> {
        py.detach(|| self.inner.step())
            .map(|value| value.map(|inner| PyNativeCausalLmTrainingSessionStep { inner }))
            .map_err(error)
    }
    #[getter]
    fn model(&self) -> PyNativeCausalLm {
        PyNativeCausalLm {
            inner: self.inner.model().clone(),
        }
    }
    #[getter]
    fn data_id(&self) -> &str {
        self.inner.data_id()
    }
    #[getter]
    fn completed_updates(&self) -> u64 {
        self.inner.completed_updates()
    }
    #[getter]
    fn updates_per_epoch(&self) -> u64 {
        self.inner.updates_per_epoch()
    }
    #[getter]
    fn batch_windows(&self) -> usize {
        self.inner.batch_windows()
    }
    #[getter]
    fn has_labels(&self) -> bool {
        self.inner.has_labels()
    }
    #[getter]
    fn window_policy(&self) -> &'static str {
        match self.inner.window_policy() {
            CausalLmSessionWindowPolicy::Disjoint => "disjoint",
            CausalLmSessionWindowPolicy::CoveringTargets => "covering_targets",
        }
    }
    #[getter]
    fn total_updates(&self) -> u64 {
        self.inner.total_updates()
    }
    #[getter]
    fn epoch(&self) -> u32 {
        self.inner.epoch()
    }
    #[getter]
    fn next_window(&self) -> usize {
        self.inner.next_window()
    }
    #[getter]
    fn exhausted(&self) -> bool {
        self.inner.exhausted()
    }
    fn config_json(&self) -> PyResult<String> {
        self.inner.config_json().map_err(error)
    }
    fn plan_json(&self) -> PyResult<String> {
        self.inner.plan().to_json().map_err(error)
    }
    fn to_json(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| self.inner.to_json()).map_err(error)
    }
    #[getter]
    fn checkpoint_id(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| self.inner.checkpoint_id()).map_err(error)
    }
    fn checkpoint_with_id(&self, py: Python<'_>) -> PyResult<(String, String)> {
        py.detach(|| self.inner.checkpoint_with_id()).map_err(error)
    }

    fn save_checkpoint_new(&self, py: Python<'_>, path: std::path::PathBuf) -> PyResult<String> {
        py.detach(|| self.inner.save_checkpoint_new(&path))
            .map_err(checkpoint_error)
    }

    fn advance_to_checkpoint_new(
        &mut self,
        py: Python<'_>,
        stop_after: &Bound<'_, PyAny>,
        path: std::path::PathBuf,
    ) -> PyResult<(Vec<PyNativeCausalLmTrainingSessionStep>, String)> {
        reject_bool(stop_after)?;
        let stop_after = stop_after.extract::<u64>()?;
        let (reports, pin) = py
            .detach(|| self.inner.advance_to_checkpoint_new(stop_after, path))
            .map_err(checkpoint_error)?;
        Ok((
            reports
                .into_iter()
                .map(|inner| PyNativeCausalLmTrainingSessionStep { inner })
                .collect(),
            pin,
        ))
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyNativeCausalLmAdaptationGradients>()?;
    module.add_class::<PyNativeCausalLmTrainingSession>()?;
    module.add_class::<PyNativeCausalLmTrainingSessionStep>()
}
