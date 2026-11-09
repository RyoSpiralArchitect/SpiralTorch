//! Token transport only; model loading, execution, loss and geometry stay in Rust.
use crate::channel_affine::{dimension, reject_bool};
use crate::geometry_artifact::PyGeometryAdapterArtifact;
use crate::geometry_runtime::{adamw_step, PyNativeGeometryAdamW};
use crate::weights::{PyNdTensor, PySafeTensorArchive};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyBytesMethods};
use st_nn::causal_lm::{
    CausalLmLimits, CausalLmObjective, CausalLmReadoutAccumulator, NativeCausalLm,
    NativeCausalLmGradients, NativeCausalLmLoraGradients, TargetClassScore,
};
use std::collections::BTreeMap;

mod accumulation;
mod context;
mod declared;
mod documents;
mod evaluation;
mod generation;
mod groups;
mod line_menu;
mod mixture_probe;
mod output_margin;
mod prefix_cache;
mod projection_probe;
mod qkv;
mod qkv_session;
mod scalar_output;
mod session;
mod session_input;
mod supervision;
mod training;
use accumulation::PyNativeCausalLmAccumulator;
use session::PyNativeCausalLmAdaptationGradients;

fn error(value: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(value.to_string())
}

fn competition_recipe(
    rivals: &Bound<'_, PyAny>,
    mix: &Bound<'_, PyAny>,
) -> PyResult<st_nn::causal_lm::RivalCrossEntropy> {
    reject_bool(mix)?;
    Ok(st_nn::causal_lm::RivalCrossEntropy {
        rivals: dimension(rivals)?,
        mix: mix.extract()?,
    })
}

fn target_score_dict(py: Python<'_>, score: TargetClassScore) -> PyResult<Py<PyAny>> {
    crate::json::json_to_py(py, &serde_json::to_value(score).map_err(error)?)
}

/// Score one saved little-endian f32 vocabulary row; does not load or execute a model.
#[pyfunction]
fn score_token_logits_f32le(
    py: Python<'_>,
    logits: &Bound<'_, PyBytes>,
    target_id: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let target = dimension(target_id)?;
    let (words, tail) = logits.as_bytes().as_chunks::<4>();
    if !tail.is_empty() || words.len() > st_core::inference::token_decoding::MAX_VOCABULARY {
        return Err(error(
            "expected one bounded row of little-endian f32 logits",
        ));
    }
    TargetClassScore::validate_domain(words.len(), target).map_err(error)?;
    let score = py.detach(|| {
        let mut row = Vec::new();
        row.try_reserve_exact(words.len()).map_err(|_| {
            pyo3::exceptions::PyMemoryError::new_err("cannot allocate token score row")
        })?;
        row.extend(words.iter().copied().map(f32::from_le_bytes));
        TargetClassScore::from_logits(&row, target).map_err(error)
    })?;
    target_score_dict(py, score)
}

fn checkpoint_error(e: Box<dyn std::error::Error + Send + Sync>) -> PyErr {
    match e.downcast::<std::io::Error>() {
        Ok(io) => match io.raw_os_error() {
            // Preserve errno; on Windows the fourth argument carries the Win32 error.
            #[cfg(windows)]
            Some(code) => {
                pyo3::exceptions::PyOSError::new_err((0, io.to_string(), None::<String>, code))
            }
            #[cfg(not(windows))]
            Some(code) => pyo3::exceptions::PyOSError::new_err((code, io.to_string())),
            None => PyErr::from(*io),
        },
        Err(other) => error(other),
    }
}

fn shape(value: &Bound<'_, PyAny>) -> PyResult<[usize; 2]> {
    if value.len()? != 2 {
        return Err(error("native causal LM shape must be [batch, sequence]"));
    }
    Ok([
        dimension(&value.get_item(0)?)?,
        dimension(&value.get_item(1)?)?,
    ])
}

fn tokens(value: &Bound<'_, PyAny>, count: usize) -> PyResult<Vec<usize>> {
    if value.len()? != count {
        return Err(error("token ID count differs from shape"));
    }
    let mut result = Vec::new();
    result.try_reserve_exact(count).map_err(error)?;
    for i in 0..count {
        result.push(dimension(&value.get_item(i)?)?);
    }
    Ok(result)
}

fn context_preference_inputs(
    model: &NativeCausalLm,
    preferred: &Bound<'_, PyAny>,
    other: &Bound<'_, PyAny>,
    target: &Bound<'_, PyAny>,
    margin: Option<&Bound<'_, PyAny>>,
    training: bool,
) -> PyResult<(Vec<usize>, Vec<usize>, usize, f64)> {
    let target = dimension(target)?;
    let margin = match margin {
        Some(m) => {
            reject_bool(m)?;
            m.extract::<f64>()?
        }
        None => 0.,
    };
    let (a, b) = (preferred.len()?, other.len()?);
    model
        .context_token_preference_shape(a, b, target, margin, training)
        .map_err(error)?;
    Ok((tokens(preferred, a)?, tokens(other, b)?, target, margin))
}

fn loss_objective(
    value: Option<&Bound<'_, PyAny>>,
    repetition: Option<&Bound<'_, PyAny>>,
) -> PyResult<CausalLmObjective> {
    let label_smoothing = match value {
        Some(value) => {
            reject_bool(value)?;
            value.extract::<f64>()?
        }
        None => 0.,
    };
    let repetition = repetition
        .map(|value| {
            let value = crate::json::py_to_json_bounded(
                value,
                65_536,
                1024,
                8,
                "native repetition objective",
            )?;
            serde_json::from_value(value).map_err(error)
        })
        .transpose()?;
    let objective = CausalLmObjective {
        label_smoothing,
        repetition,
    };
    objective.validate().map_err(error)?;
    Ok(objective)
}

fn loss_labels(value: Option<&Bound<'_, PyAny>>, count: usize) -> PyResult<Option<Vec<i64>>> {
    let Some(value) = value else { return Ok(None) };
    if value.len()? != count {
        return Err(error("label count differs from window shape"));
    }
    let mut result = Vec::new();
    result.try_reserve_exact(count).map_err(error)?;
    for i in 0..count {
        let item = value.get_item(i)?;
        reject_bool(&item)?;
        result.push(item.extract::<i64>()?);
    }
    Ok(Some(result))
}

fn limit(value: Option<&Bound<'_, PyAny>>, default: usize) -> PyResult<usize> {
    match value {
        Some(value) => {
            reject_bool(value)?;
            value.extract()
        }
        None => Ok(default),
    }
}

#[pyclass(name = "NativeCausalLm", module = "spiraltorch", frozen)]
pub struct PyNativeCausalLm {
    inner: NativeCausalLm,
}

#[pyclass(name = "NativeCausalLmGradients", module = "spiraltorch", frozen)]
pub struct PyNativeCausalLmGradients {
    inner: NativeCausalLmGradients,
}

#[pymethods]
impl PyNativeCausalLmGradients {
    #[getter]
    fn readout_accumulator(&self) -> &'static str {
        self.inner.readout_accumulator().as_str()
    }
    #[getter]
    fn loss(&self) -> f32 {
        self.inner.loss()
    }
    #[getter]
    fn causal_loss(&self) -> f32 {
        self.inner.causal_loss()
    }
    #[getter]
    fn repetition_loss(&self) -> f32 {
        self.inner.repetition_loss()
    }
    #[getter]
    fn repetition_positions(&self) -> usize {
        self.inner.repetition_positions()
    }
    fn repetition_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner.objective().repetition).map_err(error)
    }
    #[getter]
    fn base_model_identity(&self) -> &str {
        self.inner.base_model_identity()
    }
    #[getter]
    fn geometry_artifact_id(&self) -> &str {
        self.inner.geometry_artifact_id()
    }
    #[getter]
    fn window_shape(&self) -> Vec<usize> {
        self.inner.window_shape().to_vec()
    }
    #[getter]
    fn target_tokens(&self) -> usize {
        self.inner.target_tokens()
    }
    #[getter]
    fn label_smoothing(&self) -> f64 {
        self.inner.objective().label_smoothing
    }
    fn parameter_gradients(&self) -> BTreeMap<String, BTreeMap<String, Vec<f32>>> {
        self.inner.parameters().clone()
    }
}

#[pyclass(name = "NativeCausalLmLoraGradients", module = "spiraltorch", frozen)]
pub struct PyNativeCausalLmLoraGradients {
    inner: NativeCausalLmLoraGradients,
}

#[pymethods]
impl PyNativeCausalLmLoraGradients {
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
    fn lora_artifact_id(&self) -> &str {
        self.inner.lora_artifact_id()
    }
    fn parameter_gradients(&self) -> BTreeMap<String, Vec<f32>> {
        BTreeMap::from([
            ("a".into(), self.inner.a().to_vec()),
            ("b".into(), self.inner.b().to_vec()),
        ])
    }
}

#[pymethods]
impl PyNativeCausalLm {
    /// Capture each unique prompt once; indexed pair and group changes need no further decoder calls.
    #[pyo3(signature=(prompts,target_id,alternative_id,*,layer,left_f64le=None))]
    fn output_contrast_bank(
        &self,
        py: Python<'_>,
        prompts: &Bound<'_, PyAny>,
        target_id: &Bound<'_, PyAny>,
        alternative_id: &Bound<'_, PyAny>,
        layer: &Bound<'_, PyAny>,
        left_f64le: Option<&Bound<'_, PyBytes>>,
    ) -> PyResult<output_margin::PyOutputContrastBank> {
        output_margin::build(
            py,
            &self.inner,
            prompts,
            dimension(target_id)?,
            dimension(alternative_id)?,
            dimension(layer)?,
            left_f64le,
        )
    }
    /// Fix two unit factors. A new handle owns the only trainable coefficient.
    fn scalar_output_adapter(
        &self,
        py: Python<'_>,
        layer: &Bound<'_, PyAny>,
        left_f64le: &Bound<'_, PyBytes>,
        right_f64le: &Bound<'_, PyBytes>,
    ) -> PyResult<scalar_output::PyScalarOutput> {
        let layer = dimension(layer)?;
        let w = self.inner.config().width;
        let decode = |b: &[u8]| -> PyResult<Vec<f64>> {
            if b.len() != 8 * w {
                return Err(error(
                    "scalar factors must be width f64 little-endian values",
                ));
            }
            Ok(b.chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                .collect())
        };
        let left = decode(left_f64le.as_bytes())?;
        let right = decode(right_f64le.as_bytes())?;
        py.detach(|| {
            st_nn::causal_lm::NativeCausalLmScalarOutput::new(&self.inner, layer, &left, &right, 0.)
        })
        .map(scalar_output::PyScalarOutput)
        .map_err(error)
    }

    fn restore_scalar_output_adapter(
        &self,
        py: Python<'_>,
        payload: &str,
        expected_sha256: &str,
    ) -> PyResult<scalar_output::PyScalarOutput> {
        py.detach(|| {
            st_nn::causal_lm::NativeCausalLmScalarOutput::restore(
                &self.inner,
                payload,
                expected_sha256,
            )
        })
        .map(scalar_output::PyScalarOutput)
        .map_err(error)
    }
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(base, payload, expected_id, *, update, recorded_data_id, post_lora_id, post_geometry_id))]
    fn from_session_checkpoint_at(
        py: Python<'_>,
        base: &Self,
        payload: &Bound<'_, PyAny>,
        expected_id: &Bound<'_, PyAny>,
        update: &Bound<'_, PyAny>,
        recorded_data_id: &Bound<'_, PyAny>,
        post_lora_id: Option<&Bound<'_, PyAny>>,
        post_geometry_id: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        reject_bool(update)?;
        let update = update.extract::<u64>()?;
        let expected_id = session_input::pin(expected_id)?;
        let data = session_input::pin(recorded_data_id)?;
        let lora = post_lora_id.map(session_input::pin).transpose()?;
        let geometry = post_geometry_id.map(session_input::pin).transpose()?;
        let payload = session_input::payload(payload)?;
        let bytes = payload.as_bytes();
        py.detach(|| {
            NativeCausalLm::from_session_checkpoint_at(
                base.inner.clone(),
                bytes,
                &expected_id,
                update,
                &data,
                lora.as_deref(),
                geometry.as_deref(),
            )
        })
        .map(|inner| Self { inner })
        .map_err(error)
    }
    /// With no mode, expose all attached gradients. Otherwise retain the full
    /// forward function but return only the Rust-selected parameter groups.
    #[pyo3(signature=(windows, shape, *, mode=None, labels=None, observe_geometry=false))]
    fn adaptation_loss_gradients(
        &self,
        py: Python<'_>,
        windows: &Bound<'_, PyAny>,
        shape: &Bound<'_, PyAny>,
        mode: Option<&str>,
        labels: Option<&Bound<'_, PyAny>>,
        observe_geometry: bool,
    ) -> PyResult<PyNativeCausalLmAdaptationGradients> {
        let mode = mode.map(session::mode).transpose()?;
        if let Some(mode) = mode {
            self.inner.validate_adaptation_mode(mode).map_err(error)?;
        }
        let shape = self::shape(shape)?;
        let count = self
            .inner
            .adaptation_labeled_input_count(shape, labels.is_some())
            .map_err(error)?;
        let windows = tokens(windows, count)?;
        let labels = loss_labels(labels, count)?;
        py.detach(|| {
            self.inner.adaptation_window_gradients(
                &windows,
                shape,
                labels.as_deref(),
                mode,
                observe_geometry,
            )
        })
        .map(|inner| PyNativeCausalLmAdaptationGradients { inner })
        .map_err(error)
    }

    /// Train geometry with a convex blend of full CE and target-versus-top-rivals CE.
    #[pyo3(signature=(windows, shape, *, rivals, mix, labels=None))]
    fn geometry_competition_gradients(
        &self,
        py: Python<'_>,
        windows: &Bound<'_, PyAny>,
        shape: &Bound<'_, PyAny>,
        rivals: &Bound<'_, PyAny>,
        mix: &Bound<'_, PyAny>,
        labels: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyNativeCausalLmAdaptationGradients> {
        let recipe = competition_recipe(rivals, mix)?;
        let shape = self::shape(shape)?;
        let count = self
            .inner
            .geometry_competition_input_count(shape, labels.is_some(), recipe)
            .map_err(error)?;
        let ids = tokens(windows, count)?;
        let labels = loss_labels(labels, count)?;
        py.detach(|| {
            self.inner
                .geometry_competition_gradients(&ids, shape, labels.as_deref(), recipe)
        })
        .map(|inner| PyNativeCausalLmAdaptationGradients { inner })
        .map_err(error)
    }

    /// Forward-only conditional loss over complete, sorted content alternatives plus one stop.
    #[pyo3(signature=(prompt_ids, candidates, *, gold_index, stop_id, mix))]
    #[allow(clippy::too_many_arguments)]
    fn line_menu_loss(
        &self,
        py: Python<'_>,
        prompt_ids: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        gold_index: &Bound<'_, PyAny>,
        stop_id: &Bound<'_, PyAny>,
        mix: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let (b, gold, mix) = line_menu::extract(
            &self.inner,
            prompt_ids,
            candidates,
            gold_index,
            stop_id,
            mix,
            false,
        )?;
        let r = py
            .detach(|| {
                self.inner
                    .line_menu_loss(&b.prompt, &b.lines, gold, b.stop, mix)
            })
            .map_err(error)?;
        crate::json::json_to_py(py, &serde_json::to_value(r).map_err(error)?)
    }

    #[pyo3(signature=(prompt_ids, candidates, *, gold_index, stop_id, mix))]
    #[allow(clippy::too_many_arguments)]
    fn geometry_line_menu_gradients(
        &self,
        py: Python<'_>,
        prompt_ids: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        gold_index: &Bound<'_, PyAny>,
        stop_id: &Bound<'_, PyAny>,
        mix: &Bound<'_, PyAny>,
    ) -> PyResult<PyNativeCausalLmAdaptationGradients> {
        let (b, gold, mix) = line_menu::extract(
            &self.inner,
            prompt_ids,
            candidates,
            gold_index,
            stop_id,
            mix,
            true,
        )?;
        py.detach(|| {
            self.inner
                .geometry_line_menu_gradients(&b.prompt, &b.lines, gold, b.stop, mix)
        })
        .map(|inner| PyNativeCausalLmAdaptationGradients { inner })
        .map_err(error)
    }

    /// Prefer the very same next token in the first prefix, relative to the other prefix.
    #[pyo3(signature=(preferred_ids, other_ids, target_id, *, margin=None))]
    fn context_token_preference(
        &self,
        py: Python<'_>,
        preferred_ids: &Bound<'_, PyAny>,
        other_ids: &Bound<'_, PyAny>,
        target_id: &Bound<'_, PyAny>,
        margin: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let (a, b, k, m) = context_preference_inputs(
            &self.inner,
            preferred_ids,
            other_ids,
            target_id,
            margin,
            false,
        )?;
        let r = py
            .detach(|| self.inner.context_token_preference(&a, &b, k, m))
            .map_err(error)?;
        crate::json::json_to_py(py, &serde_json::to_value(r).map_err(error)?)
    }

    #[pyo3(signature=(preferred_ids, other_ids, target_id, *, margin=None))]
    fn geometry_context_token_preference_gradients(
        &self,
        py: Python<'_>,
        preferred_ids: &Bound<'_, PyAny>,
        other_ids: &Bound<'_, PyAny>,
        target_id: &Bound<'_, PyAny>,
        margin: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyNativeCausalLmAdaptationGradients> {
        let (a, b, k, m) = context_preference_inputs(
            &self.inner,
            preferred_ids,
            other_ids,
            target_id,
            margin,
            true,
        )?;
        py.detach(|| {
            self.inner
                .geometry_context_token_preference_gradients(&a, &b, k, m)
        })
        .map(|inner| PyNativeCausalLmAdaptationGradients { inner })
        .map_err(error)
    }

    /// Label-free full-line, menu-conditional and per-token NLLs including the stop.
    #[pyo3(signature=(prompt_ids, candidates, *, stop_id))]
    fn score_lines(
        &self,
        py: Python<'_>,
        prompt_ids: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        stop_id: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let b = line_menu::collect(&self.inner, prompt_ids, candidates, stop_id, None, false)?;
        let r = py
            .detach(|| self.inner.score_lines(&b.prompt, &b.lines, b.stop))
            .map_err(error)?;
        crate::json::json_to_py(py, &serde_json::to_value(r).map_err(error)?)
    }

    /// Absolute probability loss for emitting any one of the terminated alternatives.
    #[pyo3(signature=(prompt_ids, candidates, *, stop_id))]
    fn line_set_loss(
        &self,
        py: Python<'_>,
        prompt_ids: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        stop_id: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let b = line_menu::collect(&self.inner, prompt_ids, candidates, stop_id, None, false)?;
        let r = py
            .detach(|| self.inner.line_set_loss(&b.prompt, &b.lines, b.stop))
            .map_err(error)?;
        crate::json::json_to_py(py, &serde_json::to_value(r).map_err(error)?)
    }

    #[pyo3(signature=(prompt_ids, candidates, *, stop_id))]
    fn geometry_line_set_gradients(
        &self,
        py: Python<'_>,
        prompt_ids: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        stop_id: &Bound<'_, PyAny>,
    ) -> PyResult<PyNativeCausalLmAdaptationGradients> {
        let b = line_menu::collect(&self.inner, prompt_ids, candidates, stop_id, None, true)?;
        py.detach(|| {
            self.inner
                .geometry_line_set_gradients(&b.prompt, &b.lines, b.stop)
        })
        .map(|inner| PyNativeCausalLmAdaptationGradients { inner })
        .map_err(error)
    }

    /// Conditional likelihood of any accepted terminated line inside the menu.
    #[pyo3(signature=(prompt_ids, candidates, *, accepted_indices, stop_id))]
    fn line_subset_loss(
        &self,
        py: Python<'_>,
        prompt_ids: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        accepted_indices: &Bound<'_, PyAny>,
        stop_id: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let (b, ids) = line_menu::subset(
            &self.inner,
            prompt_ids,
            candidates,
            accepted_indices,
            stop_id,
            false,
        )?;
        let r = py
            .detach(|| {
                self.inner
                    .line_subset_loss(&b.prompt, &b.lines, &ids, b.stop)
            })
            .map_err(error)?;
        crate::json::json_to_py(py, &serde_json::to_value(r).map_err(error)?)
    }

    #[pyo3(signature=(prompt_ids, candidates, *, accepted_indices, stop_id))]
    fn geometry_line_subset_gradients(
        &self,
        py: Python<'_>,
        prompt_ids: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        accepted_indices: &Bound<'_, PyAny>,
        stop_id: &Bound<'_, PyAny>,
    ) -> PyResult<PyNativeCausalLmAdaptationGradients> {
        let (b, ids) = line_menu::subset(
            &self.inner,
            prompt_ids,
            candidates,
            accepted_indices,
            stop_id,
            true,
        )?;
        py.detach(|| {
            self.inner
                .geometry_line_subset_gradients(&b.prompt, &b.lines, &ids, b.stop)
        })
        .map(|inner| PyNativeCausalLmAdaptationGradients { inner })
        .map_err(error)
    }

    /// Supervised (flat unshifted target position, expected label, sorted rival IDs) triples.
    #[pyo3(signature=(windows, shape, *, contrasts, mix, labels=None))]
    fn geometry_declared_rival_gradients(
        &self,
        py: Python<'_>,
        windows: &Bound<'_, PyAny>,
        shape: &Bound<'_, PyAny>,
        contrasts: &Bound<'_, PyAny>,
        mix: &Bound<'_, PyAny>,
        labels: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyNativeCausalLmAdaptationGradients> {
        let shape = self::shape(shape)?;
        reject_bool(mix)?;
        let mix = mix.extract::<f64>()?;
        self.inner
            .validate_adaptation_mode(st_nn::causal_lm::CausalLmAdaptationMode::Geometry)
            .map_err(error)?;
        let count = self
            .inner
            .adaptation_labeled_input_count(shape, labels.is_some())
            .map_err(error)?;
        let plan = declared::extract(contrasts, shape, count, self.inner.config().vocabulary, mix)?;
        let ids = tokens(windows, count)?;
        let labels = loss_labels(labels, count)?;
        py.detach(|| {
            self.inner
                .geometry_declared_rival_gradients(&ids, shape, labels.as_deref(), &plan, mix)
        })
        .map(|inner| PyNativeCausalLmAdaptationGradients { inner })
        .map_err(error)
    }

    /// Inspect CE, declared and mixed VJPs after one f32 forward; the second
    /// path widens only the backward vocabulary projection.
    #[pyo3(signature=(prompt_ids, target_id, rival_id, *, mix))]
    fn last_attention_mixture_probe(
        &self,
        py: Python<'_>,
        prompt_ids: &Bound<'_, PyAny>,
        target_id: &Bound<'_, PyAny>,
        rival_id: &Bound<'_, PyAny>,
        mix: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let target = dimension(target_id)?;
        let rival = dimension(rival_id)?;
        reject_bool(mix)?;
        let mix = mix.extract::<f64>()?;
        let count = self
            .inner
            .mixture_probe_input_count(prompt_ids.len()?, target, rival, mix)
            .map_err(error)?;
        let ids = tokens(prompt_ids, count)?;
        let report = py
            .detach(|| {
                self.inner
                    .last_attention_mixture_probe(&ids, target, rival, mix)
            })
            .map_err(error)?;
        mixture_probe::to_python(py, report)
    }

    /// Four local channel maps on a single factual decoder forward and CE pullback.
    #[pyo3(signature=(windows, shape, config, *, labels=None))]
    fn wave_factorial_observation_json(
        &self,
        py: Python<'_>,
        windows: &Bound<'_, PyAny>,
        shape: &Bound<'_, PyAny>,
        config: &Bound<'_, PyAny>,
        labels: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<String> {
        let config = serde_json::from_value(crate::json::py_to_json_bounded(
            config,
            2048,
            16,
            3,
            "wave factorial",
        )?)
        .map_err(error)?;
        self.inner.admit_wave_factorial(config).map_err(error)?;
        let shape = self::shape(shape)?;
        let count = self
            .inner
            .adaptation_labeled_input_count(shape, labels.is_some())
            .map_err(error)?;
        let ids = tokens(windows, count)?;
        let labels = loss_labels(labels, count)?;
        py.detach(|| {
            self.inner
                .wave_factorial_observation(&ids, shape, labels.as_deref(), config)
                .and_then(|r| Ok(serde_json::to_string(&r)?))
        })
        .map_err(error)
    }

    #[pyo3(signature=(gradients, mode, learning_rate))]
    fn step_adaptation_sgd(
        &self,
        py: Python<'_>,
        gradients: &PyNativeCausalLmAdaptationGradients,
        mode: &str,
        learning_rate: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let mode = session::mode(mode)?;
        reject_bool(learning_rate)?;
        let learning_rate = learning_rate.extract::<f32>()?;
        py.detach(|| {
            self.inner
                .step_adaptation_sgd(&gradients.inner, mode, learning_rate)
        })
        .map(|inner| Self { inner })
        .map_err(error)
    }

    /// Advance only geometry using a sealed gradient and a previously measured step length.
    fn step_geometry_sgd_at_distance(
        &self,
        py: Python<'_>,
        gradients: &PyNativeCausalLmAdaptationGradients,
        learning_rate: &Bound<'_, PyAny>,
        target_l2: &Bound<'_, PyAny>,
    ) -> PyResult<(Self, Py<PyAny>)> {
        reject_bool(learning_rate)?;
        reject_bool(target_l2)?;
        let rate = learning_rate.extract::<f32>()?;
        let distance = target_l2.extract::<f64>()?;
        let (inner, report) = py
            .detach(|| {
                self.inner
                    .step_geometry_sgd_at_distance(&gradients.inner, rate, distance)
            })
            .map_err(error)?;
        Ok((
            Self { inner },
            crate::json::json_to_py(py, &serde_json::to_value(report).map_err(error)?)?,
        ))
    }

    /// Apply one equal-vote mean of sealed geometry gradients from this snapshot.
    #[pyo3(signature=(gradients, learning_rate, *, target_l2=None))]
    fn step_geometry_cohort_sgd(
        &self,
        py: Python<'_>,
        gradients: &Bound<'_, PyAny>,
        learning_rate: &Bound<'_, PyAny>,
        target_l2: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(Self, Py<PyAny>)> {
        let n = gradients.len()?;
        reject_bool(learning_rate)?;
        let rate = learning_rate.extract::<f32>()?;
        let target = target_l2
            .map(|v| {
                reject_bool(v)?;
                v.extract::<f64>()
            })
            .transpose()?;
        self.inner
            .validate_geometry_cohort(n, rate, target)
            .map_err(error)?;
        let owners = (0..n)
            .map(|i| {
                gradients
                    .get_item(i)?
                    .cast_into::<PyNativeCausalLmAdaptationGradients>()
                    .map_err(Into::into)
            })
            .collect::<PyResult<Vec<_>>>()?;
        let members: Vec<_> = owners.iter().map(|o| &o.get().inner).collect();
        let (inner, receipt) = py
            .detach(|| self.inner.step_geometry_cohort_sgd(&members, rate, target))
            .map_err(error)?;
        Ok((
            Self { inner },
            crate::json::json_to_py(py, &serde_json::to_value(receipt).map_err(error)?)?,
        ))
    }

    /// Strict common descent on 2 to 8 groups of same-snapshot geometry-only objectives.
    #[pyo3(signature=(groups, learning_rate, *, target_l2=None))]
    fn step_common_geometry_sgd(
        &self,
        py: Python<'_>,
        groups: &Bound<'_, PyAny>,
        learning_rate: &Bound<'_, PyAny>,
        target_l2: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(Self, Py<PyAny>)> {
        let n = groups.len()?;
        if !(2..=st_nn::causal_lm::MAX_COMMON_GEOMETRY_GROUPS).contains(&n) {
            return Err(error("expected 2 to 8 geometry groups"));
        }
        reject_bool(learning_rate)?;
        let rate = learning_rate.extract::<f32>()?;
        let target = target_l2
            .map(|v| {
                reject_bool(v)?;
                v.extract::<f64>()
            })
            .transpose()?;
        groups::apply(
            py,
            groups,
            n,
            None,
            |sizes| self.inner.validate_common_geometry(sizes, rate, target),
            |gs| self.inner.step_common_geometry_sgd(gs, rate, target),
        )
    }

    /// Preserve a safe geometry proposal, or repair its direction at the same length.
    #[pyo3(signature=(candidate, groups, *, min_cosine=1e-5))]
    fn guard_geometry_step(
        &self,
        py: Python<'_>,
        candidate: &Self,
        groups: &Bound<'_, PyAny>,
        min_cosine: f64,
    ) -> PyResult<(Self, Py<PyAny>)> {
        let n = groups.len()?;
        if !(1..=st_nn::causal_lm::MAX_GEOMETRY_COHORT_MEMBERS).contains(&n) {
            return Err(error("expected 1 to 256 geometry guard groups"));
        }
        groups::apply(
            py,
            groups,
            n,
            None,
            |sizes| {
                self.inner
                    .validate_geometry_step_guard(&candidate.inner, sizes, min_cosine)
            },
            |gs| {
                self.inner
                    .guard_geometry_step(&candidate.inner, gs, min_cosine)
            },
        )
    }

    /// Steer a safe anchor toward one primary objective while preserving each protected floor.
    #[pyo3(signature=(anchor, primary, protected, *, min_cosine=1e-5))]
    fn steer_geometry_step<'py>(
        &self,
        py: Python<'py>,
        anchor: &Self,
        primary: &Bound<'py, PyAny>,
        protected: &Bound<'py, PyAny>,
        min_cosine: f64,
    ) -> PyResult<(Self, Py<PyAny>)> {
        let n = protected.len()?;
        if !(1..st_nn::causal_lm::MAX_GEOMETRY_COHORT_MEMBERS).contains(&n) {
            return Err(error(
                "expected 1 to 255 protected geometry steering groups",
            ));
        }
        groups::apply(
            py,
            protected,
            n,
            Some(primary),
            |sizes| {
                self.inner
                    .validate_geometry_steering(&anchor.inner, sizes, min_cosine)
            },
            |gs| {
                self.inner
                    .steer_geometry_step(&anchor.inner, gs[0], &gs[1..], min_cosine)
            },
        )
    }

    /// Combine two sealed geometry-only gradients from this snapshot, without another forward.
    #[pyo3(signature=(left, right, learning_rate, *, policy="equal", target_l2=None))]
    fn step_paired_geometry_sgd(
        &self,
        py: Python<'_>,
        left: &PyNativeCausalLmAdaptationGradients,
        right: &PyNativeCausalLmAdaptationGradients,
        learning_rate: &Bound<'_, PyAny>,
        policy: &str,
        target_l2: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(Self, Py<PyAny>)> {
        reject_bool(learning_rate)?;
        let rate = learning_rate.extract::<f32>()?;
        let distance = target_l2
            .map(|v| {
                reject_bool(v)?;
                v.extract::<f64>()
            })
            .transpose()?;
        let policy =
            serde_json::from_value(serde_json::Value::String(policy.into())).map_err(error)?;
        let (inner, receipt) = py
            .detach(|| {
                self.inner.step_paired_geometry_sgd(
                    &left.inner,
                    &right.inner,
                    rate,
                    policy,
                    distance,
                )
            })
            .map_err(error)?;
        Ok((
            Self { inner },
            crate::json::json_to_py(py, &serde_json::to_value(receipt).map_err(error)?)?,
        ))
    }

    /// Return (updated CE, updated competition, report); neither input model is mutated.
    #[pyo3(signature=(mixed_model, windows, shape, *, learning_rate, rivals, mix, labels=None, coverage_guard=None))]
    #[allow(clippy::too_many_arguments)]
    fn geometry_competition_pair_step(
        &self,
        py: Python<'_>,
        mixed_model: &Self,
        windows: &Bound<'_, PyAny>,
        shape: &Bound<'_, PyAny>,
        learning_rate: &Bound<'_, PyAny>,
        rivals: &Bound<'_, PyAny>,
        mix: &Bound<'_, PyAny>,
        labels: Option<&Bound<'_, PyAny>>,
        coverage_guard: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(Self, Self, Py<PyAny>)> {
        reject_bool(learning_rate)?;
        let rate = learning_rate.extract::<f32>()?;
        let recipe = competition_recipe(rivals, mix)?;
        let strength = match coverage_guard {
            Some(value) => {
                reject_bool(value)?;
                value.extract::<f64>()?
            }
            None => 0.,
        };
        let pair = py
            .detach(|| {
                self.inner
                    .geometry_competition_pair(&mixed_model.inner, recipe, rate)?
                    .with_coverage_guard(strength)
            })
            .map_err(error)?;
        let shape = self::shape(shape)?;
        let count = pair.input_count(shape, labels.is_some()).map_err(error)?;
        let ids = tokens(windows, count)?;
        let labels = loss_labels(labels, count)?;
        let (ce, mixed, report) = py
            .detach(|| pair.step(&ids, shape, labels.as_deref()))
            .map_err(error)?;
        Ok((
            Self { inner: ce },
            Self { inner: mixed },
            crate::json::json_to_py(py, &serde_json::to_value(report).map_err(error)?)?,
        ))
    }

    /// Attach one native GPT-2 attention output adapter; no foreign computation.
    fn with_lora(
        &self,
        py: Python<'_>,
        layer: &Bound<'_, PyAny>,
        rank: &Bound<'_, PyAny>,
        alpha: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let layer = dimension(layer)?;
        let rank = dimension(rank)?;
        reject_bool(alpha)?;
        let alpha = alpha.extract::<f32>()?;
        py.detach(|| self.inner.with_lora(layer, rank, alpha))
            .map(|inner| Self { inner })
            .map_err(error)
    }

    #[getter]
    fn lora_artifact_id(&self) -> Option<&str> {
        self.inner.lora_artifact_id()
    }

    fn lora_to_json(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| self.inner.lora_to_json()).map_err(error)
    }

    fn with_lora_json(&self, py: Python<'_>, payload: &str) -> PyResult<Self> {
        py.detach(|| self.inner.with_lora_json(payload))
            .map(|inner| Self { inner })
            .map_err(error)
    }

    #[pyo3(signature=(windows, shape, *, labels=None))]
    fn lora_loss_gradients(
        &self,
        py: Python<'_>,
        windows: &Bound<'_, PyAny>,
        shape: &Bound<'_, PyAny>,
        labels: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyNativeCausalLmLoraGradients> {
        let shape = self::shape(shape)?;
        let count = self
            .inner
            .lora_labeled_input_count(shape, labels.is_some())
            .map_err(error)?;
        let windows = tokens(windows, count)?;
        let labels = loss_labels(labels, count)?;
        py.detach(|| {
            self.inner
                .lora_loss_gradients_with_labels(&windows, shape, labels.as_deref())
        })
        .map(|inner| PyNativeCausalLmLoraGradients { inner })
        .map_err(error)
    }

    fn step_lora_sgd(
        &self,
        py: Python<'_>,
        gradients: &PyNativeCausalLmLoraGradients,
        learning_rate: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        reject_bool(learning_rate)?;
        let learning_rate = learning_rate.extract::<f32>()?;
        py.detach(|| self.inner.step_lora_sgd(&gradients.inner, learning_rate))
            .map(|inner| Self { inner })
            .map_err(error)
    }

    #[staticmethod]
    #[pyo3(signature=(variants, token_documents, context_length, *, microbatch_windows=None, labels=None))]
    fn evaluate_variants(
        py: Python<'_>,
        variants: &Bound<'_, PyAny>,
        token_documents: &Bound<'_, PyAny>,
        context_length: &Bound<'_, PyAny>,
        microbatch_windows: Option<&Bound<'_, PyAny>>,
        labels: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<evaluation::PyNativeCausalLmEvaluationComparison> {
        evaluation::compare(
            py,
            variants,
            token_documents,
            context_length,
            microbatch_windows,
            labels,
        )
    }

    #[pyo3(signature=(token_documents, context_length, *, microbatch_windows=None, labels=None))]
    fn evaluate_documents(
        &self,
        py: Python<'_>,
        token_documents: &Bound<'_, PyAny>,
        context_length: &Bound<'_, PyAny>,
        microbatch_windows: Option<&Bound<'_, PyAny>>,
        labels: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<evaluation::PyNativeCausalLmEvaluation> {
        evaluation::evaluate(
            self,
            py,
            token_documents,
            context_length,
            microbatch_windows,
            labels,
        )
    }

    #[staticmethod]
    #[pyo3(signature=(variants, token_documents, context_length, *, microbatch_windows=None, labels=None))]
    fn evaluate_variants_by_answer(
        py: Python<'_>,
        variants: &Bound<'_, PyAny>,
        token_documents: &Bound<'_, PyAny>,
        context_length: &Bound<'_, PyAny>,
        microbatch_windows: Option<&Bound<'_, PyAny>>,
        labels: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<evaluation::PyNativeCausalLmAnswerComparison> {
        evaluation::compare_answers(
            py,
            variants,
            token_documents,
            context_length,
            microbatch_windows,
            labels,
        )
    }

    #[getter]
    fn readout_accumulator(&self) -> &'static str {
        self.inner.readout_accumulator().as_str()
    }

    fn with_readout_accumulator(&self, accumulator: &str) -> PyResult<Self> {
        let accumulator = accumulator
            .parse::<CausalLmReadoutAccumulator>()
            .map_err(error)?;
        Ok(Self {
            inner: self.inner.with_readout_accumulator(accumulator),
        })
    }

    #[getter]
    fn readout_forward_precision(&self) -> &'static str {
        self.inner
            .readout_accumulator()
            .forward_precision()
            .as_str()
    }

    #[getter]
    fn readout_backward_precision(&self) -> &'static str {
        self.inner
            .readout_accumulator()
            .backward_precision()
            .as_str()
    }

    #[staticmethod]
    #[pyo3(signature = (base, payload, expected_id, *, update, recorded_data_id, post_geometry_id))]
    fn from_training_checkpoint_at(
        py: Python<'_>,
        base: &Self,
        payload: &Bound<'_, PyAny>,
        expected_id: &Bound<'_, PyAny>,
        update: &Bound<'_, PyAny>,
        recorded_data_id: &str,
        post_geometry_id: &str,
    ) -> PyResult<Self> {
        reject_bool(update)?;
        let update = update.extract::<u64>()?;
        let pin = session_input::pin(expected_id)?;
        let payload = session_input::payload_with_limit(
            payload,
            st_nn::causal_lm::NativeCausalLmTrainer::MAX_PAYLOAD_BYTES,
        )?;
        let bytes = payload.as_bytes();
        py.detach(|| {
            NativeCausalLm::from_training_checkpoint_at(
                base.inner.clone(),
                bytes,
                &pin,
                update,
                recorded_data_id,
                post_geometry_id,
            )
        })
        .map(|inner| Self { inner })
        .map_err(error)
    }

    #[staticmethod]
    fn from_training_checkpoint(
        py: Python<'_>,
        base: &Self,
        payload: &Bound<'_, PyAny>,
        expected_id: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let pin = session_input::pin(expected_id)?;
        let payload = session_input::payload_with_limit(
            payload,
            st_nn::causal_lm::NativeCausalLmTrainer::MAX_PAYLOAD_BYTES,
        )?;
        let bytes = payload.as_bytes();
        py.detach(|| NativeCausalLm::from_training_checkpoint(base.inner.clone(), bytes, &pin))
            .map(|inner| Self { inner })
            .map_err(error)
    }

    #[staticmethod]
    #[pyo3(signature = (config_json, weights, *, max_parameter_elements=None, max_activation_elements=None, max_attention_pairs=None, max_logit_elements=None, max_training_checkpoint_elements=None))]
    fn from_gpt2(
        py: Python<'_>,
        config_json: &str,
        weights: &PySafeTensorArchive,
        max_parameter_elements: Option<&Bound<'_, PyAny>>,
        max_activation_elements: Option<&Bound<'_, PyAny>>,
        max_attention_pairs: Option<&Bound<'_, PyAny>>,
        max_logit_elements: Option<&Bound<'_, PyAny>>,
        max_training_checkpoint_elements: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let defaults = CausalLmLimits::default();
        let limits = CausalLmLimits {
            max_parameter_elements: limit(max_parameter_elements, defaults.max_parameter_elements)?,
            max_activation_elements: limit(
                max_activation_elements,
                defaults.max_activation_elements,
            )?,
            max_attention_pairs: limit(max_attention_pairs, defaults.max_attention_pairs)?,
            max_logit_elements: limit(max_logit_elements, defaults.max_logit_elements)?,
            max_training_checkpoint_elements: limit(
                max_training_checkpoint_elements,
                defaults.max_training_checkpoint_elements,
            )?,
        };
        py.detach(|| NativeCausalLm::from_gpt2_json(config_json.as_bytes(), &weights.inner, limits))
            .map(|inner| Self { inner })
            .map_err(error)
    }

    #[getter]
    fn execution_backend(&self) -> &'static str {
        "rust_f32_cpu_prepacked"
    }

    #[getter]
    fn base_model_identity(&self) -> &str {
        self.inner.base_model_identity()
    }

    #[getter]
    fn geometry_artifact_id(&self) -> Option<&str> {
        self.inner.geometry_artifact_id()
    }

    fn config_json(&self) -> PyResult<String> {
        serde_json::to_string(self.inner.config()).map_err(error)
    }

    fn geometry_artifact(&self) -> Option<PyGeometryAdapterArtifact> {
        self.inner
            .geometry_artifact()
            .cloned()
            .map(|inner| PyGeometryAdapterArtifact { inner })
    }

    #[pyo3(signature = (*, label_smoothing=None, repetition=None))]
    fn geometry_accumulator(
        &self,
        py: Python<'_>,
        label_smoothing: Option<&Bound<'_, PyAny>>,
        repetition: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyNativeCausalLmAccumulator> {
        let objective = loss_objective(label_smoothing, repetition)?;
        py.detach(|| self.inner.geometry_accumulator_with_objective(objective))
            .map(|inner| PyNativeCausalLmAccumulator { inner })
            .map_err(error)
    }

    #[pyo3(signature = (windows, shape, *, labels=None, label_smoothing=None, repetition=None))]
    fn geometry_loss_gradients(
        &self,
        py: Python<'_>,
        windows: &Bound<'_, PyAny>,
        shape: &Bound<'_, PyAny>,
        labels: Option<&Bound<'_, PyAny>>,
        label_smoothing: Option<&Bound<'_, PyAny>>,
        repetition: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyNativeCausalLmGradients> {
        let objective = loss_objective(label_smoothing, repetition)?;
        let shape = self::shape(shape)?;
        self.inner
            .validate_objective_shape(shape, objective)
            .map_err(error)?;
        let count = self
            .inner
            .geometry_objective_input_count(shape, labels.is_some())
            .map_err(error)?;
        let windows = tokens(windows, count)?;
        let labels = loss_labels(labels, count)?;
        py.detach(|| {
            self.inner.geometry_loss_gradients_with_objective(
                &windows,
                shape,
                labels.as_deref(),
                objective,
            )
        })
        .map(|inner| PyNativeCausalLmGradients { inner })
        .map_err(error)
    }

    fn step_geometry_sgd(
        &self,
        py: Python<'_>,
        gradients: &PyNativeCausalLmGradients,
        learning_rate: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        reject_bool(learning_rate)?;
        let learning_rate = learning_rate.extract::<f32>()?;
        py.detach(|| {
            self.inner
                .step_geometry_sgd(&gradients.inner, learning_rate)
        })
        .map(|inner| Self { inner })
        .map_err(error)
    }

    #[pyo3(signature = (gradients, optimizer, learning_rate, *, weight_decay=None))]
    fn step_geometry_adamw(
        &self,
        py: Python<'_>,
        gradients: &PyNativeCausalLmGradients,
        optimizer: &PyNativeGeometryAdamW,
        learning_rate: &Bound<'_, PyAny>,
        weight_decay: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(Self, PyNativeGeometryAdamW)> {
        let step = adamw_step(learning_rate, weight_decay)?;
        py.detach(|| {
            self.inner
                .step_geometry_adamw(&gradients.inner, &optimizer.inner, step)
        })
        .map(|(inner, optimizer)| (Self { inner }, PyNativeGeometryAdamW { inner: optimizer }))
        .map_err(error)
    }

    fn with_geometry(
        &self,
        py: Python<'_>,
        artifact: &PyGeometryAdapterArtifact,
    ) -> PyResult<Self> {
        py.detach(|| self.inner.with_geometry(artifact.inner.clone()))
            .map(|inner| Self { inner })
            .map_err(error)
    }

    #[pyo3(signature = (input_ids, shape, *, last_only=false))]
    fn forward(
        &self,
        py: Python<'_>,
        input_ids: &Bound<'_, PyAny>,
        shape: &Bound<'_, PyAny>,
        last_only: bool,
    ) -> PyResult<PyNdTensor> {
        let shape = self::shape(shape)?;
        let count = self.inner.input_count(shape, last_only).map_err(error)?;
        let ids = tokens(input_ids, count)?;
        py.detach(|| self.inner.forward(&ids, shape, last_only))
            .map(|inner| PyNdTensor { inner })
            .map_err(error)
    }

    fn next_token_ids(
        &self,
        py: Python<'_>,
        input_ids: &Bound<'_, PyAny>,
        shape: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<usize>> {
        let shape = self::shape(shape)?;
        let count = self.inner.input_count(shape, true).map_err(error)?;
        let ids = tokens(input_ids, count)?;
        py.detach(|| self.inner.next_token_ids(&ids, shape))
            .map_err(error)
    }

    #[pyo3(signature=(prompt_ids,target_id,alternative_id,*,layer,site,basis="diagonal"))]
    fn projection_contrast_pullback(
        &self,
        py: Python<'_>,
        prompt_ids: &Bound<'_, PyAny>,
        target_id: &Bound<'_, PyAny>,
        alternative_id: &Bound<'_, PyAny>,
        layer: &Bound<'_, PyAny>,
        site: &str,
        basis: &str,
    ) -> PyResult<Py<PyAny>> {
        let (target, alt, spec) =
            projection_probe::spec(target_id, alternative_id, layer, site, basis)?;
        let steps = prompt_ids.len()?;
        self.inner
            .projection_probe_dimensions(steps, target, alt, spec, true)
            .map_err(error)?;
        let ids = tokens(prompt_ids, steps)?;
        let r = py
            .detach(|| {
                self.inner
                    .projection_contrast_pullback(&ids, target, alt, spec)
            })
            .map_err(error)?;
        projection_probe::report(py, r)
    }

    #[pyo3(signature=(prompt_ids,target_id,alternative_id,*,layer,site,bias_f32le,weight_f32le,basis="diagonal"))]
    fn projection_perturbed_contrast(
        &self,
        py: Python<'_>,
        prompt_ids: &Bound<'_, PyAny>,
        target_id: &Bound<'_, PyAny>,
        alternative_id: &Bound<'_, PyAny>,
        layer: &Bound<'_, PyAny>,
        site: &str,
        bias_f32le: &Bound<'_, PyBytes>,
        weight_f32le: &Bound<'_, PyBytes>,
        basis: &str,
    ) -> PyResult<Py<PyAny>> {
        let (target, alt, spec) =
            projection_probe::spec(target_id, alternative_id, layer, site, basis)?;
        let steps = prompt_ids.len()?;
        let (nb, nw) = self
            .inner
            .projection_probe_dimensions(steps, target, alt, spec, false)
            .map_err(error)?;
        let (b, w) = (bias_f32le.as_bytes(), weight_f32le.as_bytes());
        if (b.len(), w.len()) != (nb * 4, nw * 4) {
            return Err(error(
                "projection f32le buffer lengths differ from the admitted basis",
            ));
        }
        let (b, w) = (projection_probe::decode(b)?, projection_probe::decode(w)?);
        let ids = tokens(prompt_ids, steps)?;
        let r = py
            .detach(|| {
                self.inner
                    .projection_perturbed_contrast(&ids, target, alt, spec, &b, &w)
            })
            .map_err(error)?;
        crate::json::json_to_py(py, &serde_json::to_value(r).map_err(error)?)
    }

    /// Raw target vs. alternative logits on the prompt alone, not a forced two-way choice.
    fn next_token_contrast(
        &self,
        py: Python<'_>,
        prompt_ids: &Bound<'_, PyAny>,
        target_id: &Bound<'_, PyAny>,
        alternative_id: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let target_id = dimension(target_id)?;
        let alternative_id = dimension(alternative_id)?;
        let count = self
            .inner
            .next_token_contrast_input_count(prompt_ids.len()?, target_id, alternative_id)
            .map_err(error)?;
        let ids = tokens(prompt_ids, count)?;
        let scores = py
            .detach(|| {
                self.inner
                    .next_token_contrast(&ids, target_id, alternative_id)
            })
            .map_err(error)?;
        crate::json::json_to_py(py, &serde_json::to_value(scores).map_err(error)?)
    }

    /// Rank the target against all vocabulary IDs on one prompt without exporting the logit vector.
    fn next_token_score(
        &self,
        py: Python<'_>,
        prompt_ids: &Bound<'_, PyAny>,
        target_id: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let target = dimension(target_id)?;
        let count = self
            .inner
            .next_token_score_input_count(prompt_ids.len()?, target)
            .map_err(error)?;
        let ids = tokens(prompt_ids, count)?;
        let score = py
            .detach(|| self.inner.next_token_score(&ids, target))
            .map_err(error)?;
        target_score_dict(py, score)
    }

    #[pyo3(signature = (windows, shape, *, labels=None, label_smoothing=None, repetition=None))]
    fn next_token_loss(
        &self,
        py: Python<'_>,
        windows: &Bound<'_, PyAny>,
        shape: &Bound<'_, PyAny>,
        labels: Option<&Bound<'_, PyAny>>,
        label_smoothing: Option<&Bound<'_, PyAny>>,
        repetition: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<f32> {
        let objective = loss_objective(label_smoothing, repetition)?;
        let shape = self::shape(shape)?;
        self.inner
            .validate_objective_shape(shape, objective)
            .map_err(error)?;
        let count = self
            .inner
            .loss_objective_input_count(shape, labels.is_some())
            .map_err(error)?;
        let windows = tokens(windows, count)?;
        let labels = loss_labels(labels, count)?;
        py.detach(|| {
            self.inner
                .next_token_loss_with_objective(&windows, shape, labels.as_deref(), objective)
        })
        .map_err(error)
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    qkv::register(module)?;
    qkv_session::register(module)?;
    module.add_function(wrap_pyfunction!(score_token_logits_f32le, module)?)?;
    module.add_class::<supervision::PyCausalTargetAlignment>()?;
    module.add_class::<supervision::PyCausalCompletion>()?;
    module.add_class::<PyNativeCausalLm>()?;
    module.add_class::<scalar_output::PyScalarOutput>()?;
    output_margin::register(module)?;
    module.add_class::<PyNativeCausalLmAccumulator>()?;
    module.add_class::<PyNativeCausalLmGradients>()?;
    module.add_class::<PyNativeCausalLmLoraGradients>()?;
    evaluation::register(module)?;
    training::register(module)?;
    session::register(module)?;
    generation::register(module)?;
    context::register(module)?;
    Ok(())
}
