//! Bounded transport; the Rust projection owns all update and retention decisions.
mod session;

use super::*;
use st_nn::causal_lm::{
    CausalLmConditionalAnswerConfig, CausalLmConditionalAnswerPrefixBank,
    CausalLmConditionalMarginRetention, CausalLmProjectionSite, CausalLmResult,
    CausalLmRetainedConditionalAnswerStep, NativeCausalLmContext, NativeCausalLmRankOneProjection,
    MAX_CONDITIONAL_BANK_ROWS, MAX_OUTPUT_CONTRAST_ROWS,
};

#[pyclass(name = "NativeCausalLmRankOneProjection", module = "spiraltorch")]
pub(super) struct PyRankOne(NativeCausalLmRankOneProjection);

#[pyclass(
    name = "NativeCausalLmConditionalPrefixBank",
    module = "spiraltorch",
    frozen
)]
pub(super) struct PyConditionalBank(CausalLmConditionalAnswerPrefixBank);

#[pyclass(name = "NativeCausalLmConditionalRetention", module = "spiraltorch")]
pub(super) struct PyConditionalRetention(CausalLmConditionalMarginRetention);

fn real(value: &Bound<'_, PyAny>) -> PyResult<f64> {
    reject_bool(value)?;
    value.extract()
}

fn config(
    length: &Bound<'_, PyAny>,
    max_trials: Option<&Bound<'_, PyAny>>,
    mass_inward_cosine: Option<&Bound<'_, PyAny>>,
) -> PyResult<CausalLmConditionalAnswerConfig> {
    let config = CausalLmConditionalAnswerConfig {
        length: real(length)?,
        max_trials: limit(max_trials, 2)?,
        mass_inward_cosine: mass_inward_cosine.map(real).transpose()?,
    };
    config.validate().map_err(error)?;
    Ok(config)
}

fn roster_size(value: &Bound<'_, PyAny>, cap: usize) -> PyResult<usize> {
    let n = value.len()?;
    if !(1..=cap).contains(&n) {
        return Err(error("invalid conditional roster length"));
    }
    Ok(n)
}

enum PromptUse {
    Training(CausalLmConditionalAnswerConfig),
    Forward,
    Retention(Option<f64>),
}

type AnswerPrompts = (Vec<Vec<usize>>, Vec<[usize; 2]>);

fn paired_prompts(
    projection: &NativeCausalLmRankOneProjection,
    value: &Bound<'_, PyAny>,
    answer_pairs: &Bound<'_, PyAny>,
    usage: PromptUse,
) -> PyResult<AnswerPrompts> {
    let n = roster_size(value, MAX_OUTPUT_CONTRAST_ROWS)?;
    let pairs = pairs(answer_pairs, n)?;
    projection
        .admit_conditional_answer_pairs(&pairs)
        .map_err(error)?;
    let prompts = read_prompts(value, n, |counts| match usage {
        PromptUse::Training(config) => {
            projection.conditional_answer_step_input_count(counts, config)
        }
        PromptUse::Forward => projection.conditional_answer_input_count(counts, false),
        PromptUse::Retention(drop) => projection.conditional_retention_input_count(counts, drop),
    })?;
    Ok((prompts, pairs))
}

fn read_prompts(
    value: &Bound<'_, PyAny>,
    n: usize,
    admit: impl FnOnce(&[usize]) -> CausalLmResult<usize>,
) -> PyResult<Vec<Vec<usize>>> {
    let rows = (0..n)
        .map(|i| value.get_item(i))
        .collect::<PyResult<Vec<_>>>()?;
    let counts = rows
        .iter()
        .map(|row| row.len())
        .collect::<PyResult<Vec<_>>>()?;
    admit(&counts).map_err(error)?;
    rows.iter()
        .zip(counts)
        .map(|(row, n)| tokens(row, n))
        .collect()
}

fn pairs(value: &Bound<'_, PyAny>, count: usize) -> PyResult<Vec<[usize; 2]>> {
    if value.len()? != count {
        return Err(error("expected one answer pair per conditional row"));
    }
    (0..count)
        .map(|i| {
            let pair = value.get_item(i)?;
            if pair.len()? != 2 {
                return Err(error("expected [gold, alternative]"));
            }
            Ok([
                dimension(&pair.get_item(0)?)?,
                dimension(&pair.get_item(1)?)?,
            ])
        })
        .collect()
}

pub(super) fn build(
    py: Python<'_>,
    model: &NativeCausalLm,
    layer: usize,
    site: &str,
    left: &Bound<'_, PyBytes>,
    right: Option<&Bound<'_, PyBytes>>,
) -> PyResult<PyRankOne> {
    let site = site.parse::<CausalLmProjectionSite>().map_err(error)?;
    let width = model.config().width;
    let decode = |v: &Bound<'_, PyBytes>| {
        let (words, tail) = v.as_bytes().as_chunks::<8>();
        if !tail.is_empty() || words.len() != width {
            return Err(error("expected width-sized f64 little-endian factor"));
        }
        Ok(words
            .iter()
            .copied()
            .map(f64::from_le_bytes)
            .collect::<Vec<_>>())
    };
    let left = decode(left)?;
    let right = right
        .map(decode)
        .transpose()?
        .unwrap_or_else(|| vec![0.; width]);
    py.detach(|| NativeCausalLmRankOneProjection::new(model, layer, site, &left, &right))
        .map(PyRankOne)
        .map_err(error)
}

pub(super) fn restore(
    py: Python<'_>,
    model: &NativeCausalLm,
    payload: &Bound<'_, PyAny>,
    expected_id: &Bound<'_, PyAny>,
    rows: &Bound<'_, PyAny>,
    answer_pairs: &Bound<'_, PyAny>,
) -> PyResult<(PyRankOne, PyConditionalRetention)> {
    let expected_id = session_input::pin(expected_id)?;
    let payload = session_input::payload_with_limit(
        payload,
        NativeCausalLmRankOneProjection::MAX_RETENTION_CHECKPOINT_BYTES,
    )?;
    let bytes = payload.as_bytes();
    let plan = py
        .detach(|| {
            NativeCausalLmRankOneProjection::admit_retained_checkpoint(model, bytes, &expected_id)
        })
        .map_err(error)?;
    let count = plan.roster_len();
    if rows.len()? != count || answer_pairs.len()? != count {
        return Err(error(
            "conditional retention checkpoint roster length changed",
        ));
    }
    let pairs = pairs(answer_pairs, count)?;
    let ids = read_prompts(rows, count, |lengths| plan.roster_input_count(lengths))?;
    py.detach(|| plan.restore(&ids, &pairs))
        .map(|(state, guard)| (PyRankOne(state), PyConditionalRetention(guard)))
        .map_err(error)
}

enum Update<'a> {
    Raw(&'a [Vec<usize>]),
    Indexed(&'a CausalLmConditionalAnswerPrefixBank, &'a [usize]),
}

impl PyRankOne {
    fn update(
        &mut self,
        py: Python<'_>,
        update: Update<'_>,
        pairs: &[[usize; 2]],
        config: CausalLmConditionalAnswerConfig,
        mut retention: Option<PyRefMut<'_, PyConditionalRetention>>,
    ) -> PyResult<Py<PyAny>> {
        let guard = retention.as_deref_mut().map(|r| &mut r.0);
        let result = py
            .detach(|| -> CausalLmResult<_> {
                Ok(match (update, guard) {
                    (Update::Raw(ids), Some(guard)) => self
                        .0
                        .step_conditional_answers_retaining(ids, pairs, config, guard)?,
                    (Update::Indexed(bank, ids), Some(guard)) => {
                        self.0.step_indexed_conditional_answers_retaining(
                            bank, ids, pairs, config, guard,
                        )?
                    }
                    (Update::Raw(ids), None) => CausalLmRetainedConditionalAnswerStep {
                        step: self.0.step_conditional_answers(ids, pairs, config)?,
                        retention: Vec::new(),
                    },
                    (Update::Indexed(bank, ids), None) => CausalLmRetainedConditionalAnswerStep {
                        step: self
                            .0
                            .step_indexed_conditional_answers(bank, ids, pairs, config)?,
                        retention: Vec::new(),
                    },
                })
            })
            .map_err(error)?;
        as_py(py, result)
    }
}

fn as_py(py: Python<'_>, result: impl serde::Serialize) -> PyResult<Py<PyAny>> {
    crate::json::json_to_py(py, &serde_json::to_value(result).map_err(error)?)
}

#[pymethods]
impl PyRankOne {
    #[getter]
    fn projection_id(&self) -> String {
        self.0.projection_id()
    }

    fn right_f64le<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        projection_probe::f64le(py, self.0.right())
    }

    fn checkpoint_with_retention<'py>(
        &self,
        py: Python<'py>,
        retention: &PyConditionalRetention,
    ) -> PyResult<(Bound<'py, PyBytes>, String)> {
        let (payload, pin) = py
            .detach(|| self.0.checkpoint_with_retention(&retention.0))
            .map_err(error)?;
        Ok((PyBytes::new(py, &payload), pin))
    }

    #[pyo3(signature=(rows, *, max_prefix_values=None))]
    fn prepare_prefix_bank(
        &self,
        py: Python<'_>,
        rows: &Bound<'_, PyAny>,
        max_prefix_values: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyConditionalBank> {
        let cap = limit(max_prefix_values, 1 << 20)?;
        let count = roster_size(rows, MAX_CONDITIONAL_BANK_ROWS)?;
        let ids = read_prompts(rows, count, |counts| {
            self.0.conditional_answer_bank_values(counts, cap)
        })?;
        py.detach(|| self.0.prepare_conditional_answer_prefix_bank(&ids, cap))
            .map(PyConditionalBank)
            .map_err(error)
    }

    #[pyo3(signature=(rows, answer_pairs, *, max_mean_mass_drop=None))]
    fn prepare_retention(
        &self,
        py: Python<'_>,
        rows: &Bound<'_, PyAny>,
        answer_pairs: &Bound<'_, PyAny>,
        max_mean_mass_drop: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyConditionalRetention> {
        let drop = max_mean_mass_drop.map(real).transpose()?;
        let (ids, pairs) = paired_prompts(&self.0, rows, answer_pairs, PromptUse::Retention(drop))?;
        py.detach(|| match drop {
            Some(d) => self
                .0
                .prepare_conditional_margin_and_mass_retention(&ids, &pairs, d),
            None => self.0.prepare_conditional_margin_retention(&ids, &pairs),
        })
        .map(PyConditionalRetention)
        .map_err(error)
    }

    #[pyo3(signature=(rows, answer_pairs, *, length, max_trials=None, mass_inward_cosine=None, retention=None))]
    #[allow(clippy::too_many_arguments)]
    fn step(
        &mut self,
        py: Python<'_>,
        rows: &Bound<'_, PyAny>,
        answer_pairs: &Bound<'_, PyAny>,
        length: &Bound<'_, PyAny>,
        max_trials: Option<&Bound<'_, PyAny>>,
        mass_inward_cosine: Option<&Bound<'_, PyAny>>,
        retention: Option<PyRefMut<'_, PyConditionalRetention>>,
    ) -> PyResult<Py<PyAny>> {
        let config = config(length, max_trials, mass_inward_cosine)?;
        let (ids, pairs) =
            paired_prompts(&self.0, rows, answer_pairs, PromptUse::Training(config))?;
        self.update(py, Update::Raw(&ids), &pairs, config, retention)
    }

    #[pyo3(signature=(bank, indices, answer_pairs, *, length, max_trials=None, mass_inward_cosine=None, retention=None))]
    #[allow(clippy::too_many_arguments)]
    fn step_indexed(
        &mut self,
        py: Python<'_>,
        bank: &PyConditionalBank,
        indices: &Bound<'_, PyAny>,
        answer_pairs: &Bound<'_, PyAny>,
        length: &Bound<'_, PyAny>,
        max_trials: Option<&Bound<'_, PyAny>>,
        mass_inward_cosine: Option<&Bound<'_, PyAny>>,
        retention: Option<PyRefMut<'_, PyConditionalRetention>>,
    ) -> PyResult<Py<PyAny>> {
        let config = config(length, max_trials, mass_inward_cosine)?;
        let count = roster_size(indices, MAX_OUTPUT_CONTRAST_ROWS)?;
        let pairs = pairs(answer_pairs, count)?;
        self.0
            .admit_conditional_answer_pairs(&pairs)
            .map_err(error)?;
        let ids = tokens(indices, count)?;
        self.update(
            py,
            Update::Indexed(&bank.0, &ids),
            &pairs,
            config,
            retention,
        )
    }

    fn score_pairs(
        &self,
        py: Python<'_>,
        rows: &Bound<'_, PyAny>,
        answer_pairs: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let (ids, pairs) = paired_prompts(&self.0, rows, answer_pairs, PromptUse::Forward)?;
        let scores = py
            .detach(|| self.0.score_answer_pairs(&ids, &pairs))
            .map_err(error)?;
        as_py(py, scores)
    }

    #[pyo3(signature=(prompt, *, max_cache_values=None))]
    fn prefill(
        &self,
        py: Python<'_>,
        prompt: &Bound<'_, PyAny>,
        max_cache_values: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<qkv::PyProjectionPrefix> {
        let cap = limit(
            max_cache_values,
            NativeCausalLmContext::DEFAULT_MAX_CACHE_VALUES,
        )?;
        let n = self
            .0
            .prefill_input_count(prompt.len()?, cap)
            .map_err(error)?;
        let ids = tokens(prompt, n)?;
        py.detach(|| self.0.prefill(ids, cap))
            .map(|inner| qkv::PyProjectionPrefix { inner })
            .map_err(error)
    }
}

#[pymethods]
impl PyConditionalBank {
    fn __len__(&self) -> usize {
        self.0.len()
    }
    #[getter]
    fn prefix_values(&self) -> usize {
        self.0.prefix_values()
    }
}

#[pymethods]
impl PyConditionalRetention {
    #[getter]
    fn protected_count(&self) -> usize {
        self.0.protected_count()
    }
    #[getter]
    fn baseline_margins(&self) -> Vec<f64> {
        self.0.baseline_margins().to_vec()
    }
    #[getter]
    fn baseline_mean_pair_mass(&self) -> Option<f64> {
        self.0.baseline_mean_pair_mass()
    }
    #[getter]
    fn minimum_mean_pair_mass(&self) -> Option<f64> {
        self.0.minimum_mean_pair_mass()
    }

    #[getter]
    fn minimum_log_mean_pair_mass(&self) -> Option<f64> {
        self.0.minimum_log_mean_pair_mass()
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRankOne>()?;
    m.add_class::<PyConditionalBank>()?;
    m.add_class::<PyConditionalRetention>()?;
    session::register(m)
}
