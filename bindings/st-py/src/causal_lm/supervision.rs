use super::{dimension, error};
use pyo3::prelude::*;
use st_nn::language_supervision::{CausalCompletion, CausalTargetAlignment};

fn extract_ids(value: &Bound<'_, PyAny>, count: usize) -> PyResult<Vec<u32>> {
    (0..count)
        .map(|i| Ok(dimension(&value.get_item(i)?)? as u32))
        .collect()
}

fn pairs(value: &Bound<'_, PyAny>, count: usize) -> PyResult<Vec<[u32; 2]>> {
    (0..count)
        .map(|i| {
            let row = value.get_item(i)?;
            if row.len()? != 2 {
                return Err(error("expected a half-open [start, end] interval"));
            }
            Ok([
                dimension(&row.get_item(0)?)? as u32,
                dimension(&row.get_item(1)?)? as u32,
            ])
        })
        .collect()
}

#[pyclass(name = "CausalTargetAlignment", module = "spiraltorch", frozen)]
pub struct PyCausalTargetAlignment {
    inner: CausalTargetAlignment,
}

#[pymethods]
impl PyCausalTargetAlignment {
    #[classattr]
    const MAX_SOURCE_UNITS: u32 = CausalTargetAlignment::MAX_SOURCE_UNITS;
    #[classattr]
    const MAX_TOKENS: usize = CausalTargetAlignment::MAX_TOKENS;
    #[classattr]
    const MAX_TARGET_SPANS: usize = CausalTargetAlignment::MAX_TARGET_SPANS;

    #[new]
    fn new(
        py: Python<'_>,
        source_length: &Bound<'_, PyAny>,
        token_ids: &Bound<'_, PyAny>,
        token_offsets: &Bound<'_, PyAny>,
        target_spans: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let length = dimension(source_length)? as u32;
        let (n, m, k) = (token_ids.len()?, token_offsets.len()?, target_spans.len()?);
        CausalTargetAlignment::admit_counts(length, n, m, k).map_err(error)?;
        let targets = pairs(target_spans, k)?;
        let offsets = pairs(token_offsets, m)?;
        let ids = extract_ids(token_ids, n)?;
        py.detach(|| CausalTargetAlignment::new(length, ids, &offsets, &targets))
            .map(|inner| Self { inner })
            .map_err(error)
    }
    #[getter]
    fn token_ids(&self) -> Vec<u32> {
        self.inner.token_ids().to_vec()
    }
    #[getter]
    fn labels(&self) -> Vec<i64> {
        self.inner.labels().to_vec()
    }
    #[getter]
    fn runs(&self) -> Vec<[u32; 4]> {
        self.inner.runs().to_vec()
    }

    fn completion(
        &self,
        py: Python<'_>,
        prompt_ids: &Bound<'_, PyAny>,
        prompt_answer_ids: &Bound<'_, PyAny>,
    ) -> PyResult<PyCausalCompletion> {
        let (p, a) = (prompt_ids.len()?, prompt_answer_ids.len()?);
        CausalCompletion::admit_counts(p, a, self.inner.token_ids().len()).map_err(error)?;
        let (prompt, answer) = (
            extract_ids(prompt_ids, p)?,
            extract_ids(prompt_answer_ids, a)?,
        );
        py.detach(|| CausalCompletion::from_alignment(&self.inner, &prompt, &answer))
            .map(|inner| PyCausalCompletion { inner })
            .map_err(error)
    }
}

#[pyclass(name = "CausalCompletion", module = "spiraltorch", frozen)]
pub struct PyCausalCompletion {
    inner: CausalCompletion,
}

#[pymethods]
impl PyCausalCompletion {
    #[new]
    fn new(
        py: Python<'_>,
        prompt_ids: &Bound<'_, PyAny>,
        prompt_answer_ids: &Bound<'_, PyAny>,
        training_ids: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let (p, a, t) = (
            prompt_ids.len()?,
            prompt_answer_ids.len()?,
            training_ids.len()?,
        );
        CausalCompletion::admit_counts(p, a, t).map_err(error)?;
        let (prompt, answer, train) = (
            extract_ids(prompt_ids, p)?,
            extract_ids(prompt_answer_ids, a)?,
            extract_ids(training_ids, t)?,
        );
        py.detach(|| CausalCompletion::from_tokenizations(&prompt, &answer, train))
            .map(|inner| Self { inner })
            .map_err(error)
    }
    #[getter]
    fn prompt_ids(&self) -> Vec<u32> {
        self.inner.prompt_ids().to_vec()
    }
    #[getter]
    fn answer_end(&self) -> usize {
        self.inner.answer_end()
    }
    fn training_document(&self) -> (Vec<u32>, Vec<i64>) {
        let (ids, labels) = self.inner.training_document();
        (ids.to_vec(), labels.to_vec())
    }
    fn evaluation_document(&self) -> (Vec<u32>, Vec<i64>) {
        let (ids, labels) = self.inner.evaluation_document();
        (ids.to_vec(), labels.to_vec())
    }
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(error)
    }
}
