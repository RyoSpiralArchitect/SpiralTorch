use super::{dimension, error};
use pyo3::prelude::*;
use st_nn::language_supervision::CausalTargetAlignment;

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
        let ids = (0..n)
            .map(|i| Ok(dimension(&token_ids.get_item(i)?)? as u32))
            .collect::<PyResult<Vec<_>>>()?;
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
}
