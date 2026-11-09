use super::*;
use st_nn::causal_lm::{
    CausalLmOutputContrastBank, CausalLmOutputMarginDirection, CausalLmOutputMarginRule,
    MAX_COMMON_GEOMETRY_GROUPS, MAX_SCALAR_OUTPUT_PAIRS,
};

#[pyclass(
    name = "NativeCausalLmOutputContrastBank",
    module = "spiraltorch",
    frozen
)]
pub(super) struct PyOutputContrastBank(CausalLmOutputContrastBank);
#[pyclass(
    name = "NativeCausalLmOutputMarginDirection",
    module = "spiraltorch",
    frozen
)]
pub(super) struct PyOutputDirection(CausalLmOutputMarginDirection);

pub(super) fn build(
    py: Python<'_>,
    m: &NativeCausalLm,
    rows: &Bound<'_, PyAny>,
    t: usize,
    a: usize,
    layer: usize,
    left: Option<&Bound<'_, PyBytes>>,
) -> PyResult<PyOutputContrastBank> {
    let n = rows.len()?;
    m.output_contrast_dimensions(layer, n, t, a, left.is_some())
        .map_err(error)?;
    let left = left
        .map(|b| -> PyResult<Vec<f64>> {
            if b.as_bytes().len() != m.config().width * 8 {
                return Err(error(
                    "expected a width-sized f64 little-endian left factor",
                ));
            }
            let v: Vec<_> = b
                .as_bytes()
                .chunks_exact(8)
                .map(|x| f64::from_le_bytes(x.try_into().unwrap()))
                .collect();
            Ok(v)
        })
        .transpose()?;
    m.validate_output_left(left.as_deref()).map_err(error)?;
    let prompts = (0..n)
        .map(|i| rows.get_item(i))
        .collect::<PyResult<Vec<_>>>()?;
    let steps = prompts
        .iter()
        .map(|r| r.len())
        .collect::<PyResult<Vec<_>>>()?;
    m.validate_output_roster_steps(layer, &steps, t, a, left.is_some())
        .map_err(error)?;
    let ids = prompts
        .iter()
        .zip(steps)
        .map(|(r, n)| tokens(r, n))
        .collect::<PyResult<Vec<_>>>()?;
    py.detach(|| m.output_contrast_bank(layer, &ids, t, a, left.as_deref()))
        .map(PyOutputContrastBank)
        .map_err(error)
}

#[pymethods]
impl PyOutputContrastBank {
    fn __len__(&self) -> usize {
        self.0.contrasts().len()
    }
    fn contrasts(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::json::json_to_py(
            py,
            &serde_json::to_value(self.0.contrasts()).map_err(error)?,
        )
    }
    fn gradient_f64le<'py>(
        &self,
        py: Python<'py>,
        index: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        projection_probe::f64le(py, self.0.gradient(dimension(index)?).map_err(error)?)
    }
    #[pyo3(signature=(pairs,groups,*,rule="common_cosine"))]
    fn direction(
        &self,
        py: Python<'_>,
        pairs: &Bound<'_, PyAny>,
        groups: &Bound<'_, PyAny>,
        rule: &str,
    ) -> PyResult<PyOutputDirection> {
        let rule = match rule {
            "equal_mean" => CausalLmOutputMarginRule::EqualMean,
            "common_cosine" => CausalLmOutputMarginRule::CommonCosine,
            _ => return Err(error("rule must be equal_mean or common_cosine")),
        };
        let n = pairs.len()?;
        if !(1..=MAX_SCALAR_OUTPUT_PAIRS).contains(&n) || groups.len()? != n {
            return Err(error("invalid pair/group roster"));
        }
        let mut labels = Vec::with_capacity(n);
        let mut indices = Vec::with_capacity(n);
        for i in 0..n {
            let g = dimension(&groups.get_item(i)?)?;
            if g >= MAX_COMMON_GEOMETRY_GROUPS {
                return Err(error("group index out of bounds"));
            }
            labels.push(g);
            let pair = pairs.get_item(i)?;
            if pair.len()? != 2 {
                return Err(error("expected a positive/negative index pair"));
            }
            indices.push([
                dimension(&pair.get_item(0)?)?,
                dimension(&pair.get_item(1)?)?,
            ]);
        }
        py.detach(|| self.0.direction(&indices, &labels, rule))
            .map(PyOutputDirection)
            .map_err(error)
    }
}

#[pymethods]
impl PyOutputDirection {
    fn certificate(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::json::json_to_py(py, &self.0.certificate())
    }
    fn coordinates_f64le<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyBytes>>> {
        self.0
            .coordinates()
            .map(|v| projection_probe::f64le(py, v))
            .transpose()
    }
    fn left_f64le<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyBytes>>> {
        self.0
            .left()
            .map(|v| projection_probe::f64le(py, v))
            .transpose()
    }
    fn scalar_adapter(
        &self,
        py: Python<'_>,
        model: &PyNativeCausalLm,
    ) -> PyResult<scalar_output::PyScalarOutput> {
        py.detach(|| self.0.scalar_adapter(&model.inner))
            .map(scalar_output::PyScalarOutput)
            .map_err(error)
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOutputContrastBank>()?;
    m.add_class::<PyOutputDirection>()
}
