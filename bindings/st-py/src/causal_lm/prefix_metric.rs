//! Python owns the prompt schedule only; never accept caller-supplied moments or parameter gradients.
use super::*;
use st_nn::causal_lm::{
    CausalLmPrefixResidualMetric, PrefixMetricRule, MAX_GEOMETRY_COHORT_MEMBERS,
};

#[pyclass(name = "NativePrefixResidualMetric", module = "spiraltorch", frozen)]
pub(super) struct PyPrefixResidualMetric {
    inner: CausalLmPrefixResidualMetric,
}

#[pymethods]
impl PyPrefixResidualMetric {
    fn report(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::json::json_to_py(py, &serde_json::to_value(&self.inner).map_err(error)?)
    }
}

impl PyNativeCausalLm {
    pub(super) fn calibrate_prefix_metric_impl(
        &self,
        py: Python<'_>,
        prompts: &Bound<'_, PyAny>,
    ) -> PyResult<PyPrefixResidualMetric> {
        let n = prompts.len()?;
        if !(1..=MAX_GEOMETRY_COHORT_MEMBERS).contains(&n) {
            return Err(error("expected 1 to 256 calibration prompts"));
        }
        let rows = (0..n)
            .map(|i| prompts.get_item(i))
            .collect::<PyResult<Vec<_>>>()?;
        let lengths = rows.iter().map(|r| r.len()).collect::<PyResult<Vec<_>>>()?;
        self.inner
            .validate_prefix_calibration(&lengths)
            .map_err(error)?;
        let ids = rows
            .iter()
            .zip(lengths)
            .map(|(r, n)| tokens(r, n))
            .collect::<PyResult<Vec<_>>>()?;
        let inner = py
            .detach(|| self.inner.calibrate_prefix_residual_metric(&ids))
            .map_err(error)?;
        Ok(PyPrefixResidualMetric { inner })
    }

    pub(super) fn step_prefix_metric_impl(
        &self,
        py: Python<'_>,
        metric: &PyPrefixResidualMetric,
        gradients: &Bound<'_, PyAny>,
        rule: &str,
        target_rms: &Bound<'_, PyAny>,
        max_parameter_l2: &Bound<'_, PyAny>,
    ) -> PyResult<(Self, Py<PyAny>)> {
        reject_bool(target_rms)?;
        reject_bool(max_parameter_l2)?;
        let (target, cap) = (
            target_rms.extract::<f64>()?,
            max_parameter_l2.extract::<f64>()?,
        );
        let rule = match rule {
            "uniform" => PrefixMetricRule::Uniform,
            "diagonal" => PrefixMetricRule::Diagonal,
            "cyclic" => PrefixMetricRule::Cyclic,
            _ => return Err(error("expected uniform, diagonal or cyclic")),
        };
        let n = gradients.len()?;
        self.inner
            .validate_geometry_cohort_ray(n, target)
            .map_err(error)?;
        let owners = groups::members(gradients, n)?;
        let gs = owners.iter().map(|o| &o.get().inner).collect::<Vec<_>>();
        let (inner, receipt) = py
            .detach(|| {
                self.inner
                    .step_prefix_residual_rms(&metric.inner, &gs, rule, target, cap)
            })
            .map_err(error)?;
        Ok((
            Self { inner },
            crate::json::json_to_py(py, &serde_json::to_value(receipt).map_err(error)?)?,
        ))
    }
}
