use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::Bound;

#[cfg(feature = "rec")]
use crate::tensor::{tensor_err_to_py, PyTensor};
#[cfg(feature = "rec")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "rec")]
use st_kdsl::{
    compile_query, Err as KdslError, Filter, OrderDirection, QueryPlan as KdslQueryPlan,
};
#[cfg(feature = "rec")]
use st_rec::{RatingTriple, RecEpochReport, Recommendation, SpiralRecError, SpiralRecommender};

#[cfg(feature = "rec")]
fn rec_err_to_py(err: SpiralRecError) -> PyErr {
    match err {
        SpiralRecError::Tensor(err) => tensor_err_to_py(err),
        SpiralRecError::OutOfBoundsRating { .. } | SpiralRecError::EmptyBatch => {
            PyValueError::new_err(err.to_string())
        }
    }
}

#[cfg(feature = "rec")]
#[pyclass(name = "QueryPlan", module = "spiraltorch.rec")]
pub(crate) struct PyQueryPlan {
    source: String,
    inner: KdslQueryPlan,
}

#[cfg(feature = "rec")]
impl PyQueryPlan {
    fn from_query(source: String, inner: KdslQueryPlan) -> Self {
        Self { source, inner }
    }

    pub(crate) fn plan(&self) -> &KdslQueryPlan {
        &self.inner
    }
}

#[cfg(feature = "rec")]
#[pymethods]
impl PyQueryPlan {
    #[new]
    pub fn new(query: String) -> PyResult<Self> {
        let plan = compile_query(&query)
            .map_err(|err: KdslError| PyValueError::new_err(err.to_string()))?;
        Ok(Self::from_query(query, plan))
    }

    pub fn query(&self) -> &str {
        &self.source
    }

    pub fn selects(&self) -> Vec<String> {
        self.inner.selects.clone()
    }

    pub fn limit(&self) -> Option<usize> {
        self.inner.limit
    }

    pub fn order(&self) -> Option<(String, String)> {
        self.inner.order.clone().map(|(column, direction)| {
            let dir = match direction {
                OrderDirection::Asc => "asc",
                OrderDirection::Desc => "desc",
            };
            (column, dir.to_string())
        })
    }

    pub fn filters(&self) -> Vec<(String, String, f64)> {
        self.inner
            .filters
            .iter()
            .map(|filter| match filter {
                Filter::Eq(column, value) => (column.clone(), "=".to_string(), *value),
                Filter::Neq(column, value) => (column.clone(), "!=".to_string(), *value),
                Filter::Gt(column, value) => (column.clone(), ">".to_string(), *value),
                Filter::Lt(column, value) => (column.clone(), "<".to_string(), *value),
                Filter::Ge(column, value) => (column.clone(), ">=".to_string(), *value),
                Filter::Le(column, value) => (column.clone(), "<=".to_string(), *value),
            })
            .collect()
    }
}

#[cfg(feature = "rec")]
#[pyclass(name = "RecEpochReport", module = "spiraltorch.rec")]
#[derive(Clone)]
pub(crate) struct PyRecEpochReport {
    #[pyo3(get)]
    rmse: f32,
    #[pyo3(get)]
    samples: usize,
    #[pyo3(get)]
    regularization_penalty: f32,
}

#[cfg(feature = "rec")]
impl From<RecEpochReport> for PyRecEpochReport {
    fn from(value: RecEpochReport) -> Self {
        Self {
            rmse: value.rmse,
            samples: value.samples,
            regularization_penalty: value.regularization_penalty,
        }
    }
}

#[cfg(feature = "rec")]
#[pymethods]
impl PyRecEpochReport {
    fn __repr__(&self) -> String {
        format!(
            "RecEpochReport(rmse={:.4}, samples={}, regularization_penalty={:.4})",
            self.rmse, self.samples, self.regularization_penalty
        )
    }
}

#[cfg(feature = "rec")]
#[pyclass(name = "Recommender", module = "spiraltorch.rec")]
pub(crate) struct PyRecommender {
    inner: SpiralRecommender,
}

#[cfg(feature = "rec")]
fn convert_ratings(ratings: Vec<(usize, usize, f32)>) -> Vec<RatingTriple> {
    ratings
        .into_iter()
        .map(|(user, item, rating)| RatingTriple::new(user, item, rating))
        .collect()
}

#[cfg(feature = "rec")]
#[pymethods]
impl PyRecommender {
    #[new]
    #[pyo3(signature = (users, items, factors, learning_rate, regularization, curvature))]
    pub fn new(
        users: usize,
        items: usize,
        factors: usize,
        learning_rate: f32,
        regularization: f32,
        curvature: f32,
    ) -> PyResult<Self> {
        let inner = SpiralRecommender::new(
            users,
            items,
            factors,
            learning_rate,
            regularization,
            curvature,
        )
        .map_err(rec_err_to_py)?;
        Ok(Self { inner })
    }

    pub fn predict(&self, user: usize, item: usize) -> PyResult<f32> {
        self.inner.predict(user, item).map_err(rec_err_to_py)
    }

    pub fn train_epoch(&mut self, ratings: Vec<(usize, usize, f32)>) -> PyResult<PyRecEpochReport> {
        let triples = convert_ratings(ratings);
        self.inner
            .train_epoch(&triples)
            .map(PyRecEpochReport::from)
            .map_err(rec_err_to_py)
    }

    #[pyo3(signature = (user, k, exclude=None))]
    pub fn recommend_top_k(
        &self,
        user: usize,
        k: usize,
        exclude: Option<Vec<usize>>,
    ) -> PyResult<Vec<(usize, f32)>> {
        self.inner
            .recommend_top_k(user, k, exclude.as_deref())
            .map(|recs| {
                recs.into_iter()
                    .map(|Recommendation { item, score }| (item, score))
                    .collect()
            })
            .map_err(rec_err_to_py)
    }

    #[pyo3(signature = (user, plan, exclude=None))]
    pub fn recommend_query(
        &self,
        user: usize,
        plan: &PyQueryPlan,
        exclude: Option<Vec<usize>>,
    ) -> PyResult<Vec<std::collections::BTreeMap<String, f64>>> {
        self.inner
            .recommend_with_query(user, plan.plan(), exclude.as_deref())
            .map_err(rec_err_to_py)
    }

    pub fn user_embedding(&self, user: usize) -> PyResult<PyTensor> {
        self.inner
            .user_embedding(user)
            .map(PyTensor::from_tensor)
            .map_err(rec_err_to_py)
    }

    pub fn item_embedding(&self, item: usize) -> PyResult<PyTensor> {
        self.inner
            .item_embedding(item)
            .map(PyTensor::from_tensor)
            .map_err(rec_err_to_py)
    }

    #[getter]
    pub fn users(&self) -> usize {
        self.inner.users()
    }

    #[getter]
    pub fn items(&self) -> usize {
        self.inner.items()
    }

    #[getter]
    pub fn factors(&self) -> usize {
        self.inner.factors()
    }
}

#[cfg(feature = "rec")]
fn register_impl(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    let module = PyModule::new_bound(py, "rec")?;
    module.add("__doc__", "SpiralTorch recommendation toolkit")?;
    module.add_class::<PyQueryPlan>()?;
    module.add_class::<PyRecEpochReport>()?;
    module.add_class::<PyRecommender>()?;

    let query_plan = module.getattr("QueryPlan")?;
    let rec_epoch_report = module.getattr("RecEpochReport")?;
    let recommender = module.getattr("Recommender")?;

    module.add(
        "__all__",
        vec!["QueryPlan", "RecEpochReport", "Recommender"],
    )?;
    parent.add_submodule(&module)?;
    parent.add("QueryPlan", query_plan)?;
    parent.add("RecEpochReport", rec_epoch_report)?;
    parent.add("Recommender", recommender)?;
    Ok(())
}

#[cfg(not(feature = "rec"))]
fn register_impl(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    let module = PyModule::new_bound(py, "rec")?;
    module.add("__doc__", "SpiralTorch recommendation toolkit")?;
    parent.add_submodule(&module)?;
    Ok(())
}

pub(crate) fn register(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    register_impl(py, parent)
}
