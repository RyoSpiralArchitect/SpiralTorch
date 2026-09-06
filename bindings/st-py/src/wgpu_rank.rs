use pyo3::prelude::*;

#[cfg(feature = "wgpu")]
mod enabled {
    use super::*;
    use crate::tensor::PyTensor;
    use pyo3::{
        exceptions::{PyRuntimeError, PyValueError},
        types::PyDict,
    };
    use st_backend_wgpu::{
        rankk_exact_2ce::{
            resident::{ResidentRank, ResidentRankError},
            DispatchError, Kind, Plan,
        },
        runtime,
    };

    fn error(err: ResidentRankError) -> PyErr {
        match err {
            ResidentRankError::Runtime(_)
            | ResidentRankError::Matmul(st_backend_wgpu::resident_matmul::MatmulError::Runtime(
                _,
            ))
            | ResidentRankError::Dispatch(
                DispatchError::Runtime(_) | DispatchError::PipelineBuild(_),
            ) => PyRuntimeError::new_err(err.to_string()),
            _ => PyValueError::new_err(err.to_string()),
        }
    }

    #[pyclass(name = "WgpuRank", module = "spiraltorch.wgpu")]
    pub(crate) struct PyWgpuRank {
        inner: ResidentRank,
    }

    impl PyWgpuRank {
        fn from_kernel_plan(py: Python<'_>, plan: Plan, timestamp_queries: bool) -> PyResult<Self> {
            py.detach(move || {
                let inner = if timestamp_queries {
                    ResidentRank::request_profiled_blocking(plan).map_err(error)?
                } else {
                    let (runtime, _) =
                        runtime::ensure_default_runtime_blocking("python.resident.rank")
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                    ResidentRank::new(runtime, plan).map_err(error)?
                };
                Ok(Self { inner })
            })
        }
    }

    #[pymethods]
    impl PyWgpuRank {
        #[new]
        #[pyo3(signature = (kind, rows, cols, k, *, tile_cols=256, timestamp_queries=false))]
        fn new(
            py: Python<'_>,
            kind: &str,
            rows: u32,
            cols: u32,
            k: u32,
            tile_cols: u32,
            timestamp_queries: bool,
        ) -> PyResult<Self> {
            let kind = match kind {
                "topk" => Kind::TopK,
                "midk" => Kind::MidK,
                "bottomk" => Kind::BottomK,
                _ => return Err(PyValueError::new_err("kind must be topk, midk, or bottomk")),
            };
            let plan = Plan::try_new(kind, rows, cols, k, tile_cols)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            if plan.is_empty() {
                return Err(PyValueError::new_err(
                    "rank workspace dimensions must be positive",
                ));
            }
            Self::from_kernel_plan(py, plan, timestamp_queries)
        }

        #[cfg(feature = "kdsl")]
        #[staticmethod]
        #[pyo3(signature = (session, candidate_index, *, timestamp_queries=false))]
        fn from_adaptation(
            py: Python<'_>,
            session: &crate::rank_adaptation::PyRankAdaptationSession,
            candidate_index: usize,
            timestamp_queries: bool,
        ) -> PyResult<Self> {
            let spec = session.wgpu_resident_candidate(candidate_index)?;
            let plan = Plan::try_new(spec.kind, spec.rows, spec.cols, spec.k, spec.tile_cols)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Self::from_kernel_plan(py, plan, timestamp_queries)
        }

        #[cfg(not(feature = "kdsl"))]
        #[staticmethod]
        #[pyo3(signature = (session, candidate_index, *, timestamp_queries=false))]
        fn from_adaptation(
            session: &Bound<'_, PyAny>,
            candidate_index: usize,
            timestamp_queries: bool,
        ) -> PyResult<Self> {
            let _ = (session, candidate_index, timestamp_queries);
            Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "resident rank adaptation requires the 'kdsl' feature",
            ))
        }
        #[getter]
        fn shape(&self) -> (u32, u32, u32) {
            let p = self.inner.plan();
            (p.rows(), p.cols(), p.k())
        }
        #[getter]
        fn kind(&self) -> &'static str {
            self.inner.plan().kind().as_str()
        }
        #[getter]
        fn tile_cols(&self) -> u32 {
            self.inner.plan().tile_cols()
        }
        #[getter]
        fn generation(&self) -> u64 {
            self.inner.generation()
        }
        #[getter]
        fn output_is_current(&self) -> bool {
            self.inner.output_is_current()
        }

        #[getter]
        fn timestamp_queries_enabled(&self) -> bool {
            self.inner.timestamp_queries_enabled()
        }

        /// Execute instrumented passes and return the Rust-owned GPU timing schema.
        #[pyo3(signature = (repetitions=1))]
        fn profile(&mut self, py: Python<'_>, repetitions: u32) -> PyResult<Py<PyAny>> {
            let profile = py
                .detach(|| self.inner.dispatch_profiled(repetitions)?.read())
                .map_err(error)?;
            crate::json::json_to_py(py, &profile.report())
        }

        fn adapter_info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
            let info = self.inner.adapter_info();
            let out = PyDict::new(py);
            out.set_item("name", &info.name)?;
            out.set_item("backend", format!("{:?}", info.backend))?;
            out.set_item("device_type", format!("{:?}", info.device_type))?;
            Ok(out)
        }
        fn upload(&mut self, py: Python<'_>, input: &PyTensor) -> PyResult<()> {
            let (rows, cols, _) = self.shape();
            if input.inner.shape() != (rows as usize, cols as usize) {
                return Err(PyValueError::new_err(
                    "input shape must match the rank workspace",
                ));
            }
            let input = input
                .inner
                .to_layout(st_tensor::Layout::RowMajor)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            py.detach(|| self.inner.upload(input.data())).map_err(error)
        }
        fn set_input_from_matmul(
            &mut self,
            py: Python<'_>,
            source: &crate::wgpu_resident::PyWgpuMatmul,
        ) -> PyResult<()> {
            py.detach(|| self.inner.set_input_from_matmul(&source.inner))
                .map_err(error)
        }
        #[pyo3(signature = (repetitions=1))]
        fn dispatch(&mut self, py: Python<'_>, repetitions: u32) -> PyResult<u64> {
            py.detach(|| self.inner.dispatch(repetitions))
                .map_err(error)
        }
        #[pyo3(signature = (source, repetitions=1))]
        fn dispatch_from_matmul(
            &mut self,
            py: Python<'_>,
            source: &mut crate::wgpu_resident::PyWgpuMatmul,
            repetitions: u32,
        ) -> PyResult<u64> {
            py.detach(|| {
                self.inner
                    .dispatch_from_matmul(&mut source.inner, repetitions)
            })
            .map_err(error)
        }
        fn synchronize(&self, py: Python<'_>) -> PyResult<()> {
            py.detach(|| self.inner.synchronize()).map_err(error)
        }
        fn readback<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
            let snapshot = self.inner.snapshot().map_err(error)?;
            let generation = snapshot.generation();
            let output = py.detach(move || snapshot.read()).map_err(error)?;
            let out = PyDict::new(py);
            out.set_item("values", output.values)?;
            out.set_item("indices", output.indices)?;
            out.set_item("generation", generation)?;
            Ok(out)
        }
    }
}

#[cfg(feature = "wgpu")]
pub(crate) use enabled::PyWgpuRank;

#[cfg(not(feature = "wgpu"))]
#[pyclass(name = "WgpuRank", module = "spiraltorch.wgpu")]
pub(crate) struct PyWgpuRank;

#[cfg(not(feature = "wgpu"))]
#[pymethods]
impl PyWgpuRank {
    #[staticmethod]
    #[pyo3(signature = (session, candidate_index, *, timestamp_queries=false))]
    fn from_adaptation(
        session: &Bound<'_, PyAny>,
        candidate_index: usize,
        timestamp_queries: bool,
    ) -> PyResult<Self> {
        let _ = (session, candidate_index, timestamp_queries);
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "WgpuRank requires a wheel built with the 'wgpu' feature",
        ))
    }

    #[new]
    #[pyo3(signature = (kind, rows, cols, k, *, tile_cols=256, timestamp_queries=false))]
    fn new(
        kind: &str,
        rows: u32,
        cols: u32,
        k: u32,
        tile_cols: u32,
        timestamp_queries: bool,
    ) -> PyResult<Self> {
        let _ = (kind, rows, cols, k, tile_cols, timestamp_queries);
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "WgpuRank requires a wheel built with the 'wgpu' feature",
        ))
    }
}
