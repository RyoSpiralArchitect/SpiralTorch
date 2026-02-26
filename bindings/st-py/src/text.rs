use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::Bound;

#[cfg(feature = "text")]
use pyo3::exceptions::PyValueError;

#[cfg(feature = "text")]
use crate::{psi_synchro::PyZPulse, tensor::tensor_err_to_py};

#[cfg(feature = "text")]
use crate::{telemetry::PyAtlasFrame, vision::chrono_summary_from_any};

#[cfg(feature = "text")]
use st_core::theory::zpulse::ZScale;
#[cfg(feature = "text")]
#[cfg(feature = "text")]
use st_logic::contextual_observation::{
    Arrangement, Label, LagrangianGate, LagrangianGateConfig, MeaningProjection, OrientationGauge,
    PureAtom,
};
#[cfg(feature = "text")]
use st_text::{LanguageWave, RealtimeNarrator, ResonanceNarrative, TextResonator};

#[cfg(feature = "text")]
fn parse_gauge(value: &str) -> PyResult<OrientationGauge> {
    match value.to_ascii_lowercase().as_str() {
        "preserve" | "keep" | "stable" => Ok(OrientationGauge::Preserve),
        "swap" | "flip" | "invert" => Ok(OrientationGauge::Swap),
        other => Err(PyValueError::new_err(format!(
            "unknown gauge '{other}' (expected 'preserve' or 'swap')"
        ))),
    }
}

#[cfg(feature = "text")]
fn parse_atoms(placements: Vec<i64>) -> PyResult<Vec<PureAtom>> {
    placements
        .into_iter()
        .map(|value| match value {
            0 => Ok(PureAtom::A),
            1 => Ok(PureAtom::B),
            other => Err(PyValueError::new_err(format!(
                "placements must be 0 or 1, got {other}"
            ))),
        })
        .collect()
}

#[cfg(feature = "text")]
fn build_arrangement(
    placements: Vec<PureAtom>,
    edges: Option<Vec<(usize, usize)>>,
) -> PyResult<Arrangement> {
    if let Some(ref edge_list) = edges {
        for &(u, v) in edge_list {
            if u >= placements.len() || v >= placements.len() {
                return Err(PyValueError::new_err(
                    "edge indices must be within the placement range",
                ));
            }
            if u == v {
                return Err(PyValueError::new_err(
                    "edge endpoints must reference distinct indices",
                ));
            }
        }
    }
    Ok(match edges {
        Some(edge_list) => Arrangement::new(placements, edge_list),
        None => Arrangement::from_line(placements),
    })
}

#[cfg(feature = "text")]
#[pyclass(module = "spiraltorch.text", name = "ResonanceNarrative")]
#[derive(Clone, Debug)]
pub(crate) struct PyResonanceNarrative {
    summary: String,
    highlights: Vec<String>,
}

#[cfg(feature = "text")]
impl PyResonanceNarrative {
    fn from_inner(narrative: ResonanceNarrative) -> Self {
        Self {
            summary: narrative.summary,
            highlights: narrative.highlights,
        }
    }

    fn as_inner(&self) -> ResonanceNarrative {
        ResonanceNarrative {
            summary: self.summary.clone(),
            highlights: self.highlights.clone(),
        }
    }
}

#[cfg(feature = "text")]
#[pymethods]
impl PyResonanceNarrative {
    #[new]
    #[pyo3(signature = (summary, highlights=None))]
    fn new(summary: String, highlights: Option<Vec<String>>) -> Self {
        Self {
            summary,
            highlights: highlights.unwrap_or_default(),
        }
    }

    #[getter]
    fn summary(&self) -> &str {
        &self.summary
    }

    #[getter]
    fn highlights(&self) -> Vec<String> {
        self.highlights.clone()
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("summary", &self.summary)?;
        dict.set_item("highlights", self.highlights.clone())?;
        Ok(dict.to_object(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "ResonanceNarrative(summary_len={}, highlights={})",
            self.summary.len(),
            self.highlights.len()
        )
    }
}

#[cfg(feature = "text")]
#[pyclass(module = "spiraltorch.text", name = "LanguageWave")]
#[derive(Clone, Debug)]
pub(crate) struct PyLanguageWave {
    inner: LanguageWave,
}

#[cfg(feature = "text")]
impl PyLanguageWave {
    fn from_inner(inner: LanguageWave) -> Self {
        Self { inner }
    }
}

#[cfg(feature = "text")]
#[pymethods]
impl PyLanguageWave {
    #[getter]
    fn summary(&self) -> &str {
        &self.inner.summary
    }

    #[getter]
    fn amplitude(&self) -> Vec<f32> {
        self.inner.amplitude.clone()
    }

    #[pyo3(signature = (sample_rate))]
    fn to_audio_samples(&self, sample_rate: usize) -> Vec<f32> {
        self.inner.to_audio_samples(sample_rate)
    }

    fn __repr__(&self) -> String {
        format!(
            "LanguageWave(amplitude_len={}, summary_len={})",
            self.inner.amplitude.len(),
            self.inner.summary.len()
        )
    }
}

#[cfg(feature = "text")]
#[pyclass(module = "spiraltorch.text", name = "TextResonator", unsendable)]
#[derive(Clone, Debug)]
pub(crate) struct PyTextResonator {
    inner: TextResonator,
}

#[cfg(feature = "text")]
#[pymethods]
impl PyTextResonator {
    #[new]
    fn new(curvature: f32, temperature: f32) -> PyResult<Self> {
        Ok(Self {
            inner: TextResonator::new(curvature, temperature).map_err(tensor_err_to_py)?,
        })
    }

    fn describe_atlas(&self, atlas: &PyAtlasFrame) -> PyResonanceNarrative {
        PyResonanceNarrative::from_inner(self.inner.describe_atlas(&atlas.to_frame()))
    }

    #[pyo3(signature = (summary))]
    fn describe_chrono_summary(
        &self,
        summary: &Bound<'_, PyAny>,
    ) -> PyResult<PyResonanceNarrative> {
        let summary = chrono_summary_from_any(summary)?;
        Ok(PyResonanceNarrative::from_inner(
            self.inner.describe_summary(&summary),
        ))
    }

    fn describe_chrono_snapshot(
        &self,
        snapshot: &crate::vision::PyChronoSnapshot,
    ) -> PyResonanceNarrative {
        PyResonanceNarrative::from_inner(self.inner.describe_summary(snapshot.summary_inner()))
    }

    fn synthesize_wave(&self, narrative: &PyResonanceNarrative) -> PyResult<PyLanguageWave> {
        self.inner
            .synthesize_wave(&narrative.as_inner())
            .map(PyLanguageWave::from_inner)
            .map_err(tensor_err_to_py)
    }

    fn wave_from_atlas(&self, atlas: &PyAtlasFrame) -> PyResult<PyLanguageWave> {
        let narrative = self.inner.describe_atlas(&atlas.to_frame());
        self.inner
            .synthesize_wave(&narrative)
            .map(PyLanguageWave::from_inner)
            .map_err(tensor_err_to_py)
    }
}

#[cfg(feature = "text")]
#[pyclass(module = "spiraltorch.text", name = "RealtimeNarrator", unsendable)]
#[derive(Clone, Debug)]
pub(crate) struct PyRealtimeNarrator {
    inner: RealtimeNarrator,
}

#[cfg(feature = "text")]
#[pymethods]
impl PyRealtimeNarrator {
    #[new]
    fn new(curvature: f32, temperature: f32, sample_rate: usize) -> PyResult<Self> {
        Ok(Self {
            inner: RealtimeNarrator::new(curvature, temperature, sample_rate)
                .map_err(tensor_err_to_py)?,
        })
    }

    fn narrate_atlas(&self, atlas: &PyAtlasFrame) -> PyResult<Vec<f32>> {
        self.inner
            .narrate_atlas(&atlas.to_frame())
            .map_err(tensor_err_to_py)
    }

    #[pyo3(signature = (summary))]
    fn narrate_chrono_summary(&self, summary: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
        let summary = chrono_summary_from_any(summary)?;
        self.inner
            .narrate_summary(&summary)
            .map_err(tensor_err_to_py)
    }
}

#[cfg(feature = "text")]
#[pyclass(module = "spiraltorch.text", name = "ContextualPulseFrame")]
pub(crate) struct PyContextualPulseFrame {
    summary: String,
    highlights: Vec<String>,
    label: Option<String>,
    lexical_weight: f32,
    signature: Option<(usize, usize, isize)>,
    support: usize,
    pulse: PyZPulse,
}

#[cfg(feature = "text")]
impl PyContextualPulseFrame {
    fn from_parts(
        narrative: ResonanceNarrative,
        projection: MeaningProjection,
        pulse: st_core::theory::zpulse::ZPulse,
    ) -> Self {
        let label = projection
            .label
            .map(|label: Label| label.as_str().to_string());
        let lexical_weight = projection.lexical_weight();
        let signature = projection.signature.as_ref().map(|signature| {
            (
                signature.boundary_edges,
                signature.absolute_population_imbalance,
                signature.cluster_imbalance,
            )
        });
        let support = projection.support;
        Self {
            summary: narrative.summary,
            highlights: narrative.highlights,
            label,
            lexical_weight,
            signature,
            support,
            pulse: PyZPulse::from_pulse(pulse),
        }
    }
}

#[cfg(feature = "text")]
#[pymethods]
impl PyContextualPulseFrame {
    #[getter]
    pub fn summary(&self) -> &str {
        &self.summary
    }

    #[getter]
    pub fn highlights(&self) -> Vec<String> {
        self.highlights.clone()
    }

    #[getter]
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    #[getter]
    pub fn lexical_weight(&self) -> f32 {
        self.lexical_weight
    }

    #[getter]
    pub fn signature(&self) -> Option<(usize, usize, i64)> {
        self.signature
            .map(|(boundary, population, cluster)| (boundary, population, cluster as i64))
    }

    #[getter]
    pub fn support(&self) -> usize {
        self.support
    }

    #[getter]
    pub fn pulse(&self) -> PyZPulse {
        self.pulse.clone()
    }
}

#[cfg(feature = "text")]
#[pyclass(
    module = "spiraltorch.text",
    name = "ContextualLagrangianGate",
    unsendable
)]
pub(crate) struct PyContextualLagrangianGate {
    narrator: TextResonator,
    gate: LagrangianGate,
    default_gauge: OrientationGauge,
}

#[cfg(feature = "text")]
#[pymethods]
impl PyContextualLagrangianGate {
    #[new]
    #[pyo3(signature = (
        curvature,
        temperature,
        *,
        gauge="preserve",
        tempo_normaliser=None,
        energy_gain=1.0,
        drift_gain=1.0,
        bias_gain=1.0,
        support_gain=1.0,
        scale=None,
        quality_floor=0.0,
        stderr_gain=1.0
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        curvature: f32,
        temperature: f32,
        gauge: &str,
        tempo_normaliser: Option<f32>,
        energy_gain: f32,
        drift_gain: f32,
        bias_gain: f32,
        support_gain: f32,
        scale: Option<(f32, f32)>,
        quality_floor: f32,
        stderr_gain: f32,
    ) -> PyResult<Self> {
        let narrator = TextResonator::new(curvature, temperature).map_err(tensor_err_to_py)?;
        let default_gauge = parse_gauge(gauge)?;
        let mut config = LagrangianGateConfig::default()
            .tempo_normaliser(tempo_normaliser.unwrap_or(1.0))
            .energy_gain(energy_gain)
            .drift_gain(drift_gain)
            .bias_gain(bias_gain)
            .support_gain(support_gain)
            .quality_floor(quality_floor)
            .stderr_gain(stderr_gain);
        if let Some((physical, log)) = scale {
            let scale = ZScale::from_components(physical, log).ok_or_else(|| {
                PyValueError::new_err("scale must have positive finite radius components")
            })?;
            config = config.scale(Some(scale));
        }
        Ok(Self {
            narrator,
            gate: LagrangianGate::new(config),
            default_gauge,
        })
    }

    #[pyo3(signature = (placements, edges=None, *, gauge=None, ts=0))]
    pub fn project(
        &self,
        placements: Vec<i64>,
        edges: Option<Vec<(usize, usize)>>,
        gauge: Option<&str>,
        ts: u64,
    ) -> PyResult<PyContextualPulseFrame> {
        let atoms = parse_atoms(placements)?;
        let arrangement = build_arrangement(atoms, edges)?;
        let gauge = match gauge {
            Some(value) => parse_gauge(value)?,
            None => self.default_gauge,
        };
        let (narrative, projection, pulse) = self
            .narrator
            .gate_contextual_meaning(&arrangement, gauge, &self.gate, ts)
            .map_err(tensor_err_to_py)?;
        Ok(PyContextualPulseFrame::from_parts(
            narrative, projection, pulse,
        ))
    }

    #[getter]
    pub fn gauge(&self) -> &'static str {
        match self.default_gauge {
            OrientationGauge::Preserve => "preserve",
            OrientationGauge::Swap => "swap",
        }
    }
}

#[cfg(feature = "text")]
fn register_impl(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    let module = PyModule::new_bound(py, "spiraltorch.text")?;
    module.add(
        "__doc__",
        "Contextual resonance narrators and Lagrangian gates",
    )?;
    module.add_class::<PyResonanceNarrative>()?;
    module.add_class::<PyLanguageWave>()?;
    module.add_class::<PyTextResonator>()?;
    module.add_class::<PyRealtimeNarrator>()?;
    module.add_class::<PyContextualLagrangianGate>()?;
    module.add_class::<PyContextualPulseFrame>()?;
    module.add(
        "__all__",
        vec![
            "ResonanceNarrative",
            "LanguageWave",
            "TextResonator",
            "RealtimeNarrator",
            "ContextualLagrangianGate",
            "ContextualPulseFrame",
        ],
    )?;
    let module_obj = module.to_object(py);
    parent.add_submodule(&module)?;
    parent.add("text", module_obj.clone_ref(py))?;
    parent.add("ResonanceNarrative", module.getattr("ResonanceNarrative")?)?;
    parent.add("LanguageWave", module.getattr("LanguageWave")?)?;
    parent.add("TextResonator", module.getattr("TextResonator")?)?;
    parent.add("RealtimeNarrator", module.getattr("RealtimeNarrator")?)?;
    parent.add(
        "ContextualLagrangianGate",
        module.getattr("ContextualLagrangianGate")?,
    )?;
    parent.add(
        "ContextualPulseFrame",
        module.getattr("ContextualPulseFrame")?,
    )?;
    Ok(())
}

#[cfg(not(feature = "text"))]
fn register_impl(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    let module = PyModule::new_bound(py, "spiraltorch.text")?;
    module.add(
        "__doc__",
        "Contextual resonance narrators (compiled without the 'text' feature)",
    )?;
    let module_obj = module.to_object(py);
    parent.add_submodule(&module)?;
    parent.add("text", module_obj)?;
    Ok(())
}

pub(crate) fn register(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    register_impl(py, parent)
}
