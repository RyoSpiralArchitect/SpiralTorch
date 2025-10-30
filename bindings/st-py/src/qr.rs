use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::wrap_pyfunction;
use pyo3::{Bound, PyRef};

use crate::spiralk::PyMaxwellPulse;

use st_core::maxwell::MaxwellZPulse;
use st_qr_studio::{QuantumMeasurement, QuantumOverlayConfig, ZOverlayCircuit, ZResonance};
use std::cmp::Ordering;
use std::collections::HashMap;

fn pulses_from_any(py: Python<'_>, pulses: &Bound<'_, PyAny>) -> PyResult<Vec<MaxwellZPulse>> {
    if let Ok(single) = pulses.extract::<Py<PyMaxwellPulse>>() {
        let pulse = single.borrow(py);
        return Ok(vec![pulse.to_pulse()]);
    }

    if let Ok(list) = pulses.extract::<Vec<Py<PyMaxwellPulse>>>() {
        let mut out = Vec::with_capacity(list.len());
        for pulse in list {
            out.push(pulse.borrow(py).to_pulse());
        }
        return Ok(out);
    }

    Err(PyTypeError::new_err(
        "pulses must be iterables of spiraltorch.spiralk.MaxwellPulse",
    ))
}

fn fractal_patch_to_pulses(patch: &Bound<'_, PyAny>, eta_scale: f64) -> Vec<MaxwellZPulse> {
    let density: Vec<f64> = patch
        .getattr("density")
        .and_then(|obj| obj.extract())
        .unwrap_or_default();
    let support: (f64, f64) = patch
        .getattr("support")
        .and_then(|obj| obj.extract())
        .unwrap_or((0.0, 1.0));
    let mut start = support.0.min(support.1);
    let mut end = support.0.max(support.1);
    if !start.is_finite() {
        start = 0.0;
    }
    if !end.is_finite() || (end - start).abs() < 1e-6 {
        end = start + 1.0;
    }
    let span = (end - start).abs().max(1e-6);
    let dimension = patch
        .getattr("dimension")
        .and_then(|obj| obj.extract::<f64>())
        .unwrap_or(2.0)
        .abs()
        .max(1.0);
    let zoom = patch
        .getattr("zoom")
        .and_then(|obj| obj.extract::<f64>())
        .unwrap_or(1.0)
        .abs()
        .max(1e-6);
    let eta_scale = eta_scale.max(0.0);
    let limit = density.len();
    let steps = if limit > 1 { (limit - 1) as f64 } else { 1.0 };
    let mut pulses = Vec::with_capacity(limit.max(1));
    for (index, raw) in density.into_iter().enumerate() {
        let amplitude = if raw.is_finite() { raw.abs() } else { 0.0 };
        let phase = if limit <= 1 {
            0.0
        } else {
            index as f64 / steps
        };
        let mean = start + span * phase;
        let standard_error = (1.0 / (index as f64 + 1.0)).sqrt();
        let spectral = amplitude * (dimension + 1.0);
        let radial = amplitude * (phase * span + 1.0);
        let axial = amplitude * (index as f64 + 1.0);
        let z_score = amplitude * zoom * (dimension + phase);
        let z_bias = (z_score * eta_scale).tanh() as f32;
        pulses.push(MaxwellZPulse {
            blocks: index as u64,
            mean,
            standard_error,
            z_score,
            band_energy: (spectral as f32, radial as f32, axial as f32),
            z_bias,
        });
    }
    if pulses.is_empty() {
        pulses.push(MaxwellZPulse {
            blocks: 0,
            mean: start,
            standard_error: 1.0,
            z_score: 0.0,
            band_energy: (0.0, 0.0, 0.0),
            z_bias: 0.0,
        });
    }
    pulses
}

#[pyclass(module = "spiraltorch.qr", name = "QuantumOverlayConfig")]
#[derive(Clone)]
pub(crate) struct PyQuantumOverlayConfig {
    inner: QuantumOverlayConfig,
}

impl PyQuantumOverlayConfig {
    fn rebuild(&mut self, curvature: f64, qubits: usize, packing_bias: f64, leech_shells: usize) {
        self.inner = QuantumOverlayConfig::new(curvature, qubits)
            .with_packing_bias(packing_bias)
            .with_leech_shells(leech_shells);
    }
}

#[pymethods]
impl PyQuantumOverlayConfig {
    #[new]
    #[pyo3(signature = (curvature=-1.0, qubits=24, packing_bias=0.35, leech_shells=24))]
    pub fn new(curvature: f64, qubits: usize, packing_bias: f64, leech_shells: usize) -> Self {
        let config = QuantumOverlayConfig::new(curvature, qubits)
            .with_packing_bias(packing_bias)
            .with_leech_shells(leech_shells);
        Self { inner: config }
    }

    #[getter]
    pub fn curvature(&self) -> f64 {
        self.inner.curvature()
    }

    pub fn set_curvature(&mut self, curvature: f64) {
        self.rebuild(
            curvature,
            self.inner.qubits(),
            self.inner.packing_bias(),
            self.inner.leech_shells(),
        );
    }

    #[getter]
    pub fn qubits(&self) -> usize {
        self.inner.qubits()
    }

    pub fn set_qubits(&mut self, qubits: usize) {
        self.rebuild(
            self.inner.curvature(),
            qubits,
            self.inner.packing_bias(),
            self.inner.leech_shells(),
        );
    }

    #[getter]
    pub fn packing_bias(&self) -> f64 {
        self.inner.packing_bias()
    }

    pub fn set_packing_bias(&mut self, packing_bias: f64) {
        self.rebuild(
            self.inner.curvature(),
            self.inner.qubits(),
            packing_bias,
            self.inner.leech_shells(),
        );
    }

    #[getter]
    pub fn leech_shells(&self) -> usize {
        self.inner.leech_shells()
    }

    pub fn set_leech_shells(&mut self, leech_shells: usize) {
        self.rebuild(
            self.inner.curvature(),
            self.inner.qubits(),
            self.inner.packing_bias(),
            leech_shells,
        );
    }
}

#[pyclass(module = "spiraltorch.qr", name = "ZResonance")]
#[derive(Clone)]
pub(crate) struct PyZResonance {
    inner: ZResonance,
}

impl PyZResonance {
    pub(crate) fn inner(&self) -> &ZResonance {
        &self.inner
    }
}

#[pymethods]
impl PyZResonance {
    #[new]
    #[pyo3(signature = (spectrum=None, eta_hint=0.0, shell_weights=None))]
    pub fn new(spectrum: Option<Vec<f64>>, eta_hint: f32, shell_weights: Option<Vec<f64>>) -> Self {
        let resonance = match (spectrum, shell_weights) {
            (Some(spectrum), Some(shells)) => ZResonance {
                spectrum,
                eta_hint: eta_hint.max(0.0),
                shell_weights: shells,
            },
            (Some(spectrum), None) => ZResonance::from_spectrum(spectrum, eta_hint),
            (None, Some(shells)) => ZResonance {
                spectrum: Vec::new(),
                eta_hint: eta_hint.max(0.0),
                shell_weights: shells,
            },
            (None, None) => ZResonance::from_spectrum(Vec::new(), eta_hint),
        };
        Self { inner: resonance }
    }

    #[staticmethod]
    #[pyo3(signature = (spectrum, eta_hint=None))]
    pub fn from_spectrum(spectrum: Vec<f64>, eta_hint: Option<f32>) -> Self {
        let hint = eta_hint.unwrap_or_default();
        Self {
            inner: ZResonance::from_spectrum(spectrum, hint),
        }
    }

    #[staticmethod]
    pub fn from_pulses(py: Python<'_>, pulses: &Bound<'_, PyAny>) -> PyResult<Self> {
        let collected = pulses_from_any(py, pulses)?;
        Ok(Self {
            inner: ZResonance::from_pulses(&collected),
        })
    }

    #[getter]
    pub fn spectrum(&self) -> Vec<f64> {
        self.inner.spectrum.clone()
    }

    #[getter]
    pub fn eta_hint(&self) -> f32 {
        self.inner.eta_hint
    }

    #[getter]
    pub fn shell_weights(&self) -> Vec<f64> {
        self.inner.shell_weights.clone()
    }
}

#[pyclass(module = "spiraltorch.qr", name = "ZOverlayCircuit")]
#[derive(Clone)]
pub(crate) struct PyZOverlayCircuit {
    inner: ZOverlayCircuit,
}

impl From<ZOverlayCircuit> for PyZOverlayCircuit {
    fn from(inner: ZOverlayCircuit) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyZOverlayCircuit {
    pub fn weights(&self) -> Vec<f64> {
        self.inner.weights().to_vec()
    }

    pub fn eta_bar(&self) -> f64 {
        self.inner.eta_bar()
    }

    pub fn packing_pressure(&self) -> f64 {
        self.inner.packing_pressure()
    }

    #[pyo3(signature = (threshold=0.0))]
    pub fn measure(&self, threshold: f64) -> PyQuantumMeasurement {
        PyQuantumMeasurement::from(self.inner.measure(threshold))
    }
}

#[pyclass(module = "spiraltorch.qr", name = "QuantumMeasurement")]
#[derive(Clone)]
pub(crate) struct PyQuantumMeasurement {
    inner: QuantumMeasurement,
}

impl From<QuantumMeasurement> for PyQuantumMeasurement {
    fn from(inner: QuantumMeasurement) -> Self {
        Self { inner }
    }
}

fn sorted_logits(measurement: &QuantumMeasurement) -> Vec<(usize, f32)> {
    let mut enumerated: Vec<(usize, f32)> = measurement
        .policy_logits()
        .iter()
        .copied()
        .enumerate()
        .collect();
    enumerated.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    enumerated
}

#[pymethods]
impl PyQuantumMeasurement {
    #[getter]
    pub fn active_qubits(&self) -> Vec<usize> {
        self.inner.active_qubits().to_vec()
    }

    #[getter]
    pub fn eta_bar(&self) -> f32 {
        self.inner.eta_bar()
    }

    #[getter]
    pub fn policy_logits(&self) -> Vec<f32> {
        self.inner.policy_logits().to_vec()
    }

    #[getter]
    pub fn packing_pressure(&self) -> f32 {
        self.inner.packing_pressure()
    }

    #[pyo3(signature = (count=None))]
    pub fn top_qubits(&self, count: Option<usize>) -> Vec<(usize, f32)> {
        let mut ranked = sorted_logits(&self.inner);
        if let Some(limit) = count {
            let limit = limit.max(1);
            if ranked.len() > limit {
                ranked.truncate(limit);
            }
        }
        ranked
    }

    pub fn activation_density(&self) -> f32 {
        let total = self.inner.policy_logits().len();
        if total == 0 {
            return 0.0;
        }
        let active = self.inner.active_qubits().len();
        (active as f32 / total as f32).clamp(0.0, 1.0)
    }

    #[pyo3(signature = (base_rate=1.0))]
    pub fn to_policy_update(&self, base_rate: f32) -> HashMap<String, f32> {
        let mut ranked = sorted_logits(&self.inner);
        let active = self.inner.active_qubits().len().max(1);
        if ranked.len() > active {
            ranked.truncate(active);
        }
        let active_mean = if ranked.is_empty() {
            0.0
        } else {
            ranked.iter().map(|(_, weight)| weight.abs()).sum::<f32>() / ranked.len() as f32
        };
        let activation = self.activation_density();
        let eta = self.inner.eta_bar().abs();
        let pressure = self.inner.packing_pressure().abs();
        let novelty = eta + pressure * 0.5;
        let base = base_rate.max(0.0);
        let mut update = HashMap::with_capacity(5);
        update.insert("learning_rate".to_string(), base + novelty.max(0.0));
        update.insert("gauge".to_string(), base + activation + active_mean);
        update.insert("eta_bar".to_string(), self.inner.eta_bar());
        update.insert(
            "packing_pressure".to_string(),
            self.inner.packing_pressure(),
        );
        update.insert("activation_density".to_string(), activation);
        update
    }
}

#[pyclass(module = "spiraltorch.qr", name = "QuantumRealityStudio")]
#[derive(Clone)]
pub(crate) struct PyQuantumRealityStudio {
    config: QuantumOverlayConfig,
}

impl PyQuantumRealityStudio {
    fn overlay_internal(&self, resonance: &PyZResonance) -> PyZOverlayCircuit {
        PyZOverlayCircuit::from(ZOverlayCircuit::synthesize(&self.config, resonance.inner()))
    }
}

#[pymethods]
impl PyQuantumRealityStudio {
    #[new]
    #[pyo3(signature = (curvature=-1.0, qubits=24, packing_bias=0.35, leech_shells=24))]
    pub fn new(curvature: f64, qubits: usize, packing_bias: f64, leech_shells: usize) -> Self {
        let config = QuantumOverlayConfig::new(curvature, qubits)
            .with_packing_bias(packing_bias)
            .with_leech_shells(leech_shells);
        Self { config }
    }

    #[pyo3(signature = (curvature=None, qubits=None, packing_bias=None, leech_shells=None))]
    pub fn configure(
        &mut self,
        curvature: Option<f64>,
        qubits: Option<usize>,
        packing_bias: Option<f64>,
        leech_shells: Option<usize>,
    ) {
        let mut config = QuantumOverlayConfig::new(
            curvature.unwrap_or(self.config.curvature()),
            qubits.unwrap_or(self.config.qubits()),
        );
        config = config
            .with_packing_bias(packing_bias.unwrap_or(self.config.packing_bias()))
            .with_leech_shells(leech_shells.unwrap_or(self.config.leech_shells()));
        self.config = config;
    }

    pub fn overlay_zspace(&self, resonance: &PyZResonance) -> PyZOverlayCircuit {
        self.overlay_internal(resonance)
    }

    pub fn overlay(&self, resonance: &PyZResonance) -> PyZOverlayCircuit {
        self.overlay_internal(resonance)
    }

    #[pyo3(signature = (pulses, threshold=0.0))]
    pub fn record_quantum_policy(
        &self,
        py: Python<'_>,
        pulses: &Bound<'_, PyAny>,
        threshold: f64,
    ) -> PyResult<PyQuantumMeasurement> {
        let pulses = pulses_from_any(py, pulses)?;
        let resonance = ZResonance::from_pulses(&pulses);
        let circuit = ZOverlayCircuit::synthesize(&self.config, &resonance);
        Ok(PyQuantumMeasurement::from(circuit.measure(threshold)))
    }
}

#[pyfunction(name = "resonance_from_fractal_patch")]
#[pyo3(signature = (patch, eta_scale=1.0))]
fn resonance_from_fractal_patch_py(patch: &Bound<'_, PyAny>, eta_scale: f64) -> PyZResonance {
    PyZResonance {
        inner: ZResonance::from_pulses(&fractal_patch_to_pulses(patch, eta_scale)),
    }
}

#[pyfunction(name = "quantum_measurement_from_fractal")]
#[pyo3(signature = (studio, patch, threshold=0.0, eta_scale=1.0))]
fn quantum_measurement_from_fractal_py(
    studio: PyRef<PyQuantumRealityStudio>,
    patch: &Bound<'_, PyAny>,
    threshold: f64,
    eta_scale: f64,
) -> PyQuantumMeasurement {
    let resonance = ZResonance::from_pulses(&fractal_patch_to_pulses(patch, eta_scale));
    let circuit = ZOverlayCircuit::synthesize(&studio.config, &resonance);
    PyQuantumMeasurement::from(circuit.measure(threshold))
}

pub(crate) fn register(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    let module = PyModule::new_bound(py, "qr")?;
    module.add("__doc__", "Quantum overlay helpers")?;
    module.add_class::<PyQuantumOverlayConfig>()?;
    module.add_class::<PyZResonance>()?;
    module.add_class::<PyZOverlayCircuit>()?;
    module.add_class::<PyQuantumMeasurement>()?;
    module.add_class::<PyQuantumRealityStudio>()?;
    module.add_function(wrap_pyfunction!(resonance_from_fractal_patch_py, &module)?)?;
    module.add_function(wrap_pyfunction!(
        quantum_measurement_from_fractal_py,
        &module
    )?)?;
    module.add(
        "__all__",
        vec![
            "QuantumOverlayConfig",
            "ZResonance",
            "ZOverlayCircuit",
            "QuantumMeasurement",
            "QuantumRealityStudio",
            "resonance_from_fractal_patch",
            "quantum_measurement_from_fractal",
        ],
    )?;

    parent.add_submodule(&module)?;
    let module_obj = module.to_object(py);
    parent.add("qr", module_obj)?;

    for name in [
        "QuantumOverlayConfig",
        "ZResonance",
        "ZOverlayCircuit",
        "QuantumMeasurement",
        "QuantumRealityStudio",
        "resonance_from_fractal_patch",
        "quantum_measurement_from_fractal",
    ] {
        let attr = module.getattr(name)?;
        parent.add(name, attr.to_object(py))?;
    }

    Ok(())
}
