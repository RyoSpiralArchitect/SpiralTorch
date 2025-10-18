// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::{convert, intern_label, PyOpenTopos, PyTensor, PyTensorBiome};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use pyo3::wrap_pyfunction;
use pyo3::Bound;
use st_tensor::{Tensor, TensorBiome};

const GOLDEN_ANGLE: f64 = 2.399_963_229_728_653; // π(3 − √5)
const GOLDEN_RATIO: f64 = 1.618_033_988_749_895; // (1 + √5) / 2

#[derive(Clone, Copy)]
pub(crate) struct Sot3DParams {
    pub(crate) base_radius: f64,
    pub(crate) radial_growth: f64,
    pub(crate) base_height: f64,
    pub(crate) meso_gain: f64,
    pub(crate) micro_gain: f64,
}

#[derive(Clone)]
struct Sot3DStep {
    index: usize,
    angle: f64,
    radius: f64,
    x: f64,
    y: f64,
    height: f64,
    z_track: f64,
    curvature: f64,
    macro_index: usize,
    macro_length: usize,
    macro_position: usize,
    macro_phase: f64,
    meso_index: usize,
    meso_length: usize,
    meso_position: usize,
    meso_phase: f64,
    meso_role: &'static str,
    meso_role_index: usize,
    micro_index: usize,
    micro_length: usize,
    micro_position: usize,
    micro_phase: f64,
    micro_role: &'static str,
    micro_role_index: usize,
    macro_reflection: bool,
    meso_reflection: bool,
    micro_reflection: bool,
}

#[derive(Clone)]
struct MacroSummary {
    index: usize,
    start: usize,
    length: usize,
    ideal_length: usize,
    reflection_step: Option<usize>,
    height_gain: f64,
    mean_curvature: f64,
}

struct OrbitLength {
    actual: usize,
    ideal: usize,
}

const MESO_ROLES: [&str; 3] = ["explore", "structure", "verify"];
const MICRO_ROLES: [&str; 4] = ["diverge", "ground", "design", "test"];

fn fibonacci_orbits(total_steps: usize) -> Vec<OrbitLength> {
    if total_steps == 0 {
        return Vec::new();
    }
    let mut orbits = Vec::new();
    let mut remaining = total_steps;
    let mut a = 1usize;
    let mut b = 1usize;
    while remaining > 0 {
        let ideal = a;
        let take = ideal.min(remaining);
        orbits.push(OrbitLength {
            actual: take,
            ideal,
        });
        remaining -= take;
        let next = a + b;
        a = b;
        b = next;
    }
    orbits
}

fn nacci_numbers(order: usize, seeds: &[usize], limit: usize) -> Vec<usize> {
    assert!(order > 0);
    assert!(seeds.len() == order);
    let mut seq: Vec<usize> = seeds.to_vec();
    while *seq.last().unwrap() < limit {
        let len = seq.len();
        let next: usize = seq[len - order..].iter().sum();
        seq.push(next);
    }
    let mut filtered: Vec<usize> = seq.into_iter().filter(|&v| v <= limit).collect();
    filtered.sort_unstable();
    filtered.dedup();
    if filtered.is_empty() {
        filtered.push(1);
    }
    filtered
}

fn pack_with_sequence(length: usize, order: usize, seeds: &[usize]) -> Vec<usize> {
    if length == 0 {
        return Vec::new();
    }
    let mut remaining = length;
    let mut chunks = Vec::new();
    while remaining > 0 {
        let options = nacci_numbers(order, seeds, remaining);
        let target = remaining as f64 / 2.0;
        let mut best = options[0];
        let mut best_score = (best as f64 - target).abs();
        for &candidate in &options[1..] {
            let score = (candidate as f64 - target).abs();
            if score < best_score - f64::EPSILON {
                best = candidate;
                best_score = score;
            } else if (score - best_score).abs() <= f64::EPSILON && candidate > best {
                best = candidate;
                best_score = score;
            }
        }
        chunks.push(best);
        remaining -= best;
    }
    chunks
}

fn pack_tribonacci(length: usize) -> Vec<usize> {
    pack_with_sequence(length, 3, &[1, 1, 2])
}

fn pack_tetranacci(length: usize) -> Vec<usize> {
    pack_with_sequence(length, 4, &[1, 1, 2, 4])
}

pub(crate) fn build_plan(total_steps: usize, params: Sot3DParams) -> PyResult<PySoT3DPlan> {
    if total_steps == 0 {
        return Ok(PySoT3DPlan {
            steps: Vec::new(),
            macros: Vec::new(),
            params,
            total_steps,
        });
    }

    let orbits = fibonacci_orbits(total_steps);
    let mut steps = Vec::with_capacity(total_steps);
    let mut macros = Vec::with_capacity(orbits.len());
    let mut z_track = 0.0;
    let phi = GOLDEN_RATIO;
    let curvature_factor = (1.0 + params.radial_growth * params.radial_growth).sqrt();

    let mut global_step_index = 0usize;
    let mut global_meso_index = 0usize;
    let mut global_micro_index = 0usize;

    for (macro_index, orbit) in orbits.iter().enumerate() {
        let macro_length = orbit.actual;
        if macro_length == 0 {
            continue;
        }
        let macro_start = global_step_index;
        let delta_height = params.base_height / phi.powi(macro_index as i32);
        let meso_chunks = pack_tribonacci(macro_length);
        let mut macro_step_position = 0usize;
        let mut macro_curvature_acc = 0.0;

        for meso_length in meso_chunks {
            let meso_index = global_meso_index;
            let meso_role = MESO_ROLES[meso_index % MESO_ROLES.len()];
            let meso_role_index = meso_index % MESO_ROLES.len();
            let mut meso_step_position = 0usize;
            let micro_chunks = pack_tetranacci(meso_length);

            for micro_length in micro_chunks {
                let micro_index = global_micro_index;
                let micro_role = MICRO_ROLES[micro_index % MICRO_ROLES.len()];
                let micro_role_index = micro_index % MICRO_ROLES.len();

                for micro_step in 0..micro_length {
                    let step_index = global_step_index;
                    let macro_phase = if macro_length > 1 {
                        macro_step_position as f64 / (macro_length - 1) as f64
                    } else {
                        0.0
                    };
                    let meso_phase = if meso_length > 1 {
                        meso_step_position as f64 / (meso_length - 1) as f64
                    } else {
                        0.0
                    };
                    let micro_phase = if micro_length > 1 {
                        micro_step as f64 / (micro_length - 1) as f64
                    } else {
                        0.0
                    };

                    z_track += delta_height / macro_length as f64;
                    let base_height = z_track;
                    let height = base_height
                        + params.meso_gain * (meso_phase - 0.5)
                        + params.micro_gain * (micro_phase - 0.5);

                    let angle = GOLDEN_ANGLE * step_index as f64;
                    let radius =
                        params.base_radius * f64::exp(params.radial_growth * step_index as f64);
                    let x = radius * angle.cos();
                    let y = radius * angle.sin();
                    let curvature = if radius.abs() <= f64::EPSILON {
                        f64::INFINITY
                    } else {
                        curvature_factor / radius.abs()
                    };

                    macro_curvature_acc +=
                        curvature.is_finite().then_some(curvature).unwrap_or(0.0);

                    steps.push(Sot3DStep {
                        index: step_index,
                        angle,
                        radius,
                        x,
                        y,
                        height,
                        z_track: base_height,
                        curvature,
                        macro_index,
                        macro_length,
                        macro_position: macro_step_position,
                        macro_phase,
                        meso_index,
                        meso_length,
                        meso_position: meso_step_position,
                        meso_phase,
                        meso_role,
                        meso_role_index,
                        micro_index,
                        micro_length,
                        micro_position: micro_step,
                        micro_phase,
                        micro_role,
                        micro_role_index,
                        macro_reflection: macro_step_position + 1 == macro_length,
                        meso_reflection: meso_step_position + 1 == meso_length,
                        micro_reflection: micro_step + 1 == micro_length,
                    });

                    global_step_index += 1;
                    macro_step_position += 1;
                    meso_step_position += 1;
                }
                global_micro_index += 1;
            }
            global_meso_index += 1;
        }

        let reflection = if macro_length > 0 {
            Some(macro_start + macro_length - 1)
        } else {
            None
        };
        let mean_curvature = if macro_length > 0 {
            macro_curvature_acc / macro_length as f64
        } else {
            0.0
        };
        macros.push(MacroSummary {
            index: macro_index,
            start: macro_start,
            length: macro_length,
            ideal_length: orbit.ideal,
            reflection_step: reflection,
            height_gain: delta_height,
            mean_curvature,
        });
    }

    Ok(PySoT3DPlan {
        steps,
        macros,
        params,
        total_steps,
    })
}

#[pyclass(module = "spiraltorch.sot", name = "SoT3DStep")]
struct PySoT3DStep {
    step: Sot3DStep,
}

#[pymethods]
impl PySoT3DStep {
    #[getter]
    fn index(&self) -> usize {
        self.step.index
    }

    #[getter]
    fn angle(&self) -> f64 {
        self.step.angle
    }

    #[getter]
    fn radius(&self) -> f64 {
        self.step.radius
    }

    #[getter]
    fn x(&self) -> f64 {
        self.step.x
    }

    #[getter]
    fn y(&self) -> f64 {
        self.step.y
    }

    #[getter]
    fn height(&self) -> f64 {
        self.step.height
    }

    #[getter]
    fn z_track(&self) -> f64 {
        self.step.z_track
    }

    #[getter]
    fn curvature(&self) -> f64 {
        self.step.curvature
    }

    #[getter]
    fn macro_index(&self) -> usize {
        self.step.macro_index
    }

    #[getter]
    fn macro_length(&self) -> usize {
        self.step.macro_length
    }

    #[getter]
    fn macro_position(&self) -> usize {
        self.step.macro_position
    }

    #[getter]
    fn macro_phase(&self) -> f64 {
        self.step.macro_phase
    }

    #[getter]
    fn meso_index(&self) -> usize {
        self.step.meso_index
    }

    #[getter]
    fn meso_length(&self) -> usize {
        self.step.meso_length
    }

    #[getter]
    fn meso_position(&self) -> usize {
        self.step.meso_position
    }

    #[getter]
    fn meso_phase(&self) -> f64 {
        self.step.meso_phase
    }

    #[getter]
    fn meso_role(&self) -> &'static str {
        self.step.meso_role
    }

    #[getter]
    fn meso_role_index(&self) -> usize {
        self.step.meso_role_index
    }

    #[getter]
    fn micro_index(&self) -> usize {
        self.step.micro_index
    }

    #[getter]
    fn micro_length(&self) -> usize {
        self.step.micro_length
    }

    #[getter]
    fn micro_position(&self) -> usize {
        self.step.micro_position
    }

    #[getter]
    fn micro_phase(&self) -> f64 {
        self.step.micro_phase
    }

    #[getter]
    fn micro_role(&self) -> &'static str {
        self.step.micro_role
    }

    #[getter]
    fn micro_role_index(&self) -> usize {
        self.step.micro_role_index
    }

    #[getter]
    fn macro_reflection(&self) -> bool {
        self.step.macro_reflection
    }

    #[getter]
    fn meso_reflection(&self) -> bool {
        self.step.meso_reflection
    }

    #[getter]
    fn micro_reflection(&self) -> bool {
        self.step.micro_reflection
    }

    fn position(&self) -> (f64, f64, f64) {
        (self.step.x, self.step.y, self.step.height)
    }

    fn as_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item("index", self.step.index)?;
        dict.set_item("angle", self.step.angle)?;
        dict.set_item("radius", self.step.radius)?;
        dict.set_item("x", self.step.x)?;
        dict.set_item("y", self.step.y)?;
        dict.set_item("height", self.step.height)?;
        dict.set_item("z_track", self.step.z_track)?;
        dict.set_item("curvature", self.step.curvature)?;
        let macro_info = PyDict::new_bound(py);
        macro_info.set_item("index", self.step.macro_index)?;
        macro_info.set_item("length", self.step.macro_length)?;
        macro_info.set_item("position", self.step.macro_position)?;
        macro_info.set_item("phase", self.step.macro_phase)?;
        macro_info.set_item("reflection", self.step.macro_reflection)?;
        dict.set_item("macro", macro_info)?;
        let meso_info = PyDict::new_bound(py);
        meso_info.set_item("index", self.step.meso_index)?;
        meso_info.set_item("length", self.step.meso_length)?;
        meso_info.set_item("position", self.step.meso_position)?;
        meso_info.set_item("phase", self.step.meso_phase)?;
        meso_info.set_item("role", self.step.meso_role)?;
        meso_info.set_item("role_index", self.step.meso_role_index)?;
        meso_info.set_item("reflection", self.step.meso_reflection)?;
        dict.set_item("meso", meso_info)?;
        let micro_info = PyDict::new_bound(py);
        micro_info.set_item("index", self.step.micro_index)?;
        micro_info.set_item("length", self.step.micro_length)?;
        micro_info.set_item("position", self.step.micro_position)?;
        micro_info.set_item("phase", self.step.micro_phase)?;
        micro_info.set_item("role", self.step.micro_role)?;
        micro_info.set_item("role_index", self.step.micro_role_index)?;
        micro_info.set_item("reflection", self.step.micro_reflection)?;
        dict.set_item("micro", micro_info)?;
        Ok(dict.into_py(py))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "SoT3DStep(index={}, macro={}@{}, meso={}@{}, micro={}@{})",
            self.step.index,
            self.step.macro_index,
            self.step.macro_position,
            self.step.meso_index,
            self.step.meso_position,
            self.step.micro_index,
            self.step.micro_position
        ))
    }
}

#[pyclass(module = "spiraltorch.sot", name = "MacroSummary")]
struct PyMacroSummary {
    summary: MacroSummary,
}

#[pymethods]
impl PyMacroSummary {
    #[getter]
    fn index(&self) -> usize {
        self.summary.index
    }

    #[getter]
    fn start(&self) -> usize {
        self.summary.start
    }

    #[getter]
    fn length(&self) -> usize {
        self.summary.length
    }

    #[getter]
    fn ideal_length(&self) -> usize {
        self.summary.ideal_length
    }

    #[getter]
    fn reflection_step(&self) -> Option<usize> {
        self.summary.reflection_step
    }

    #[getter]
    fn height_gain(&self) -> f64 {
        self.summary.height_gain
    }

    #[getter]
    fn mean_curvature(&self) -> f64 {
        self.summary.mean_curvature
    }

    fn as_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item("index", self.summary.index)?;
        dict.set_item("start", self.summary.start)?;
        dict.set_item("length", self.summary.length)?;
        dict.set_item("ideal_length", self.summary.ideal_length)?;
        dict.set_item("reflection_step", self.summary.reflection_step)?;
        dict.set_item("height_gain", self.summary.height_gain)?;
        dict.set_item("mean_curvature", self.summary.mean_curvature)?;
        Ok(dict.into_py(py))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "MacroSummary(index={}, length={}, reflection={:?})",
            self.summary.index, self.summary.length, self.summary.reflection_step
        ))
    }
}

#[pyclass(module = "spiraltorch.sot", name = "SoT3DPlan")]
#[derive(Clone)]
pub struct PySoT3DPlan {
    steps: Vec<Sot3DStep>,
    macros: Vec<MacroSummary>,
    params: Sot3DParams,
    total_steps: usize,
}

impl PySoT3DPlan {
    pub(crate) fn positions_tensor(&self) -> Result<Tensor, st_tensor::TensorError> {
        if self.steps.is_empty() {
            return Tensor::from_vec(0, 3, Vec::new());
        }
        let mut data = Vec::with_capacity(self.steps.len() * 3);
        for step in &self.steps {
            data.push(step.x as f32);
            data.push(step.y as f32);
            data.push(step.height as f32);
        }
        Tensor::from_vec(self.steps.len(), 3, data)
    }

    fn feature_tensor_internal(&self) -> Result<Tensor, st_tensor::TensorError> {
        if self.steps.is_empty() {
            return Tensor::from_vec(0, 9, Vec::new());
        }
        let mut feature_data = Vec::with_capacity(self.steps.len() * 9);
        for step in &self.steps {
            feature_data.push(step.radius as f32);
            feature_data.push(step.angle as f32);
            feature_data.push(step.height as f32);
            feature_data.push(step.curvature as f32);
            feature_data.push(step.macro_phase as f32);
            feature_data.push(step.meso_phase as f32);
            feature_data.push(step.micro_phase as f32);
            feature_data.push(self.params.meso_gain as f32);
            feature_data.push(self.params.micro_gain as f32);
        }
        Tensor::from_vec(self.steps.len(), 9, feature_data)
    }

    fn reflection_tensor_internal(&self) -> Result<Tensor, st_tensor::TensorError> {
        if self.steps.is_empty() {
            return Tensor::from_vec(0, 3, Vec::new());
        }
        let mut reflection_data = Vec::with_capacity(self.steps.len() * 3);
        for step in &self.steps {
            reflection_data.push(if step.macro_reflection { 1.0 } else { 0.0 });
            reflection_data.push(if step.meso_reflection { 1.0 } else { 0.0 });
            reflection_data.push(if step.micro_reflection { 1.0 } else { 0.0 });
        }
        Tensor::from_vec(self.steps.len(), 3, reflection_data)
    }

    fn role_tensor_internal(&self) -> Result<Tensor, st_tensor::TensorError> {
        if self.steps.is_empty() {
            return Tensor::from_vec(0, 2, Vec::new());
        }
        let mut role_data = Vec::with_capacity(self.steps.len() * 2);
        for step in &self.steps {
            role_data.push(step.meso_role_index as f32);
            role_data.push(step.micro_role_index as f32);
        }
        Tensor::from_vec(self.steps.len(), 2, role_data)
    }

    fn macro_summary_tensor_internal(&self) -> Result<Tensor, st_tensor::TensorError> {
        if self.macros.is_empty() {
            return Tensor::from_vec(0, 6, Vec::new());
        }
        let mut macro_data = Vec::with_capacity(self.macros.len() * 6);
        for summary in &self.macros {
            macro_data.push(summary.index as f32);
            macro_data.push(summary.length as f32);
            macro_data.push(summary.ideal_length as f32);
            macro_data.push(summary.height_gain as f32);
            macro_data.push(summary.mean_curvature as f32);
            macro_data.push(
                summary
                    .reflection_step
                    .map(|idx| idx as f32)
                    .unwrap_or(-1.0),
            );
        }
        Tensor::from_vec(self.macros.len(), 6, macro_data)
    }

    fn deposit_into_biome(
        &self,
        biome: &mut TensorBiome,
        prefix: &str,
        include_reflections: bool,
        include_roles: bool,
    ) -> PyResult<()> {
        if self.steps.is_empty() {
            return Ok(());
        }
        let positions = convert(self.positions_tensor())?;
        convert(biome.absorb(intern_label(&format!("{}_positions", prefix)), positions))?;

        let features = convert(self.feature_tensor_internal())?;
        let feature_weight = (1.0 + self.params.meso_gain + self.params.micro_gain) as f32;
        convert(biome.absorb_weighted(
            intern_label(&format!("{}_features", prefix)),
            features,
            feature_weight,
        ))?;

        if include_roles {
            let roles = convert(self.role_tensor_internal())?;
            convert(biome.absorb(intern_label(&format!("{}_roles", prefix)), roles))?;
        }

        if include_reflections {
            let reflections = convert(self.reflection_tensor_internal())?;
            convert(biome.absorb(
                intern_label(&format!("{}_reflections", prefix)),
                reflections,
            ))?;
        }

        if !self.macros.is_empty() {
            let macros_tensor = convert(self.macro_summary_tensor_internal())?;
            let macro_weight = (self.macros.len() as f32).max(1.0);
            convert(biome.absorb_weighted(
                intern_label(&format!("{}_macro_summary", prefix)),
                macros_tensor,
                macro_weight,
            ))?;
        }

        Ok(())
    }
}

#[pymethods]
impl PySoT3DPlan {
    #[getter]
    fn total_steps(&self) -> usize {
        self.total_steps
    }

    #[getter]
    fn base_radius(&self) -> f64 {
        self.params.base_radius
    }

    #[getter]
    fn radial_growth(&self) -> f64 {
        self.params.radial_growth
    }

    #[getter]
    fn base_height(&self) -> f64 {
        self.params.base_height
    }

    #[getter]
    fn meso_gain(&self) -> f64 {
        self.params.meso_gain
    }

    #[getter]
    fn micro_gain(&self) -> f64 {
        self.params.micro_gain
    }

    fn steps(&self, py: Python<'_>) -> PyResult<Vec<Py<PySoT3DStep>>> {
        self.steps
            .iter()
            .cloned()
            .map(|step| Py::new(py, PySoT3DStep { step }))
            .collect()
    }

    fn as_dicts(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        let mut out = Vec::with_capacity(self.steps.len());
        for step in &self.steps {
            let py_step = Py::new(py, PySoT3DStep { step: step.clone() })?;
            let dict = py_step.bind(py).call_method0("as_dict")?;
            out.push(dict.into_py(py));
        }
        Ok(out)
    }

    fn as_tensor(&self) -> PyResult<PyTensor> {
        let tensor = self
            .positions_tensor()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(PyTensor::from_tensor(tensor))
    }

    fn feature_tensor(&self) -> PyResult<PyTensor> {
        let tensor = self
            .feature_tensor_internal()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(PyTensor::from_tensor(tensor))
    }

    fn role_tensor(&self) -> PyResult<PyTensor> {
        let tensor = self
            .role_tensor_internal()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(PyTensor::from_tensor(tensor))
    }

    fn reflection_tensor(&self) -> PyResult<PyTensor> {
        let tensor = self
            .reflection_tensor_internal()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(PyTensor::from_tensor(tensor))
    }

    fn macro_summary_tensor(&self) -> PyResult<PyTensor> {
        let tensor = self
            .macro_summary_tensor_internal()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(PyTensor::from_tensor(tensor))
    }

    fn macro_summaries(&self, py: Python<'_>) -> PyResult<Vec<Py<PyMacroSummary>>> {
        self.macros
            .iter()
            .cloned()
            .map(|summary| Py::new(py, PyMacroSummary { summary }))
            .collect()
    }

    fn polyline(&self) -> Vec<(f64, f64, f64)> {
        self.steps
            .iter()
            .map(|step| (step.x, step.y, step.height))
            .collect()
    }

    fn reflection_points(&self) -> Vec<(usize, &'static str)> {
        let mut points = Vec::new();
        for step in &self.steps {
            if step.micro_reflection {
                points.push((step.index, "micro"));
            }
            if step.meso_reflection {
                points.push((step.index, "meso"));
            }
            if step.macro_reflection {
                points.push((step.index, "macro"));
            }
        }
        points
    }

    #[pyo3(signature = (topos, label_prefix=None, include_reflections=true, include_roles=true))]
    fn grow_biome(
        &self,
        topos: &PyOpenTopos,
        label_prefix: Option<&str>,
        include_reflections: bool,
        include_roles: bool,
    ) -> PyResult<PyTensorBiome> {
        let prefix = label_prefix.unwrap_or("sot");
        let mut biome = TensorBiome::new(topos.inner.clone());
        self.deposit_into_biome(&mut biome, prefix, include_reflections, include_roles)?;
        Ok(PyTensorBiome::from_biome(biome))
    }

    #[pyo3(signature = (biome, label_prefix=None, include_reflections=true, include_roles=true))]
    fn infuse_biome(
        &self,
        biome: &mut PyTensorBiome,
        label_prefix: Option<&str>,
        include_reflections: bool,
        include_roles: bool,
    ) -> PyResult<()> {
        let prefix = label_prefix.unwrap_or("sot");
        self.deposit_into_biome(&mut biome.inner, prefix, include_reflections, include_roles)
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "SoT3DPlan(steps={}, base_radius={:.3}, growth={:.3})",
            self.total_steps, self.params.base_radius, self.params.radial_growth
        ))
    }
}

#[pyfunction]
#[pyo3(signature = (total_steps, base_radius=1.0, radial_growth=0.05, base_height=1.0, meso_gain=0.2, micro_gain=0.05))]
fn generate_plan(
    total_steps: usize,
    base_radius: f64,
    radial_growth: f64,
    base_height: f64,
    meso_gain: f64,
    micro_gain: f64,
) -> PyResult<PySoT3DPlan> {
    let params = Sot3DParams {
        base_radius,
        radial_growth,
        base_height,
        meso_gain,
        micro_gain,
    };
    build_plan(total_steps, params)
}

pub(crate) fn generate_plan_with_params(
    total_steps: usize,
    params: Sot3DParams,
) -> PyResult<PySoT3DPlan> {
    build_plan(total_steps, params)
}

#[pyfunction]
fn pack_tetranacci_chunks(length: usize) -> Vec<usize> {
    pack_tetranacci(length)
}

#[pyfunction]
fn pack_tribonacci_chunks(length: usize) -> Vec<usize> {
    pack_tribonacci(length)
}

#[pyfunction]
fn fibonacci_pacing(total_steps: usize) -> Vec<usize> {
    fibonacci_orbits(total_steps)
        .into_iter()
        .map(|orbit| orbit.actual)
        .collect()
}

#[pyfunction]
fn golden_angle() -> f64 {
    GOLDEN_ANGLE
}

#[pyfunction]
fn golden_ratio() -> f64 {
    GOLDEN_RATIO
}

pub fn module(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySoT3DPlan>()?;
    m.add_class::<PySoT3DStep>()?;
    m.add_class::<PyMacroSummary>()?;
    m.add_function(wrap_pyfunction!(generate_plan, m)?)?;
    m.add_function(wrap_pyfunction!(pack_tetranacci_chunks, m)?)?;
    m.add_function(wrap_pyfunction!(pack_tribonacci_chunks, m)?)?;
    m.add_function(wrap_pyfunction!(fibonacci_pacing, m)?)?;
    m.add_function(wrap_pyfunction!(golden_angle, m)?)?;
    m.add_function(wrap_pyfunction!(golden_ratio, m)?)?;
    m.setattr(
        "__all__",
        vec![
            "generate_plan",
            "pack_tetranacci_chunks",
            "pack_tribonacci_chunks",
            "fibonacci_pacing",
            "golden_angle",
            "golden_ratio",
            "SoT3DPlan",
            "SoT3DStep",
            "MacroSummary",
        ],
    )?;
    m.setattr(
        "__doc__",
        "SoT-3Dφ spiral reasoning kernel with Fibonacci/Tribonacci/Tetranacci pacing.",
    )?;
    Ok(())
}
