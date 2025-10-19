//! Minimal one-binary PyO3 module: `import spiraltorch`
//! - ルート1つだけ（#[pymodule] fn spiraltorch）
//! - サブモジュールは動的生成（nn/frac/... は import 可）
//! - extras はトップレベルに直登録

use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::wrap_pyfunction;

// =======================
// extras（安全・自己完結）
// =======================
mod extras {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    pub const GOLDEN_RATIO: f64 = 1.618_033_988_749_894_8_f64;
    pub const GOLDEN_ANGLE: f64 = 2.0 * std::f64::consts::PI / (GOLDEN_RATIO * GOLDEN_RATIO);

    static GLOBAL_SEED: AtomicU64 = AtomicU64::new(0);

    #[pyfunction]
    pub fn set_global_seed(seed: u64) { GLOBAL_SEED.store(seed, Ordering::SeqCst); }

    #[pyfunction]
    pub fn golden_angle() -> f64 { GOLDEN_ANGLE }

    #[pyfunction]
    pub fn golden_ratio() -> f64 { GOLDEN_RATIO }

    #[derive(Clone, Copy, Debug)]
    struct OrbitLength { pub actual: usize, #[allow(dead_code)] pub ideal: usize }

    fn nacci_orbits(order: usize, seeds: &[usize], total_steps: usize) -> Vec<OrbitLength> {
        use std::collections::VecDeque;
        assert!(order > 0, "order must be >= 1");
        let mut window: VecDeque<usize> = if seeds.len() >= order {
            seeds[seeds.len() - order..].iter().copied().collect()
        } else {
            let mut w = vec![1usize; order - seeds.len()];
            w.extend_from_slice(seeds);
            w.into_iter().collect()
        };
        let mut remaining = total_steps;
        let mut out = Vec::new();
        while remaining > 0 {
            let ideal: usize = window.iter().sum::<usize>().max(1);
            let take = ideal.min(remaining);
            out.push(OrbitLength { actual: take, ideal });
            remaining -= take;
            window.pop_front();
            window.push_back(ideal);
        }
        out
    }

    #[pyfunction]
    pub fn fibonacci_pacing(total_steps: usize) -> Vec<usize> {
        nacci_orbits(2, &[1, 1], total_steps).into_iter().map(|o| o.actual).collect()
    }

    #[pyfunction]
    pub fn pack_nacci_chunks(order: usize, total_steps: usize) -> Vec<usize> {
        let mut seeds = vec![1usize; order.saturating_sub(1)];
        if order > 0 { seeds.push(2); }
        nacci_orbits(order, &seeds, total_steps).into_iter().map(|o| o.actual).collect()
    }

    #[pyfunction]
    pub fn pack_tribonacci_chunks(total_steps: usize) -> Vec<usize> {
        nacci_orbits(3, &[1, 1, 2], total_steps).into_iter().map(|o| o.actual).collect()
    }

    #[pyfunction]
    pub fn pack_tetranacci_chunks(total_steps: usize) -> Vec<usize> {
        nacci_orbits(4, &[1, 1, 2, 4], total_steps).into_iter().map(|o| o.actual).collect()
    }

    // 既存 generate_plan へ配線するための薄い入口（今は NotImplemented）
    #[pyfunction]
    #[pyo3(signature = (n, total_steps, base_radius, radial_growth, base_height, meso_gain, micro_gain, seed=None))]
    pub fn generate_plan_batch_ex(
        _py: Python<'_>,
        _n: usize,
        _total_steps: usize,
        _base_radius: f64,
        _radial_growth: f64,
        _base_height: f64,
        _meso_gain: f64,
        _micro_gain: f64,
        _seed: Option<u64>,
    ) -> PyResult<Vec<PyObject>> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "wire generate_plan_batch_ex() to your existing generate_plan()",
        ))
    }

    pub fn register(py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(set_global_seed, m)?)?;
        m.add_function(wrap_pyfunction!(golden_angle, m)?)?;
        m.add_function(wrap_pyfunction!(golden_ratio, m)?)?;
        m.add_function(wrap_pyfunction!(fibonacci_pacing, m)?)?;
        m.add_function(wrap_pyfunction!(pack_nacci_chunks, m)?)?;
        m.add_function(wrap_pyfunction!(pack_tribonacci_chunks, m)?)?;
        m.add_function(wrap_pyfunction!(pack_tetranacci_chunks, m)?)?;
        m.add_function(wrap_pyfunction!(generate_plan_batch_ex, m)?)?;
        m.add("__doc__", "SpiralTorch extras: seeds/golden/n-bonacci/chunking/plan-batch")?;
        let _ = py;
        Ok(())
    }
}

// =======================
// st-frac の実APIを公開
// =======================
mod frac_bindings {
    use super::*;
    use st_frac::{Pad, gl_coeffs_adaptive as gl_coeffs_adaptive_rs, fracdiff_gl_1d as fracdiff_gl_1d_rs};

    #[pyfunction]
    #[pyo3(signature = (alpha, tol=1e-6, max_len=8192))]
    fn gl_coeffs_adaptive(alpha: f32, tol: f32, max_len: usize) -> Vec<f32> {
        gl_coeffs_adaptive_rs(alpha, tol, max_len)
    }

    #[pyfunction]
    #[pyo3(signature = (x, alpha, kernel_len, pad="zero", pad_constant=None, scale=None))]
    fn fracdiff_gl_1d(
        x: Vec<f32>,
        alpha: f32,
        kernel_len: usize,
        pad: &str,
        pad_constant: Option<f32>,
        scale: Option<f32>,
    ) -> PyResult<Vec<f32>> {
        let pad = match pad.to_ascii_lowercase().as_str() {
            "zero"     => Pad::Zero,
            "reflect"  => Pad::Reflect,
            "constant" => Pad::Constant(pad_constant.unwrap_or(0.0)),
            other => return Err(pyo3::exceptions::PyValueError::new_err(
                format!("unknown pad '{other}', expected 'zero'|'reflect'|'constant'")))
        };
        fracdiff_gl_1d_rs(&x, alpha, kernel_len, pad, scale)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))
    }

    pub fn register(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(gl_coeffs_adaptive, m)?)?;
        m.add_function(wrap_pyfunction!(fracdiff_gl_1d, m)?)?;
        m.add("__doc__", "Fractional differencing (Grünwald–Letnikov) and helpers.")?;
        Ok(())
    }
}

// =======================
// ルート #[pymodule]
// =======================
#[pymodule]
fn spiraltorch(py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    // 1) トップレベル（そのまま import できる）
    extras::register(py, m)?;

    // 2) サブモジュール（nn/frac/...）
    //    PyO3 0.22 以降は new_bound / Bound<PyModule> を使う
    let nn = PyModule::new_bound(py, "nn")?;
    nn.add("__doc__", "SpiralTorch neural network primitives")?;
    m.add_submodule(&nn)?;

    let frac = PyModule::new_bound(py, "frac")?;
    frac_bindings::register(py, &frac)?; // ← ここで実APIを公開
    m.add_submodule(&frac)?;

    let dataset = PyModule::new_bound(py, "dataset")?;
    dataset.add("__doc__", "Datasets & loaders")?;
    m.add_submodule(&dataset)?;

    let linalg = PyModule::new_bound(py, "linalg")?;
    linalg.add("__doc__", "Linear algebra utilities")?;
    m.add_submodule(&linalg)?;

    let rl = PyModule::new_bound(py, "rl")?;
    rl.add("__doc__", "Reinforcement learning components")?;
    m.add_submodule(&rl)?;

    let rec = PyModule::new_bound(py, "rec")?;
    rec.add("__doc__", "Reconstruction / signal processing")?;
    m.add_submodule(&rec)?;

    let telemetry = PyModule::new_bound(py, "telemetry")?;
    telemetry.add("__doc__", "Telemetry / dashboards / metrics")?;
    m.add_submodule(&telemetry)?;

    let ecosystem = PyModule::new_bound(py, "ecosystem")?;
    ecosystem.add("__doc__", "Integrations & ecosystem glue")?;
    m.add_submodule(&ecosystem)?;

    // 3) __all__（見栄え）
    m.add("__all__", vec![
        "nn","frac","dataset","linalg","rl","rec","telemetry","ecosystem",
        "golden_ratio","golden_angle","set_global_seed",
        "fibonacci_pacing","pack_nacci_chunks","pack_tribonacci_chunks","pack_tetranacci_chunks",
        "generate_plan_batch_ex",
        "gl_coeffs_adaptive","fracdiff_gl_1d",
    ])?;
    Ok(())
}
