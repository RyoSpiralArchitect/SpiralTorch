//! Minimal one-binary PyO3 module: `import spiraltorch`
//! - ルート1つだけ（#[pymodule] fn spiraltorch）
//! - サブモジュールは動的生成（nn/frac/... は空でも import 可）
//! - extras はトップレベルに直登録（UX良し）

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

//
// =======================
// extras (safe, self-contained)
// =======================
//
mod extras {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    pub const GOLDEN_RATIO: f64 = 1.618_033_988_749_894_8_f64;
    pub const GOLDEN_ANGLE: f64 = 2.0 * std::f64::consts::PI / (GOLDEN_RATIO * GOLDEN_RATIO);

    static GLOBAL_SEED: AtomicU64 = AtomicU64::new(0);

    #[pyfunction]
    pub fn set_global_seed(seed: u64) {
        GLOBAL_SEED.store(seed, Ordering::SeqCst);
    }

    #[pyfunction]
    pub fn golden_angle() -> f64 { GOLDEN_ANGLE }

    #[pyfunction]
    pub fn golden_ratio() -> f64 { GOLDEN_RATIO }

    #[derive(Clone, Copy, Debug)]
    struct OrbitLength { pub actual: usize, #[allow(dead_code)] pub ideal: usize }

    fn fibonacci_orbits(total_steps: usize) -> Vec<OrbitLength> {
        nacci_orbits(2, &[1, 1], total_steps)
    }

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
        fibonacci_orbits(total_steps).into_iter().map(|o| o.actual).collect()
    }

    #[pyfunction]
    pub fn pack_nacci_chunks(order: usize, total_steps: usize) -> Vec<usize> {
        let mut seeds = vec![1usize; order.saturating_sub(1)];
        if order > 0 { seeds.push(2); }
        nacci_orbits(order, &seeds, total_steps).into_iter().map(|o| o.actual).collect()
    }

    #[pyfunction]
    pub fn pack_tribonacci_chunks(total_steps: usize) -> Vec<usize> {
        let seeds = [1usize, 1, 2];
        nacci_orbits(3, &seeds, total_steps).into_iter().map(|o| o.actual).collect()
    }

    #[pyfunction]
    pub fn pack_tetranacci_chunks(total_steps: usize) -> Vec<usize> {
        let seeds = [1usize, 1, 2, 4];
        nacci_orbits(4, &seeds, total_steps).into_iter().map(|o| o.actual).collect()
    }

    #[pyfunction]
    #[pyo3(signature = (n, total_steps, base_radius, radial_growth, base_height, meso_gain, micro_gain, seed=None))]
    pub fn generate_plan_batch_ex(
        py: Python<'_>,
        n: usize,
        total_steps: usize,
        base_radius: f64,
        radial_growth: f64,
        base_height: f64,
        meso_gain: f64,
        micro_gain: f64,
        seed: Option<u64>,
    ) -> PyResult<Vec<PyObject>> {
        fn call_generate_plan(
            _py: Python<'_>,
            _total_steps: usize,
            _base_radius: f64,
            _radial_growth: f64,
            _base_height: f64,
            _meso_gain: f64,
            _micro_gain: f64,
            _seed: Option<u64>,
        ) -> PyResult<PyObject> {
            Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "wire generate_plan_batch_ex() to your existing generate_plan()",
            ))
        }

        let base_seed = seed.unwrap_or_else(|| GLOBAL_SEED.load(Ordering::SeqCst));
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let s = base_seed.wrapping_add(i as u64);
            let plan_obj = call_generate_plan(
                py, total_steps, base_radius, radial_growth, base_height, meso_gain, micro_gain, Some(s)
            )?;
            out.push(plan_obj);
        }
        Ok(out)
    }

    pub fn register(py: Python<'_>, m: &Bound<pyo3::types::PyModule>) -> PyResult<()> {
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

//
// =======================
// ルート拡張モジュール（唯一の #[pymodule]）
// =======================
//
#[pymodule]
fn spiraltorch(py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    // 1) トップレベル：extras を直登録（from spiraltorch import golden_ratio など）
    extras::register(py, m)?;

    // 2) サブモジュールを空でも作っておく（将来ここで各 crate の register を呼ぶ）
    //    これで `import spiraltorch.nn` などが今から可能になる。
    let nn = PyModule::new(py, "nn")?;
    // st_nn::py::register(py, &nn)?;  // ← 実装が出来次第ここで配線
    nn.add("__doc__", "SpiralTorch neural network primitives")?;
    m.add_submodule(&nn)?;

    let frac = PyModule::new(py, "frac")?;
    frac.add("__doc__", "Fractal & fractional tools")?;
    m.add_submodule(&frac)?;

    let dataset = PyModule::new(py, "dataset")?;
    dataset.add("__doc__", "Datasets & loaders")?;
    m.add_submodule(&dataset)?;

    let linalg = PyModule::new(py, "linalg")?;
    linalg.add("__doc__", "Linear algebra utilities")?;
    m.add_submodule(&linalg)?;

    let rl = PyModule::new(py, "rl")?;
    rl.add("__doc__", "Reinforcement learning components")?;
    m.add_submodule(&rl)?;

    let rec = PyModule::new(py, "rec")?;
    rec.add("__doc__", "Reconstruction / signal processing")?;
    m.add_submodule(&rec)?;

    let telemetry = PyModule::new(py, "telemetry")?;
    telemetry.add("__doc__", "Telemetry / dashboards / metrics")?;
    m.add_submodule(&telemetry)?;

    let ecosystem = PyModule::new(py, "ecosystem")?;
    ecosystem.add("__doc__", "Integrations & ecosystem glue")?;
    m.add_submodule(&ecosystem)?;

    // 3) 見栄え
    m.add("__all__", vec![
        "nn","frac","dataset","linalg","rl","rec","telemetry","ecosystem",
        // ルート直下の関数/クラスを列挙したければここに追加
        "golden_ratio","golden_angle","set_global_seed",
        "fibonacci_pacing","pack_nacci_chunks","pack_tribonacci_chunks","pack_tetranacci_chunks",
        "generate_plan_batch_ex",
    ])?;
    Ok(())
}
