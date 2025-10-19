// bindings/st-py/src/lib.rs

//! pyo3 entry for spiraltorch._native
//! NOTE: このファイルは “単一の” #[pymodule] をエクスポートします。
//! 既存で複数の #[pymodule] (nn, frac, ...) を出していた構成は衝突の原因。
//! サブモジュール化したい場合は Rust 側は 1つの _native に集約し、
//! その中で m.add_submodule(...) or m.add_function(...) でぶら下げます。

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::sync::atomic::{AtomicU64, Ordering};

// ========= extras（あなたが書いた関数群） =========

static GLOBAL_SEED: AtomicU64 = AtomicU64::new(0);

// GOLDEN_RATIO/GOLDEN_ANGLE は const fn を噛ませると安定します
const fn sqrt5() -> f64 {
    // 近似でも充分、ただし定数式なのでニュートン法で十分精度を確保
    // ここでは単純化のため、展開済みの近似値を使用（double精度で十分）
    2.236_067_977_499_79_f64
}
pub const GOLDEN_RATIO: f64 = (1.0 + sqrt5()) / 2.0;
pub const GOLDEN_ANGLE: f64 = 2.0 * std::f64::consts::PI * (1.0 - 1.0 / GOLDEN_RATIO);

#[pyfunction]
pub fn set_global_seed(seed: u64) {
    GLOBAL_SEED.store(seed, Ordering::SeqCst);
}

#[pyfunction]
pub fn golden_angle() -> f64 {
    GOLDEN_ANGLE
}

#[pyfunction]
pub fn golden_ratio() -> f64 {
    GOLDEN_RATIO
}

#[derive(Clone, Copy)]
struct OrbitLength { actual: usize, ideal: usize }

fn fibonacci_orbits(total_steps: usize) -> Vec<OrbitLength> {
    let mut a = 1usize;
    let mut b = 1usize;
    let mut remaining = total_steps;
    let mut out = Vec::new();
    while remaining > 0 {
        let ideal = b.max(1);
        let take = ideal.min(remaining);
        out.push(OrbitLength { actual: take, ideal });
        remaining -= take;
        let next = a + b;
        a = b;
        b = next;
    }
    out
}

#[pyfunction]
pub fn fibonacci_pacing(total_steps: usize) -> Vec<usize> {
    fibonacci_orbits(total_steps).into_iter().map(|o| o.actual).collect()
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
pub fn pack_nacci_chunks(order: usize, total_steps: usize) -> Vec<usize> {
    let mut seeds = vec![1usize; order.saturating_sub(1)];
    if order > 0 {
        seeds.push(2);
    }
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

// 既存 generate_plan に配線するなら、ここに紐づける
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
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "wire this to existing generate_plan()",
        ))
    }

    let base_seed = seed.unwrap_or_else(|| GLOBAL_SEED.load(Ordering::SeqCst));
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let s = base_seed.wrapping_add(i as u64);
        let plan_obj = call_generate_plan(
            py, total_steps, base_radius, radial_growth, base_height, meso_gain, micro_gain, Some(s),
        )?;
        out.push(plan_obj);
    }
    Ok(out)
}

// ここに “登録ヘルパ” をまとめる
fn register_extras(py: Python<'_>, m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(set_global_seed, m)?)?;
    m.add_function(wrap_pyfunction!(golden_angle, m)?)?;
    m.add_function(wrap_pyfunction!(golden_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(fibonacci_pacing, m)?)?;
    m.add_function(wrap_pyfunction!(pack_nacci_chunks, m)?)?;
    m.add_function(wrap_pyfunction!(pack_tribonacci_chunks, m)?)?;
    m.add_function(wrap_pyfunction!(pack_tetranacci_chunks, m)?)?;
    m.add_function(wrap_pyfunction!(generate_plan_batch_ex, m)?)?;
    Ok(())
}

// === ルートの pyo3 モジュール: spiraltorch._native ===
// Cargo.toml 側で module 名を "spiraltorch._native" に設定（後述）しておくと
// `from spiraltorch import *` で __init__.py 側から再エクスポートしやすい。
#[pymodule]
fn _native(py: Python<'_>, m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    // 既存のサブモジュール登録があるならここで追加してOK:
    // let sub = PyModule::new_bound(py, "nn")?; ...; m.add_submodule(&sub)?;
    register_extras(py, m)?;
    Ok(())
}
