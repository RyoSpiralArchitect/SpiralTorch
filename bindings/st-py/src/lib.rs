//! Minimal PyO3 bindings scaffold + extras
//! - No `mod xxx;` declarations (E0583/E0428回避)
//! - 既存の実装に段階的に繋げられるよう、まずは空モジュールで公開
//! - "extras" はここで登録（seed/golden/fibonacci/n-bonacci/chunk/plan-batch）

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

    // ---- constants (局所定義。将来コア側の定義に寄せるならここを差し替え) ----
    // sqrt は const で使えないため、定数値を埋め込みます。
    pub const GOLDEN_RATIO: f64 = 1.618_033_988_749_894_8_f64;
    pub const GOLDEN_ANGLE: f64 = 2.0 * std::f64::consts::PI / (GOLDEN_RATIO * GOLDEN_RATIO);

    // グローバルシード（明示指定がなければこれを使う）
    static GLOBAL_SEED: AtomicU64 = AtomicU64::new(0);

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

    // ---- orbit utilities ----
    #[derive(Clone, Copy, Debug)]
    struct OrbitLength {
        pub actual: usize,
        #[allow(dead_code)]
        pub ideal: usize,
    }

    // フィボナッチ系列のパッキング（F(1)=1, F(2)=1…）
    fn fibonacci_orbits(total_steps: usize) -> Vec<OrbitLength> {
        nacci_orbits(2, &[1, 1], total_steps)
    }

    // n-bonacci の理想長を生成しつつ、総ステップに収まるように実長(actual)を切り出す
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
            // 現在の理想長は window の合計
            let ideal: usize = window.iter().sum();
            let ideal = ideal.max(1); // 0 を避ける
            let take = ideal.min(remaining);
            out.push(OrbitLength { actual: take, ideal });
            remaining -= take;

            // 次の項 = 直近 order 個の合計
            window.pop_front();
            window.push_back(ideal);
        }
        out
    }

    #[pyfunction]
    pub fn fibonacci_pacing(total_steps: usize) -> Vec<usize> {
        fibonacci_orbits(total_steps)
            .into_iter()
            .map(|o| o.actual)
            .collect()
    }

    #[pyfunction]
    pub fn pack_nacci_chunks(order: usize, total_steps: usize) -> Vec<usize> {
        // デフォルトシード: [1, 1, ..., 2]（order-1 個の 1 と末尾 2）
        let mut seeds = vec![1usize; order.saturating_sub(1)];
        if order > 0 {
            seeds.push(2);
        }
        nacci_orbits(order, &seeds, total_steps)
            .into_iter()
            .map(|o| o.actual)
            .collect()
    }

    #[pyfunction]
    pub fn pack_tribonacci_chunks(total_steps: usize) -> Vec<usize> {
        let seeds = [1usize, 1, 2];
        nacci_orbits(3, &seeds, total_steps)
            .into_iter()
            .map(|o| o.actual)
            .collect()
    }

    #[pyfunction]
    pub fn pack_tetranacci_chunks(total_steps: usize) -> Vec<usize> {
        let seeds = [1usize, 1, 2, 4];
        nacci_orbits(4, &seeds, total_steps)
            .into_iter()
            .map(|o| o.actual)
            .collect()
    }

    // 既存の generate_plan を n 回呼ぶ軽量バッチ API。
    // 既存 generate_plan に seed 引数が無い場合でも、下の call_generate_plan を編集すればOK。
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
            // TODO: ここを既存の Rust/Python 実装に接続してください。
            // 例（Rust直呼び）:
            // let plan = crate::sot::generate_plan(...)?;
            // Ok(plan.into_py(py))
            Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "wire generate_plan_batch_ex() to your existing generate_plan()",
            ))
        }

        let base_seed = seed.unwrap_or_else(|| GLOBAL_SEED.load(Ordering::SeqCst));
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let s = base_seed.wrapping_add(i as u64);
            let plan_obj = call_generate_plan(
                py,
                total_steps,
                base_radius,
                radial_growth,
                base_height,
                meso_gain,
                micro_gain,
                Some(s),
            )?;
            out.push(plan_obj);
        }
        Ok(out)
    }

    // モジュール登録ヘルパ
    pub fn register(py: Python<'_>, m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(set_global_seed, m)?)?;
        m.add_function(wrap_pyfunction!(golden_angle, m)?)?;
        m.add_function(wrap_pyfunction!(golden_ratio, m)?)?;
        m.add_function(wrap_pyfunction!(fibonacci_pacing, m)?)?;
        m.add_function(wrap_pyfunction!(pack_nacci_chunks, m)?)?;
        m.add_function(wrap_pyfunction!(pack_tribonacci_chunks, m)?)?;
        m.add_function(wrap_pyfunction!(pack_tetranacci_chunks, m)?)?;
        m.add_function(wrap_pyfunction!(generate_plan_batch_ex, m)?)?;

        // 簡単なdoc
        m.add("__doc__", "SpiralTorch extras: seeds/golden/n-bonacci/chunking/plan-batch")?;
        let _ = py; // silence unused
        Ok(())
    }
}

// =======================
// PyO3 modules (空の器 + extras)
// =======================

// 1) ルート拡張モジュール
#[pymodule]
fn spiraltorch(py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    extras::register(py, m)?;
    Ok(())
}

// 2) 既存の下位モジュールたち（まずは空で公開。必要なら順次APIを追加）
#[pymodule]
fn nn(_py: Python, _m: &Bound<PyModule>) -> PyResult<()> { Ok(()) }

#[pymodule]
fn frac(_py: Python, _m: &Bound<PyModule>) -> PyResult<()> { Ok(()) }

#[pymodule]
fn dataset(_py: Python, _m: &Bound<PyModule>) -> PyResult<()> { Ok(()) }

#[pymodule]
fn linalg(_py: Python, _m: &Bound<PyModule>) -> PyResult<()> { Ok(()) }

#[pymodule]
fn rl(_py: Python, _m: &Bound<PyModule>) -> PyResult<()> { Ok(()) }

#[pymodule]
fn rec(_py: Python, _m: &Bound<PyModule>) -> PyResult<()> { Ok(()) }

#[pymodule]
fn telemetry(_py: Python, _m: &Bound<PyModule>) -> PyResult<()> { Ok(()) }

#[pymodule]
fn ecosystem(_py: Python, _m: &Bound<PyModule>) -> PyResult<()> { Ok(()) }
