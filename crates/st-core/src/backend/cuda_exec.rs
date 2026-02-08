// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#[cfg(feature = "cuda")]
use crate::backend::cuda_runtime;
use crate::backend::rankk_launch::with_registered_buffers_cuda;
use crate::backend::rankk_software::{run_selection, Selection};
use crate::ops::rank_entry::{RankKExecutor, RankPlan};

#[cfg(test)]
use crate::backend::rankk_launch::{with_launch_buffers_cuda, LaunchBuffers};

#[derive(Default)]
pub struct CudaExecutor;

impl RankKExecutor for CudaExecutor {
    type Error = String;
    fn launch_topk(&self, plan: &RankPlan) -> Result<(), Self::Error> {
        dispatch_topk(plan)
    }
    fn launch_midk(&self, plan: &RankPlan) -> Result<(), Self::Error> {
        dispatch_midk(plan)
    }
    fn launch_bottomk(&self, plan: &RankPlan) -> Result<(), Self::Error> {
        dispatch_bottomk(plan)
    }
}

fn is_two_ce(plan: &RankPlan) -> bool {
    let c = &plan.choice;
    // Future: if generated has two_ce_hint, weigh it here.
    c.use_2ce || (plan.cols as u64 >= (c.ctile.max(256) as u64) * 64)
}

fn dispatch_topk(plan: &RankPlan) -> Result<(), String> {
    run_cuda_selection(plan, Selection::Top)
}

fn dispatch_midk(plan: &RankPlan) -> Result<(), String> {
    let _ = is_two_ce(plan);
    run_cuda_selection(plan, Selection::Mid)
}
fn dispatch_bottomk(plan: &RankPlan) -> Result<(), String> {
    let _ = is_two_ce(plan);
    run_cuda_selection(plan, Selection::Bottom)
}

fn run_cuda_selection(plan: &RankPlan, selection: Selection) -> Result<(), String> {
    with_registered_buffers_cuda(|mut buffers| {
        #[cfg(feature = "cuda")]
        {
            match cuda_runtime::run_selection(selection, plan, &mut buffers) {
                Ok(()) => return Ok(()),
                Err(err) => {
                    if strict_gpu_path() {
                        return Err(format!("cuda launch failed ({err}); fallback disabled"));
                    }
                    return run_selection(selection, plan, buffers).map_err(|soft_err| {
                        format!(
                            "cuda launch failed ({err}); software fallback also failed: {soft_err}"
                        )
                    });
                }
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            return run_selection(selection, plan, buffers);
        }
    })
}

fn strict_gpu_path() -> bool {
    std::env::var("SPIRALTORCH_STRICT_GPU")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE"))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::device_caps::DeviceCaps;
    use crate::backend::rankk_launch::LaunchSlices;
    use crate::backend::unison_heuristics::RankKind;
    use crate::ops::rank_entry::plan_rank;
    use std::ffi::OsString;
    use std::sync::{Mutex, OnceLock};

    const ROWS: u32 = 2;
    const COLS: u32 = 5;
    const LARGE_COLS: u32 = 4097;

    fn plan(kind: RankKind, k: u32) -> RankPlan {
        plan_rank(
            kind,
            ROWS,
            COLS,
            k,
            DeviceCaps::cuda(32, 1024, Some(64 * 1024)),
        )
    }

    fn plan_with_shape(kind: RankKind, rows: u32, cols: u32, k: u32) -> RankPlan {
        plan_rank(
            kind,
            rows,
            cols,
            k,
            DeviceCaps::cuda(32, 1024, Some(64 * 1024)),
        )
    }

    fn sample_input() -> Vec<f32> {
        vec![
            1.0, 3.5, -2.0, 0.5, 7.0, // row 0
            -1.0, 4.0, 0.25, -3.0, 2.0, // row 1
        ]
    }

    fn sample_input_shape(rows: u32, cols: u32) -> Vec<f32> {
        let mut out = Vec::with_capacity((rows * cols) as usize);
        for r in 0..rows {
            for c in 0..cols {
                let centered = c as f32 - (cols as f32 * 0.5);
                out.push(centered + (r as f32) * 0.125);
            }
        }
        out
    }

    fn launch_buffers<'a>(
        input: &'a [f32],
        out_vals: &'a mut [f32],
        out_idx: &'a mut [i32],
        k: u32,
    ) -> LaunchBuffers<'a> {
        LaunchBuffers::new(input, ROWS, COLS, k, out_vals, out_idx).expect("valid launch buffers")
    }

    fn launch_buffers_shape<'a>(
        input: &'a [f32],
        rows: u32,
        cols: u32,
        out_vals: &'a mut [f32],
        out_idx: &'a mut [i32],
        k: u32,
    ) -> LaunchBuffers<'a> {
        LaunchBuffers::new(input, rows, cols, k, out_vals, out_idx).expect("valid launch buffers")
    }

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("env lock available")
    }

    struct EnvVarRestore {
        key: &'static str,
        previous: Option<OsString>,
    }

    impl EnvVarRestore {
        fn capture(key: &'static str) -> Self {
            Self {
                key,
                previous: std::env::var_os(key),
            }
        }
    }

    impl Drop for EnvVarRestore {
        fn drop(&mut self) {
            if let Some(prev) = &self.previous {
                unsafe { std::env::set_var(self.key, prev) };
            } else {
                unsafe { std::env::remove_var(self.key) };
            }
        }
    }

    fn with_strict_gpu_env<T>(value: Option<&str>, f: impl FnOnce() -> T) -> T {
        let _lock = env_lock();
        let _restore = EnvVarRestore::capture("SPIRALTORCH_STRICT_GPU");
        if let Some(value) = value {
            unsafe { std::env::set_var("SPIRALTORCH_STRICT_GPU", value) };
        } else {
            unsafe { std::env::remove_var("SPIRALTORCH_STRICT_GPU") };
        }
        f()
    }

    #[test]
    fn cuda_topk_selects_largest_values() {
        let plan = plan(RankKind::TopK, 2);
        let input = sample_input();
        let mut out_vals = vec![0.0f32; (ROWS * plan.k) as usize];
        let mut out_idx = vec![0i32; (ROWS * plan.k) as usize];

        with_launch_buffers_cuda(
            launch_buffers(&input, &mut out_vals, &mut out_idx, plan.k),
            || {
                CudaExecutor::default().launch_topk(&plan).unwrap();
            },
        );

        assert_eq!(out_vals, vec![7.0, 3.5, 4.0, 2.0]);
        assert_eq!(out_idx, vec![4, 1, 1, 4]);
    }

    #[test]
    fn cuda_midk_selects_central_band() {
        let plan = plan(RankKind::MidK, 2);
        let input = sample_input();
        let mut out_vals = vec![0.0f32; (ROWS * plan.k) as usize];
        let mut out_idx = vec![0i32; (ROWS * plan.k) as usize];

        with_launch_buffers_cuda(
            launch_buffers(&input, &mut out_vals, &mut out_idx, plan.k),
            || {
                CudaExecutor::default().launch_midk(&plan).unwrap();
            },
        );

        assert_eq!(out_vals, vec![0.5, 1.0, -1.0, 0.25]);
        assert_eq!(out_idx, vec![3, 0, 0, 2]);
    }

    #[test]
    fn cuda_bottomk_selects_smallest_values() {
        let plan = plan(RankKind::BottomK, 2);
        let input = sample_input();
        let mut out_vals = vec![0.0f32; (ROWS * plan.k) as usize];
        let mut out_idx = vec![0i32; (ROWS * plan.k) as usize];

        with_launch_buffers_cuda(
            launch_buffers(&input, &mut out_vals, &mut out_idx, plan.k),
            || {
                CudaExecutor::default().launch_bottomk(&plan).unwrap();
            },
        );

        assert_eq!(out_vals, vec![-2.0, 0.5, -3.0, -1.0]);
        assert_eq!(out_idx, vec![2, 3, 3, 0]);
    }

    #[test]
    fn cuda_errors_without_registered_buffers() {
        let err = CudaExecutor::default()
            .launch_topk(&plan(RankKind::TopK, 2))
            .unwrap_err();
        assert!(err.contains("no launch buffers"));
    }

    #[test]
    fn strict_gpu_env_defaults_to_false_when_unset() {
        with_strict_gpu_env(None, || {
            assert!(!strict_gpu_path());
        });
    }

    #[test]
    fn strict_gpu_env_accepts_truthy_values() {
        with_strict_gpu_env(Some("1"), || {
            assert!(strict_gpu_path());
        });
        with_strict_gpu_env(Some("true"), || {
            assert!(strict_gpu_path());
        });
        with_strict_gpu_env(Some("TRUE"), || {
            assert!(strict_gpu_path());
        });
    }

    #[test]
    fn strict_gpu_env_rejects_non_truthy_values() {
        with_strict_gpu_env(Some("0"), || {
            assert!(!strict_gpu_path());
        });
        with_strict_gpu_env(Some("false"), || {
            assert!(!strict_gpu_path());
        });
        with_strict_gpu_env(Some("yes"), || {
            assert!(!strict_gpu_path());
        });
    }

    #[test]
    fn cuda_midk_non_strict_falls_back_when_gpu_precheck_fails() {
        let rows = 1;
        let cols = LARGE_COLS;
        let k = 5;
        let plan = plan_with_shape(RankKind::MidK, rows, cols, k);
        let input = sample_input_shape(rows, cols);

        let mut expected_vals = vec![0.0f32; (rows * k) as usize];
        let mut expected_idx = vec![0i32; (rows * k) as usize];
        run_selection(
            Selection::Mid,
            &plan,
            LaunchSlices {
                input: &input,
                out_vals: &mut expected_vals,
                out_idx: &mut expected_idx,
                rows,
                cols,
                k,
            },
        )
        .expect("software midk reference should succeed");

        let mut out_vals = vec![0.0f32; (rows * k) as usize];
        let mut out_idx = vec![0i32; (rows * k) as usize];
        with_strict_gpu_env(Some("0"), || {
            with_launch_buffers_cuda(
                launch_buffers_shape(&input, rows, cols, &mut out_vals, &mut out_idx, k),
                || {
                    CudaExecutor::default()
                        .launch_midk(&plan)
                        .expect("non-strict mode should fallback to software");
                },
            );
        });

        assert_eq!(out_vals, expected_vals);
        assert_eq!(out_idx, expected_idx);
    }

    #[test]
    fn cuda_midk_strict_mode_blocks_fallback_when_gpu_precheck_fails() {
        let rows = 1;
        let cols = LARGE_COLS;
        let k = 5;
        let plan = plan_with_shape(RankKind::MidK, rows, cols, k);
        let input = sample_input_shape(rows, cols);

        let mut out_vals = vec![0.0f32; (rows * k) as usize];
        let mut out_idx = vec![0i32; (rows * k) as usize];
        let err = with_strict_gpu_env(Some("1"), || {
            with_launch_buffers_cuda(
                launch_buffers_shape(&input, rows, cols, &mut out_vals, &mut out_idx, k),
                || CudaExecutor::default().launch_midk(&plan),
            )
            .expect_err("strict mode should reject software fallback")
        });

        assert!(err.contains("fallback disabled"));
        assert!(err.contains("supports cols"));
    }
}
