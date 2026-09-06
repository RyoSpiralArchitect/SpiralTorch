//! JSON-lines input/output for strict native rank benchmarking. No fallback.
use serde::Deserialize;
use serde_json::json;
use st_core::backend::cuda_exec::CudaExecutor;
use st_core::backend::device_caps::BackendKind;
use st_core::backend::execution_plan::{AcceleratorFallback, ExecutionConfig};
use st_core::backend::rankk_launch::{
    with_launch_buffers_cuda, with_launch_buffers_wgpu, LaunchBuffers,
};
use st_core::backend::unison_heuristics::RankKind;
use st_core::backend::wgpu_exec::WgpuExecutor;
use st_core::ops::rank_entry::{execute_rank, try_plan_rank_with_config, RankPlan};
use st_core::runtime::blackcat::bandit::SoftBanditMode;
use st_core::runtime::rank_adaptation::RankAdaptationSession;
use std::ffi::OsStr;
use std::io::{self, BufRead};
use std::time::Instant;

const BUILD_IDENTITY_SCHEMA: &str = "spiraltorch.native_build_identity.v1";

fn build_identity() -> Result<serde_json::Value, String> {
    let manifest = serde_json::from_str::<serde_json::Value>(st_core::build_manifest_json())
        .map_err(|error| format!("invalid embedded build manifest: {error}"))?;
    Ok(json!({
        "schema": BUILD_IDENTITY_SCHEMA,
        "build_fingerprint": st_core::build_fingerprint(),
        "manifest": manifest,
    }))
}

#[derive(Deserialize)]
struct Request {
    backend: String,
    kind: String,
    rows: u32,
    cols: u32,
    k: u32,
    input: Vec<f32>,
    iterations: usize,
    warmup: usize,
    seed: u64,
    // Multiple scripts enable a seeded Black Cat selector, with measured reward.
    scripts: Vec<String>,
}

fn launch(
    plan: &RankPlan,
    backend: BackendKind,
    input: &[f32],
) -> Result<(Vec<f32>, Vec<i32>), String> {
    let len = plan.rows as usize * plan.k as usize;
    let mut values = vec![0.; len];
    let mut indices = vec![0; len];
    let buffers = LaunchBuffers::new(
        input,
        plan.rows,
        plan.cols,
        plan.k,
        &mut values,
        &mut indices,
    )?;
    match backend {
        BackendKind::Cuda => {
            with_launch_buffers_cuda(buffers, || execute_rank(&CudaExecutor, plan))?
        }
        BackendKind::Wgpu => {
            with_launch_buffers_wgpu(buffers, || execute_rank(&WgpuExecutor, plan))?
        }
        _ => return Err("only explicit cuda or wgpu is supported".into()),
    }
    Ok((values, indices))
}

fn benchmark(request: Request) -> Result<serde_json::Value, String> {
    let backend = match request.backend.as_str() {
        "cuda" => BackendKind::Cuda,
        "wgpu" => BackendKind::Wgpu,
        _ => return Err("only explicit cuda or wgpu is supported".into()),
    };
    let kind: RankKind = request.kind.parse().map_err(|e| format!("{e}"))?;
    if request.rows == 0
        || request.cols == 0
        || request.k == 0
        || request.k > request.cols
        || u64::from(request.rows) * u64::from(request.cols) > 1_048_576
        || request.input.len() as u64 != u64::from(request.rows) * u64::from(request.cols)
        || request.input.iter().any(|x| !x.is_finite())
        || request.iterations == 0
        || request.iterations > 1000
        || request.warmup > 100
        || request.scripts.is_empty()
        || request.scripts.len() > 8
    {
        return Err("invalid or oversized benchmark request".into());
    }
    let adapter = if backend == BackendKind::Wgpu {
        st_core::backend::wgpu_rt::ensure_default_ctx()?;
        let (runtime, _) =
            st_backend_wgpu::runtime::ensure_default_runtime_blocking("rank benchmark")
                .map_err(|e| e.to_string())?;
        let info = runtime.adapter_info();
        json!({"name": info.name, "backend": format!("{:?}", info.backend),
            "vendor": info.vendor, "device": info.device, "driver": info.driver, "driver_info": info.driver_info})
    } else {
        json!(null)
    };
    let base = try_plan_rank_with_config(
        kind,
        request.rows,
        request.cols,
        request.k,
        backend.default_caps(),
        ExecutionConfig::new(AcceleratorFallback::Forbid, 1024),
    )
    .map_err(|e| e.to_string())?;
    let mut adaptation = RankAdaptationSession::try_from_spiralk(
        &base,
        &request.scripts,
        SoftBanditMode::UCB,
        request.seed,
    )
    .map_err(|e| e.to_string())?;
    let mut expected_values = Vec::new();
    let mut expected_indices = Vec::new();
    for row in request.input.chunks_exact(request.cols as usize) {
        let mut order: Vec<usize> = (0..row.len()).collect();
        order.sort_by(|&a, &b| {
            let cmp = if kind == RankKind::TopK {
                row[b].total_cmp(&row[a])
            } else {
                row[a].total_cmp(&row[b])
            };
            cmp.then(a.cmp(&b))
        });
        let start = if kind == RankKind::MidK {
            (row.len() - request.k as usize) / 2
        } else {
            0
        };
        for &id in &order[start..start + request.k as usize] {
            expected_values.push(row[id]);
            expected_indices.push(id as i32);
        }
    }
    let check = |values: &[f32], indices: &[i32]| {
        if values == expected_values && indices == expected_indices {
            Ok(())
        } else {
            Err(format!("rank correctness failed: values={values:?}, indices={indices:?}; expected indices={expected_indices:?}"))
        }
    };
    // No Black Cat decision is admitted until every candidate passes once.
    let candidate_plans = adaptation.candidate_plans().cloned().collect::<Vec<_>>();
    let candidate_snapshots = adaptation.snapshot().candidates;
    let mut first_call_ms = Vec::new();
    for (candidate_index, plan) in candidate_plans.iter().enumerate() {
        let start = Instant::now();
        let result = launch(plan, backend, &request.input);
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        first_call_ms.push(json!({
            "candidate_index": candidate_index,
            "elapsed_ms": elapsed_ms,
        }));
        let (values, indices) = match result {
            Ok(result) => result,
            Err(error) => {
                return Ok(json!({
                    "status": "error",
                    "stage": "candidate_preflight_launch",
                    "error": error,
                    "candidate": candidate_snapshots[candidate_index],
                    "elapsed_ms": elapsed_ms,
                    "first_call_ms": first_call_ms,
                    "rank_adaptation": adaptation.snapshot(),
                }));
            }
        };
        if let Err(error) = check(&values, &indices) {
            return Ok(json!({
                "status": "error",
                "stage": "candidate_preflight_correctness",
                "error": error,
                "candidate": candidate_snapshots[candidate_index],
                "elapsed_ms": elapsed_ms,
                "actual_values": values,
                "actual_indices": indices,
                "expected_values": expected_values,
                "expected_indices": expected_indices,
                "first_call_ms": first_call_ms,
                "rank_adaptation": adaptation.snapshot(),
            }));
        }
        for warmup_index in 0..request.warmup {
            let result = launch(plan, backend, &request.input);
            let (values, indices) = match result {
                Ok(result) => result,
                Err(error) => {
                    return Ok(json!({
                        "status": "error",
                        "stage": "candidate_warmup_launch",
                        "error": error,
                        "candidate": candidate_snapshots[candidate_index],
                        "warmup_index": warmup_index,
                        "first_call_ms": first_call_ms,
                        "rank_adaptation": adaptation.snapshot(),
                    }));
                }
            };
            if let Err(error) = check(&values, &indices) {
                return Ok(json!({
                    "status": "error",
                    "stage": "candidate_warmup_correctness",
                    "error": error,
                    "candidate": candidate_snapshots[candidate_index],
                    "warmup_index": warmup_index,
                    "actual_values": values,
                    "actual_indices": indices,
                    "expected_values": expected_values,
                    "expected_indices": expected_indices,
                    "first_call_ms": first_call_ms,
                    "rank_adaptation": adaptation.snapshot(),
                }));
            }
        }
    }

    // Measure every arm equally in a rotating order. These control samples are
    // separate from the adaptive observations and never train the posterior.
    let mut control_samples = Vec::new();
    let candidate_count = candidate_plans.len();
    let rotation = request.seed as usize % candidate_count;
    for round in 0..request.iterations {
        for offset in 0..candidate_count {
            let candidate_index = (rotation + round + offset) % candidate_count;
            let plan = &candidate_plans[candidate_index];
            let started = Instant::now();
            let result = launch(plan, backend, &request.input);
            let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
            let (values, indices) = match result {
                Ok(result) => result,
                Err(error) => {
                    return Ok(json!({
                        "status": "error",
                        "stage": "balanced_control_launch",
                        "error": error,
                        "candidate": candidate_snapshots[candidate_index],
                        "round": round,
                        "elapsed_ms": elapsed_ms,
                        "rank_adaptation": adaptation.snapshot(),
                    }));
                }
            };
            if let Err(error) = check(&values, &indices) {
                return Ok(json!({
                    "status": "error",
                    "stage": "balanced_control_correctness",
                    "error": error,
                    "candidate": candidate_snapshots[candidate_index],
                    "round": round,
                    "elapsed_ms": elapsed_ms,
                    "actual_values": values,
                    "actual_indices": indices,
                    "expected_values": expected_values,
                    "expected_indices": expected_indices,
                    "rank_adaptation": adaptation.snapshot(),
                }));
            }
            control_samples.push(json!({
                "round": round,
                "order_offset": offset,
                "candidate_index": candidate_index,
                "execution_signature": candidate_snapshots[candidate_index].execution_signature,
                "elapsed_ms": elapsed_ms,
            }));
        }
    }
    let mut samples = Vec::new();
    for _ in 0..request.iterations {
        let selection = adaptation.try_choose().map_err(|e| e.to_string())?;
        let selection_id = selection.receipt().selection_id;
        let start = Instant::now();
        let result = launch(selection.plan(), backend, &request.input);
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        let (values, indices) = match result {
            Ok(result) => result,
            Err(error) => {
                let abandonment = adaptation
                    .try_abandon(selection_id)
                    .map(|receipt| json!(receipt))
                    .unwrap_or_else(|receipt_error| {
                        json!({"status": "error", "error": receipt_error.to_string()})
                    });
                return Ok(json!({
                    "status": "error",
                    "stage": "adaptive_launch",
                    "error": error,
                    "selection": selection.receipt(),
                    "abandonment": abandonment,
                    "elapsed_ms": elapsed_ms,
                    "rank_adaptation": adaptation.snapshot(),
                    "control_samples": control_samples,
                }));
            }
        };
        let correctness = check(&values, &indices);
        let observation = adaptation
            .try_observe(selection_id, elapsed_ms, correctness.is_ok())
            .map_err(|e| e.to_string())?;
        if let Err(error) = correctness {
            return Ok(json!({
                "status": "error",
                "stage": "adaptive_correctness",
                "error": error,
                "selection": selection.receipt(),
                "observation": observation,
                "actual_values": values,
                "actual_indices": indices,
                "expected_values": expected_values,
                "expected_indices": expected_indices,
                "rank_adaptation": adaptation.snapshot(),
                "control_samples": control_samples,
            }));
        }
        samples.push(json!({"selection": selection.receipt(), "observation": observation}));
    }
    let adaptation_snapshot = adaptation.snapshot();
    let observation_counts = adaptation_snapshot.observation_counts.clone();
    Ok(
        json!({"status": "passed", "backend": request.backend, "adapter": adapter,
        "kind": request.kind, "rows": request.rows, "cols": request.cols, "k": request.k,
        "scripts": request.scripts, "rank_adaptation": adaptation_snapshot,
        "first_call_ms": first_call_ms,
        "control_design": "equal-count rotating round-robin; does not update Black Cat",
        "control_samples": control_samples, "samples": samples,
        "blackcat_observation_counts": observation_counts,
        "values": expected_values, "indices": expected_indices,
        "boundary": "Rust host buffers -> strict GPU execution -> host buffers; planning excluded"}),
    )
}

fn main() {
    let mut args = std::env::args_os().skip(1);
    match (args.next(), args.next()) {
        (Some(flag), None) if flag == OsStr::new("--build-info") => {
            match build_identity() {
                Ok(identity) => println!("{identity}"),
                Err(error) => {
                    eprintln!("{error}");
                    std::process::exit(1);
                }
            }
            return;
        }
        (None, None) => {}
        _ => {
            eprintln!("usage: backend_rank_bench [--build-info]");
            std::process::exit(2);
        }
    }

    let mut failed = false;
    for line in io::stdin().lock().lines() {
        let result = line
            .map_err(|e| e.to_string())
            .and_then(|line| serde_json::from_str(&line).map_err(|e| e.to_string()))
            .and_then(benchmark);
        let report = result.unwrap_or_else(|error| {
            failed = true;
            json!({"status": "error", "error": error})
        });
        if report.get("status").and_then(|value| value.as_str()) == Some("error") {
            failed = true;
        }
        println!("{report}");
    }
    if failed {
        std::process::exit(1);
    }
}
