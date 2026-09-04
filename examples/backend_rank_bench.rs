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
use st_core::runtime::blackcat::{bandit::SoftBanditMode, ChoiceGroups, MultiBandit};
use std::collections::HashMap;
use std::io::{self, BufRead};
use std::time::Instant;

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
    let context = base.spiralk_context().map_err(|e| e.to_string())?;
    let plans = request
        .scripts
        .iter()
        .map(|script| {
            let out = st_kdsl::eval_program(script, &context).map_err(|e| e.to_string())?;
            base.try_with_spiralk_hard(&out.hard)
                .map_err(|e| e.to_string())
        })
        .collect::<Result<Vec<_>, String>>()?;
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
    // Every candidate must execute correctly before it enters the bandit's domain.
    let mut first_call_ms = Vec::new();
    for plan in &plans {
        let start = Instant::now();
        let (values, indices) = launch(plan, backend, &request.input)?;
        first_call_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        check(&values, &indices)?;
        for _ in 0..request.warmup {
            launch(plan, backend, &request.input)?;
        }
    }
    let groups = ChoiceGroups {
        groups: HashMap::from([(
            "variant".to_owned(),
            (0..plans.len()).map(|i| i.to_string()).collect(),
        )]),
    };
    let mut bandit = MultiBandit::try_new_seeded(&groups, 1, SoftBanditMode::UCB, request.seed)
        .map_err(|e| e.to_string())?;
    let mut samples = Vec::new();
    for _ in 0..request.iterations {
        let (picks, _) = bandit.try_select_all(&[1.0]).map_err(|e| e.to_string())?;
        let index: usize = picks["variant"].parse().map_err(|e| format!("{e}"))?;
        let start = Instant::now();
        let result = launch(&plans[index], backend, &request.input);
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        let (values, indices) = match result {
            Ok(result) => result,
            Err(error) => {
                bandit.try_abandon_all().map_err(str::to_owned)?;
                return Err(error);
            }
        };
        if let Err(error) = check(&values, &indices) {
            bandit.try_abandon_all().map_err(str::to_owned)?;
            return Err(error);
        }
        let reward = 1.0 / (1.0 + elapsed_ms);
        bandit
            .try_update_all(&[1.0], reward)
            .map_err(str::to_owned)?;
        samples.push(json!({"variant": index, "elapsed_ms": elapsed_ms, "reward": reward}));
    }
    Ok(
        json!({"status": "passed", "backend": request.backend, "adapter": adapter,
        "kind": request.kind, "rows": request.rows, "cols": request.cols, "k": request.k,
        "scripts": request.scripts, "plans": plans.iter().map(RankPlan::snapshot).collect::<Vec<_>>(),
        "first_call_ms": first_call_ms, "samples": samples,
        "blackcat_observation_counts": bandit.observation_counts(),
        "values": expected_values, "indices": expected_indices,
        "boundary": "Rust host buffers -> strict GPU execution -> host buffers; planning excluded"}),
    )
}

fn main() {
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
        println!("{report}");
    }
    if failed {
        std::process::exit(1);
    }
}
