//! Bounded JSON-lines diagnostic: same exact kernels, host versus resident boundary.
use serde::Deserialize;
use serde_json::json;
use st_backend_wgpu::{
    rankk_exact_2ce::{dispatch_host, resident::ResidentRank, Kind, Pipelines, Plan},
    runtime,
};
use std::{
    io::{self, BufRead},
    time::Instant,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    kind: String,
    rows: u32,
    cols: u32,
    k: u32,
    tile: u32,
    input: Vec<f32>,
    seed: u64,
}

fn run(
    r: Request,
    runtime: &runtime::WgpuRuntime,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    if r.rows == 0
        || r.cols == 0
        || r.k == 0
        || r.k > r.cols
        || u64::from(r.rows) * u64::from(r.cols) > 1_048_576
        || r.input.len() as u64 != u64::from(r.rows) * u64::from(r.cols)
        || r.input.iter().any(|x| !x.is_finite())
    {
        return Err("invalid bounded rank request".into());
    }
    let kind = match r.kind.as_str() {
        "topk" => Kind::TopK,
        "midk" => Kind::MidK,
        "bottomk" => Kind::BottomK,
        _ => return Err("invalid kind".into()),
    };
    let plan = Plan::try_new(kind, r.rows, r.cols, r.k, r.tile)?;
    let context = runtime.context();
    let cold_start = Instant::now();
    let mut workspace = ResidentRank::new(runtime.clone(), plan)?;
    let create_ms = cold_start.elapsed().as_secs_f64() * 1000.0;
    let pipelines = Pipelines::new(context.device())?;
    workspace.upload(&r.input)?;
    workspace.dispatch(1)?;
    let actual = workspace.snapshot()?.read()?;
    let host = dispatch_host(
        context.device(),
        context.queue(),
        &pipelines,
        plan,
        &r.input,
    )?;
    if actual != host {
        return Err("resident versus host parity mismatch".into());
    }
    let mut samples = [Vec::new(), Vec::new(), Vec::new()];
    for block in 0..14 {
        // Rotate within a block to reduce fixed-order bias between native boundaries.
        for slot in 0..3 {
            let mode = (slot + block + r.seed as usize % 3) % 3;
            workspace.synchronize()?;
            let start = Instant::now();
            let reps = if mode == 2 { 16 } else { 1 };
            match mode {
                0 => {
                    std::hint::black_box(dispatch_host(
                        context.device(),
                        context.queue(),
                        &pipelines,
                        plan,
                        &r.input,
                    )?);
                }
                1 => {
                    workspace.upload(&r.input)?;
                    workspace.dispatch(1)?;
                    std::hint::black_box(workspace.snapshot()?.read()?);
                }
                _ => {
                    workspace.dispatch(reps)?;
                    workspace.synchronize()?;
                }
            }
            let ms = start.elapsed().as_secs_f64() * 1000.0 / f64::from(reps);
            if block >= 2 {
                samples[mode].push(ms);
            }
        }
    }
    let final_output = workspace.snapshot()?.read()?;
    if final_output != actual {
        return Err("post-timing output changed".into());
    }
    Ok(
        json!({"status":"passed", "kind":r.kind, "rows":r.rows,"cols":r.cols,"k":r.k,
        "tile":plan.tile_cols(), "seed":r.seed,"create_ms":create_ms,
        "adapter":{"name":runtime.adapter_info().name,"backend":format!("{:?}",runtime.adapter_info().backend)},
        "values":actual.values,"indices":actual.indices,
        "samples_ms":{"host_api":samples[0],"resident_host_to_host":samples[1],"resident_dispatch_fence_per_op":samples[2]},
        "resident_repetitions":16}),
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args == ["--build-info"] {
        println!(
            "{}",
            json!({"schema":"spiraltorch.native_build_identity.v1",
            "build_fingerprint":st_core::build_fingerprint(),
            "manifest":serde_json::from_str::<serde_json::Value>(st_core::build_manifest_json())?})
        );
        return Ok(());
    }
    if !args.is_empty() {
        return Err("usage: resident_rank_bench [--build-info]".into());
    }
    let (runtime, _) = runtime::ensure_default_runtime_blocking("resident.rank.bench")?;
    let mut failed = false;
    for line in io::stdin().lock().lines() {
        let line = line?;
        let result = serde_json::from_str(&line)
            .map_err(|e| e.into())
            .and_then(|r| run(r, &runtime));
        match result {
            Ok(value) => println!("{value}"),
            Err(error) => {
                failed = true;
                println!("{}", json!({"status":"error","error":error.to_string()}));
            }
        }
    }
    if failed {
        std::process::exit(1);
    }
    Ok(())
}
