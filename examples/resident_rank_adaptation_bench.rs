//! Fixed-input resident rank controls and a separately observed Black Cat loop.
#[cfg(not(target_arch = "wasm32"))]
mod native {
    use serde::Deserialize;
    use serde_json::json;
    use st_backend_wgpu::{
        rankk_exact_2ce::{resident::ResidentRank, Plan},
        runtime,
    };
    use st_core::{
        backend::{
            device_caps::BackendKind,
            execution_plan::{AcceleratorFallback, ExecutionConfig},
            unison::RankKind,
        },
        ops::rank_entry::try_plan_rank_with_config,
        runtime::{
            blackcat::bandit::SoftBanditMode,
            rank_adaptation::{declared_native_rank_execution_signature, RankAdaptationSession},
        },
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
        scripts: Vec<String>,
        input: Vec<f32>,
        seed: u64,
        policy: String,
        rounds: u32,
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
            || r.rounds == 0
            || r.rounds > 256
        {
            return Err("invalid bounded resident adaptation request".into());
        }
        let kind = r.kind.parse::<RankKind>()?;
        let policy = match r.policy.as_str() {
            "ucb" => SoftBanditMode::UCB,
            "thompson_sampling" => SoftBanditMode::TS,
            _ => return Err("policy must be ucb or thompson_sampling".into()),
        };
        let base = try_plan_rank_with_config(
            kind,
            r.rows,
            r.cols,
            r.k,
            BackendKind::Wgpu.default_caps(),
            ExecutionConfig::new(AcceleratorFallback::Forbid, 1024),
        )?;
        let mut session = RankAdaptationSession::try_from_spiralk_with_execution_signature(
            &base,
            &r.scripts,
            policy,
            r.seed,
            declared_native_rank_execution_signature,
        )?;
        let initial = session.snapshot();
        let mut expected_values = Vec::new();
        let mut expected_indices = Vec::new();
        for row in r.input.chunks_exact(r.cols as usize) {
            let mut ids: Vec<usize> = (0..row.len()).collect();
            ids.sort_by(|&a, &b| {
                (if kind == RankKind::TopK {
                    row[b].total_cmp(&row[a])
                } else {
                    row[a].total_cmp(&row[b])
                })
                .then(a.cmp(&b))
            });
            let start = if kind == RankKind::MidK {
                (row.len() - r.k as usize) / 2
            } else {
                0
            };
            for &id in &ids[start..start + r.k as usize] {
                expected_values.push(row[id]);
                expected_indices.push(id as i32);
            }
        }
        let validate = |workspace: &ResidentRank| -> Result<(), Box<dyn std::error::Error>> {
            let output = workspace.snapshot()?.read()?;
            if output.indices != expected_indices || output.values != expected_values {
                return Err("resident candidate differs from exact Rust reference".into());
            }
            Ok(())
        };
        let mut workspaces = Vec::new();
        let mut create_ms = Vec::new();
        for index in 0..initial.candidates.len() {
            let spec = session.wgpu_resident_candidate(index)?;
            let start = Instant::now();
            let plan = Plan::try_new(spec.kind, spec.rows, spec.cols, spec.k, spec.tile_cols)?;
            let mut workspace = ResidentRank::new(runtime.clone(), plan)?;
            create_ms.push(start.elapsed().as_secs_f64() * 1000.0);
            workspace.upload(&r.input)?;
            workspace.dispatch(1)?;
            validate(&workspace)?;
            workspaces.push(workspace);
        }
        // No constructor, upload, readback, or policy work is inside these intervals.
        let measure = |workspace: &mut ResidentRank| -> Result<f64, Box<dyn std::error::Error>> {
            workspace.synchronize()?;
            let start = Instant::now();
            workspace.dispatch(16)?;
            workspace.synchronize()?;
            Ok(start.elapsed().as_secs_f64() * 1000.0)
        };
        let mut controls = vec![Vec::new(); workspaces.len()];
        for block in 0..14 {
            for slot in 0..workspaces.len() {
                let index = (slot + block + r.seed as usize % workspaces.len()) % workspaces.len();
                let elapsed_ms = measure(&mut workspaces[index])?;
                validate(&workspaces[index])?;
                if block >= 2 {
                    controls[index].push(elapsed_ms / 16.0);
                }
            }
        }
        for workspace in &workspaces {
            validate(workspace)?;
        }
        let mut observations = Vec::new();
        for _ in 0..r.rounds {
            let selection = session.try_choose()?;
            let receipt = selection.receipt();
            let index = receipt.candidate_index;
            let elapsed_ms = match measure(&mut workspaces[index]) {
                Ok(elapsed_ms) => elapsed_ms,
                Err(error) => {
                    session.try_abandon(receipt.selection_id)?;
                    return Err(error);
                }
            };
            // Correctness is checked outside timing, before this measurement is credited.
            let correctness = validate(&workspaces[index]);
            let observation =
                session.try_observe(receipt.selection_id, elapsed_ms, correctness.is_ok())?;
            observations.push(json!({"selection":receipt,"observation":observation,"per_op_ms":elapsed_ms / 16.0}));
            if let Err(error) = correctness {
                return Ok(
                    json!({"status":"error","stage":"adaptive_correctness","error":error.to_string(),
                    "observations":observations,"rank_adaptation":session.snapshot()}),
                );
            }
        }
        for workspace in &workspaces {
            validate(workspace)?;
        }
        Ok(
            json!({"status":"passed","schema":"spiraltorch.resident_rank_adaptation.v1",
            "kind":r.kind,"rows":r.rows,"cols":r.cols,"k":r.k,"seed":r.seed,"policy":r.policy,
            "adapter":{"name":runtime.adapter_info().name,"backend":format!("{:?}",runtime.adapter_info().backend)},
            "repetitions":16,"reward_boundary":"elapsed_ms is one 16-pair dispatch plus completion fence, not per-op latency",
            "validation_boundary":"every candidate before controls, after every control batch, every adaptive observation before credit, and final output",
            "initial":initial,"rank_adaptation":session.snapshot(),"observations":observations,
            "create_ms":create_ms,"control_samples_per_op_ms":controls,
            "values":expected_values,"indices":expected_indices} ),
        )
    }

    pub fn main() -> Result<(), Box<dyn std::error::Error>> {
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
            return Err("usage: resident_rank_adaptation_bench [--build-info]".into());
        }
        let (runtime, _) = runtime::ensure_default_runtime_blocking("resident.rank.adaptation")?;
        let mut failed = false;
        for line in io::stdin().lock().lines() {
            let line = line?;
            let result = serde_json::from_str(&line)
                .map_err(|e| e.into())
                .and_then(|r| run(r, &runtime));
            match result {
                Ok(value) => {
                    failed |= value["status"] != "passed";
                    println!("{value}");
                }
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
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        native::main()
    }
    #[cfg(target_arch = "wasm32")]
    {
        Err("use the browser resident rank adaptation fixture on WASM".into())
    }
}
