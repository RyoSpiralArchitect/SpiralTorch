//! Source-bound, paired pass timestamps and uninstrumented host boundaries.

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use serde::Deserialize;
    use serde_json::{json, Value};
    use st_backend_wgpu::{
        rankk_exact_2ce::{resident::ResidentRank, Kind, Plan},
        runtime::WgpuRuntime,
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
    }

    fn run(r: Request, runtime: &WgpuRuntime) -> Result<Value, Box<dyn std::error::Error>> {
        let kind = match r.kind.as_str() {
            "topk" => Kind::TopK,
            "midk" => Kind::MidK,
            "bottomk" => Kind::BottomK,
            _ => return Err("invalid rank kind".into()),
        };
        let plan = Plan::try_new(kind, r.rows, r.cols, r.k, r.tile)?;
        if plan.is_empty()
            || r.input.len() != plan.input_elements() as usize
            || r.input.iter().any(|x| !x.is_finite())
        {
            return Err("profile fixture must be nonempty, finite, and match the plan".into());
        }
        let mut values = Vec::new();
        let mut indices = Vec::new();
        for row in r.input.chunks_exact(r.cols as usize) {
            let mut ids: Vec<usize> = (0..row.len()).collect();
            ids.sort_by(|&a, &b| {
                (if kind == Kind::TopK {
                    row[b].total_cmp(&row[a])
                } else {
                    row[a].total_cmp(&row[b])
                })
                .then(a.cmp(&b))
            });
            let start = if kind == Kind::MidK {
                (row.len() - r.k as usize) / 2
            } else {
                0
            };
            for &id in &ids[start..start + r.k as usize] {
                values.push(row[id]);
                indices.push(id as i32);
            }
        }
        let check = |ws: &ResidentRank| -> Result<(), Box<dyn std::error::Error>> {
            let output = ws.snapshot()?.read()?;
            if output.values != values || output.indices != indices {
                return Err("profiled output differs from canonical reference".into());
            }
            Ok(())
        };
        let mut ws = ResidentRank::new(runtime.clone(), plan)?;
        ws.upload(&r.input)?;
        let mut batches = Vec::new();
        for batch in 0..14 {
            let mut sample = json!({"order": if batch % 2 == 0 { ["plain", "profiled"] } else { ["profiled", "plain"] }});
            for profiled in if batch % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                ws.synchronize()?;
                let start = Instant::now();
                if profiled {
                    let pending = ws.dispatch_profiled(16)?;
                    let dispatched = start.elapsed().as_secs_f64() * 1000.0;
                    let profile = pending.read()?;
                    let total = start.elapsed().as_secs_f64() * 1000.0;
                    sample["profiled"] = json!({"dispatch_call_ms":dispatched,"query_readback_ms":total-dispatched,"total_ms":total,"gpu":profile.report()});
                } else {
                    ws.dispatch(16)?;
                    let dispatched = start.elapsed().as_secs_f64() * 1000.0;
                    ws.synchronize()?;
                    let total = start.elapsed().as_secs_f64() * 1000.0;
                    sample["plain"] = json!({"dispatch_call_ms":dispatched,"completion_wait_ms":total-dispatched,"total_ms":total});
                }
                check(&ws)?;
            }
            if batch >= 2 {
                batches.push(sample);
            }
        }
        Ok(
            json!({"schema":"spiraltorch.rank_stage_probe.v1","status":"passed",
            "kind":r.kind,"rows":r.rows,"cols":r.cols,"k":r.k,"tile":r.tile,"repetitions":16,
            "warmup_pairs":2,"retained_pairs":12,"values":values,"indices":indices,
            "adapter":{"name":runtime.adapter_info().name,"backend":format!("{:?}",runtime.adapter_info().backend)},
            "boundary":"plain: dispatch call plus completion; profiled: query allocation/encoding/submit plus query readback; GPU pass clocks are diagnostic, not subtractable from host clocks",
            "batches":batches}),
        )
    }

    pub fn main() -> Result<(), Box<dyn std::error::Error>> {
        let args = std::env::args().skip(1).collect::<Vec<_>>();
        if args == ["--build-info"] {
            println!(
                "{}",
                json!({"schema":"spiraltorch.native_build_identity.v1",
                "build_fingerprint":st_core::build_fingerprint(),
                "manifest":serde_json::from_str::<Value>(st_core::build_manifest_json())?})
            );
            return Ok(());
        }
        if !args.is_empty() {
            return Err("usage: resident_rank_profile_bench [--build-info]".into());
        }
        let runtime = WgpuRuntime::request_profiled_headless_blocking("resident.rank.profile")?;
        let mut failed = false;
        for line in io::stdin().lock().lines() {
            let result = serde_json::from_str(&line?)
                .map_err(|e| e.into())
                .and_then(|r| run(r, &runtime));
            match result {
                Ok(report) => println!("{report}"),
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
        Err("use the browser rank profiling fixture on WASM".into())
    }
}
