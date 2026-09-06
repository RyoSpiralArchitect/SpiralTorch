//! Source-bound, small projection-head diagnostic, not full-model throughput.
#[cfg(not(target_arch = "wasm32"))]
mod native {
    use serde::Deserialize;
    use serde_json::json;
    use st_backend_wgpu::{
        rankk_exact_2ce::{resident::ResidentRank, Kind, Plan},
        resident_matmul::{MatmulShape, ResidentMatmul},
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
        inner: u32,
        cols: u32,
        k: u32,
        lhs: Vec<f32>,
        rhs: Vec<f32>,
        seed: u64,
    }

    fn run(
        r: Request,
        runtime: &runtime::WgpuRuntime,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let left = u64::from(r.rows) * u64::from(r.inner);
        let right = u64::from(r.inner) * u64::from(r.cols);
        if r.inner > 4096
            || left + right > 1_048_576
            || u64::from(r.rows) * u64::from(r.cols) > 1_048_576
            || left != r.lhs.len() as u64
            || right != r.rhs.len() as u64
            || r.lhs
                .iter()
                .chain(&r.rhs)
                .any(|v| !v.is_finite() || v.fract() != 0. || v.abs() > 4.)
        {
            return Err("expected bounded integer projection inputs".into());
        }
        let kind = match r.kind.as_str() {
            "topk" => Kind::TopK,
            "midk" => Kind::MidK,
            "bottomk" => Kind::BottomK,
            _ => return Err("unknown rank kind".into()),
        };
        let mut matmul = ResidentMatmul::new(
            runtime.clone(),
            MatmulShape::new(r.rows as usize, r.inner as usize, r.cols as usize)?,
        )?;
        let mut rank = ResidentRank::new(
            runtime.clone(),
            Plan::try_new(kind, r.rows, r.cols, r.k, 256)?,
        )?;
        matmul.upload(&r.lhs, &r.rhs)?;
        matmul.dispatch(1)?;
        rank.set_input_from_matmul(&matmul)?;
        rank.dispatch(1)?;
        let expected = rank.snapshot()?.read()?;
        let mut samples = [Vec::new(), Vec::new(), Vec::new()];
        for block in 0..14 {
            for slot in 0..3 {
                let mode = (slot + block + r.seed as usize % 3) % 3;
                rank.synchronize()?;
                let start = Instant::now();
                let repetitions = if mode == 2 { 16 } else { 1 };
                let mut actual = None;
                for _ in 0..repetitions {
                    matmul.dispatch(1)?;
                    if mode == 0 {
                        rank.upload(&matmul.snapshot()?.read()?)?;
                    } else {
                        rank.set_input_from_matmul(&matmul)?;
                    }
                    rank.dispatch(1)?;
                    if mode != 2 {
                        actual = Some(rank.snapshot()?.read()?);
                    }
                }
                if mode == 2 {
                    rank.synchronize()?;
                }
                let ms = start.elapsed().as_secs_f64() * 1000. / f64::from(repetitions);
                if actual.is_some_and(|output| output != expected) {
                    return Err("bridge parity failed".into());
                }
                if block >= 2 {
                    samples[mode].push(ms);
                }
            }
        }
        if rank.snapshot()?.read()? != expected {
            return Err("post-timing parity failed".into());
        }
        Ok(
            json!({"status":"passed", "kind":r.kind,"rows":r.rows,"inner":r.inner,"cols":r.cols,"k":r.k,"seed":r.seed,
            "matmul_kernel":matmul.kernel().as_str(),"matmul_accumulation":matmul.accumulation().as_str(),
            "adapter":{"name":runtime.adapter_info().name,"backend":format!("{:?}",runtime.adapter_info().backend)},
            "values":expected.values,"indices":expected.indices,"resident_repetitions":16,
            "samples_ms":{"host_bridge":samples[0],"device_copy_bridge":samples[1],"resident_copy_bridge_per_op":samples[2]}}),
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
            return Err("usage: resident_matmul_rank_bench [--build-info]".into());
        }
        let (runtime, _) = runtime::ensure_default_runtime_blocking("resident.matmul.rank.bench")?;
        for line in io::stdin().lock().lines() {
            match run(serde_json::from_str(&line?)?, &runtime) {
                Ok(result) => println!("{result}"),
                Err(error) => {
                    println!("{}", json!({"status":"error","error":error.to_string()}));
                    return Err(error);
                }
            }
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
        Err("native diagnostic only; use resident_matmul_rank_webgpu.html in a browser".into())
    }
}
