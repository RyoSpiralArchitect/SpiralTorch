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

    fn readback_probe(
        runtime: &runtime::WgpuRuntime,
        payload: &[u8],
    ) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let context = runtime.context();
        let device = context.device();
        let queue = context.queue();
        let size = payload.len() as u64;
        let source = runtime::upload_slice(
            device,
            "probe.source",
            payload,
            wgpu::BufferUsages::COPY_SRC,
        )?;
        let allocate = || {
            runtime::empty_buffer::<u8>(
                device,
                "probe.staging",
                payload.len(),
                wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            )
        };
        let reused = allocate()?;
        let copy = |target: &wgpu::Buffer| {
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(&source, 0, target, 0, size);
            queue.submit(Some(encoder.finish()));
        };
        copy(&reused);
        let mut samples: [Vec<serde_json::Value>; 3] = std::array::from_fn(|_| Vec::new());
        for block in 0..14 {
            for slot in 0..3 {
                let mode = (slot + block) % 3;
                runtime::submit_with_timeout(
                    device,
                    queue,
                    [],
                    std::time::Duration::from_secs(30),
                    "readback.probe",
                )?;
                let started = Instant::now();
                let fresh = if mode == 0 { Some(allocate()?) } else { None };
                let target = fresh.as_ref().unwrap_or(&reused);
                let allocated = Instant::now();
                if mode != 2 {
                    copy(target);
                }
                let submitted = Instant::now();
                let data = runtime::map_read_bytes_with_timeout(
                    device,
                    target,
                    0..size,
                    std::time::Duration::from_secs(30),
                    "readback.probe",
                )?;
                let mapped = Instant::now();
                drop(fresh);
                let released = Instant::now();
                if data != payload {
                    return Err("readback control bytes changed".into());
                }
                if block >= 2 {
                    let ms = |end: Instant, start: Instant| (end - start).as_secs_f64() * 1000.;
                    samples[mode].push(json!({
                        "allocate":ms(allocated,started), "copy_submit":ms(submitted,allocated),
                        "map_and_owned_copy":ms(mapped,submitted), "release":ms(released,mapped),
                        "total":ms(released,started),
                    }));
                }
            }
        }
        Ok(json!({"status":"passed", "bytes":size,
            "boundary":"precompleted synthetic byte-copy control, not rank shader time; map includes owned host copy and unmap",
            "samples_ms":{"fresh":samples[0],"reused":samples[1],"map_only_no_submit":samples[2]}}))
    }

    fn run(
        r: Request,
        runtime: &runtime::WgpuRuntime,
        probe: bool,
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
        let mut samples: [Vec<f64>; 6] = std::array::from_fn(|_| Vec::new());
        for block in 0..14 {
            for slot in 0..6 {
                let mode = (slot + block + r.seed as usize % 6) % 6;
                rank.synchronize()?;
                let start = Instant::now();
                let resident = matches!(mode, 2 | 4 | 5);
                let repetitions = if resident { 16 } else { 1 };
                let mut actual = None;
                if mode == 5 {
                    rank.dispatch_from_matmul(&mut matmul, repetitions)?;
                } else {
                    for _ in 0..repetitions {
                        if mode >= 3 {
                            rank.dispatch_from_matmul(&mut matmul, 1)?;
                        } else {
                            matmul.dispatch(1)?;
                            if mode == 0 {
                                rank.upload(&matmul.snapshot()?.read()?)?;
                            } else {
                                rank.set_input_from_matmul(&matmul)?;
                            }
                            rank.dispatch(1)?;
                        }
                        if !resident {
                            actual = Some(rank.snapshot()?.read()?);
                        }
                    }
                }
                if resident {
                    rank.synchronize()?;
                }
                let ms = start.elapsed().as_secs_f64() * 1000. / f64::from(repetitions);
                // Validate every timed mode; resident snapshots are outside timing.
                let actual = match actual {
                    Some(output) => output,
                    None => rank.snapshot()?.read()?,
                };
                if actual != expected {
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
        let readback_probe = if probe {
            let payload = expected
                .values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .chain(expected.indices.iter().flat_map(|v| v.to_le_bytes()))
                .collect::<Vec<_>>();
            Some(readback_probe(runtime, &payload)?)
        } else {
            None
        };
        Ok(
            json!({"status":"passed", "kind":r.kind,"rows":r.rows,"inner":r.inner,"cols":r.cols,"k":r.k,"seed":r.seed,
            "matmul_kernel":matmul.kernel().as_str(),"matmul_accumulation":matmul.accumulation().as_str(),
            "adapter":{"name":runtime.adapter_info().name,"backend":format!("{:?}",runtime.adapter_info().backend)},
            "readback_probe":readback_probe,
            "values":expected.values,"indices":expected.indices,"resident_repetitions":16,
            "samples_ms":{"host_bridge":samples[0],"device_copy_bridge":samples[1],"resident_copy_bridge_per_op":samples[2],
                "single_submit_bridge":samples[3],"resident_single_submit_per_op":samples[4],"resident_batched_submit_per_op":samples[5]}}),
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
        let probe = args == ["--readback-probe"];
        if !args.is_empty() && !probe {
            return Err("usage: resident_matmul_rank_bench [--build-info|--readback-probe]".into());
        }
        let (runtime, _) = runtime::ensure_default_runtime_blocking("resident.matmul.rank.bench")?;
        for line in io::stdin().lock().lines() {
            match run(serde_json::from_str(&line?)?, &runtime, probe) {
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
