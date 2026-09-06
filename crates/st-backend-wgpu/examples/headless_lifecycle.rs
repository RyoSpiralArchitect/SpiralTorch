//! Concurrent headless startup, copy/readback, destruction and thread-exit probe.
//! Run with an external process timeout: foreign driver calls can block in Drop.

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<(), String> {
    use st_backend_wgpu::runtime::{read_buffer, upload_slice, WgpuRuntime};
    use std::sync::Barrier;

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() > 3 {
        return Err("usage: headless_lifecycle [threads=4] [rounds=4] [drop|destroy]".into());
    }
    let shutdown = args.get(2).map(String::as_str).unwrap_or("drop");
    if !matches!(shutdown, "drop" | "destroy") {
        return Err("shutdown must be drop or destroy".into());
    }
    let parse = |index: usize, default: usize, max: usize| -> Result<usize, String> {
        let value = args
            .get(index)
            .map(|value| value.parse::<usize>())
            .transpose()
            .map_err(|error| error.to_string())?
            .unwrap_or(default);
        if value == 0 || value > max {
            return Err(format!("argument {index} must be in 1..={max}"));
        }
        Ok(value)
    };
    let threads = parse(0, 4, 8)?;
    let rounds = parse(1, 4, 16)?;
    let mut receipts = Vec::new();
    for round in 0..rounds {
        let barrier = Barrier::new(threads);
        let results = std::thread::scope(|scope| {
            let handles = (0..threads)
                .map(|worker| {
                    let barrier = &barrier;
                    scope.spawn(move || -> Result<_, String> {
                        barrier.wait();
                        eprintln!("round={round} worker={worker} phase=request");
                        let runtime = pollster::block_on(WgpuRuntime::request_headless(
                            "st.backend.headless_lifecycle",
                        ))
                        .map_err(|error| error.to_string())?;
                        let info = runtime.adapter_info().clone();
                        if info.device_type == wgpu::DeviceType::Cpu {
                            return Err("software adapter is not real-GPU coverage".into());
                        }
                        eprintln!(
                            "round={round} worker={worker} phase=ready backend={:?}",
                            info.backend
                        );
                        let context = runtime.context();
                        let expected = [round as u32, worker as u32, 0x12345678, u32::MAX];
                        let buffer = upload_slice(
                            context.device(),
                            "lifecycle.input",
                            &expected,
                            wgpu::BufferUsages::COPY_SRC,
                        )
                        .map_err(|error| error.to_string())?;
                        let actual = read_buffer::<u32>(
                            context.device(),
                            context.queue(),
                            &buffer,
                            expected.len(),
                            "lifecycle.read",
                        )
                        .map_err(|error| error.to_string())?;
                        if actual != expected {
                            return Err("lifecycle readback mismatch".into());
                        }
                        drop(buffer);
                        if shutdown == "destroy" {
                            context.device().destroy();
                            context.device().poll(wgpu::Maintain::Wait);
                            eprintln!("round={round} worker={worker} phase=device_destroyed");
                        }
                        drop(runtime);
                        eprintln!("round={round} worker={worker} phase=runtime_dropped");
                        Ok(serde_json::json!({
                            "round": round, "worker": worker, "adapter": info.name,
                            "backend": format!("{:?}", info.backend),
                            "device_type": format!("{:?}", info.device_type),
                            "status": "passed"
                        }))
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().map_err(|_| "worker panicked".to_owned())?)
                .collect::<Result<Vec<_>, String>>()
        })?;
        receipts.extend(results);
        eprintln!("round={round} phase=threads_joined");
    }
    println!(
        "{}",
        serde_json::json!({
            "schema": "spiraltorch.headless_lifecycle.v1", "status": "passed",
        "threads": threads, "rounds": rounds, "shutdown": shutdown, "receipts": receipts,
            "boundary": "startup/copy/readback/drop/thread-join correctness, not throughput"
        })
    );
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn main() {
    // Browser lifetime is exercised by the asynchronous WebGPU fixtures instead.
}
