#[cfg(not(all(feature = "wgpu", feature = "wgpu-rt")))]
fn main() {
    eprintln!("enable --features 'wgpu wgpu-rt' to run this example");
}

#[cfg(all(feature = "wgpu", feature = "wgpu-rt"))]
fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

#[cfg(all(feature = "wgpu", feature = "wgpu-rt"))]
fn run() -> Result<(), String> {
    use std::sync::Arc;

    use st_core::backend::device_caps::DeviceCaps;
    use st_core::backend::wgpu_rt::{self, WgpuCtx};
    use st_core::ops::rank_entry::{plan_rank, RankKind};
    use wgpu::util::DeviceExt;

    let rows = 2u32;
    let cols = 5u32;
    let k = 2u32;
    let row_stride = cols;
    let x: Vec<f32> = vec![
        1.0, 3.0, 2.0, 3.0, -1.0, //
        0.0, -2.0, 5.0, 4.0, 5.0,
    ];

    let instance = wgpu::Instance::default();
    let Some(adapter) =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
    else {
        println!("no WGPU adapter found; skipping");
        return Ok(());
    };
    let limits = adapter.limits();
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("st.rank.example.device"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
        },
        None,
    ))
    .map_err(|e| format!("request_device failed: {e}"))?;

    let ctx = Arc::new(WgpuCtx::new(device, queue));
    wgpu_rt::install_ctx(ctx.clone());

    let x_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("st.rank.example.x"),
            contents: bytemuck::cast_slice(&x),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let out_vals = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("st.rank.example.out_vals"),
        size: rows as u64 * k as u64 * std::mem::size_of::<f32>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let out_idx = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("st.rank.example.out_idx"),
        size: rows as u64 * k as u64 * std::mem::size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    run_rank_case(
        "topk",
        &ctx,
        &x,
        &x_buf,
        &out_vals,
        &out_idx,
        row_stride,
        plan_rank(
            RankKind::TopK,
            rows,
            cols,
            k,
            DeviceCaps::wgpu(32, false, 256),
        ),
    )?;
    run_rank_case(
        "bottomk",
        &ctx,
        &x,
        &x_buf,
        &out_vals,
        &out_idx,
        row_stride,
        plan_rank(
            RankKind::BottomK,
            rows,
            cols,
            k,
            DeviceCaps::wgpu(32, false, 256),
        ),
    )?;
    Ok(())
}

#[cfg(all(feature = "wgpu", feature = "wgpu-rt"))]
fn run_rank_case(
    label: &str,
    ctx: &std::sync::Arc<st_core::backend::wgpu_rt::WgpuCtx>,
    x: &[f32],
    x_buf: &wgpu::Buffer,
    out_vals: &wgpu::Buffer,
    out_idx: &wgpu::Buffer,
    row_stride: u32,
    plan: st_core::ops::rank_entry::RankPlan,
) -> Result<(), String> {
    let mut exec =
        st_core::backend::wgpu_exec::WgpuBufferExecutor::rank(x_buf, row_stride, out_vals, out_idx);
    st_core::ops::rank_entry::execute_rank(&mut exec, &plan)?;

    let got_vals = readback::<f32>(
        &ctx.device,
        &ctx.queue,
        out_vals,
        (plan.rows * plan.k) as usize,
    );
    let got_idx = readback::<u32>(
        &ctx.device,
        &ctx.queue,
        out_idx,
        (plan.rows * plan.k) as usize,
    );
    let expected = st_core::ops::rank_cpu::select_rank_cpu(&plan, x, row_stride)
        .map_err(|e| format!("cpu reference failed: {e}"))?;

    assert_eq!(got_vals, expected.values);
    assert_eq!(got_idx, expected.indices);

    println!("{label} values={got_vals:?}");
    println!("{label} idx={got_idx:?}");
    Ok(())
}

#[cfg(all(feature = "wgpu", feature = "wgpu-rt"))]
fn readback<T: bytemuck::Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    len: usize,
) -> Vec<T> {
    let size = len * std::mem::size_of::<T>();
    let read = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("st.rank.example.readback"),
        size: size as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("st.rank.example.readback.enc"),
    });
    enc.copy_buffer_to_buffer(src, 0, &read, 0, size as u64);
    queue.submit(Some(enc.finish()));

    let slice = read.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    device.poll(wgpu::Maintain::Wait);
    receiver
        .recv()
        .expect("map_async callback dropped")
        .expect("buffer map failed");

    let bytes = slice.get_mapped_range();
    let out = bytemuck::cast_slice(&bytes).to_vec();
    drop(bytes);
    read.unmap();
    out
}
