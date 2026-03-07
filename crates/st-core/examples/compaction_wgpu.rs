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

    use st_core::backend::wgpu_rt::{self, WgpuCtx};
    use st_core::ops::compaction::compact_between;
    use wgpu::util::DeviceExt;

    let rows = 2u32;
    let cols = 8u32;
    let row_stride = cols;
    let lower = 2.0f32;
    let upper = 4.0f32;
    let x: Vec<f32> = vec![
        0.0, 2.0, 1.0, 4.0, 3.0, 9.0, 8.0, 7.0, //
        3.0, 2.0, 1.0, 0.0, -1.0, -2.0, 5.0, 6.0,
    ];
    let mask: Vec<u32> = x
        .iter()
        .map(|&v| u32::from(v >= lower && v <= upper))
        .collect();

    let instance = wgpu::Instance::default();
    let Some(adapter) =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
    else {
        println!("no WGPU adapter found; skipping");
        return Ok(());
    };
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("st.compaction.example.device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
        },
        None,
    ))
    .map_err(|e| format!("request_device failed: {e}"))?;

    let ctx = Arc::new(WgpuCtx::new(device, queue));
    wgpu_rt::install_ctx(ctx.clone());

    let x_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("st.compaction.example.x"),
            contents: bytemuck::cast_slice(&x),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let mask_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("st.compaction.example.mask"),
            contents: bytemuck::cast_slice(&mask),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let out_counts = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("st.compaction.example.out_counts"),
        size: rows as u64 * std::mem::size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let out_vals = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("st.compaction.example.out_vals"),
        size: rows as u64 * cols as u64 * std::mem::size_of::<f32>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let out_idx = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("st.compaction.example.out_idx"),
        size: rows as u64 * cols as u64 * std::mem::size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    wgpu_rt::dispatch_compaction_1ce_buffers(
        rows,
        cols,
        row_stride,
        0,
        &x_buf,
        &mask_buf,
        &out_counts,
        &out_vals,
        &out_idx,
    )?;

    let got_counts = readback::<u32>(&ctx.device, &ctx.queue, &out_counts, rows as usize);
    let got_vals = readback::<f32>(&ctx.device, &ctx.queue, &out_vals, (rows * cols) as usize);
    let got_idx = readback::<u32>(&ctx.device, &ctx.queue, &out_idx, (rows * cols) as usize);
    let cpu = compact_between(&x, rows, cols, row_stride, lower, upper)
        .map_err(|e| format!("cpu compaction failed: {e:?}"))?;

    assert_eq!(got_counts, cpu.counts);
    for row in 0..rows as usize {
        let base = row * cols as usize;
        let valid = got_counts[row] as usize;
        assert_eq!(&got_vals[base..base + valid], &cpu.values[base..base + valid]);
        assert_eq!(&got_idx[base..base + valid], &cpu.indices[base..base + valid]);
    }

    println!("between counts={got_counts:?}");
    println!("between row0 values={:?}", &got_vals[..cols as usize]);
    println!("between row0 idx={:?}", &got_idx[..cols as usize]);
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
        label: Some("st.compaction.example.readback"),
        size: size as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("st.compaction.example.readback.enc"),
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
