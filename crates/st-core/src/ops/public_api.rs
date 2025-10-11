
use crate::error::{Result, device as dev_err};

// Availability probes
pub fn wgpu_is_available() -> bool {
    #[cfg(feature="wgpu")]
    {
        let instance = wgpu::Instance::default();
        if let Some(_adapter) = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())) {
            return true;
        }
        return false;
    }
    #[cfg(not(feature="wgpu"))] { false }
}
pub fn cuda_is_available() -> bool {
    #[cfg(feature="cuda")]
    { cust::quick_init().is_ok() }
    #[cfg(not(feature="cuda"))] { false }
}
pub fn mps_is_available() -> bool {
    #[cfg(feature="mps")]
    { metal::Device::system_default().is_some() }
    #[cfg(not(feature="mps"))] { false }
}

// Capability flags (future extension)
pub fn supports_where_nd_cuda() -> bool { false }
pub fn supports_where_nd_mps()  -> bool { false }
pub fn supports_topk_cuda()     -> bool { false }
pub fn supports_topk_mps()      -> bool { false }

fn strides_of_usize(shape:&[usize])->Vec<usize>{ let nd=shape.len(); let mut st=vec![0usize;nd]; let mut acc=1usize; for d in (0..nd).rev(){ st[d]=acc; acc*=shape[d]; } st }
fn broadcast_strides_strided(in_shape:&[usize], in_strides:&[usize], out_shape:&[usize]) -> Result<Vec<usize>> {
    if in_shape.len()>out_shape.len(){ return Err(dev_err("broadcast: rank larger than out")); }
    let pad = out_shape.len()-in_shape.len();
    let mut shp = vec![1usize; pad]; shp.extend_from_slice(in_shape);
    let mut st  = vec![0usize; pad]; st.extend_from_slice(in_strides);
    let mut bs = vec![0usize; out_shape.len()];
    for d in 0..out_shape.len(){
        if shp[d]==out_shape[d] { bs[d]=st[d]; }
        else if shp[d]==1 && out_shape[d]>1 { bs[d]=0; }
        else { return Err(dev_err("broadcast: incompatible")); }
    }
    Ok(bs)
}

fn where_nd_cpu_strided_bool(cond:&[bool], cshape:&[usize], cstrides:&[usize],
                             x:&[f32], xshape:&[usize], xstrides:&[usize],
                             y:&[f32], yshape:&[usize], ystrides:&[usize],
                             out_shape:&[usize]) -> Result<Vec<f32>>
{
    let nd = out_shape.len();
    let st_out = strides_of_usize(out_shape);
    let bc = broadcast_strides_strided(cshape, cstrides, out_shape)?;
    let bx = broadcast_strides_strided(xshape, xstrides, out_shape)?;
    let by = broadcast_strides_strided(yshape, ystrides, out_shape)?;
    let n: usize = out_shape.iter().product();
    let mut out = vec![0f32; n];
    for i in 0..n {
        let mut off_c=0usize; let mut off_x=0usize; let mut off_y=0usize;
        let mut rem = i;
        for d in 0..nd {
            let s = st_out[d]; let cd = (rem / s) % out_shape[d]; rem %= s;
            off_c += cd * bc[d]; off_x += cd * bx[d]; off_y += cd * by[d];
        }
        let c = cond[off_c];
        out[i] = if c { x[off_x] } else { y[off_y] };
    }
    Ok(out)
}

// Public host-facing (STRIDED) with GPU/CPU switch. cond_bool path
pub fn where_nd_host_use_wgpu_strided(cond_bool:&[bool], cshape:&[usize], cstrides:&[usize],
                                      x:&[f32], xshape:&[usize], xstrides:&[usize],
                                      y:&[f32], yshape:&[usize], ystrides:&[usize],
                                      out_shape:&[usize], use_wgpu: bool) -> Result<Vec<f32>>
{
    if use_wgpu {
        #[cfg(feature="wgpu")]
        {
            use wgpu::util::DeviceExt;
            use crate::backend::wgpu_where_nd::WgpuWhereND;
            // Upload cond as u8 bytes and use u8-optimized kernel
            let cond_u8: Vec<u8> = cond_bool.iter().map(|&b| if b {1u8} else {0u8}).collect();
            // pack into u32 words (4 bytes per word)
            let words = (cond_u8.len() + 3) / 4;
            let mut packed: Vec<u32> = vec![0u32; words];
            for i in 0..cond_u8.len() {
                let w = i / 4; let shift = (i % 4) * 8;
                packed[w] |= (cond_u8[i] as u32) << shift;
            }
            let instance = wgpu::Instance::default();
            let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).expect("wgpu adapter");
            let (dev, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor{
                label: Some("st-ops-where-nd-strided-u8"), features: wgpu::Features::empty(), limits: wgpu::Limits::downlevel_defaults()
            }, None)).expect("device");
            let b_cond_bytes = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor{ label: Some("cond_bytes"), contents: bytemuck::cast_slice(&packed), usage: wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_DST });
            let b_x = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor{ label: Some("x"), contents: bytemuck::cast_slice(x), usage: wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_DST });
            let b_y = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor{ label: Some("y"), contents: bytemuck::cast_slice(y), usage: wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_DST });
            let to_u32 = |v:&[usize]| -> Vec<u32> { v.iter().map(|&u| u as u32).collect() };
            let out_strides_u32: Vec<u32> = { let mut st=vec![0usize; out_shape.len()]; let mut acc=1; for d in (0..out_shape.len()).rev(){ st[d]=acc; acc*=out_shape[d]; } st }.into_iter().map(|u| u as u32).collect();
            let out_buf = WgpuWhereND::new().where_nd_strided_u8(
                &b_cond_bytes,
                &b_x, &to_u32(xshape), &to_u32(xstrides),
                &b_y, &to_u32(yshape), &to_u32(ystrides),
                &to_u32(out_shape), &out_strides_u32,
                &to_u32(cshape), &to_u32(cstrides)
            )?;
            let n: usize = out_shape.iter().product();
            let staging = dev.create_buffer(&wgpu::BufferDescriptor{ label: Some("out-staging"), size: (n*4) as u64, usage: wgpu::BufferUsages::MAP_READ|wgpu::BufferUsages::COPY_DST, mapped_at_creation:false });
            let mut e = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("rb-enc") });
            e.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, (n*4) as u64);
            queue.submit(std::iter::once(e.finish()));
            let slice = staging.slice(..); let _ = slice.map_async(wgpu::MapMode::Read);
            dev.poll(wgpu::Maintain::Wait);
            let data = slice.get_mapped_range(); let vec = bytemuck::cast_slice::<u8,f32>(&data).to_vec(); drop(data); staging.unmap();
            return Ok(vec);
        }
        #[cfg(not(feature="wgpu"))]
        { return Err(dev_err("wgpu feature not enabled")); }
    }
    // CPU fallback
    where_nd_cpu_strided_bool(cond_bool, cshape, cstrides, x, xshape, xstrides, y, yshape, ystrides, out_shape)
}

// 2D TopK host wrapper (GPU/CPU); device probing is done at Python side.
pub fn topk_lastdim_host_use_wgpu(x:&[f32], rows:usize, cols:usize, k:usize, use_wgpu: bool) -> Result<(Vec<f32>, Vec<i32>)> {
    if k==0 || k>cols { return Err(dev_err("topk: invalid k")); }
    if use_wgpu {
        #[cfg(feature="wgpu")]
        {
            use wgpu::util::DeviceExt;
            use crate::backend::{BackendArrayF32, wgpu_topk_unified::WgpuTopKUnified};
            let instance = wgpu::Instance::default();
            let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).expect("wgpu adapter");
            let (dev, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor{
                label: Some("st-ops-topk"), features: wgpu::Features::empty(), limits: wgpu::Limits::downlevel_defaults()
            }, None)).expect("device");
            let b_x = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor{
                label: Some("x"), contents: bytemuck::cast_slice(x), usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
            });
            let x_dev = BackendArrayF32::Wgpu{ rows, cols, buffer: b_x };
            let (v,i) = WgpuTopKUnified::new().topk_lastdim(&x_dev, rows, cols, k)?;
            let vals = v.into_dimensionality::<ndarray::Ix2>().unwrap().into_raw_vec();
            let idxs = i.into_dimensionality::<ndarray::Ix2>().unwrap().into_raw_vec();
            return Ok((vals, idxs));
        }
        #[cfg(not(feature="wgpu"))]
        { return Err(dev_err("wgpu feature not enabled")); }
    }
    // CPU fallback
    use std::cmp::Ordering;
    let mut vals = vec![0f32; rows*k];
    let mut idxs = vec![0i32; rows*k];
    for r in 0..rows {
        let base = r*cols;
        let mut v: Vec<(f32,i32)> = (0..cols).map(|c|(x[base+c], c as i32)).collect();
        v.sort_unstable_by(|a,b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        for t in 0..k { vals[r*k+t]=v[t].0; idxs[r*k+t]=v[t].1; }
    }
    Ok((vals, idxs))
}
