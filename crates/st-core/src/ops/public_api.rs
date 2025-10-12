
use crate::error::{Result, device as dev_err};

pub fn wgpu_is_available() -> bool {
    #[cfg(feature="wgpu")]
    { pollster::block_on(wgpu::Instance::default().request_adapter(&wgpu::RequestAdapterOptions::default())).is_some() }
    #[cfg(not(feature="wgpu"))] { false }
}
pub fn cuda_is_available() -> bool {
    #[cfg(feature="cuda")] { cust::quick_init().is_ok() }
    #[cfg(not(feature="cuda"))] { false }
}
pub fn mps_is_available() -> bool {
    #[cfg(feature="mps")] { metal::Device::system_default().is_some() }
    #[cfg(not(feature="mps"))] { false }
}

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

pub fn where_nd_host_select_strided_bytes(
    cond_bytes:&[u8], cshape:&[usize], cstrides_elems:&[usize], c_base_elems: usize,
    x_bytes:&[u8], xshape:&[usize], xstrides_elems:&[usize], x_base_elems: usize,
    y_bytes:&[u8], yshape:&[usize], ystrides_elems:&[usize], y_base_elems: usize,
    out_shape:&[usize], device: &str
) -> Result<Vec<f32>>
{
    let st_out = strides_of_usize(out_shape);
    match device {
        "wgpu" => {
            #[cfg(feature="wgpu")]
            { return crate::backend::wgpu_where_nd::WgpuWhereND::new().where_nd_strided_u8_with_base(
                cond_bytes,
                x_bytes, &xshape.iter().map(|&u| u as u32).collect::<Vec<_>>(), &xstrides_elems.iter().map(|&u| u as u32).collect::<Vec<_>>(), x_base_elems as u32,
                y_bytes, &yshape.iter().map(|&u| u as u32).collect::<Vec<_>>(), &ystrides_elems.iter().map(|&u| u as u32).collect::<Vec<_>>(), y_base_elems as u32,
                &out_shape.iter().map(|&u| u as u32).collect::<Vec<_>>(), &st_out.iter().map(|&u| u as u32).collect::<Vec<_>>(),
                &cshape.iter().map(|&u| u as u32).collect::<Vec<_>>(), &cstrides_elems.iter().map(|&u| u as u32).collect::<Vec<_>>(), c_base_elems as u32
            ); }
            #[cfg(not(feature="wgpu"))] { return Err(dev_err("wgpu feature not enabled")); }
        }
        "mps" => {
            #[cfg(feature="mps")]
            { return Err(dev_err("mps bytes path not wired in this artifact")); }
            #[cfg(not(feature="mps"))] { return Err(dev_err("mps feature not enabled")); }
        }
        _ => Err(dev_err("device not supported for bytes path")),
    }
}

pub fn where_nd_host_select_strided(
    cond_bool:&[bool], cshape:&[usize], cstrides:&[usize],
    x:&[f32], xshape:&[usize], xstrides:&[usize],
    y:&[f32], yshape:&[usize], ystrides:&[usize],
    out_shape:&[usize], device:&str
) -> Result<Vec<f32>>
{
    match device {
        "cpu" => where_nd_cpu_strided_bool(cond_bool, cshape, cstrides, x, xshape, xstrides, y, yshape, ystrides, out_shape),
        "cuda" => {
            #[cfg(feature="cuda")] {
                // Fallback to CPU logic for this artifact (kernel wire to be added)
                where_nd_cpu_strided_bool(cond_bool, cshape, cstrides, x, xshape, xstrides, y, yshape, ystrides, out_shape)
            }
            #[cfg(not(feature="cuda"))] { Err(dev_err("cuda feature not enabled")) }
        }
        "wgpu" => Err(dev_err("wgpu path requires bytes API; call where_nd_host_select_strided_bytes")),
        "mps" => Err(dev_err("mps path requires bytes/segments API; call bytes/segments variant")),
        _ => Err(dev_err("unsupported device for this path"))
    }
}

// Segments API (WGPU/MPS)
pub fn where_nd_host_select_strided_segments_bytes(
    cond_total: usize, cond_offsets: &[u32], cond_sizes: &[u32], cond_starts: &[u32], cond_blob: &[u8],
    cshape:&[usize], cstrides:&[usize], c_base_elems: usize,
    x_total: usize, x_offsets: &[u32], x_sizes: &[u32], x_starts: &[u32], x_blob: &[u8],
    xshape:&[usize], xstrides:&[usize], x_base_elems: usize,
    y_total: usize, y_offsets: &[u32], y_sizes: &[u32], y_starts: &[u32], y_blob: &[u8],
    yshape:&[usize], ystrides:&[usize], y_base_elems: usize,
    out_shape:&[usize], device:&str
) -> Result<Vec<f32>> {
    match device {
        "wgpu" => Err(dev_err("segments path not wired into WGSL upload in this artifact")), // wiring left for later patch
        "mps"  => Err(dev_err("segments path not wired into MPS upload in this artifact")),
        _ => Err(dev_err("segments path supports wgpu/mps only")),
    }
}

// minimal TopK (CPU) used by Python
pub fn topk_lastdim_host_select(x:&[f32], rows:usize, cols:usize, k:usize, _device:&str) -> Result<(Vec<f32>, Vec<i32>)> {
    if k==0 || k>cols { return Err(dev_err("topk: invalid k")); }
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
