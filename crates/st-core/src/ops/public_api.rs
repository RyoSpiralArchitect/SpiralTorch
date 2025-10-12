use crate::error::{Result, invalid, device as dev_err};

pub fn topk_lastdim_host_select(x:&[f32], rows:usize, cols:usize, k:usize, _want:&str) -> Result<(Vec<f32>, Vec<i32>)> {
    if k==0 || k>cols { return Err(invalid("topk: invalid k")); }
    let mut outv = vec![0f32; rows*k];
    let mut outi = vec![0i32; rows*k];
    for r in 0..rows {
        let base = r*cols;
        let mut v: Vec<(f32, i32)> = (0..cols).map(|c| (x[base+c], c as i32)).collect();
        v.select_nth_unstable_by(k-1, |a,b| b.0.partial_cmp(&a.0).unwrap());
        v[..k].sort_by(|a,b| b.0.partial_cmp(&a.0).unwrap());
        for i in 0..k { outv[r*k+i] = v[i].0; outi[r*k+i] = v[i].1; }
    }
    Ok((outv, outi))
}

#[cfg(feature="wgpu")]
pub fn topk_lastdim_wgpu_2d_autotuned(x:&[f32], rows:usize, cols:usize, k:usize) -> Result<(Vec<f32>, Vec<i32>)> {
    crate::backend::wgpu_topk_kway::topk_kway_2d_autotuned(x, rows as u32, cols as u32, k as u32)
}
#[cfg(feature="mps")]
pub fn topk_lastdim_mps_2d_autotuned(x:&[f32], rows:usize, cols:usize, k:usize) -> Result<(Vec<f32>, Vec<i32>)> {
    crate::backend::mps_topk_kway::topk_kway_2d_autotuned(x, rows as u32, cols as u32, k as u32)
}
#[cfg(feature="cuda")]
pub fn topk_lastdim_cuda_2d(x:&[f32], rows:usize, cols:usize, k:usize) -> Result<(Vec<f32>, Vec<i32>)> {
    crate::backend::cuda_topk_kway::topk_lastdim_cuda_2d(x, rows, cols, k)
}

#[cfg(feature="wgpu")]
#[allow(clippy::too_many_arguments)]
pub fn where_nd_host_select_strided_bytes_direct_wgpu(
    cond_blob:&[u8], c_base_elems: usize, cshape:&[usize], cstrides:&[usize],
    x_blob:&[u8], x_base_elems: usize, xshape:&[usize], xstrides:&[usize],
    y_blob:&[u8], y_base_elems: usize, yshape:&[usize], ystrides:&[usize],
    out_shape:&[usize]
) -> Result<Vec<f32>> {
    let to_u32 = |v:&[usize]| v.iter().map(|&u| u as u32).collect::<Vec<u32>>();
    crate::backend::wgpu_where_direct::where_nd_strided_bytes_direct(
        cond_blob, c_base_elems as u32, &to_u32(cshape), &to_u32(cstrides),
        x_blob, x_base_elems as u32, &to_u32(xshape), &to_u32(xstrides),
        y_blob, y_base_elems as u32, &to_u32(yshape), &to_u32(ystrides),
        &to_u32(out_shape)
    )
}

#[cfg(feature="mps")]
#[allow(clippy::too_many_arguments)]
pub fn where_nd_host_select_strided_bytes_direct_mps(
    _cond_blob:&[u8], _c_base_elems: usize, _cshape:&[usize], _cstrides:&[usize],
    _x_blob:&[u8], _x_base_elems: usize, _xshape:&[usize], _xstrides:&[usize],
    _y_blob:&[u8], _y_base_elems: usize, _yshape:&[usize], _ystrides:&[usize],
    _out_shape:&[usize], _out_strides:&[usize]
) -> Result<Vec<f32>> {
    Err(dev_err("mps where_nd stub"))
}
