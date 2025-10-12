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

#[cfg(feature="cuda")]
pub fn topk_lastdim_cuda_2d(x:&[f32], rows:usize, cols:usize, k:usize) -> Result<(Vec<f32>, Vec<i32>)> {
    crate::backend::cuda_topk_kway::topk_lastdim_cuda_2d(x, rows, cols, k)
}

#[allow(clippy::too_many_arguments)]
pub fn where_nd_host_select_strided_bytes_direct_wgpu(
    cond_blob:&[u8], c_base_elems: usize, cshape:&[usize], cstrides:&[usize],
    x_blob:&[u8], x_base_elems: usize, xshape:&[usize], xstrides:&[usize],
    y_blob:&[u8], y_base_elems: usize, yshape:&[usize], ystrides:&[usize],
    out_shape:&[usize]
) -> Result<Vec<f32>> {
    #[cfg(feature="wgpu")] {
        let to_u32 = |v:&[usize]| v.iter().map(|&u| u as u32).collect::<Vec<u32>>();
        return crate::backend::wgpu_where_direct::where_nd_strided_bytes_direct(
            cond_blob, c_base_elems as u32, &to_u32(cshape), &to_u32(cstrides),
            x_blob, x_base_elems as u32, &to_u32(xshape), &to_u32(xstrides),
            y_blob, y_base_elems as u32, &to_u32(yshape), &to_u32(ystrides),
            &to_u32(out_shape)
        );
    }
    #[cfg(not(feature="wgpu"))] { Err(dev_err("wgpu feature not enabled")) }
}

#[cfg(feature="mps")]
#[allow(clippy::too_many_arguments)]
pub fn where_nd_host_select_strided_bytes_direct_mps(
    cond_blob:&[u8], _c_base_elems: usize, cshape:&[usize], cstrides:&[usize],
    x_blob:&[u8], _x_base_elems: usize, xshape:&[usize], xstrides:&[usize],
    y_blob:&[u8], _y_base_elems: usize, yshape:&[usize], ystrides:&[usize],
    out_shape:&[usize], out_strides:&[usize]
) -> Result<Vec<f32>> {
    use crate::backend::mps_where_direct::MpsWhereDirect;
    let dev = metal::Device::system_default().ok_or_else(|| dev_err("MPS device not found"))?;
    let to_u32b = |v:&[usize]| bytemuck::cast_slice::<u32,u8>(&v.iter().map(|&u| u as u32).collect::<Vec<u32>>());
    let b_out_shape   = dev.new_buffer_with_data(to_u32b(out_shape), metal::MTLResourceOptions::CPUCacheModeDefaultCache);
    let b_out_strides = dev.new_buffer_with_data(to_u32b(out_strides), metal::MTLResourceOptions::CPUCacheModeDefaultCache);
    let b_c_shape     = dev.new_buffer_with_data(to_u32b(cshape), metal::MTLResourceOptions::CPUCacheModeDefaultCache);
    let b_c_strides   = dev.new_buffer_with_data(to_u32b(cstrides), metal::MTLResourceOptions::CPUCacheModeDefaultCache);
    let b_x_shape     = dev.new_buffer_with_data(to_u32b(xshape), metal::MTLResourceOptions::CPUCacheModeDefaultCache);
    let b_x_strides   = dev.new_buffer_with_data(to_u32b(xstrides), metal::MTLResourceOptions::CPUCacheModeDefaultCache);
    let b_y_shape     = dev.new_buffer_with_data(to_u32b(yshape), metal::MTLResourceOptions::CPUCacheModeDefaultCache);
    let b_y_strides   = dev.new_buffer_with_data(to_u32b(ystrides), metal::MTLResourceOptions::CPUCacheModeDefaultCache);
    let n: u32 = out_shape.iter().copied().fold(1u32, |a,b| a.saturating_mul(b as u32));
    let out = MpsWhereDirect::new()?.run_direct(
        cond_blob, x_blob, y_blob,
        &b_out_shape, &b_out_strides,
        &b_c_shape, &b_c_strides, 0,
        &b_x_shape, &b_x_strides, 0,
        &b_y_shape, &b_y_strides, 0,
        out_shape.len() as u32, n as u32
    )?;
    Ok(out)
}
