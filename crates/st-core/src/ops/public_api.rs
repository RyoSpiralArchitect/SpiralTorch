use crate::error::{Result, invalid, device as dev_err};

/// CPU TopK along the last dimension (2D: rows x cols)
pub fn topk_lastdim_host_select(x:&[f32], rows:usize, cols:usize, k:usize, _want:&str) -> Result<(Vec<f32>, Vec<i32>)> {
    if k==0 || k>cols { return Err(invalid("topk: invalid k")); }
    let mut outv = vec![0f32; rows*k];
    let mut outi = vec![0i32; rows*k];
    for r in 0..rows {
        let base = r*cols;
        let mut v: Vec<(f32, i32)> = (0..cols).map(|c| (x[base+c], c as i32)).collect();
        v.select_nth_unstable_by(k-1, |a,b| b.0.partial_cmp(&a.0).unwrap());
        v[..k].sort_by(|a,b| b.0.partial_cmp(&a.0).unwrap());
        for i in 0..k {
            outv[r*k+i] = v[i].0;
            outi[r*k+i] = v[i].1;
        }
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
        "wgpu" => {
            #[cfg(feature="wgpu")] {
                let to_u32 = |v:&[usize]| v.iter().map(|&u| u as u32).collect::<Vec<u32>>();
                return crate::backend::wgpu_where_segments::where_nd_strided_segments_u8_with_base(
                    cond_total, cond_offsets, cond_sizes, cond_starts, cond_blob,
                    x_total, x_offsets, x_sizes, x_starts, x_blob,
                    y_total, y_offsets, y_sizes, y_starts, y_blob,
                    &to_u32(cshape), &to_u32(cstrides), c_base_elems as u32,
                    &to_u32(xshape), &to_u32(xstrides), x_base_elems as u32,
                    &to_u32(yshape), &to_u32(ystrides), y_base_elems as u32,
                    &to_u32(out_shape)
                );
            }
            #[cfg(not(feature="wgpu"))] { return Err(dev_err("wgpu feature not enabled")); }
        }
        "mps" => {
            #[cfg(feature="mps")] { return Err(dev_err("use where_nd_host_select_strided_segments_bytes_mps for MPS")); }
            #[cfg(not(feature="mps"))] { return Err(dev_err("mps feature not enabled")); }
        }
        _ => Err(dev_err("segments path supports wgpu/mps only")),
    }
}

#[cfg(feature="mps")]
#[allow(clippy::too_many_arguments)]
pub fn where_nd_host_select_strided_segments_bytes_mps(
    cond_total: usize, cond_offsets: &[u32], cond_sizes: &[u32], cond_starts: &[u32], cond_blob: &[u8],
    x_total: usize, x_offsets: &[u32], x_sizes: &[u32], x_starts: &[u32], x_blob: &[u8],
    y_total: usize, y_offsets: &[u32], y_sizes: &[u32], y_starts: &[u32], y_blob: &[u8],
    out_shape:&[usize], out_strides:&[usize],
    cshape:&[usize], cstrides:&[usize], c_base_elems: usize,
    xshape:&[usize], xstrides:&[usize], x_base_elems: usize,
    yshape:&[usize], ystrides:&[usize], y_base_elems: usize,
) -> Result<Vec<f32>> {
    let to_u32 = |v:&[usize]| v.iter().map(|&u| u as u32).collect::<Vec<u32>>();
    crate::backend::mps_where_segments::where_nd_strided_segments_u8_with_base_mps(
        cond_total, cond_offsets, cond_sizes, cond_starts, cond_blob,
        x_total, x_offsets, x_sizes, x_starts, x_blob,
        y_total, y_offsets, y_sizes, y_starts, y_blob,
        &to_u32(out_shape), &to_u32(out_strides),
        &to_u32(cshape), &to_u32(cstrides), c_base_elems as u32,
        &to_u32(xshape), &to_u32(xstrides), x_base_elems as u32,
        &to_u32(yshape), &to_u32(ystrides), y_base_elems as u32,
    )
}
