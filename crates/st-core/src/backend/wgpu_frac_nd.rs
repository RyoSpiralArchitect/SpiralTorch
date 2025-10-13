//! Permute-to-last axis wrapper for WGPU fracdiff/specmul kernels (ND convenience).
#![cfg(feature = "wgpu")]
use ndarray::{ArrayD, IxDyn, Axis};
use st_frac::alpha::gl_coeffs;
use crate::backend::wgpu_frac; // assume existing frac kernels for last-axis

pub fn fracdiff_gl_nd_wgpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    x: &ArrayD<f32>,
    alpha: f64,
    axis: usize,
    kernel_len: usize,
) -> anyhow::Result<ArrayD<f32>> {
    assert!(axis < x.ndim());
    // Permute axis to last; if already last, pass through
    let mut axes: Vec<usize> = (0..x.ndim()).collect();
    axes.swap(axis, x.ndim()-1);
    let x_perm = x.view().permuted_axes(axes.clone()).to_owned();
    let shape = x_perm.shape().to_vec();
    let rows = shape[..shape.len()-1].iter().product::<usize>();
    let cols = *shape.last().unwrap();
    let coeff = gl_coeffs(alpha, kernel_len).into_iter().map(|v| v as f32).collect::<Vec<_>>();
    // Flatten to 2D and call wgpu fracdiff last-axis kernel
    let flat = x_perm.into_shape((rows, cols)).unwrap();
    let y2d = wgpu_frac::fracdiff_gl_wgpu_2d(device, queue, flat.as_standard_layout().to_owned(), &coeff)?;
    // Reshape back and inverse permute
    let mut y = ArrayD::<f32>::from_shape_vec(IxDyn(&[rows, cols]), y2d.into_raw_vec()).unwrap();
    let mut y_nd = y.into_shape(IxDyn(&shape)).unwrap();
    // inverse permute
    let mut inv = vec![0; axes.len()];
    for (i,&a) in axes.iter().enumerate() { inv[a]=i; }
    let y_ret = y_nd.view().permuted_axes(inv).to_owned();
    Ok(y_ret)
}
