
use ndarray::{ArrayD, Axis};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum FracErr {
    #[error("axis out of range")]
    Axis,
    #[error("kernel_len must be > 0")]
    Kernel,
}

#[derive(Clone, Copy, Debug)]
pub enum Pad { Zero, Reflect }

pub fn gl_coeffs(alpha: f32, len: usize) -> Vec<f32> {
    assert!(len>0);
    let mut c = vec![0.0f32; len];
    c[0] = 1.0;
    for k in 1..len {
        let prev = c[k-1];
        let num = alpha - (k as f32 - 1.0);
        c[k] = prev * (num / (k as f32)) * -1.0;
    }
    c
}

fn conv1d_gl_line(x:&[f32], y:&mut [f32], coeff:&[f32], pad:Pad, scale:f32) {
    let n = x.len(); let klen = coeff.len();
    for i in 0..n {
        let mut acc = 0.0f32;
        for k in 0..klen {
            if i>=k { acc += coeff[k] * x[i-k]; }
            else {
                match pad {
                    Pad::Zero => {}
                    Pad::Reflect => {
                        let idx = k - i - 1;
                        let jj = if idx < n { idx } else { n.saturating_sub(1) - ((idx - (n-1)) % n) };
                        acc += coeff[k] * x[jj];
                    }
                }
            }
        }
        y[i] = scale * acc;
    }
}

pub fn fracdiff_gl_nd(x:&ArrayD<f32>, alpha:f32, axis:usize, kernel_len:usize, pad:Pad, scale:Option<f32>) -> Result<ArrayD<f32>, FracErr> {
    if axis >= x.ndim() { return Err(FracErr::Axis); }
    if kernel_len==0 { return Err(FracErr::Kernel); }
    let mut y = x.clone();
    let coeff = gl_coeffs(alpha, kernel_len);
    let s = scale.unwrap_or(1.0);
    let ax = Axis(axis);
    let mut yv = y.view_mut();
let dst_lanes = yv.lanes_mut(ax);
let xv = x.view();
let src_lanes = xv.lanes(ax);
 
    for (mut dst, src) in dst_lanes.into_iter().zip(src_lanes.into_iter()) {
        conv1d_gl_line(src.as_slice().unwrap(), dst.as_slice_mut().unwrap(), &coeff, pad, s);
    }
    Ok(y)
}

pub fn fracdiff_gl_nd_backward(gy:&ArrayD<f32>, alpha:f32, axis:usize, kernel_len:usize, pad:Pad, scale:Option<f32>) -> Result<ArrayD<f32>, FracErr> {
    if axis >= gy.ndim() { return Err(FracErr::Axis); }
    if kernel_len==0 { return Err(FracErr::Kernel); }
    let mut gx = gy.clone();
    let mut coeff = gl_coeffs(alpha, kernel_len);
    coeff.reverse();
    let s = scale.unwrap_or(1.0);
    let ax = Axis(axis);
    let mut gxv = gx.view_mut();
let dst_lanes = gxv.lanes_mut(ax);
let gyv = gy.view();
let src_lanes = gyv.lanes(ax);
 
    for (mut dst, src) in dst_lanes.into_iter().zip(src_lanes.into_iter()) {
        conv1d_gl_line(src.as_slice().unwrap(), dst.as_slice_mut().unwrap(), &coeff, pad, s);
    }
    Ok(gx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    #[test] fn smoke() {
        let x = array![[0.,1.,2.,3.,4.,5.,6.,7.]].into_dyn();
        let y = fracdiff_gl_nd(&x, 0.5, 1, 4, Pad::Zero, None).unwrap();
        assert_eq!(y.shape(), &[1,8]);
    }
}
