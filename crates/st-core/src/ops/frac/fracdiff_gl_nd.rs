use ndarray::{ArrayD, IxDyn, Axis};
use st_frac::alpha::gl_coeffs;

#[derive(Clone, Copy)]
pub enum PadMode { Zero, Reflect }

/// Apply GL fractional difference along arbitrary axis, CPU.
/// x: f32 array, alpha in [0,1], kernel_len > 0.
pub fn fracdiff_gl_nd_cpu(x: &ArrayD<f32>, alpha: f64, axis: usize, kernel_len: usize, pad: PadMode) -> ArrayD<f32> {
    assert!(axis < x.ndim());
    let coeff = gl_coeffs(alpha, kernel_len);
    let mut out = x.clone();
    let len = x.shape()[axis];
    let mut lane = x.clone().into_dimensionality::<IxDyn>().unwrap();
    // Iterate over all lanes (all dims except axis), then convolve 1D along axis.
    // Simplicity: use ndarray lane APIs
    let mut it = x.view().lanes(Axis(axis));
    let mut oit = out.view_mut().lanes_mut(Axis(axis));
    for (src, mut dst) in it.into_iter().zip(oit.into_iter()) {
        // 1D conv with padding
        for i in 0..len {
            let mut acc = 0.0f32;
            for k in 0..kernel_len {
                let j = if i>=k { i-k } else {
                    match pad {
                        PadMode::Zero => { continue; }
                        PadMode::Reflect => { k - i } // crude reflect
                    }
                };
                let w = coeff[k] as f32;
                acc += w * src[j];
            }
            dst[i] = acc;
        }
    }
    out
}
