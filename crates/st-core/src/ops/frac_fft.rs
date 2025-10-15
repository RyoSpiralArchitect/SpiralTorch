// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.


// Fractional Laplacian along last axis (CPU): y = IFFT( |k|^{2s} * FFT(x) )
// Implementation uses rustfft for 1D transforms along the last axis.
use ndarray::{ArrayD, IxDyn, Axis, ArrayBase, Data, DataMut};
use num_complex::Complex32;
use rustfft::{FftPlanner, num_traits::Zero};

pub fn frac_laplacian_last_axis_cpu(x:&ArrayD<f32>, s:f32) -> ArrayD<f32> {
    let n = x.shape()[x.ndim()-1];
    let rows = x.len() / n;
    let mut planner = FftPlanner::<f32>::new();
    let fft  = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);

    let mut out = x.clone();

    // Temporary buffers
    let mut buf: Vec<Complex32> = vec![Complex32::zero(); n];

    let ax = Axis(x.ndim()-1);
    let mut lanes = out.view_mut().lanes_mut(ax);
    let mut src = x.view().lanes(ax);
    for (mut dst, src_lane) in lanes.into_iter().zip(src.into_iter()) {
        // Load real → complex
        for i in 0..n {
            buf[i] = Complex32::new(src_lane[i], 0.0);
        }
        // FFT
        fft.process(&mut buf);
        // Multiply |k|^{2s}; use symmetric indexing min(k, N-k)
        let power = 2.0*s.max(0.0);
        for k in 0..n {
            let kk = (k as i32).min((n - k) as i32) as f32;
            let scale = kk.powf(power);
            buf[k] = buf[k] * scale;
        }
        // IFFT
        ifft.process(&mut buf);
        // Store real / n (rustfft not normalized)
        for i in 0..n {
            dst[i] = buf[i].re / (n as f32);
        }
    }
    out
}
