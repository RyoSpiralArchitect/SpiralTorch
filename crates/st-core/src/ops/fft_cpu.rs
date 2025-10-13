use num_complex::Complex32;
use rustfft::{FftPlanner, FftNum};
use ndarray::{Array3, Axis, s};

#[derive(Clone, Copy)]
pub enum FftNorm { None, Backward, Ortho }

/// x_reim: shape (rows, cols, 2) with interleaved real/imag per element.
pub fn fft_1d_rows_cpu(x_reim: &Array3<f32>, rows: usize, cols: usize, inverse: bool, norm: FftNorm) -> Array3<f32> {
    assert_eq!(x_reim.shape(), &[rows, cols, 2]);
    let mut out = x_reim.clone();
    let mut planner = FftPlanner::<f32>::new();
    let fft = if inverse { planner.plan_fft_inverse(cols) } else { planner.plan_fft_forward(cols) };
    let scale = match (inverse, norm) {
        (true, FftNorm::Backward) => 1.0 / (cols as f32),
        (true, FftNorm::Ortho) => 1.0 / (cols as f32).sqrt(),
        (false, FftNorm::Ortho) => 1.0 / (cols as f32).sqrt(),
        _ => 1.0,
    };
    for r in 0..rows {
        // Gather row into Vec<Complex32>
        let mut buf: Vec<Complex32> = (0..cols).map(|c| {
            let re = x_reim[[r,c,0]];
            let im = x_reim[[r,c,1]];
            Complex32::new(re, im)
        }).collect();
        fft.process(&mut buf);
        for c in 0..cols {
            out[[r,c,0]] = buf[c].re * scale;
            out[[r,c,1]] = buf[c].im * scale;
        }
    }
    out
}
