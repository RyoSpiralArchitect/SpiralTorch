//! Grünwald–Letnikov coefficients and analytical derivative wrt alpha.
//! c_k(α) = (-1)^k * C(α,k) where C(α,k) = Γ(α+1)/[Γ(k+1)Γ(α-k+1)]
//! ∂c_k/∂α = c_k * (ψ(α+1) - ψ(α-k+1))
use crate::special::digamma;

pub fn gl_coeffs(alpha: f64, klen: usize) -> Vec<f64> {
    // Stable recurrence for generalized binomial coefficients
    // C(α,0)=1, C(α,k)=C(α,k-1)*(α-(k-1))/k
    let mut c = vec![0.0; klen];
    if klen == 0 { return c; }
    let mut cur = 1.0f64;
    c[0] = 1.0;
    for k in 1..klen {
        cur *= (alpha - (k as f64 - 1.0)) / (k as f64);
        c[k] = cur;
    }
    // apply (-1)^k
    for (k, v) in c.iter_mut().enumerate() {
        if k % 2 == 1 { *v = -*v; }
    }
    c
}

pub fn d_gl_coeffs(alpha: f64, klen: usize) -> Vec<f64> {
    // Use analytical derivative: dc_k = c_k * (ψ(α+1) - ψ(α-k+1))
    let c = gl_coeffs(alpha, klen);
    let psi_ap1 = digamma(alpha + 1.0);
    let mut out = vec![0.0; klen];
    for k in 0..klen {
        let psi_apkm1 = digamma(alpha - (k as f64) + 1.0);
        out[k] = c[k] * (psi_ap1 - psi_apkm1);
    }
    out
}
