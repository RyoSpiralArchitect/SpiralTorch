//! Special functions (digamma) — lightweight approximation.
//! We use recurrence to push x to > 8, then asymptotic expansion.
pub fn digamma(mut x: f64) -> f64 {
    // Handle non-positive by reflection if needed (simple guard)
    if x <= 0.0 {
        // ψ(x) = ψ(1-x) + π cot(π x), but avoid singularities; shift to positive side
        // For practicality: shift until x>0 using ψ(x) = ψ(x+1) - 1/x
        // (This is a crude guard; for production clip/return finite)
        let mut s = 0.0;
        while x <= 0.0 {
            s -= 1.0 / x;
            x += 1.0;
        }
        return digamma(x) + s;
    }
    // Use recurrence to large x
    let mut s = 0.0;
    while x < 8.0 {
        s -= 1.0 / x;
        x += 1.0;
    }
    // Asymptotic expansion (Bernoulli numbers up to 1/x^8)
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    s + (x.ln() - 0.5*inv - inv2*(1.0/12.0) + inv2*inv2*(1.0/120.0) - inv2*inv2*inv2*(1.0/252.0))
}
