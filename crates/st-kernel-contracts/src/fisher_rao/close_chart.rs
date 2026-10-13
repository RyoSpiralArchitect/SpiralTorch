use crate::compensated::Sum;

fn exponential_remainder(x: f64) -> f64 {
    // exp(x)-1-x for |x|<=0.5, without subtracting its linear part.
    let mut term = x;
    let mut sum = Sum::default();
    for n in 2..=17 {
        term *= x / f64::from(n);
        sum.add(term);
    }
    sum.total()
}

#[derive(Clone, Copy, Debug)]
pub(super) struct CloseChart {
    pivot: usize,
    inverse_norm: f64,
    normalization_delta: f64,
}

fn relative_logit(x: &[f32], y: &[f32], pivot: usize, i: usize) -> f64 {
    let mut delta = Sum::default();
    // Do not round (x_i-x_p) and (y_i-y_p) separately: either can erase a
    // subnormal perturbation next to an otherwise identical large pivot.
    for value in [x[i], -x[pivot], -y[i], y[pivot]] {
        delta.add(f64::from(value));
    }
    delta.total()
}

fn mass_parts(value: f32, maximum: f32) -> [f64; 4] {
    let a = f64::from(value);
    let b = -f64::from(maximum);
    let leading = a + b;
    let rounded_b = leading - a;
    let residual = (a - (leading - rounded_b)) + (b - rounded_b);
    // Even maximal f32 gain/seed and u32-sized reductions cannot rescue
    // exp(-700) to an f32 gradient on the bounded close chart.
    if leading < -700. {
        return [0.; 4];
    }
    let exponent = (leading * std::f64::consts::LOG2_E).round();
    let remainder = (-exponent).mul_add(std::f64::consts::LN_2, leading);
    let base = 2f64.powi(exponent as i32);
    let body = remainder.exp_m1();
    let tail = residual.exp_m1();
    // Contract each part before adding them: 1 + expm1(tiny) would erase
    // the change in the key measure that contributes to the covariance.
    [base, base * body, base * tail, base * body * tail]
}

impl CloseChart {
    pub(super) fn new(x: &[f32], y: &[f32], roots: &[f64]) -> Option<Self> {
        let pivot = (1..y.len()).fold(0, |p, i| if y[i] > y[p] { i } else { p });
        let mut square = Sum::default();
        let mut change = Sum::default();
        for (i, &root) in roots.iter().enumerate() {
            let delta = relative_logit(x, y, pivot, i);
            if delta.abs() > 0.25 {
                return None;
            }
            square.product(root, root);
            change.product(root * root, delta);
            change.product(root * root, exponential_remainder(delta));
        }
        let square = square.total();
        let change = change.total();
        let a = (square + change).sqrt();
        let b = square.sqrt();
        Some(Self {
            pivot,
            inverse_norm: a.recip(),
            // A-B = (A^2-B^2)/(A+B), without subtracting rounded norms.
            normalization_delta: change / (a * b * (a + b)),
        })
    }

    pub(super) fn delta(self, x: &[f32], y: &[f32], root: f64, i: usize) -> f64 {
        let change = (0.5 * relative_logit(x, y, self.pivot, i)).exp_m1();
        root * (change * self.inverse_norm - self.normalization_delta)
    }

    pub(super) fn pullbacks(x: &[f32], y: &[f32], i: usize) -> (f64, f64) {
        // Recenter at the coordinate being differentiated. This exposes its
        // covariance directly instead of projecting huge cancelling root VJPs.
        // For t=||r_x-r_y||^2, E=sum b_j^2 exp(delta_j/2), A^2=sum
        // b_j^2 exp(delta_j), B^2=sum b_j^2: dt/dx_i=b_i^2(E-A^2)/(A^3 B),
        // dt/dy_i=b_i^2(E-B^2)/(A B^3). This close-only reference is O(cols^2).
        // Unnormalized key masses give the same covariance, but retain the
        // key's sub-ulp perturbations that rounded root coordinates cannot.
        let maximum = y.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut square = Sum::default();
        let mut linear = Sum::default();
        let mut half_remainder = Sum::default();
        let mut full_remainder = Sum::default();
        for (j, &value) in y.iter().enumerate() {
            let delta = relative_logit(x, y, i, j);
            let half = exponential_remainder(0.5 * delta);
            let full = exponential_remainder(delta);
            for mass in mass_parts(value, maximum) {
                square.add(mass);
                linear.product(mass, delta);
                half_remainder.product(mass, half);
                full_remainder.product(mass, full);
            }
        }
        let b2 = square.total();
        let linear = linear.total();
        let half = half_remainder.total();
        let full = full_remainder.total();
        let a2 = b2 + linear + full;
        let ab = (a2 * b2).sqrt();
        let mut dx = Sum::default();
        dx.add(-0.5 * linear);
        dx.add(half);
        dx.add(-full);
        let mut dy = Sum::default();
        dy.add(0.5 * linear);
        dy.add(half);
        let mass: f64 = mass_parts(y[i], maximum).into_iter().sum();
        (mass * dx.total() / (a2 * ab), mass * dy.total() / (b2 * ab))
    }
}
