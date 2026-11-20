/// Compensated f64 addition, including cancellation between parameter groups.
#[derive(Default)]
pub(super) struct Sum {
    high: f64,
    low: f64,
}

/// Add nonnegative quantities represented by their logarithms. None is exact zero.
pub(super) fn add_logs(a: Option<f64>, b: Option<f64>) -> Option<f64> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.max(b) + (-(a - b).abs()).exp().ln_1p()),
        _ => a.or(b),
    }
}

impl Sum {
    pub(super) fn add(&mut self, v: f64) {
        let n = self.high + v;
        self.low += if self.high.abs() >= v.abs() {
            (self.high - n) + v
        } else {
            (v - n) + self.high
        };
        self.high = n;
    }

    pub(super) fn merge(&mut self, other: &Self) {
        self.add(other.high);
        self.add(other.low);
    }

    pub(super) fn add_scaled(&mut self, other: &Self, scale: f64) {
        self.add(other.high * scale);
        self.add(other.low * scale);
    }

    pub(super) fn value(&self) -> f64 {
        self.high + self.low
    }

    pub(super) fn mean(self, count: u64) -> f64 {
        self.value() / count as f64
    }
}
