/// Compensated f64 addition, including cancellation between parameter groups.
#[derive(Default)]
pub(super) struct Sum {
    high: f64,
    low: f64,
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

    pub(super) fn value(&self) -> f64 {
        self.high + self.low
    }

    pub(super) fn mean(self, count: u64) -> f64 {
        self.value() / count as f64
    }
}
