// SPDX-License-Identifier: AGPL-3.0-or-later
//! Private finite reductions that retain small terms across cancellation.

#[derive(Clone, Copy, Default)]
pub(crate) struct Sum {
    value: f64,
    correction: f64,
}

impl Sum {
    pub(crate) fn add(&mut self, x: f64) {
        let total = self.value + x;
        self.correction += if self.value.abs() >= x.abs() {
            (self.value - total) + x
        } else {
            (x - total) + self.value
        };
        self.value = total;
    }

    pub(crate) fn product(&mut self, a: f64, b: f64) {
        let product = a * b;
        self.add(product);
        self.add(a.mul_add(b, -product));
    }

    pub(crate) fn total(self) -> f64 {
        self.value + self.correction
    }
}
