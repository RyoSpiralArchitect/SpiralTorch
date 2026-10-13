// SPDX-License-Identifier: AGPL-3.0-or-later
//! Private finite reductions that retain small terms across cancellation.

#[derive(Clone, Default)]
pub(crate) struct Sum {
    partials: Vec<f64>,
}

impl Sum {
    pub(crate) fn add(&mut self, mut x: f64) {
        // A non-overlapping expansion, not one uncompensated correction.
        // Keep every lost term until later cancellation has participated.
        let mut count = 0;
        for i in 0..self.partials.len() {
            let mut y = self.partials[i];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let high = x + y;
            let low = y - (high - x);
            if low != 0. {
                self.partials[count] = low;
                count += 1;
            }
            x = high;
        }
        self.partials.truncate(count);
        if x != 0. {
            self.partials.push(x);
        }
    }

    pub(crate) fn product(&mut self, a: f64, b: f64) {
        let product = a * b;
        self.add(product);
        self.add(a.mul_add(b, -product));
    }

    pub(crate) fn total(&self) -> f64 {
        self.partials.iter().sum()
    }
}
