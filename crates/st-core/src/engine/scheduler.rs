use std::fmt;

/// Density summary emitted by runtime observers.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Density {
    pub act: f32,
    pub grad: f32,
    pub token_run: f32,
}

impl Density {
    pub fn clamp(self) -> Self {
        Self {
            act: self.act.clamp(0.0, 1.0),
            grad: self.grad.clamp(0.0, 1.0),
            token_run: self.token_run.clamp(0.0, 1.0),
        }
    }
}

/// Simple adaptive scheduler that modulates the learning rate and exploration
/// temperature based on density statistics.
#[derive(Clone, Debug)]
pub struct Scheduler {
    pub lr: f32,
    pub lr_min: f32,
    pub lr_max: f32,
    pub z_tau: f32,
    pub tau_min: f32,
    pub tau_max: f32,
}

impl Scheduler {
    pub fn new(lr: f32, z_tau: f32) -> Self {
        Self {
            lr,
            lr_min: lr * 0.1,
            lr_max: lr * 10.0,
            z_tau,
            tau_min: 0.1,
            tau_max: 4.0,
        }
    }

    /// Ingests density metrics and nudges the scheduler parameters toward more
    /// stable configurations.
    pub fn feed_density(&mut self, density: Density) {
        let d = density.clamp();
        let dense_act = d.act;
        let sparse_grad = 1.0 - d.grad;
        let monotony = d.token_run;

        let lr_decay = dense_act * 0.2 + monotony * 0.1;
        let lr_boost = sparse_grad * 0.15;
        self.lr = (self.lr * (1.0 + lr_boost - lr_decay)).clamp(self.lr_min, self.lr_max);

        let tau_inc = sparse_grad * 0.25 + (1.0 - monotony) * 0.1;
        let tau_dec = dense_act * 0.2;
        self.z_tau = (self.z_tau * (1.0 + tau_inc - tau_dec)).clamp(self.tau_min, self.tau_max);
    }
}

impl fmt::Display for Scheduler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Scheduler(lr={:.4}, tau={:.3})", self.lr, self.z_tau)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheduler_adapts_to_density() {
        let mut sched = Scheduler::new(0.05, 1.0);
        let dense = Density {
            act: 0.9,
            grad: 0.2,
            token_run: 0.8,
        };
        sched.feed_density(dense);
        assert!(sched.lr < 0.05);
        let sparse = Density {
            act: 0.2,
            grad: 0.1,
            token_run: 0.2,
        };
        sched.feed_density(sparse);
        assert!(sched.lr > 0.04);
    }
}
