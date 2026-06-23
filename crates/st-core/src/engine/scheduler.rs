// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::fmt;

use spiral_config::determinism;
use st_tensor::{emit_tensor_op, emit_tensor_op_meta};

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
            act: clamp_unit_density(self.act),
            grad: clamp_unit_density(self.grad),
            token_run: clamp_unit_density(self.token_run),
        }
    }

    fn is_finite(self) -> bool {
        self.act.is_finite() && self.grad.is_finite() && self.token_run.is_finite()
    }
}

fn clamp_unit_density(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn density_sanitized(raw: Density, clamped: Density) -> bool {
    !raw.is_finite()
        || raw.act != clamped.act
        || raw.grad != clamped.grad
        || raw.token_run != clamped.token_run
}

fn finite_nonnegative(value: f32, fallback: f32) -> f32 {
    if value.is_finite() && value >= 0.0 {
        value
    } else {
        fallback
    }
}

fn finite_positive(value: f32, fallback: f32) -> f32 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        fallback
    }
}

fn finite_scaled_clamped(value: f32, factor: f32, min: f32, max: f32) -> f32 {
    let next = f64::from(value) * f64::from(factor);
    next.clamp(f64::from(min), f64::from(max)) as f32
}

fn finite_meta_f32(value: f32) -> serde_json::Value {
    serde_json::Value::from(if value.is_finite() { value as f64 } else { 0.0 })
}

fn density_drive_label(density: Density) -> &'static str {
    if density.act >= 0.75 && density.token_run >= 0.6 {
        "dense_monotony"
    } else if density.act >= 0.75 {
        "dense_activation"
    } else if density.grad <= 0.25 {
        "sparse_gradient"
    } else if density.token_run >= 0.75 {
        "token_monotony"
    } else {
        "balanced"
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "Metadata mirrors scheduler control terms"
)]
fn emit_scheduler_feed_density_meta(
    scheduler: &Scheduler,
    raw_density: Density,
    density: Density,
    locked: bool,
    density_valid: bool,
    state_sanitized: bool,
    lr_before: f32,
    tau_before: f32,
    lr_decay: f32,
    lr_boost: f32,
    tau_inc: f32,
    tau_dec: f32,
) {
    emit_tensor_op("scheduler_feed_density", &[1, 3], &[1, 2]);
    emit_tensor_op_meta("scheduler_feed_density", || {
        let mut payload = serde_json::Map::new();
        payload.insert("backend".into(), "cpu".into());
        payload.insert("requested_backend".into(), "auto".into());
        payload.insert("kind".into(), "st_core_scheduler_feed_density".into());
        payload.insert("locked".into(), locked.into());
        payload.insert("density_valid".into(), density_valid.into());
        payload.insert(
            "density_sanitized".into(),
            density_sanitized(raw_density, density).into(),
        );
        payload.insert("state_sanitized".into(), state_sanitized.into());
        payload.insert("drive_label".into(), density_drive_label(density).into());
        payload.insert("raw_act_density".into(), finite_meta_f32(raw_density.act));
        payload.insert("raw_grad_density".into(), finite_meta_f32(raw_density.grad));
        payload.insert(
            "raw_token_run_density".into(),
            finite_meta_f32(raw_density.token_run),
        );
        payload.insert("act_density".into(), finite_meta_f32(density.act));
        payload.insert("grad_density".into(), finite_meta_f32(density.grad));
        payload.insert(
            "token_run_density".into(),
            finite_meta_f32(density.token_run),
        );
        payload.insert("sparse_grad".into(), finite_meta_f32(1.0 - density.grad));
        payload.insert("lr_before".into(), finite_meta_f32(lr_before));
        payload.insert("lr_after".into(), finite_meta_f32(scheduler.lr));
        payload.insert("lr_delta".into(), finite_meta_f32(scheduler.lr - lr_before));
        payload.insert("lr_min".into(), finite_meta_f32(scheduler.lr_min));
        payload.insert("lr_max".into(), finite_meta_f32(scheduler.lr_max));
        payload.insert("lr_decay".into(), finite_meta_f32(lr_decay));
        payload.insert("lr_boost".into(), finite_meta_f32(lr_boost));
        payload.insert("tau_before".into(), finite_meta_f32(tau_before));
        payload.insert("tau_after".into(), finite_meta_f32(scheduler.z_tau));
        payload.insert(
            "tau_delta".into(),
            finite_meta_f32(scheduler.z_tau - tau_before),
        );
        payload.insert("tau_min".into(), finite_meta_f32(scheduler.tau_min));
        payload.insert("tau_max".into(), finite_meta_f32(scheduler.tau_max));
        payload.insert("tau_inc".into(), finite_meta_f32(tau_inc));
        payload.insert("tau_dec".into(), finite_meta_f32(tau_dec));
        serde_json::Value::Object(payload)
    });
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
        let lr = finite_nonnegative(lr, 1.0e-3);
        let lr_min = finite_nonnegative(lr * 0.1, 0.0);
        let lr_max = if (lr * 10.0).is_finite() {
            (lr * 10.0).max(lr_min)
        } else {
            f32::MAX
        };
        let z_tau = finite_positive(z_tau, 1.0);
        let mut scheduler = Self {
            lr,
            lr_min,
            lr_max,
            z_tau,
            tau_min: 0.1,
            tau_max: 4.0,
        };
        scheduler.normalize_state();
        scheduler
    }

    fn normalize_state(&mut self) -> bool {
        let before = (
            self.lr,
            self.lr_min,
            self.lr_max,
            self.z_tau,
            self.tau_min,
            self.tau_max,
        );
        self.lr_min = finite_nonnegative(self.lr_min, 0.0);
        self.lr_max = finite_nonnegative(self.lr_max, self.lr_min.max(1.0e-6));
        if self.lr_max < self.lr_min {
            self.lr_max = self.lr_min;
        }
        self.lr = finite_nonnegative(self.lr, self.lr_min);
        self.lr = self.lr.clamp(self.lr_min, self.lr_max);

        self.tau_min = finite_positive(self.tau_min, 0.1);
        self.tau_max = finite_positive(self.tau_max, self.tau_min.max(4.0));
        if self.tau_max < self.tau_min {
            self.tau_max = self.tau_min;
        }
        self.z_tau = finite_positive(self.z_tau, self.tau_min.max(1.0));
        self.z_tau = self.z_tau.clamp(self.tau_min, self.tau_max);
        before
            != (
                self.lr,
                self.lr_min,
                self.lr_max,
                self.z_tau,
                self.tau_min,
                self.tau_max,
            )
    }

    /// Ingests density metrics and nudges the scheduler parameters toward more
    /// stable configurations.
    pub fn feed_density(&mut self, density: Density) {
        let raw_density = density;
        let d = density.clamp();
        let density_valid = raw_density.is_finite();
        let state_sanitized = self.normalize_state();
        let lr_before = self.lr;
        let tau_before = self.z_tau;
        let locked = determinism::lock_scheduler();
        if locked || !density_valid {
            emit_scheduler_feed_density_meta(
                self,
                raw_density,
                d,
                locked,
                density_valid,
                state_sanitized,
                lr_before,
                tau_before,
                0.0,
                0.0,
                0.0,
                0.0,
            );
            return;
        }
        let dense_act = d.act;
        let sparse_grad = 1.0 - d.grad;
        let monotony = d.token_run;

        let lr_decay = dense_act * 0.2 + monotony * 0.1;
        let lr_boost = sparse_grad * 0.15;
        self.lr =
            finite_scaled_clamped(self.lr, 1.0 + lr_boost - lr_decay, self.lr_min, self.lr_max);

        let tau_inc = sparse_grad * 0.25 + (1.0 - monotony) * 0.1;
        let tau_dec = dense_act * 0.2;
        self.z_tau = finite_scaled_clamped(
            self.z_tau,
            1.0 + tau_inc - tau_dec,
            self.tau_min,
            self.tau_max,
        );

        emit_scheduler_feed_density_meta(
            self,
            raw_density,
            d,
            false,
            density_valid,
            state_sanitized,
            lr_before,
            tau_before,
            lr_decay,
            lr_boost,
            tau_inc,
            tau_dec,
        );
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
    use std::sync::{Arc, Mutex};

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

    #[test]
    fn scheduler_feed_density_emits_backend_meta() {
        let _lock = crate::telemetry::tensor_observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let mut sched = Scheduler::new(0.05, 1.0);
        sched.feed_density(Density {
            act: 0.9,
            grad: 0.2,
            token_run: 0.8,
        });
        st_tensor::set_tensor_op_meta_observer(previous);

        let events = events.lock().unwrap();
        let meta = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "scheduler_feed_density"
                    && data["kind"] == "st_core_scheduler_feed_density"
                    && data["drive_label"] == "dense_monotony"
            })
            .expect("scheduler metadata event");
        assert_eq!(meta.1["backend"], "cpu");
        assert_eq!(meta.1["locked"], false);
        assert!(meta.1["lr_after"].as_f64().unwrap() < meta.1["lr_before"].as_f64().unwrap());
        assert!(meta.1["tau_after"].as_f64().unwrap() > meta.1["tau_before"].as_f64().unwrap());
        assert!(meta.1["lr_decay"].as_f64().unwrap() > meta.1["lr_boost"].as_f64().unwrap());
        assert!(meta.1["tau_inc"].as_f64().unwrap() > meta.1["tau_dec"].as_f64().unwrap());
    }

    #[test]
    fn scheduler_ignores_non_finite_density_without_poisoning_state() {
        let _lock = crate::telemetry::tensor_observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let mut sched = Scheduler::new(f32::NAN, f32::INFINITY);
        sched.lr = f32::NAN;
        sched.lr_min = f32::NAN;
        sched.lr_max = f32::NAN;
        sched.z_tau = f32::INFINITY;
        sched.tau_min = f32::NAN;
        sched.tau_max = f32::NAN;
        sched.feed_density(Density {
            act: f32::NAN,
            grad: f32::INFINITY,
            token_run: 0.8,
        });
        st_tensor::set_tensor_op_meta_observer(previous);

        assert!(sched.lr.is_finite());
        assert!(sched.lr_min.is_finite());
        assert!(sched.lr_max.is_finite());
        assert!(sched.z_tau.is_finite());
        assert!(sched.tau_min.is_finite());
        assert!(sched.tau_max.is_finite());
        assert!(sched.lr_min <= sched.lr && sched.lr <= sched.lr_max);
        assert!(sched.tau_min <= sched.z_tau && sched.z_tau <= sched.tau_max);

        let events = events.lock().unwrap();
        let meta = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "scheduler_feed_density"
                    && data["kind"] == "st_core_scheduler_feed_density"
                    && data["density_valid"] == false
                    && data["density_sanitized"] == true
            })
            .expect("scheduler metadata event");
        assert_eq!(meta.1["density_valid"], false);
        assert_eq!(meta.1["density_sanitized"], true);
        assert_eq!(meta.1["state_sanitized"], true);
        assert_eq!(meta.1["lr_decay"], 0.0);
        assert_eq!(meta.1["tau_inc"], 0.0);
    }
}
