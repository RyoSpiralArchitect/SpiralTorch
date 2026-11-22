use super::super::{sum::Sum, CausalLmResult};

struct Normalizer {
    peak: f64,
    log_sum: f64,
}

impl Normalizer {
    fn new(logits: &[f32]) -> CausalLmResult<Self> {
        if logits.is_empty() || logits.iter().any(|x| !x.is_finite()) {
            return Err("anchor KL needs a finite nonempty vocabulary row".into());
        }
        let peak = f64::from(logits.iter().copied().fold(f32::NEG_INFINITY, f32::max));
        let mut sum = Sum::default();
        for &x in logits {
            sum.add((f64::from(x) - peak).exp());
        }
        Ok(Self {
            peak,
            log_sum: sum.value().ln(),
        })
    }

    fn log_p(&self, x: f32) -> f64 {
        (f64::from(x) - self.peak) - self.log_sum
    }
}

pub(super) struct Reference(Vec<(f64, f64)>);

impl Reference {
    pub(super) fn new(logits: &[f32]) -> CausalLmResult<Self> {
        let norm = Normalizer::new(logits)?;
        Ok(Self(
            logits
                .iter()
                .map(|&x| {
                    let lp = norm.log_p(x);
                    (lp, lp.exp())
                })
                .collect(),
        ))
    }

    pub(super) fn probability_delta(&self, proposed: &[f32]) -> CausalLmResult<Vec<f32>> {
        if self.0.len() != proposed.len() {
            return Err("anchor KL vocabulary mismatch".into());
        }
        let norm = Normalizer::new(proposed)?;
        Ok(self
            .0
            .iter()
            .zip(proposed)
            .map(|(&(lp, p), &x)| {
                let lq = norm.log_p(x);
                let t = lq - lp;
                (if t.abs() < 0.5 {
                    p * t.exp_m1()
                } else {
                    lq.exp() - p
                }) as f32
            })
            .collect())
    }

    pub(super) fn forward_kl(&self, proposed: &[f32]) -> CausalLmResult<f64> {
        if self.0.len() != proposed.len() {
            return Err("anchor KL vocabulary mismatch".into());
        }
        let norm = Normalizer::new(proposed)?;
        let mut kl = Sum::default();
        for (&(lp, p), &x) in self.0.iter().zip(proposed) {
            let lq = norm.log_p(x);
            let t = lq - lp;
            // Add p log(p/q) - p + q. These terms are nonnegative, avoid
            // cancellation near equality, and do not form q/p in the tails.
            kl.add(if t.abs() < 1e-4 {
                p * (t * t) * (0.5 + t * (1. / 6. + t / 24.))
            } else if t.abs() < 0.5 {
                p * (t.exp_m1() - t)
            } else {
                lq.exp() - p * (1. + t)
            });
        }
        let kl = kl.value();
        if !kl.is_finite() || kl < 0. {
            return Err("anchor KL is not finite and nonnegative".into());
        }
        Ok(kl)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_for_offsets_tiny_deltas_and_extreme_tails() {
        let p = Reference::new(&[0., 0.]).unwrap();
        assert_eq!(p.forward_kl(&[800., 800.]).unwrap(), 0.);
        let delta = 1e-7f32;
        let tiny = p.forward_kl(&[-delta, delta]).unwrap();
        assert!((tiny / (0.5 * f64::from(delta).powi(2)) - 1.).abs() < 1e-8);
        let check = |v: f64, want: f64| assert!((v - want).abs() < 1e-12);
        check(p.forward_kl(&[-1000., 1000.]).unwrap(), 1000. - 2f64.ln());
        let concentrated = Reference::new(&[-1000., 1000.]).unwrap();
        assert_eq!(
            concentrated.probability_delta(&[1000., -1000.]).unwrap(),
            vec![1., -1.]
        );
        assert_eq!(p.probability_delta(&[800., 800.]).unwrap(), vec![0., 0.]);
        assert!(p.probability_delta(&[1.]).is_err());
        check(concentrated.forward_kl(&[0., 0.]).unwrap(), 2f64.ln());
        check(concentrated.forward_kl(&[1000., -1000.]).unwrap(), 2000.);
        assert!(p.forward_kl(&[0.]).is_err());
        assert!(p.forward_kl(&[f32::NAN, 0.]).is_err());
        assert!(Reference::new(&[]).is_err());
        assert!(Reference::new(&[f32::INFINITY]).is_err());
    }
}
