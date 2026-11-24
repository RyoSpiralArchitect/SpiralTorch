use super::super::{sum::Sum, CausalLmResult};

struct Normalizer {
    peak: f64,
    log_sum: f64,
}

impl Normalizer {
    fn peak(logits: &[f32]) -> CausalLmResult<f64> {
        if logits.is_empty() || logits.iter().any(|x| !x.is_finite()) {
            return Err("anchor KL needs a finite nonempty vocabulary row".into());
        }
        Ok(f64::from(
            logits.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        ))
    }

    fn at_peak(logits: &[f32], peak: f64) -> Self {
        let mut sum = Sum::default();
        for &x in logits {
            sum.add((f64::from(x) - peak).exp());
        }
        Self {
            peak,
            log_sum: sum.value().ln(),
        }
    }

    fn log_p(&self, x: f32) -> f64 {
        (f64::from(x) - self.peak) - self.log_sum
    }
}

pub(super) struct Reference {
    norm: Normalizer,
    row: Vec<(f32, f64)>,
}

impl Reference {
    pub(super) fn new(logits: &[f32]) -> CausalLmResult<Self> {
        let norm = Normalizer::at_peak(logits, Normalizer::peak(logits)?);
        let row = logits.iter().map(|&x| (x, norm.log_p(x).exp())).collect();
        Ok(Self { norm, row })
    }

    fn ratios(&self, proposed: &[f32], mut visit: impl FnMut(f64, f64, f64)) -> CausalLmResult<()> {
        if self.row.len() != proposed.len() {
            return Err("anchor KL vocabulary mismatch".into());
        }
        let peak = Normalizer::peak(proposed)?;
        let shifted_delta = |old: f32, new: f32| {
            if self.norm.peak == peak {
                return f64::from(new) - f64::from(old);
            }
            let mut d = Sum::default();
            for term in [f64::from(new), -peak, -f64::from(old), self.norm.peak] {
                d.add(term);
            }
            d.value()
        };
        // For peak-shifted partition sums, (Zq - Zp)/Zp = sum p * expm1(delta).
        // This preserves shifts too small to survive subtracting two log softmaxes.
        let mut change = Sum::default();
        for (&(old, p), &new) in self.row.iter().zip(proposed) {
            let d = shifted_delta(old, new);
            change.add(if d.abs() < 0.5 {
                p * d.exp_m1()
            } else {
                ((f64::from(new) - peak) - self.norm.log_sum).exp() - p
            });
        }
        let relative = change.value();
        let (shift, log_sum) = if relative < -0.5 {
            // Avoid recovering a small partition ratio by adding nearly -1 to 1.
            let direct = Normalizer::at_peak(proposed, peak).log_sum;
            (direct - self.norm.log_sum, direct)
        } else {
            let shift = relative.ln_1p();
            (shift, self.norm.log_sum + shift)
        };
        let norm = Normalizer { peak, log_sum };
        for (&(old, p), &new) in self.row.iter().zip(proposed) {
            visit(p, norm.log_p(new), shifted_delta(old, new) - shift);
        }
        Ok(())
    }

    pub(super) fn probability_delta(&self, proposed: &[f32]) -> CausalLmResult<Vec<f32>> {
        let mut delta = Vec::with_capacity(self.row.len().min(proposed.len()));
        self.ratios(proposed, |p, lq, t| {
            delta.push(
                (if t.abs() < 0.5 {
                    p * t.exp_m1()
                } else {
                    lq.exp() - p
                }) as f32,
            );
        })?;
        Ok(delta)
    }

    pub(super) fn forward_kl(&self, proposed: &[f32]) -> CausalLmResult<f64> {
        let mut kl = Sum::default();
        self.ratios(proposed, |p, lq, t| {
            // Add p log(p/q) - p + q. These terms are nonnegative, avoid
            // cancellation near equality, and do not form q/p in the tails.
            kl.add(if t.abs() < 1e-4 {
                p * (t * t) * (0.5 + t * (1. / 6. + t / 24.))
            } else if t.abs() < 0.5 {
                p * (t.exp_m1() - t)
            } else {
                lq.exp() - p * (1. + t)
            });
        })?;
        let kl = kl.value();
        if !kl.is_finite() || kl < 0. {
            return Err("anchor KL is not finite and nonnegative".into());
        }
        Ok(kl)
    }
}

#[cfg(test)]
mod tests {
    use super::super::criterion::{CausalLmAnchorKlCriterion, Position};
    use super::*;

    #[test]
    fn below_floor_for_sub_ulp_binary_logit_motion() {
        let d = 2f32.powi(-53);
        let expected = f64::from(d).powi(2) / 8.; // log(cosh(d/2)) to f64 precision
        for (origin, candidate, sign) in [
            ([0., 0.], [0., d], 1.),
            ([0., 0.], [d, 0.], -1.),
            ([0., d], [1024., 1024.], -1.),
        ] {
            let reference = Reference::new(&origin).unwrap();
            let kl = reference.forward_kl(&candidate).unwrap();
            assert_eq!(
                CausalLmAnchorKlCriterion::Mean {}.position(2. * expected, kl, kl),
                Position::Below,
                "{origin:?} -> {candidate:?}: {kl:e}"
            );
            assert!((kl / expected - 1.).abs() < 1e-12);
            assert_eq!(
                reference.probability_delta(&candidate).unwrap(),
                vec![-sign * d / 4., sign * d / 4.]
            );
        }
    }

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
