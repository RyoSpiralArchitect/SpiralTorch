// SPDX-License-Identifier: AGPL-3.0-or-later
//! Headwise convex mixtures. The router is shared over batch/positions, not
//! conditioned on a future token. Gradients are sums, never implicit means.
use crate::compensated::Sum;
use thiserror::Error;

mod querywise;
pub use querywise::{QuerywiseMixtureForward, QuerywiseMixtureSpec, QuerywiseMixtureVjp};

pub const MAX_COMPONENTS: usize = 8;

#[derive(Debug, Error, PartialEq)]
pub enum MixtureError {
    #[error("mixture needs nonempty [batch,heads,positions,components], with 2..=8 components")]
    Shape,
    #[error("mixture dimensions exceed portable addressing")]
    Overflow,
    #[error("mixture input, router or cotangent shape mismatch")]
    Length,
    #[error("mixture input, output or derivative is non-finite")]
    NonFinite,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadwiseMixtureSpec {
    shape: [usize; 4],
    len: usize,
}

impl HeadwiseMixtureSpec {
    pub fn new(shape: [usize; 4]) -> Result<Self, MixtureError> {
        if shape.contains(&0) || !(2..=MAX_COMPONENTS).contains(&shape[3]) {
            return Err(MixtureError::Shape);
        }
        let len = shape
            .iter()
            .try_fold(1usize, |a, b| a.checked_mul(*b))
            .ok_or(MixtureError::Overflow)?;
        u32::try_from(len).map_err(|_| MixtureError::Overflow)?;
        Ok(Self { shape, len })
    }
    pub fn shape(self) -> [usize; 4] {
        self.shape
    }
    pub fn len(self) -> usize {
        self.len
    }
    pub fn is_empty(self) -> bool {
        false
    }
    pub fn output_len(self) -> usize {
        self.len / self.shape[3]
    }
    pub fn router_len(self) -> usize {
        self.shape[1] * self.shape[3]
    }
    pub fn validate_lengths(self, input: usize, logits: usize) -> Result<(), MixtureError> {
        if input != self.len || logits != self.router_len() {
            return Err(MixtureError::Length);
        }
        Ok(())
    }
}

fn finite(v: f32) -> Result<f32, MixtureError> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(MixtureError::NonFinite)
    }
}

#[derive(Clone, Debug)]
pub struct HeadwiseMixtureForward {
    spec: HeadwiseMixtureSpec,
    input: Vec<f32>,
    weights: Vec<f32>,
    output: Vec<f32>,
}

#[derive(Debug)]
pub struct HeadwiseMixtureVjp {
    pub input: Vec<f32>,
    pub logits: Vec<f32>,
}

impl HeadwiseMixtureForward {
    /// output[b,h,p] = sum_k softmax(logits[h,:])[k] * input[b,h,p,k].
    pub fn new(
        spec: HeadwiseMixtureSpec,
        input: &[f32],
        logits: &[f32],
    ) -> Result<Self, MixtureError> {
        spec.validate_lengths(input.len(), logits.len())?;
        for &v in input.iter().chain(logits) {
            finite(v)?;
        }
        let [_, heads, positions, components] = spec.shape;
        let mut weights = vec![0.; logits.len()];
        for (row, weight) in logits
            .chunks_exact(components)
            .zip(weights.chunks_exact_mut(components))
        {
            let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut total = 0.;
            for (w, &v) in weight.iter_mut().zip(row) {
                // A negative infinite difference of finite extreme logits is
                // a valid zero softmax mass, not a non-finite model parameter.
                *w = (v - maximum).exp();
                total = finite(total + *w)?;
            }
            for w in weight {
                *w = finite(*w / total)?;
            }
        }
        let mut output = vec![0.; spec.output_len()];
        for (i, (row, y)) in input.chunks_exact(components).zip(&mut output).enumerate() {
            let h = i / positions % heads;
            for (x, w) in row
                .iter()
                .zip(&weights[h * components..(h + 1) * components])
            {
                *y = finite(*y + finite(x * w)?)?;
            }
        }
        Ok(Self {
            spec,
            input: input.to_vec(),
            weights,
            output,
        })
    }
    pub fn output(&self) -> &[f32] {
        &self.output
    }
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }
    pub fn backward(&self, seed: &[f32]) -> Result<HeadwiseMixtureVjp, MixtureError> {
        if seed.len() != self.spec.output_len() {
            return Err(MixtureError::Length);
        }
        for &s in seed {
            finite(s)?;
        }
        let [_, heads, positions, components] = self.spec.shape;
        let mut input = vec![0.; self.spec.len];
        let mut reduced = vec![Sum::default(); self.spec.router_len()];
        for (i, &s) in seed.iter().enumerate() {
            let h = i / positions % heads;
            let row = &self.input[i * components..(i + 1) * components];
            let weights = &self.weights[h * components..(h + 1) * components];
            for k in 0..components {
                input[i * components + k] = finite(s * weights[k])?;
                // Center each position before reduction so a large common
                // component cannot erase a later position's learning signal.
                let mut centered = Sum::default();
                for (&x, &w) in row.iter().zip(weights) {
                    centered.add(f64::from(w) * (f64::from(row[k]) - f64::from(x)));
                }
                reduced[h * components + k].product(f64::from(s), centered.total());
            }
        }
        let logits = reduced
            .iter()
            .zip(&self.weights)
            .map(|(&g, &w)| finite((f64::from(w) * g.total()) as f32))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(HeadwiseMixtureVjp { input, logits })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dynamic_range_router_reduction_keeps_cancelled_residual() {
        let x = [0., 2., 0., 2., 0., 2.];
        for exponent in [24, 60, 96, 124] {
            let seed = [2f32.powi(exponent), 1., -2f32.powi(exponent)];
            let shared = HeadwiseMixtureForward::new(
                HeadwiseMixtureSpec::new([1, 1, 3, 2]).unwrap(),
                &x,
                &[0., 0.],
            )
            .unwrap();
            assert_eq!(shared.backward(&seed).unwrap().logits, [-0.5, 0.5]);
            let query = QuerywiseMixtureForward::new(
                QuerywiseMixtureSpec::new([1, 1, 1, 3, 2]).unwrap(),
                &x,
                &[0., 0.],
            )
            .unwrap();
            assert_eq!(query.backward(&seed).unwrap().logits, [-0.5, 0.5]);
        }
    }

    #[test]
    fn shape_and_portable_lengths_are_checked() {
        for shape in [
            [0, 1, 1, 3],
            [1, 0, 1, 3],
            [1, 1, 0, 3],
            [1, 1, 1, 1],
            [1, 1, 1, 9],
        ] {
            assert_eq!(HeadwiseMixtureSpec::new(shape), Err(MixtureError::Shape));
        }
        assert_eq!(
            HeadwiseMixtureSpec::new([usize::MAX, 2, 1, 3]),
            Err(MixtureError::Overflow)
        );
        assert_eq!(
            HeadwiseMixtureSpec::new([u32::MAX as usize, 1, 1, 3]),
            Err(MixtureError::Overflow)
        );
        let spec = HeadwiseMixtureSpec::new([2, 3, 5, 3]).unwrap();
        assert_eq!(
            (spec.len(), spec.output_len(), spec.router_len()),
            (90, 30, 9)
        );
        assert!(spec.validate_lengths(90, 9).is_ok());
        assert!(spec.validate_lengths(90, 3).is_err());
    }

    #[test]
    fn vjp_matches_directional_differences_for_inputs_and_router() {
        let spec = HeadwiseMixtureSpec::new([2, 2, 3, 3]).unwrap();
        let x: Vec<_> = (0..spec.len()).map(|i| (i as f32 * 0.7).sin()).collect();
        let logits = vec![-0.2, 0.7, -0.8, 1.2, -0.3, 0.6];
        let seed: Vec<_> = (0..spec.output_len())
            .map(|i| (i as f32 * 0.3).cos())
            .collect();
        let f = HeadwiseMixtureForward::new(spec, &x, &logits).unwrap();
        let g = f.backward(&seed).unwrap();
        let loss = |x: &[f32], l: &[f32]| -> f32 {
            HeadwiseMixtureForward::new(spec, x, l)
                .unwrap()
                .output()
                .iter()
                .zip(&seed)
                .map(|(a, b)| a * b)
                .sum()
        };
        for (values, grad, router) in [(&x, &g.input, false), (&logits, &g.logits, true)] {
            for i in 0..values.len() {
                let mut a = values.clone();
                let mut b = values.clone();
                a[i] += 0.002;
                b[i] -= 0.002;
                let numeric = if router {
                    (loss(&x, &a) - loss(&x, &b)) / 0.004
                } else {
                    (loss(&a, &logits) - loss(&b, &logits)) / 0.004
                };
                assert!(
                    (numeric - grad[i]).abs() < 2e-4,
                    "{router} {i}: {numeric} != {}",
                    grad[i]
                );
            }
        }
        for row in g.logits.as_chunks::<3>().0 {
            assert!(row.iter().sum::<f32>().abs() < 1e-6);
        }
    }

    #[test]
    fn saturated_finite_logits_and_shared_batch_router() {
        let spec = HeadwiseMixtureSpec::new([2, 1, 1, 3]).unwrap();
        let f = HeadwiseMixtureForward::new(
            spec,
            &[1., 2., 3., 4., 5., 6.],
            &[-f32::MAX, f32::MAX, 0.],
        )
        .unwrap();
        assert_eq!(f.weights(), &[0., 1., 0.]);
        assert_eq!(f.output(), &[2., 5.]);
        let g = f.backward(&[2., 3.]).unwrap();
        assert_eq!(g.input, [0., 2., 0., 0., 3., 0.]);
        assert_eq!(g.logits, [0., 0., 0.]);
    }

    #[test]
    fn nonfinite_and_shape_mismatches_are_not_silently_ignored() {
        let s = HeadwiseMixtureSpec::new([1, 1, 1, 2]).unwrap();
        assert!(HeadwiseMixtureForward::new(s, &[1., f32::NAN], &[0., 0.]).is_err());
        assert!(HeadwiseMixtureForward::new(s, &[1., 2.], &[0., f32::INFINITY]).is_err());
        assert!(HeadwiseMixtureForward::new(s, &[1.], &[0., 0.]).is_err());
        let f = HeadwiseMixtureForward::new(s, &[1., 2.], &[0., 0.]).unwrap();
        assert!(f.backward(&[f32::NAN]).is_err());
        assert!(f.backward(&[]).is_err());
        let gradient = f.backward(&[f32::MAX]).unwrap();
        assert_eq!(gradient.input, [f32::MAX / 2.; 2]);
        assert_eq!(gradient.logits, [-f32::MAX / 4., f32::MAX / 4.]);
    }

    #[test]
    fn wide_centering_keeps_finite_vjps_and_removes_identical_components() {
        let s = HeadwiseMixtureSpec::new([2, 1, 1, 3]).unwrap();
        let same = HeadwiseMixtureForward::new(s, &[f32::MAX; 6], &[-0.3, 0.2, 0.7]).unwrap();
        assert_eq!(same.backward(&[f32::MAX; 2]).unwrap().logits, [0.; 3]);
        let two = HeadwiseMixtureForward::new(
            HeadwiseMixtureSpec::new([1, 1, 1, 2]).unwrap(),
            &[1., 3.],
            &[0., 0.],
        )
        .unwrap();
        assert_eq!(
            two.backward(&[f32::MAX]).unwrap().logits,
            [-f32::MAX / 2., f32::MAX / 2.]
        );
        let overflow = HeadwiseMixtureForward::new(
            HeadwiseMixtureSpec::new([1, 1, 1, 2]).unwrap(),
            &[1., 10.],
            &[0., 0.],
        )
        .unwrap();
        assert!(overflow.backward(&[f32::MAX]).is_err());
    }

    #[test]
    fn centering_precedes_position_and_batch_reduction() {
        for shape in [[1, 1, 2, 2], [2, 1, 1, 2]] {
            let forward = HeadwiseMixtureForward::new(
                HeadwiseMixtureSpec::new(shape).unwrap(),
                &[f32::MAX, f32::MAX, 1., 2.],
                &[0., 0.],
            )
            .unwrap();
            assert_eq!(forward.output(), &[f32::MAX, 1.5]);
            let gradient = forward.backward(&[1., 1.]).unwrap();
            assert_eq!(gradient.input, [0.5; 4]);
            assert_eq!(gradient.logits, [-0.25, 0.25]);
        }
    }
}
