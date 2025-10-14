//! Measure-theoretic helpers for working with Z-space actions.
//!
//! The utilities implemented here follow the categorical view outlined in the
//! user guide: a quasi-invariant group action \(G \curvearrowright (\Omega, \Sigma, \mu)\)
//! induces a Koopman representation \((U_g)_{g \in G}\) on \(L^2(\mu)\).
//! Averaging the pullbacks of a function through a Følner sequence recovers the
//! conditional expectation onto the invariant \(\sigma\)-algebra.  These
//! helpers expose that construction directly on top of the pure tensor core so
//! higher layers can project activations onto the shared "vocabulary" slice
//! without destroying the phase information that lives in the complementary
//! directions.

use super::{PureResult, Tensor, TensorError};

/// Trait describing how a group element acts on a tensor through the associated
/// Koopman operator.  Callers provide a concrete implementation that performs
/// the pullback for their representation.
pub trait KoopmanAction<G> {
    /// Applies the Koopman operator associated with `element` to `input`.
    fn apply(&self, element: &G, input: &Tensor) -> PureResult<Tensor>;
}

/// Computes the conditional expectation `E[f | Σ^G]` by averaging the images of
/// `f` under the Koopman operators indexed by `elements`.
///
/// The caller is responsible for supplying a Følner set (or any finite subset of
/// the acting group).  The result converges to the invariant projection when
/// the supplied sequence grows along a Følner exhaustion.
pub fn conditional_expectation<G, A>(action: &A, elements: &[G], f: &Tensor) -> PureResult<Tensor>
where
    A: KoopmanAction<G>,
{
    if elements.is_empty() {
        return Err(TensorError::EmptyInput("conditional_expectation"));
    }
    let (rows, cols) = f.shape();
    let mut accumulator = Tensor::zeros(rows, cols)?;
    for element in elements {
        let transformed = action.apply(element, f)?;
        accumulator.add_scaled(&transformed, 1.0)?;
    }
    accumulator.scale(1.0 / elements.len() as f32)
}

/// Produces the running Cesàro averages associated with a Følner sequence.
///
/// Each entry in `sequence` is treated as a finite subset of the acting group.
/// The returned vector contains the partial conditional expectations after each
/// step which makes it easy to monitor convergence in practice.
pub fn cesaro_averages<G, A, I>(action: &A, sequence: I, f: &Tensor) -> PureResult<Vec<Tensor>>
where
    A: KoopmanAction<G>,
    I: IntoIterator,
    I::Item: AsRef<[G]>,
{
    let mut averages = Vec::new();
    for (idx, subset) in sequence.into_iter().enumerate() {
        let subset_ref = subset.as_ref();
        if subset_ref.is_empty() {
            return Err(TensorError::EmptyInput("cesaro_averages"));
        }
        let projection = conditional_expectation(action, subset_ref, f)?;
        // To avoid accidental aliasing we clone once before pushing.
        averages.push(projection.clone());
        // We overwrite the vector entry with the newly computed projection so
        // the caller can observe convergence while preserving ownership of the
        // tensor.
        averages[idx] = projection;
    }
    Ok(averages)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug)]
    struct CyclicShift {
        width: usize,
    }

    impl KoopmanAction<usize> for CyclicShift {
        fn apply(&self, element: &usize, input: &Tensor) -> PureResult<Tensor> {
            let (rows, cols) = input.shape();
            let width = self.width;
            assert_eq!(cols % width, 0, "input columns must be a multiple of width");
            let channels = cols / width;
            let mut out = Tensor::zeros(rows, cols)?;
            for r in 0..rows {
                let row = &input.data()[r * cols..(r + 1) * cols];
                let out_row = &mut out.data_mut()[r * cols..(r + 1) * cols];
                for c in 0..channels {
                    for x in 0..width {
                        let src = (x + element) % width;
                        out_row[c * width + x] = row[c * width + src];
                    }
                }
            }
            Ok(out)
        }
    }

    #[test]
    fn projection_recovers_invariants() {
        let action = CyclicShift { width: 3 };
        let input = Tensor::from_vec(1, 3, vec![1.0, 2.0, 3.0]).unwrap();
        let elements = vec![0usize, 1, 2];
        let projected = conditional_expectation(&action, &elements, &input).unwrap();
        assert_eq!(projected.data(), &[2.0, 2.0, 2.0]);
    }

    #[test]
    fn cesaro_sequence_converges() {
        let action = CyclicShift { width: 3 };
        let input = Tensor::from_vec(1, 3, vec![3.0, 0.0, 0.0]).unwrap();
        let sequence = vec![vec![0usize], vec![0usize, 1], vec![0usize, 1, 2]];
        let averages = cesaro_averages(&action, sequence, &input).unwrap();
        assert_eq!(averages.len(), 3);
        assert_eq!(averages[0].data(), &[3.0, 0.0, 0.0]);
        assert_eq!(averages[1].data(), &[1.5, 1.5, 0.0]);
        assert_eq!(averages[2].data(), &[1.0, 1.0, 1.0]);
    }
}
