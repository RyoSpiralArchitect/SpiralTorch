// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

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

/// Intermediate densities emitted while interpolating the barycenter objective.
#[derive(Debug, Clone)]
pub struct BarycenterIntermediate {
    /// Interpolation factor between the arithmetic baseline and the barycenter density.
    pub interpolation: f32,
    /// Interpolated density at the current interpolation step.
    pub density: Tensor,
    /// KL energy of the interpolated density with respect to the input charts.
    pub kl_energy: f32,
    /// Shannon entropy of the interpolated density.
    pub entropy: f32,
    /// Full barycenter objective for the interpolated density.
    pub objective: f32,
}

/// Result of the Z-space barycenter solver.  Besides the density itself we surface
/// individual energy contributions so callers can inspect how the KL attraction,
/// entropy regulariser, and inter-chart coupling interact.
#[derive(Debug, Clone)]
pub struct ZSpaceBarycenter {
    /// Minimiser of the weighted KL + entropy functional.
    pub density: Tensor,
    /// \(\sum_u W_u \mathrm{KL}(\rho^* \| \hat\rho_u)\).
    pub kl_energy: f32,
    /// Shannon entropy \(S(\rho^*)\).
    pub entropy: f32,
    /// Coupling penalty \(\tfrac{\beta_J}{2} \sum_{u,v} J_{uv} \mathcal{D}(\hat\rho_u, \hat\rho_v)\).
    pub coupling_energy: f32,
    /// Total objective value at the minimiser.
    pub objective: f32,
    /// Aggregated KL weight after subtracting the entropy regulariser.
    pub effective_weight: f32,
    /// Intermediate densities describing the loss curve towards the barycenter.
    pub intermediates: Vec<BarycenterIntermediate>,
}

const LOG_FLOOR: f32 = 1.0e-12;

fn guard_probability_mass(label: &'static str, value: f32) -> PureResult<f32> {
    if !value.is_finite() {
        return Err(TensorError::NonFiniteValue { label, value });
    }
    Ok(value.max(LOG_FLOOR))
}

fn normalise_distribution(tensor: &Tensor) -> PureResult<Vec<f32>> {
    let mut data = tensor.data().to_vec();
    let mut sum = 0.0f32;
    for value in data.iter_mut() {
        *value = guard_probability_mass("density entry", *value)?;
        sum += *value;
    }
    if sum <= 0.0 {
        return Err(TensorError::NonFiniteValue {
            label: "density mass",
            value: sum,
        });
    }
    for value in data.iter_mut() {
        *value /= sum;
    }
    Ok(data)
}

fn kl_divergence(p: &[f32], q: &[f32]) -> PureResult<f32> {
    let mut acc = 0.0f32;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        let qi = guard_probability_mass("kl denominator", qi)?;
        acc += pi * (pi / qi).ln();
    }
    Ok(acc)
}

fn symmetric_kl(p: &[f32], q: &[f32]) -> PureResult<f32> {
    Ok(kl_divergence(p, q)? + kl_divergence(q, p)?)
}

fn entropy(dist: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for &value in dist {
        let value = value.max(LOG_FLOOR);
        acc -= value * value.ln();
    }
    acc
}

fn barycenter_objective(
    candidate: &[f32],
    weights: &[f32],
    normalised: &[Vec<f32>],
    entropy_weight: f32,
    coupling_energy: f32,
) -> PureResult<(f32, f32, f32)> {
    let mut kl_acc = 0.0f32;
    for (weight, dist) in weights.iter().zip(normalised.iter()) {
        kl_acc += *weight * kl_divergence(candidate, dist)?;
    }
    let entropy_value = entropy(candidate);
    Ok((
        kl_acc,
        entropy_value,
        kl_acc + entropy_weight * entropy_value + coupling_energy,
    ))
}

fn barycenter_intermediates(
    weights: &[f32],
    normalised: &[Vec<f32>],
    weight_sum: f32,
    bary: &[f32],
    entropy_weight: f32,
    coupling_energy: f32,
    rows: usize,
    cols: usize,
) -> PureResult<Vec<BarycenterIntermediate>> {
    let mut baseline = vec![0.0f32; bary.len()];
    for (weight, dist) in weights.iter().zip(normalised.iter()) {
        for (slot, value) in baseline.iter_mut().zip(dist.iter()) {
            *slot += *weight * *value;
        }
    }
    for value in baseline.iter_mut() {
        *value /= weight_sum;
    }

    let schedule = [0.0f32, 0.25, 0.5, 0.75, 1.0];
    let mut intermediates = Vec::with_capacity(schedule.len());
    for &alpha in &schedule {
        let mut mix = Vec::with_capacity(bary.len());
        for (&start, &target) in baseline.iter().zip(bary.iter()) {
            let value = (1.0 - alpha) * start + alpha * target;
            mix.push(guard_probability_mass("barycenter intermediate", value)?);
        }
        let mut total = 0.0f32;
        for value in mix.iter_mut() {
            total += *value;
        }
        if total <= 0.0 {
            return Err(TensorError::NonFiniteValue {
                label: "barycenter intermediate mass",
                value: total,
            });
        }
        for value in mix.iter_mut() {
            *value /= total;
        }
        let (kl_energy, entropy_value, objective) =
            barycenter_objective(&mix, weights, normalised, entropy_weight, coupling_energy)?;
        intermediates.push(BarycenterIntermediate {
            interpolation: alpha,
            density: Tensor::from_vec(rows, cols, mix.clone())?,
            kl_energy,
            entropy: entropy_value,
            objective,
        });
    }
    Ok(intermediates)
}

fn barycenter_mode(
    weights: &[f32],
    normalised: &[Vec<f32>],
    effective: f32,
) -> PureResult<Vec<f32>> {
    let volume = normalised
        .get(0)
        .map(|dist| dist.len())
        .ok_or(TensorError::EmptyInput("z_space_barycenter"))?;
    let mut out = vec![0.0f32; volume];
    for (idx, slot) in out.iter_mut().enumerate() {
        let mut log_sum = 0.0f32;
        for (weight, dist) in weights.iter().zip(normalised.iter()) {
            let value = guard_probability_mass("barycenter component", dist[idx])?;
            log_sum += weight * value.ln();
        }
        *slot = (log_sum / effective).exp();
    }
    let mut total = 0.0f32;
    for value in out.iter_mut() {
        if !value.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "barycenter value",
                value: *value,
            });
        }
        *value = guard_probability_mass("barycenter value", *value)?;
        total += *value;
    }
    if total <= 0.0 {
        return Err(TensorError::NonFiniteValue {
            label: "barycenter mass",
            value: total,
        });
    }
    for value in out.iter_mut() {
        *value /= total;
    }
    Ok(out)
}

/// Solve the variational barycentre problem described in the Z-space note.
///
/// The solver works with discrete probability measures stored inside `Tensor`
/// rows.  Every supplied density is renormalised to guard against numerical
/// drift and entries are clipped at `1e-12` to keep the logarithms well-defined.
///
/// * `weights` --- non-negative coefficients \(W_u\)
/// * `entropy_weight` --- entropy regulariser \(\gamma_S\)
/// * `beta_j` --- coupling scale \(\beta_J\)
/// * `coupling` --- optional matrix \(J_{uv}\)
pub fn z_space_barycenter(
    weights: &[f32],
    densities: &[Tensor],
    entropy_weight: f32,
    beta_j: f32,
    coupling: Option<&Tensor>,
) -> PureResult<ZSpaceBarycenter> {
    if densities.is_empty() {
        return Err(TensorError::EmptyInput("z_space_barycenter"));
    }
    if weights.len() != densities.len() {
        return Err(TensorError::DataLength {
            expected: densities.len(),
            got: weights.len(),
        });
    }
    if entropy_weight.is_nan() {
        return Err(TensorError::NonFiniteValue {
            label: "entropy_weight",
            value: entropy_weight,
        });
    }
    if beta_j.is_nan() {
        return Err(TensorError::NonFiniteValue {
            label: "beta_j",
            value: beta_j,
        });
    }

    let mut weight_sum = 0.0f32;
    for &weight in weights {
        if weight < 0.0 || !weight.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "barycenter weight",
                value: weight,
            });
        }
        weight_sum += weight;
    }
    if weight_sum <= 0.0 {
        return Err(TensorError::NonFiniteValue {
            label: "weight_sum",
            value: weight_sum,
        });
    }

    let effective_weight = weight_sum - entropy_weight;
    if effective_weight <= 0.0 {
        return Err(TensorError::DegenerateBarycenter { effective_weight });
    }

    let (rows, cols) = densities[0].shape();
    let mut normalised = Vec::with_capacity(densities.len());
    for density in densities {
        if density.shape() != (rows, cols) {
            return Err(TensorError::ShapeMismatch {
                left: (rows, cols),
                right: density.shape(),
            });
        }
        normalised.push(normalise_distribution(density)?);
    }

    let bary = barycenter_mode(weights, &normalised, effective_weight)?;
    let bary_tensor = Tensor::from_vec(rows, cols, bary.clone())?;
    let kl_energy = {
        let mut acc = 0.0f32;
        for (weight, dist) in weights.iter().zip(normalised.iter()) {
            acc += *weight * kl_divergence(&bary, dist)?;
        }
        acc
    };
    let entropy_value = entropy(&bary);

    let coupling_energy = if let Some(coupling_tensor) = coupling {
        let (c_rows, c_cols) = coupling_tensor.shape();
        if (c_rows, c_cols) != (densities.len(), densities.len()) {
            return Err(TensorError::ShapeMismatch {
                left: (densities.len(), densities.len()),
                right: (c_rows, c_cols),
            });
        }
        let matrix = coupling_tensor.data();
        let mut acc = 0.0f32;
        for u in 0..densities.len() {
            for v in 0..densities.len() {
                let weight = matrix[u * densities.len() + v];
                if weight == 0.0 {
                    continue;
                }
                acc += weight * symmetric_kl(&normalised[u], &normalised[v])?;
            }
        }
        0.5 * beta_j * acc
    } else {
        0.0
    };

    let intermediates = barycenter_intermediates(
        weights,
        &normalised,
        weight_sum,
        &bary,
        entropy_weight,
        coupling_energy,
        rows,
        cols,
    )?;

    Ok(ZSpaceBarycenter {
        objective: kl_energy + entropy_weight * entropy_value + coupling_energy,
        density: bary_tensor,
        kl_energy,
        entropy: entropy_value,
        coupling_energy,
        effective_weight,
        intermediates,
    })
}

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

    #[test]
    fn barycenter_respects_symmetry() {
        let densities = vec![
            Tensor::from_vec(1, 2, vec![0.8, 0.2]).unwrap(),
            Tensor::from_vec(1, 2, vec![0.2, 0.8]).unwrap(),
        ];
        let weights = vec![1.0, 1.0];
        let result = z_space_barycenter(&weights, &densities, 0.25, 0.0, None).unwrap();
        let data = result.density.data();
        assert!((data[0] - data[1]).abs() < 1e-6);
        assert!((result.kl_energy - 0.0).abs() < 1e-6);
        assert!(result.entropy > 0.0);
        assert_eq!(result.coupling_energy, 0.0);
    }

    #[test]
    fn barycenter_degeneracy_detected() {
        let densities = vec![Tensor::from_vec(1, 2, vec![0.6, 0.4]).unwrap()];
        let weights = vec![1.0];
        let err = z_space_barycenter(&weights, &densities, 1.0, 0.0, None).unwrap_err();
        assert!(matches!(
            err,
            TensorError::DegenerateBarycenter { effective_weight } if effective_weight <= 0.0
        ));
    }

    #[test]
    fn barycenter_coupling_energy_matches() {
        let densities = vec![
            Tensor::from_vec(1, 2, vec![0.9, 0.1]).unwrap(),
            Tensor::from_vec(1, 2, vec![0.4, 0.6]).unwrap(),
        ];
        let weights = vec![2.0, 1.0];
        let coupling = Tensor::from_vec(2, 2, vec![0.0, 1.0, 1.0, 0.0]).unwrap();
        let result = z_space_barycenter(&weights, &densities, 0.1, 2.0, Some(&coupling)).unwrap();
        assert!(result.coupling_energy > 0.0);
        assert!(result.objective > result.kl_energy);
        assert!(!result.intermediates.is_empty());
    }

    #[test]
    fn barycenter_loss_curve_descends() {
        let densities = vec![
            Tensor::from_vec(1, 3, vec![0.7, 0.2, 0.1]).unwrap(),
            Tensor::from_vec(1, 3, vec![0.1, 0.4, 0.5]).unwrap(),
        ];
        let weights = vec![1.5, 0.5];
        let result = z_space_barycenter(&weights, &densities, 0.2, 0.0, None).unwrap();
        assert!(result.intermediates.len() >= 2);
        let objectives: Vec<f32> = result
            .intermediates
            .iter()
            .map(|stage| stage.objective)
            .collect();
        for window in objectives.windows(2) {
            assert!(window[1] <= window[0] + 1e-5);
        }
        let last = result.intermediates.last().unwrap();
        assert!((last.objective - result.objective).abs() < 1e-4);
        assert!((last.interpolation - 1.0).abs() < f32::EPSILON);
    }
}
