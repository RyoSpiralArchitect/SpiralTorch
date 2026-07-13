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

use super::{PureResult, RewriteMonad, Tensor, TensorError};

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
    /// Total weight temperature \(\sum_u W_u + \gamma_S\) that scales the barycenter mode.
    pub effective_weight: f32,
    /// Intermediate densities describing the loss curve towards the barycenter.
    pub intermediates: Vec<BarycenterIntermediate>,
}

/// Spectral line emitted by the Tesla tail diagnostic.
#[derive(Debug, Clone, PartialEq)]
pub struct TeslaTailLine {
    /// Spectral frequency \(\omega_k\).
    pub frequency: f32,
    /// Weighted magnitude \(w_k r_k\).
    pub amplitude: f32,
    /// Line weight \(w_k\) recovered from the left eigenvectors.
    pub weight: f32,
}

/// Aggregated Tesla tail spectrum and coherence indicator.
#[derive(Debug, Clone, PartialEq)]
pub struct TeslaTail {
    /// Discrete spectral lines \(\mathcal{T}(\omega) = \sum_k w_k r_k \delta(\omega - \omega_k)\).
    pub lines: Vec<TeslaTailLine>,
    /// Coherence retention rate \(\mathrm{CR} = 1 - \kappa \sum_k w_k (1 - r_k)\).
    pub coherence_rate: f32,
}

const LOG_FLOOR: f32 = 1.0e-12;

fn checked_f64_to_f32(label: &'static str, value: f64) -> PureResult<f32> {
    if !value.is_finite() || value.abs() > f32::MAX as f64 {
        let value = if value.is_nan() {
            f32::NAN
        } else if value.is_sign_negative() {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
        return Err(TensorError::NonFiniteValue { label, value });
    }
    Ok(value as f32)
}

fn guard_probability_mass(label: &'static str, value: f32) -> PureResult<f32> {
    if !value.is_finite() {
        return Err(TensorError::NonFiniteValue { label, value });
    }
    Ok(value.max(LOG_FLOOR))
}

fn legacy_guard_probability_slice(label: &'static str, slice: &mut [f32]) -> PureResult<()> {
    if slice.is_empty() {
        return Err(TensorError::EmptyInput(label));
    }
    let mut candidate = Vec::with_capacity(slice.len());
    let mut sum = 0.0f64;
    for &value in slice.iter() {
        let guarded = guard_probability_mass(label, value)?;
        candidate.push(guarded);
        sum += guarded as f64;
    }
    if !sum.is_finite() || sum <= 0.0 {
        return Err(TensorError::NonFiniteValue {
            label,
            value: sum as f32,
        });
    }
    let mut normalised_sum = 0.0f64;
    for value in &mut candidate {
        *value = checked_f64_to_f32(label, *value as f64 / sum)?;
        normalised_sum += *value as f64;
    }
    let correction = 1.0 / normalised_sum;
    for value in &mut candidate {
        *value = checked_f64_to_f32(label, *value as f64 * correction)?;
    }
    slice.copy_from_slice(&candidate);
    Ok(())
}

fn guard_probability_slice_with(
    guard: Option<RewriteMonad<'_>>,
    label: &'static str,
    slice: &mut [f32],
) -> PureResult<()> {
    if let Some(monad) = guard {
        monad.guard_probability_slice(label, slice)
    } else {
        legacy_guard_probability_slice(label, slice)
    }
}

fn normalise_distribution(
    guard: Option<RewriteMonad<'_>>,
    tensor: &Tensor,
) -> PureResult<Vec<f32>> {
    let mut data = tensor.data().to_vec();
    guard_probability_slice_with(guard, "z_space_barycenter_density", &mut data)?;
    Ok(data)
}

fn kl_divergence(p: &[f32], q: &[f32]) -> PureResult<f32> {
    let mut acc = 0.0f64;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        if !pi.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "kl numerator",
                value: pi,
            });
        }
        if pi < 0.0 {
            return Err(TensorError::InvalidValue {
                label: "kl numerator",
            });
        }
        if pi == 0.0 {
            continue;
        }
        let qi = guard_probability_mass("kl denominator", qi)?;
        acc += pi as f64 * (pi as f64 / qi as f64).ln();
    }
    checked_f64_to_f32("kl_divergence", acc.max(0.0))
}

fn symmetric_kl(p: &[f32], q: &[f32]) -> PureResult<f32> {
    checked_f64_to_f32(
        "symmetric_kl",
        kl_divergence(p, q)? as f64 + kl_divergence(q, p)? as f64,
    )
}

fn entropy(dist: &[f32]) -> PureResult<f32> {
    let mut acc = 0.0f64;
    for &value in dist {
        if !value.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "entropy component",
                value,
            });
        }
        if value < 0.0 {
            return Err(TensorError::InvalidValue {
                label: "entropy component",
            });
        }
        if value == 0.0 {
            continue;
        }
        let value = value as f64;
        acc -= value * value.ln();
    }
    checked_f64_to_f32("entropy", acc.max(0.0))
}

fn barycenter_objective(
    candidate: &[f32],
    weights: &[f32],
    normalised: &[Vec<f32>],
    entropy_weight: f32,
    coupling_energy: f32,
) -> PureResult<(f32, f32, f32)> {
    let mut kl_energy = 0.0f64;
    for (weight, dist) in weights.iter().zip(normalised.iter()) {
        kl_energy += *weight as f64 * kl_divergence(candidate, dist)? as f64;
    }
    let entropy_value = entropy(candidate)?;
    let objective =
        kl_energy + coupling_energy as f64 - entropy_weight as f64 * entropy_value as f64;
    Ok((
        checked_f64_to_f32("barycenter_kl_energy", kl_energy)?,
        entropy_value,
        checked_f64_to_f32("barycenter_objective", objective)?,
    ))
}

fn weighted_baseline(
    guard: Option<RewriteMonad<'_>>,
    weights: &[f32],
    normalised: &[Vec<f32>],
    weight_sum: f64,
) -> PureResult<Vec<f32>> {
    let volume = normalised
        .first()
        .map(|dist| dist.len())
        .ok_or(TensorError::EmptyInput("z_space_barycenter"))?;
    if !weight_sum.is_finite() || weight_sum < 0.0 {
        return Err(TensorError::NonFiniteValue {
            label: "weight_sum",
            value: weight_sum as f32,
        });
    }
    if weight_sum == 0.0 {
        let uniform = checked_f64_to_f32("baseline component", 1.0 / volume as f64)?;
        let mut baseline = vec![uniform; volume];
        guard_probability_slice_with(guard, "z_space_barycenter_baseline", &mut baseline)?;
        return Ok(baseline);
    }
    let mut accumulator = vec![0.0f64; volume];
    for (weight, dist) in weights.iter().zip(normalised.iter()) {
        let ratio = *weight as f64 / weight_sum;
        for (slot, value) in accumulator.iter_mut().zip(dist.iter()) {
            *slot += ratio * guard_probability_mass("baseline component", *value)? as f64;
        }
    }
    let mut baseline = Vec::with_capacity(volume);
    for value in accumulator {
        baseline.push(guard_probability_mass(
            "baseline component",
            checked_f64_to_f32("baseline component", value)?,
        )?);
    }
    guard_probability_slice_with(guard, "z_space_barycenter_baseline", &mut baseline)?;
    Ok(baseline)
}

#[allow(clippy::too_many_arguments)]
fn barycenter_intermediates(
    guard: Option<RewriteMonad<'_>>,
    baseline: &[f32],
    bary: &[f32],
    weights: &[f32],
    normalised: &[Vec<f32>],
    entropy_weight: f32,
    coupling_energy: f32,
    rows: usize,
    cols: usize,
) -> PureResult<Vec<BarycenterIntermediate>> {
    let schedule = [0.0f32, 0.25, 0.5, 0.75, 1.0];
    let mut intermediates = Vec::with_capacity(schedule.len());
    for &alpha in &schedule {
        let mut mix = Vec::with_capacity(bary.len());
        for (&start, &target) in baseline.iter().zip(bary.iter()) {
            let value = (1.0 - alpha as f64) * start as f64 + alpha as f64 * target as f64;
            let value = checked_f64_to_f32("barycenter intermediate", value)?;
            mix.push(guard_probability_mass("barycenter intermediate", value)?);
        }
        guard_probability_slice_with(guard, "barycenter intermediate", &mut mix)?;
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
    guard: Option<RewriteMonad<'_>>,
    weights: &[f32],
    normalised: &[Vec<f32>],
    effective: f64,
) -> PureResult<Vec<f32>> {
    let volume = normalised
        .first()
        .map(|dist| dist.len())
        .ok_or(TensorError::EmptyInput("z_space_barycenter"))?;
    if !effective.is_finite() || effective <= 0.0 {
        return Err(TensorError::DegenerateBarycenter {
            effective_weight: effective as f32,
        });
    }
    let mut log_modes = vec![0.0f64; volume];
    for (idx, slot) in log_modes.iter_mut().enumerate() {
        let mut log_sum = 0.0f64;
        for (weight, dist) in weights.iter().zip(normalised.iter()) {
            let value = guard_probability_mass("barycenter component", dist[idx])? as f64;
            log_sum += *weight as f64 * value.ln();
        }
        *slot = log_sum / effective;
    }
    let max_log = log_modes.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max_log.is_finite() {
        return Err(TensorError::NonFiniteValue {
            label: "barycenter_log_mode",
            value: max_log as f32,
        });
    }
    let mut total = 0.0f64;
    let mut out = Vec::with_capacity(volume);
    for log_mode in log_modes {
        let value = (log_mode - max_log).exp();
        total += value;
        out.push(value);
    }
    if !total.is_finite() || total <= 0.0 {
        return Err(TensorError::NonFiniteValue {
            label: "barycenter mass",
            value: total as f32,
        });
    }
    let mut normalised_out = Vec::with_capacity(volume);
    for value in out {
        normalised_out.push(guard_probability_mass(
            "barycenter value",
            checked_f64_to_f32("barycenter value", value / total)?,
        )?);
    }
    guard_probability_slice_with(guard, "z_space_barycenter_mode", &mut normalised_out)?;
    Ok(normalised_out)
}

/// Solve the variational barycentre problem described in the Z-space note.
///
/// The solver works with discrete probability measures stored inside `Tensor`
/// rows.  Every supplied density is renormalised to guard against numerical
/// drift and entries are clipped at `1e-12` to keep the logarithms well-defined.
///
/// * `weights` --- non-negative coefficients \(W_u\); all-zero weights are valid
///   when the entropy term keeps the effective temperature positive
/// * `entropy_weight` --- finite entropy regulariser \(\gamma_S\)
/// * `beta_j` --- finite non-negative coupling scale \(\beta_J\)
/// * `coupling` --- optional non-negative finite matrix \(J_{uv}\)
fn z_space_barycenter_inner(
    guard: Option<RewriteMonad<'_>>,
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
    if !entropy_weight.is_finite() {
        return Err(TensorError::NonFiniteValue {
            label: "entropy_weight",
            value: entropy_weight,
        });
    }
    if !beta_j.is_finite() {
        return Err(TensorError::NonFiniteValue {
            label: "beta_j",
            value: beta_j,
        });
    }
    if beta_j < 0.0 {
        return Err(TensorError::InvalidValue { label: "beta_j" });
    }

    let mut weight_sum = 0.0f64;
    for &weight in weights {
        if !weight.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "barycenter weight",
                value: weight,
            });
        }
        if weight < 0.0 {
            return Err(TensorError::NonPositiveWeight { weight });
        }
        weight_sum += weight as f64;
    }
    if !weight_sum.is_finite() || weight_sum < 0.0 {
        return Err(TensorError::NonFiniteValue {
            label: "weight_sum",
            value: weight_sum as f32,
        });
    }

    let effective = weight_sum + entropy_weight as f64;
    if !effective.is_finite() {
        return Err(TensorError::NonFiniteValue {
            label: "barycenter_effective_weight",
            value: effective as f32,
        });
    }
    if effective <= 0.0 {
        return Err(TensorError::DegenerateBarycenter {
            effective_weight: effective as f32,
        });
    }
    let effective_weight = checked_f64_to_f32("barycenter_effective_weight", effective)?;

    let (rows, cols) = densities[0].shape();
    let mut normalised = Vec::with_capacity(densities.len());
    for density in densities {
        if density.shape() != (rows, cols) {
            return Err(TensorError::ShapeMismatch {
                left: (rows, cols),
                right: density.shape(),
            });
        }
        normalised.push(normalise_distribution(guard, density)?);
    }

    let baseline = weighted_baseline(guard, weights, &normalised, weight_sum)?;
    let bary = barycenter_mode(guard, weights, &normalised, effective)?;
    let bary_tensor = Tensor::from_vec(rows, cols, bary.clone())?;

    let coupling_energy = if let Some(coupling_tensor) = coupling {
        let (c_rows, c_cols) = coupling_tensor.shape();
        if (c_rows, c_cols) != (densities.len(), densities.len()) {
            return Err(TensorError::ShapeMismatch {
                left: (densities.len(), densities.len()),
                right: (c_rows, c_cols),
            });
        }
        let matrix = coupling_tensor.data();
        let mut acc = 0.0f64;
        for u in 0..densities.len() {
            for v in 0..densities.len() {
                let weight = matrix[u * densities.len() + v];
                if !weight.is_finite() {
                    return Err(TensorError::NonFiniteValue {
                        label: "barycenter_coupling",
                        value: weight,
                    });
                }
                if weight < 0.0 {
                    return Err(TensorError::InvalidValue {
                        label: "barycenter_coupling",
                    });
                }
                if weight == 0.0 {
                    continue;
                }
                acc += weight as f64 * symmetric_kl(&normalised[u], &normalised[v])? as f64;
            }
        }
        checked_f64_to_f32("barycenter_coupling_energy", 0.5 * beta_j as f64 * acc)?
    } else {
        0.0
    };

    let (kl_energy, entropy_value, objective) =
        barycenter_objective(&bary, weights, &normalised, entropy_weight, coupling_energy)?;
    let intermediates = barycenter_intermediates(
        guard,
        &baseline,
        &bary,
        weights,
        &normalised,
        entropy_weight,
        coupling_energy,
        rows,
        cols,
    )?;

    Ok(ZSpaceBarycenter {
        objective,
        density: bary_tensor,
        kl_energy,
        entropy: entropy_value,
        coupling_energy,
        effective_weight,
        intermediates,
    })
}

/// Solves the Z-space barycenter while projecting every probability slice
/// through the supplied open-topos rewrite contract.
pub fn z_space_barycenter_guarded(
    monad: RewriteMonad<'_>,
    weights: &[f32],
    densities: &[Tensor],
    entropy_weight: f32,
    beta_j: f32,
    coupling: Option<&Tensor>,
) -> PureResult<ZSpaceBarycenter> {
    z_space_barycenter_inner(
        Some(monad),
        weights,
        densities,
        entropy_weight,
        beta_j,
        coupling,
    )
}

/// Solves the Z-space barycenter with the standalone finite probability guard.
pub fn z_space_barycenter(
    weights: &[f32],
    densities: &[Tensor],
    entropy_weight: f32,
    beta_j: f32,
    coupling: Option<&Tensor>,
) -> PureResult<ZSpaceBarycenter> {
    z_space_barycenter_inner(None, weights, densities, entropy_weight, beta_j, coupling)
}

/// Builds the Tesla tail spectrum and coherence indicator from the dominant eigenpairs.
///
/// Radii and `kappa` must lie in `[0, 1]`. Input line weights are normalised to
/// unit mass before amplitudes and the coherence rate are evaluated.
pub fn tesla_tail_spectrum(
    radii: &[f32],
    frequencies: &[f32],
    weights: &[f32],
    kappa: f32,
) -> PureResult<TeslaTail> {
    if radii.is_empty() {
        return Err(TensorError::EmptyInput("tesla_tail_spectrum"));
    }
    if radii.len() != frequencies.len() {
        return Err(TensorError::DataLength {
            expected: radii.len(),
            got: frequencies.len(),
        });
    }
    if radii.len() != weights.len() {
        return Err(TensorError::DataLength {
            expected: radii.len(),
            got: weights.len(),
        });
    }
    if !kappa.is_finite() {
        return Err(TensorError::NonFiniteValue {
            label: "tesla_tail_kappa",
            value: kappa,
        });
    }
    if !(0.0..=1.0).contains(&kappa) {
        return Err(TensorError::InvalidValue {
            label: "tesla_tail_kappa",
        });
    }

    let mut weight_sum = 0.0f64;
    for idx in 0..radii.len() {
        let r = radii[idx];
        if !r.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "tesla_tail_radius",
                value: r,
            });
        }
        if !(0.0..=1.0).contains(&r) {
            return Err(TensorError::InvalidValue {
                label: "tesla_tail_radius",
            });
        }
        let omega = frequencies[idx];
        if !omega.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "tesla_tail_frequency",
                value: omega,
            });
        }
        let weight = weights[idx];
        if !weight.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "tesla_tail_weight",
                value: weight,
            });
        }
        if weight < 0.0 {
            return Err(TensorError::NonPositiveWeight { weight });
        }
        weight_sum += weight as f64;
    }
    if !weight_sum.is_finite() || weight_sum <= 0.0 {
        return Err(TensorError::DegenerateBarycenter {
            effective_weight: weight_sum as f32,
        });
    }

    let mut normalised_weights = Vec::with_capacity(weights.len());
    let mut normalised_sum = 0.0f64;
    for &weight in weights {
        let normalised = checked_f64_to_f32("tesla_tail_weight", weight as f64 / weight_sum)?;
        normalised_sum += normalised as f64;
        normalised_weights.push(normalised);
    }
    let correction = 1.0 / normalised_sum;
    let mut lines = Vec::with_capacity(radii.len());
    let mut penalty = 0.0f64;
    for idx in 0..radii.len() {
        let weight = checked_f64_to_f32(
            "tesla_tail_weight",
            normalised_weights[idx] as f64 * correction,
        )?;
        let amplitude =
            checked_f64_to_f32("tesla_tail_amplitude", weight as f64 * radii[idx] as f64)?;
        lines.push(TeslaTailLine {
            frequency: frequencies[idx],
            amplitude,
            weight,
        });
        penalty += weight as f64 * (1.0 - radii[idx] as f64);
    }

    let coherence_rate =
        checked_f64_to_f32("tesla_tail_coherence_rate", 1.0 - kappa as f64 * penalty)?;
    Ok(TeslaTail {
        lines,
        coherence_rate,
    })
}

/// Applies the NIRT weight update \(W_{t+1} = W_t + \eta\, \mathrm{Sim}(\rho^*, \tilde\rho)\, \mathrm{CR}\).
///
/// Similarities must lie in `[-1, 1]` and the coherence rate in `[0, 1]`.
/// The caller's weights are committed only after every candidate is finite and
/// the complete update can be normalised to unit mass.
pub fn nirt_weight_update(
    weights: &mut [f32],
    similarities: &[f32],
    coherence_rate: f32,
    eta: f32,
) -> PureResult<()> {
    if weights.is_empty() {
        return Err(TensorError::EmptyInput("nirt_weight_update_weights"));
    }
    if weights.len() != similarities.len() {
        return Err(TensorError::DataLength {
            expected: weights.len(),
            got: similarities.len(),
        });
    }
    if !eta.is_finite() || eta <= 0.0 {
        return Err(TensorError::NonPositiveLearningRate { rate: eta });
    }
    if !coherence_rate.is_finite() {
        return Err(TensorError::NonFiniteValue {
            label: "nirt_coherence_rate",
            value: coherence_rate,
        });
    }
    if !(0.0..=1.0).contains(&coherence_rate) {
        return Err(TensorError::InvalidValue {
            label: "nirt_coherence_rate",
        });
    }

    let mut candidate = Vec::with_capacity(weights.len());
    let mut total = 0.0f64;
    for (weight, similarity) in weights.iter().zip(similarities.iter()) {
        if !weight.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "nirt_weight",
                value: *weight,
            });
        }
        if *weight < 0.0 {
            return Err(TensorError::NonPositiveWeight { weight: *weight });
        }
        if !similarity.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "nirt_similarity",
                value: *similarity,
            });
        }
        if !(-1.0..=1.0).contains(similarity) {
            return Err(TensorError::InvalidValue {
                label: "nirt_similarity",
            });
        }
        let updated = *weight as f64 + eta as f64 * coherence_rate as f64 * *similarity as f64;
        let updated = updated.max(0.0);
        if !updated.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "nirt_updated_weight",
                value: updated as f32,
            });
        }
        candidate.push(updated);
        total += updated;
    }

    if !total.is_finite() || total <= 0.0 {
        return Err(TensorError::DegenerateBarycenter {
            effective_weight: total as f32,
        });
    }

    let mut normalised = Vec::with_capacity(candidate.len());
    let mut normalised_sum = 0.0f64;
    for weight in candidate {
        let weight = checked_f64_to_f32("nirt_weight", weight / total)?;
        normalised_sum += weight as f64;
        normalised.push(weight);
    }
    let correction = 1.0 / normalised_sum;
    for weight in &mut normalised {
        *weight = checked_f64_to_f32("nirt_weight", *weight as f64 * correction)?;
    }
    weights.copy_from_slice(&normalised);
    Ok(())
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
/// the supplied sequence grows along a Følner exhaustion. Action outputs are
/// accumulated in `f64` and must preserve the input shape and finite value domain.
pub fn conditional_expectation<G, A>(action: &A, elements: &[G], f: &Tensor) -> PureResult<Tensor>
where
    A: KoopmanAction<G>,
{
    if elements.is_empty() {
        return Err(TensorError::EmptyInput("conditional_expectation"));
    }
    let (rows, cols) = f.shape();
    for &value in f.data() {
        if !value.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "conditional_expectation_input",
                value,
            });
        }
    }
    let mut accumulator = vec![0.0f64; f.data().len()];
    for element in elements {
        let transformed = action.apply(element, f)?;
        if transformed.shape() != (rows, cols) {
            return Err(TensorError::ShapeMismatch {
                left: (rows, cols),
                right: transformed.shape(),
            });
        }
        for (slot, &value) in accumulator.iter_mut().zip(transformed.data()) {
            if !value.is_finite() {
                return Err(TensorError::NonFiniteValue {
                    label: "conditional_expectation_action",
                    value,
                });
            }
            *slot += value as f64;
        }
    }
    let denominator = elements.len() as f64;
    let mut averaged = Vec::with_capacity(accumulator.len());
    for value in accumulator {
        averaged.push(checked_f64_to_f32(
            "conditional_expectation_output",
            value / denominator,
        )?);
    }
    Tensor::from_vec(rows, cols, averaged)
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
    for subset in sequence {
        let subset_ref = subset.as_ref();
        if subset_ref.is_empty() {
            return Err(TensorError::EmptyInput("cesaro_averages"));
        }
        let projection = conditional_expectation(action, subset_ref, f)?;
        averages.push(projection);
    }
    if averages.is_empty() {
        return Err(TensorError::EmptyInput("cesaro_averages"));
    }
    Ok(averages)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pure::topos::OpenCartesianTopos;

    #[track_caller]
    fn unwrap_ok<T, E: core::fmt::Debug>(result: Result<T, E>) -> T {
        match result {
            Ok(value) => value,
            Err(error) => panic!("expected Ok(..), got Err({error:?})"),
        }
    }

    #[track_caller]
    fn unwrap_err<T, E: core::fmt::Debug>(result: Result<T, E>) -> E {
        match result {
            Ok(_) => panic!("expected Err(..), got Ok(..)"),
            Err(error) => error,
        }
    }

    #[track_caller]
    fn unwrap_some<T>(option: Option<T>) -> T {
        match option {
            Some(value) => value,
            None => panic!("expected Some(..), got None"),
        }
    }

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
                        let dest = (x + element) % width;
                        out_row[c * width + dest] = row[c * width + x];
                    }
                }
            }
            Ok(out)
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct ExtremeAction;

    impl KoopmanAction<bool> for ExtremeAction {
        fn apply(&self, element: &bool, _input: &Tensor) -> PureResult<Tensor> {
            Tensor::from_vec(1, 1, vec![if *element { f32::MAX } else { -f32::MAX }])
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct MalformedAction;

    impl KoopmanAction<usize> for MalformedAction {
        fn apply(&self, element: &usize, _input: &Tensor) -> PureResult<Tensor> {
            if *element == 0 {
                Tensor::from_vec(1, 2, vec![0.0, 0.0])
            } else {
                Tensor::from_vec(1, 1, vec![f32::NAN])
            }
        }
    }

    #[test]
    fn projection_recovers_invariants() {
        let action = CyclicShift { width: 3 };
        let input = unwrap_ok(Tensor::from_vec(1, 3, vec![1.0, 2.0, 3.0]));
        let elements = vec![0usize, 1, 2];
        let projected = unwrap_ok(conditional_expectation(&action, &elements, &input));
        assert_eq!(projected.data(), &[2.0, 2.0, 2.0]);
    }

    #[test]
    fn cesaro_sequence_converges() {
        let action = CyclicShift { width: 3 };
        let input = unwrap_ok(Tensor::from_vec(1, 3, vec![3.0, 0.0, 0.0]));
        let sequence = vec![vec![0usize], vec![0usize, 1], vec![0usize, 1, 2]];
        let averages = unwrap_ok(cesaro_averages(&action, sequence, &input));
        assert_eq!(averages.len(), 3);
        assert_eq!(averages[0].data(), &[3.0, 0.0, 0.0]);
        assert_eq!(averages[1].data(), &[1.5, 1.5, 0.0]);
        assert_eq!(averages[2].data(), &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn conditional_expectation_uses_finite_f64_accumulation() {
        let input = unwrap_ok(Tensor::from_vec(1, 1, vec![0.0]));
        let projected = unwrap_ok(conditional_expectation(
            &ExtremeAction,
            &[true, false],
            &input,
        ));
        assert_eq!(projected.data(), &[0.0]);
    }

    #[test]
    fn conditional_expectation_rejects_malformed_actions() {
        let input = unwrap_ok(Tensor::from_vec(1, 1, vec![0.0]));
        let error = unwrap_err(conditional_expectation(&MalformedAction, &[0], &input));
        assert!(matches!(error, TensorError::ShapeMismatch { .. }));

        let error = unwrap_err(conditional_expectation(&MalformedAction, &[1], &input));
        assert!(matches!(
            error,
            TensorError::NonFiniteValue {
                label: "conditional_expectation_action",
                ..
            }
        ));
        let error = unwrap_err(cesaro_averages::<usize, _, Vec<Vec<usize>>>(
            &MalformedAction,
            Vec::new(),
            &input,
        ));
        assert!(matches!(error, TensorError::EmptyInput("cesaro_averages")));
    }

    #[test]
    fn barycenter_respects_symmetry() {
        let densities = vec![
            unwrap_ok(Tensor::from_vec(1, 2, vec![0.8, 0.2])),
            unwrap_ok(Tensor::from_vec(1, 2, vec![0.2, 0.8])),
        ];
        let weights = vec![1.0, 1.0];
        let result = unwrap_ok(z_space_barycenter(&weights, &densities, 0.25, 0.0, None));
        let data = result.density.data();
        assert!((data[0] - data[1]).abs() < 1e-6);
        assert!(result.kl_energy > 0.0);
        assert!(result.entropy > 0.0);
        assert_eq!(result.coupling_energy, 0.0);
        assert!(result.objective <= result.kl_energy);
    }

    #[test]
    fn information_energies_respect_zero_support_limits() {
        let kl = unwrap_ok(kl_divergence(&[1.0, 0.0], &[0.5, 0.5]));
        assert!((kl - core::f32::consts::LN_2).abs() < 1e-6);
        assert_eq!(unwrap_ok(entropy(&[1.0, 0.0])), 0.0);
    }

    #[test]
    fn barycenter_degeneracy_detected() {
        let densities = vec![unwrap_ok(Tensor::from_vec(1, 2, vec![0.6, 0.4]))];
        let weights = vec![1.0];
        let err = unwrap_err(z_space_barycenter(&weights, &densities, -1.5, 0.0, None));
        assert!(matches!(
            err,
            TensorError::DegenerateBarycenter { effective_weight } if effective_weight <= 0.0
        ));
    }

    #[test]
    fn barycenter_rejects_invalid_parameters_and_coupling() {
        let densities = vec![unwrap_ok(Tensor::from_vec(1, 2, vec![0.6, 0.4]))];
        let weights = vec![1.0];
        let error = unwrap_err(z_space_barycenter(
            &weights,
            &densities,
            f32::INFINITY,
            0.0,
            None,
        ));
        assert!(matches!(
            error,
            TensorError::NonFiniteValue {
                label: "entropy_weight",
                ..
            }
        ));
        let error = unwrap_err(z_space_barycenter(
            &weights,
            &densities,
            0.0,
            f32::INFINITY,
            None,
        ));
        assert!(matches!(
            error,
            TensorError::NonFiniteValue {
                label: "beta_j",
                ..
            }
        ));
        let error = unwrap_err(z_space_barycenter(&weights, &densities, 0.0, -0.5, None));
        assert!(matches!(
            error,
            TensorError::InvalidValue { label: "beta_j" }
        ));

        let coupling = unwrap_ok(Tensor::from_vec(1, 1, vec![f32::NAN]));
        let error = unwrap_err(z_space_barycenter(
            &weights,
            &densities,
            0.0,
            1.0,
            Some(&coupling),
        ));
        assert!(matches!(
            error,
            TensorError::NonFiniteValue {
                label: "barycenter_coupling",
                ..
            }
        ));
        let coupling = unwrap_ok(Tensor::from_vec(1, 1, vec![-1.0]));
        let error = unwrap_err(z_space_barycenter(
            &weights,
            &densities,
            0.0,
            1.0,
            Some(&coupling),
        ));
        assert!(matches!(
            error,
            TensorError::InvalidValue {
                label: "barycenter_coupling"
            }
        ));
    }

    #[test]
    fn barycenter_normalises_extreme_finite_densities_and_weights() {
        let densities = vec![
            unwrap_ok(Tensor::from_vec(1, 2, vec![f32::MAX, f32::MAX])),
            unwrap_ok(Tensor::from_vec(
                1,
                2,
                vec![f32::MIN_POSITIVE, f32::MIN_POSITIVE],
            )),
        ];
        let weight = f32::MAX * 0.25;
        let result = unwrap_ok(z_space_barycenter(
            &[weight, weight],
            &densities,
            0.0,
            0.0,
            None,
        ));
        assert!(result.density.data().iter().all(|value| value.is_finite()));
        assert!((result.density.data()[0] - 0.5).abs() < 1e-6);
        assert!((result.density.data()[1] - 0.5).abs() < 1e-6);
        assert!(result.objective.is_finite());
        assert!(result.effective_weight.is_finite());
    }

    #[test]
    fn barycenter_mode_remains_stable_near_degenerate_temperature() {
        let densities = vec![unwrap_ok(Tensor::from_vec(1, 2, vec![0.9, 0.1]))];
        let result = unwrap_ok(z_space_barycenter(&[1.0], &densities, -0.999, 0.0, None));
        assert!(result.density.data()[0] > 0.999);
        assert!(result.density.data()[1] >= 0.0);
        assert!((result.density.data().iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn entropy_only_barycenter_is_uniform() {
        let densities = vec![
            unwrap_ok(Tensor::from_vec(1, 3, vec![0.98, 0.01, 0.01])),
            unwrap_ok(Tensor::from_vec(1, 3, vec![0.1, 0.2, 0.7])),
        ];
        let result = unwrap_ok(z_space_barycenter(&[0.0, 0.0], &densities, 0.5, 0.0, None));
        for value in result.density.data() {
            assert!((*value - 1.0 / 3.0).abs() < 1e-6);
        }
        assert_eq!(result.kl_energy, 0.0);
        assert!(result.entropy > 0.0);
        assert!(result.objective < 0.0);
    }

    #[test]
    fn barycenter_rejects_unrepresentable_effective_weight() {
        let densities = vec![
            unwrap_ok(Tensor::from_vec(1, 1, vec![1.0])),
            unwrap_ok(Tensor::from_vec(1, 1, vec![1.0])),
        ];
        let error = unwrap_err(z_space_barycenter(
            &[f32::MAX, f32::MAX],
            &densities,
            0.0,
            0.0,
            None,
        ));
        assert!(matches!(
            error,
            TensorError::NonFiniteValue {
                label: "barycenter_effective_weight",
                ..
            }
        ));
    }

    #[test]
    fn barycenter_coupling_energy_matches() {
        let densities = vec![
            unwrap_ok(Tensor::from_vec(1, 2, vec![0.9, 0.1])),
            unwrap_ok(Tensor::from_vec(1, 2, vec![0.4, 0.6])),
        ];
        let weights = vec![2.0, 1.0];
        let coupling = unwrap_ok(Tensor::from_vec(2, 2, vec![0.0, 1.0, 1.0, 0.0]));
        let result = unwrap_ok(z_space_barycenter(
            &weights,
            &densities,
            0.1,
            2.0,
            Some(&coupling),
        ));
        assert!(result.coupling_energy > 0.0);
        assert!(result.objective < result.kl_energy + result.coupling_energy);
        assert!(!result.intermediates.is_empty());
    }

    #[test]
    fn barycenter_loss_curve_descends() {
        let densities = vec![
            unwrap_ok(Tensor::from_vec(1, 3, vec![0.7, 0.2, 0.1])),
            unwrap_ok(Tensor::from_vec(1, 3, vec![0.1, 0.4, 0.5])),
        ];
        let weights = vec![1.5, 0.5];
        let result = unwrap_ok(z_space_barycenter(&weights, &densities, 0.2, 0.0, None));
        assert!(result.intermediates.len() >= 2);
        let objectives: Vec<f32> = result
            .intermediates
            .iter()
            .map(|stage| stage.objective)
            .collect();
        for window in objectives.windows(2) {
            assert!(window[1] <= window[0] + 1e-5);
        }
        let last = unwrap_some(result.intermediates.last());
        assert!((last.objective - result.objective).abs() < 1e-4);
        assert!((last.interpolation - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn guarded_barycenter_projects_through_topos() {
        let topos = unwrap_ok(OpenCartesianTopos::new(-1.0, 1e-5, 10.0, 64, 4096));
        let weights = vec![1.0, 2.0];
        let densities = vec![
            unwrap_ok(Tensor::from_vec(1, 3, vec![0.2, f32::NAN, 0.6])),
            unwrap_ok(Tensor::from_vec(1, 3, vec![-5.0, 0.3, f32::INFINITY])),
        ];
        let result = unwrap_ok(z_space_barycenter_guarded(
            RewriteMonad::new(&topos),
            &weights,
            &densities,
            0.05,
            0.0,
            None,
        ));
        let data = result.density.data();
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(data.iter().all(|value| value.is_finite() && *value >= 0.0));
    }

    #[test]
    fn tesla_tail_reports_spectrum_and_coherence() {
        let radii = [0.9, 0.75, 0.5];
        let freqs = [1.0, 2.0, 3.5];
        let weights = [0.4, 0.35, 0.25];
        let tail = unwrap_ok(tesla_tail_spectrum(&radii, &freqs, &weights, 0.6));
        assert_eq!(tail.lines.len(), radii.len());
        assert!(tail.lines.iter().all(|line| line.amplitude >= 0.0));
        let expected = 1.0
            - 0.6
                * weights
                    .iter()
                    .zip(radii.iter())
                    .map(|(w, r)| *w * (1.0 - *r))
                    .sum::<f32>();
        assert!((tail.coherence_rate - expected).abs() < 1e-6);
    }

    #[test]
    fn tesla_tail_normalises_weights_and_enforces_rate_domain() {
        let tail = unwrap_ok(tesla_tail_spectrum(
            &[1.0, 0.5],
            &[1.0, 2.0],
            &[3.0, 1.0],
            1.0,
        ));
        assert!((tail.lines[0].weight - 0.75).abs() < 1e-6);
        assert!((tail.lines[1].weight - 0.25).abs() < 1e-6);
        assert!((tail.coherence_rate - 0.875).abs() < 1e-6);

        let error = unwrap_err(tesla_tail_spectrum(&[1.1], &[1.0], &[1.0], 0.5));
        assert!(matches!(
            error,
            TensorError::InvalidValue {
                label: "tesla_tail_radius"
            }
        ));
        let error = unwrap_err(tesla_tail_spectrum(&[1.0], &[1.0], &[1.0], 1.1));
        assert!(matches!(
            error,
            TensorError::InvalidValue {
                label: "tesla_tail_kappa"
            }
        ));
        let error = unwrap_err(tesla_tail_spectrum(&[1.0], &[1.0], &[0.0], 0.5));
        assert!(matches!(error, TensorError::DegenerateBarycenter { .. }));
    }

    #[test]
    fn nirt_weight_update_shifts_mass_towards_similar_modes() {
        let mut weights = vec![0.3, 0.4, 0.3];
        let similarities = vec![0.9, 0.1, 0.4];
        unwrap_ok(nirt_weight_update(&mut weights, &similarities, 0.8, 0.2));
        let sum = weights.iter().sum::<f32>();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(weights[0] > weights[1]);
        assert!(weights[0] > 0.3);
    }

    #[test]
    fn nirt_weight_update_rolls_back_late_failures() {
        let mut weights = vec![0.4, 0.6];
        let original = weights.clone();
        let error = unwrap_err(nirt_weight_update(&mut weights, &[0.5, f32::NAN], 0.8, 0.2));
        assert!(matches!(
            error,
            TensorError::NonFiniteValue {
                label: "nirt_similarity",
                ..
            }
        ));
        assert_eq!(weights, original);

        let error = unwrap_err(nirt_weight_update(&mut weights, &[0.5, 1.1], 0.8, 0.2));
        assert!(matches!(
            error,
            TensorError::InvalidValue {
                label: "nirt_similarity"
            }
        ));
        assert_eq!(weights, original);
    }

    #[test]
    fn nirt_weight_update_rolls_back_degenerate_candidates() {
        let mut weights = vec![0.0, 0.0];
        let original = weights.clone();
        let error = unwrap_err(nirt_weight_update(&mut weights, &[-1.0, -1.0], 1.0, 1.0));
        assert!(matches!(error, TensorError::DegenerateBarycenter { .. }));
        assert_eq!(weights, original);
    }
}
