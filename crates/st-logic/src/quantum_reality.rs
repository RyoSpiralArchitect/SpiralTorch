// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! A strongly typed transcription of the "Quantum Reality Engine ∞" LaTeX
//! specification into idiomatic Rust data structures and helper routines.
//!
//! The module models the layered superconducting hardware, qubit state
//! trajectories, bloom signatures, interference transport, and metaphoric
//! Z-space/Monad-biome couplings described in the document.  Each mathematical
//! relation is expressed through explicit functions so it can participate in
//! downstream simulations without needing to parse the original TeX.

use std::f64::consts::PI;
use std::iter;
use std::sync::Arc;

use num_complex::Complex64;

/// Convenience alias for scalar-valued time functions.
pub type TimeFn = Arc<dyn Fn(f64) -> f64 + Send + Sync + 'static>;

/// Soft Gaussian kernel used as the "soft delta" in the document.
fn soft_delta(x: f64, sigma: f64) -> f64 {
    let sigma = sigma.max(1e-12);
    let norm = (2.0 * PI).sqrt() * sigma;
    let expo = -x * x / (2.0 * sigma * sigma);
    expo.exp() / norm
}

/// Specification of the superconducting hardware stack.
#[derive(Clone, Debug)]
pub struct HardwareStack {
    pub qubit_layer: QubitLayer,
    pub junction: JosephsonJunction,
    pub on_chip_lines: OnChipLines,
    pub ground_plane: GroundPlane,
    pub cryo_enclosure: CryoEnclosure,
}

#[derive(Clone, Debug)]
pub struct QubitLayer {
    pub material: &'static str,
    pub geometry: &'static str,
}

#[derive(Clone, Debug)]
pub struct JosephsonJunction {
    pub barrier_material: &'static str,
    pub barrier_thickness_nm: f64,
}

#[derive(Clone, Debug)]
pub struct OnChipLines {
    pub topology: &'static str,
    pub center_width_microns: f64,
    pub impedance_ohms: f64,
}

#[derive(Clone, Debug)]
pub struct GroundPlane {
    pub material: &'static str,
    pub shielded: bool,
}

#[derive(Clone, Debug)]
pub struct CryoEnclosure {
    pub temperature_mk: f64,
    pub shields: &'static str,
}

/// Physical constants for the sapphire substrate.
#[derive(Clone, Debug)]
pub struct SubstratePhysicalModel {
    pub lattice_a_angstrom: f64,
    pub lattice_c_angstrom: f64,
    pub die_dims_mm: [f64; 3],
    pub epsilon_r: f64,
    pub thermal_conductivity_w_mk: f64,
    pub cte_per_k: f64,
}

/// Microwave control and decoherence parameters.
#[derive(Clone, Debug)]
pub struct SimulationParameters {
    pub qubit_frequency_ghz: f64,
    pub anharmonicity_mhz: f64,
    pub coupling_mhz: f64,
    pub t1_microseconds: f64,
    pub t2_microseconds: f64,
    pub pi_pulse_ns: f64,
}

/// Cluster definition following the taxicab embedding.
#[derive(Clone, Debug)]
pub struct ClusterStructure {
    segments: Vec<usize>,
}

impl ClusterStructure {
    pub fn new(segments: Vec<usize>) -> Self {
        assert!(!segments.is_empty(), "cluster structure requires segments");
        ClusterStructure { segments }
    }

    pub fn total_qubits(&self) -> usize {
        self.segments.iter().copied().sum()
    }

    pub fn cluster_bounds(&self, k: usize) -> Option<(usize, usize)> {
        let mut start = 0usize;
        for (idx, len) in self.segments.iter().copied().enumerate() {
            let end = start + len;
            if idx == k {
                return Some((start, end));
            }
            start = end;
        }
        None
    }

    pub fn cluster_indices(&self, k: usize) -> impl Iterator<Item = usize> {
        self.cluster_bounds(k)
            .map(|(s, e)| s..e)
            .into_iter()
            .flatten()
    }

    pub fn cluster_of(&self, n: usize) -> Option<usize> {
        let mut start = 0usize;
        for (idx, len) in self.segments.iter().copied().enumerate() {
            let end = start + len;
            if n >= start && n < end {
                return Some(idx);
            }
            start = end;
        }
        None
    }
}

/// Time-parametrised angles for a single qubit.
#[derive(Clone)]
pub struct QubitTrajectory {
    pub theta: TimeFn,
    pub phi: TimeFn,
    pub theta_dot: TimeFn,
    pub phi_dot: TimeFn,
}

impl QubitTrajectory {
    pub fn theta(&self, t: f64) -> f64 {
        (self.theta)(t)
    }

    pub fn phi(&self, t: f64) -> f64 {
        (self.phi)(t)
    }

    pub fn theta_dot(&self, t: f64) -> f64 {
        (self.theta_dot)(t)
    }

    pub fn phi_dot(&self, t: f64) -> f64 {
        (self.phi_dot)(t)
    }
}

/// Complete qubit description.
#[derive(Clone)]
pub struct QubitState {
    pub index: usize,
    pub omega: f64,
    pub trajectory: QubitTrajectory,
}

impl QubitState {
    pub fn bloch_vector(&self, t: f64) -> [f64; 3] {
        let theta = self.trajectory.theta(t);
        let phi = self.trajectory.phi(t);
        [
            theta.sin() * phi.cos(),
            theta.sin() * phi.sin(),
            theta.cos(),
        ]
    }

    pub fn bloch_velocity(&self, t: f64) -> [f64; 3] {
        // d/dt of the Bloch vector under spherical coordinates.
        let theta = self.trajectory.theta(t);
        let phi = self.trajectory.phi(t);
        let theta_dot = self.trajectory.theta_dot(t);
        let phi_dot = self.trajectory.phi_dot(t);

        let sin_theta = theta.sin();
        let cos_theta = theta.cos();
        let sin_phi = phi.sin();
        let cos_phi = phi.cos();

        [
            theta_dot * cos_theta * cos_phi - sin_theta * sin_phi * phi_dot,
            theta_dot * cos_theta * sin_phi + sin_theta * cos_phi * phi_dot,
            -theta_dot * sin_theta,
        ]
    }
}

/// Fidelity between two single-qubit pure states.
pub fn fidelity(m: &QubitState, n: &QubitState, t: f64) -> f64 {
    let theta_m = m.trajectory.theta(t);
    let theta_n = n.trajectory.theta(t);
    let phi_m = m.trajectory.phi(t);
    let phi_n = n.trajectory.phi(t);

    let cos_term = (phi_m - phi_n).cos();
    0.5 * (1.0 + theta_m.cos() * theta_n.cos() + theta_m.sin() * theta_n.sin() * cos_term)
}

/// Bloom amplitude definition.
#[derive(Clone)]
pub struct BloomSignature {
    pub varphi0: f64,
    pub epsilon_t: f64,
    pub psi: TimeFn,
}

impl BloomSignature {
    pub fn amplitude(&self, n: usize, t: f64, state: &QubitState) -> f64 {
        let theta = state.trajectory.theta(t);
        let envelope = (1.0 + n as f64 + self.epsilon_t * t).ln();
        let psi_t = (self.psi)(t);
        self.varphi0.powi(4) * theta.sin() * envelope.powf(psi_t)
    }

    pub fn d_theta(&self, n: usize, t: f64, state: &QubitState) -> f64 {
        let theta = state.trajectory.theta(t);
        let envelope = (1.0 + n as f64 + self.epsilon_t * t).ln();
        let psi_t = (self.psi)(t);
        self.varphi0.powi(4) * theta.cos() * envelope.powf(psi_t)
    }

    pub fn d_phi(&self, _n: usize, _t: f64, _state: &QubitState) -> f64 {
        0.0
    }
}

/// Wavelet modulation over a cluster.
pub fn wavelet_modulation(
    cluster: &ClusterStructure,
    cluster_index: usize,
    omega: f64,
    t: f64,
    bloom: &BloomSignature,
    states: &[QubitState],
) -> Complex64 {
    let mut acc = Complex64::new(0.0, 0.0);
    for n in cluster.cluster_indices(cluster_index) {
        if let Some(state) = states.get(n) {
            let phase = -omega * n as f64;
            let amp = bloom.amplitude(n, t, state);
            acc += Complex64::new(amp * phase.cos(), amp * phase.sin());
        }
    }
    acc
}

/// Spiral phase affinity matrix entry.
pub fn spiral_phase_affinity(
    m: &QubitState,
    n: &QubitState,
    t: f64,
    bloom: &BloomSignature,
) -> Complex64 {
    let amp = (bloom.amplitude(m.index, t, m) * bloom.amplitude(n.index, t, n))
        .abs()
        .sqrt();
    let fidelity = fidelity(m, n, t);
    let phase = (m.omega - n.omega) * t;
    Complex64::from_polar(amp * fidelity, phase)
}

/// Drift potential control describing the fractal-enhanced field.
#[derive(Clone)]
pub struct DriftField {
    pub phi: TimeFn,
    pub omega: f64,
    pub zeta: Vec<Complex64>,
}

#[derive(Clone, Copy, Debug)]
pub struct DriftVector {
    pub theta_component: f64,
    pub phi_component: f64,
}

impl DriftField {
    pub fn drift_vector(&self, state: &QubitState, t: f64) -> DriftVector {
        let phi_n = state.trajectory.phi(t);
        let phase = self.omega * t + phi_n;
        let idx = state.index;
        let zeta = self
            .zeta
            .get(idx)
            .copied()
            .unwrap_or_else(|| Complex64::new(1.0, 0.0));
        let cos_term = phase.cos();
        let d_phi = (self.phi)(t) * (zeta * cos_term).re;
        let sin_theta = state.trajectory.theta(t).sin();
        let phi_component = if sin_theta.abs() < 1e-9 {
            0.0
        } else {
            d_phi / sin_theta
        };
        DriftVector {
            theta_component: 0.0,
            phi_component,
        }
    }
}

/// Gaussian-weighted interference kernel.
pub fn interference_kernel(
    m: &QubitState,
    n: &QubitState,
    t: f64,
    bloom: &BloomSignature,
    drift: &DriftField,
    sigma_d: f64,
    sigma_bloom: f64,
) -> f64 {
    let dm = drift.drift_vector(m, t);
    let dn = drift.drift_vector(n, t);
    let norm_sq = (dm.theta_component - dn.theta_component).powi(2)
        + (dm.phi_component - dn.phi_component).powi(2);
    let drift_kernel = (-norm_sq / (2.0 * sigma_d.powi(2))).exp();

    let bm = bloom.amplitude(m.index, t, m);
    let bn = bloom.amplitude(n.index, t, n);
    let bloom_kernel = soft_delta(bm - bn, sigma_bloom);

    drift_kernel * bloom_kernel
}

/// Row-normalised transport coefficients.
pub fn row_normalised_transport(
    states: &[QubitState],
    t: f64,
    bloom: &BloomSignature,
    drift: &DriftField,
    sigma_d: f64,
    sigma_bloom: f64,
) -> Vec<Vec<f64>> {
    states
        .iter()
        .map(|m| {
            let mut row: Vec<f64> = states
                .iter()
                .map(|n| interference_kernel(m, n, t, bloom, drift, sigma_d, sigma_bloom))
                .collect();
            let sum: f64 = row.iter().sum();
            if sum > 0.0 {
                for val in &mut row {
                    *val /= sum;
                }
            }
            row
        })
        .collect()
}

/// Thermo-pressure functional.
pub fn thermo_pressure(n: &QubitState, t: f64, bloom: &BloomSignature, alpha_th: f64) -> f64 {
    let grad_sq = bloom.d_theta(n.index, t, n).powi(2) + bloom.d_phi(n.index, t, n).powi(2);
    (-alpha_th * grad_sq).exp()
}

/// Shadow entropy evaluation.
pub fn shadow_entropy(states: &[QubitState], t: f64, bloom: &BloomSignature) -> f64 {
    let n = states.len().max(1) as f64;
    let mut acc = 0.0;
    for state in states {
        let theta = state.trajectory.theta(t);
        let w_n = (1.0 + theta.cos()) / 2.0;
        let bloom_val = bloom.amplitude(state.index, t, state);
        let theta_grad = bloom.d_theta(state.index, t, state);
        let delta = (bloom_val - theta_grad).abs();
        acc += w_n * (1.0 + delta).ln();
    }
    acc / n
}

/// Count near-degeneracies for the taxicab entropy manifold.
pub fn degeneracy_counts(
    states: &[QubitState],
    t: f64,
    bloom: &BloomSignature,
    epsilon: f64,
) -> Vec<usize> {
    states
        .iter()
        .map(|m| {
            states
                .iter()
                .filter(|n| {
                    m.index != n.index
                        && (bloom.amplitude(m.index, t, m) - bloom.amplitude(n.index, t, n)).abs()
                            < epsilon
                })
                .count()
        })
        .collect()
}

/// Taxicab entropic collapse manifold per cluster.
pub fn taxicab_entropy(
    cluster: &ClusterStructure,
    cluster_index: usize,
    states: &[QubitState],
    t: f64,
    bloom: &BloomSignature,
    degeneracy: &[usize],
) -> f64 {
    cluster
        .cluster_indices(cluster_index)
        .filter_map(|idx| {
            let state = states.get(idx)?;
            let bloom_val = bloom.amplitude(state.index, t, state);
            let theta_grad = bloom.d_theta(state.index, t, state);
            let delta = (bloom_val - theta_grad).abs();
            let deg = degeneracy.get(idx).copied().unwrap_or(0) as f64;
            Some((1.0 + delta).ln() * deg)
        })
        .sum()
}

/// Collapse probability redistribution.
pub fn collapse_redistribution(
    lambda: &[Vec<f64>],
    mut probability: Vec<f64>,
    mu: f64,
) -> Vec<f64> {
    let mu = mu.clamp(0.0, 1.0);
    let mut mixed = vec![0.0; probability.len()];
    for (m, row) in lambda.iter().enumerate() {
        let mut acc = 0.0;
        for (n, &weight) in row.iter().enumerate() {
            let p_n = *probability.get(n).unwrap_or(&0.0);
            acc += weight * p_n;
        }
        mixed[m] = acc;
    }

    for (p, m) in probability.iter_mut().zip(mixed.into_iter()) {
        *p = (1.0 - mu) * *p + mu * m;
    }
    probability
}

/// Superposed drift according to the Lambda operator.
pub fn superposed_drift(
    lambda: &[Vec<f64>],
    states: &[QubitState],
    t: f64,
    drift: &DriftField,
) -> Vec<DriftVector> {
    lambda
        .iter()
        .enumerate()
        .map(|(_m, row)| {
            let mut theta_component = 0.0;
            let mut phi_component = 0.0;
            for (n, &weight) in row.iter().enumerate() {
                if let Some(state) = states.get(n) {
                    let dv = drift.drift_vector(state, t);
                    theta_component += weight * dv.theta_component;
                    phi_component += weight * dv.phi_component;
                }
            }
            DriftVector {
                theta_component,
                phi_component,
            }
        })
        .collect()
}

/// Monad biome descriptor.
#[derive(Clone, Debug)]
pub struct MonadBiome {
    pub weights: Vec<f64>,
}

/// Z-space descriptor.
#[derive(Clone, Debug)]
pub struct ZSpace {
    pub signature: Vec<f64>,
}

/// Result of linking Z-space and Monad biomes.
#[derive(Clone, Debug)]
pub struct ZMonadBridge {
    pub probability: Vec<f64>,
    pub shadow_entropy: f64,
    pub coherence: f64,
}

/// Complete engine tying the various components together.
#[derive(Clone)]
pub struct QuantumRealityEngine {
    pub hardware: HardwareStack,
    pub substrate: SubstratePhysicalModel,
    pub simulation: SimulationParameters,
    pub clusters: ClusterStructure,
    pub states: Vec<QubitState>,
    pub bloom: BloomSignature,
    pub drift: DriftField,
    pub sigma_d: f64,
    pub sigma_bloom: f64,
}

impl QuantumRealityEngine {
    pub fn lambda(&self, t: f64) -> Vec<Vec<f64>> {
        row_normalised_transport(
            &self.states,
            t,
            &self.bloom,
            &self.drift,
            self.sigma_d,
            self.sigma_bloom,
        )
    }

    pub fn connect_zspace_monad(
        &self,
        z_space: &ZSpace,
        biome: &MonadBiome,
        t: f64,
        mu: f64,
    ) -> ZMonadBridge {
        let lambda = self.lambda(t);
        let mut probability = normalize(&z_space.signature);
        probability = collapse_redistribution(&lambda, probability, mu);

        let entropy = shadow_entropy(&self.states, t, &self.bloom);
        let coherence = probability
            .iter()
            .zip(biome.weights.iter().chain(iter::repeat(&0.0)))
            .map(|(p, w)| p * w)
            .sum();

        ZMonadBridge {
            probability,
            shadow_entropy: entropy,
            coherence,
        }
    }
}

fn normalize(weights: &[f64]) -> Vec<f64> {
    let mut sum: f64 = weights.iter().copied().filter(|w| w.is_finite()).sum();
    if sum <= 0.0 || !sum.is_finite() {
        sum = 1.0;
    }
    weights.iter().map(|w| w.max(0.0) / sum).collect()
}
