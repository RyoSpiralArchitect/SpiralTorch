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

use std::cmp::Ordering;
use std::f64::consts::PI;
use std::iter;
use std::sync::Arc;

use nalgebra::{DMatrix, DVector, SymmetricEigen};
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

    pub fn cluster_count(&self) -> usize {
        self.segments.len()
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

/// Manhattan grid coordinate used for virtual taxicab embeddings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GridCoordinate {
    pub x: i32,
    pub y: i32,
}

impl GridCoordinate {
    pub fn manhattan_distance(&self, other: &GridCoordinate) -> i32 {
        (self.x - other.x).abs() + (self.y - other.y).abs()
    }
}

/// Virtual physical layout for qubit indices.
#[derive(Clone, Debug)]
pub struct TaxicabLayout {
    coordinates: Vec<GridCoordinate>,
}

impl TaxicabLayout {
    pub fn new(coordinates: Vec<GridCoordinate>) -> Self {
        TaxicabLayout { coordinates }
    }

    pub fn coordinate_of(&self, index: usize) -> GridCoordinate {
        self.coordinates
            .get(index)
            .copied()
            .unwrap_or(GridCoordinate {
                x: index as i32,
                y: 0,
            })
    }

    pub fn ensure_capacity(&mut self, len: usize) {
        while self.coordinates.len() < len {
            let idx = self.coordinates.len() as i32;
            self.coordinates.push(GridCoordinate { x: idx, y: 0 });
        }
    }

    pub fn from_cluster(cluster: &ClusterStructure) -> Self {
        let total = cluster.total_qubits();
        let mut coords = Vec::with_capacity(total);
        let mut offsets = vec![0usize; cluster.cluster_count().max(1)];
        for idx in 0..total {
            let cluster_idx = cluster.cluster_of(idx).unwrap_or(0);
            let offset = offsets[cluster_idx];
            offsets[cluster_idx] += 1;
            coords.push(GridCoordinate {
                x: offset as i32,
                y: cluster_idx as i32,
            });
        }
        TaxicabLayout::new(coords)
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

/// Coherent noise sources affecting the interference atlas dynamics.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoiseModel {
    pub t2_drift: f64,
    pub phi_drift: f64,
    pub omega_variation: f64,
}

impl NoiseModel {
    fn is_active(&self) -> bool {
        self.t2_drift > 0.0 || self.phi_drift.abs() > 0.0 || self.omega_variation.abs() > 0.0
    }
}

/// Result of constructing an interference atlas.
#[derive(Clone, Debug)]
pub struct AtlasResult {
    pub kernel: Vec<Vec<f64>>,
    pub diffusion_coordinates: Vec<Vec<f64>>,
    pub pseudo_cluster_ratio: f64,
}

/// Comparison between pristine and noise-perturbed atlas projections.
#[derive(Clone, Debug)]
pub struct AtlasAnalysis {
    pub base: AtlasResult,
    pub noisy: Option<AtlasResult>,
    pub stability: f64,
}

/// Selected parameters for the collapse drive auto-tuning.
#[derive(Clone, Debug)]
pub struct AutoTuneResult {
    pub k_nn: usize,
    pub k_scale: f64,
    pub bandwidth: f64,
    pub objective: f64,
    pub connectivity: usize,
    pub lambda2: f64,
    pub pseudo_ratio: f64,
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
        .map(|row| {
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

fn atlas_feature(
    state: &QubitState,
    t: f64,
    bloom: &BloomSignature,
    drift: &DriftField,
    coord: GridCoordinate,
) -> [f64; 5] {
    let bloom_amp = bloom.amplitude(state.index, t, state);
    let drift_vec = drift.drift_vector(state, t);
    [
        bloom_amp,
        drift_vec.phi_component,
        state.omega,
        coord.x as f64,
        coord.y as f64,
    ]
}

fn dual_local_kernel(
    states: &[QubitState],
    t: f64,
    bloom: &BloomSignature,
    drift: &DriftField,
    layout: &TaxicabLayout,
    k_scale: f64,
    bandwidth: f64,
) -> Vec<Vec<f64>> {
    let sigma = bandwidth.max(1e-9);
    let freq_scale = k_scale.abs().max(1e-9);
    states
        .iter()
        .map(|m| {
            let coord_m = layout.coordinate_of(m.index);
            let feat_m = atlas_feature(m, t, bloom, drift, coord_m);
            states
                .iter()
                .map(|n| {
                    let coord_n = layout.coordinate_of(n.index);
                    let feat_n = atlas_feature(n, t, bloom, drift, coord_n);
                    let delta_b = feat_m[0] - feat_n[0];
                    let delta_dphi = feat_m[1] - feat_n[1];
                    let delta_omega = (feat_m[2] - feat_n[2]) / freq_scale;
                    let delta_x = feat_m[3] - feat_n[3];
                    let delta_y = feat_m[4] - feat_n[4];
                    let manhattan = coord_m.manhattan_distance(&coord_n) as f64;
                    let feature_distance =
                        delta_b.powi(2) + delta_dphi.powi(2) + delta_omega.powi(2);
                    let spatial_distance = delta_x.abs() + delta_y.abs() + manhattan;
                    let norm = feature_distance + spatial_distance;
                    (-norm / (sigma * sigma)).exp()
                })
                .collect()
        })
        .collect()
}

fn knn_adjacency(kernel: &[Vec<f64>], k: usize) -> Vec<Vec<f64>> {
    let n = kernel.len();
    let mut adjacency = vec![vec![0.0; n]; n];
    for (i, row) in kernel.iter().enumerate() {
        let mut pairs: Vec<(usize, f64)> = row
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(j, &w)| (j, w))
            .collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        for &(idx, weight) in pairs.iter().take(k) {
            let j = idx;
            let w = weight.max(0.0);
            if adjacency[i][j] < w {
                adjacency[i][j] = w;
            }
            if adjacency[j][i] < w {
                adjacency[j][i] = w;
            }
        }
    }
    adjacency
}

fn row_normalise_matrix(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    matrix
        .iter()
        .map(|row| {
            let sum: f64 = row.iter().sum();
            if sum > 0.0 {
                row.iter().map(|val| val / sum).collect()
            } else {
                row.iter().map(|_| 0.0).collect()
            }
        })
        .collect()
}

fn connected_components(adjacency: &[Vec<f64>]) -> usize {
    let n = adjacency.len();
    if n == 0 {
        return 0;
    }
    let mut visited = vec![false; n];
    let mut components = 0usize;
    for start in 0..n {
        if visited[start] {
            continue;
        }
        components += 1;
        let mut stack = vec![start];
        visited[start] = true;
        while let Some(node) = stack.pop() {
            for (nbr, &weight) in adjacency[node].iter().enumerate() {
                if weight > 0.0 && !visited[nbr] {
                    visited[nbr] = true;
                    stack.push(nbr);
                }
            }
        }
    }
    components
}

fn laplacian_lambda2(adjacency: &[Vec<f64>]) -> f64 {
    let n = adjacency.len();
    if n <= 1 {
        return 0.0;
    }
    let laplacian = DMatrix::from_fn(n, n, |i, j| {
        if i == j {
            adjacency[i].iter().sum::<f64>()
        } else {
            -adjacency[i][j]
        }
    });
    let eigen = SymmetricEigen::new(laplacian);
    let mut values: Vec<f64> = eigen.eigenvalues.iter().copied().collect();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    if values.len() >= 2 {
        values[1].max(0.0)
    } else {
        0.0
    }
}

fn pseudo_ratio_from_matrix(matrix: &[Vec<f64>], clusters: &ClusterStructure) -> f64 {
    let mut within = 0.0;
    let mut between = 0.0;
    for (i, row) in matrix.iter().enumerate() {
        for (j, &weight) in row.iter().enumerate() {
            if i == j || weight <= 0.0 {
                continue;
            }
            let ci = clusters.cluster_of(i).unwrap_or(usize::MAX);
            let cj = clusters.cluster_of(j).unwrap_or(usize::MAX);
            if ci == cj {
                within += weight;
            } else {
                between += weight;
            }
        }
    }
    within / (between + 1e-9)
}

fn matrix_to_vec(matrix: &DMatrix<f64>) -> Vec<Vec<f64>> {
    (0..matrix.nrows())
        .map(|i| (0..matrix.ncols()).map(|j| matrix[(i, j)]).collect())
        .collect()
}

fn project_to_psd(matrix: DMatrix<f64>) -> DMatrix<f64> {
    let symmetric = (matrix.clone() + matrix.transpose()) * 0.5;
    let eigen = SymmetricEigen::new(symmetric);
    let mut diag = DVector::zeros(eigen.eigenvalues.len());
    for (idx, val) in eigen.eigenvalues.iter().enumerate() {
        diag[idx] = val.max(0.0);
    }
    let diag_matrix = DMatrix::from_diagonal(&diag);
    &eigen.eigenvectors * diag_matrix * eigen.eigenvectors.transpose()
}

fn diffusion_map(kernel: &DMatrix<f64>, components: usize) -> Vec<Vec<f64>> {
    let n = kernel.nrows();
    if n == 0 {
        return Vec::new();
    }
    let mut degrees = DVector::zeros(n);
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..n {
            sum += kernel[(i, j)];
        }
        degrees[i] = sum.max(1e-9);
    }
    let mut norm = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            norm[(i, j)] = kernel[(i, j)] / (degrees[i] * degrees[j]).sqrt();
        }
    }
    let eigen = SymmetricEigen::new(norm);
    let mut pairs: Vec<(f64, Vec<f64>)> = eigen
        .eigenvalues
        .iter()
        .zip(eigen.eigenvectors.column_iter())
        .map(|(val, vec)| (val.abs(), vec.iter().copied().collect()))
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    let embed_dim = components.min(n);
    let mut embedding = vec![vec![0.0; embed_dim]; n];
    for (comp_idx, (_eig, vector)) in pairs.into_iter().take(embed_dim).enumerate() {
        for (row, value) in vector.into_iter().enumerate() {
            embedding[row][comp_idx] = value;
        }
    }
    embedding
}

fn build_interference_matrix(
    states: &[QubitState],
    t: f64,
    bloom: &BloomSignature,
    drift: &DriftField,
    layout: &TaxicabLayout,
    bandwidth: f64,
    noise: Option<NoiseModel>,
) -> DMatrix<f64> {
    let sigma = bandwidth.max(1e-6);
    let mut matrix = DMatrix::zeros(states.len(), states.len());
    for (i, m) in states.iter().enumerate() {
        let coord_m = layout.coordinate_of(m.index);
        for (j, n) in states.iter().enumerate() {
            let coord_n = layout.coordinate_of(n.index);
            let amp_m = bloom.amplitude(m.index, t, m);
            let amp_n = bloom.amplitude(n.index, t, n);
            let amp = 0.5 * (amp_m + amp_n);
            let base_phase = (m.omega - n.omega) * t;
            let spatial = {
                let dist = coord_m.manhattan_distance(&coord_n) as f64;
                (-dist * dist / (sigma * sigma)).exp()
            };
            let mut phase = base_phase;
            let mut amplitude = amp;
            if let Some(model) = noise {
                if model.t2_drift > 0.0 {
                    let decay = (-t.abs() / model.t2_drift.max(1e-9)).exp();
                    amplitude *= decay;
                }
                phase += model.phi_drift * (m.trajectory.phi(t) - n.trajectory.phi(t));
                phase += model.omega_variation * t * (m.index as f64 - n.index as f64);
            }
            let drift_delta =
                drift.drift_vector(m, t).phi_component - drift.drift_vector(n, t).phi_component;
            phase += drift_delta;
            matrix[(i, j)] = amplitude * phase.cos() * spatial;
        }
    }
    matrix
}

#[allow(clippy::too_many_arguments)]
pub fn interference_atlas(
    states: &[QubitState],
    t: f64,
    bloom: &BloomSignature,
    drift: &DriftField,
    layout: &TaxicabLayout,
    bandwidth: f64,
    clusters: &ClusterStructure,
    noise: Option<NoiseModel>,
) -> AtlasAnalysis {
    let base_matrix = build_interference_matrix(states, t, bloom, drift, layout, bandwidth, None);
    let psd = project_to_psd(base_matrix);
    let kernel_vec = matrix_to_vec(&psd);
    let diffusion = diffusion_map(&psd, 3);
    let pseudo_ratio = pseudo_ratio_from_matrix(&kernel_vec, clusters);
    let base_result = AtlasResult {
        kernel: kernel_vec,
        diffusion_coordinates: diffusion,
        pseudo_cluster_ratio: pseudo_ratio,
    };

    let noisy_result = noise.and_then(|model| {
        if model.is_active() {
            let noisy_matrix =
                build_interference_matrix(states, t, bloom, drift, layout, bandwidth, Some(model));
            let noisy_psd = project_to_psd(noisy_matrix);
            let kernel_vec = matrix_to_vec(&noisy_psd);
            let diffusion = diffusion_map(&noisy_psd, 3);
            let pseudo_ratio = pseudo_ratio_from_matrix(&kernel_vec, clusters);
            Some(AtlasResult {
                kernel: kernel_vec,
                diffusion_coordinates: diffusion,
                pseudo_cluster_ratio: pseudo_ratio,
            })
        } else {
            None
        }
    });

    let stability = noisy_result
        .as_ref()
        .map(|noisy| {
            let denom = base_result.pseudo_cluster_ratio.abs() + 1e-9;
            1.0 - ((base_result.pseudo_cluster_ratio - noisy.pseudo_cluster_ratio).abs() / denom)
        })
        .unwrap_or(1.0)
        .clamp(0.0, 1.0);

    AtlasAnalysis {
        base: base_result,
        noisy: noisy_result,
        stability,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn auto_tune_collapse_drive(
    states: &[QubitState],
    t: f64,
    bloom: &BloomSignature,
    drift: &DriftField,
    layout: &mut TaxicabLayout,
    clusters: &ClusterStructure,
    k_nn_candidates: &[usize],
    k_scale_candidates: &[f64],
    bandwidth_candidates: &[f64],
) -> Option<AutoTuneResult> {
    layout.ensure_capacity(states.len());
    let mut best: Option<AutoTuneResult> = None;
    for &k_nn in k_nn_candidates {
        if k_nn == 0 {
            continue;
        }
        for &k_scale in k_scale_candidates {
            for &bandwidth in bandwidth_candidates {
                let kernel = dual_local_kernel(states, t, bloom, drift, layout, k_scale, bandwidth);
                let adjacency = knn_adjacency(&kernel, k_nn.min(states.len().saturating_sub(1)));
                let components = connected_components(&adjacency);
                let connectivity_score = if components == 0 {
                    0.0
                } else {
                    (1.0 / components as f64).clamp(0.0, 1.0)
                };
                let lambda2 = laplacian_lambda2(&adjacency);
                let lambda_score = (-((lambda2 - 0.5).powi(2)) / 0.25).exp();
                let pseudo_ratio = pseudo_ratio_from_matrix(&adjacency, clusters);
                let ratio_score = pseudo_ratio / (1.0 + pseudo_ratio);
                let objective = connectivity_score * lambda_score * ratio_score;
                let candidate = AutoTuneResult {
                    k_nn,
                    k_scale,
                    bandwidth,
                    objective,
                    connectivity: components,
                    lambda2,
                    pseudo_ratio,
                };
                match &best {
                    Some(current) if current.objective >= candidate.objective => {}
                    _ => {
                        best = Some(candidate);
                    }
                }
            }
        }
    }
    best
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

/// Summary of the collapse manifold state used for diagnostics.
#[derive(Clone, Debug)]
pub struct CollapseManifold {
    /// Entropy contributions per cluster on the taxicab manifold.
    pub cluster_entropies: Vec<f64>,
    /// Degeneracy counts per qubit used when constructing the manifold.
    pub degeneracy: Vec<usize>,
    /// Thermo-pressure samples per qubit at the query time.
    pub pressure_map: Vec<f64>,
    /// Mean entropy across all clusters for a quick stability indicator.
    pub average_entropy: f64,
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

    pub fn lambda_with_dual_locality(
        &self,
        t: f64,
        layout: &mut TaxicabLayout,
        params: &AutoTuneResult,
    ) -> Vec<Vec<f64>> {
        layout.ensure_capacity(self.states.len());
        let kernel = dual_local_kernel(
            &self.states,
            t,
            &self.bloom,
            &self.drift,
            layout,
            params.k_scale,
            params.bandwidth,
        );
        let adjacency = knn_adjacency(
            &kernel,
            params.k_nn.min(self.states.len().saturating_sub(1)),
        );
        row_normalise_matrix(&adjacency)
    }

    pub fn tuned_collapse_drive(
        &self,
        t: f64,
        layout: &mut TaxicabLayout,
        k_nn_candidates: &[usize],
        k_scale_candidates: &[f64],
        bandwidth_candidates: &[f64],
    ) -> Option<(AutoTuneResult, Vec<Vec<f64>>)> {
        let params = auto_tune_collapse_drive(
            &self.states,
            t,
            &self.bloom,
            &self.drift,
            layout,
            &self.clusters,
            k_nn_candidates,
            k_scale_candidates,
            bandwidth_candidates,
        )?;
        let lambda = self.lambda_with_dual_locality(t, layout, &params);
        Some((params, lambda))
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

    /// Construct a diagnostic collapse manifold snapshot at time `t`.
    ///
    /// `degeneracy_epsilon` controls the tolerance when counting near-degenerate
    /// bloom amplitudes, while `alpha_th` governs the thermo-pressure decay.
    pub fn collapse_manifold(
        &self,
        t: f64,
        degeneracy_epsilon: f64,
        alpha_th: f64,
    ) -> CollapseManifold {
        let epsilon = degeneracy_epsilon.max(1e-9);
        let degeneracy = degeneracy_counts(&self.states, t, &self.bloom, epsilon);

        let mut cluster_entropies = Vec::new();
        for cluster_idx in 0..self.clusters.cluster_count() {
            let entropy = taxicab_entropy(
                &self.clusters,
                cluster_idx,
                &self.states,
                t,
                &self.bloom,
                &degeneracy,
            );
            cluster_entropies.push(entropy);
        }

        let pressure_map = self
            .states
            .iter()
            .map(|state| thermo_pressure(state, t, &self.bloom, alpha_th))
            .collect::<Vec<_>>();

        let average_entropy = if cluster_entropies.is_empty() {
            0.0
        } else {
            cluster_entropies.iter().sum::<f64>() / cluster_entropies.len() as f64
        };

        CollapseManifold {
            cluster_entropies,
            degeneracy,
            pressure_map,
            average_entropy,
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;
    use std::sync::Arc;

    fn constant_fn(value: f64) -> TimeFn {
        Arc::new(move |_| value)
    }

    #[test]
    fn collapse_manifold_reports_cluster_metrics() {
        let hardware = HardwareStack {
            qubit_layer: QubitLayer {
                material: "NbTiN",
                geometry: "planar",
            },
            junction: JosephsonJunction {
                barrier_material: "AlOx",
                barrier_thickness_nm: 1.8,
            },
            on_chip_lines: OnChipLines {
                topology: "coplanar",
                center_width_microns: 10.0,
                impedance_ohms: 50.0,
            },
            ground_plane: GroundPlane {
                material: "Nb",
                shielded: true,
            },
            cryo_enclosure: CryoEnclosure {
                temperature_mk: 10.0,
                shields: "mu-metal",
            },
        };

        let substrate = SubstratePhysicalModel {
            lattice_a_angstrom: 4.76,
            lattice_c_angstrom: 12.99,
            die_dims_mm: [8.0, 8.0, 0.5],
            epsilon_r: 9.4,
            thermal_conductivity_w_mk: 35.0,
            cte_per_k: 5.6e-6,
        };

        let simulation = SimulationParameters {
            qubit_frequency_ghz: 5.0,
            anharmonicity_mhz: -300.0,
            coupling_mhz: 20.0,
            t1_microseconds: 30.0,
            t2_microseconds: 40.0,
            pi_pulse_ns: 25.0,
        };

        let clusters = ClusterStructure::new(vec![2, 1]);

        let states = vec![
            QubitState {
                index: 0,
                omega: 5.0,
                trajectory: QubitTrajectory {
                    theta: constant_fn(PI / 4.0),
                    phi: constant_fn(0.0),
                    theta_dot: constant_fn(0.0),
                    phi_dot: constant_fn(0.0),
                },
            },
            QubitState {
                index: 1,
                omega: 5.2,
                trajectory: QubitTrajectory {
                    theta: constant_fn(PI / 3.0),
                    phi: constant_fn(PI / 8.0),
                    theta_dot: constant_fn(0.01),
                    phi_dot: constant_fn(0.02),
                },
            },
            QubitState {
                index: 2,
                omega: 4.8,
                trajectory: QubitTrajectory {
                    theta: constant_fn(PI / 6.0),
                    phi: constant_fn(PI / 7.0),
                    theta_dot: constant_fn(0.0),
                    phi_dot: constant_fn(0.0),
                },
            },
        ];

        let bloom = BloomSignature {
            varphi0: 0.7,
            epsilon_t: 0.05,
            psi: constant_fn(1.2),
        };

        let drift = DriftField {
            phi: constant_fn(0.3),
            omega: 0.15,
            zeta: vec![Complex64::new(1.0, 0.0); 3],
        };

        let engine = QuantumRealityEngine {
            hardware,
            substrate,
            simulation,
            clusters,
            states,
            bloom,
            drift,
            sigma_d: 0.2,
            sigma_bloom: 0.3,
        };

        let cluster_count = engine.clusters.cluster_count();
        let state_count = engine.states.len();

        let manifold = engine.collapse_manifold(0.1, 0.05, 0.4);

        assert_eq!(manifold.degeneracy.len(), state_count);
        assert_eq!(manifold.pressure_map.len(), state_count);
        assert_eq!(manifold.cluster_entropies.len(), cluster_count);

        let expected_average = if manifold.cluster_entropies.is_empty() {
            0.0
        } else {
            manifold.cluster_entropies.iter().sum::<f64>() / manifold.cluster_entropies.len() as f64
        };
        assert!((manifold.average_entropy - expected_average).abs() < 1e-9);

        for pressure in manifold.pressure_map {
            assert!(pressure.is_finite());
        }
    }
}
