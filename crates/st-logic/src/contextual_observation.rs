// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! Contextual observation primitives inspired by the "pure/mixture"
//! correspondence shared in the research notes.
//!
//! The goal of this module is to turn the informal axioms `(A1)–(A5)`
//! into concrete Rust data structures that can be used by higher-level
//! logic engines.  We encode three layers:
//!
//! * **Arrangements** — placements of latent pure atoms `Â`/`𝐵̂` on a
//!   discrete support `Λ`.
//! * **Gauge-invariant observation** — an observer that only sees
//!   relational structure and therefore cannot distinguish a global swap
//!   of `Â ↔ 𝐵̂`.
//! * **Orientation lifts** — optional gauge choices (the refined context
//!   `c′`) that can turn a symmetric signature into oriented labels
//!   `a`/`b` whenever the underlying arrangement supports such a lift.
//!
//! Together these components provide a lightweight reference
//! implementation of the descriptive-to-existential duality highlighted
//! in the accompanying memo.

use std::cmp::Ordering;
use std::collections::VecDeque;
use std::f32::consts::TAU;

use num_complex::Complex32;
use st_core::theory::zpulse::{ZPulse, ZScale, ZSource, ZSupport};
use st_tensor::{PureResult, Tensor};

/// Latent pure atoms — the unobservable `Â` and `𝐵̂` units.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PureAtom {
    A,
    B,
}

impl PureAtom {
    fn flipped(self) -> Self {
        match self {
            PureAtom::A => PureAtom::B,
            PureAtom::B => PureAtom::A,
        }
    }
}

/// Discrete arrangement (placement) of pure atoms on the support `Λ`.
///
/// The arrangement remembers the adjacency of the support as an edge
/// list, so we can talk about connected components and boundary counts
/// without assuming a particular geometry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Arrangement {
    placements: Vec<PureAtom>,
    edges: Vec<(usize, usize)>,
    adjacency: Vec<Vec<usize>>,
}

impl Arrangement {
    /// Creates a new arrangement.  The edge list is interpreted as an
    /// undirected simple graph; invalid indices cause a panic, keeping
    /// the implementation honest for test-time usage.
    pub fn new(placements: Vec<PureAtom>, mut edges: Vec<(usize, usize)>) -> Self {
        let n = placements.len();
        for &(u, v) in &edges {
            assert!(u < n && v < n, "edge indices must be in-bounds");
            assert!(u != v, "self-loops are not allowed");
        }
        edges.sort();
        edges.dedup();
        let mut adjacency = vec![Vec::new(); n];
        for (u, v) in &edges {
            adjacency[*u].push(*v);
            adjacency[*v].push(*u);
        }
        Arrangement {
            placements,
            edges,
            adjacency,
        }
    }

    /// Constructs a 1-D lattice (path graph) arrangement — the toy
    /// example described in the memo.
    pub fn from_line(placements: Vec<PureAtom>) -> Self {
        let len = placements.len();
        let edges = (0..len.saturating_sub(1)).map(|i| (i, i + 1)).collect();
        Arrangement::new(placements, edges)
    }

    /// Returns the support size `|Λ|`.
    pub fn len(&self) -> usize {
        self.placements.len()
    }

    /// Returns the number of edges on which neighbouring atoms disagree
    /// — the "boundary gate" quantity.
    pub fn boundary_edges(&self) -> usize {
        self.edges
            .iter()
            .filter(|&&(u, v)| self.placements[u] != self.placements[v])
            .count()
    }

    /// Determines whether the arrangement is pure (all `Â` or all `𝐵̂`).
    pub fn is_pure(&self) -> bool {
        self.boundary_edges() == 0
    }

    /// Returns a flipped arrangement with every latent atom swapped.
    pub fn flipped(&self) -> Self {
        let placements = self
            .placements
            .iter()
            .copied()
            .map(PureAtom::flipped)
            .collect();
        Arrangement::new(placements, self.edges.clone())
    }

    /// Counts how many nodes carry each atom.
    pub fn population(&self) -> [usize; 2] {
        let mut counts = [0usize; 2];
        for atom in &self.placements {
            match atom {
                PureAtom::A => counts[0] += 1,
                PureAtom::B => counts[1] += 1,
            }
        }
        counts
    }

    /// Computes the number of connected clusters for each atom type
    /// using a BFS that respects the latent placement.
    pub fn cluster_counts(&self) -> [usize; 2] {
        let mut visited = vec![false; self.placements.len()];
        let mut counts = [0usize; 2];
        for start in 0..self.placements.len() {
            if visited[start] {
                continue;
            }
            let atom = self.placements[start];
            counts[atom.index()] += 1;
            let mut queue = VecDeque::from([start]);
            visited[start] = true;
            while let Some(node) = queue.pop_front() {
                for &nbr in &self.adjacency[node] {
                    if !visited[nbr] && self.placements[nbr] == atom {
                        visited[nbr] = true;
                        queue.push_back(nbr);
                    }
                }
            }
        }
        counts
    }

    /// Signed cluster imbalance `#clusters(𝐵̂) - #clusters(Â)`.
    pub fn cluster_imbalance(&self) -> isize {
        let counts = self.cluster_counts();
        counts[1] as isize - counts[0] as isize
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContextualSignature {
    pub boundary_edges: usize,
    pub absolute_population_imbalance: usize,
    pub cluster_imbalance: isize,
}

impl ContextualSignature {
    pub fn meaning_weight(&self) -> f32 {
        let boundary = self.boundary_edges as f32;
        let population = self.absolute_population_imbalance as f32;
        let cluster = self.cluster_imbalance.abs() as f32;
        if boundary == 0.0 && population == 0.0 && cluster == 0.0 {
            0.0
        } else {
            boundary + 0.5 * population + 0.25 * cluster
        }
    }
}

impl PureAtom {
    fn index(self) -> usize {
        match self {
            PureAtom::A => 0,
            PureAtom::B => 1,
        }
    }
}

/// Result of the gauge-invariant observer `ρ : Arr → {a, b, ⊥}`.
///
/// * `Undetermined` corresponds to the pure placements; the observer has
///   no access to the latent identity.
/// * `Signature` contains the relational invariants that survive the
///   quotient by the global swap.  The absolute population imbalance
///   keeps track of the "degree of asymmetry" without committing to an
///   orientation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Observation {
    Undetermined,
    Signature {
        boundary_edges: usize,
        absolute_population_imbalance: usize,
    },
}

/// Gauge-invariant observation engine implementing axioms `(A1)–(A4)`.
pub struct ContextObserver;

impl ContextObserver {
    /// Applies the contextual observer to an arrangement.
    pub fn observe(arrangement: &Arrangement) -> Observation {
        let boundary = arrangement.boundary_edges();
        if boundary == 0 {
            return Observation::Undetermined;
        }
        let counts = arrangement.population();
        let imbalance = counts[1].abs_diff(counts[0]);
        Observation::Signature {
            boundary_edges: boundary,
            absolute_population_imbalance: imbalance,
        }
    }

    /// Verifies the invariance `ρ(x) = ρ(σ(x))` by comparing the
    /// observation of an arrangement with that of its flipped copy.
    pub fn is_swap_invariant(arrangement: &Arrangement) -> bool {
        let flipped = arrangement.flipped();
        Self::observe(arrangement) == Self::observe(&flipped)
    }

    /// Returns the contextual signature when the observer detects a
    /// distinguishable pattern.
    pub fn signature(arrangement: &Arrangement) -> Option<ContextualSignature> {
        match Self::observe(arrangement) {
            Observation::Signature {
                boundary_edges,
                absolute_population_imbalance,
            } => Some(ContextualSignature {
                boundary_edges,
                absolute_population_imbalance,
                cluster_imbalance: arrangement.cluster_imbalance(),
            }),
            Observation::Undetermined => None,
        }
    }
}

/// Orientation preferences (`c′`) for lifting the invariant signature
/// back to labelled outcomes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrientationGauge {
    /// Choose the label that tracks the dominant population.
    Preserve,
    /// Swap the labels (useful when we re-anchor the latent atoms).
    Swap,
}

impl OrientationGauge {
    fn apply(self, label: Label) -> Label {
        match (self, label) {
            (OrientationGauge::Preserve, _) => label,
            (OrientationGauge::Swap, label) => label.flipped(),
        }
    }
}

/// Oriented labels that become accessible once a gauge is fixed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Label {
    A,
    B,
}

impl Label {
    fn flipped(self) -> Self {
        match self {
            Label::A => Label::B,
            Label::B => Label::A,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Label::A => "a",
            Label::B => "b",
        }
    }
}

/// Lift the symmetric observation into an oriented label when possible.
/// Returns `None` for pure arrangements or when the population balance is
/// neutral — matching the intuition that pure `Â`/`𝐵̂` placements remain
/// unobservable even after gauge fixing.
pub fn lift_to_label(arrangement: &Arrangement, gauge: OrientationGauge) -> Option<Label> {
    if arrangement.is_pure() {
        return None;
    }
    let counts = arrangement.population();
    if counts[0] == counts[1] {
        return None;
    }
    let base = if counts[1] > counts[0] {
        Label::B
    } else {
        Label::A
    };
    Some(gauge.apply(base))
}

#[derive(Clone, Debug, PartialEq)]
pub struct MeaningBasis {
    signal: Tensor,
    spectrum: Tensor,
}

impl MeaningBasis {
    pub fn from_arrangement(arrangement: &Arrangement) -> PureResult<Self> {
        let signal_values = arrangement_signal_values(arrangement);
        let spectrum = arrangement_spectrum_from_signal(&signal_values)?;
        let signal = Tensor::from_vec(1, signal_values.len(), signal_values)?;
        Ok(Self { signal, spectrum })
    }

    pub fn signal(&self) -> &Tensor {
        &self.signal
    }

    pub fn spectrum(&self) -> &Tensor {
        &self.spectrum
    }

    pub fn dominant_frequency(&self) -> Option<(usize, f32)> {
        self.spectrum
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| match left.partial_cmp(right) {
                Some(ordering) => ordering,
                None => Ordering::Equal,
            })
            .map(|(index, magnitude)| (index, *magnitude))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MeaningProjection {
    pub observation: Observation,
    pub signature: Option<ContextualSignature>,
    pub label: Option<Label>,
    pub basis: MeaningBasis,
    pub support: usize,
}

/// Aggregate statistics describing how meaning evolves across a timeline of
/// projections. Instead of treating each observation as a static imitation,
/// the metrics emphasise the interplay between contextual stability and
/// transformative shifts—mirroring the manifesto's call for emergent meaning.
#[derive(Clone, Debug, PartialEq)]
pub struct MeaningEmergenceMetrics {
    pub lexical_mean: f32,
    pub lexical_std: f32,
    pub lexical_flux: f32,
    pub support_mean: f32,
    pub support_std: f32,
    pub orientation_flip_rate: f32,
    pub indeterminate_share: f32,
    pub frequency_flux: f32,
    pub coherence_score: f32,
    pub emergence_score: f32,
}

impl MeaningEmergenceMetrics {
    /// Builds the metrics from a chronological series of projections.
    pub fn from_projections(projections: &[MeaningProjection]) -> Option<Self> {
        if projections.is_empty() {
            return None;
        }

        let count = projections.len() as f32;
        let lexical_values: Vec<f32> = projections.iter().map(|p| p.lexical_weight()).collect();
        let lexical_mean = lexical_values.iter().sum::<f32>() / count;
        let lexical_std = if projections.len() > 1 {
            let variance = lexical_values
                .iter()
                .map(|value| {
                    let diff = *value - lexical_mean;
                    diff * diff
                })
                .sum::<f32>()
                / count;
            variance.sqrt()
        } else {
            0.0
        };

        let support_values: Vec<f32> = projections.iter().map(|p| p.support as f32).collect();
        let support_mean = support_values.iter().sum::<f32>() / count;
        let support_std = if projections.len() > 1 {
            let variance = support_values
                .iter()
                .map(|value| {
                    let diff = *value - support_mean;
                    diff * diff
                })
                .sum::<f32>()
                / count;
            variance.sqrt()
        } else {
            0.0
        };

        let mut lexical_flux = 0.0;
        let mut orientation_flips = 0;
        let mut orientation_pairs = 0;
        let mut indeterminate = 0;
        let mut frequency_flux = 0.0;
        let mut frequency_pairs = 0;
        let mut previous_orientation = None;
        let mut previous_lexical = None;
        let mut previous_frequency = None;

        for projection in projections {
            if projection.label.is_none() {
                indeterminate += 1;
            }

            let orientation = orientation_sign(projection.label);
            if let Some(prev) = previous_orientation {
                orientation_pairs += 1;
                if orientation != 0.0 && prev != 0.0 && orientation.signum() != prev.signum() {
                    orientation_flips += 1;
                }
            }
            previous_orientation = Some(orientation);

            let lexical = projection.lexical_weight();
            if let Some(prev) = previous_lexical {
                lexical_flux += (lexical - prev).abs();
            }
            previous_lexical = Some(lexical);

            if let Some((bin, _)) = projection.dominant_frequency_bin() {
                let bin = bin as f32;
                if let Some(prev) = previous_frequency {
                    frequency_flux += (bin - prev).abs();
                    frequency_pairs += 1;
                }
                previous_frequency = Some(bin);
            } else {
                previous_frequency = None;
            }
        }

        let normaliser = (projections.len().saturating_sub(1)).max(1) as f32;
        let lexical_flux = lexical_flux / normaliser;
        let orientation_flip_rate = if orientation_pairs == 0 {
            0.0
        } else {
            orientation_flips as f32 / orientation_pairs as f32
        };
        let frequency_flux = if frequency_pairs == 0 {
            0.0
        } else {
            frequency_flux / frequency_pairs as f32
        };
        let indeterminate_share = indeterminate as f32 / count;

        let coherence_score = (1.0 / (1.0 + lexical_std))
            * (1.0 - orientation_flip_rate).max(0.0)
            * (1.0 / (1.0 + frequency_flux * 0.1));
        let emergence_score =
            (lexical_flux.tanh() + frequency_flux.tanh()) * 0.5 + orientation_flip_rate;

        Some(Self {
            lexical_mean,
            lexical_std,
            lexical_flux,
            support_mean,
            support_std,
            orientation_flip_rate,
            indeterminate_share,
            frequency_flux,
            coherence_score,
            emergence_score,
        })
    }
}

impl MeaningProjection {
    pub fn from_arrangement(
        arrangement: &Arrangement,
        gauge: OrientationGauge,
    ) -> PureResult<Self> {
        let observation = ContextObserver::observe(arrangement);
        let signature = ContextObserver::signature(arrangement);
        let label = lift_to_label(arrangement, gauge);
        let basis = MeaningBasis::from_arrangement(arrangement)?;
        Ok(Self {
            observation,
            signature,
            label,
            basis,
            support: arrangement.len(),
        })
    }

    pub fn lexical_weight(&self) -> f32 {
        self.signature
            .as_ref()
            .map(|signature| {
                let support = self.support.max(1) as f32;
                signature.meaning_weight() / support
            })
            .unwrap_or(0.0)
    }

    pub fn dominant_frequency_bin(&self) -> Option<(usize, f32)> {
        self.basis.dominant_frequency()
    }

    fn orientation_sign(&self) -> f32 {
        orientation_sign(self.label)
    }
}

fn orientation_sign(label: Option<Label>) -> f32 {
    match label {
        Some(Label::A) => -1.0,
        Some(Label::B) => 1.0,
        None => 0.0,
    }
}

/// Configuration describing how contextual meaning should be pushed through the
/// Desire Lagrangian gate and into Z-space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LagrangianGateConfig {
    pub tempo_normaliser: f32,
    pub energy_gain: f32,
    pub drift_gain: f32,
    pub bias_gain: f32,
    pub support_gain: f32,
    pub quality_floor: f32,
    pub stderr_gain: f32,
    pub scale: Option<ZScale>,
}

impl Default for LagrangianGateConfig {
    fn default() -> Self {
        Self {
            tempo_normaliser: 1.0,
            energy_gain: 1.0,
            drift_gain: 1.0,
            bias_gain: 1.0,
            support_gain: 1.0,
            quality_floor: 0.0,
            stderr_gain: 1.0,
            scale: Some(ZScale::ONE),
        }
    }
}

impl LagrangianGateConfig {
    pub fn tempo_normaliser(mut self, value: f32) -> Self {
        self.tempo_normaliser = value.max(0.0);
        self
    }

    pub fn energy_gain(mut self, value: f32) -> Self {
        self.energy_gain = value.max(0.0);
        self
    }

    pub fn drift_gain(mut self, value: f32) -> Self {
        self.drift_gain = value;
        self
    }

    pub fn bias_gain(mut self, value: f32) -> Self {
        self.bias_gain = value;
        self
    }

    pub fn support_gain(mut self, value: f32) -> Self {
        self.support_gain = value.max(0.0);
        self
    }

    pub fn quality_floor(mut self, value: f32) -> Self {
        self.quality_floor = value.max(0.0);
        self
    }

    pub fn stderr_gain(mut self, value: f32) -> Self {
        self.stderr_gain = value.max(0.0);
        self
    }

    pub fn scale(mut self, scale: Option<ZScale>) -> Self {
        self.scale = scale;
        self
    }
}

/// Pushes contextual meaning projections through a Desire Lagrangian gate,
/// emitting Z pulses that can be fed into training loops and automation layers.
#[derive(Clone, Debug)]
pub struct LagrangianGate {
    config: LagrangianGateConfig,
}

impl LagrangianGate {
    pub fn new(config: LagrangianGateConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &LagrangianGateConfig {
        &self.config
    }

    pub fn emit(&self, projection: &MeaningProjection, ts: u64) -> ZPulse {
        let mut pulse = ZPulse::default();
        pulse.source = ZSource::Other("contextual-lagrangian");
        pulse.ts = ts;
        pulse.scale = self.config.scale;

        let support = projection.support.max(1) as f32;
        pulse.tempo = projection
            .dominant_frequency_bin()
            .map(|(bin, _)| (bin as f32 / support) * self.config.tempo_normaliser)
            .unwrap_or(0.0);

        let lexical = projection.lexical_weight().max(0.0);
        let signature = projection.signature.as_ref();
        let boundary = signature.map(|s| s.boundary_edges as f32).unwrap_or(0.0);
        let population = signature
            .map(|s| s.absolute_population_imbalance as f32)
            .unwrap_or(0.0);
        let cluster = signature.map(|s| s.cluster_imbalance as f32).unwrap_or(0.0);

        let orientation = projection.orientation_sign();
        let energy_gate = (boundary + population * 0.5).max(0.0) * self.config.energy_gain;
        let orientation_bias = orientation * lexical * self.config.bias_gain;
        let drift = orientation * energy_gate * self.config.drift_gain;

        let above = (energy_gate * 0.5 + orientation_bias.max(0.0)).max(0.0);
        let beneath = (energy_gate * 0.5 + (-orientation_bias).max(0.0)).max(0.0);
        let central = (lexical + cluster.abs() * 0.1).max(0.0);

        let gate_factor = (1.0 + boundary.max(0.0)).ln_1p().max(1e-3);
        let leading_support = (lexical + boundary * 0.25 + orientation_bias.max(0.0)).max(0.0)
            * self.config.support_gain
            * gate_factor;
        let trailing_support = (lexical + boundary * 0.25 + (-orientation_bias).max(0.0)).max(0.0)
            * self.config.support_gain
            * gate_factor;
        let central_support =
            (lexical + population * 0.25 + cluster.abs() * 0.1).max(0.0) * self.config.support_gain;

        pulse.band_energy = (above, central, beneath);
        pulse.support = ZSupport::new(leading_support, central_support, trailing_support);
        pulse.drift = drift;
        pulse.z_bias = orientation_bias;
        pulse.quality = (self.config.quality_floor + lexical + gate_factor * 0.1).min(1.0);
        pulse.stderr = ((1.0 - lexical).max(0.0) + cluster.abs() * 0.01) * self.config.stderr_gain;
        pulse
    }
}

fn arrangement_signal_values(arrangement: &Arrangement) -> Vec<f32> {
    let mut signal = Vec::with_capacity(arrangement.len());
    for atom in arrangement.placements.iter().copied() {
        let value = match atom {
            PureAtom::A => -1.0,
            PureAtom::B => 1.0,
        };
        signal.push(value);
    }
    signal
}

fn arrangement_spectrum_from_signal(signal: &[f32]) -> PureResult<Tensor> {
    let magnitudes = compute_spectrum(signal);
    Tensor::from_vec(1, magnitudes.len(), magnitudes)
}

fn compute_spectrum(signal: &[f32]) -> Vec<f32> {
    let n = signal.len();
    if n == 0 {
        return Vec::new();
    }
    let mut spectrum = Vec::with_capacity(n);
    let scale = 1.0 / n as f32;
    for k in 0..n {
        let mut accumulator = Complex32::new(0.0, 0.0);
        for (t, &sample) in signal.iter().enumerate() {
            let phase = -TAU * k as f32 * t as f32 / n as f32;
            accumulator += Complex32::from_polar(1.0, phase) * sample;
        }
        spectrum.push(accumulator.norm() * scale);
    }
    spectrum
}

#[cfg(test)]
mod tests {
    use super::*;
    use st_core::theory::zpulse::ZSource;

    #[test]
    fn pure_arrangements_are_hidden() {
        let pure_a = Arrangement::from_line(vec![PureAtom::A; 5]);
        let pure_b = Arrangement::from_line(vec![PureAtom::B; 5]);
        assert_eq!(ContextObserver::observe(&pure_a), Observation::Undetermined);
        assert_eq!(ContextObserver::observe(&pure_b), Observation::Undetermined);
    }

    #[test]
    fn mixed_arrangement_produces_signature() {
        let arrangement = Arrangement::from_line(vec![
            PureAtom::A,
            PureAtom::A,
            PureAtom::B,
            PureAtom::B,
            PureAtom::A,
        ]);
        match ContextObserver::observe(&arrangement) {
            Observation::Signature {
                boundary_edges,
                absolute_population_imbalance,
            } => {
                assert_eq!(boundary_edges, 2);
                assert_eq!(absolute_population_imbalance, 1);
            }
            Observation::Undetermined => panic!("mixed arrangement should not be hidden"),
        }
    }

    #[test]
    fn observation_is_swap_invariant() {
        let arrangement =
            Arrangement::from_line(vec![PureAtom::A, PureAtom::B, PureAtom::A, PureAtom::B]);
        assert!(ContextObserver::is_swap_invariant(&arrangement));
    }

    #[test]
    fn lifting_respects_gauge_choice() {
        let arrangement = Arrangement::from_line(vec![
            PureAtom::A,
            PureAtom::A,
            PureAtom::B,
            PureAtom::B,
            PureAtom::B,
        ]);
        let preserve = lift_to_label(&arrangement, OrientationGauge::Preserve);
        let swap = lift_to_label(&arrangement, OrientationGauge::Swap);
        assert_eq!(preserve, Some(Label::B));
        assert_eq!(swap, Some(Label::A));
    }

    #[test]
    fn lifting_fails_for_balanced_clusters() {
        let arrangement =
            Arrangement::from_line(vec![PureAtom::A, PureAtom::B, PureAtom::B, PureAtom::A]);
        assert_eq!(
            lift_to_label(&arrangement, OrientationGauge::Preserve),
            None
        );
    }

    #[test]
    fn contextual_signature_tracks_cluster_imbalance() {
        let arrangement =
            Arrangement::from_line(vec![PureAtom::A, PureAtom::B, PureAtom::B, PureAtom::B]);
        let signature = ContextObserver::signature(&arrangement).expect("signature");
        assert_eq!(signature.boundary_edges, 1);
        assert_eq!(signature.absolute_population_imbalance, 2);
        assert_eq!(signature.cluster_imbalance, 0);
        assert!(signature.meaning_weight() > 0.0);
    }

    #[test]
    fn meaning_basis_extracts_signal_and_spectrum() {
        let arrangement =
            Arrangement::from_line(vec![PureAtom::A, PureAtom::B, PureAtom::A, PureAtom::B]);
        let basis = MeaningBasis::from_arrangement(&arrangement).expect("basis");
        assert_eq!(basis.signal().data(), &[-1.0, 1.0, -1.0, 1.0]);
        let (bin, magnitude) = basis.dominant_frequency().expect("dominant frequency");
        assert_eq!(bin, 2);
        assert!(magnitude > 0.9);
    }

    #[test]
    fn meaning_projection_combines_label_and_spectrum() {
        let arrangement =
            Arrangement::from_line(vec![PureAtom::A, PureAtom::B, PureAtom::B, PureAtom::B]);
        let projection =
            MeaningProjection::from_arrangement(&arrangement, OrientationGauge::Preserve)
                .expect("projection");
        assert_eq!(projection.label, Some(Label::B));
        assert_eq!(projection.support, 4);
        assert!(projection.lexical_weight() > 0.0);
        assert!(projection.dominant_frequency_bin().is_some());
    }

    #[test]
    fn lagrangian_gate_emits_contextual_pulse() {
        let arrangement = Arrangement::from_line(vec![
            PureAtom::A,
            PureAtom::B,
            PureAtom::B,
            PureAtom::B,
            PureAtom::A,
        ]);
        let projection =
            MeaningProjection::from_arrangement(&arrangement, OrientationGauge::Preserve)
                .expect("projection");
        let gate = LagrangianGate::new(LagrangianGateConfig::default());
        let pulse = gate.emit(&projection, 88);
        assert_eq!(pulse.source, ZSource::Other("contextual-lagrangian"));
        assert_eq!(pulse.ts, 88);
        assert!(pulse.total_energy() > 0.0);
        assert!(pulse.support_mass() > 0.0);
        assert!(pulse.quality >= 0.0);
    }

    #[test]
    fn meaning_emergence_metrics_capture_flux_and_coherence() {
        let arrangement_a =
            Arrangement::from_line(vec![PureAtom::A, PureAtom::B, PureAtom::B, PureAtom::A]);
        let arrangement_b =
            Arrangement::from_line(vec![PureAtom::B, PureAtom::B, PureAtom::A, PureAtom::A]);
        let arrangement_c =
            Arrangement::from_line(vec![PureAtom::A, PureAtom::A, PureAtom::B, PureAtom::B]);

        let proj_a =
            MeaningProjection::from_arrangement(&arrangement_a, OrientationGauge::Preserve)
                .expect("projection a");
        let proj_b =
            MeaningProjection::from_arrangement(&arrangement_b, OrientationGauge::Preserve)
                .expect("projection b");
        let proj_c =
            MeaningProjection::from_arrangement(&arrangement_c, OrientationGauge::Preserve)
                .expect("projection c");

        let metrics =
            MeaningEmergenceMetrics::from_projections(&[proj_a, proj_b, proj_c]).expect("metrics");

        assert!(metrics.lexical_mean >= 0.0);
        assert!(metrics.lexical_std >= 0.0);
        assert!(metrics.support_mean > 0.0);
        assert!(metrics.orientation_flip_rate >= 0.0);
        assert!(metrics.orientation_flip_rate <= 1.0);
        assert!(metrics.frequency_flux >= 0.0);
        assert!(metrics.indeterminate_share >= 0.0);
        assert!(metrics.coherence_score >= 0.0);
        assert!(metrics.coherence_score <= 1.0);
        assert!(metrics.emergence_score >= metrics.orientation_flip_rate);
    }
}
