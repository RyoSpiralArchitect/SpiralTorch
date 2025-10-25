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

use std::collections::VecDeque;

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
