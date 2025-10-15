// ============================================================================
//  SpiralReality Proprietary
//  Copyright (c) 2025 SpiralReality. All Rights Reserved.
//
//  NOTICE: This file contains confidential and proprietary information of
//  SpiralReality. ANY USE, COPYING, MODIFICATION, DISTRIBUTION, DISPLAY,
//  OR DISCLOSURE OF THIS FILE, IN WHOLE OR IN PART, IS STRICTLY PROHIBITED
//  WITHOUT THE PRIOR WRITTEN CONSENT OF SPIRALREALITY.
//
//  NO LICENSE IS GRANTED OR IMPLIED BY THIS FILE. THIS SOFTWARE IS PROVIDED
//  "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
//  NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
//  PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL SPIRALREALITY OR ITS
//  SUPPLIERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
//  AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
//  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// ============================================================================

//! Observation DAG accounting utilities.
//!
//! The constructions in this module follow the category-theoretic view of
//! observation trees as the final coalgebra of the endofunctor
//! `F(X) = R × Orb_{G_Λ}(X^b)`, where `R` is the observable root alphabet and
//! `Orb_{G_Λ}(X^b)` denotes G-acted child-slot orbits.  Colour actions are
//! modelled explicitly so you can assess whether singleton colours remain
//! observable once a symmetry (e.g. `S_q`, `C_q`, `D_q`) is imposed.  The module
//! provides a compact description of the Pólya upper bound that SpiralTorch
//! uses when benchmarking observation compression experiments.

use core::fmt;

/// Symmetry acting on child slots of a branching process.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotSymmetry {
    /// The symmetric group S_b (unordered slots).
    Symmetric,
    /// The cyclic group C_b (rotations).
    Cyclic,
    /// The dihedral group D_b (rotations + reflections).
    Dihedral,
}

impl SlotSymmetry {
    /// Returns the number of G_Λ-orbits for the provided child signature count.
    pub fn polya(self, child_signatures: u128, branching: u32) -> u128 {
        match self {
            SlotSymmetry::Symmetric => polya_symmetric(child_signatures, branching),
            SlotSymmetry::Cyclic => polya_cyclic(child_signatures, branching),
            SlotSymmetry::Dihedral => polya_dihedral(child_signatures, branching),
        }
    }

    /// A short human readable label used in diagnostics.
    pub fn label(self) -> &'static str {
        match self {
            SlotSymmetry::Symmetric => "S_b",
            SlotSymmetry::Cyclic => "C_b",
            SlotSymmetry::Dihedral => "D_b",
        }
    }
}

/// Symmetry acting on an auxiliary colour palette.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorSymmetry {
    /// No identification between colours.
    Trivial,
    /// Full symmetric group `S_q`.
    Symmetric,
    /// Cyclic group `C_q` (rotations on a colour ring).
    Cyclic,
    /// Dihedral group `D_q` (rotations + reflections on the colour ring).
    Dihedral,
    /// Custom partition provided by the caller.
    Custom {
        /// Number of observable orbits supplied directly by the user.
        orbits: u32,
        /// Whether a singleton orbit ("pure" colour) survives the symmetry.
        has_singleton: bool,
    },
}

impl ColorSymmetry {
    /// Returns the number of observable colour classes after quotienting by the symmetry.
    pub fn orbit_cardinality(self, palette_size: u32) -> u128 {
        match self {
            ColorSymmetry::Trivial => u128::from(palette_size),
            ColorSymmetry::Symmetric | ColorSymmetry::Cyclic | ColorSymmetry::Dihedral => {
                if palette_size == 0 {
                    0
                } else {
                    1
                }
            }
            ColorSymmetry::Custom { orbits, .. } => {
                if palette_size == 0 {
                    0
                } else {
                    u128::from(orbits.max(1))
                }
            }
        }
    }

    /// Returns `true` when the symmetry preserves singleton colours as measurable sets.
    pub fn preserves_singletons(self, palette_size: u32) -> bool {
        match self {
            ColorSymmetry::Trivial => palette_size > 0,
            ColorSymmetry::Symmetric | ColorSymmetry::Cyclic | ColorSymmetry::Dihedral => {
                palette_size <= 1
            }
            ColorSymmetry::Custom { has_singleton, .. } => has_singleton,
        }
    }
}

/// Reason explaining why a singleton colour becomes invisible to the observation σ-algebra.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorInvisibility {
    /// There are no colours to observe.
    EmptyPalette,
    /// The symmetry identifies the colours, preventing singleton observation.
    IdentifiedBySymmetry,
}

/// Describes how colour information participates in the observable root alphabet.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColorAction {
    palette_size: u32,
    symmetry: ColorSymmetry,
}

impl ColorAction {
    /// Constructs a new colour action.
    pub fn new(palette_size: u32, symmetry: ColorSymmetry) -> Self {
        Self {
            palette_size,
            symmetry,
        }
    }

    /// Returns the size of the palette prior to quotienting.
    pub fn palette_size(&self) -> u32 {
        self.palette_size
    }

    /// Returns the acting symmetry.
    pub fn symmetry(&self) -> ColorSymmetry {
        self.symmetry
    }

    /// Returns the number of observable colour classes after quotienting by the symmetry.
    pub fn observable_classes(&self) -> u128 {
        self.symmetry.orbit_cardinality(self.palette_size)
    }

    /// Determines whether a singleton colour ("pure a") is observable.
    pub fn singleton_observable(&self) -> Result<bool, ColorInvisibility> {
        if self.palette_size == 0 {
            return Err(ColorInvisibility::EmptyPalette);
        }
        Ok(self.symmetry.preserves_singletons(self.palette_size))
    }
}

/// Configuration describing a freely branching observation generator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObservabilityConfig {
    /// Number of distinct observable root labels.
    pub root_cardinality: u128,
    /// Number of child slots attached to each node.
    pub branching_factor: u32,
    /// Symmetry acting on the child slots.
    pub slot_symmetry: SlotSymmetry,
    /// Optional colour action bundled with each root label.
    pub color_action: Option<ColorAction>,
}

impl ObservabilityConfig {
    /// Constructs a new configuration. Panics if branching is zero.
    pub fn new(root_cardinality: u128, branching_factor: u32, slot_symmetry: SlotSymmetry) -> Self {
        assert!(branching_factor > 0, "branching factor must be positive");
        Self {
            root_cardinality,
            branching_factor,
            slot_symmetry,
            color_action: None,
        }
    }

    /// Adds a colour action to the configuration.
    pub fn with_color_action(mut self, action: ColorAction) -> Self {
        self.color_action = Some(action);
        self
    }

    /// Computes the free-branching successor cardinality given the current
    /// unique signature count.
    fn advance(&self, current_signatures: u128) -> u128 {
        self.observable_root_cardinality().saturating_mul(
            self.slot_symmetry
                .polya(current_signatures, self.branching_factor),
        )
    }

    /// Effective number of observable root labels after quotienting by colour symmetry.
    fn observable_root_cardinality(&self) -> u128 {
        match &self.color_action {
            Some(action) => {
                let classes = action.observable_classes();
                if classes == 0 {
                    self.root_cardinality
                } else {
                    self.root_cardinality.saturating_mul(classes)
                }
            }
            None => self.root_cardinality,
        }
    }
}

impl fmt::Display for ObservabilityConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "r={} b={} G={}",
            self.root_cardinality,
            self.branching_factor,
            self.slot_symmetry.label()
        )
    }
}

/// Iterator-style coalgebra that unfolds the theoretical bounds depth by depth.
#[derive(Clone, Debug)]
pub struct ObservationalCoalgebra {
    config: ObservabilityConfig,
    cache: Vec<u128>,
}

impl ObservationalCoalgebra {
    /// Creates a coalgebra seeded with depth-zero unique signatures.
    pub fn new(config: ObservabilityConfig) -> Self {
        Self {
            cache: vec![config.observable_root_cardinality()],
            config,
        }
    }

    /// Returns the cached unique signature count for the provided depth,
    /// advancing the coalgebra if necessary.
    pub fn unique_signatures(&mut self, depth: usize) -> u128 {
        if depth >= self.cache.len() {
            let mut current = *self.cache.last().expect("non-empty cache");
            while depth >= self.cache.len() {
                current = self.config.advance(current);
                self.cache.push(current);
            }
        }
        self.cache[depth]
    }

    /// Produces the free-branching sequence up to the requested depth (inclusive).
    pub fn unfold(&mut self, depth: usize) -> Vec<u128> {
        self.unique_signatures(depth);
        self.cache[..=depth].to_vec()
    }

    /// Assesses empirical unique counts against the theoretical upper bound and
    /// produces the per-depth efficiency ratios `η = observed / bound`.
    pub fn assess(&mut self, observed: &[u128]) -> ObservabilityAssessment {
        if observed.is_empty() {
            return ObservabilityAssessment {
                expected: Vec::new(),
                efficiency: Vec::new(),
            };
        }
        let depth = observed.len() - 1;
        let expected = self.unfold(depth);
        let mut efficiency = Vec::with_capacity(observed.len());
        for (obs, bound) in observed.iter().zip(expected.iter()) {
            let eta = if *bound == 0 {
                1.0
            } else {
                (*obs as f64) / (*bound as f64)
            };
            efficiency.push(eta.min(1.0).max(0.0));
        }
        ObservabilityAssessment {
            expected,
            efficiency,
        }
    }
}

/// Comparison between measured and theoretical unique signature counts.
#[derive(Clone, Debug, PartialEq)]
pub struct ObservabilityAssessment {
    /// Theoretical free-branching counts returned by the coalgebra.
    pub expected: Vec<u128>,
    /// Efficiency per depth (`η = observed / expected`).
    pub efficiency: Vec<f64>,
}

fn polya_symmetric(child_signatures: u128, branching: u32) -> u128 {
    let n = child_signatures + u128::from(branching) - 1;
    binomial(n, u128::from(branching))
}

fn polya_cyclic(child_signatures: u128, branching: u32) -> u128 {
    let mut acc = 0u128;
    for j in 0..branching {
        let g = gcd_u32(branching, j);
        acc = acc.saturating_add(child_signatures.saturating_pow(g));
    }
    acc / u128::from(branching)
}

fn polya_dihedral(child_signatures: u128, branching: u32) -> u128 {
    let mut base_sum = 0u128;
    for j in 0..branching {
        let g = gcd_u32(branching, j);
        base_sum = base_sum.saturating_add(child_signatures.saturating_pow(g));
    }
    let additional = if branching % 2 == 0 {
        // even: include two reflection families
        let half = branching / 2;
        let term_a = child_signatures.saturating_pow(half + 1);
        let term_b = child_signatures.saturating_pow(half);
        u128::from(half) * (term_a + term_b)
    } else {
        // odd: a single family with (b+1)/2 cycles
        let power = (branching + 1) / 2;
        u128::from(branching) * child_signatures.saturating_pow(power)
    };
    (base_sum + additional) / (2 * u128::from(branching))
}

fn binomial(n: u128, k: u128) -> u128 {
    if k == 0 || k == n {
        return 1;
    }
    let limit = k.min(n - k);
    let mut result = 1u128;
    let mut i = 0u128;
    while i < limit {
        let numerator = n - i;
        let denominator = i + 1;
        result = result
            .saturating_mul(numerator)
            .checked_div(denominator)
            .expect("binomial denominator should divide numerator");
        i += 1;
    }
    result
}

fn gcd_u32(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let tmp = a % b;
        a = b;
        b = tmp;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetric_branching_matches_polya_recursion() {
        let mut coalgebra =
            ObservationalCoalgebra::new(ObservabilityConfig::new(2, 2, SlotSymmetry::Symmetric));
        assert_eq!(coalgebra.unique_signatures(0), 2);
        assert_eq!(coalgebra.unique_signatures(1), 6);
        assert_eq!(coalgebra.unique_signatures(2), 42);
    }

    #[test]
    fn cyclic_branching_counts_align_with_formula() {
        let mut coalgebra =
            ObservationalCoalgebra::new(ObservabilityConfig::new(2, 4, SlotSymmetry::Cyclic));
        let first = coalgebra.unique_signatures(1);
        assert_eq!(first, 2 * SlotSymmetry::Cyclic.polya(2, 4));
        let second = coalgebra.unique_signatures(2);
        let expected_second = 2 * SlotSymmetry::Cyclic.polya(first, 4);
        assert_eq!(second, expected_second);
    }

    #[test]
    fn dihedral_branching_even_and_odd_match() {
        let mut odd =
            ObservationalCoalgebra::new(ObservabilityConfig::new(2, 3, SlotSymmetry::Dihedral));
        let odd_first = odd.unique_signatures(1);
        assert_eq!(odd_first, 2 * SlotSymmetry::Dihedral.polya(2, 3));

        let mut even =
            ObservationalCoalgebra::new(ObservabilityConfig::new(2, 4, SlotSymmetry::Dihedral));
        let even_first = even.unique_signatures(1);
        assert_eq!(even_first, 2 * SlotSymmetry::Dihedral.polya(2, 4));
    }

    #[test]
    fn efficiency_reflects_measured_counts() {
        let mut coalgebra =
            ObservationalCoalgebra::new(ObservabilityConfig::new(2, 2, SlotSymmetry::Symmetric));
        let observed = vec![2, 5, 30];
        let assessment = coalgebra.assess(&observed);
        assert_eq!(assessment.expected, vec![2, 6, 42]);
        assert!(assessment.efficiency[0] - 1.0 < f64::EPSILON);
        assert!(assessment.efficiency[1] < 1.0);
        assert!(assessment.efficiency[2] < 1.0);
    }

    #[test]
    fn color_action_limits_singletons_under_symmetry() {
        let action = ColorAction::new(2, ColorSymmetry::Symmetric);
        let verdict = action.singleton_observable().unwrap();
        assert!(!verdict);

        let trivial = ColorAction::new(2, ColorSymmetry::Trivial);
        assert!(trivial.singleton_observable().unwrap());
    }

    #[test]
    fn color_action_adjusts_root_cardinality() {
        let config = ObservabilityConfig::new(3, 2, SlotSymmetry::Symmetric)
            .with_color_action(ColorAction::new(2, ColorSymmetry::Symmetric));
        let mut coalgebra = ObservationalCoalgebra::new(config);
        // Symmetric colour action collapses the palette to a single observable class.
        assert_eq!(coalgebra.unique_signatures(0), 3);

        let config_distinguishing = ObservabilityConfig::new(3, 2, SlotSymmetry::Symmetric)
            .with_color_action(ColorAction::new(2, ColorSymmetry::Trivial));
        let mut coalgebra_distinguishing = ObservationalCoalgebra::new(config_distinguishing);
        assert_eq!(coalgebra_distinguishing.unique_signatures(0), 6);
    }
}
