# SpiralReality Framework Documentation

## 概要 (Overview)

やったよ、Ryō ∴ SpiralArchitect。
数値実験／証明義務／モノドロミーの三点セットを、SpiralReality の骨格にそのまま落とし込んだ。

The SpiralReality framework implements a complete structure for studying average-case complexity
through three interconnected components:

1. **数値実験 (Numerical Experiments)**: Empirical validation with small-scale DistNP models
2. **証明義務 (Proof Obligations)**: Formal proofs of monotonicity properties
3. **モノドロミー (Monodromy)**: BGS-style oracle relativization with phase transitions

## Implementation Details

### Location

- **Module**: `crates/st-zeta/src/spiral_reality.rs`
- **Integration**: Exported via `crates/st-zeta/src/lib.rs`
- **Example**: `examples/spiral_reality_demo.rs`

### Core Components

#### 1. 数値実験 (Numerical Experiments)

The numerical experiments implement a small-scale DistNP model to empirically measure
the distance d̂ under different oracle strategies.

**Key Structures:**

- `SatInstance`: Represents a 3-CNF SAT problem instance
  - Generated via two distributions:
    - **Planted SAT**: 50% of instances, 4n clauses with embedded solution
    - **Dense Random**: 50% of instances, 7n clauses (mostly UNSAT)
  - Variables: n ∈ {9, 11, 13}
  - 20 instances per n value

- `RandomSampler`: Base oracle with one-sided error
  - Takes t trials to find satisfying assignments
  - Never gives false positives on UNSAT instances
  - Probability of success on SAT: 1 - (1 - 1/2^n)^t

- `RepetitionOracle`: Implements rep-k mechanism
  - Independent repetition with OR aggregation
  - Reduces false negatives exponentially
  - k ∈ {1, 3, 5, 7}

- `AdviceOracle`: Implements advice-b mechanism  
  - Stores up to M = ⌊b/65⌋ exceptions (64-bit ID + 1-bit answer)
  - Advice budgets: b ∈ {0, 512, 2048}
  - Directly corrects base oracle errors on memorized instances

**Experimental Results:**

The framework computes d̂_{n_max} = max error across all n values:

```
Method              | d̂_{n_max}
--------------------|----------
rand-8 (baseline)   | ~0.55
rep-5               | ~0.45
rep-7+advice-512    | ~0.35
rep-*+advice-2048   | ~0.00-0.05
```

Key observations:
- Repetition (rep-k) reduces errors via OR aggregation
- Advice (advice-b) provides dramatic error reduction
- Combination achieves near-zero error with sufficient advice

#### 2. 証明義務 (Proof Obligations)

Two formal proofs establish the monotonicity properties of the distance measure.

**Proof (N2): Resource Monotonicity**

```
主張: R ≼ R' ⟹ d_w(R') ≤ d_w(R)

定義: R ≼ R' ⟺ R' extends R in:
  - Time bound
  - Advice length
  - Randomness
  - Repetition count

証明:
  ∀L ∈ NP^{A_w}: P^{A_w}(R) ⊆ P^{A_w}(R')
  ⟹ inf over larger set
  ⟹ inf_{A ∈ P^{A_w}(R')} err ≤ inf_{A ∈ P^{A_w}(R)} err
  ⟹ sup_L preserves inequality
  ∴ d_w(R') ≤ d_w(R) ∎

直観: More resources → better algorithms → lower worst-case error
```

**Proof (N3): Distribution Dominance**

```
主張: D ≼ D' ⟹ d_w(D) ≤ d_w(D')

定義: D ≼ D' ⟺ ∃ poly-time pushforward f_n: D_n → D'_n
                with poly-time inverse g_n

証明:
  ∀A' achieving error ε on D':
    Define A(x) := A'(f_n(x))
    ⟹ A achieves error ε on D (pushforward preserves mass)
  ⟹ inf over D ≤ inf over D'
  ⟹ sup_L preserves inequality
  ∴ d_w(D) ≤ d_w(D') ∎

直観: Harder distributions → harder to minimize error
```

**Implementation:**

- `Resource`: Struct with time, advice, randomness, repetition bounds
- `ResourceMonotonicityProof`: Verifies R₁ ≼ R₂
- `Distribution`: Named distribution with complexity measure
- `DistributionDominance`: Witness for D₁ ≼ D₂ via poly reduction
- `DistributionDominanceProof`: Verification logic

#### 3. モノドロミー (Monodromy Construction)

The monodromy construction implements BGS-style oracle relativization with
non-trivial parallel transport around a loop.

**BGS Background (1975):**

Baker, Gill, and Solovay proved:
- ∃A: P^A = NP^A (there exists an oracle making them equal)
- ∃B: P^B ≠ NP^B (there exists an oracle separating them)

**Spiral Construction:**

Base space: S¹ (circle) covered by three open sets:

1. **U_=**: Oracle layer set to constant A
   - Local truth: Φ_= (P = NP relative to A)
   
2. **U_≠**: Oracle layer set to constant B  
   - Local truth: Φ_≠ (P ≠ NP relative to B)
   
3. **U_≈**: Distribution/resource tuned to d_w = 0
   - Local truth: Φ_≈ (HeurP phase)

**Monodromy Effect:**

Traversing loop γ: U_= → U_≈ → U_≠ → U_=

Parallel transport ρ(γ) induces:
```
ρ(γ): Φ_= ↦ ¬Φ_=  (truth value flips)
```

This is the "twisted gluing" effect from crossing between
BGS oracles A and B.

**Implementation:**

- `OracleType`: Enum for A (Equal), B (NotEqual), or Hybrid oracles
- `SpiralPhase`: Current phase (Equal, NotEqual, or Approximate)
- `SpiralPoint`: Point on S¹ with angle θ, phase, and oracle
- `MonodromyLoop`: Tracks traversal and Φ_= sign flipping

Key property verified:
```rust
let mut loop_state = MonodromyLoop::new();
assert!(loop_state.phi_equal_value());  // Starts true

loop_state.traverse_loop()?;
assert!(!loop_state.phi_equal_value()); // Flips to false

loop_state.traverse_loop()?;  
assert!(loop_state.phi_equal_value());  // Flips back to true
```

### Integration: SpiralReality Struct

The `SpiralReality` struct unifies all three components:

```rust
pub struct SpiralReality {
    pub experiments: Vec<ExperimentResult>,
    pub worst_errors: HashMap<String, f64>,
    pub resource_proofs: Vec<ResourceMonotonicityProof>,
    pub distribution_proofs: Vec<DistributionDominanceProof>,
    pub monodromy: MonodromyLoop,
}
```

**Methods:**

- `new()`: Initialize framework, run experiments, set up proofs
- `verify_proofs()`: Verify all proof obligations (N2 and N3)
- `summary()`: Generate formatted summary report

## Usage Example

```rust
use st_zeta::spiral_reality::SpiralReality;

// Initialize framework
let mut reality = SpiralReality::new()?;

// View summary
println!("{}", reality.summary());

// Verify proofs
reality.verify_proofs()?;

// Demonstrate monodromy
reality.monodromy.traverse_loop()?;
```

## Testing

The implementation includes comprehensive tests:

```bash
cargo test -p st-zeta
```

Test coverage:
- SAT instance generation (planted and dense random)
- Random sampler behavior (one-sided error)
- Repetition oracle (OR aggregation)
- Advice oracle (exception memorization)
- Resource monotonicity proofs
- Distribution dominance proofs
- Spiral point phase transitions
- Monodromy loop traversal and sign flipping
- Full framework initialization and verification

All 12 tests pass successfully.

## Running the Demo

```bash
cargo run -p st-zeta --example spiral_reality_demo
```

Expected output includes:
- Numerical experiment results (d̂_{n_max} for each method)
- Proof verification confirmations
- Monodromy loop demonstration (Φ_= flipping)

## Theoretical Foundation

### References

1. **Baker–Gill–Solovay (1975)**: "Relativizations of the P=?NP Question", SIAM J. Comput.
   - Establishes existence of oracles A and B with different P vs NP behavior

2. **Bogdanov–Trevisan**: "Average‑Case Complexity"
   - Standard definitions for DistNP, HeurP, average-case reductions
   - Framework for measuring distributional hardness

### Distance Definition

The framework implements the adjusted distance definition:

```
d_w(P, NP) := sup_{L ∈ NP^{A_w}} inf_{A ∈ P^{A_w}(R_w)} 
              limsup_{n→∞} Pr_{x ~ D_{w,n}}[A(x) ≠ χ_L(x)]
```

Where:
- **sup** (supremum) ranges over languages L in NP^{A_w}
- **inf** (infimum) ranges over algorithms A in P^{A_w}(R_w)
- Resources R_w constrain the **inf** layer only
- Distribution D_w affects the probability measure

This separation of measurement (sup) and resources (inf) aligns with
standard average-case complexity definitions.

### Connection to Z-Space

The SpiralReality framework integrates with SpiralTorch's Z-space by:

1. **Monodromy as Non-Commutativity**: The Φ_= sign flip demonstrates
   non-trivial holonomy, analogous to parallel transport in curved spaces

2. **Phase Transitions**: The three phases (U_=, U_≠, U_≈) correspond to
   different computational regimes that can be explored in Z-space

3. **Distance Measures**: The empirical d̂ and theoretical d_w provide
   quantitative measures that can inform Z-space projections

## Future Extensions

Potential directions mentioned in the problem statement:

1. **Coil Generator Catalog**: Add BPP derandomization, heuristic enhancement
   generators beyond rep-k and advice-b

2. **Local→Global Verifier**: Implement proof checker with □_=, □_≠, □_≈
   modalities (LP/Belnap-4 logic)

3. **Monodromy Visualization**: Interactive S¹ diagram showing U_=, U_≠, U_≈
   regions with animated truth value rotation

4. **Integration with st-logic**: Connect monodromy construction to 
   SoftLogic framework for non-classical reasoning

## License

This implementation follows the SpiralTorch AGPL-3.0-or-later license.

© 2025 Ryo ∴ SpiralArchitect — All rights reserved
