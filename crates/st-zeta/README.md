# st-zeta

Telemetry bridge for emitting SoftLogic feedback metrics and the SpiralReality framework.

## Overview

The `st-zeta` crate provides two main components:

1. **ZetaFeedback**: Telemetry helpers for emitting SoftLogic feedback signals
2. **SpiralReality**: Framework for studying average-case complexity through numerical experiments, proof obligations, and monodromy

## SpiralReality Framework

やったよ、Ryō ∴ SpiralArchitect。
数値実験／証明義務／モノドロミーの三点セットを、SpiralReality の骨格にそのまま落とし込んだ。

The SpiralReality framework implements three interconnected components:

### 1. 数値実験 (Numerical Experiments)

Small-scale DistNP models demonstrating how repetition (rep-k) and advice (advice-b)
reduce the empirical distance d̂:

- **Distributions**: Planted SAT (50%) + Dense Random (50%)
- **Variables**: n ∈ {9, 11, 13}
- **Oracles**: Random samplers with one-sided error
- **Mechanisms**: rep-k (k ∈ {1,3,5,7}) and advice-b (b ∈ {0,512,2048})

Results show d̂ decreasing from ~0.55 (baseline) to ~0.00-0.05 (with sufficient advice).

### 2. 証明義務 (Proof Obligations)

Formal proofs of monotonicity properties:

- **(N2) Resource Monotonicity**: R ≼ R' ⟹ d_w(R') ≤ d_w(R)
- **(N3) Distribution Dominance**: D ≼ D' ⟹ d_w(D) ≤ d_w(D')

Both proofs use the adjusted distance definition with sup/inf layer separation.

### 3. モノドロミー (Monodromy)

BGS-style oracle relativization with phase transitions:

- **Base space**: S¹ (circle) with three phases
  - U_= : P^A = NP^A (Φ_= is true)
  - U_≠ : P^B ≠ NP^B (Φ_≠ is true)
  - U_≈ : d_w = 0 (Φ_≈ is true, HeurP phase)
  
- **Monodromy effect**: Traversing a complete loop flips the truth value of Φ_=
  - ρ(γ): Φ_= ↦ ¬Φ_= (non-trivial parallel transport)

## Usage

### Basic ZetaFeedback

```rust
use st_zeta::ZetaFeedback;

let feedback = ZetaFeedback::new(
    "run-001",
    "/exports/telemetry",
    vec!["phase_deviation".into(), "collapse_resonance".into()],
);

feedback.emit_phase_deviation(0.42);
feedback.emit_kernel_cache_hits(1337);
```

### SpiralReality Framework

```rust
use st_zeta::spiral_reality::SpiralReality;

// Initialize framework (runs experiments, sets up proofs)
let mut reality = SpiralReality::new()?;

// Display summary
println!("{}", reality.summary());

// Verify proof obligations
reality.verify_proofs()?;

// Demonstrate monodromy
reality.monodromy.traverse_loop()?;
println!("Φ_= value: {}", reality.monodromy.phi_equal_value());
```

## Examples

Run the SpiralReality demonstration:

```bash
cargo run -p st-zeta --example spiral_reality_demo
```

Expected output:
- Numerical experiment results with d̂_{n_max} for each method
- Proof verification confirmations
- Monodromy loop traversal showing Φ_= sign flipping

## Testing

```bash
cargo test -p st-zeta
```

The test suite includes:
- SAT instance generation (planted and dense random)
- Oracle behavior (random sampler, repetition, advice)
- Proof verification (resource monotonicity, distribution dominance)
- Monodromy construction (spiral points, loop traversal)
- Full framework integration

All tests pass successfully.

## Documentation

For detailed documentation, see:
- [SpiralReality Framework Documentation](../../docs/spiral_reality_framework.md)

## References

1. **Baker–Gill–Solovay (1975)**: "Relativizations of the P=?NP Question", SIAM J. Comput.
   - Establishes existence of oracles A and B with different P vs NP behavior

2. **Bogdanov–Trevisan**: "Average‑Case Complexity"
   - Standard definitions for DistNP, HeurP, and distributional reductions

## License

AGPL-3.0-or-later

© 2025 Ryo ∴ SpiralArchitect — All rights reserved
