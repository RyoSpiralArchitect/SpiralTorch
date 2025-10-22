#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# © 2025 Ryo ∴ SpiralArchitect

import spiraltorch as st
from spiraltorch.nn import CoherenceDiagnostics, ZSpaceCoherenceSequencer
from spiraltorch import Tensor

print("🌀 SpiralTorch ZSpaceCoherenceSequencer Demo\n")
print("=" * 60)
print("NOT Attention. NOT Transformer.")
print("=" * 60)
print()

print("Using:")
print("  ✓ Maxwell pulses for phase synchronization")
print("  ✓ Desire Lagrangian for linguistic bias")
print("  ✓ Hyperbolic geometry for token hierarchies")
print("  ✓ Fractional calculus operators (not softmax)")
print()

# Create Z-space coherence sequencer
model = ZSpaceCoherenceSequencer(
    dim=768,
    num_heads=12,
    curvature=-1.0  # Hyperbolic
)

print(f"✅ Model created: {model}")
print(f"   - Dimension: 768")
print(f"   - Heads: 12")
print(f"   - Curvature: -1.0 (hyperbolic)")
print()

# Create random input
x = Tensor(2, 768, [0.1] * (2 * 768))
print(f"✅ Input shape: {x.shape()}")

# Forward pass
try:
    out, coherence, diagnostics = model.forward_with_diagnostics(x)
    print(f"✅ Output shape: {out.shape()}")

    print("✅ Coherence diagnostics:")
    dominant = diagnostics.dominant_channel()
    dominant_str = f"ch{dominant:02d}" if dominant is not None else "—"
    print(f"   - Mean coherence: {diagnostics.mean_coherence():.4f}")
    print(f"   - Entropy: {diagnostics.coherence_entropy():.4f}")
    print(f"   - Dominant channel: {dominant_str}")
    print(f"   - Energy ratio: {diagnostics.energy_ratio():.3f}")
    print(f"   - Z-bias: {diagnostics.z_bias():.3f}")
    print(f"   - Maxwell channels: {len(coherence)}")

    contour = model.emit_linguistic_contour(x)
    print("✅ Linguistic contour:")
    print(f"   - Prosody index: {contour.prosody_index():.3f}")
    print(f"   - Coherence strength: {contour.coherence_strength():.3f}")
    print(f"   - Articulation bias: {contour.articulation_bias():.3f}")
    print()
    reports = model.describe_channels(x)
    print("✅ Channel reports (first 5):")
    for report in reports[:5]:
        concept = report.dominant_concept()
        concept_label = concept.label() if concept else "baseline"
        descriptor = report.descriptor() or "—"
        print(
            f"   - ch{report.channel():02d}: weight={report.weight():.4f}"
            f" concept={concept_label} emphasis={report.emphasis():.2f}"
            f" descriptor={descriptor} backend={report.backend().label()}"
        )
    print()
    print("🎯 ZSpaceCoherenceSequencer is working!")
    print()
    print("Key differences from Attention:")
    print("  ✓ No Q/K/V projections")
    print("  ✓ No softmax (not differentiable in Z-space)")
    print("  ✓ Maxwell pulse detection for coherence")
    print("  ✓ Desire Lagrangian for linguistic safety")
    print("  ✓ Hyperbolic geometry for hierarchy")
    print("  ✓ Fractional calculus for spectra")
    print()
    print("📖 Full implementation coming this week!")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
