#!/usr/bin/env -S python3 -s
# SPDX-License-Identifier: AGPL-3.0-or-later
# © 2025 Ryo ∴ SpiralArchitect

"""ZSpaceCoherenceSequencer demo.

Run from a source checkout:
`python3 -s examples/05_new_layers/zspace_coherence_demo.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import spiraltorch as st
except ModuleNotFoundError:
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    import spiraltorch as st

from spiraltorch.nn import ZSpaceCoherenceSequencer

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

# Create deterministic input
x = st.Tensor.rand(2, 768, seed=1)
print(f"✅ Input shape: {x.shape()}")

# Forward pass
try:
    out, coherence, diagnostics = model.forward_with_diagnostics(x)
    print(f"✅ Output shape: {out.shape()}")

    print("✅ Coherence diagnostics:")
    print(f"   - Observation label: {diagnostics.observation.label}")
    print(f"   - Preserved channels: {diagnostics.preserved_channels}")
    print(f"   - Discarded channels: {diagnostics.discarded_channels}")
    print(f"   - Maxwell channels: {len(coherence)}")

    signature = diagnostics.observation.signature
    if signature is not None:
        dominant = signature.dominant_channel
        dominant_str = f"ch{dominant:02d}" if dominant is not None else "—"
        print(f"   - Mean coherence: {signature.mean_coherence:.6f}")
        print(f"   - Entropy: {signature.entropy:.4f}")
        print(f"   - Dominant channel: {dominant_str}")
        print(f"   - Energy ratio: {signature.energy_ratio:.6f}")
        print(f"   - Swap invariant: {signature.swap_invariant}")
    else:
        print("   - Signature: —")
    print()
    print("✅ Channel reports (first 5):")
    for report in diagnostics.channel_reports[:5]:
        concept_label = report.dominant_concept or "baseline"
        descriptor = report.descriptor or "—"
        print(
            f"   - ch{report.channel:02d}: weight={report.weight:.4f}"
            f" concept={concept_label} emphasis={report.emphasis:.2f}"
            f" descriptor={descriptor} backend={report.backend}"
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
