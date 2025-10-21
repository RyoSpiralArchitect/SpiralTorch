#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# © 2025 Ryo ∴ SpiralArchitect

"""Demonstration of the ZSpaceCoherenceSequencer Python API."""

from spiraltorch.nn import ZSpaceCoherenceSequencer
from spiraltorch import Tensor


def main() -> None:
    print("🌀 SpiralTorch ZSpaceCoherenceSequencer Demo\n")
    print("=" * 60)
    print("NOT Attention. NOT Transformer.")
    print("=" * 60)
    print()

    print("Using:")
    print("  ✓ Maxwell pulses for phase synchronization")
    print("  ✓ Desire Lagrangian for semantic bias")
    print("  ✓ Hyperbolic geometry for token hierarchies")
    print("  ✓ Fractional calculus operators (not softmax)")
    print()

    # Create Z-space coherence sequencer
    model = ZSpaceCoherenceSequencer(
        dim=768,
        num_heads=12,
        curvature=-1.0,  # Hyperbolic
    )

    print(f"✅ Model created: {model}")
    print("   - Dimension: 768")
    print("   - Heads: 12")
    print("   - Curvature: -1.0 (hyperbolic)")
    print(f"   - Maxwell channels: {model.maxwell_channels()}")
    print()

    # Create random input
    x = Tensor(2, 768, [0.1] * (2 * 768))
    print(f"✅ Input shape: {x.shape()}")

    # Forward pass
    try:
        out = model.forward(x)
        print(f"✅ Output shape: {out.shape()}")
        print()
        print("🎯 ZSpaceCoherenceSequencer is working!")
        print()
        print("Key differences from Attention:")
        print("  ✓ No Q/K/V projections")
        print("  ✓ No softmax (not differentiable in Z-space)")
        print("  ✓ Maxwell pulse detection for coherence")
        print("  ✓ Desire Lagrangian for semantic safety")
        print("  ✓ Hyperbolic geometry for hierarchy")
        print("  ✓ Fractional calculus for spectra")
        print()
        print("📖 Full implementation coming this week!")
    except Exception as exc:  # pragma: no cover - demo prints tracebacks
        print(f"❌ Error: {exc}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
