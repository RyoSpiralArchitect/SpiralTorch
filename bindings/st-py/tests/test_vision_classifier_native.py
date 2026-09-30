"""The wheel's ordinary model factory must reach the Rust ConvNeXt model."""
from __future__ import annotations

import pytest

st = pytest.importorskip("spiraltorch")
pytest.importorskip("spiraltorch.spiraltorch")


def test_convnext_factory_owns_backbone_and_classification_head() -> None:
    model = st.vision_create_classification_model("convnext_tiny", num_classes=3, seed=42)
    metadata = model.metadata()
    assert metadata["name"] == "convnext_tiny"
    assert metadata["num_classes"] == 3
    assert metadata["has_pretrained"] is False
    # The actual default backbone and a three-class head, not SimpleCnn.
    assert model.parameter_count() == 27_888_003
    with pytest.raises(ValueError, match="vision_batch_forward"):
        model.forward([])


def test_legacy_factory_remains_available() -> None:
    model = st.vision_create_classification_model("mobilenet_v3_small", num_classes=3, seed=42)
    assert model.metadata()["num_classes"] == 3
    assert model.parameter_count() is None
