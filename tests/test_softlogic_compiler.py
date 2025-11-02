"""Tests for the Python SpiralK compiler shim."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_compiler_module():
    path = Path(__file__).resolve().parent.parent / "spiraltorch" / "softlogic" / "compiler.py"
    spec = importlib.util.spec_from_file_location("spiraltorch_softlogic_compiler", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError("unable to load spiraltorch.softlogic.compiler")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


compiler = _load_compiler_module()
Backend = compiler.Backend
Layout = compiler.Layout
Precision = compiler.Precision


SAMPLE = """
refract main {
  target: graph:ZSpaceEncoder
  precision: bf16
  layout: nhwc
  schedule: cooperative
  backend: WGPU
  op: attention -> fuse_softmax, stable_grad
}

sync merge_01 {
  pairs: [Σeve, Raqel]
  tolerance: 0.06
}

feedback z01143A {
  export: "runs/01143A"
  metrics: [phase_deviation, collapse_resonance, kernel_cache_hits]
}
"""


def test_compile_spiralk_parses_reference_document():
    doc = compiler.compile_spiralk(SAMPLE)
    assert len(doc.refracts) == 1
    refract = doc.refracts[0]
    assert refract.name == "main"
    assert refract.target.kind == "graph"
    assert refract.target.name == "ZSpaceEncoder"
    assert refract.precision is Precision.BF16
    assert refract.layout is Layout.NHWC
    assert refract.backend is Backend.WGPU
    assert refract.policies[0].flags == ["fuse_softmax", "stable_grad"]

    assert len(doc.syncs) == 1
    assert doc.syncs[0].pairs == ["Σeve", "Raqel"]
    assert doc.syncs[0].tolerance == pytest.approx(0.06)

    assert len(doc.feedbacks) == 1
    assert doc.feedbacks[0].export_path == "runs/01143A"


def test_compile_spiralk_falls_back_when_native_fails(monkeypatch: pytest.MonkeyPatch):
    original = compiler._NATIVE_PARSE

    def _boom(_: str):  # pragma: no cover - ensures fallback path is exercised
        raise RuntimeError("boom")

    monkeypatch.setattr(compiler, "_NATIVE_PARSE", _boom)
    try:
        doc = compiler.compile_spiralk("refract demo { target: graph:Demo }")
        assert len(doc.refracts) == 1
        assert doc.refracts[0].target.name == "Demo"
    finally:
        monkeypatch.setattr(compiler, "_NATIVE_PARSE", original)
