from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT / "bindings" / "st-py" / "python"))

try:
    from spiral.export import DeploymentTarget, ExportConfig, ExportPipeline, load_benchmark_report
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    pytest.skip(f"spiraltorch native module unavailable: {exc}", allow_module_level=True)


def test_export_pipeline(tmp_path: Path) -> None:
    weights = [((i * 5) % 23) / 29.0 for i in range(96)]
    config = ExportConfig(
        quantization_bit_width=8,
        pruning_block_size=8,
        target_sparsity=0.5,
        latency_budget_ms=8.0,
        name="unit",
    )

    pipeline = ExportPipeline(weights, config)
    compression = pipeline.run()
    assert compression["estimated_latency_reduction"] >= 0.2
    bench = pipeline.benchmark(iterations=64)
    assert bench["average_latency_ms"] > 0.0

    artefact = pipeline.export(tmp_path, DeploymentTarget.TFLITE)
    assert artefact.exists()

    report = json.loads(artefact.read_text())
    assert report["target"] == "tflite"

    loaded = load_benchmark_report(artefact)
    assert loaded["iterations"] == bench["iterations"]
