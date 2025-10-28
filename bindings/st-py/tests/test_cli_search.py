"""Integration tests for the SpiralTorch training CLI."""

from __future__ import annotations

import copy
import importlib
import json
import sys
import types
from pathlib import Path

import pytest

try:
    pytest.importorskip("spiraltorch")
except AttributeError as exc:  # pragma: no cover - environment-specific
    pytest.skip(
        f"spiraltorch import failed because torch is unavailable: {exc}",
        allow_module_level=True,
    )

from spiral.cli import main as cli_main

CONFIG_TEMPLATE = {
    "space": [
        {"name": "lr", "type": "float", "low": 0.0001, "high": 0.1},
        {"name": "layers", "type": "int", "low": 1, "high": 4},
        {"name": "activation", "type": "categorical", "choices": ["relu", "gelu", "tanh"]},
    ],
    "strategy": {"name": "bayesian", "exploration": 0.2, "seed": 123},
    "objective": {
        "callable": "spiral.tests.objectives:quadratic_objective",
        "maximize": False,
    },
    "resource": {"max_concurrent": 1, "min_interval_ms": 0},
    "max_trials": 4,
}


def write_config(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config, indent=2))
    return path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def test_cli_resume_matches_baseline(tmp_path: Path) -> None:
    config_path = write_config(tmp_path, CONFIG_TEMPLATE)

    baseline_ckpt = tmp_path / "baseline_ckpt.json"
    baseline_output = tmp_path / "baseline_output.json"
    cli_main([
        "search",
        "--config",
        str(config_path),
        "--checkpoint",
        str(baseline_ckpt),
        "--output",
        str(baseline_output),
        "--max-trials",
        "4",
    ])

    staged_ckpt = tmp_path / "staged_ckpt.json"
    staged_output = tmp_path / "staged_output.json"
    cli_main([
        "search",
        "--config",
        str(config_path),
        "--checkpoint",
        str(staged_ckpt),
        "--output",
        str(staged_output),
        "--max-trials",
        "2",
    ])

    cli_main([
        "search",
        "--config",
        str(config_path),
        "--checkpoint",
        str(staged_ckpt),
        "--resume",
        str(staged_ckpt),
        "--output",
        str(staged_output),
        "--max-trials",
        "4",
    ])

    baseline = read_json(baseline_output)
    resumed = read_json(staged_output)
    assert baseline == resumed


def test_cli_surfaces_resource_errors(tmp_path: Path) -> None:
    config = copy.deepcopy(CONFIG_TEMPLATE)
    config["resource"] = {"max_concurrent": 0}
    config_path = write_config(tmp_path, config)
    checkpoint = tmp_path / "ckpt.json"

    with pytest.raises(RuntimeError):
        cli_main([
            "search",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--max-trials",
            "1",
        ])


def test_cli_rejects_non_mapping_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps([{"foo": "bar"}]))

    with pytest.raises(
        TypeError,
        match=r"トップレベルはオブジェクト（マッピング）でなければならない; got list",
    ):
        cli_main([
            "search",
            "--config",
            str(config_path),
            "--max-trials",
            "1",
        ])


def test_cli_writes_summary(tmp_path: Path) -> None:
    config_path = write_config(tmp_path, CONFIG_TEMPLATE)
    summary_path = tmp_path / "summary.json"
    output_path = tmp_path / "best.json"

    cli_main(
        [
            "search",
            "--config",
            str(config_path),
            "--max-trials",
            "3",
            "--summary",
            str(summary_path),
            "--output",
            str(output_path),
        ]
    )

    summary = read_json(summary_path)
    best = summary.get("best_trial")
    assert summary["objective"] == "minimize"
    assert summary["completed_trials"] == 3
    assert best is not None
    assert "metric" in best

    output = read_json(output_path)
    assert output["id"] == best["id"]
    assert output["metric"] == best["metric"]
