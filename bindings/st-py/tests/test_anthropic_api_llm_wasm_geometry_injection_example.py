from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("spiraltorch")


def _load_example_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "anthropic_api_llm_wasm_geometry_injection.py"
    )
    spec = importlib.util.spec_from_file_location(
        "anthropic_api_llm_wasm_geometry_injection",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_geometry_injection_example_runs_keyless_comparison() -> None:
    module = _load_example_module()

    result = module.run_geometry_injection(
        prompt="Route the geometry.",
        conditions=("baseline", "calm", "turbulent"),
        live_anthropic=False,
        max_tokens=32,
    )

    assert result["kind"] == "spiraltorch.anthropic_wasm_geometry_injection"
    assert result["probe_source"] == "builtin-wasm-shaped"
    assert result["conditions"] == ["baseline", "calm", "turbulent"]
    assert result["results"]["baseline"]["context_origins"] == []
    assert result["results"]["calm"]["context_origins"][-1] == "geometry:consensus"
    assert result["results"]["turbulent"]["context_origins"][-1] == "geometry:consensus"
    assert result["results"]["calm"]["telemetry"][
        "geometry.consensus.probe_count"
    ] == pytest.approx(3.0)
    assert result["results"]["turbulent"]["telemetry"][
        "geometry.log_z_series.3.projection_stability"
    ] < 0.001
    assert "Geometry context routed decoding" in result["results"]["calm"]["text"]


def test_geometry_injection_example_cli_writes_json(tmp_path, capsys) -> None:
    module = _load_example_module()
    out = tmp_path / "geometry-injection.json"

    exit_code = module.main(
        [
            "--prompt",
            "Route one condition.",
            "--condition",
            "calm",
            "--json-out",
            str(out),
            "--indent",
            "2",
        ]
    )

    printed = json.loads(capsys.readouterr().out)
    written = json.loads(out.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert printed == written
    assert printed["conditions"] == ["calm"]
    assert printed["results"]["calm"]["context_origins"] == [
        "geometry:scale",
        "geometry:field",
        "geometry:logz",
        "geometry:consensus",
    ]
