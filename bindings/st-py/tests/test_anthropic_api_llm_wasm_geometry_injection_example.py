from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

st = pytest.importorskip("spiraltorch")


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
    assert result["context_mode"] == "consensus-only"
    assert result["conditions"] == ["baseline", "calm", "turbulent"]
    assert result["results"]["baseline"]["context_origins"] == []
    assert result["results"]["calm"]["context_origins"] == ["geometry:consensus"]
    assert result["results"]["turbulent"]["context_origins"] == ["geometry:consensus"]
    assert result["results"]["calm"]["telemetry"][
        "geometry.consensus.probe_count"
    ] == pytest.approx(3.0)
    assert result["results"]["turbulent"]["telemetry"][
        "geometry.consensus.log_z_series_projection_stability_mean"
    ] < 0.001
    assert "Geometry context routed decoding" in result["results"]["calm"]["text"]


def test_geometry_injection_example_writes_trace_artifacts(tmp_path) -> None:
    module = _load_example_module()
    trace_dir = tmp_path / "traces"

    result = module.run_geometry_injection(
        prompt="Route trace artifacts.",
        conditions=("baseline", "calm"),
        live_anthropic=False,
        trace_dir=trace_dir,
    )

    assert result["trace_paths"].keys() == {"baseline", "calm"}
    assert result["trace_comparison"]["count"] == 2
    assert result["trace_summaries"]["calm"]["count"] == 1
    calm_events = st.load_api_llm_trace_events(result["trace_paths"]["calm"])
    assert calm_events[0]["telemetry"][
        "geometry.consensus.probe_count"
    ] == pytest.approx(3.0)
    assert Path(result["trace_paths"]["baseline"]).exists()
    assert Path(result["trace_paths"]["calm"]).exists()


def test_geometry_injection_example_repeats_trace_artifacts(tmp_path) -> None:
    module = _load_example_module()
    trace_dir = tmp_path / "repeated-traces"

    result = module.run_geometry_injection(
        prompt="Route repeated trace artifacts.",
        conditions=("baseline", "calm"),
        live_anthropic=False,
        repeat=2,
        trace_dir=trace_dir,
    )

    assert result["repeat"] == 2
    assert len(result["runs"]["baseline"]) == 2
    assert len(result["runs"]["calm"]) == 2
    assert result["trace_comparison"]["count"] == 2
    assert result["trace_summaries"]["baseline"]["count"] == 2
    assert result["trace_summaries"]["calm"]["count"] == 2
    assert result["run_trace_comparison"]["count"] == 4
    calm_events = st.load_api_llm_trace_events(result["trace_paths"]["calm"])
    assert len(calm_events) == 2
    assert set(result["run_trace_paths"]) == {
        "baseline.1",
        "baseline.2",
        "calm.1",
        "calm.2",
    }


def test_geometry_injection_example_cli_writes_json(tmp_path, capsys) -> None:
    module = _load_example_module()
    out = tmp_path / "geometry-injection.json"

    exit_code = module.main(
        [
            "--prompt",
            "Route one condition.",
            "--condition",
            "calm",
            "--full-context",
            "--trace-dir",
            str(tmp_path / "traces"),
            "--repeat",
            "2",
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
    assert printed["context_mode"] == "full"
    assert printed["conditions"] == ["calm"]
    assert printed["repeat"] == 2
    assert printed["trace_comparison"]["count"] == 1
    assert printed["trace_summaries"]["calm"]["count"] == 2
    assert printed["run_trace_comparison"]["count"] == 2
    assert printed["results"]["calm"]["context_origins"] == [
        "geometry:scale",
        "geometry:field",
        "geometry:logz",
        "geometry:consensus",
    ]


def test_geometry_injection_repeat_sample_output_is_sanitized() -> None:
    sample_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "anthropic_wasm_geometry_injection_repeat_sample.json"
    )

    sample_text = sample_path.read_text(encoding="utf-8")
    sample = json.loads(sample_text)

    assert sample["kind"] == "spiraltorch.anthropic_wasm_geometry_injection.sample"
    assert sample["probe_source"] == "node-st-wasm"
    assert sample["repeat"] == 2
    assert sample["comparison"]["run_trace_count"] == 4
    assert sample["comparison"]["winners"]["best_score"] in sample["conditions"]
    assert "ANTHROPIC_API_KEY" not in sample_text
    assert "sk-" not in sample_text
    assert "/tmp/" not in sample_text
    assert sample["runs"]["calm"][0]["finish_reason"] == "end_turn"
    assert sample["runs"]["turbulent"][0]["telemetry"][
        "geometry.consensus.log_z_series_projection_stability_mean"
    ] < 0.001
