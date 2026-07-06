from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

st = pytest.importorskip("spiraltorch")


def _load_example_module():
    examples_dir = Path(__file__).resolve().parents[1] / "examples"
    path = examples_dir / "openai_api_llm_wasm_geometry_injection.py"
    spec = importlib.util.spec_from_file_location(
        "openai_api_llm_wasm_geometry_injection",
        path,
    )
    assert spec is not None and spec.loader is not None
    sys.path.insert(0, str(examples_dir))
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(examples_dir))


def test_openai_geometry_injection_example_runs_keyless_comparison() -> None:
    module = _load_example_module()

    result = module.run_openai_geometry_injection(
        prompt="Route OpenAI geometry.",
        conditions=("baseline", "calm", "turbulent"),
        live_openai=False,
        max_output_tokens=32,
    )

    assert result["kind"] == "spiraltorch.openai_wasm_geometry_injection"
    assert result["provider"] == "local-demo"
    assert result["model"] == module.DEFAULT_MODEL
    assert result["reasoning_effort"] == "none"
    assert result["probe_source"] == "builtin-wasm-shaped"
    assert result["context_mode"] == "consensus-only"
    assert result["conditions"] == ["baseline", "calm", "turbulent"]
    assert result["results"]["baseline"]["context_origins"] == []
    assert result["results"]["calm"]["context_origins"] == ["geometry:consensus"]
    assert result["results"]["calm"]["telemetry"][
        "geometry.consensus.probe_count"
    ] == pytest.approx(3.0)
    assert result["results"]["turbulent"]["telemetry"][
        "geometry.consensus.log_z_series_projection_stability_mean"
    ] < 0.001


def test_openai_geometry_injection_example_writes_trace_artifacts(tmp_path) -> None:
    module = _load_example_module()
    trace_dir = tmp_path / "openai-traces"

    result = module.run_openai_geometry_injection(
        prompt="Route OpenAI trace artifacts.",
        conditions=("baseline", "calm"),
        live_openai=False,
        repeat=2,
        trace_dir=trace_dir,
    )

    assert result["repeat"] == 2
    assert result["trace_paths"].keys() == {"baseline", "calm"}
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


def test_openai_geometry_injection_example_cli_writes_json(tmp_path, capsys) -> None:
    module = _load_example_module()
    out = tmp_path / "openai-geometry-injection.json"

    exit_code = module.main(
        [
            "--prompt",
            "Route one OpenAI condition.",
            "--condition",
            "calm",
            "--full-context",
            "--reasoning-effort",
            "none",
            "--trace-dir",
            str(tmp_path / "traces"),
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
    assert printed["provider"] == "local-demo"
    assert printed["context_mode"] == "full"
    assert printed["conditions"] == ["calm"]
    assert printed["reasoning_effort"] == "none"
    assert printed["trace_comparison"]["count"] == 1
    assert printed["results"]["calm"]["context_origins"] == [
        "geometry:scale",
        "geometry:field",
        "geometry:logz",
        "geometry:consensus",
    ]
