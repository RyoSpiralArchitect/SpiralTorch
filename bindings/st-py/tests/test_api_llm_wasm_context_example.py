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
        / "api_llm_wasm_context_runtime.py"
    )
    spec = importlib.util.spec_from_file_location("api_llm_wasm_context_runtime", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canvas_report() -> dict[str, object]:
    return {
        "schema": "spiraltorch.wasm.canvas_hypertrain_report.v1",
        "kind": "canvas-hypertrain-training",
        "runtime": {
            "wasm": True,
            "webgpuAvailable": True,
            "webgpuDeviceReady": True,
        },
        "currentFrame": {
            "width": 4,
            "height": 4,
            "relationStats": {"count": 16, "finiteCount": 16, "rms": 0.2},
            "fieldStats": {"count": 64, "finiteCount": 64, "rms": 0.3},
            "trailStats": {"count": 112, "finiteCount": 112, "rms": 0.4},
            "desire": {"balance": 0.5, "stability": 0.9, "saturation": 0.1},
            "gradients": {"hypergradRms": 0.12, "realgradRms": 0.08},
            "learningControl": {"operatorMix": 0.4, "operatorGain": 0.7},
        },
        "metrics": {
            "step": 2,
            "historyLength": 2,
            "last": {"loss": 0.05},
            "lossStats": {"count": 2, "finiteCount": 2, "mean": 0.08, "rms": 0.09},
        },
    }


def test_wasm_context_example_accepts_report_file_and_writes_outputs(
    tmp_path, capsys
) -> None:
    module = _load_example_module()
    report = tmp_path / "canvas-report.json"
    summary = tmp_path / "summary.json"
    trace_jsonl = tmp_path / "trace.jsonl"
    report.write_text(json.dumps(_canvas_report()), encoding="utf-8")

    exit_code = module.main(
        [
            "--wasm-report",
            str(report),
            "--prompt",
            "Use this browser-side context.",
            "--json-out",
            str(summary),
            "--trace-jsonl",
            str(trace_jsonl),
            "--indent",
            "2",
        ]
    )

    printed = json.loads(capsys.readouterr().out)
    written = json.loads(summary.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert printed == written
    assert printed["report_count"] == 1
    assert printed["reports"][0]["artifact_path"] == str(report)
    assert printed["reports"][0]["loss"] == pytest.approx(0.05)
    assert printed["trace_count"] == 1
    assert printed["trace_jsonl"] == str(trace_jsonl)
    assert printed["wasm_context_seen"]["family_canvas"] == pytest.approx(1.0)
    assert printed["wasm_context_seen"]["webgpu_device_ready"] == pytest.approx(1.0)
    assert trace_jsonl.exists()


def test_wasm_context_example_runs_with_builtin_report() -> None:
    module = _load_example_module()

    result = module.run_demo(prompts=["Use the built-in report."])

    assert result["report_count"] == 1
    assert result["reports"][0]["family"] == "canvas"
    assert result["context_origins"] == ["wasm:canvas"]
    assert result["trace_count"] == 1
    assert result["wasm_context_seen"]["loss"] == pytest.approx(0.041)
