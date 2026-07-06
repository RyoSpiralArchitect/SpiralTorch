from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

import pytest

st = pytest.importorskip("spiraltorch")


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def _load_examples(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLES_DIR))
    sys.modules.pop("api_llm_live_provider_matrix_sweep", None)
    sys.modules.pop("api_llm_live_provider_matrix", None)
    live_matrix = importlib.import_module("api_llm_live_provider_matrix")
    sweep = importlib.import_module("api_llm_live_provider_matrix_sweep")
    return live_matrix, sweep


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


def _write_canvas_report(tmp_path: Path) -> Path:
    report = tmp_path / "canvas-report.json"
    report.write_text(json.dumps(_canvas_report()), encoding="utf-8")
    return report


def test_live_matrix_builds_wasm_context_from_report(tmp_path, monkeypatch) -> None:
    live_matrix, _ = _load_examples(monkeypatch)
    report = _write_canvas_report(tmp_path)

    context, metadata = live_matrix.build_wasm_context(
        str(report),
        gradient_dim=6,
        bundle_weight=0.25,
        telemetry_prefix="browser",
    )

    assert metadata["report_count"] == 1
    assert metadata["reports"][0]["artifact_path"] == str(report)
    assert metadata["reports"][0]["loss"] == pytest.approx(0.05)
    assert metadata["context_origins"] == ["wasm:canvas"]
    assert len(context) == 1
    partial = context[0]
    assert partial.origin == "wasm:canvas"
    assert partial.weight == pytest.approx(0.25)
    assert len(partial.resolved()["gradient"]) == 6
    telemetry = partial.telemetry_payload()
    assert telemetry is not None
    assert telemetry["browser.family_canvas"] == pytest.approx(1.0)
    assert telemetry["browser.webgpu_device_ready"] == pytest.approx(1.0)


def test_sweep_dry_run_records_wasm_context(tmp_path, monkeypatch, capsys) -> None:
    _, sweep = _load_examples(monkeypatch)
    report = _write_canvas_report(tmp_path)
    out_dir = tmp_path / "sweep"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "api_llm_live_provider_matrix_sweep.py",
            "--dry-run",
            "--out-dir",
            str(out_dir),
            "--prompt-limit",
            "1",
            "--budget-pairs",
            "64:128",
            "--wasm-report",
            str(report),
            "--wasm-gradient-dim",
            "6",
        ],
    )

    sweep.main()

    printed = json.loads(capsys.readouterr().out)
    written = json.loads((out_dir / "sweep-plan.json").read_text(encoding="utf-8"))
    assert printed == written
    assert printed["wasm_context"]["report_count"] == 1
    assert printed["wasm_context"]["context_origins"] == ["wasm:canvas"]
    assert printed["wasm_context"]["reports"][0]["loss"] == pytest.approx(0.05)


def test_sweep_run_config_passes_wasm_context_into_trace(
    tmp_path,
    monkeypatch,
) -> None:
    live_matrix, sweep = _load_examples(monkeypatch)
    report = _write_canvas_report(tmp_path)
    context, wasm_context = live_matrix.build_wasm_context(
        [str(report)],
        gradient_dim=6,
        bundle_weight=1.0,
        telemetry_prefix="wasm",
    )

    def fake_invoke(prompt: str, **_: object) -> dict[str, object]:
        text = f"Local route uses browser-side context for {prompt}"
        return {
            "model": "local-model",
            "output_text": text,
            "status": "completed",
            "usage": {
                "input_tokens": max(1, len(prompt.split())),
                "output_tokens": max(1, len(text.split())),
            },
        }

    def fake_build_routes(**_: object):
        return (
            {"local-route": fake_invoke},
            {"local-route": "local"},
            {"local-route": "local-model"},
            {"local-route": {}},
            {},
        )

    monkeypatch.setattr(live_matrix, "build_routes", fake_build_routes)

    summary = sweep._run_config(
        label="00-o64-a128",
        out_dir=tmp_path / "matrix",
        prompts=["Use the WASM context."],
        z_state=[0.12, -0.04, 0.33, -0.11],
        backend=None,
        create_session=False,
        openai_model="unused-openai",
        openai_max_output_tokens=64,
        anthropic_models=["unused-anthropic"],
        anthropic_efforts=["low"],
        anthropic_max_tokens=128,
        near_best_tolerance=0.02,
        context_partials=context,
        wasm_context=wasm_context,
        resume_existing=False,
        force=False,
    )

    payload = json.loads(Path(summary["report"]).read_text(encoding="utf-8"))
    assert payload["wasm_context"]["report_count"] == 1
    assert payload["trace_paths"]["local-route"].endswith("00-local-route.jsonl")
    events = st.load_api_llm_trace_events(payload["trace_paths"]["local-route"])
    telemetry = events[0]["inference"]["telemetry"]["payload"]
    assert telemetry["wasm.family_canvas"] == pytest.approx(1.0)
    assert telemetry["wasm.webgpu_device_ready"] == pytest.approx(1.0)
