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
        / "openai_api_llm_wasm_context_runtime.py"
    )
    spec = importlib.util.spec_from_file_location(
        "openai_api_llm_wasm_context_runtime", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canvas_report(last_loss: float = 0.05) -> dict[str, object]:
    return {
        "schema": "spiraltorch.wasm.canvas_hypertrain_report.v1",
        "kind": "canvas-hypertrain-training",
        "createdAt": "2026-07-06T00:00:00.000Z",
        "runtime": {
            "wasm": True,
            "webgpuAvailable": True,
            "webgpuDeviceReady": True,
            "webgpuTrainerReady": True,
        },
        "config": {"width": 4, "height": 4, "mode": "hyper"},
        "currentFrame": {
            "width": 4,
            "height": 4,
            "relationStats": {"count": 16, "finiteCount": 16, "rms": 0.2},
            "fieldStats": {"count": 64, "finiteCount": 64, "rms": 0.3},
            "trailStats": {"count": 112, "finiteCount": 112, "rms": 0.4},
            "desire": {"balance": 0.55, "stability": 0.82, "saturation": 0.18},
            "gradients": {"hypergradRms": 0.12, "realgradRms": 0.08},
            "learningControl": {"operatorMix": 0.35, "operatorGain": 0.7},
        },
        "metrics": {
            "step": 3,
            "historyLength": 3,
            "last": {"step": 2, "loss": last_loss, "hyperRms": 0.12},
            "lossStats": {
                "count": 3,
                "finiteCount": 3,
                "min": last_loss,
                "max": 0.2,
                "mean": 0.11,
                "rms": 0.13,
            },
        },
    }


def test_openai_wasm_context_example_writes_artifact_and_trace(tmp_path) -> None:
    module = _load_example_module()
    report = tmp_path / "canvas-report.json"
    trace_jsonl = tmp_path / "trace.jsonl"
    context_artifact = tmp_path / "handoff" / "wasm-context.json"
    report.write_text(json.dumps(_canvas_report()), encoding="utf-8")

    calls: list[tuple[str, dict[str, object]]] = []

    def fake_openai(prompt: str, **kwargs: object) -> dict[str, object]:
        calls.append((prompt, dict(kwargs)))
        text = f"OpenAI route consumed browser context: {prompt}"
        return {
            "id": "resp-openai-wasm-context-test",
            "object": "response",
            "model": "gpt-test",
            "output_text": text,
            "status": "completed",
            "usage": {
                "input_tokens": 5,
                "output_tokens": max(1, len(text.split())),
                "total_tokens": 5 + max(1, len(text.split())),
            },
        }

    result = module.run_openai_wasm_context(
        prompts=["Use the selected browser report."],
        model="gpt-test",
        backend="cpu",
        invoke=fake_openai,
        wasm_reports=[report],
        write_wasm_context_artifact=context_artifact,
        trace_jsonl=trace_jsonl,
        instructions="Use supplied SpiralTorch telemetry only.",
    )

    assert calls == [
        (
            "Use the selected browser report.",
            {"instructions": "Use supplied SpiralTorch telemetry only."},
        )
    ]
    assert result["kind"] == "spiraltorch.openai_wasm_context_runtime"
    assert result["provider"] == "openai"
    assert result["model"] == "gpt-test"
    assert result["trace_count"] == 1
    assert result["trace_jsonl"] == str(trace_jsonl)
    assert result["context_artifact"] == str(context_artifact)
    assert result["wasm_context"]["source"] == "built-and-written"
    assert result["wasm_context"]["report_count"] == 1
    assert result["wasm_context_seen"]["family_canvas"] == pytest.approx(1.0)
    assert result["wasm_context_seen"]["loss"] == pytest.approx(0.05)
    assert result["wasm_context_seen"]["webgpu_device_ready"] == pytest.approx(1.0)
    assert "OpenAI route consumed browser context" in result["first_text_preview"]

    artifact = json.loads(context_artifact.read_text(encoding="utf-8"))
    assert artifact["schema"] == "spiraltorch.wasm_report_context.v1"
    events = st.load_api_llm_trace_events(trace_jsonl)
    assert events[0]["telemetry"]["wasm.loss"] == pytest.approx(0.05)
    assert events[0]["telemetry"]["wasm.webgpu_device_ready"] == pytest.approx(1.0)


def test_openai_wasm_context_example_loads_persisted_artifact(tmp_path) -> None:
    module = _load_example_module()
    report = tmp_path / "canvas-report.json"
    artifact = tmp_path / "wasm-context.json"
    report.write_text(json.dumps(_canvas_report(last_loss=0.03)), encoding="utf-8")
    st.write_wasm_report_context_artifact(artifact, [report], gradient_dim=4)

    def fake_openai(prompt: str, **kwargs: object) -> dict[str, object]:
        return {
            "model": "gpt-test",
            "output_text": f"Artifact replay: {prompt}",
            "status": "completed",
            "usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
        }

    result = module.run_openai_wasm_context(
        prompts=["Replay the artifact."],
        model="gpt-test",
        backend="cpu",
        invoke=fake_openai,
        wasm_context_artifact=artifact,
    )

    assert result["context_artifact"] == str(artifact)
    assert result["wasm_context"]["source"] == "artifact"
    assert result["wasm_context"]["artifact_schema"] == "spiraltorch.wasm_report_context.v1"
    assert result["wasm_context_seen"]["loss"] == pytest.approx(0.03)


def test_openai_wasm_context_example_requires_context_input() -> None:
    module = _load_example_module()

    with pytest.raises(ValueError, match="provide --wasm-report"):
        module.run_openai_wasm_context(
            prompts=["No context."],
            model="gpt-test",
            invoke=lambda prompt, **kwargs: {"output_text": prompt},
        )
