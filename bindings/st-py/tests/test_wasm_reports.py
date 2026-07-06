from __future__ import annotations

import json

import pytest

st = pytest.importorskip("spiraltorch")


def _mellin_report(final_loss: float = 0.125) -> dict[str, object]:
    return {
        "schema": "spiraltorch.wasm.mellin_learning_report.v1",
        "kind": "mellin-log-grid-training",
        "createdAt": "2026-07-06T00:00:00.000Z",
        "runtime": {"wasm": True, "webgpuAvailable": True},
        "config": {"len": 128, "trainSteps": 3, "trainLr": 0.08},
        "target": {
            "grid": {
                "len": 128,
                "logStart": -5.0,
                "logStep": 0.01,
                "hilbertNorm": 1.5,
                "sampleStats": {"count": 256, "finiteCount": 256, "rms": 0.1},
            },
            "magnitudeStats": {"count": 8, "finiteCount": 8, "mean": 0.8, "rms": 0.9},
        },
        "learned": {
            "grid": {
                "len": 128,
                "logStart": -5.0,
                "logStep": 0.01,
                "hilbertNorm": 1.45,
                "sampleStats": {"count": 256, "finiteCount": 256, "rms": 0.11},
            },
            "magnitudeStats": {"count": 8, "finiteCount": 8, "mean": 0.78, "rms": 0.88},
        },
        "plan": {"len": 8, "shape": [1, 8], "logStart": -5.0, "logStep": 0.01},
        "training": {
            "steps": 3,
            "lr": 0.08,
            "durationMs": 4.5,
            "finalLoss": final_loss,
            "traceStride": 1,
            "trace": [
                {"step": 1, "loss": 0.5},
                {"step": 2, "loss": 0.25},
                {"step": 3, "loss": final_loss},
            ],
        },
        "inferenceProbe": {
            "mode": "vertical",
            "sReal": 2.5,
            "absDiffStats": {
                "count": 8,
                "finiteCount": 8,
                "mean": 0.05,
                "rms": 0.06,
                "linf": 0.1,
            },
        },
    }


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
        "config": {"width": 4, "height": 4, "mode": "hyper", "objective": "target-mse"},
        "target": {
            "dims": {"width": 4, "height": 4},
            "stats": {"count": 16, "finiteCount": 16, "rms": 0.25},
        },
        "currentFrame": {
            "width": 4,
            "height": 4,
            "relationStats": {"count": 16, "finiteCount": 16, "mean": 0.01, "rms": 0.2},
            "fieldStats": {"count": 64, "finiteCount": 64, "rms": 0.3},
            "trailStats": {"count": 112, "finiteCount": 112, "rms": 0.4},
            "desire": {
                "balance": 0.55,
                "stability": 0.82,
                "saturation": 0.18,
                "eventsMask": 1,
            },
            "gradients": {
                "hypergradRms": 0.12,
                "realgradRms": 0.08,
                "hypergradCount": 16,
                "realgradCount": 16,
            },
            "learningControl": {
                "hyperLearningRateScale": 0.9,
                "realLearningRateScale": 0.8,
                "operatorMix": 0.35,
                "operatorGain": 0.7,
            },
        },
        "metrics": {
            "step": 3,
            "historyLength": 3,
            "truncated": False,
            "last": {"step": 2, "loss": last_loss, "hyperRms": 0.12, "realRms": 0.08},
            "lossStats": {
                "count": 3,
                "finiteCount": 3,
                "min": last_loss,
                "max": 0.2,
                "mean": 0.11,
                "rms": 0.13,
            },
            "hyperRmsStats": {"count": 3, "finiteCount": 3, "mean": 0.12, "rms": 0.12},
            "realRmsStats": {"count": 3, "finiteCount": 3, "mean": 0.08, "rms": 0.08},
            "lrStats": {"count": 3, "finiteCount": 3, "mean": 0.02, "rms": 0.02},
        },
    }


def test_wasm_report_helpers_exported_from_top_level() -> None:
    assert "build_wasm_report_context" in st.__all__
    assert "build_wasm_report_context_artifact" in st.__all__
    assert "collect_wasm_report_paths" in st.__all__
    assert "load_wasm_report_context_artifact" in st.__all__
    assert "load_wasm_report" in st.__all__
    assert "summarize_wasm_report" in st.__all__
    assert "compare_wasm_reports" in st.__all__
    assert "wasm_report_to_zspace_partial" in st.__all__
    assert "write_wasm_report_context_artifact" in st.__all__


def test_load_and_summarize_mellin_wasm_report(tmp_path) -> None:
    path = tmp_path / "mellin.json"
    path.write_text(json.dumps(_mellin_report()), encoding="utf-8")

    loaded = st.load_wasm_report(path)
    summary = st.summarize_wasm_report(loaded)

    assert loaded["artifact_path"] == str(path)
    assert summary["family"] == "mellin"
    assert summary["learning"]["final_loss"] == pytest.approx(0.125)
    assert summary["learning"]["trace"]["improved"] is True
    assert summary["inference_probe"]["abs_diff"]["rms"] == pytest.approx(0.06)


def test_summarize_canvas_wasm_report_and_convert_to_partial() -> None:
    summary = st.summarize_wasm_report(_canvas_report())

    assert summary["family"] == "canvas"
    assert summary["runtime"]["webgpu_device_ready"] is True
    assert summary["learning"]["last_loss"] == pytest.approx(0.05)
    assert summary["desire"]["stability"] == pytest.approx(0.82)

    partial = st.wasm_report_to_zspace_partial(_canvas_report(), gradient_dim=6)
    metrics = partial.resolved()

    assert partial.origin == "wasm:canvas"
    assert metrics["stability"] == pytest.approx(0.82)
    assert len(metrics["gradient"]) == 6
    telemetry = partial.telemetry_payload()
    assert telemetry is not None
    assert telemetry["wasm.family_canvas"] == pytest.approx(1.0)
    assert telemetry["wasm.webgpu_device_ready"] == pytest.approx(1.0)


def test_compare_wasm_reports_selects_best_loss() -> None:
    comparison = st.compare_wasm_reports(
        {
            "slow": _mellin_report(final_loss=0.2),
            "better": _mellin_report(final_loss=0.05),
        }
    )

    assert comparison["count"] == 2
    assert comparison["families"] == {"mellin": 2}
    assert comparison["best_loss"]["label"] == "better"
    assert comparison["best_loss"]["loss"] == pytest.approx(0.05)


def test_collect_and_build_wasm_report_context_selects_best_runs(tmp_path) -> None:
    runs = tmp_path / "runs"
    nested = runs / "nested"
    runs.mkdir()
    nested.mkdir()
    slow = runs / "slow.json"
    best = runs / "best.json"
    middle = nested / "middle.json"
    slow.write_text(json.dumps(_canvas_report(last_loss=0.2)), encoding="utf-8")
    best.write_text(json.dumps(_canvas_report(last_loss=0.01)), encoding="utf-8")
    middle.write_text(json.dumps(_canvas_report(last_loss=0.05)), encoding="utf-8")

    paths = st.collect_wasm_report_paths(
        [slow, slow],
        globs=[str(best)],
        dirs=[runs],
        recursive=True,
    )

    assert set(paths) == {str(slow), str(best), str(middle)}

    context, metadata = st.build_wasm_report_context(
        [slow, slow],
        report_globs=[str(best)],
        report_dirs=[runs],
        recursive=True,
        max_reports=2,
        gradient_dim=5,
    )

    selected = {row["artifact_path"] for row in metadata["reports"]}
    assert metadata["candidate_count"] == 3
    assert metadata["report_count"] == 2
    assert selected == {str(best), str(middle)}
    assert metadata["comparison"]["best_loss"]["label"] == "best"
    assert metadata["comparison"]["best_loss"]["loss"] == pytest.approx(0.01)
    assert len(context) == 2
    assert all(len(partial.resolved()["gradient"]) == 5 for partial in context)


def test_wasm_report_context_artifact_roundtrips_into_api_llm_suite(tmp_path) -> None:
    slow = tmp_path / "slow.json"
    best = tmp_path / "best.json"
    artifact = tmp_path / "handoff" / "wasm-context.json"
    slow.write_text(json.dumps(_canvas_report(last_loss=0.2)), encoding="utf-8")
    best.write_text(json.dumps(_canvas_report(last_loss=0.01)), encoding="utf-8")

    written = st.write_wasm_report_context_artifact(
        artifact,
        [slow, best],
        max_reports=1,
        gradient_dim=6,
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    context, metadata = st.load_wasm_report_context_artifact(written)

    assert written == str(artifact)
    assert payload["schema"] == "spiraltorch.wasm_report_context.v1"
    assert payload["metadata"]["report_count"] == 1
    assert payload["metadata"]["reports"][0]["artifact_path"] == str(best)
    assert metadata["artifact_path"] == str(artifact)
    assert metadata["artifact_schema"] == "spiraltorch.wasm_report_context.v1"
    assert len(context) == 1

    def fake_api(prompt: str) -> dict[str, object]:
        text = f"Hosted route reuses selected browser context for {prompt}"
        return {
            "model": "local-wasm-context-artifact",
            "output_text": text,
            "status": "completed",
            "usage": {
                "input_tokens": max(1, len(prompt.split())),
                "output_tokens": max(1, len(text.split())),
            },
        }

    suite = st.run_api_llm_prompt_suite(
        ["Use selected WASM context."],
        fake_api,
        z_state=[0.12, -0.04, 0.33, -0.11],
        create_session=False,
        context_partials=context,
    )
    telemetry = suite["traces"][0]["inference"]["telemetry"]["payload"]
    assert telemetry["wasm.family_canvas"] == pytest.approx(1.0)
    assert telemetry["wasm.loss"] == pytest.approx(0.01)
    assert telemetry["wasm.webgpu_device_ready"] == pytest.approx(1.0)
