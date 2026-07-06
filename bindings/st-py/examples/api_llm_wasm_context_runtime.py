# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of SpiralTorch -- Licensed under AGPL-3.0-or-later.
"""Blend a browser-exported WASM learning report into API-model Z-space runtime.

This example is intentionally keyless and network-free. It uses a small local
WASM-report-shaped mapping plus a fake hosted-model response so the whole bridge
can be exercised in CI, notebooks, or first-run shells:

1. Browser WASM demo exports a report JSON.
2. Python converts that report into a Z-space context partial.
3. API-model responses are inferred with that browser-side learning context.

Replace ``canvas_wasm_report()`` with ``st.load_wasm_report("report.json")`` and
``fake_api`` with ``runtime.call_openai_responses(...)`` or
``runtime.call_anthropic_messages(...)`` for a real hosted-model route.
"""

from __future__ import annotations

import json

import spiraltorch as st


def canvas_wasm_report() -> dict[str, object]:
    """Return a compact Canvas Hypertrain report in the browser export shape."""

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
        "config": {
            "width": 8,
            "height": 8,
            "mode": "hyper",
            "objective": "target-mse",
        },
        "target": {
            "dims": {"width": 8, "height": 8},
            "stats": {"count": 64, "finiteCount": 64, "rms": 0.24},
        },
        "currentFrame": {
            "width": 8,
            "height": 8,
            "relationStats": {"count": 64, "finiteCount": 64, "mean": 0.02, "rms": 0.18},
            "fieldStats": {"count": 256, "finiteCount": 256, "rms": 0.31},
            "trailStats": {"count": 448, "finiteCount": 448, "rms": 0.37},
            "desire": {
                "balance": 0.56,
                "stability": 0.86,
                "saturation": 0.14,
                "eventsMask": 1,
            },
            "gradients": {
                "hypergradRms": 0.11,
                "realgradRms": 0.07,
                "hypergradCount": 64,
                "realgradCount": 64,
            },
            "learningControl": {
                "hyperLearningRateScale": 0.92,
                "realLearningRateScale": 0.81,
                "operatorMix": 0.38,
                "operatorGain": 0.74,
            },
        },
        "metrics": {
            "step": 6,
            "historyLength": 6,
            "truncated": False,
            "last": {"step": 5, "loss": 0.041, "hyperRms": 0.11, "realRms": 0.07},
            "lossStats": {
                "count": 6,
                "finiteCount": 6,
                "min": 0.041,
                "max": 0.19,
                "mean": 0.082,
                "rms": 0.095,
            },
            "hyperRmsStats": {"count": 6, "finiteCount": 6, "mean": 0.11, "rms": 0.11},
            "realRmsStats": {"count": 6, "finiteCount": 6, "mean": 0.07, "rms": 0.07},
            "lrStats": {"count": 6, "finiteCount": 6, "mean": 0.018, "rms": 0.018},
        },
    }


def fake_api(prompt: str) -> dict[str, object]:
    """Return an OpenAI Responses-shaped local payload."""

    text = (
        "The hosted model reads the browser WASM context as a live Z-space "
        f"prior before answering: {prompt}"
    )
    return {
        "id": "resp-wasm-context-demo",
        "object": "response",
        "model": "local-wasm-context-demo",
        "output_text": text,
        "status": "completed",
        "usage": {
            "input_tokens": max(1, len(prompt.split())),
            "output_tokens": max(1, len(text.split())),
        },
    }


def main() -> None:
    report = canvas_wasm_report()
    report_summary = st.summarize_wasm_report(report)
    context_partials = st.api_llm_wasm_context_partials(report, gradient_dim=6)

    suite = st.run_api_llm_prompt_suite(
        [
            "Use the browser-side training signal as context.",
            "Explain one safe next step for runtime fine-tuning.",
        ],
        fake_api,
        z_state=[0.12, -0.04, 0.33, -0.11],
        provider="local-demo",
        model="local-wasm-context-demo",
        create_session=False,
        context_partials=context_partials,
    )

    first_inference = suite["traces"][0]["inference"]
    telemetry = first_inference["telemetry"]["payload"] if first_inference else {}
    print(
        json.dumps(
            {
                "wasm_family": report_summary["family"],
                "wasm_last_loss": report_summary["learning"]["last_loss"],
                "context_origin": context_partials[0].origin,
                "trace_count": suite["count"],
                "first_confidence": first_inference["confidence"],
                "wasm_context_seen": {
                    "family_canvas": telemetry.get("wasm.family_canvas"),
                    "webgpu_device_ready": telemetry.get("wasm.webgpu_device_ready"),
                    "loss": telemetry.get("wasm.loss"),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
