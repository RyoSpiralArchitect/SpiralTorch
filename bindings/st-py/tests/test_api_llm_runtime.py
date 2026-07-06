from __future__ import annotations

import math

import pytest

st = pytest.importorskip("spiraltorch")


def test_api_llm_runtime_exports_from_top_level() -> None:
    assert "ApiLLMZSpaceRuntime" in st.__all__
    assert "api_llm_partial_from_response" in st.__all__


def _chat_response() -> dict[str, object]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "api-model-test",
        "choices": [
            {
                "message": {"role": "assistant", "content": "Z-space runtime is awake."},
                "finish_reason": "stop",
                "logprobs": {
                    "content": [
                        {
                            "token": "Z",
                            "logprob": -0.2,
                            "top_logprobs": [
                                {"token": "Z", "logprob": -0.2},
                                {"token": "The", "logprob": -1.6},
                            ],
                        }
                    ]
                },
            }
        ],
        "usage": {
            "prompt_tokens": 6,
            "completion_tokens": 5,
            "total_tokens": 11,
        },
    }


def test_api_llm_text_from_chat_completion_shape() -> None:
    assert st.api_llm_text_from_response(_chat_response()) == "Z-space runtime is awake."


def test_api_llm_text_from_responses_shape() -> None:
    response = {
        "model": "api-response-test",
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "Spiral "},
                    {"type": "output_text", "text": "route"},
                ],
            }
        ],
        "usage": {"input_tokens": 4, "output_tokens": 2},
    }

    assert st.api_llm_text_from_response(response) == "Spiral route"
    usage = st.api_llm_usage_from_response(response)
    assert usage["prompt_tokens"] == 4
    assert usage["completion_tokens"] == 2
    assert usage["total_tokens"] == 6


def test_api_llm_partial_derives_zspace_bundle() -> None:
    bundle = st.api_llm_partial_from_response(
        _chat_response(),
        prompt="Where does the runtime live?",
        provider="example",
        latency_ms=250.0,
        gradient_dim=6,
    )

    metrics = bundle.resolved()
    assert bundle.origin == "api_llm"
    assert 0.0 <= metrics["speed"] <= 1.0
    assert 0.0 <= metrics["memory"] <= 1.0
    assert metrics["stability"] > 0.7
    assert len(metrics["gradient"]) == 6
    telemetry = bundle.telemetry_payload()
    assert telemetry is not None
    assert telemetry["api_llm.total_tokens"] == 11
    assert telemetry["api_llm.provider_present"] == 1.0
    assert math.isclose(telemetry["api_llm.finish_reason_stop"], 1.0)


def test_api_llm_runtime_calls_callable_and_records_inference() -> None:
    runtime = st.ApiLLMZSpaceRuntime(
        [0.12, -0.04, 0.33, -0.11],
        provider="example",
        create_session=False,
    )

    def fake_api(prompt: str) -> dict[str, object]:
        assert prompt == "trace the API model"
        return _chat_response()

    trace = runtime.call(fake_api, "trace the API model")
    payload = trace.as_dict()

    assert trace.text == "Z-space runtime is awake."
    assert trace.inference is not None
    assert payload["model"] == "api-model-test"
    assert payload["inference"]["confidence"] > 0.0
    assert payload["metrics"]["stability"] > 0.7
    assert runtime.as_dict()["traces"][0]["text"] == trace.text
