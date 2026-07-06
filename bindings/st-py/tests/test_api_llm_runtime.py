from __future__ import annotations

import math

import pytest

st = pytest.importorskip("spiraltorch")


def test_api_llm_runtime_exports_from_top_level() -> None:
    assert "ApiLLMZSpaceRuntime" in st.__all__
    assert "api_llm_partial_from_response" in st.__all__
    assert "make_openai_chat_invoke" in st.__all__
    assert "make_openai_responses_invoke" in st.__all__


class _FakeResponses:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **request: object) -> dict[str, object]:
        self.calls.append(dict(request))
        return {
            "id": "resp-test",
            "object": "response",
            "model": request.get("model", "response-model-test"),
            "output_text": "Adapter response entered Z-space.",
            "status": "completed",
            "usage": {"input_tokens": 7, "output_tokens": 5, "total_tokens": 12},
        }


class _FakeResponsesClient:
    def __init__(self) -> None:
        self.responses = _FakeResponses()


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **request: object) -> dict[str, object]:
        self.calls.append(dict(request))
        return {
            "id": "chat-test",
            "object": "chat.completion",
            "model": request.get("model", "chat-model-test"),
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Chat adapter entered Z-space.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 5,
                "total_tokens": 14,
            },
        }


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeChatClient:
    def __init__(self) -> None:
        self.chat = _FakeChat()


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


def test_make_openai_responses_invoke_uses_client_factory_lazily() -> None:
    created: list[dict[str, object]] = []
    client = _FakeResponsesClient()

    def factory(**kwargs: object) -> _FakeResponsesClient:
        created.append(dict(kwargs))
        return client

    invoke = st.make_openai_responses_invoke(
        client_factory=factory,
        client_kwargs={"api_key": "test-key"},
        model="response-model-test",
        max_output_tokens=12,
    )
    response = invoke("trace this response", temperature=0.1)

    assert created == [{"api_key": "test-key"}]
    assert st.api_llm_text_from_response(response) == "Adapter response entered Z-space."
    assert client.responses.calls == [
        {
            "max_output_tokens": 12,
            "temperature": 0.1,
            "model": "response-model-test",
            "input": "trace this response",
        }
    ]


def test_runtime_call_openai_responses_records_trace() -> None:
    client = _FakeResponsesClient()
    runtime = st.ApiLLMZSpaceRuntime(
        [0.2, -0.1, 0.4, 0.05],
        model="response-model-test",
        create_session=False,
    )

    trace = runtime.call_openai_responses(
        "route the hosted model",
        client=client,
        max_output_tokens=8,
    )

    assert trace.provider == "openai"
    assert trace.model == "response-model-test"
    assert trace.text == "Adapter response entered Z-space."
    assert trace.inference is not None
    assert client.responses.calls[0]["input"] == "route the hosted model"
    assert client.responses.calls[0]["model"] == "response-model-test"


def test_make_openai_chat_invoke_builds_messages() -> None:
    client = _FakeChatClient()
    invoke = st.make_openai_chat_invoke(
        client=client,
        model="chat-model-test",
        system="Answer in compact Z-space language.",
        max_tokens=10,
    )
    response = invoke("trace this chat", temperature=0.2)

    assert st.api_llm_text_from_response(response) == "Chat adapter entered Z-space."
    request = client.chat.completions.calls[0]
    assert request["model"] == "chat-model-test"
    assert request["max_tokens"] == 10
    assert request["temperature"] == 0.2
    assert request["messages"] == [
        {"role": "system", "content": "Answer in compact Z-space language."},
        {"role": "user", "content": "trace this chat"},
    ]


def test_runtime_call_openai_chat_records_trace() -> None:
    client = _FakeChatClient()
    runtime = st.ApiLLMZSpaceRuntime(
        [0.05, 0.15, -0.2, 0.31],
        create_session=False,
    )

    trace = runtime.call_openai_chat(
        "route the chat model",
        client=client,
        model="chat-model-test",
        messages=[{"role": "assistant", "content": "Prior trace acknowledged."}],
        max_tokens=6,
    )

    assert trace.provider == "openai"
    assert trace.model == "chat-model-test"
    assert trace.text == "Chat adapter entered Z-space."
    assert trace.inference is not None
    messages = client.chat.completions.calls[0]["messages"]
    assert messages[-2:] == [
        {"role": "assistant", "content": "Prior trace acknowledged."},
        {"role": "user", "content": "route the chat model"},
    ]
