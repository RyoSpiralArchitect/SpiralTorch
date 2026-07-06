from __future__ import annotations

import math

import pytest

st = pytest.importorskip("spiraltorch")


def test_api_llm_runtime_exports_from_top_level() -> None:
    assert "ApiLLMZSpaceRuntime" in st.__all__
    assert "api_llm_partial_from_response" in st.__all__
    assert "compare_api_llm_trace_runs" in st.__all__
    assert "load_api_llm_trace_events" in st.__all__
    assert "make_anthropic_messages_invoke" in st.__all__
    assert "make_openai_chat_invoke" in st.__all__
    assert "make_openai_responses_invoke" in st.__all__
    assert "run_api_llm_prompt_suite" in st.__all__
    assert "run_api_llm_prompt_suite_matrix" in st.__all__
    assert "summarize_api_llm_trace_events" in st.__all__
    assert "write_api_llm_trace_jsonl" in st.__all__


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


class _FakeAnthropicMessages:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **request: object) -> dict[str, object]:
        self.calls.append(dict(request))
        return {
            "id": "msg-test",
            "type": "message",
            "role": "assistant",
            "model": request.get("model", "claude-test"),
            "content": [
                {
                    "type": "text",
                    "text": "Anthropic message entered bipolar Z-space geometry.",
                }
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 8, "output_tokens": 7},
        }


class _FakeAnthropicClient:
    def __init__(self) -> None:
        self.messages = _FakeAnthropicMessages()


class _SdkTextBlock:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _SdkThinkingBlock:
    type = "thinking"

    def __init__(self, thinking: str) -> None:
        self.thinking = thinking


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


def test_api_llm_text_from_anthropic_messages_shape() -> None:
    response = {
        "model": "claude-test",
        "content": [
            {"type": "thinking", "thinking": "private route scratchpad"},
            {"type": "text", "text": "Bipolar geometry route "},
            _SdkThinkingBlock("sdk private route scratchpad"),
            _SdkTextBlock("entered SDK Z-space, "),
            {"type": "text", "text": "entered Z-space."},
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 6, "output_tokens": 4},
    }

    assert st.api_llm_text_from_response(response) == (
        "Bipolar geometry route entered SDK Z-space, entered Z-space."
    )
    usage = st.api_llm_usage_from_response(response)
    assert usage["prompt_tokens"] == 6
    assert usage["completion_tokens"] == 4
    assert usage["total_tokens"] == 10


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


def test_make_anthropic_messages_invoke_builds_messages() -> None:
    client = _FakeAnthropicClient()
    invoke = st.make_anthropic_messages_invoke(
        client=client,
        model="claude-test",
        system="Reason in bipolar geometry coordinates.",
        max_tokens=16,
    )
    response = invoke("trace this Anthropic message", temperature=0.1)

    assert st.api_llm_text_from_response(response) == (
        "Anthropic message entered bipolar Z-space geometry."
    )
    request = client.messages.calls[0]
    assert request["model"] == "claude-test"
    assert request["system"] == "Reason in bipolar geometry coordinates."
    assert request["max_tokens"] == 16
    assert request["temperature"] == 0.1
    assert request["messages"] == [
        {"role": "user", "content": "trace this Anthropic message"}
    ]


def test_runtime_call_anthropic_messages_records_trace() -> None:
    client = _FakeAnthropicClient()
    runtime = st.ApiLLMZSpaceRuntime(
        [0.05, 0.15, -0.2, 0.31],
        model="claude-test",
        create_session=False,
    )

    trace = runtime.call_anthropic_messages(
        "route Anthropic through bipolar geometry",
        client=client,
        max_tokens=20,
    )

    assert trace.provider == "anthropic"
    assert trace.model == "claude-test"
    assert trace.text == "Anthropic message entered bipolar Z-space geometry."
    assert trace.finish_reason == "end_turn"
    assert trace.usage["total_tokens"] == 15
    assert trace.inference is not None
    assert client.messages.calls[0]["messages"] == [
        {"role": "user", "content": "route Anthropic through bipolar geometry"}
    ]


def test_api_llm_runtime_writes_loads_and_summarizes_jsonl(tmp_path) -> None:
    runtime = st.ApiLLMZSpaceRuntime(
        [0.12, -0.04, 0.33, -0.11],
        provider="example",
        model="api-model-test",
        create_session=False,
    )

    runtime.record_response(
        _chat_response(),
        prompt="first trace",
        latency_ms=100.0,
    )
    runtime.record_response(
        {
            "model": "api-model-test",
            "output_text": "Second Z-space runtime trace.",
            "status": "completed",
            "usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
        },
        prompt="second trace",
        latency_ms=200.0,
    )

    runtime_summary = runtime.summary()
    assert runtime_summary["count"] == 2
    assert runtime_summary["models"] == {"api-model-test": 2}
    assert runtime_summary["usage"]["total_tokens"]["mean"] == 9.0
    assert runtime_summary["empty_text_rate"] == 0.0
    assert runtime_summary["refusal_rate"] == 0.0
    assert runtime_summary["completion_rate"] == 1.0
    assert runtime_summary["text_quality"]["text_quality_score"]["mean"] > 0.0
    assert runtime.as_dict()["summary"]["count"] == 2

    path = tmp_path / "api-llm-trace.jsonl"
    assert runtime.write_jsonl(path) == str(path)

    events = st.load_api_llm_trace_events(path)
    assert len(events) == 2
    assert events[0]["step"] == 0
    assert events[1]["text"] == "Second Z-space runtime trace."

    summary = st.summarize_api_llm_trace_events(path)
    assert summary["count"] == 2
    assert summary["first_step"] == 0
    assert summary["last_step"] == 1
    assert summary["last_text_preview"] == "Second Z-space runtime trace."
    assert summary["models"] == {"api-model-test": 2}
    assert summary["total_tokens"] == 18.0
    assert summary["empty_text_count"] == 0
    assert summary["empty_text_rate"] == 0.0
    assert summary["refusal_count"] == 0
    assert summary["completion_rate"] == 1.0
    assert summary["latency_ms"]["max"] == 200.0
    assert summary["confidence"]["min"] > 0.0
    assert "stability" in summary["metrics"]
    assert summary["text_quality"]["text_quality_score"]["mean"] > 0.0
    assert summary["text_quality"]["response_signal_rate"]["mean"] > 0.0
    assert summary["text_quality"]["repetition_rate"]["max"] < 1.0


def test_api_llm_trace_health_penalizes_empty_refusals(tmp_path) -> None:
    healthy = st.ApiLLMZSpaceRuntime(
        [0.12, -0.04, 0.33, -0.11],
        provider="example",
        model="healthy-model",
        create_session=False,
    )
    healthy.record_response(
        {
            "model": "healthy-model",
            "output_text": "A visible answer entered Z-space.",
            "status": "completed",
            "usage": {"input_tokens": 4, "output_tokens": 6, "total_tokens": 10},
        },
        prompt="visible route",
        latency_ms=100.0,
    )
    healthy_path = tmp_path / "healthy.jsonl"
    healthy.write_jsonl(healthy_path)

    refusal = st.ApiLLMZSpaceRuntime(
        [0.12, -0.04, 0.33, -0.11],
        provider="example",
        model="refusal-model",
        create_session=False,
    )
    refusal.record_response(
        {
            "id": "refusal-test",
            "model": "refusal-model",
            "content": [{"type": "thinking", "thinking": "hidden route"}],
            "stop_reason": "refusal",
            "stop_details": {"category": "policy"},
            "usage": {"input_tokens": 4, "output_tokens": 8, "total_tokens": 12},
        },
        prompt="refusal route",
        latency_ms=100.0,
    )
    refusal_path = tmp_path / "refusal.jsonl"
    refusal.write_jsonl(refusal_path)

    summary = st.summarize_api_llm_trace_events(refusal_path)
    assert summary["empty_text_count"] == 1
    assert summary["empty_text_rate"] == 1.0
    assert summary["refusal_count"] == 1
    assert summary["refusal_rate"] == 1.0
    assert summary["completion_rate"] == 0.0
    assert summary["stop_detail_categories"] == {"policy": 1}

    event = st.load_api_llm_trace_events(refusal_path)[0]
    assert event["response_metadata"]["stop_details"] == {"category": "policy"}
    assert event["telemetry"]["api_llm.empty_text"] == 1.0
    assert event["telemetry"]["api_llm.finish_reason_refusal"] == 1.0

    comparison = st.compare_api_llm_trace_runs(
        {"healthy": healthy_path, "refusal": refusal_path}
    )
    rows = {row["label"]: row for row in comparison["runs"]}
    assert rows["refusal"]["empty_text_rate"] == 1.0
    assert rows["refusal"]["refusal_rate"] == 1.0
    assert rows["refusal"]["completion_rate"] == 0.0
    assert rows["refusal"]["stop_detail_category"] == "policy"
    assert rows["healthy"]["route_score"] > rows["refusal"]["route_score"]
    assert comparison["winners"]["lowest_empty_text"] == "healthy"
    assert comparison["winners"]["lowest_refusal"] == "healthy"


def test_api_llm_runtime_runs_prompt_suite_and_writes_jsonl(tmp_path) -> None:
    runtime = st.ApiLLMZSpaceRuntime(
        [0.12, -0.04, 0.33, -0.11],
        provider="suite-provider",
        model="suite-model",
        create_session=False,
    )
    calls: list[str] = []

    def fake_api(prompt: str, *, suffix: str) -> dict[str, object]:
        calls.append(prompt)
        return {
            "model": "suite-model",
            "output_text": f"{prompt} {suffix}",
            "status": "completed",
            "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
        }

    path = tmp_path / "suite.jsonl"
    result = runtime.run_prompts(
        ["first route", "second route"],
        fake_api,
        suffix="entered Z-space.",
        jsonl_out=path,
    )

    assert calls == ["first route", "second route"]
    assert result["kind"] == "spiraltorch.api_llm_prompt_suite"
    assert result["count"] == 2
    assert result["runtime_trace_count"] == 2
    assert result["provider"] == "suite-provider"
    assert result["model"] == "suite-model"
    assert result["jsonl"] == str(path)
    assert result["summary"]["count"] == 2
    assert len(result["traces"]) == 2
    assert result["traces"][0]["prompt"] == "first route"

    events = st.load_api_llm_trace_events(path)
    assert [event["prompt"] for event in events] == ["first route", "second route"]
    summary = st.summarize_api_llm_trace_events(path)
    assert summary["count"] == 2
    assert summary["models"] == {"suite-model": 2}
    assert summary["total_tokens"] == 14.0


def test_run_api_llm_prompt_suite_creates_runtime(tmp_path) -> None:
    calls: list[tuple[str, float]] = []

    def fake_api(prompt: str, *, temperature: float) -> dict[str, object]:
        calls.append((prompt, temperature))
        return {
            "model": "top-level-suite",
            "output_text": "Top-level prompt suite entered Z-space.",
            "usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
        }

    path = tmp_path / "top-level-suite.jsonl"
    result = st.run_api_llm_prompt_suite(
        ["route the API model"],
        fake_api,
        z_state=[0.2, -0.1, 0.4, 0.05],
        provider="suite-provider",
        model="top-level-suite",
        create_session=False,
        jsonl_out=path,
        temperature=0.2,
    )

    assert calls == [("route the API model", 0.2)]
    assert result["count"] == 1
    assert result["runtime_trace_count"] == 1
    assert result["requested_backend"] == "auto"
    assert result["device_preflight"] is None
    assert result["summary"]["models"] == {"top-level-suite": 1}
    assert result["jsonl"] == str(path)


def test_run_api_llm_prompt_suite_matrix_compares_providers(tmp_path) -> None:
    calls: list[tuple[str, str, str]] = []

    def fast_api(prompt: str, *, suffix: str) -> dict[str, object]:
        calls.append(("fast api", prompt, suffix))
        return {
            "model": "fast-model",
            "output_text": f"Fast bipolar route {suffix}",
            "status": "completed",
            "usage": {"prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9},
        }

    def deep_api(prompt: str, *, suffix: str) -> dict[str, object]:
        calls.append(("deep/api", prompt, suffix))
        return {
            "model": "deep-model",
            "content": [
                {
                    "type": "text",
                    "text": f"Deep bipolar route with more generated tokens {suffix}",
                }
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 12, "output_tokens": 24},
        }

    result = st.run_api_llm_prompt_suite_matrix(
        ["first bipolar prompt", "second bipolar prompt"],
        {"fast api": fast_api, "deep/api": deep_api},
        z_state=[0.2, -0.1, 0.4, 0.05],
        providers={"fast api": "openai", "deep/api": "anthropic"},
        models={"fast api": "fast-model", "deep/api": "deep-model"},
        create_session=False,
        jsonl_dir=tmp_path,
        request_kwargs={
            "fast api": {"suffix": "entered fast Z-space."},
            "deep/api": {"suffix": "entered deep Z-space."},
        },
        near_best_tolerance=0.05,
        suffix="entered shared Z-space.",
    )

    assert calls == [
        ("fast api", "first bipolar prompt", "entered fast Z-space."),
        ("fast api", "second bipolar prompt", "entered fast Z-space."),
        ("deep/api", "first bipolar prompt", "entered deep Z-space."),
        ("deep/api", "second bipolar prompt", "entered deep Z-space."),
    ]
    assert result["kind"] == "spiraltorch.api_llm_prompt_suite_matrix"
    assert result["count"] == 2
    assert result["prompt_count"] == 2
    assert result["labels"] == ["fast api", "deep/api"]
    assert set(result["suites"]) == {"fast api", "deep/api"}
    assert result["suites"]["fast api"]["summary"]["models"] == {"fast-model": 2}
    assert result["suites"]["deep/api"]["summary"]["models"] == {"deep-model": 2}
    assert set(result["trace_paths"]) == {"fast api", "deep/api"}
    assert result["trace_paths"]["fast api"].endswith("00-fast-api.jsonl")
    assert result["trace_paths"]["deep/api"].endswith("01-deep-api.jsonl")

    comparison = result["comparison"]
    assert comparison["kind"] == "spiraltorch.api_llm_trace_comparison"
    assert comparison["count"] == 2
    assert comparison["near_best_tolerance"] == 0.05
    assert {row["label"] for row in comparison["runs"]} == {"fast api", "deep/api"}
    assert comparison["winners"]["lowest_total_tokens"] == "fast api"


def test_compare_api_llm_trace_runs_ranks_compact_artifacts(tmp_path) -> None:
    fast = st.ApiLLMZSpaceRuntime(
        [0.12, -0.04, 0.33, -0.11],
        provider="example",
        model="fast-model",
        create_session=False,
    )
    fast.record_response(
        _chat_response(),
        prompt="fast route",
        model="fast-model",
        latency_ms=50.0,
    )
    fast_path = tmp_path / "fast.jsonl"
    fast.write_jsonl(fast_path)

    slow = st.ApiLLMZSpaceRuntime(
        [0.12, -0.04, 0.33, -0.11],
        provider="example",
        model="slow-model",
        create_session=False,
    )
    slow.record_response(
        {
            "model": "slow-model",
            "output_text": "A slower API model response with more generated text.",
            "status": "completed",
            "usage": {"input_tokens": 64, "output_tokens": 96, "total_tokens": 160},
        },
        prompt="slow route",
        latency_ms=2500.0,
    )
    slow_path = tmp_path / "slow.jsonl"
    slow.write_jsonl(slow_path)

    comparison = st.compare_api_llm_trace_runs({"fast": fast_path, "slow": slow_path})

    assert comparison["kind"] == "spiraltorch.api_llm_trace_comparison"
    assert comparison["count"] == 2
    assert comparison["runs"][0]["label"] == "fast"
    assert comparison["winners"]["best_score"] == "fast"
    assert comparison["winners"]["lowest_latency"] == "fast"
    assert comparison["winners"]["lowest_total_tokens"] == "fast"
    assert comparison["runs"][0]["route_score"] > comparison["runs"][1]["route_score"]
    assert comparison["recommendations"] == [
        "prefer fast for the highest aggregate API LLM route score"
    ]


def test_compare_api_llm_trace_runs_exposes_route_tradeoffs(tmp_path) -> None:
    compact = st.ApiLLMZSpaceRuntime(
        [0.12, -0.04, 0.33, -0.11],
        provider="example",
        model="compact-model",
        create_session=False,
    )
    compact.record_response(
        {
            "model": "compact-model",
            "output_text": "Compact route entered Z-space.",
            "status": "completed",
            "usage": {"prompt_tokens": 8, "completion_tokens": 8, "total_tokens": 16},
        },
        prompt="compact route",
        latency_ms=100.0,
    )
    compact_path = tmp_path / "compact.jsonl"
    compact.write_jsonl(compact_path)

    expanded = st.ApiLLMZSpaceRuntime(
        [0.12, -0.04, 0.33, -0.11],
        provider="example",
        model="expanded-model",
        create_session=False,
    )
    expanded.record_response(
        {
            "model": "expanded-model",
            "output_text": (
                "Expanded route entered Z-space with additional context for a "
                "near-best tradeoff comparison."
            ),
            "status": "completed",
            "usage": {"prompt_tokens": 8, "completion_tokens": 18, "total_tokens": 26},
        },
        prompt="expanded route",
        latency_ms=180.0,
    )
    expanded_path = tmp_path / "expanded.jsonl"
    expanded.write_jsonl(expanded_path)

    comparison = st.compare_api_llm_trace_runs(
        {"compact": compact_path, "expanded": expanded_path},
        near_best_tolerance=0.25,
    )
    rows = {row["label"]: row for row in comparison["runs"]}

    assert comparison["near_best_tolerance"] == 0.25
    assert {row["label"] for row in comparison["near_best"]} == {
        "compact",
        "expanded",
    }
    assert "compare near-best routes within 0.250" in comparison["recommendations"][1]
    assert comparison["winners"]["highest_efficiency"] == "compact"
    assert comparison["winners"]["highest_quality"] in {"compact", "expanded"}
    assert comparison["winners"]["highest_text_quality"] in {"compact", "expanded"}
    assert comparison["selection_profiles"]["balanced"]["label"] in {
        "compact",
        "expanded",
    }
    assert comparison["selection_profiles"]["efficiency"]["label"] == "compact"
    assert comparison["selection_profiles"]["latency"]["label"] == "compact"
    assert 0.0 <= comparison["selection_profiles"]["quality"]["score"] <= 1.0
    assert rows["compact"]["latency_cost"] < rows["expanded"]["latency_cost"]
    assert rows["compact"]["token_cost"] < rows["expanded"]["token_cost"]
    assert rows["compact"]["health_penalty"] == 0.0
    assert "grounded" in rows["expanded"]["selection_scores"]
    assert rows["compact"]["prompt_coverage_mean"] > 0.0
    assert rows["compact"]["text_quality_score"] > 0.0
    assert 0.0 <= rows["expanded"]["quality_score"] <= 1.0
    assert 0.0 <= rows["expanded"]["efficiency_score"] <= 1.0


def test_compare_api_llm_trace_runs_flags_prompt_text_quality(tmp_path) -> None:
    aligned = st.ApiLLMZSpaceRuntime(
        [0.12, -0.04, 0.33, -0.11],
        provider="example",
        model="aligned-model",
        create_session=False,
    )
    aligned.record_response(
        {
            "model": "aligned-model",
            "output_text": "Z-space routing audits latency, tokens, and prompt coverage.",
            "status": "completed",
            "usage": {"prompt_tokens": 8, "completion_tokens": 9, "total_tokens": 17},
        },
        prompt="Audit Z-space routing latency tokens and prompt coverage",
        latency_ms=150.0,
    )
    aligned_path = tmp_path / "aligned.jsonl"
    aligned.write_jsonl(aligned_path)

    off_topic = st.ApiLLMZSpaceRuntime(
        [0.12, -0.04, 0.33, -0.11],
        provider="example",
        model="off-topic-model",
        create_session=False,
    )
    off_topic.record_response(
        {
            "model": "off-topic-model",
            "output_text": "Bananas repeat bananas repeat bananas repeat.",
            "status": "completed",
            "usage": {"prompt_tokens": 8, "completion_tokens": 6, "total_tokens": 14},
        },
        prompt="Audit Z-space routing latency tokens and prompt coverage",
        latency_ms=120.0,
    )
    off_topic_path = tmp_path / "off-topic.jsonl"
    off_topic.write_jsonl(off_topic_path)

    comparison = st.compare_api_llm_trace_runs(
        {"aligned": aligned_path, "off-topic": off_topic_path},
        near_best_tolerance=1.0,
    )
    rows = {row["label"]: row for row in comparison["runs"]}

    assert comparison["winners"]["highest_text_quality"] == "aligned"
    assert comparison["selection_profiles"]["grounded"]["label"] == "aligned"
    assert rows["aligned"]["prompt_coverage_mean"] > rows["off-topic"]["prompt_coverage_mean"]
    assert rows["aligned"]["text_quality_score"] > rows["off-topic"]["text_quality_score"]
    assert rows["off-topic"]["repetition_rate_mean"] > rows["aligned"]["repetition_rate_mean"]
