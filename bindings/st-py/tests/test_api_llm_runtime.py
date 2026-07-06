from __future__ import annotations

import json
import math

import pytest

st = pytest.importorskip("spiraltorch")


def test_api_llm_runtime_exports_from_top_level() -> None:
    assert "ApiLLMZSpaceRuntime" in st.__all__
    assert "api_llm_geometry_context_partials" in st.__all__
    assert "api_llm_partial_from_response" in st.__all__
    assert "api_llm_wasm_context_partials" in st.__all__
    assert "compare_api_llm_matrix_reports" in st.__all__
    assert "compare_api_llm_trace_runs" in st.__all__
    assert "format_api_llm_context_prompt" in st.__all__
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


def _geometry_log_z_probe() -> dict[str, object]:
    return {
        "kind": "spiraltorch.wasm_log_z_series_probe",
        "source_crate": "st-frac::cosmology",
        "mode": "log_z_series",
        "log_lattice": {"log_start": 0.0, "log_step": 0.25, "len": 4},
        "options": {"window": "hann", "normalisation": "l1"},
        "sample_count": 4,
        "sample_stats": {"count": 4, "mean": 2.5, "min": 1.0, "max": 4.0, "energy": 7.5},
        "weight_stats": {"count": 4, "mean": 0.25, "min": 0.0, "max": 0.5, "energy": 0.125},
        "z_count": 2,
        "projection": {
            "count": 2,
            "mean_abs": 1.1,
            "max_abs": 1.4,
            "energy": 1.25,
            "phase_drift": 0.1,
            "stability_score": 0.9,
            "preview_count": 1,
            "preview": [{"index": 0, "re": 1.0, "im": 0.0, "abs": 1.0}],
        },
    }


def _canvas_wasm_report(
    *,
    last_loss: float = 0.05,
    stability: float = 0.9,
    webgpu_device_ready: bool = True,
) -> dict[str, object]:
    return {
        "schema": "spiraltorch.wasm.canvas_hypertrain_report.v1",
        "kind": "canvas-hypertrain-training",
        "runtime": {
            "wasm": True,
            "webgpuAvailable": True,
            "webgpuDeviceReady": webgpu_device_ready,
        },
        "currentFrame": {
            "width": 4,
            "height": 4,
            "relationStats": {"count": 16, "finiteCount": 16, "rms": 0.2},
            "desire": {"balance": 0.5, "stability": stability, "saturation": 0.1},
            "gradients": {"hypergradRms": 0.12, "realgradRms": 0.08},
            "learningControl": {"operatorMix": 0.4, "operatorGain": 0.7},
        },
        "metrics": {
            "step": 2,
            "historyLength": 2,
            "last": {"loss": last_loss},
            "lossStats": {"count": 2, "finiteCount": 2, "mean": 0.08, "rms": 0.09},
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


def test_api_llm_text_from_responses_skips_text_config_object() -> None:
    class ResponseTextConfig:
        def __str__(self) -> str:
            return "ResponseTextConfig(format=ResponseFormatText(type='text'))"

    response = {
        "model": "api-response-test",
        "text": ResponseTextConfig(),
        "output": [{"type": "reasoning", "content": []}],
        "usage": {"input_tokens": 4, "output_tokens": 64},
    }

    assert st.api_llm_text_from_response(response) == ""


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


def test_api_llm_runtime_blends_wasm_context_partials_for_each_prompt() -> None:
    context = st.api_llm_wasm_context_partials(_canvas_wasm_report(), gradient_dim=6)
    runtime = st.ApiLLMZSpaceRuntime(
        [0.12, -0.04, 0.33, -0.11],
        provider="example",
        create_session=False,
    )

    result = runtime.run_prompts(
        ["first", "second"],
        lambda _prompt: _chat_response(),
        context_partials=context,
    )

    assert len(context) == 1
    assert context[0].origin == "wasm:canvas"
    assert result["count"] == 2
    for trace in result["traces"]:
        inference = trace["inference"]
        assert inference is not None
        telemetry = inference["telemetry"]["payload"]
        assert telemetry["wasm.family_canvas"] == pytest.approx(1.0)
        assert telemetry["wasm.webgpu_device_ready"] == pytest.approx(1.0)
        assert inference["confidence"] > 0.0


def test_format_api_llm_context_prompt_includes_bounded_wasm_telemetry() -> None:
    context = st.api_llm_wasm_context_partials(
        _canvas_wasm_report(last_loss=0.0415, stability=0.86),
        gradient_dim=6,
    )

    prompt = st.format_api_llm_context_prompt(
        "Diagnose the run.",
        context,
        max_metrics=4,
        max_telemetry=8,
    )

    assert prompt.startswith("SpiralTorch Z-space context:")
    assert "origin=wasm:canvas" in prompt
    assert "wasm.loss=0.0415" in prompt
    assert "wasm.stability_hint=0.86" in prompt
    assert "User prompt: Diagnose the run." in prompt


def test_api_llm_geometry_context_partials_feed_context_prompt() -> None:
    context = st.api_llm_geometry_context_partials(
        {"logz": _geometry_log_z_probe()},
        gradient_dim=5,
        include_consensus=True,
    )

    prompt = st.format_api_llm_context_prompt(
        "Use geometry context.",
        context,
        max_partials=2,
        max_telemetry=20,
    )

    assert len(context) == 2
    assert context[0].origin == "geometry:logz"
    assert context[1].origin == "geometry:consensus"
    assert "origin=geometry:logz" in prompt
    assert "origin=geometry:consensus" in prompt
    assert "geometry.log_z_series.1.projection_stability=0.9" in prompt
    assert "geometry.consensus.probe_count=1" in prompt
    assert "User prompt: Use geometry context." in prompt


def test_api_llm_geometry_context_partials_can_send_consensus_only() -> None:
    context = st.api_llm_geometry_context_partials(
        {"logz": _geometry_log_z_probe()},
        gradient_dim=5,
        consensus_only=True,
    )

    prompt = st.format_api_llm_context_prompt(
        "Use compact geometry context.",
        context,
        max_partials=1,
        max_telemetry=20,
    )

    assert len(context) == 1
    assert context[0].origin == "geometry:consensus"
    assert "origin=geometry:consensus" in prompt
    assert "origin=geometry:logz" not in prompt
    assert "geometry.consensus.probe_count=1" in prompt


def test_runtime_context_prompt_injection_keeps_trace_prompt_original() -> None:
    context = st.api_llm_wasm_context_partials(_canvas_wasm_report(), gradient_dim=6)
    runtime = st.ApiLLMZSpaceRuntime(
        [0.12, -0.04, 0.33, -0.11],
        provider="example",
        create_session=False,
    )
    calls: list[str] = []

    def fake_api(prompt: str) -> dict[str, object]:
        calls.append(prompt)
        return _chat_response()

    trace = runtime.call(
        fake_api,
        "Use the browser report.",
        context_partials=iter(context),
        context_prompt=True,
        context_prompt_options={"max_telemetry": 8},
    )
    payload = trace.as_dict()

    assert len(calls) == 1
    assert calls[0] != "Use the browser report."
    assert "SpiralTorch Z-space context:" in calls[0]
    assert "wasm.loss=0.05" in calls[0]
    assert payload["prompt"] == "Use the browser report."
    assert payload["inference"]["telemetry"]["payload"]["wasm.loss"] == pytest.approx(0.05)


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


def test_compare_api_llm_matrix_reports_tracks_stable_profile_winners(tmp_path) -> None:
    def write_report(
        name: str,
        compact_score: float,
        expanded_score: float,
        *,
        wasm_loss: float,
        wasm_stability: float,
        webgpu_ready: bool,
    ) -> str:
        report = {
            "kind": "spiraltorch.api_llm_live_provider_matrix",
            "created_at": f"2026-07-06T00:00:0{len(name)}Z",
            "prompt_count": 24,
            "route_count": 2,
            "near_best_tolerance": 0.02,
            "skipped": {},
            "client_errors": [],
            "comparison": {
                "winners": {
                    "best_score": "openai-compact",
                    "highest_quality": "openai-expanded",
                    "highest_text_quality": "openai-expanded",
                    "highest_efficiency": "openai-compact",
                    "lowest_latency": "openai-compact",
                    "lowest_total_tokens": "openai-compact",
                },
                "selection_profiles": {
                    "balanced": {
                        "label": "openai-compact",
                        "score": compact_score,
                    },
                    "quality": {
                        "label": "openai-expanded",
                        "score": expanded_score,
                    },
                    "grounded": {
                        "label": "openai-expanded",
                        "score": expanded_score - 0.02,
                    },
                    "efficiency": {
                        "label": "openai-compact",
                        "score": compact_score - 0.05,
                    },
                    "latency": {
                        "label": "openai-compact",
                        "score": compact_score - 0.03,
                    },
                },
                "near_best": [{"label": "openai-compact"}, {"label": "openai-expanded"}],
                "runs": [
                    {
                        "label": "openai-compact",
                        "count": 24,
                        "route_score": compact_score,
                        "quality_score": 0.91,
                        "text_quality_score": 0.68,
                        "efficiency_score": 0.58,
                        "completion_rate": 1.0,
                        "latency_ms_mean": 2400.0,
                        "total_tokens": 4800.0,
                        "empty_text_rate": 0.0,
                        "refusal_rate": 0.0,
                    },
                    {
                        "label": "openai-expanded",
                        "count": 24,
                        "route_score": expanded_score,
                        "quality_score": 0.90,
                        "text_quality_score": 0.74,
                        "efficiency_score": 0.55,
                        "completion_rate": 1.0,
                        "latency_ms_mean": 3200.0,
                        "total_tokens": 5200.0,
                        "empty_text_rate": 0.0,
                        "refusal_rate": 0.0,
                    },
                ],
            },
            "wasm_context": {
                "report_count": 1,
                "context_origins": ["wasm:canvas"],
                "reports": [
                    {
                        "label": f"{name}-canvas",
                        "family": "canvas",
                        "loss": wasm_loss,
                        "stability": wasm_stability,
                        "webgpu_device_ready": webgpu_ready,
                    }
                ],
                "comparison": {
                    "families": {"canvas": 1},
                    "best_loss": {
                        "label": f"{name}-canvas",
                        "family": "canvas",
                        "loss": wasm_loss,
                        "stability": wasm_stability,
                    },
                    "best_stability": {
                        "label": f"{name}-canvas",
                        "family": "canvas",
                        "loss": wasm_loss,
                        "stability": wasm_stability,
                    },
                },
            },
        }
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        return str(path)

    first = write_report(
        "first",
        0.83,
        0.82,
        wasm_loss=0.05,
        wasm_stability=0.86,
        webgpu_ready=True,
    )
    second = write_report(
        "second",
        0.84,
        0.81,
        wasm_loss=0.02,
        wasm_stability=0.91,
        webgpu_ready=False,
    )

    comparison = st.compare_api_llm_matrix_reports(
        {"first": first, "second": second}
    )
    routes = {row["label"]: row for row in comparison["routes"]}

    assert comparison["kind"] == "spiraltorch.api_llm_matrix_report_comparison"
    assert comparison["count"] == 2
    assert comparison["profile_winners"]["balanced"][0]["label"] == "openai-compact"
    assert comparison["profile_winners"]["balanced"][0]["win_rate"] == 1.0
    assert comparison["profile_winners"]["quality"][0]["label"] == "openai-expanded"
    assert comparison["wasm_context"]["observed_reports"] == 2
    assert comparison["wasm_context"]["total_wasm_report_count"] == 2
    assert comparison["wasm_context"]["families"] == {"canvas": 2}
    assert comparison["wasm_context"]["context_origins"] == {"wasm:canvas": 2}
    assert comparison["wasm_context"]["lowest_best_loss"] == "second"
    assert comparison["wasm_context"]["highest_best_stability"] == "second"
    assert comparison["wasm_context"]["highest_webgpu_device_ready"] == "first"
    consistency = comparison["wasm_context"]["consistency"]
    assert consistency["status"] == "varied_selected_reports"
    assert consistency["consistent_families"] is True
    assert consistency["consistent_context_origins"] is True
    assert consistency["consistent_report_count"] is True
    assert consistency["best_loss"]["range"] == pytest.approx(0.03)
    report_rows = {row["label"]: row for row in comparison["reports"]}
    assert report_rows["second"]["wasm_best_loss"] == pytest.approx(0.02)
    assert report_rows["first"]["wasm_webgpu_device_ready_rate"] == pytest.approx(1.0)
    assert report_rows["second"]["wasm_webgpu_device_ready_rate"] == pytest.approx(0.0)
    assert routes["openai-compact"]["best_score_wins"] == 2
    assert routes["openai-compact"]["profile_wins"]["latency"] == 2
    assert routes["openai-expanded"]["profile_wins"]["grounded"] == 2
    assert routes["openai-compact"]["route_score_mean"] > routes["openai-expanded"][
        "route_score_mean"
    ]
    assert "balanced profile is stable on openai-compact" in " ".join(
        comparison["recommendations"]
    )
    assert "lowest selected WASM report loss" in " ".join(
        comparison["recommendations"]
    )
    assert "selected WASM report variation" in " ".join(
        comparison["recommendations"]
    )


def test_compare_api_llm_matrix_reports_flags_mixed_wasm_context(tmp_path) -> None:
    def write_report(name: str, *, family: str, origin: str) -> str:
        report = {
            "kind": "spiraltorch.api_llm_live_provider_matrix",
            "prompt_count": 1,
            "route_count": 0,
            "wasm_context": {
                "report_count": 1,
                "context_origins": [origin],
                "reports": [
                    {
                        "label": f"{name}-{family}",
                        "family": family,
                        "loss": 0.05,
                        "stability": 0.8,
                        "webgpu_device_ready": True,
                    }
                ],
                "comparison": {
                    "families": {family: 1},
                    "best_loss": {
                        "label": f"{name}-{family}",
                        "family": family,
                        "loss": 0.05,
                        "stability": 0.8,
                    },
                    "best_stability": {
                        "label": f"{name}-{family}",
                        "family": family,
                        "loss": 0.05,
                        "stability": 0.8,
                    },
                },
            },
        }
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        return str(path)

    comparison = st.compare_api_llm_matrix_reports(
        {
            "canvas-report": write_report(
                "canvas-report",
                family="canvas",
                origin="wasm:canvas",
            ),
            "mellin-report": write_report(
                "mellin-report",
                family="mellin",
                origin="wasm:mellin",
            ),
        }
    )
    consistency = comparison["wasm_context"]["consistency"]

    assert comparison["wasm_context"]["observed_reports"] == 2
    assert comparison["wasm_context"]["families"] == {"canvas": 1, "mellin": 1}
    assert consistency["status"] == "mixed_context"
    assert consistency["consistent_families"] is False
    assert consistency["consistent_context_origins"] is False
    assert "audit mixed WASM context" in " ".join(comparison["recommendations"])


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


def test_compare_api_llm_trace_runs_surfaces_wasm_context_tradeoffs(tmp_path) -> None:
    low_loss_context = st.api_llm_wasm_context_partials(
        _canvas_wasm_report(last_loss=0.02, stability=0.94),
        gradient_dim=6,
    )
    low_loss = st.ApiLLMZSpaceRuntime(
        [0.12, -0.04, 0.33, -0.11],
        provider="example",
        model="low-loss-model",
        create_session=False,
    )
    low_loss.run_prompts(
        ["Use browser context."],
        lambda _prompt: {
            "model": "low-loss-model",
            "output_text": "Low loss browser context entered Z-space.",
            "status": "completed",
            "usage": {"prompt_tokens": 8, "completion_tokens": 7, "total_tokens": 15},
        },
        context_partials=low_loss_context,
    )
    low_path = tmp_path / "low-loss.jsonl"
    low_loss.write_jsonl(low_path)

    high_loss_context = st.api_llm_wasm_context_partials(
        _canvas_wasm_report(
            last_loss=0.18,
            stability=0.62,
            webgpu_device_ready=False,
        ),
        gradient_dim=6,
    )
    high_loss = st.ApiLLMZSpaceRuntime(
        [0.12, -0.04, 0.33, -0.11],
        provider="example",
        model="high-loss-model",
        create_session=False,
    )
    high_loss.run_prompts(
        ["Use browser context."],
        lambda _prompt: {
            "model": "high-loss-model",
            "output_text": "High loss browser context entered Z-space.",
            "status": "completed",
            "usage": {"prompt_tokens": 8, "completion_tokens": 7, "total_tokens": 15},
        },
        context_partials=high_loss_context,
    )
    high_path = tmp_path / "high-loss.jsonl"
    high_loss.write_jsonl(high_path)

    summary = st.summarize_api_llm_trace_events(low_path)
    assert summary["wasm_context"]["observed_count"] == 1
    assert summary["wasm_context"]["families"] == {"canvas": 1}
    assert summary["wasm_context"]["loss"]["mean"] == pytest.approx(0.02)
    assert summary["wasm_context"]["stability_hint"]["mean"] == pytest.approx(0.94)
    assert summary["wasm_context"]["webgpu_device_ready"]["ready_rate"] == pytest.approx(
        1.0
    )

    comparison = st.compare_api_llm_trace_runs(
        {"low-loss": low_path, "high-loss": high_path},
        near_best_tolerance=1.0,
    )
    rows = {row["label"]: row for row in comparison["runs"]}

    assert comparison["wasm_context"]["observed_runs"] == 2
    assert comparison["wasm_context"]["observed_run_rate"] == pytest.approx(1.0)
    assert comparison["wasm_context"]["families"] == {"canvas": 2}
    assert comparison["winners"]["lowest_wasm_loss"] == "low-loss"
    assert comparison["winners"]["highest_wasm_stability_hint"] == "low-loss"
    assert comparison["winners"]["highest_wasm_webgpu_device_ready"] == "low-loss"
    assert rows["low-loss"]["wasm_family"] == "canvas"
    assert rows["low-loss"]["wasm_loss_mean"] == pytest.approx(0.02)
    assert rows["low-loss"]["wasm_webgpu_device_ready_rate"] == pytest.approx(1.0)
    assert rows["high-loss"]["wasm_loss_mean"] == pytest.approx(0.18)
    assert rows["high-loss"]["wasm_webgpu_device_ready_rate"] == pytest.approx(0.0)
    assert any(
        "lowest browser-side WASM context loss" in recommendation
        for recommendation in comparison["recommendations"]
    )


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
