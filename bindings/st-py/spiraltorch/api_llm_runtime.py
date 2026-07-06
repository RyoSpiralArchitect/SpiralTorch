"""Bridge API-model LLM inference into the SpiralTorch Z-space runtime.

The helpers in this module intentionally avoid hard dependencies on hosted LLM
SDKs.  Callers can pass an already materialised response mapping, an SDK
response object, or a callable that performs the API request.  SpiralTorch then
derives a bounded Z-space partial observation from text, usage, latency, and
probability-like fields.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

from .zspace_inference import (
    ZSpaceInferencePipeline,
    ZSpacePartialBundle,
)

__all__ = [
    "ApiLLMTrace",
    "ApiLLMZSpaceRuntime",
    "api_llm_partial_from_response",
    "api_llm_text_from_response",
    "api_llm_trace_from_response",
    "api_llm_usage_from_response",
    "api_llm_wasm_context_partials",
    "compare_api_llm_matrix_reports",
    "compare_api_llm_trace_runs",
    "load_api_llm_trace_events",
    "make_anthropic_messages_invoke",
    "make_openai_chat_invoke",
    "make_openai_responses_invoke",
    "run_api_llm_prompt_suite",
    "run_api_llm_prompt_suite_matrix",
    "summarize_api_llm_trace_events",
    "write_api_llm_trace_jsonl",
]


_OPENAI_INSTALL_HINT = "pip install openai"
_ANTHROPIC_INSTALL_HINT = "pip install anthropic"
_API_LLM_TRACE_SCHEMA = "spiraltorch.api_llm_trace.v1"
_SELECTION_PROFILES = ("balanced", "quality", "grounded", "efficiency", "latency")
_REPORT_ROUTE_METRICS = (
    "route_score",
    "quality_score",
    "text_quality_score",
    "efficiency_score",
    "completion_rate",
    "latency_ms_mean",
    "total_tokens",
    "empty_text_rate",
    "refusal_rate",
    "wasm_context_observed_rate",
    "wasm_loss_mean",
    "wasm_stability_hint_mean",
    "wasm_webgpu_device_ready_rate",
)
_TEXT_QUALITY_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "give",
        "how",
        "if",
        "in",
        "into",
        "is",
        "it",
        "name",
        "of",
        "on",
        "one",
        "or",
        "should",
        "that",
        "the",
        "this",
        "to",
        "with",
    }
)


def _value(source: Any, key: str, default: Any = None) -> Any:
    if source is None:
        return default
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def _maybe_mapping(source: Any) -> Mapping[str, Any] | None:
    if isinstance(source, Mapping):
        return source
    for name in ("model_dump", "dict", "as_dict"):
        fn = getattr(source, name, None)
        if not callable(fn):
            continue
        try:
            payload = fn()
        except Exception:
            continue
        if isinstance(payload, Mapping):
            return payload
    return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _finite_float(value: Any) -> float | None:
    numeric = _as_float(value)
    if numeric is None:
        return None
    return numeric if math.isfinite(numeric) else None


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, bytes):
        return content.decode("utf-8", "replace")
    if isinstance(content, Mapping):
        for key in ("text", "output_text", "content"):
            text = _content_text(content.get(key))
            if text:
                return text
        if content.get("type"):
            return ""
        return ""
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
        return "".join(_content_text(part) for part in content)
    for key in ("text", "output_text", "content"):
        candidate = getattr(content, key, None)
        if candidate is None or candidate is content:
            continue
        text = _content_text(candidate)
        if text:
            return text
    if getattr(content, "type", None):
        return ""
    return str(content)


def _first_choice(response: Any) -> Any:
    choices = _value(response, "choices")
    if isinstance(choices, Sequence) and choices:
        return choices[0]
    return None


def _response_output_text(response: Any) -> str:
    output_text = _content_text(_value(response, "output_text"))
    if output_text:
        return output_text

    output = _value(response, "output")
    if isinstance(output, Sequence) and not isinstance(output, (str, bytes, bytearray)):
        chunks: list[str] = []
        for item in output:
            text = _content_text(_value(item, "text"))
            if text:
                chunks.append(text)
                continue
            content = _value(item, "content")
            text = _content_text(content)
            if text:
                chunks.append(text)
        if chunks:
            return "".join(chunks)
    return ""


def api_llm_text_from_response(response: Any) -> str:
    """Extract the primary text from common hosted-LLM response shapes."""

    text = _response_output_text(response)
    if text:
        return text

    choice = _first_choice(response)
    if choice is not None:
        message = _value(choice, "message")
        for candidate in (
            _value(message, "content") if message is not None else None,
            _value(choice, "text"),
            _value(choice, "content"),
            _value(choice, "delta"),
        ):
            text = _content_text(candidate)
            if text:
                return text

    for key in ("text", "content", "message"):
        text = _content_text(_value(response, key))
        if text:
            return text
    return ""


def api_llm_usage_from_response(response: Any) -> dict[str, float]:
    """Return normalized usage counters from OpenAI-compatible response shapes."""

    usage = _value(response, "usage")
    mapping = _maybe_mapping(usage) or {}

    prompt_tokens = _as_float(
        mapping.get("prompt_tokens")
        or mapping.get("input_tokens")
        or mapping.get("prompt")
        or _value(usage, "prompt_tokens")
        or _value(usage, "input_tokens")
    )
    completion_tokens = _as_float(
        mapping.get("completion_tokens")
        or mapping.get("output_tokens")
        or mapping.get("completion")
        or _value(usage, "completion_tokens")
        or _value(usage, "output_tokens")
    )
    total_tokens = _as_float(
        mapping.get("total_tokens")
        or mapping.get("total")
        or _value(usage, "total_tokens")
    )
    if total_tokens is None:
        total_tokens = (prompt_tokens or 0.0) + (completion_tokens or 0.0)

    result: dict[str, float] = {}
    if prompt_tokens is not None:
        result["prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        result["completion_tokens"] = completion_tokens
    if total_tokens is not None:
        result["total_tokens"] = total_tokens
    return result


def _finish_reason_from_response(response: Any) -> str | None:
    choice = _first_choice(response)
    for source in (choice, response):
        for key in ("finish_reason", "stop_reason", "status"):
            value = _value(source, key)
            if value:
                return str(value)
    return None


def _response_metadata(response: Any) -> dict[str, Any]:
    metadata = _maybe_mapping(response) or {}
    payload = {
        key: value
        for key, value in metadata.items()
        if key in {"id", "created", "object", "system_fingerprint", "stop_details"}
    }
    stop_details = _value(response, "stop_details")
    stop_details_mapping = _maybe_mapping(stop_details)
    if stop_details_mapping:
        payload["stop_details"] = dict(stop_details_mapping)
    return payload


def _logprob_to_probability(logprob: Any) -> float | None:
    numeric = _as_float(logprob)
    if numeric is None:
        return None
    if 0.0 <= numeric <= 1.0:
        return numeric
    try:
        return max(0.0, min(1.0, math.exp(numeric)))
    except OverflowError:
        return None


def _top_probability_from_response(response: Any) -> float | None:
    candidates: list[Any] = [response, _first_choice(response)]
    for source in candidates:
        if source is None:
            continue
        for key in ("top_probability", "probability", "confidence"):
            prob = _logprob_to_probability(_value(source, key))
            if prob is not None:
                return prob
        logprobs = _value(source, "logprobs")
        if isinstance(logprobs, Mapping):
            content = logprobs.get("content")
            if isinstance(content, Sequence) and content:
                first = content[0]
                top = _value(first, "top_logprobs")
                if isinstance(top, Sequence) and top:
                    values = [
                        _logprob_to_probability(_value(item, "logprob"))
                        for item in top
                    ]
                    values = [value for value in values if value is not None]
                    if values:
                        return max(values)
                prob = _logprob_to_probability(_value(first, "logprob"))
                if prob is not None:
                    return prob
    return None


def _text_stats(text: str) -> dict[str, float]:
    chars = float(len(text))
    words = float(len([part for part in text.split() if part]))
    if not text:
        return {
            "chars": 0.0,
            "words": 0.0,
            "unique_chars": 0.0,
            "entropy": 0.0,
            "entropy_norm": 0.0,
        }
    counts: dict[str, int] = {}
    for char in text:
        counts[char] = counts.get(char, 0) + 1
    entropy = 0.0
    for count in counts.values():
        probability = count / len(text)
        entropy -= probability * math.log2(probability)
    max_entropy = math.log2(max(2, len(counts)))
    return {
        "chars": chars,
        "words": words,
        "unique_chars": float(len(counts)),
        "entropy": entropy,
        "entropy_norm": entropy / max_entropy if max_entropy > 0.0 else 0.0,
    }


def _word_tokens(value: Any) -> list[str]:
    text = str(value or "").lower()
    tokens: list[str] = []
    current: list[str] = []
    for char in text:
        if char.isalnum():
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current.clear()
    if current:
        tokens.append("".join(current))
    return tokens


def _significant_tokens(value: Any) -> list[str]:
    return [
        token
        for token in _word_tokens(value)
        if token not in _TEXT_QUALITY_STOPWORDS
        and (len(token) > 2 or token in {"ai", "api", "llm", "z"})
    ]


def _trace_text_quality(event: Mapping[str, Any]) -> dict[str, float | None]:
    prompt_tokens = set(_significant_tokens(event.get("prompt")))
    text_tokens = _significant_tokens(event.get("text"))
    text_token_set = set(text_tokens)
    raw_text_token_count = len(_word_tokens(event.get("text")))
    text_token_count = len(text_tokens)
    if text_token_count == 0:
        response_signal_rate = 0.0
        repetition_rate = 1.0
    else:
        response_signal_rate = text_token_count / max(1, raw_text_token_count)
        repetition_rate = 1.0 - (len(text_token_set) / text_token_count)

    prompt_coverage: float | None
    prompt_echo_rate: float | None
    if prompt_tokens:
        overlap = len(prompt_tokens & text_token_set)
        prompt_coverage = overlap / len(prompt_tokens)
        prompt_echo_rate = overlap / max(1, text_token_count)
    else:
        prompt_coverage = None
        prompt_echo_rate = None

    repetition_reward = 1.0 - max(0.0, min(1.0, repetition_rate))
    if prompt_coverage is None:
        text_quality_score = 0.65 * response_signal_rate + 0.35 * repetition_reward
    else:
        text_quality_score = (
            0.60 * prompt_coverage
            + 0.25 * response_signal_rate
            + 0.15 * repetition_reward
        )
    return {
        "prompt_coverage": prompt_coverage,
        "prompt_echo_rate": prompt_echo_rate,
        "response_signal_rate": response_signal_rate,
        "repetition_rate": repetition_rate,
        "text_quality_score": max(0.0, min(1.0, text_quality_score)),
    }


def _estimate_tokens(text: str) -> float:
    if not text:
        return 0.0
    return max(1.0, len(text) / 4.0)


def _runtime_metrics(
    *,
    prompt: str | None,
    text: str,
    usage: Mapping[str, float],
    latency_ms: float | None,
    finish_reason: str | None,
    top_probability: float | None,
) -> tuple[dict[str, float], dict[str, float]]:
    stats = _text_stats(text)
    prompt_tokens = float(usage.get("prompt_tokens") or _estimate_tokens(prompt or ""))
    completion_tokens = float(usage.get("completion_tokens") or _estimate_tokens(text))
    total_tokens = float(usage.get("total_tokens") or prompt_tokens + completion_tokens)
    latency_seconds = max(0.0, float(latency_ms or 0.0) / 1000.0)
    tokens_per_second = (
        completion_tokens / max(latency_seconds, 1e-6)
        if latency_ms is not None
        else completion_tokens
    )

    finish = (finish_reason or "").lower()
    if finish in {"stop", "completed", "complete", "success", "end_turn", "stop_sequence"}:
        stop_stability = 0.95
    elif finish in {"length", "max_tokens", "incomplete"}:
        stop_stability = 0.45
    elif finish in {"content_filter", "safety"}:
        stop_stability = 0.15
    else:
        stop_stability = 0.65

    probability = top_probability if top_probability is not None else 0.65
    probability = max(0.0, min(1.0, float(probability)))
    speed = math.tanh(tokens_per_second / 64.0)
    memory = math.tanh(total_tokens / 2048.0)
    stability = max(0.0, min(1.0, 0.65 * stop_stability + 0.35 * probability))
    frac = max(0.0, min(1.0, stats["entropy_norm"]))
    drs = math.tanh((completion_tokens - prompt_tokens) / max(1.0, total_tokens))
    gradient_source = [
        speed,
        memory,
        stability,
        frac,
        drs,
        math.tanh(stats["chars"] / 4096.0),
        math.tanh(stats["words"] / 512.0),
        probability,
    ]

    telemetry = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "latency_ms": float(latency_ms) if latency_ms is not None else 0.0,
        "tokens_per_second": tokens_per_second,
        "response_chars": stats["chars"],
        "response_words": stats["words"],
        "response_unique_chars": stats["unique_chars"],
        "response_entropy": stats["entropy"],
        "response_entropy_norm": stats["entropy_norm"],
        "top_probability": probability,
    }
    metrics = {
        "speed": speed,
        "memory": memory,
        "stability": stability,
        "frac": frac,
        "drs": drs,
        "gradient": gradient_source,
    }
    return metrics, telemetry


def _prefixed(payload: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    if not prefix:
        return dict(payload)
    return {f"{prefix}.{key}": value for key, value in payload.items()}


def _response_model(response: Any) -> str | None:
    value = _value(response, "model")
    return str(value) if value else None


def _create_openai_client(
    *,
    client: Any | None,
    client_factory: Callable[..., Any] | None,
    client_kwargs: Mapping[str, Any] | None,
) -> Any:
    if client is not None:
        return client
    if client_factory is None:
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - depends on optional SDK.
            raise RuntimeError(
                "OpenAI SDK is required for the OpenAI adapter; "
                f"install it with `{_OPENAI_INSTALL_HINT}` or pass a client."
            ) from exc
        client_factory = OpenAI
    return client_factory(**dict(client_kwargs or {}))


def _create_anthropic_client(
    *,
    client: Any | None,
    client_factory: Callable[..., Any] | None,
    client_kwargs: Mapping[str, Any] | None,
) -> Any:
    if client is not None:
        return client
    if client_factory is None:
        try:
            from anthropic import Anthropic  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - depends on optional SDK.
            raise RuntimeError(
                "Anthropic SDK is required for the Anthropic adapter; "
                f"install it with `{_ANTHROPIC_INSTALL_HINT}` or pass a client."
            ) from exc
        client_factory = Anthropic
    return client_factory(**dict(client_kwargs or {}))


def _resolve_create(root: Any, path: Sequence[str]) -> Callable[..., Any]:
    target = root
    for name in path:
        target = _value(target, name)
        if target is None:
            dotted = ".".join(path)
            raise AttributeError(f"client does not expose {dotted}.create")
    create = _value(target, "create")
    if not callable(create):
        dotted = ".".join(path)
        raise AttributeError(f"client does not expose callable {dotted}.create")
    return create


def _merge_request(
    defaults: Mapping[str, Any],
    overrides: Mapping[str, Any],
    *,
    model: str | None,
) -> dict[str, Any]:
    request = dict(defaults)
    request.update(overrides)
    if model is not None:
        request.setdefault("model", model)
    return request


def make_openai_responses_invoke(
    *,
    client: Any | None = None,
    client_factory: Callable[..., Any] | None = None,
    client_kwargs: Mapping[str, Any] | None = None,
    model: str | None = None,
    input_key: str = "input",
    **request_defaults: Any,
) -> Callable[..., Any]:
    """Return a callable that sends prompts through OpenAI's Responses API.

    The returned callable is compatible with :meth:`ApiLLMZSpaceRuntime.call`.
    It imports ``openai`` lazily, so SpiralTorch remains usable without the SDK
    until this adapter is invoked.
    """

    cached_client = client

    def _invoke(prompt: str, **request_overrides: Any) -> Any:
        nonlocal cached_client
        if cached_client is None:
            cached_client = _create_openai_client(
                client=None,
                client_factory=client_factory,
                client_kwargs=client_kwargs,
            )
        request = _merge_request(request_defaults, request_overrides, model=model)
        request.setdefault(input_key, prompt)
        create = _resolve_create(cached_client, ("responses",))
        return create(**request)

    return _invoke


def _anthropic_messages(
    prompt: str,
    *,
    messages: Sequence[Mapping[str, Any]] | None,
) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    if messages:
        result.extend(dict(message) for message in messages)
    result.append({"role": "user", "content": prompt})
    return result


def make_anthropic_messages_invoke(
    *,
    client: Any | None = None,
    client_factory: Callable[..., Any] | None = None,
    client_kwargs: Mapping[str, Any] | None = None,
    model: str | None = None,
    system: str | None = None,
    messages: Sequence[Mapping[str, Any]] | None = None,
    **request_defaults: Any,
) -> Callable[..., Any]:
    """Return a callable that sends prompts through Anthropic Messages."""

    cached_client = client

    def _invoke(prompt: str, **request_overrides: Any) -> Any:
        nonlocal cached_client
        if cached_client is None:
            cached_client = _create_anthropic_client(
                client=None,
                client_factory=client_factory,
                client_kwargs=client_kwargs,
            )
        request = _merge_request(request_defaults, request_overrides, model=model)
        if system is not None:
            request.setdefault("system", system)
        override_messages = request.pop("messages", None)
        request["messages"] = _anthropic_messages(
            prompt,
            messages=override_messages if override_messages is not None else messages,
        )
        create = _resolve_create(cached_client, ("messages",))
        return create(**request)

    return _invoke


def _chat_messages(
    prompt: str,
    *,
    system: str | None,
    messages: Sequence[Mapping[str, Any]] | None,
) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    if system:
        result.append({"role": "system", "content": system})
    if messages:
        result.extend(dict(message) for message in messages)
    result.append({"role": "user", "content": prompt})
    return result


def make_openai_chat_invoke(
    *,
    client: Any | None = None,
    client_factory: Callable[..., Any] | None = None,
    client_kwargs: Mapping[str, Any] | None = None,
    model: str | None = None,
    system: str | None = None,
    messages: Sequence[Mapping[str, Any]] | None = None,
    **request_defaults: Any,
) -> Callable[..., Any]:
    """Return a callable that sends prompts through OpenAI chat completions."""

    cached_client = client

    def _invoke(prompt: str, **request_overrides: Any) -> Any:
        nonlocal cached_client
        if cached_client is None:
            cached_client = _create_openai_client(
                client=None,
                client_factory=client_factory,
                client_kwargs=client_kwargs,
            )
        request = _merge_request(request_defaults, request_overrides, model=model)
        override_messages = request.pop("messages", None)
        request["messages"] = _chat_messages(
            prompt,
            system=system,
            messages=override_messages if override_messages is not None else messages,
        )
        create = _resolve_create(cached_client, ("chat", "completions"))
        return create(**request)

    return _invoke


def api_llm_partial_from_response(
    response: Any,
    *,
    prompt: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    latency_ms: float | None = None,
    bundle_weight: float = 1.0,
    origin: str | None = "api_llm",
    telemetry_prefix: str = "api_llm",
    gradient_dim: int = 4,
) -> ZSpacePartialBundle:
    """Convert an API-model LLM response into a Z-space partial bundle."""

    text = api_llm_text_from_response(response)
    usage = api_llm_usage_from_response(response)
    finish_reason = _finish_reason_from_response(response)
    top_probability = _top_probability_from_response(response)
    metrics, telemetry = _runtime_metrics(
        prompt=prompt,
        text=text,
        usage=usage,
        latency_ms=latency_ms,
        finish_reason=finish_reason,
        top_probability=top_probability,
    )
    gradient = list(metrics["gradient"])
    dim = max(1, int(gradient_dim))
    if len(gradient) < dim:
        gradient.extend(0.0 for _ in range(dim - len(gradient)))
    metrics["gradient"] = gradient[:dim]

    numeric_telemetry: dict[str, Any] = dict(telemetry)
    if provider:
        numeric_telemetry["provider_present"] = 1.0
    if model or _response_model(response):
        numeric_telemetry["model_present"] = 1.0
    if finish_reason:
        finish = finish_reason.lower()
        numeric_telemetry["finish_reason_stop"] = 1.0 if finish == "stop" else 0.0
        numeric_telemetry["finish_reason_refusal"] = 1.0 if finish == "refusal" else 0.0
    numeric_telemetry["empty_text"] = 1.0 if not text.strip() else 0.0

    return ZSpacePartialBundle(
        metrics,
        weight=max(0.0, float(bundle_weight)),
        origin=origin,
        telemetry=_prefixed(numeric_telemetry, telemetry_prefix),
    )


@dataclass(frozen=True)
class ApiLLMTrace:
    """Serializable trace of an API LLM response fused into Z-space."""

    provider: str | None
    model: str | None
    prompt: str | None
    text: str
    finish_reason: str | None
    latency_ms: float | None
    usage: Mapping[str, float]
    metrics: Mapping[str, Any]
    telemetry: Mapping[str, Any]
    inference: Any | None = None
    device_preflight: Mapping[str, Any] | None = None
    response_metadata: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        inference_payload = None
        if self.inference is not None:
            as_dict = getattr(self.inference, "as_dict", None)
            inference_payload = as_dict() if callable(as_dict) else self.inference
        telemetry = dict(self.telemetry)
        telemetry.update(_inference_telemetry_payload(inference_payload))
        return {
            "provider": self.provider,
            "model": self.model,
            "prompt": self.prompt,
            "text": self.text,
            "finish_reason": self.finish_reason,
            "latency_ms": self.latency_ms,
            "usage": dict(self.usage),
            "metrics": dict(self.metrics),
            "telemetry": telemetry,
            "inference": inference_payload,
            "device_preflight": None if self.device_preflight is None else dict(self.device_preflight),
            "response_metadata": None if self.response_metadata is None else dict(self.response_metadata),
        }


def _inference_telemetry_payload(inference: Any) -> dict[str, Any]:
    if not isinstance(inference, Mapping):
        return {}
    telemetry = inference.get("telemetry")
    if not isinstance(telemetry, Mapping):
        return {}
    payload = telemetry.get("payload")
    if isinstance(payload, Mapping):
        return dict(payload)
    return dict(telemetry)


def api_llm_trace_from_response(
    response: Any,
    *,
    prompt: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    latency_ms: float | None = None,
    bundle_weight: float = 1.0,
    telemetry_prefix: str = "api_llm",
    inference: Any | None = None,
    device_preflight: Mapping[str, Any] | None = None,
    gradient_dim: int = 4,
) -> ApiLLMTrace:
    """Build a serializable trace without owning the runtime lifecycle."""

    bundle = api_llm_partial_from_response(
        response,
        prompt=prompt,
        provider=provider,
        model=model,
        latency_ms=latency_ms,
        bundle_weight=bundle_weight,
        telemetry_prefix=telemetry_prefix,
        gradient_dim=gradient_dim,
    )
    return ApiLLMTrace(
        provider=provider,
        model=model or _response_model(response),
        prompt=prompt,
        text=api_llm_text_from_response(response),
        finish_reason=_finish_reason_from_response(response),
        latency_ms=latency_ms,
        usage=api_llm_usage_from_response(response),
        metrics=bundle.resolved(),
        telemetry=bundle.telemetry_payload() or {},
        inference=inference,
        device_preflight=device_preflight,
        response_metadata=_response_metadata(response),
    )


def _trace_payload(trace: ApiLLMTrace | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(trace, ApiLLMTrace):
        return trace.as_dict()
    return dict(trace)


def _normalise_context_partials(context_partials: Any) -> list[ZSpacePartialBundle]:
    if context_partials is None:
        return []
    if isinstance(context_partials, ZSpacePartialBundle):
        return [context_partials]
    if isinstance(context_partials, Mapping):
        return [ZSpacePartialBundle(context_partials)]
    if isinstance(context_partials, (str, bytes, bytearray)):
        raise TypeError(
            "context_partials must be partial mappings or ZSpacePartialBundle values"
        )

    bundles: list[ZSpacePartialBundle] = []
    try:
        iterator = iter(context_partials)
    except TypeError as exc:
        raise TypeError(
            "context_partials must be partial mappings or ZSpacePartialBundle values"
        ) from exc
    for partial in iterator:
        if partial is None:
            continue
        if isinstance(partial, ZSpacePartialBundle):
            bundles.append(partial)
        elif isinstance(partial, Mapping):
            bundles.append(ZSpacePartialBundle(partial))
        else:
            raise TypeError(
                "context_partials must contain partial mappings or ZSpacePartialBundle values"
            )
    return bundles


def _iter_wasm_report_inputs(reports: Any) -> list[tuple[str | None, Any]]:
    if reports is None:
        return []
    if isinstance(reports, (str, os.PathLike)):
        return [(None, reports)]
    if isinstance(reports, Mapping):
        if "schema" in reports or "kind" in reports:
            return [(None, reports)]
        return [(str(label), report) for label, report in reports.items()]
    if isinstance(reports, (bytes, bytearray)):
        raise TypeError("wasm reports must be paths, mappings, or sequences")
    try:
        return [(None, report) for report in reports]
    except TypeError as exc:
        raise TypeError("wasm reports must be paths, mappings, or sequences") from exc


def api_llm_wasm_context_partials(
    reports: Any,
    *,
    bundle_weight: float = 1.0,
    origin: str | None = None,
    telemetry_prefix: str = "wasm",
    gradient_dim: int = 8,
) -> list[ZSpacePartialBundle]:
    """Convert one or more WASM reports into API-LLM runtime context partials."""

    from .wasm_reports import wasm_report_to_zspace_partial

    partials: list[ZSpacePartialBundle] = []
    for label, report in _iter_wasm_report_inputs(reports):
        partial_origin = origin
        if partial_origin is None and label:
            partial_origin = f"wasm:{label}"
        partials.append(
            wasm_report_to_zspace_partial(
                report,
                bundle_weight=bundle_weight,
                origin=partial_origin,
                telemetry_prefix=telemetry_prefix,
                gradient_dim=gradient_dim,
            )
        )
    return partials


def write_api_llm_trace_jsonl(
    traces: Iterable[ApiLLMTrace | Mapping[str, Any]],
    path: str | Path,
    *,
    event_type: str = "ApiLLMTrace",
) -> str:
    """Write API LLM Z-space traces as JSONL events."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for step, trace in enumerate(traces):
            payload = _trace_payload(trace)
            record = {
                "event_type": event_type,
                "schema": _API_LLM_TRACE_SCHEMA,
                "step": step,
                "ts": time.time(),
                "payload": payload,
            }
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    return str(out_path)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                yield row


def _normalise_trace_record(
    record: Mapping[str, Any],
    *,
    event_type: str,
) -> dict[str, Any] | None:
    payload = record.get("payload")
    record_type = record.get("event_type") or record.get("type") or record.get("kind")
    if isinstance(payload, Mapping) and (record_type in {event_type, None}):
        event = dict(payload)
        telemetry = dict(_mapping_at(event, "telemetry"))
        telemetry.update(_inference_telemetry_payload(event.get("inference")))
        if telemetry:
            event["telemetry"] = telemetry
        for key in ("step", "ts", "schema"):
            if key in record and key not in event:
                event[key] = record[key]
        return event
    if {"text", "metrics", "usage"}.issubset(record.keys()):
        event = dict(record)
        telemetry = dict(_mapping_at(event, "telemetry"))
        telemetry.update(_inference_telemetry_payload(event.get("inference")))
        if telemetry:
            event["telemetry"] = telemetry
        return event
    return None


def load_api_llm_trace_events(
    path: str | Path,
    *,
    event_type: str = "ApiLLMTrace",
) -> list[dict[str, Any]]:
    """Load API LLM Z-space trace JSONL rows written by SpiralTorch."""

    events: list[dict[str, Any]] = []
    for record in _iter_jsonl(Path(path)):
        event = _normalise_trace_record(record, event_type=event_type)
        if event is not None:
            events.append(event)
    return events


def _count_labels(values: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if value is None or value == "":
            continue
        label = str(value)
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _stats(values: Iterable[Any]) -> dict[str, float]:
    numeric = [value for value in (_finite_float(item) for item in values) if value is not None]
    if not numeric:
        return {"count": 0.0, "min": 0.0, "max": 0.0, "mean": 0.0, "last": 0.0}
    return {
        "count": float(len(numeric)),
        "min": min(numeric),
        "max": max(numeric),
        "mean": sum(numeric) / len(numeric),
        "last": numeric[-1],
    }


def _mapping_at(source: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = source.get(key)
    return value if isinstance(value, Mapping) else {}


def _trace_confidence(event: Mapping[str, Any]) -> float | None:
    inference = _mapping_at(event, "inference")
    return _finite_float(inference.get("confidence"))


def _trace_runtime_status(event: Mapping[str, Any]) -> str | None:
    preflight = _mapping_at(event, "device_preflight")
    for key in (
        "runtime_status",
        "effective_backend_runtime_status",
        "requested_backend_runtime_status",
    ):
        value = preflight.get(key)
        if value:
            return str(value)
    return None


def _trace_runtime_ready(event: Mapping[str, Any]) -> bool | None:
    preflight = _mapping_at(event, "device_preflight")
    value = preflight.get("runtime_ready")
    if isinstance(value, bool):
        return value
    value = preflight.get("effective_backend_runtime_ready")
    return value if isinstance(value, bool) else None


def _preview(value: Any, *, limit: int = 160) -> str | None:
    if value is None:
        return None
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def _trace_step(event: Mapping[str, Any]) -> int | None:
    step = event.get("step")
    if isinstance(step, bool):
        return None
    if isinstance(step, (int, float)):
        return int(step)
    return None


def _finish_reason_label(event: Mapping[str, Any]) -> str:
    value = event.get("finish_reason")
    return str(value).lower() if value else ""


def _trace_has_text(event: Mapping[str, Any]) -> bool:
    return bool(str(event.get("text") or "").strip())


def _trace_stop_details_category(event: Mapping[str, Any]) -> str | None:
    metadata = _mapping_at(event, "response_metadata")
    stop_details = metadata.get("stop_details")
    if not isinstance(stop_details, Mapping):
        return None
    category = stop_details.get("category")
    return str(category) if category not in {None, ""} else None


def _wasm_ready_summary(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, float]:
    values = [_finite_float(row.get(key)) for row in rows]
    observed = [value for value in values if value is not None]
    ready_count = sum(1 for value in observed if value > 0.0)
    return {
        "observed_count": float(len(observed)),
        "ready_count": float(ready_count),
        "ready_rate": ready_count / len(observed) if observed else 0.0,
    }


def _summarize_wasm_context_telemetry(
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize browser-side WASM telemetry fused into API LLM traces."""

    rows = [
        _mapping_at(event, "telemetry")
        for event in events
        if any(str(key).startswith("wasm.") for key in _mapping_at(event, "telemetry"))
    ]
    families: dict[str, int] = {}
    for row in rows:
        for key, value in row.items():
            key_text = str(key)
            if not key_text.startswith("wasm.family_"):
                continue
            if (_finite_float(value) or 0.0) <= 0.0:
                continue
            family = key_text.removeprefix("wasm.family_")
            families[family] = families.get(family, 0) + 1
    return {
        "observed_count": len(rows),
        "observed_rate": len(rows) / len(events) if events else 0.0,
        "families": dict(sorted(families.items())),
        "loss": _stats(row.get("wasm.loss") for row in rows),
        "stability_hint": _stats(row.get("wasm.stability_hint") for row in rows),
        "work_units": _stats(row.get("wasm.work_units") for row in rows),
        "webgpu_available": _wasm_ready_summary(rows, "wasm.webgpu_available"),
        "webgpu_device_ready": _wasm_ready_summary(
            rows,
            "wasm.webgpu_device_ready",
        ),
    }


def summarize_api_llm_trace_events(
    path: str | Path,
    *,
    event_type: str = "ApiLLMTrace",
) -> dict[str, Any]:
    """Summarize API LLM Z-space trace JSONL rows for experiment comparison."""

    events = load_api_llm_trace_events(path, event_type=event_type)
    usage_rows = [_mapping_at(event, "usage") for event in events]
    metric_rows = [_mapping_at(event, "metrics") for event in events]
    text_quality_rows = [_trace_text_quality(event) for event in events]
    confidence_values = [_trace_confidence(event) for event in events]
    runtime_ready_values = [_trace_runtime_ready(event) for event in events]
    ready_observed = [value for value in runtime_ready_values if value is not None]
    ready_count = sum(1 for value in ready_observed if value)
    text_lengths = [len(str(event.get("text") or "")) for event in events]
    empty_text_count = sum(1 for event in events if not _trace_has_text(event))
    refusal_count = sum(1 for event in events if _finish_reason_label(event) == "refusal")
    incomplete_count = sum(
        1
        for event in events
        if _finish_reason_label(event) in {"incomplete", "length", "max_tokens"}
    )
    completed_count = sum(
        1
        for event in events
        if _finish_reason_label(event)
        in {"stop", "completed", "complete", "success", "end_turn", "stop_sequence"}
    )

    metric_keys = sorted(
        {
            key
            for row in metric_rows
            for key, value in row.items()
            if key != "gradient" and _finite_float(value) is not None
        }
    )
    metrics = {
        key: _stats(row.get(key) for row in metric_rows)
        for key in metric_keys
    }
    usage = {
        "prompt_tokens": _stats(row.get("prompt_tokens") for row in usage_rows),
        "completion_tokens": _stats(row.get("completion_tokens") for row in usage_rows),
        "total_tokens": _stats(row.get("total_tokens") for row in usage_rows),
    }
    total_tokens = sum(
        value
        for value in (_finite_float(row.get("total_tokens")) for row in usage_rows)
        if value is not None
    )
    latency = _stats(event.get("latency_ms") for event in events)
    first = events[0] if events else {}
    last = events[-1] if events else {}
    return {
        "event_type": event_type,
        "schema": _API_LLM_TRACE_SCHEMA,
        "count": len(events),
        "first_step": _trace_step(first),
        "last_step": _trace_step(last),
        "first_text_preview": _preview(first.get("text")),
        "last_text_preview": _preview(last.get("text")),
        "providers": _count_labels(event.get("provider") for event in events),
        "models": _count_labels(event.get("model") for event in events),
        "finish_reasons": _count_labels(event.get("finish_reason") for event in events),
        "stop_detail_categories": _count_labels(
            _trace_stop_details_category(event) for event in events
        ),
        "empty_text_count": empty_text_count,
        "empty_text_rate": empty_text_count / len(events) if events else 0.0,
        "refusal_count": refusal_count,
        "refusal_rate": refusal_count / len(events) if events else 0.0,
        "incomplete_count": incomplete_count,
        "incomplete_rate": incomplete_count / len(events) if events else 0.0,
        "completed_count": completed_count,
        "completion_rate": completed_count / len(events) if events else 0.0,
        "runtime_statuses": _count_labels(_trace_runtime_status(event) for event in events),
        "runtime_ready_count": ready_count,
        "runtime_ready_rate": ready_count / len(ready_observed) if ready_observed else 0.0,
        "usage": usage,
        "total_tokens": total_tokens,
        "latency_ms": latency,
        "text_chars": _stats(text_lengths),
        "text_quality": {
            "prompt_coverage": _stats(
                row.get("prompt_coverage") for row in text_quality_rows
            ),
            "prompt_echo_rate": _stats(
                row.get("prompt_echo_rate") for row in text_quality_rows
            ),
            "response_signal_rate": _stats(
                row.get("response_signal_rate") for row in text_quality_rows
            ),
            "repetition_rate": _stats(
                row.get("repetition_rate") for row in text_quality_rows
            ),
            "text_quality_score": _stats(
                row.get("text_quality_score") for row in text_quality_rows
            ),
        },
        "confidence": _stats(confidence_values),
        "metrics": metrics,
        "wasm_context": _summarize_wasm_context_telemetry(events),
    }


def _summary_stat(
    summary: Mapping[str, Any],
    section: str,
    *,
    key: str | None = None,
    stat: str = "mean",
) -> float:
    source = summary.get(section)
    if key is not None and isinstance(source, Mapping):
        source = source.get(key)
    if isinstance(source, Mapping):
        numeric = _finite_float(source.get(stat))
        return 0.0 if numeric is None else numeric
    numeric = _finite_float(source)
    return 0.0 if numeric is None else numeric


def _summary_metric(
    summary: Mapping[str, Any],
    metric: str,
    *,
    stat: str = "mean",
) -> float:
    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping):
        return 0.0
    source = metrics.get(metric)
    if not isinstance(source, Mapping):
        return 0.0
    numeric = _finite_float(source.get(stat))
    return 0.0 if numeric is None else numeric


def _wasm_context_stat(
    summary: Mapping[str, Any],
    section: str,
    *,
    stat: str = "mean",
) -> float | None:
    wasm_context = summary.get("wasm_context")
    if not isinstance(wasm_context, Mapping):
        return None
    source = wasm_context.get(section)
    if not isinstance(source, Mapping):
        return None
    if (_finite_float(source.get("count")) or 0.0) <= 0.0:
        return None
    return _finite_float(source.get(stat))


def _wasm_context_ready_rate(
    summary: Mapping[str, Any],
    section: str,
) -> float | None:
    wasm_context = summary.get("wasm_context")
    if not isinstance(wasm_context, Mapping):
        return None
    source = wasm_context.get(section)
    if not isinstance(source, Mapping):
        return None
    if (_finite_float(source.get("observed_count")) or 0.0) <= 0.0:
        return None
    return _finite_float(source.get("ready_rate"))


def _dominant_label(counts: Any) -> str | None:
    if not isinstance(counts, Mapping) or not counts:
        return None
    return max(
        ((str(label), int(count)) for label, count in counts.items()),
        key=lambda item: (item[1], item[0]),
    )[0]


def _trace_entries(
    traces: Mapping[str, str | Path] | Sequence[str | Path] | str | Path,
    *,
    labels: Sequence[str] | None,
) -> list[tuple[str, Path]]:
    if isinstance(traces, Mapping):
        return [(str(label), Path(path)) for label, path in traces.items()]
    if isinstance(traces, (str, Path)):
        path = Path(traces)
        return [(path.stem or "run_0", path)]
    paths = list(traces)
    explicit = list(labels or [])
    entries: list[tuple[str, Path]] = []
    for index, raw_path in enumerate(paths):
        path = Path(raw_path)
        label = explicit[index] if index < len(explicit) else path.stem
        entries.append((label or f"run_{index}", path))
    return entries


def _safe_trace_label(label: str, *, fallback: str) -> str:
    safe = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "-"
        for char in label
    )
    safe = safe.strip(".-_")
    return safe or fallback


def _bounded_unit(value: Any) -> float:
    return max(0.0, min(1.0, _finite_float(value) or 0.0))


def _quality_score(row: Mapping[str, Any]) -> float:
    confidence = _finite_float(row.get("confidence_mean")) or 0.0
    stability = _finite_float(row.get("stability_mean")) or 0.0
    frac = _finite_float(row.get("frac_mean")) or 0.0
    return max(
        0.0,
        min(1.0, 0.5 * confidence + 0.3 * stability + 0.2 * frac),
    )


def _latency_cost(row: Mapping[str, Any]) -> float:
    latency = max(0.0, _finite_float(row.get("latency_ms_mean")) or 0.0)
    return min(1.0, math.log1p(latency) / math.log1p(10_000.0))


def _token_cost(row: Mapping[str, Any]) -> float:
    tokens = max(0.0, _finite_float(row.get("total_tokens")) or 0.0)
    return min(1.0, math.log1p(tokens) / math.log1p(4096.0))


def _health_penalty(row: Mapping[str, Any]) -> float:
    empty_text_rate = _bounded_unit(row.get("empty_text_rate"))
    refusal_rate = _bounded_unit(row.get("refusal_rate"))
    incomplete_rate = _bounded_unit(row.get("incomplete_rate"))
    return 0.35 * empty_text_rate + 0.25 * refusal_rate + 0.08 * incomplete_rate


def _route_score(row: Mapping[str, Any]) -> float:
    quality = _quality_score(row)
    runtime_ready = _bounded_unit(row.get("runtime_ready_rate"))
    latency_penalty = 0.05 * _latency_cost(row)
    token_penalty = 0.05 * _token_cost(row)
    health_penalty = _health_penalty(row)
    return max(
        0.0,
        min(
            1.0,
            quality
            + 0.05 * runtime_ready
            - latency_penalty
            - token_penalty
            - health_penalty,
        ),
    )


def _efficiency_score(row: Mapping[str, Any]) -> float:
    quality = _quality_score(row)
    runtime_ready = _bounded_unit(row.get("runtime_ready_rate"))
    cost_reward = 0.5 * (1.0 - _latency_cost(row)) + 0.5 * (1.0 - _token_cost(row))
    return max(
        0.0,
        min(
            1.0,
            0.60 * quality
            + 0.35 * cost_reward
            + 0.05 * runtime_ready
            - _health_penalty(row),
        ),
    )


def _selection_profile_score(row: Mapping[str, Any], profile: str) -> float:
    quality = _quality_score(row)
    text_quality = _bounded_unit(row.get("text_quality_score"))
    completion = _bounded_unit(row.get("completion_rate"))
    runtime_ready = _bounded_unit(row.get("runtime_ready_rate"))
    latency_reward = 1.0 - _latency_cost(row)
    token_reward = 1.0 - _token_cost(row)
    health_penalty = _health_penalty(row)
    if profile == "quality":
        score = (
            0.65 * quality
            + 0.20 * text_quality
            + 0.10 * completion
            + 0.05 * runtime_ready
        )
    elif profile == "grounded":
        score = (
            0.55 * text_quality
            + 0.25 * quality
            + 0.15 * completion
            + 0.05 * runtime_ready
        )
    elif profile == "efficiency":
        score = (
            0.40 * quality
            + 0.20 * text_quality
            + 0.20 * latency_reward
            + 0.15 * token_reward
            + 0.05 * runtime_ready
        )
    elif profile == "latency":
        score = (
            0.35 * latency_reward
            + 0.25 * quality
            + 0.20 * text_quality
            + 0.15 * completion
            + 0.05 * runtime_ready
        )
    else:
        score = _route_score(row)
    return max(0.0, min(1.0, score - health_penalty))


def _comparison_row(label: str, path: Path, summary: Mapping[str, Any]) -> dict[str, Any]:
    wasm_context = summary.get("wasm_context")
    wasm_context_map = wasm_context if isinstance(wasm_context, Mapping) else {}
    row: dict[str, Any] = {
        "label": label,
        "path": str(path),
        "count": int(summary.get("count") or 0),
        "provider": _dominant_label(summary.get("providers")),
        "model": _dominant_label(summary.get("models")),
        "runtime_status": _dominant_label(summary.get("runtime_statuses")),
        "runtime_ready_rate": _finite_float(summary.get("runtime_ready_rate")) or 0.0,
        "completion_rate": _finite_float(summary.get("completion_rate")) or 0.0,
        "incomplete_rate": _finite_float(summary.get("incomplete_rate")) or 0.0,
        "empty_text_rate": _finite_float(summary.get("empty_text_rate")) or 0.0,
        "refusal_rate": _finite_float(summary.get("refusal_rate")) or 0.0,
        "stop_detail_category": _dominant_label(summary.get("stop_detail_categories")),
        "total_tokens": _finite_float(summary.get("total_tokens")) or 0.0,
        "latency_ms_mean": _summary_stat(summary, "latency_ms"),
        "confidence_mean": _summary_stat(summary, "confidence"),
        "text_chars_mean": _summary_stat(summary, "text_chars"),
        "prompt_coverage_mean": _summary_stat(
            summary,
            "text_quality",
            key="prompt_coverage",
        ),
        "prompt_echo_rate_mean": _summary_stat(
            summary,
            "text_quality",
            key="prompt_echo_rate",
        ),
        "response_signal_rate_mean": _summary_stat(
            summary,
            "text_quality",
            key="response_signal_rate",
        ),
        "repetition_rate_mean": _summary_stat(
            summary,
            "text_quality",
            key="repetition_rate",
        ),
        "text_quality_score": _summary_stat(
            summary,
            "text_quality",
            key="text_quality_score",
        ),
        "prompt_tokens_mean": _summary_stat(summary, "usage", key="prompt_tokens"),
        "completion_tokens_mean": _summary_stat(summary, "usage", key="completion_tokens"),
        "stability_mean": _summary_metric(summary, "stability"),
        "speed_mean": _summary_metric(summary, "speed"),
        "memory_mean": _summary_metric(summary, "memory"),
        "frac_mean": _summary_metric(summary, "frac"),
        "drs_mean": _summary_metric(summary, "drs"),
        "wasm_context_observed_count": int(
            _finite_float(wasm_context_map.get("observed_count")) or 0
        ),
        "wasm_context_observed_rate": _finite_float(
            wasm_context_map.get("observed_rate")
        )
        or 0.0,
        "wasm_family": _dominant_label(wasm_context_map.get("families")),
        "wasm_loss_mean": _wasm_context_stat(summary, "loss"),
        "wasm_loss_last": _wasm_context_stat(summary, "loss", stat="last"),
        "wasm_stability_hint_mean": _wasm_context_stat(summary, "stability_hint"),
        "wasm_work_units_mean": _wasm_context_stat(summary, "work_units"),
        "wasm_webgpu_available_rate": _wasm_context_ready_rate(
            summary,
            "webgpu_available",
        ),
        "wasm_webgpu_device_ready_rate": _wasm_context_ready_rate(
            summary,
            "webgpu_device_ready",
        ),
        "last_text_preview": summary.get("last_text_preview"),
    }
    row["quality_score"] = _quality_score(row)
    row["latency_cost"] = _latency_cost(row)
    row["token_cost"] = _token_cost(row)
    row["health_penalty"] = _health_penalty(row)
    row["efficiency_score"] = _efficiency_score(row)
    row["route_score"] = _route_score(row)
    row["selection_scores"] = {
        profile: _selection_profile_score(row, profile)
        for profile in _SELECTION_PROFILES
    }
    return row


def _winner(
    rows: Sequence[Mapping[str, Any]],
    key: str,
    *,
    higher_is_better: bool = True,
) -> str | None:
    candidates: list[tuple[float, str]] = []
    for row in rows:
        if int(row.get("count") or 0) <= 0:
            continue
        value = _finite_float(row.get(key))
        label = row.get("label")
        if value is not None and label is not None:
            score = value if higher_is_better else -value
            candidates.append((score, str(label)))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _near_best_routes(
    rows: Sequence[Mapping[str, Any]],
    *,
    tolerance: float,
) -> list[dict[str, Any]]:
    candidates: list[tuple[float, Mapping[str, Any]]] = []
    for row in rows:
        if int(row.get("count") or 0) <= 0:
            continue
        score = _finite_float(row.get("route_score"))
        if score is not None:
            candidates.append((score, row))
    if not candidates:
        return []
    best_score = max(score for score, _row in candidates)
    bounded_tolerance = max(0.0, float(tolerance))
    near: list[dict[str, Any]] = []
    for score, row in sorted(candidates, key=lambda item: item[0], reverse=True):
        delta = best_score - score
        if delta > bounded_tolerance:
            continue
        near.append(
            {
                "label": row.get("label"),
                "route_score": score,
                "route_score_delta": delta,
                "quality_score": _finite_float(row.get("quality_score")) or 0.0,
                "text_quality_score": _finite_float(row.get("text_quality_score")) or 0.0,
                "prompt_coverage_mean": _finite_float(
                    row.get("prompt_coverage_mean")
                )
                or 0.0,
                "efficiency_score": _finite_float(row.get("efficiency_score")) or 0.0,
                "latency_ms_mean": _finite_float(row.get("latency_ms_mean")) or 0.0,
                "total_tokens": _finite_float(row.get("total_tokens")) or 0.0,
                "completion_rate": _finite_float(row.get("completion_rate")) or 0.0,
                "empty_text_rate": _finite_float(row.get("empty_text_rate")) or 0.0,
                "refusal_rate": _finite_float(row.get("refusal_rate")) or 0.0,
                "wasm_family": row.get("wasm_family"),
                "wasm_loss_mean": _finite_float(row.get("wasm_loss_mean")),
                "wasm_webgpu_device_ready_rate": _finite_float(
                    row.get("wasm_webgpu_device_ready_rate")
                ),
            }
        )
    return near


def _selection_profiles(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    for profile in _SELECTION_PROFILES:
        candidates: list[tuple[float, Mapping[str, Any]]] = []
        for row in rows:
            if int(row.get("count") or 0) <= 0:
                continue
            selection_scores = row.get("selection_scores")
            score = (
                _finite_float(selection_scores.get(profile))
                if isinstance(selection_scores, Mapping)
                else None
            )
            if score is not None:
                candidates.append((score, row))
        if not candidates:
            profiles[profile] = {"label": None, "score": 0.0}
            continue
        score, row = max(
            candidates,
            key=lambda item: (
                item[0],
                _finite_float(item[1].get("route_score")) or 0.0,
                str(item[1].get("label")),
            ),
        )
        profiles[profile] = {
            "label": row.get("label"),
            "score": score,
            "route_score": _finite_float(row.get("route_score")) or 0.0,
            "quality_score": _finite_float(row.get("quality_score")) or 0.0,
            "text_quality_score": _finite_float(row.get("text_quality_score")) or 0.0,
            "efficiency_score": _finite_float(row.get("efficiency_score")) or 0.0,
            "latency_ms_mean": _finite_float(row.get("latency_ms_mean")) or 0.0,
            "total_tokens": _finite_float(row.get("total_tokens")) or 0.0,
            "completion_rate": _finite_float(row.get("completion_rate")) or 0.0,
            "wasm_loss_mean": _finite_float(row.get("wasm_loss_mean")),
            "wasm_webgpu_device_ready_rate": _finite_float(
                row.get("wasm_webgpu_device_ready_rate")
            ),
        }
    return profiles


def _comparison_wasm_context(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    observed = [
        row
        for row in rows
        if int(row.get("count") or 0) > 0
        and int(row.get("wasm_context_observed_count") or 0) > 0
    ]
    families: dict[str, int] = {}
    for row in observed:
        family = row.get("wasm_family")
        if family in {None, ""}:
            continue
        family_text = str(family)
        families[family_text] = families.get(family_text, 0) + 1
    return {
        "observed_runs": len(observed),
        "observed_run_rate": len(observed) / len(rows) if rows else 0.0,
        "families": dict(sorted(families.items())),
        "lowest_loss": _winner(rows, "wasm_loss_mean", higher_is_better=False),
        "highest_stability_hint": _winner(rows, "wasm_stability_hint_mean"),
        "highest_webgpu_device_ready": _winner(rows, "wasm_webgpu_device_ready_rate"),
    }


def _read_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return payload


def _report_entries(
    reports: Mapping[str, str | Path] | Sequence[str | Path] | str | Path,
    *,
    labels: Sequence[str] | None,
) -> list[tuple[str, Path]]:
    if isinstance(reports, Mapping):
        return [(str(label), Path(path)) for label, path in reports.items()]
    if isinstance(reports, (str, Path)):
        path = Path(reports)
        return [(path.parent.name or path.stem or "report_0", path)]
    paths = list(reports)
    explicit = list(labels or [])
    entries: list[tuple[str, Path]] = []
    for index, raw_path in enumerate(paths):
        path = Path(raw_path)
        label = explicit[index] if index < len(explicit) else path.parent.name or path.stem
        entries.append((str(label or f"report_{index}"), path))
    return entries


def _report_profile_winners(
    selection_profiles: Mapping[str, Any],
) -> tuple[dict[str, str | None], dict[str, float]]:
    winners: dict[str, str | None] = {}
    scores: dict[str, float] = {}
    for profile in _SELECTION_PROFILES:
        payload = selection_profiles.get(profile)
        if isinstance(payload, Mapping):
            label = payload.get("label")
            winners[profile] = str(label) if label not in {None, ""} else None
            scores[profile] = _finite_float(payload.get("score")) or 0.0
        else:
            winners[profile] = None
            scores[profile] = 0.0
    return winners, scores


def _counts_from_sequence(values: Any) -> dict[str, int]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return {}
    counts: dict[str, int] = {}
    for value in values:
        if value is None:
            continue
        label = str(value)
        if not label:
            continue
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _list_from_sequence(values: Any) -> list[Any]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return []
    return list(values)


def _wasm_context_report_rows(wasm_context: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    reports = wasm_context.get("reports")
    if not isinstance(reports, Sequence) or isinstance(reports, (str, bytes, bytearray)):
        return []
    return [row for row in reports if isinstance(row, Mapping)]


def _wasm_context_report_count(wasm_context: Mapping[str, Any]) -> int:
    count = _finite_float(wasm_context.get("report_count"))
    if count is not None:
        return int(count)
    return len(_wasm_context_report_rows(wasm_context))


def _wasm_context_families(wasm_context: Mapping[str, Any]) -> dict[str, int]:
    comparison = wasm_context.get("comparison")
    comparison_map = comparison if isinstance(comparison, Mapping) else {}
    families = comparison_map.get("families")
    if isinstance(families, Mapping):
        result: dict[str, int] = {}
        for label, count in families.items():
            numeric = _finite_float(count)
            if numeric is not None and numeric > 0.0:
                result[str(label)] = int(numeric)
        if result:
            return dict(sorted(result.items()))

    result: dict[str, int] = {}
    for row in _wasm_context_report_rows(wasm_context):
        family = row.get("family")
        if family in {None, ""}:
            continue
        family_text = str(family)
        result[family_text] = result.get(family_text, 0) + 1
    return dict(sorted(result.items()))


def _wasm_context_best_value(
    wasm_context: Mapping[str, Any],
    key: str,
    *,
    comparison_key: str,
    higher_is_better: bool,
) -> float | None:
    comparison = wasm_context.get("comparison")
    comparison_map = comparison if isinstance(comparison, Mapping) else {}
    row = comparison_map.get(comparison_key)
    if isinstance(row, Mapping):
        value = _finite_float(row.get(key))
        if value is not None:
            return value

    values = [
        value
        for value in (
            _finite_float(row.get(key))
            for row in _wasm_context_report_rows(wasm_context)
        )
        if value is not None
    ]
    if not values:
        return None
    return max(values) if higher_is_better else min(values)


def _wasm_context_ready_rate_from_reports(wasm_context: Mapping[str, Any]) -> float | None:
    values = [
        row.get("webgpu_device_ready")
        for row in _wasm_context_report_rows(wasm_context)
        if row.get("webgpu_device_ready") is not None
    ]
    if not values:
        return None
    ready = sum(1 for value in values if bool(value))
    return ready / len(values)


def _matrix_report_row(label: str, path: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    comparison = report.get("comparison")
    comparison_map = comparison if isinstance(comparison, Mapping) else {}
    winners = comparison_map.get("winners")
    winners_map = winners if isinstance(winners, Mapping) else {}
    selection_profiles = comparison_map.get("selection_profiles")
    selection_map = selection_profiles if isinstance(selection_profiles, Mapping) else {}
    profile_winners, profile_scores = _report_profile_winners(selection_map)
    skipped = report.get("skipped")
    skipped_map = dict(skipped) if isinstance(skipped, Mapping) else {}
    client_errors = report.get("client_errors")
    client_error_count = (
        len(client_errors)
        if isinstance(client_errors, Sequence) and not isinstance(client_errors, (str, bytes))
        else int(_finite_float(report.get("client_error_count")) or 0)
    )
    near_best = comparison_map.get("near_best")
    wasm_context = report.get("wasm_context")
    wasm_context_map = wasm_context if isinstance(wasm_context, Mapping) else {}
    return {
        "label": label,
        "path": str(path),
        "kind": report.get("kind"),
        "created_at": report.get("created_at"),
        "prompt_count": int(_finite_float(report.get("prompt_count")) or 0),
        "route_count": int(_finite_float(report.get("route_count")) or 0),
        "near_best_tolerance": _finite_float(report.get("near_best_tolerance")) or 0.0,
        "best_score": winners_map.get("best_score"),
        "highest_quality": winners_map.get("highest_quality"),
        "highest_text_quality": winners_map.get("highest_text_quality"),
        "highest_efficiency": winners_map.get("highest_efficiency"),
        "lowest_latency": winners_map.get("lowest_latency"),
        "lowest_total_tokens": winners_map.get("lowest_total_tokens"),
        "profile_winners": profile_winners,
        "profile_scores": profile_scores,
        "near_best_count": len(near_best) if isinstance(near_best, Sequence) else 0,
        "skipped": skipped_map,
        "skipped_count": len(skipped_map),
        "client_error_count": client_error_count,
        "wasm_report_count": _wasm_context_report_count(wasm_context_map),
        "wasm_context_origins": _list_from_sequence(
            wasm_context_map.get("context_origins")
        ),
        "wasm_context_origin_counts": _counts_from_sequence(
            wasm_context_map.get("context_origins")
        ),
        "wasm_families": _wasm_context_families(wasm_context_map),
        "wasm_best_loss": _wasm_context_best_value(
            wasm_context_map,
            "loss",
            comparison_key="best_loss",
            higher_is_better=False,
        ),
        "wasm_best_stability": _wasm_context_best_value(
            wasm_context_map,
            "stability",
            comparison_key="best_stability",
            higher_is_better=True,
        ),
        "wasm_webgpu_device_ready_rate": _wasm_context_ready_rate_from_reports(
            wasm_context_map
        ),
    }


def _matrix_report_winner(
    rows: Sequence[Mapping[str, Any]],
    key: str,
    *,
    higher_is_better: bool = True,
) -> str | None:
    candidates: list[tuple[float, str]] = []
    for row in rows:
        if int(row.get("wasm_report_count") or 0) <= 0:
            continue
        value = _finite_float(row.get(key))
        label = row.get("label")
        if value is None or label in {None, ""}:
            continue
        score = value if higher_is_better else -value
        candidates.append((score, str(label)))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _matrix_reports_wasm_context(
    report_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    observed = [
        row
        for row in report_rows
        if int(row.get("wasm_report_count") or 0) > 0
    ]
    family_counts: dict[str, int] = {}
    origin_counts: dict[str, int] = {}
    for row in observed:
        families = row.get("wasm_families")
        if isinstance(families, Mapping):
            for family, count in families.items():
                numeric = int(_finite_float(count) or 0)
                if numeric > 0:
                    family_text = str(family)
                    family_counts[family_text] = (
                        family_counts.get(family_text, 0) + numeric
                    )
        origins = row.get("wasm_context_origin_counts")
        if isinstance(origins, Mapping):
            for origin, count in origins.items():
                numeric = int(_finite_float(count) or 0)
                if numeric > 0:
                    origin_text = str(origin)
                    origin_counts[origin_text] = (
                        origin_counts.get(origin_text, 0) + numeric
                    )

    return {
        "observed_reports": len(observed),
        "observed_report_rate": (
            len(observed) / len(report_rows) if report_rows else 0.0
        ),
        "total_wasm_report_count": sum(
            int(row.get("wasm_report_count") or 0) for row in observed
        ),
        "families": dict(sorted(family_counts.items())),
        "context_origins": dict(sorted(origin_counts.items())),
        "lowest_best_loss": _matrix_report_winner(
            report_rows,
            "wasm_best_loss",
            higher_is_better=False,
        ),
        "highest_best_stability": _matrix_report_winner(
            report_rows,
            "wasm_best_stability",
        ),
        "highest_webgpu_device_ready": _matrix_report_winner(
            report_rows,
            "wasm_webgpu_device_ready_rate",
        ),
    }


def _report_run_rows(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    comparison = report.get("comparison")
    if not isinstance(comparison, Mapping):
        return []
    runs = comparison.get("runs")
    if not isinstance(runs, Sequence) or isinstance(runs, (str, bytes)):
        return []
    return [row for row in runs if isinstance(row, Mapping)]


def _profile_winner_summary(
    report_rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    total_reports = max(1, len(report_rows))
    result: dict[str, list[dict[str, Any]]] = {}
    for profile in _SELECTION_PROFILES:
        counts: dict[str, int] = {}
        scores: dict[str, list[float]] = {}
        observed_reports = 0
        for row in report_rows:
            winners = row.get("profile_winners")
            profile_scores = row.get("profile_scores")
            label = winners.get(profile) if isinstance(winners, Mapping) else None
            if label in {None, ""}:
                continue
            observed_reports += 1
            label_value = str(label)
            counts[label_value] = counts.get(label_value, 0) + 1
            if isinstance(profile_scores, Mapping):
                score = _finite_float(profile_scores.get(profile))
                if score is not None:
                    scores.setdefault(label_value, []).append(score)
        result[profile] = [
            {
                "label": label,
                "wins": wins,
                "observed_reports": observed_reports,
                "report_coverage": observed_reports / total_reports,
                "win_rate": wins / max(1, observed_reports),
                "score_mean": _stats(scores.get(label, []))["mean"],
            }
            for label, wins in sorted(
                counts.items(),
                key=lambda item: (item[1], item[0]),
                reverse=True,
            )
        ]
    return result


def _route_report_summary(
    report_rows: Sequence[Mapping[str, Any]],
    reports_by_label: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    route_values: dict[str, dict[str, list[float]]] = {}
    best_wins: dict[str, int] = {}
    profile_wins: dict[str, dict[str, int]] = {}
    appearances: dict[str, int] = {}
    for report_row in report_rows:
        report_label = str(report_row.get("label"))
        report = reports_by_label.get(report_label, {})
        best = report_row.get("best_score")
        if best not in {None, ""}:
            best_label = str(best)
            best_wins[best_label] = best_wins.get(best_label, 0) + 1
        winners = report_row.get("profile_winners")
        if isinstance(winners, Mapping):
            for profile, route_label in winners.items():
                if route_label in {None, ""}:
                    continue
                route_value = str(route_label)
                profile_wins.setdefault(route_value, {})
                profile_wins[route_value][str(profile)] = (
                    profile_wins[route_value].get(str(profile), 0) + 1
                )
        for run_row in _report_run_rows(report):
            route_label = run_row.get("label")
            if route_label in {None, ""}:
                continue
            route_value = str(route_label)
            appearances[route_value] = appearances.get(route_value, 0) + 1
            metrics = route_values.setdefault(route_value, {})
            for metric in _REPORT_ROUTE_METRICS:
                value = _finite_float(run_row.get(metric))
                if value is not None:
                    metrics.setdefault(metric, []).append(value)

    summaries: list[dict[str, Any]] = []
    for route_label, metrics in route_values.items():
        profile_win_counts = {
            profile: profile_wins.get(route_label, {}).get(profile, 0)
            for profile in _SELECTION_PROFILES
        }
        row: dict[str, Any] = {
            "label": route_label,
            "appearances": appearances.get(route_label, 0),
            "best_score_wins": best_wins.get(route_label, 0),
            "profile_wins": profile_win_counts,
            "profile_win_total": sum(profile_win_counts.values()),
        }
        for metric in _REPORT_ROUTE_METRICS:
            stats = _stats(metrics.get(metric, []))
            prefix = metric[:-5] if metric.endswith("_mean") else metric
            row[f"{prefix}_mean"] = stats["mean"]
            row[f"{prefix}_min"] = stats["min"]
            row[f"{prefix}_max"] = stats["max"]
        summaries.append(row)
    summaries.sort(
        key=lambda row: (
            int(row.get("best_score_wins") or 0),
            int(row.get("profile_win_total") or 0),
            _finite_float(row.get("route_score_mean")) or 0.0,
            str(row.get("label")),
        ),
        reverse=True,
    )
    return summaries


def compare_api_llm_matrix_reports(
    reports: Mapping[str, str | Path] | Sequence[str | Path] | str | Path,
    *,
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compare multiple live API LLM provider-matrix report JSON files."""

    entries = _report_entries(reports, labels=labels)
    report_rows: list[dict[str, Any]] = []
    reports_by_label: dict[str, dict[str, Any]] = {}
    for label, path in entries:
        report = _read_json_mapping(path)
        reports_by_label[label] = report
        report_rows.append(_matrix_report_row(label, path, report))

    profile_winners = _profile_winner_summary(report_rows)
    route_summaries = _route_report_summary(report_rows, reports_by_label)
    wasm_context = _matrix_reports_wasm_context(report_rows)
    recommendations: list[str] = []
    if route_summaries:
        best_route = route_summaries[0]
        recommendations.append(
            f"prefer {best_route['label']} when stability across matrix reports matters"
        )
    if wasm_context["observed_reports"]:
        recommendations.append(
            f"{wasm_context['observed_reports']} matrix reports carry WASM context "
            f"({wasm_context['total_wasm_report_count']} selected browser reports)"
        )
    lowest_wasm_loss = wasm_context.get("lowest_best_loss")
    if lowest_wasm_loss:
        recommendations.append(
            f"inspect {lowest_wasm_loss} for the lowest selected WASM report loss"
        )
    for profile, winners in profile_winners.items():
        if not winners:
            continue
        leader = winners[0]
        if leader["win_rate"] >= 1.0 and leader.get("report_coverage") == 1.0:
            recommendations.append(
                f"{profile} profile is stable on {leader['label']} across all reports"
            )
        elif leader["win_rate"] >= 1.0:
            recommendations.append(
                f"{profile} profile is stable on {leader['label']} across "
                f"observed reports ({leader['observed_reports']}/{len(report_rows)} reports)"
            )
        elif leader["win_rate"] >= 0.5:
            recommendations.append(
                f"{profile} profile leans toward {leader['label']} "
                f"({leader['wins']}/{leader['observed_reports']} observed reports)"
            )

    return {
        "kind": "spiraltorch.api_llm_matrix_report_comparison",
        "count": len(report_rows),
        "reports": report_rows,
        "profile_winners": profile_winners,
        "routes": route_summaries,
        "wasm_context": wasm_context,
        "recommendations": recommendations,
    }


def compare_api_llm_trace_runs(
    traces: Mapping[str, str | Path] | Sequence[str | Path] | str | Path,
    *,
    labels: Sequence[str] | None = None,
    event_type: str = "ApiLLMTrace",
    near_best_tolerance: float = 0.02,
) -> dict[str, Any]:
    """Compare multiple API LLM Z-space trace runs.

    The comparison is intentionally lightweight: it consumes JSONL traces already
    written by :func:`write_api_llm_trace_jsonl` or ``runtime.write_jsonl(...)``
    and returns compact per-run rows plus common winners for notebooks and CI.
    """

    entries = _trace_entries(traces, labels=labels)
    rows: list[dict[str, Any]] = []
    for label, path in entries:
        summary = summarize_api_llm_trace_events(path, event_type=event_type)
        rows.append(_comparison_row(label, path, summary))
    rows.sort(
        key=lambda row: (row["route_score"], row["confidence_mean"], row["label"]),
        reverse=True,
    )
    winners = {
        "best_score": _winner(rows, "route_score"),
        "highest_quality": _winner(rows, "quality_score"),
        "highest_text_quality": _winner(rows, "text_quality_score"),
        "highest_efficiency": _winner(rows, "efficiency_score"),
        "highest_confidence": _winner(rows, "confidence_mean"),
        "highest_stability": _winner(rows, "stability_mean"),
        "highest_completion_rate": _winner(rows, "completion_rate"),
        "lowest_empty_text": _winner(rows, "empty_text_rate", higher_is_better=False),
        "lowest_refusal": _winner(rows, "refusal_rate", higher_is_better=False),
        "lowest_latency": _winner(rows, "latency_ms_mean", higher_is_better=False),
        "lowest_total_tokens": _winner(rows, "total_tokens", higher_is_better=False),
        "highest_runtime_ready": _winner(rows, "runtime_ready_rate"),
        "lowest_wasm_loss": _winner(
            rows,
            "wasm_loss_mean",
            higher_is_better=False,
        ),
        "highest_wasm_stability_hint": _winner(rows, "wasm_stability_hint_mean"),
        "highest_wasm_webgpu_device_ready": _winner(
            rows,
            "wasm_webgpu_device_ready_rate",
        ),
    }
    best = winners.get("best_score")
    near_best_tolerance_value = max(0.0, _finite_float(near_best_tolerance) or 0.0)
    near_best = _near_best_routes(rows, tolerance=near_best_tolerance_value)
    selection_profiles = _selection_profiles(rows)
    wasm_context = _comparison_wasm_context(rows)
    recommendations: list[str] = []
    if best:
        recommendations.append(f"prefer {best} for the highest aggregate API LLM route score")
    if len(near_best) > 1:
        labels_text = ", ".join(str(row["label"]) for row in near_best[:4])
        recommendations.append(
            f"compare near-best routes within {near_best_tolerance_value:.3f}: {labels_text}"
        )
    if winners.get("lowest_latency") and winners["lowest_latency"] != best:
        recommendations.append(f"inspect {winners['lowest_latency']} for latency-sensitive routing")
    if winners.get("highest_efficiency") and winners["highest_efficiency"] != best:
        recommendations.append(f"inspect {winners['highest_efficiency']} for cost-sensitive routing")
    if winners.get("highest_text_quality") and winners["highest_text_quality"] != best:
        recommendations.append(
            f"inspect {winners['highest_text_quality']} for stronger prompt-text coverage"
        )
    if winners.get("lowest_refusal") and winners["lowest_refusal"] != best:
        recommendations.append(f"inspect {winners['lowest_refusal']} for fewer refusals")
    grounded_label = selection_profiles.get("grounded", {}).get("label")
    if grounded_label and grounded_label != best:
        recommendations.append(f"use {grounded_label} for grounded prompt-following routes")
    lowest_wasm_loss = winners.get("lowest_wasm_loss")
    if lowest_wasm_loss and lowest_wasm_loss != best:
        recommendations.append(
            f"inspect {lowest_wasm_loss} for the lowest browser-side WASM context loss"
        )
    elif lowest_wasm_loss:
        recommendations.append(
            f"{lowest_wasm_loss} also carries the lowest browser-side WASM context loss"
        )
    return {
        "kind": "spiraltorch.api_llm_trace_comparison",
        "event_type": event_type,
        "count": len(rows),
        "runs": rows,
        "near_best_tolerance": near_best_tolerance_value,
        "near_best": near_best,
        "selection_profiles": selection_profiles,
        "wasm_context": wasm_context,
        "winners": winners,
        "recommendations": recommendations,
    }


def _session_from_spiraltorch(
    backend: str | None,
    session_factory: Callable[..., Any] | None,
) -> tuple[Any | None, dict[str, Any] | None, str | None]:
    if backend is None:
        return None, None, None
    try:
        if session_factory is None:
            module = sys.modules.get("spiraltorch")
            session_factory = getattr(module, "SpiralSession", None) if module is not None else None
        if not callable(session_factory):
            return None, None, "SpiralSession is unavailable"
        session = session_factory(backend=backend)
        report = getattr(session, "device_preflight", None)
        return session, dict(report) if isinstance(report, Mapping) else None, None
    except Exception as exc:  # pragma: no cover - backend availability varies.
        return None, None, f"{exc.__class__.__name__}: {exc}"


class ApiLLMZSpaceRuntime:
    """Runtime bridge for hosted LLM inference plus Z-space observations."""

    def __init__(
        self,
        z_state: Sequence[float],
        *,
        backend: str | None = "auto",
        provider: str | None = None,
        model: str | None = None,
        session: Any | None = None,
        session_factory: Callable[..., Any] | None = None,
        create_session: bool = True,
        alpha: float = 0.35,
        smoothing: float = 0.35,
        strategy: str = "mean",
    ) -> None:
        self.provider = provider
        self.model = model
        self.requested_backend = backend
        self.pipeline = ZSpaceInferencePipeline(
            z_state,
            alpha=alpha,
            smoothing=smoothing,
            strategy=strategy,
        )
        self.session_error: str | None = None
        self.session = session
        self.device_preflight: dict[str, Any] | None = None
        if session is not None:
            report = getattr(session, "device_preflight", None)
            self.device_preflight = dict(report) if isinstance(report, Mapping) else None
        elif create_session:
            self.session, self.device_preflight, self.session_error = _session_from_spiraltorch(
                backend,
                session_factory,
            )
        self.traces: list[ApiLLMTrace] = []

    def record_response(
        self,
        response: Any,
        *,
        prompt: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        latency_ms: float | None = None,
        bundle_weight: float = 1.0,
        telemetry_prefix: str = "api_llm",
        gradient_dim: int = 4,
        context_partials: Any = None,
        clear: bool = True,
    ) -> ApiLLMTrace:
        """Record an API response, fuse it into Z-space, and return a trace."""

        provider_value = provider or self.provider
        model_value = model or self.model or _response_model(response)
        bundle = api_llm_partial_from_response(
            response,
            prompt=prompt,
            provider=provider_value,
            model=model_value,
            latency_ms=latency_ms,
            bundle_weight=bundle_weight,
            telemetry_prefix=telemetry_prefix,
            gradient_dim=gradient_dim,
        )
        for context in _normalise_context_partials(context_partials):
            self.pipeline.add_partial(context)
        self.pipeline.add_partial(bundle)
        inference = self.pipeline.infer(clear=clear)
        trace = api_llm_trace_from_response(
            response,
            prompt=prompt,
            provider=provider_value,
            model=model_value,
            latency_ms=latency_ms,
            inference=inference,
            device_preflight=self.device_preflight,
            bundle_weight=bundle_weight,
            telemetry_prefix=telemetry_prefix,
            gradient_dim=gradient_dim,
        )
        self.traces.append(trace)
        return trace

    def call(
        self,
        invoke: Callable[..., Any],
        prompt: str,
        *args: Any,
        provider: str | None = None,
        model: str | None = None,
        context_partials: Any = None,
        **kwargs: Any,
    ) -> ApiLLMTrace:
        """Call an API-model function and immediately record the Z-space trace."""

        start = time.perf_counter()
        response = invoke(prompt, *args, **kwargs)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return self.record_response(
            response,
            prompt=prompt,
            provider=provider,
            model=model,
            latency_ms=latency_ms,
            context_partials=context_partials,
        )

    def run_prompts(
        self,
        prompts: Iterable[str],
        invoke: Callable[..., Any],
        *args: Any,
        provider: str | None = None,
        model: str | None = None,
        jsonl_out: str | Path | None = None,
        context_partials: Any = None,
        clear: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run a prompt suite through one runtime and return trace artifacts."""

        traces: list[ApiLLMTrace] = []
        context_bundles = _normalise_context_partials(context_partials)
        for prompt in prompts:
            start = time.perf_counter()
            response = invoke(prompt, *args, **kwargs)
            latency_ms = (time.perf_counter() - start) * 1000.0
            traces.append(
                self.record_response(
                    response,
                    prompt=prompt,
                    provider=provider,
                    model=model,
                    latency_ms=latency_ms,
                    context_partials=context_bundles,
                    clear=clear,
                )
            )

        result: dict[str, Any] = {
            "kind": "spiraltorch.api_llm_prompt_suite",
            "count": len(traces),
            "runtime_trace_count": len(self.traces),
            "provider": provider or self.provider,
            "model": model or self.model,
            "requested_backend": self.requested_backend,
            "device_preflight": self.device_preflight,
            "summary": self.summary(),
            "traces": [trace.as_dict() for trace in traces],
        }
        if jsonl_out is not None:
            result["jsonl"] = self.write_jsonl(jsonl_out)
        return result

    def call_openai_responses(
        self,
        prompt: str,
        *,
        client: Any | None = None,
        client_factory: Callable[..., Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
        model: str | None = None,
        input_key: str = "input",
        provider: str | None = "openai",
        context_partials: Any = None,
        **request: Any,
    ) -> ApiLLMTrace:
        """Call OpenAI's Responses API and record the resulting Z-space trace."""

        model_value = model or self.model
        invoke = make_openai_responses_invoke(
            client=client,
            client_factory=client_factory,
            client_kwargs=client_kwargs,
            model=model_value,
            input_key=input_key,
            **request,
        )
        return self.call(
            invoke,
            prompt,
            provider=provider,
            model=model_value,
            context_partials=context_partials,
        )

    def call_openai_chat(
        self,
        prompt: str,
        *,
        client: Any | None = None,
        client_factory: Callable[..., Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
        model: str | None = None,
        system: str | None = None,
        messages: Sequence[Mapping[str, Any]] | None = None,
        provider: str | None = "openai",
        context_partials: Any = None,
        **request: Any,
    ) -> ApiLLMTrace:
        """Call OpenAI chat completions and record the resulting Z-space trace."""

        model_value = model or self.model
        invoke = make_openai_chat_invoke(
            client=client,
            client_factory=client_factory,
            client_kwargs=client_kwargs,
            model=model_value,
            system=system,
            messages=messages,
            **request,
        )
        return self.call(
            invoke,
            prompt,
            provider=provider,
            model=model_value,
            context_partials=context_partials,
        )

    def call_anthropic_messages(
        self,
        prompt: str,
        *,
        client: Any | None = None,
        client_factory: Callable[..., Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
        model: str | None = None,
        system: str | None = None,
        messages: Sequence[Mapping[str, Any]] | None = None,
        provider: str | None = "anthropic",
        context_partials: Any = None,
        **request: Any,
    ) -> ApiLLMTrace:
        """Call Anthropic Messages and record the resulting Z-space trace."""

        model_value = model or self.model
        invoke = make_anthropic_messages_invoke(
            client=client,
            client_factory=client_factory,
            client_kwargs=client_kwargs,
            model=model_value,
            system=system,
            messages=messages,
            **request,
        )
        return self.call(
            invoke,
            prompt,
            provider=provider,
            model=model_value,
            context_partials=context_partials,
        )

    def summary(self) -> dict[str, Any]:
        """Summarize traces already recorded by this runtime instance."""

        trace_dicts = [trace.as_dict() for trace in self.traces]
        usage_rows = [_mapping_at(trace, "usage") for trace in trace_dicts]
        metric_rows = [_mapping_at(trace, "metrics") for trace in trace_dicts]
        text_quality_rows = [_trace_text_quality(trace) for trace in trace_dicts]
        ready_values = [_trace_runtime_ready(trace) for trace in trace_dicts]
        ready_observed = [value for value in ready_values if value is not None]
        ready_count = sum(1 for value in ready_observed if value)
        empty_text_count = sum(1 for trace in trace_dicts if not _trace_has_text(trace))
        refusal_count = sum(
            1 for trace in trace_dicts if _finish_reason_label(trace) == "refusal"
        )
        incomplete_count = sum(
            1
            for trace in trace_dicts
            if _finish_reason_label(trace) in {"incomplete", "length", "max_tokens"}
        )
        completed_count = sum(
            1
            for trace in trace_dicts
            if _finish_reason_label(trace)
            in {"stop", "completed", "complete", "success", "end_turn", "stop_sequence"}
        )
        metric_keys = sorted(
            {
                key
                for row in metric_rows
                for key, value in row.items()
                if key != "gradient" and _finite_float(value) is not None
            }
        )
        return {
            "count": len(trace_dicts),
            "providers": _count_labels(trace.get("provider") for trace in trace_dicts),
            "models": _count_labels(trace.get("model") for trace in trace_dicts),
            "finish_reasons": _count_labels(
                trace.get("finish_reason") for trace in trace_dicts
            ),
            "stop_detail_categories": _count_labels(
                _trace_stop_details_category(trace) for trace in trace_dicts
            ),
            "empty_text_count": empty_text_count,
            "empty_text_rate": empty_text_count / len(trace_dicts) if trace_dicts else 0.0,
            "refusal_count": refusal_count,
            "refusal_rate": refusal_count / len(trace_dicts) if trace_dicts else 0.0,
            "incomplete_count": incomplete_count,
            "incomplete_rate": incomplete_count / len(trace_dicts) if trace_dicts else 0.0,
            "completed_count": completed_count,
            "completion_rate": completed_count / len(trace_dicts) if trace_dicts else 0.0,
            "runtime_statuses": _count_labels(
                _trace_runtime_status(trace) for trace in trace_dicts
            ),
            "runtime_ready_count": ready_count,
            "runtime_ready_rate": ready_count / len(ready_observed) if ready_observed else 0.0,
            "usage": {
                "prompt_tokens": _stats(row.get("prompt_tokens") for row in usage_rows),
                "completion_tokens": _stats(
                    row.get("completion_tokens") for row in usage_rows
                ),
                "total_tokens": _stats(row.get("total_tokens") for row in usage_rows),
            },
            "latency_ms": _stats(trace.get("latency_ms") for trace in trace_dicts),
            "text_quality": {
                "prompt_coverage": _stats(
                    row.get("prompt_coverage") for row in text_quality_rows
                ),
                "prompt_echo_rate": _stats(
                    row.get("prompt_echo_rate") for row in text_quality_rows
                ),
                "response_signal_rate": _stats(
                    row.get("response_signal_rate") for row in text_quality_rows
                ),
                "repetition_rate": _stats(
                    row.get("repetition_rate") for row in text_quality_rows
                ),
                "text_quality_score": _stats(
                    row.get("text_quality_score") for row in text_quality_rows
                ),
            },
            "confidence": _stats(_trace_confidence(trace) for trace in trace_dicts),
            "metrics": {
                key: _stats(row.get(key) for row in metric_rows)
                for key in metric_keys
            },
        }

    def write_jsonl(
        self,
        path: str | Path,
        *,
        event_type: str = "ApiLLMTrace",
    ) -> str:
        """Persist recorded traces as JSONL events."""

        return write_api_llm_trace_jsonl(
            self.traces,
            path,
            event_type=event_type,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "requested_backend": self.requested_backend,
            "session_error": self.session_error,
            "device_preflight": self.device_preflight,
            "summary": self.summary(),
            "traces": [trace.as_dict() for trace in self.traces],
        }


def run_api_llm_prompt_suite(
    prompts: Iterable[str],
    invoke: Callable[..., Any],
    *args: Any,
    z_state: Sequence[float],
    backend: str | None = "auto",
    provider: str | None = None,
    model: str | None = None,
    session: Any | None = None,
    session_factory: Callable[..., Any] | None = None,
    create_session: bool = True,
    alpha: float = 0.35,
    smoothing: float = 0.35,
    strategy: str = "mean",
    jsonl_out: str | Path | None = None,
    context_partials: Any = None,
    clear: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run multiple hosted/API-model prompts through a fresh Z-space runtime."""

    runtime = ApiLLMZSpaceRuntime(
        z_state,
        backend=backend,
        provider=provider,
        model=model,
        session=session,
        session_factory=session_factory,
        create_session=create_session,
        alpha=alpha,
        smoothing=smoothing,
        strategy=strategy,
    )
    return runtime.run_prompts(
        prompts,
        invoke,
        *args,
        provider=provider,
        model=model,
        jsonl_out=jsonl_out,
        context_partials=context_partials,
        clear=clear,
        **kwargs,
    )


def run_api_llm_prompt_suite_matrix(
    prompts: Iterable[str],
    invokes: Mapping[str, Callable[..., Any]],
    *args: Any,
    z_state: Sequence[float],
    backend: str | None = "auto",
    providers: Mapping[str, str | None] | None = None,
    models: Mapping[str, str | None] | None = None,
    session: Any | None = None,
    session_factory: Callable[..., Any] | None = None,
    create_session: bool = True,
    alpha: float = 0.35,
    smoothing: float = 0.35,
    strategy: str = "mean",
    jsonl_dir: str | Path | None = None,
    context_partials: Any = None,
    request_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
    near_best_tolerance: float = 0.02,
    clear: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the same prompt suite through multiple API-model callables.

    ``kwargs`` are shared across all routes.  Use ``request_kwargs`` when each
    route needs a provider-specific request shape, such as Anthropic effort
    controls versus OpenAI response token limits.
    """

    prompt_list = list(prompts)
    provider_map = dict(providers or {})
    model_map = dict(models or {})
    request_kwargs_map = dict(request_kwargs or {})
    out_dir = Path(jsonl_dir) if jsonl_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    suites: dict[str, dict[str, Any]] = {}
    trace_paths: dict[str, str] = {}
    for index, (label, invoke) in enumerate(invokes.items()):
        label_value = str(label)
        jsonl_out: Path | None = None
        if out_dir is not None:
            safe_label = _safe_trace_label(label_value, fallback=f"run-{index}")
            filename = f"{index:02d}-{safe_label}.jsonl"
            jsonl_out = out_dir / filename
        route_kwargs = dict(kwargs)
        route_kwargs.update(dict(request_kwargs_map.get(label_value, {})))
        suite = run_api_llm_prompt_suite(
            prompt_list,
            invoke,
            *args,
            z_state=z_state,
            backend=backend,
            provider=provider_map.get(label_value, label_value),
            model=model_map.get(label_value),
            session=session,
            session_factory=session_factory,
            create_session=create_session,
            alpha=alpha,
            smoothing=smoothing,
            strategy=strategy,
            jsonl_out=jsonl_out,
            context_partials=context_partials,
            clear=clear,
            **route_kwargs,
        )
        suites[label_value] = suite
        path = suite.get("jsonl")
        if isinstance(path, str):
            trace_paths[label_value] = path

    comparison = (
        compare_api_llm_trace_runs(trace_paths, near_best_tolerance=near_best_tolerance)
        if trace_paths
        else None
    )
    return {
        "kind": "spiraltorch.api_llm_prompt_suite_matrix",
        "count": len(suites),
        "prompt_count": len(prompt_list),
        "labels": list(suites.keys()),
        "jsonl_dir": str(out_dir) if out_dir is not None else None,
        "trace_paths": trace_paths,
        "suites": suites,
        "comparison": comparison,
    }
