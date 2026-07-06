"""Bridge API-model LLM inference into the SpiralTorch Z-space runtime.

The helpers in this module intentionally avoid hard dependencies on hosted LLM
SDKs.  Callers can pass an already materialised response mapping, an SDK
response object, or a callable that performs the API request.  SpiralTorch then
derives a bounded Z-space partial observation from text, usage, latency, and
probability-like fields.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
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
]


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
        return ""
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
        return "".join(_content_text(part) for part in content)
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
    if finish in {"stop", "completed", "complete", "success"}:
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
        numeric_telemetry["finish_reason_stop"] = 1.0 if finish_reason.lower() == "stop" else 0.0

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
        return {
            "provider": self.provider,
            "model": self.model,
            "prompt": self.prompt,
            "text": self.text,
            "finish_reason": self.finish_reason,
            "latency_ms": self.latency_ms,
            "usage": dict(self.usage),
            "metrics": dict(self.metrics),
            "telemetry": dict(self.telemetry),
            "inference": inference_payload,
            "device_preflight": None if self.device_preflight is None else dict(self.device_preflight),
            "response_metadata": None if self.response_metadata is None else dict(self.response_metadata),
        }


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
    metadata = _maybe_mapping(response) or {}
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
        response_metadata={
            key: value
            for key, value in metadata.items()
            if key in {"id", "created", "object", "system_fingerprint"}
        },
    )


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
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "requested_backend": self.requested_backend,
            "session_error": self.session_error,
            "device_preflight": self.device_preflight,
            "traces": [trace.as_dict() for trace in self.traces],
        }
