"""Safety-aware inference helpers built on top of the SpiralTorch runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from spiraltorch import inference as _native

SafetyViolation = _native.SafetyViolation
SafetyVerdict = _native.SafetyVerdict
InferenceResult = _native.InferenceResult
AuditEvent = _native.AuditEvent
AuditLog = _native.AuditLog


@dataclass(frozen=True)
class SafetyEvent:
    """Snapshot of a safety audit event returned by the native runtime."""

    channel: str
    verdict: SafetyVerdict
    content_preview: str
    timestamp: str

    @classmethod
    def from_native(cls, event: AuditEvent) -> "SafetyEvent":
        return cls(
            channel=event.channel,
            verdict=event.verdict,
            content_preview=event.content_preview,
            timestamp=event.timestamp,
        )


@dataclass(frozen=True)
class ChatMessage:
    """Lightweight container representing a single conversational turn."""

    role: str
    content: str

    def __post_init__(self) -> None:
        role = self.role.strip().lower()
        if not role:
            raise ValueError("chat message role must be non-empty")
        if not isinstance(self.content, str):
            raise TypeError("chat message content must be a string")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "content", self.content.rstrip())

    @classmethod
    def system(cls, content: str) -> "ChatMessage":
        return cls("system", content)

    @classmethod
    def user(cls, content: str) -> "ChatMessage":
        return cls("user", content)

    @classmethod
    def assistant(cls, content: str) -> "ChatMessage":
        return cls("assistant", content)

    @classmethod
    def tool(cls, content: str) -> "ChatMessage":
        return cls("tool", content)

    def format(self, *, style: str = "spiral") -> str:
        """Render the message into a textual representation."""

        style_key = style.lower()
        text = self.content.strip()
        if style_key in {"spiral", "role"}:
            return f"{self.role.upper()}: {text}"
        if style_key == "markdown":
            return f"**{self.role}:** {text}"
        if style_key in {"openai", "tagged"}:
            return f"<|{self.role}|>\n{text}"
        raise ValueError(f"unknown chat message style: {style}")


def _coerce_message(message: "ChatMessageLike") -> ChatMessage:
    if isinstance(message, ChatMessage):
        return message
    if isinstance(message, Mapping):
        role = message.get("role")
        content = message.get("content")
        if role is None or content is None:
            raise ValueError("mapping-based chat messages require 'role' and 'content'")
        return ChatMessage(str(role), str(content))
    if isinstance(message, Sequence) and not isinstance(message, (str, bytes, bytearray)):
        if len(message) != 2:
            raise ValueError("sequence-based chat messages must be (role, content)")
        role, content = message
        return ChatMessage(str(role), str(content))
    raise TypeError("chat messages must be ChatMessage, mapping, or (role, content) pairs")


@dataclass(frozen=True)
class ChatPrompt:
    """Helper that renders multi-turn chat prompts consistently."""

    messages: tuple[ChatMessage, ...]
    style: str = "spiral"
    separator: str = "\n\n"

    def __post_init__(self) -> None:
        object.__setattr__(self, "messages", tuple(self.messages))
        if not self.messages:
            raise ValueError("ChatPrompt requires at least one message")

    def __iter__(self):
        return iter(self.messages)

    def __len__(self) -> int:
        return len(self.messages)

    def append(self, message: "ChatMessageLike") -> "ChatPrompt":
        return ChatPrompt(self.messages + (_coerce_message(message),), self.style, self.separator)

    def extend(self, messages: Iterable["ChatMessageLike"]) -> "ChatPrompt":
        additions = tuple(_coerce_message(msg) for msg in messages)
        return ChatPrompt(self.messages + additions, self.style, self.separator)

    def render(self, *, style: Optional[str] = None, separator: Optional[str] = None) -> str:
        style_key = (style or self.style).lower()
        sep = self.separator if separator is None else separator
        if not sep:
            sep = "\n"
        return sep.join(message.format(style=style_key) for message in self.messages)

    def __str__(self) -> str:
        return self.render()

    @classmethod
    def from_messages(
        cls,
        messages: Iterable["ChatMessageLike"],
        *,
        style: str = "spiral",
        separator: str = "\n\n",
    ) -> "ChatPrompt":
        return cls(tuple(_coerce_message(msg) for msg in messages), style=style, separator=separator)


ChatMessageLike = ChatMessage | Mapping[str, Any] | Sequence[Any]


def format_chat_prompt(
    messages: Iterable[ChatMessageLike],
    *,
    style: str = "spiral",
    separator: str = "\n\n",
) -> str:
    """Render chat-style messages into a single prompt string."""

    prompt = ChatPrompt.from_messages(messages, style=style, separator=separator)
    return prompt.render()


def _render_prompt(
    prompt: str | ChatPrompt | Sequence[ChatMessageLike] | Iterable[ChatMessageLike],
) -> str:
    if isinstance(prompt, ChatPrompt):
        return prompt.render()
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, Mapping):
        return format_chat_prompt([prompt])
    if isinstance(prompt, Iterable) and not isinstance(prompt, (bytes, bytearray)):
        return format_chat_prompt(prompt)
    raise TypeError("prompt must be a string, ChatPrompt, or iterable of chat messages")


class InferenceClient:
    """High-level wrapper that injects policy enforcement before returning results."""

    def __init__(self, *, refusal_threshold: float | None = None) -> None:
        self._runtime = _native.InferenceRuntime(refusal_threshold=refusal_threshold)

    def generate(
        self,
        prompt: str | ChatPrompt | Sequence[ChatMessageLike] | Iterable[ChatMessageLike],
        *,
        candidate: str | None = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> InferenceResult:
        payload: Dict[str, Any] = dict(metadata or {})
        if candidate is not None:
            payload.setdefault("candidate", candidate)
        prompt_text = _render_prompt(prompt)
        return self._runtime.generate(prompt_text, metadata=payload or None)

    def chat(
        self,
        messages: Iterable[ChatMessageLike],
        *,
        candidate: str | None = None,
        metadata: Optional[Dict[str, Any]] = None,
        style: str = "spiral",
        separator: str = "\n\n",
    ) -> InferenceResult:
        prompt_text = format_chat_prompt(messages, style=style, separator=separator)
        return self.generate(prompt_text, candidate=candidate, metadata=metadata)

    @property
    def audit_log(self) -> AuditLog:
        return self._runtime.audit_log

    def audit_events(self) -> list[SafetyEvent]:
        return [SafetyEvent.from_native(event) for event in self.audit_log.entries()]


__all__ = [
    "AuditEvent",
    "AuditLog",
    "ChatMessage",
    "ChatPrompt",
    "InferenceClient",
    "InferenceResult",
    "SafetyEvent",
    "SafetyVerdict",
    "SafetyViolation",
    "format_chat_prompt",
]

