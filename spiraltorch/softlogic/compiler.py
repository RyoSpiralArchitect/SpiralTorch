"""Lightweight SpiralK parser used by the Python shim.

The canonical parser lives in the Rust crate ``st-softlogic``.  This module
mirrors the behaviour so that developers can experiment with SoftLogic flows
without having to compile the native bindings.  Once the bindings are linked,
the same API can delegate to the Rust implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Any, Iterator, List, Mapping, NamedTuple, Optional, Sequence, Type, TypeVar

try:  # pragma: no cover - optional native bridge
    from spiraltorch import _softlogic_native as _softlogic_native
except Exception:  # pragma: no cover - native module missing
    _NATIVE_PARSE = None
else:  # pragma: no cover - exercised when extension is present
    _NATIVE_PARSE = getattr(_softlogic_native, "parse_document", None)


logger = logging.getLogger(__name__)

_EnumT = TypeVar("_EnumT", bound=Enum)


class Precision(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


class Layout(str, Enum):
    NHWC = "nhwc"
    NCHW = "nchw"
    BLOCKED = "blocked"


class Backend(str, Enum):
    WGPU = "WGPU"
    MPS = "MPS"
    CUDA = "CUDA"
    CPU = "CPU"


@dataclass(frozen=True)
class TargetSpec:
    kind: str
    name: str

    @classmethod
    def graph(cls, name: str) -> "TargetSpec":
        return cls("graph", name)

    @classmethod
    def prsn(cls, name: str) -> "TargetSpec":
        return cls("prsn", name)

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return f"{self.kind}:{self.name}"


@dataclass(frozen=True)
class RefractOpPolicy:
    op: str
    flags: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class RefractBlock:
    name: str
    target: TargetSpec
    precision: Optional[Precision] = None
    layout: Optional[Layout] = None
    schedule: Optional[str] = None
    backend: Optional[Backend] = None
    policies: List[RefractOpPolicy] = field(default_factory=list)


@dataclass(frozen=True)
class SyncBlock:
    name: str
    pairs: List[str]
    tolerance: float


@dataclass(frozen=True)
class FeedbackBlock:
    name: str
    export_path: str
    metrics: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class Document:
    refracts: List[RefractBlock] = field(default_factory=list)
    syncs: List[SyncBlock] = field(default_factory=list)
    feedbacks: List[FeedbackBlock] = field(default_factory=list)


class _Token(NamedTuple):
    kind: str
    value: Optional[str] = None


def compile_spiralk(src: str) -> Document:
    if _NATIVE_PARSE is not None:
        try:
            payload = _NATIVE_PARSE(src)
        except Exception:  # pragma: no cover - bridge failures fall back
            logger.debug(
                "Native SpiralK parser unavailable; falling back to Python",
                exc_info=True,
            )
        else:
            try:
                return _document_from_payload(payload)
            except Exception:  # pragma: no cover - ensure fallback
                logger.exception(
                    "SoftLogic: native parser returned unexpected data; using Python fallback"
                )
    return _compile_spiralk_python(src)


def _compile_spiralk_python(src: str) -> Document:
    tokens = list(_tokenize(src))
    parser = _Parser(tokens)
    return parser.parse_document()


def _document_from_payload(payload: Mapping[str, Any]) -> Document:
    if not isinstance(payload, Mapping):
        raise TypeError("native parse must return a mapping")  # pragma: no cover - defensive

    refracts = [
        _refract_from_payload(entry)
        for entry in _as_sequence(payload.get("refracts"), "refracts")
    ]
    syncs = [
        _sync_from_payload(entry) for entry in _as_sequence(payload.get("syncs"), "syncs")
    ]
    feedbacks = [
        _feedback_from_payload(entry)
        for entry in _as_sequence(payload.get("feedbacks"), "feedbacks")
    ]
    return Document(refracts=list(refracts), syncs=list(syncs), feedbacks=list(feedbacks))


def _refract_from_payload(data: Mapping[str, Any]) -> RefractBlock:
    if not isinstance(data, Mapping):
        raise TypeError("refract entry must be a mapping")  # pragma: no cover - defensive

    target_data = data.get("target")
    if not isinstance(target_data, Mapping):
        raise TypeError("refract.target must be a mapping")  # pragma: no cover - defensive

    kind = _coerce_string(target_data.get("kind"), "target.kind").lower()
    name = _coerce_string(target_data.get("name"), "target.name")
    if kind == "graph":
        target = TargetSpec.graph(name)
    elif kind == "prsn":
        target = TargetSpec.prsn(name)
    else:
        target = TargetSpec(kind, name)

    precision = _coerce_enum_optional(Precision, data.get("precision"))
    layout = _coerce_enum_optional(Layout, data.get("layout"))
    schedule = _coerce_optional_string(data.get("schedule"))
    backend = _coerce_enum_optional(Backend, data.get("backend"))

    policies: List[RefractOpPolicy] = []
    for entry in _as_sequence(data.get("policies"), "refract.policies"):
        if not isinstance(entry, Mapping):
            raise TypeError(
                "refract.policies entries must be mappings"
            )  # pragma: no cover - defensive
        op = _coerce_string(entry.get("op"), "policy.op")
        flags = _coerce_string_list(entry.get("flags"), "policy.flags")
        policies.append(RefractOpPolicy(op=op, flags=flags))

    return RefractBlock(
        name=_coerce_string(data.get("name"), "refract.name"),
        target=target,
        precision=precision,
        layout=layout,
        schedule=schedule,
        backend=backend,
        policies=policies,
    )


def _sync_from_payload(data: Mapping[str, Any]) -> SyncBlock:
    if not isinstance(data, Mapping):
        raise TypeError("sync entry must be a mapping")  # pragma: no cover - defensive

    pairs = _coerce_string_list(data.get("pairs"), "sync.pairs")
    tolerance_value = data.get("tolerance")
    if tolerance_value is None:
        raise TypeError("sync.tolerance is required")  # pragma: no cover - defensive

    return SyncBlock(
        name=_coerce_string(data.get("name"), "sync.name"),
        pairs=list(pairs),
        tolerance=float(tolerance_value),
    )


def _feedback_from_payload(data: Mapping[str, Any]) -> FeedbackBlock:
    if not isinstance(data, Mapping):
        raise TypeError("feedback entry must be a mapping")  # pragma: no cover - defensive

    return FeedbackBlock(
        name=_coerce_string(data.get("name"), "feedback.name"),
        export_path=_coerce_string(data.get("export"), "feedback.export"),
        metrics=list(_coerce_string_list(data.get("metrics"), "feedback.metrics")),
    )


def _as_sequence(value: Any, field: str) -> Sequence[Any]:
    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    raise TypeError(
        f"native parse field '{field}' must be a sequence"
    )  # pragma: no cover - defensive


def _coerce_enum(enum_cls: Type[_EnumT], value: Any) -> _EnumT:
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        for member in enum_cls:
            if (
                value == member.value
                or lowered == str(member.value).lower()
                or lowered == member.name.lower()
            ):
                return member
    raise ValueError(f"unknown {enum_cls.__name__} value: {value}")


def _coerce_enum_optional(enum_cls: Type[_EnumT], value: Any) -> Optional[_EnumT]:
    if value is None:
        return None
    return _coerce_enum(enum_cls, value)


def _coerce_string(value: Any, field: str) -> str:
    if value is None:
        raise TypeError(
            f"native parse missing {field}"
        )  # pragma: no cover - defensive
    if isinstance(value, str):
        return value
    if isinstance(value, Enum):
        return value.value
    return str(value)


def _coerce_optional_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, Enum):
        return value.value
    return str(value)


def _coerce_string_list(values: Any, field: str) -> List[str]:
    return [
        _coerce_string(item, f"{field}[{index}]")
        for index, item in enumerate(_as_sequence(values, field))
    ]


def _tokenize(src: str) -> Iterator[_Token]:
    length = len(src)
    i = 0
    while i < length:
        ch = src[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "{":
            yield _Token("LBRACE")
            i += 1
            continue
        if ch == "}":
            yield _Token("RBRACE")
            i += 1
            continue
        if ch == "[":
            yield _Token("LBRACKET")
            i += 1
            continue
        if ch == "]":
            yield _Token("RBRACKET")
            i += 1
            continue
        if ch == ":":
            yield _Token("COLON")
            i += 1
            continue
        if ch == ",":
            yield _Token("COMMA")
            i += 1
            continue
        if ch == "-":
            if i + 1 >= length:
                raise ValueError("dangling '-' at end of source")
            nxt = src[i + 1]
            if nxt == ">":
                yield _Token("ARROW")
                i += 2
                continue
            if nxt.isdigit():
                start = i
                i += 2
                while i < length and (src[i].isdigit() or src[i] == "."):
                    i += 1
                yield _Token("NUMBER", src[start:i])
                continue
            raise ValueError("unexpected '-' without '>' or digits")
        if ch == '"':
            i += 1
            literal: List[str] = []
            escaped = False
            closed = False
            while i < length:
                nxt = src[i]
                i += 1
                if escaped:
                    literal.append(nxt)
                    escaped = False
                    continue
                if nxt == "\\":
                    escaped = True
                    continue
                if nxt == '"':
                    closed = True
                    break
                literal.append(nxt)
            if not closed:
                raise ValueError("unterminated string literal")
            yield _Token("STRING", "".join(literal))
            continue
        if _is_ident_start(ch):
            start = i
            i += 1
            while i < length and _is_ident_continue(src[i]):
                i += 1
            yield _Token("IDENT", src[start:i])
            continue
        if ch.isdigit():
            start = i
            i += 1
            while i < length and (src[i].isdigit() or src[i] == "."):
                i += 1
            yield _Token("NUMBER", src[start:i])
            continue
        raise ValueError(f"unexpected character '{ch}' in SpiralK source")


def _is_ident_start(ch: str) -> bool:
    return ch == "_" or ch.isalpha()


def _is_ident_continue(ch: str) -> bool:
    return ch == "_" or ch == "-" or ch.isalnum()


class _Parser:
    def __init__(self, tokens: List[_Token]):
        self._tokens = tokens
        self._index = 0

    def parse_document(self) -> Document:
        refracts: List[RefractBlock] = []
        syncs: List[SyncBlock] = []
        feedbacks: List[FeedbackBlock] = []

        while not self._eof:
            token = self._peek()
            if token is None:
                break
            if token.kind != "IDENT":
                raise ValueError(f"unexpected token {token.kind}")
            keyword = token.value or ""
            self._advance()
            if keyword == "refract":
                refracts.append(self._parse_refract())
            elif keyword == "sync":
                syncs.append(self._parse_sync())
            elif keyword == "feedback":
                feedbacks.append(self._parse_feedback())
            else:
                raise ValueError(f"unknown top-level keyword '{keyword}'")

        return Document(refracts=refracts, syncs=syncs, feedbacks=feedbacks)

    @property
    def _eof(self) -> bool:
        return self._index >= len(self._tokens)

    def _peek(self) -> Optional[_Token]:
        if self._eof:
            return None
        return self._tokens[self._index]

    def _advance(self) -> Optional[_Token]:
        if self._eof:
            return None
        token = self._tokens[self._index]
        self._index += 1
        return token

    def _expect(self, kind: str) -> _Token:
        token = self._advance()
        if token is None or token.kind != kind:
            raise ValueError(f"expected {kind}, got {token}")
        return token

    def _consume(self, kind: str) -> bool:
        if not self._eof and self._tokens[self._index].kind == kind:
            self._index += 1
            return True
        return False

    def _parse_refract(self) -> RefractBlock:
        name = self._expect("IDENT").value or ""
        self._expect("LBRACE")

        target: Optional[TargetSpec] = None
        precision: Optional[Precision] = None
        layout: Optional[Layout] = None
        schedule: Optional[str] = None
        backend: Optional[Backend] = None
        policies: List[RefractOpPolicy] = []

        while not self._consume("RBRACE"):
            key = self._expect("IDENT").value or ""
            if key == "target":
                self._expect("COLON")
                target = self._parse_target()
            elif key == "precision":
                self._expect("COLON")
                ident = (self._expect("IDENT").value or "").lower()
                try:
                    precision = Precision(ident)
                except ValueError as exc:  # pragma: no cover - defensive
                    raise ValueError(f"unknown precision '{ident}'") from exc
            elif key == "layout":
                self._expect("COLON")
                ident = (self._expect("IDENT").value or "").lower()
                try:
                    layout = Layout(ident)
                except ValueError as exc:
                    raise ValueError(f"unknown layout '{ident}'") from exc
            elif key == "schedule":
                self._expect("COLON")
                schedule = self._expect("IDENT").value or ""
            elif key == "backend":
                self._expect("COLON")
                ident = self._expect("IDENT").value or ""
                try:
                    backend = Backend(ident.upper())
                except ValueError as exc:
                    raise ValueError(f"unknown backend '{ident}'") from exc
            elif key == "op":
                self._expect("COLON")
                op = self._expect("IDENT").value or ""
                flags: List[str] = []
                if self._consume("ARROW"):
                    while True:
                        flag_tok = self._expect("IDENT")
                        flags.append(flag_tok.value or "")
                        if not self._consume("COMMA"):
                            break
                policies.append(RefractOpPolicy(op, flags))
            else:
                raise ValueError(f"unknown statement '{key}' in refract block")
            self._consume("COMMA")

        if target is None:
            raise ValueError(f"refract block '{name}' missing target")

        return RefractBlock(
            name=name,
            target=target,
            precision=precision,
            layout=layout,
            schedule=schedule,
            backend=backend,
            policies=policies,
        )

    def _parse_target(self) -> TargetSpec:
        kind = self._expect("IDENT").value or ""
        self._expect("COLON")
        name = self._expect("IDENT").value or ""
        if kind == "graph":
            return TargetSpec.graph(name)
        if kind == "prsn":
            return TargetSpec.prsn(name)
        raise ValueError(f"unknown target kind '{kind}'")

    def _parse_sync(self) -> SyncBlock:
        name = self._expect("IDENT").value or ""
        self._expect("LBRACE")

        pairs: Optional[List[str]] = None
        tolerance: Optional[float] = None

        while not self._consume("RBRACE"):
            key = self._expect("IDENT").value or ""
            if key == "pairs":
                self._expect("COLON")
                self._expect("LBRACKET")
                values: List[str] = []
                while not self._consume("RBRACKET"):
                    values.append(self._expect("IDENT").value or "")
                    if not self._consume("COMMA"):
                        self._expect("RBRACKET")
                        break
                pairs = values
            elif key == "tolerance":
                self._expect("COLON")
                num = self._expect("NUMBER").value or "0"
                tolerance = float(num)
            else:
                raise ValueError(f"unknown field '{key}' in sync block")
            self._consume("COMMA")

        if pairs is None:
            raise ValueError(f"sync block '{name}' missing pairs")
        if tolerance is None:
            raise ValueError(f"sync block '{name}' missing tolerance")

        return SyncBlock(name=name, pairs=pairs, tolerance=tolerance)

    def _parse_feedback(self) -> FeedbackBlock:
        name = self._expect("IDENT").value or ""
        self._expect("LBRACE")

        export_path: Optional[str] = None
        metrics: List[str] = []

        while not self._consume("RBRACE"):
            key = self._expect("IDENT").value or ""
            if key == "export":
                self._expect("COLON")
                export_path = self._expect("STRING").value or ""
            elif key == "metrics":
                self._expect("COLON")
                self._expect("LBRACKET")
                values: List[str] = []
                while not self._consume("RBRACKET"):
                    values.append(self._expect("IDENT").value or "")
                    if not self._consume("COMMA"):
                        self._expect("RBRACKET")
                        break
                metrics = values
            else:
                raise ValueError(f"unknown field '{key}' in feedback block")
            self._consume("COMMA")

        if export_path is None:
            raise ValueError(f"feedback block '{name}' missing export")

        return FeedbackBlock(name=name, export_path=export_path, metrics=metrics)

