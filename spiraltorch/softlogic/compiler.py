"""Lightweight SpiralK parser used by the Python shim.

The canonical parser lives in the Rust crate ``st-softlogic``.  This module
mirrors the behaviour so that developers can experiment with SoftLogic flows
without having to compile the native bindings.  Once the bindings are linked,
the same API can delegate to the Rust implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, List, NamedTuple, Optional


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
    tokens = list(_tokenize(src))
    parser = _Parser(tokens)
    return parser.parse_document()


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

