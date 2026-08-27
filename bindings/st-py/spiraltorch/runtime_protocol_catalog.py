"""Thin Python facade for the Rust-owned cross-client protocol catalog."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Any


def _native_constant(name: str, fallback: str) -> str:
    package = sys.modules.get(__package__ or "spiraltorch")
    native = getattr(package, "_rs", None)
    return str(getattr(native, name, fallback))


ZSPACE_RUNTIME_PROTOCOL_CATALOG_CONTRACT_VERSION = _native_constant(
    "ZSPACE_RUNTIME_PROTOCOL_CATALOG_CONTRACT_VERSION",
    "spiraltorch.zspace_runtime_protocol_catalog.v1",
)
ZSPACE_RUNTIME_PROTOCOL_CATALOG_KIND = _native_constant(
    "ZSPACE_RUNTIME_PROTOCOL_CATALOG_KIND",
    "spiraltorch.zspace_runtime_protocol_catalog",
)
ZSPACE_RUNTIME_PROTOCOL_CATALOG_ID_RULE = _native_constant(
    "ZSPACE_RUNTIME_PROTOCOL_CATALOG_ID_RULE",
    "sha256(contract_version UTF-8 || NUL || compact catalog JSON with catalog_id empty)",
)
ZSPACE_RUNTIME_PROTOCOL_CATALOG_SEMANTIC_OWNER = _native_constant(
    "ZSPACE_RUNTIME_PROTOCOL_CATALOG_SEMANTIC_OWNER",
    "st-core::runtime::zspace_runtime_protocol_catalog",
)
ZSPACE_RUNTIME_PROTOCOL_CATALOG_SEMANTIC_BACKEND = _native_constant(
    "ZSPACE_RUNTIME_PROTOCOL_CATALOG_SEMANTIC_BACKEND",
    "rust",
)
ZSPACE_RUNTIME_PROTOCOL_CATALOG_STATUS = _native_constant(
    "ZSPACE_RUNTIME_PROTOCOL_CATALOG_STATUS",
    "ready",
)

__all__ = [
    "ZSPACE_RUNTIME_PROTOCOL_CATALOG_CONTRACT_VERSION",
    "ZSPACE_RUNTIME_PROTOCOL_CATALOG_ID_RULE",
    "ZSPACE_RUNTIME_PROTOCOL_CATALOG_KIND",
    "ZSPACE_RUNTIME_PROTOCOL_CATALOG_SEMANTIC_BACKEND",
    "ZSPACE_RUNTIME_PROTOCOL_CATALOG_SEMANTIC_OWNER",
    "ZSPACE_RUNTIME_PROTOCOL_CATALOG_STATUS",
    "validate_zspace_runtime_protocol_catalog",
    "zspace_runtime_protocol_catalog",
]


def _validate_catalog(catalog: Mapping[str, Any]) -> None:
    protocols = catalog.get("protocols")
    protocol_count = catalog.get("protocol_count")
    if (
        catalog.get("contract_version")
        != ZSPACE_RUNTIME_PROTOCOL_CATALOG_CONTRACT_VERSION
        or catalog.get("kind") != ZSPACE_RUNTIME_PROTOCOL_CATALOG_KIND
        or catalog.get("catalog_id_rule") != ZSPACE_RUNTIME_PROTOCOL_CATALOG_ID_RULE
        or catalog.get("semantic_owner")
        != ZSPACE_RUNTIME_PROTOCOL_CATALOG_SEMANTIC_OWNER
        or catalog.get("semantic_backend")
        != ZSPACE_RUNTIME_PROTOCOL_CATALOG_SEMANTIC_BACKEND
        or catalog.get("catalog_validated") is not True
        or catalog.get("status") != ZSPACE_RUNTIME_PROTOCOL_CATALOG_STATUS
        or not isinstance(protocol_count, int)
        or isinstance(protocol_count, bool)
        or protocol_count <= 0
        or not isinstance(protocols, list)
        or protocol_count != len(protocols)
    ):
        raise RuntimeError("native Z-space core returned an untrusted protocol catalog")
    catalog_id = catalog.get("catalog_id")
    if (
        not isinstance(catalog_id, str)
        or not catalog_id.startswith("sha256:")
        or len(catalog_id) != 71
    ):
        raise RuntimeError(
            "native Z-space core returned an invalid protocol catalog ID"
        )

    protocol_order = str(catalog.get("protocol_order_rule") or "").split(",")
    client_order = str(catalog.get("client_order_rule") or "").split(",")
    transport_order = ["native", "bounded_mapping", "bounded_json"]
    if len(protocol_order) != protocol_count or client_order != [
        "rust",
        "python",
        "wasm",
    ]:
        raise RuntimeError("native Z-space core returned invalid catalog ordering")
    for index, protocol in enumerate(protocols):
        if not isinstance(protocol, Mapping):
            raise RuntimeError(
                "native Z-space core returned a malformed protocol entry"
            )
        clients = protocol.get("clients")
        artifacts = protocol.get("artifacts")
        if (
            protocol.get("name") != protocol_order[index]
            or protocol.get("semantic_backend") != "rust"
            or protocol.get("admission_owner") != "rust"
            or not isinstance(artifacts, list)
            or not artifacts
            or not isinstance(clients, list)
            or len(clients) != len(client_order)
            or not all(isinstance(client, Mapping) for client in clients)
            or [client.get("client") for client in clients] != client_order
            or [client.get("transport") for client in clients] != transport_order
        ):
            raise RuntimeError("native Z-space core returned invalid protocol metadata")
        for artifact in artifacts:
            if (
                not isinstance(artifact, Mapping)
                or not isinstance(artifact.get("name"), str)
                or not artifact["name"]
                or not isinstance(artifact.get("contract_version"), str)
                or not artifact["contract_version"]
                or artifact.get("discriminator_field") not in {"kind", "schema"}
                or not isinstance(artifact.get("discriminator_value"), str)
                or not artifact["discriminator_value"]
            ):
                raise RuntimeError(
                    "native Z-space core returned invalid artifact metadata"
                )
        for client in clients:
            operations = client.get("operations")
            if (
                not isinstance(operations, list)
                or not operations
                or not all(
                    isinstance(operation, str) and operation for operation in operations
                )
            ):
                raise RuntimeError(
                    "native Z-space core returned an empty client surface"
                )
            if (
                client.get("client") == "wasm"
                and client.get("trusted_legacy_replay") is not False
            ):
                raise RuntimeError(
                    "native Z-space core exposed trusted legacy replay to WASM"
                )


def _native_operation(
    name: str, payload: Mapping[str, object] | None = None
) -> dict[str, Any]:
    package = sys.modules.get(__package__ or "spiraltorch")
    native = getattr(package, "_rs", None)
    operation = getattr(native, name, None)
    if not callable(operation):
        raise RuntimeError(
            "Z-space runtime protocol catalog requires the compiled Rust semantic "
            f"core; rebuild or reinstall SpiralTorch with {name}"
        )
    result = operation() if payload is None else operation(payload)
    if not isinstance(result, Mapping):
        raise RuntimeError(f"native {name} returned a non-mapping payload")
    catalog = dict(result)
    _validate_catalog(catalog)
    return catalog


def zspace_runtime_protocol_catalog() -> dict[str, Any]:
    """Return the exact Rust-owned protocol surface shared by all clients."""

    return _native_operation("_zspace_runtime_protocol_catalog")


def validate_zspace_runtime_protocol_catalog(
    catalog: Mapping[str, object],
) -> dict[str, Any]:
    """Replay an archived catalog against the current exact Rust surface."""

    if not isinstance(catalog, Mapping):
        raise TypeError("catalog must be a mapping")
    return _native_operation("_zspace_runtime_protocol_catalog_validate", catalog)
