from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path

import pytest

import spiraltorch as st


WASM_DECLARATIONS = (
    Path(__file__).resolve().parents[2] / "st-wasm" / "types" / "spiraltorch-wasm.d.ts"
)


def test_runtime_protocol_catalog_is_rust_owned_and_replayable() -> None:
    catalog = st.zspace_runtime_protocol_catalog()

    assert (
        catalog["contract_version"]
        == st.ZSPACE_RUNTIME_PROTOCOL_CATALOG_CONTRACT_VERSION
    )
    assert catalog["kind"] == st.ZSPACE_RUNTIME_PROTOCOL_CATALOG_KIND
    assert (
        catalog["semantic_owner"] == st.ZSPACE_RUNTIME_PROTOCOL_CATALOG_SEMANTIC_OWNER
    )
    assert catalog["semantic_backend"] == "rust"
    assert catalog["catalog_validated"] is True
    assert catalog["status"] == "ready"
    assert catalog["catalog_id_rule"] == st.ZSPACE_RUNTIME_PROTOCOL_CATALOG_ID_RULE
    assert catalog["catalog_id"].startswith("sha256:")
    assert [protocol["name"] for protocol in catalog["protocols"]] == [
        "generation_evidence",
        "repetition_unlikelihood",
        "semantic_review",
    ]
    assert st.validate_zspace_runtime_protocol_catalog(catalog) == catalog

    tampered = copy.deepcopy(catalog)
    tampered["protocols"][0]["clients"][2]["operations"][0] = "browserOwnedSemantics"
    with pytest.raises(
        ValueError, match="does not match the current Rust-owned surface"
    ):
        st.validate_zspace_runtime_protocol_catalog(tampered)


def test_runtime_protocol_catalog_matches_python_and_wasm_public_surfaces() -> None:
    catalog = st.zspace_runtime_protocol_catalog()
    declarations = WASM_DECLARATIONS.read_text(encoding="utf-8")

    for protocol in catalog["protocols"]:
        assert isinstance(protocol, Mapping)
        surfaces = {
            surface["client"]: surface
            for surface in protocol["clients"]
            if isinstance(surface, Mapping)
        }
        assert list(surfaces) == ["rust", "python", "wasm"]
        for operation in surfaces["python"]["operations"]:
            assert hasattr(st, operation), f"missing Python operation {operation}"
            assert operation in st.__all__
        for operation in surfaces["wasm"]["operations"]:
            assert f"function {operation}(" in declarations
            assert "legacy" not in operation.lower()
            assert operation.endswith("Json")
            assert "Object" not in operation
        assert surfaces["python"]["transport"] == "bounded_mapping"
        assert surfaces["wasm"]["transport"] == "bounded_json"
        assert surfaces["wasm"]["trusted_legacy_replay"] is False

    assert "function validateZspaceRuntimeProtocolCatalogObject(" not in declarations


def test_runtime_protocol_catalog_public_surface_is_typed_and_exported() -> None:
    expected = {
        "ZSPACE_RUNTIME_PROTOCOL_CATALOG_CONTRACT_VERSION",
        "ZSPACE_RUNTIME_PROTOCOL_CATALOG_ID_RULE",
        "ZSPACE_RUNTIME_PROTOCOL_CATALOG_KIND",
        "ZSPACE_RUNTIME_PROTOCOL_CATALOG_SEMANTIC_BACKEND",
        "ZSPACE_RUNTIME_PROTOCOL_CATALOG_SEMANTIC_OWNER",
        "ZSPACE_RUNTIME_PROTOCOL_CATALOG_STATUS",
        "validate_zspace_runtime_protocol_catalog",
        "zspace_runtime_protocol_catalog",
    }

    assert expected <= set(st.__all__)
    stub = (Path(st.__file__).with_name("__init__.pyi")).read_text(encoding="utf-8")
    for symbol in expected:
        assert symbol in stub
