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
        "periodicity",
        "stochastic_schrodinger",
        "stochastic_schrodinger_complex",
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
        rust_admission = surfaces["rust"]["normal_admission"]
        python_admission = surfaces["python"]["normal_admission"]
        wasm_admission = surfaces["wasm"]["normal_admission"]
        assert rust_admission["profile"] == "typed_native"
        assert rust_admission["limits"] is None
        assert python_admission["profile"] == "passive_json_containers"
        assert wasm_admission["profile"] == "bounded_json_string"
        assert python_admission["limits"] == wasm_admission["limits"]
        assert all(
            isinstance(python_admission["limits"][field], int)
            and python_admission["limits"][field] > 0
            for field in ("maximum_bytes", "maximum_nodes", "maximum_depth")
        )
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
    assert (
        st.validate_zspace_runtime_protocol_catalog.__annotations__["catalog"]
        == "dict[str, object]"
    )
    assert "catalog: Dict[str, object]" in stub
    assert "samples: Sequence[Mapping[str, object]]" not in stub
    assert "samples: List[Dict[str, object]] | Tuple[Dict[str, object], ...]" in stub
    periodicity_stub = stub.split("def zspace_periodicity(", 1)[1].split(
        "def validate_zspace_periodicity(", 1
    )[0]
    assert "token_ids: Sequence[int]" not in periodicity_stub
    assert "token_ids: List[int] | Tuple[int, ...]" in periodicity_stub
    assert (
        st.zspace_periodicity.__annotations__["token_ids"]
        == "list[int] | tuple[int, ...]"
    )


def test_runtime_protocol_catalog_rejects_active_mapping_hooks_in_rust() -> None:
    class ActiveMapping(Mapping[str, object]):
        @property
        def __class__(self) -> type[object]:
            raise AssertionError("custom __class__ must not run")

        def __getitem__(self, _key: str) -> object:
            raise AssertionError("custom __getitem__ must not run")

        def __iter__(self):
            raise AssertionError("custom __iter__ must not run")

        def __len__(self) -> int:
            raise AssertionError("custom __len__ must not run")

        def items(self):
            raise AssertionError("custom items must not run")

    class HostileDict(dict[str, object]):
        def __iter__(self):
            raise AssertionError("overridden __iter__ must not run")

        def __len__(self) -> int:
            raise AssertionError("overridden __len__ must not run")

        def items(self):
            raise AssertionError("overridden items must not run")

    class ActiveClassProxy:
        @property
        def __class__(self) -> type[object]:
            raise AssertionError("custom __class__ must not run")

    with pytest.raises(ValueError, match="payload must be JSON-like"):
        st.validate_zspace_runtime_protocol_catalog(ActiveMapping())
    with pytest.raises(ValueError, match="payload must be JSON-like"):
        st.validate_zspace_runtime_protocol_catalog(ActiveClassProxy())  # type: ignore[arg-type]

    catalog = st.zspace_runtime_protocol_catalog()
    assert st.validate_zspace_runtime_protocol_catalog(HostileDict(catalog)) == catalog
