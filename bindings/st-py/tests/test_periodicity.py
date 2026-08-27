from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence

import pytest

import spiraltorch as st


def _sha_id(character: str) -> str:
    return "sha256:" + character * 64


def test_periodicity_is_rust_owned_replayable_and_exported() -> None:
    report = st.zspace_periodicity([9, 1, 2, 1, 2, 1], appended_token_id=2)

    assert report["contract_version"] == st.ZSPACE_PERIODICITY_CONTRACT_VERSION
    assert report["kind"] == st.ZSPACE_PERIODICITY_KIND
    assert report["semantic_owner"] == st.ZSPACE_PERIODICITY_SEMANTIC_OWNER
    assert report["semantic_backend"] == "rust"
    assert report["analysis_scope"] == "observed_sequence_with_appended_token"
    assert report["input_token_count"] == 6
    assert report["effective_token_count"] == 7
    assert report["periodic_loop_detected"] is True
    assert report["periodic_suffix"] == {
        "period": 2,
        "token_count": 6,
        "repeated_token_count": 4,
        "repetition_count": 3,
    }
    assert report["efficacy_claim_ready"] is False
    assert st.validate_zspace_periodicity(report) == report
    assert {
        "ZSPACE_PERIODICITY_CONTRACT_VERSION",
        "ZSPACE_PERIODICITY_RULE",
        "validate_zspace_periodicity",
        "zspace_periodicity",
    } <= set(st.__all__)


def test_periodicity_matches_generation_evidence_shared_kernel() -> None:
    token_ids = [1, 2, 1, 2, 1, 2]
    periodicity = st.zspace_periodicity(token_ids)
    evidence = st.zspace_generation_evidence(
        protocol_id=_sha_id("a"),
        runtime_identity_id=_sha_id("b"),
        model_artifact_id=_sha_id("c"),
        prompt_set_id=_sha_id("d"),
        decoding_config_id=_sha_id("e"),
        samples=[
            {
                "prompt_id": _sha_id("f"),
                "seed": 17,
                "continuation_token_ids": token_ids,
            }
        ],
    )
    sample = evidence["samples"][0]
    suffix = periodicity["periodic_suffix"]

    assert suffix is not None
    assert sample["periodic_suffix_period"] == suffix["period"]
    assert sample["periodic_suffix_token_count"] == suffix["token_count"]
    assert (
        sample["periodic_suffix_repeated_token_count"]
        == suffix["repeated_token_count"]
    )
    assert sample["periodic_suffix_repetition_count"] == suffix["repetition_count"]
    assert (
        sample["periodic_suffix_repeated_token_ratio"]
        == periodicity["periodic_suffix_repeated_token_ratio"]
    )


def test_periodicity_reports_no_match_without_inventing_a_suffix() -> None:
    report = st.zspace_periodicity([1, 2, 3])

    assert report["analysis_scope"] == "observed_sequence"
    assert report["periodic_loop_detected"] is False
    assert report["periodic_suffix"] is None
    assert report["periodic_suffix_token_ratio"] == 0.0
    assert report["periodic_suffix_repeated_token_ratio"] == 0.0


def test_periodicity_fails_closed_on_unsafe_or_unbounded_requests() -> None:
    with pytest.raises(ValueError, match="cross-client maximum"):
        st.zspace_periodicity([st.ZSPACE_PERIODICITY_MAX_SAFE_INTEGER + 1])

    with pytest.raises(ValueError, match="maximum_period"):
        st.zspace_periodicity([1], maximum_period=0)

    with pytest.raises(ValueError, match="minimum_repetitions"):
        st.zspace_periodicity([1], minimum_repetitions=1)

    with pytest.raises(ValueError, match="comparison work"):
        st.zspace_periodicity(
            [1] * 8_192,
            maximum_period=st.ZSPACE_PERIODICITY_MAX_PERIOD,
            minimum_repetitions=2,
        )


def test_periodicity_rejects_active_container_hooks_before_native_work() -> None:
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

    class ActiveSequence(Sequence[int]):
        @property
        def __class__(self) -> type[object]:
            raise AssertionError("custom __class__ must not run")

        def __getitem__(self, _index: int) -> int:
            raise AssertionError("custom __getitem__ must not run")

        def __iter__(self):
            raise AssertionError("custom __iter__ must not run")

        def __len__(self) -> int:
            raise AssertionError("custom __len__ must not run")

    class HostileList(list[int]):
        def __iter__(self):
            raise AssertionError("overridden __iter__ must not run")

        def __len__(self) -> int:
            raise AssertionError("overridden __len__ must not run")

    class HostileDict(dict[str, object]):
        def __iter__(self):
            raise AssertionError("overridden __iter__ must not run")

        def __len__(self) -> int:
            raise AssertionError("overridden __len__ must not run")

        def items(self):
            raise AssertionError("overridden items must not run")

    report = st.zspace_periodicity(HostileList([1, 2, 1, 2, 1, 2]))
    assert st.validate_zspace_periodicity(HostileDict(report)) == report

    with pytest.raises(TypeError, match="list or tuple for bounded admission"):
        st.zspace_periodicity(ActiveSequence())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="dict-backed mapping"):
        st.validate_zspace_periodicity(ActiveMapping())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="payload must be JSON-like"):
        st._rs._zspace_periodicity_validate(ActiveMapping())


def test_periodicity_validator_rejects_request_and_result_tampering() -> None:
    report = st.zspace_periodicity([1, 2, 1, 2, 1, 2])

    tampered_result = copy.deepcopy(report)
    tampered_result["periodic_loop_detected"] = False
    with pytest.raises(ValueError, match="canonical Rust periodicity analysis"):
        st.validate_zspace_periodicity(tampered_result)

    tampered_request = copy.deepcopy(report)
    tampered_request["request"]["token_ids"][0] = 2
    with pytest.raises(ValueError, match="canonical Rust periodicity analysis"):
        st.validate_zspace_periodicity(tampered_request)
