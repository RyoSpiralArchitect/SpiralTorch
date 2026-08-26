from __future__ import annotations

import copy

import pytest

import spiraltorch as st


def _plan(token_ids: list[int]) -> dict[str, object]:
    return st.zspace_repetition_unlikelihood_plan(
        strength=0.1,
        ngram_order=3,
        context_window=16,
        max_candidates_per_position=8,
        sequences=[
            {
                "token_ids": token_ids,
                "token_mask": [True] * len(token_ids),
                "label_mask": [True] * len(token_ids),
            }
        ],
    )


def test_repetition_unlikelihood_plan_is_rust_owned_and_tamper_evident() -> None:
    plan = _plan([1, 2, 3, 1, 2, 4])

    assert (
        plan["contract_version"] == st.ZSPACE_REPETITION_UNLIKELIHOOD_CONTRACT_VERSION
    )
    assert plan["kind"] == st.ZSPACE_REPETITION_UNLIKELIHOOD_KIND
    assert plan["semantic_backend"] == "rust"
    assert plan["differentiation_owner"] == "model-client-autograd"
    assert plan["plan_validated"] is True
    assert plan["efficacy_claim_ready"] is False
    position = plan["positions"][0]  # type: ignore[index]
    assert position["prediction_index"] == 4
    assert position["target_token_id"] == 4
    assert position["candidates"] == [  # type: ignore[index]
        {"token_id": 3, "occurrence_count": 1, "most_recent_distance": 3}
    ]
    assert st.validate_zspace_repetition_unlikelihood_plan(plan) == plan

    tampered = copy.deepcopy(plan)
    tampered["positions"][0]["candidates"][0]["token_id"] = 7  # type: ignore[index]
    with pytest.raises(ValueError, match="canonical Rust plan"):
        st.validate_zspace_repetition_unlikelihood_plan(tampered)


def test_repetition_unlikelihood_excludes_the_supervised_target() -> None:
    plan = _plan([1, 2, 3, 1, 2, 3])

    assert plan["positions"] == []
    aggregate = plan["aggregate"]
    assert aggregate["excluded_target_match_count"] == 1  # type: ignore[index]
    assert aggregate["candidate_count"] == 0  # type: ignore[index]


def test_repetition_unlikelihood_rejects_invalid_masks() -> None:
    with pytest.raises(ValueError, match="label 1 is supervised but masked as invalid"):
        st.zspace_repetition_unlikelihood_plan(
            sequences=[
                {
                    "token_ids": [1, 2, 3],
                    "token_mask": [True, False, True],
                    "label_mask": [True, True, True],
                }
            ]
        )


def test_repetition_unlikelihood_public_surface_is_exported() -> None:
    expected = {
        "ZSPACE_REPETITION_UNLIKELIHOOD_CONTRACT_VERSION",
        "ZSPACE_REPETITION_UNLIKELIHOOD_OBJECTIVE_RULE",
        "ZSPACE_REPETITION_UNLIKELIHOOD_SEMANTIC_OWNER",
        "validate_zspace_repetition_unlikelihood_plan",
        "zspace_repetition_unlikelihood_plan",
    }

    assert expected <= set(st.__all__)
