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


def test_repetition_unlikelihood_filters_model_topk_against_history() -> None:
    plan = st.zspace_repetition_unlikelihood_plan(
        strength=0.1,
        candidate_source="model_topk_history",
        proposal_top_k=4,
        context_window=16,
        max_candidates_per_position=8,
        sequences=[
            {
                "token_ids": [7, 2, 3, 2, 4],
                "token_mask": [True] * 5,
                "label_mask": [False, False, False, False, True],
                "proposal_token_ids": [[], [], [], [], [2, 3, 4, 9]],
            }
        ],
    )

    assert plan["proposal_owner"] == "model-client-no-grad"
    position = plan["positions"][0]  # type: ignore[index]
    assert position["candidates"] == [  # type: ignore[index]
        {
            "token_id": 2,
            "occurrence_count": 2,
            "most_recent_distance": 1,
            "proposal_rank": 0,
        },
        {
            "token_id": 3,
            "occurrence_count": 1,
            "most_recent_distance": 2,
            "proposal_rank": 1,
        },
    ]
    aggregate = plan["aggregate"]
    assert aggregate["proposal_count"] == 4  # type: ignore[index]
    assert aggregate["excluded_target_proposal_count"] == 1  # type: ignore[index]
    assert aggregate["excluded_out_of_history_proposal_count"] == 1  # type: ignore[index]
    assert st.validate_zspace_repetition_unlikelihood_plan(plan) == plan


def test_repetition_unlikelihood_filters_model_topk_to_periodic_suffixes() -> None:
    plan = st.zspace_repetition_unlikelihood_plan(
        strength=0.1,
        candidate_source="model_topk_periodic",
        proposal_top_k=4,
        context_window=16,
        max_candidates_per_position=8,
        sequences=[
            {
                "token_ids": [9, 1, 2, 1, 2, 1, 7],
                "token_mask": [True] * 7,
                "label_mask": [False, False, False, False, False, False, True],
                "proposal_token_ids": [[], [], [], [], [], [], [2, 1, 7, 8]],
            }
        ],
    )

    assert plan["contract_version"].endswith(".v3")
    assert plan["periodic_suffix_max_period"] == 16
    assert plan["periodic_suffix_min_repetitions"] == 3
    position = plan["positions"][0]  # type: ignore[index]
    assert position["candidates"] == [  # type: ignore[index]
        {
            "token_id": 2,
            "occurrence_count": 2,
            "most_recent_distance": 2,
            "proposal_rank": 0,
            "periodic_suffix_period": 2,
            "periodic_suffix_token_count": 6,
            "periodic_suffix_repeated_token_count": 4,
            "periodic_suffix_repetition_count": 3,
        }
    ]
    aggregate = plan["aggregate"]
    assert aggregate["excluded_target_proposal_count"] == 1  # type: ignore[index]
    assert aggregate["excluded_out_of_history_proposal_count"] == 1  # type: ignore[index]
    assert aggregate["excluded_non_periodic_proposal_count"] == 1  # type: ignore[index]
    assert aggregate["periodic_candidate_count"] == 1  # type: ignore[index]
    assert st.validate_zspace_repetition_unlikelihood_plan(plan) == plan


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
        "ZSPACE_REPETITION_UNLIKELIHOOD_PROPOSAL_RULE",
        "ZSPACE_REPETITION_UNLIKELIHOOD_PERIODIC_SUFFIX_MAX_PERIOD",
        "ZSPACE_REPETITION_UNLIKELIHOOD_PERIODIC_SUFFIX_MIN_REPETITIONS",
        "ZSPACE_REPETITION_UNLIKELIHOOD_SEMANTIC_OWNER",
        "validate_zspace_repetition_unlikelihood_plan",
        "zspace_repetition_unlikelihood_plan",
    }

    assert expected <= set(st.__all__)
