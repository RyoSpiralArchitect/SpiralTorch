from __future__ import annotations

import copy

import pytest

import spiraltorch as st


def _request():
    return {
        "config": {
            "normalization": "eligible_targets",
            "schedule": {
                "kind": "linear_decay",
                "start_update": 2,
                "end_update": 6,
                "final_scale": 0.0,
            },
        },
        "base_strength": 0.2,
        "completed_update_slots": 4,
        "active_position_count": 2,
        "eligible_target_count": 8,
    }


def test_public_rust_coefficient_and_portable_policy_identity():
    assert "zspace_repetition_objective_control" in st.__all__
    request = _request()
    control = st.zspace_repetition_objective_control(**request)
    assert control["effective_strength"] == 0.025
    assert (
        control["policy"]["semantic_owner"]
        == "st-core::runtime::zspace_repetition_objective"
    )
    request["completed_update_slots"] = 6
    resumed = st.zspace_repetition_objective_control(**request)
    assert resumed["effective_strength"] == 0.0
    assert resumed["policy"] == control["policy"]
    request["config"]["normalization"] = "active_positions"
    assert (
        st.zspace_repetition_objective_control(**request)["policy"] != control["policy"]
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("completed_update_slots", True),
        ("completed_update_slots", 1.5),
        ("completed_update_slots", -1),
        ("completed_update_slots", 2**53),
        ("base_strength", float("nan")),
        ("base_strength", float("inf")),
        ("active_position_count", 9),
        ("eligible_target_count", 0),
    ],
)
def test_invalid_requests_reach_rust_validation(field, value):
    request = _request()
    request[field] = value
    with pytest.raises((TypeError, ValueError)):
        st.zspace_repetition_objective_control(**request)


@pytest.mark.parametrize("kind", ["constant", "linear_decay"])
def test_unknown_schedule_fields_are_not_silently_ignored(kind):
    request = copy.deepcopy(_request())
    if kind == "constant":
        request["config"]["schedule"] = {"kind": kind}
    request["config"]["schedule"]["end_udpate"] = 6
    with pytest.raises(ValueError):
        st.zspace_repetition_objective_control(**request)
