from __future__ import annotations

import copy
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

import spiraltorch as st
from spiraltorch import semantic_review


TRACKED_ALICE_PACKET = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "benchmarks"
    / "hf_periodic_baseline_replication_pythia70m_alice_semantic_review_packet_20260823.json"
)
TRACKED_DISTILGPT2_PACKET = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "benchmarks"
    / "hf_periodic_second_model_distilgpt2_alice_eval_parity_semantic_review_packet_20260823.json"
)
TRACKED_DISTILGPT2_REPORT = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "benchmarks"
    / "hf_periodic_second_model_distilgpt2_alice_eval_parity_64step_20260823.json"
)


def _identity(character: str) -> str:
    return "sha256:" + character * 64


def _packet() -> dict[str, object]:
    request: dict[str, object] = {
        "protocol_id": _identity("a"),
        "prompt_set_id": _identity("b"),
        "blinding_key_sha256": "c" * 64,
        "instructions": "Score each candidate while blind.",
        "rubric": {
            "fluency": "integer 1 through 5",
            "prompt_relevance": "integer 1 through 5",
            "local_coherence": "integer 1 through 5",
            "non_repetition": "integer 1 through 5",
            "preference": "A, B, C, or tie",
        },
        "groups": [
            {
                "group_id": _identity("1"),
                "prompt": "A lighthouse",
                "candidates": [
                    {"candidate_label": label, "continuation": f" shines {label}"}
                    for label in ("A", "B", "C")
                ],
            },
            {
                "group_id": _identity("2"),
                "prompt": "The quiet train",
                "candidates": [
                    {"candidate_label": label, "continuation": f" arrives {label}"}
                    for label in ("A", "B", "C")
                ],
            },
        ],
    }
    request["blinding_map_id"] = st.zspace_semantic_review_map_id(_map_entries(request))
    receipt = st.seal_zspace_semantic_review_packet(**request)
    packet = receipt["packet"]
    assert isinstance(packet, dict)
    return packet


def _score(label: str, value: int) -> dict[str, object]:
    return {
        "candidate_label": label,
        "fluency": value,
        "prompt_relevance": value,
        "local_coherence": value,
        "non_repetition": value,
    }


def _response(
    group_id: str, preference: str, values: tuple[int, int, int]
) -> dict[str, object]:
    return {
        "group_id": group_id,
        "scores": [
            _score("C", values[2]),
            _score("A", values[0]),
            _score("B", values[1]),
        ],
        "preference": preference,
    }


def _draft(packet: dict[str, object], *, complete: bool) -> dict[str, object]:
    groups = packet["groups"]
    assert isinstance(groups, list)
    responses = [_response(str(groups[0]["group_id"]), "A", (5, 3, 1))]
    if complete:
        responses.append(_response(str(groups[1]["group_id"]), "tie", (4, 2, 3)))
    return st.new_zspace_semantic_review_draft(
        packet_id=str(packet["packet_id"]),
        reviewer_id=_identity("d"),
        review_session_id=_identity("e"),
        responses=responses,
    )


def _map_entries(packet: dict[str, object]) -> list[dict[str, object]]:
    groups = packet["groups"]
    assert isinstance(groups, list)
    return [
        {
            "group_id": groups[1]["group_id"],
            "seed": 17,
            "prompt_id": _identity("4"),
            "candidate_to_arm": {
                "A": "periodic",
                "B": "baseline",
                "C": "history",
            },
        },
        {
            "group_id": groups[0]["group_id"],
            "seed": 13,
            "prompt_id": _identity("3"),
            "candidate_to_arm": {
                "A": "baseline",
                "B": "history",
                "C": "periodic",
            },
        },
    ]


def _map(packet: dict[str, object]) -> dict[str, object]:
    return {
        "schema": st.ZSPACE_SEMANTIC_REVIEW_MAP_SCHEMA,
        "status": "sealed_pending_review",
        "protocol_id": packet["protocol_id"],
        "packet_id": packet["packet_id"],
        "blinding_key_sha256": packet["blinding_key_sha256"],
        "entries": _map_entries(packet),
    }


@pytest.mark.parametrize(
    ("packet_path", "expected_packet_id", "expected_blinding_map_id"),
    [
        (
            TRACKED_ALICE_PACKET,
            "sha256:0ab7cc9853886e895074bce4cfb5338be8087521971b2a92346cdab6f8924f83",
            "sha256:4b696e405b0db68a870613065c9516da953adac73bc17c086dbebf77e3ca8aec",
        ),
        (
            TRACKED_DISTILGPT2_PACKET,
            "sha256:a19f0c5ecf948edbe1df30d99ab6ab1e803f3d7dd51c98383447ac7312446191",
            "sha256:79d7da15d41cfe04d2528e142c52d2f874fc04d22d9e08b11b8fe03b71198bf8",
        ),
    ],
)
def test_tracked_packets_match_rust_commitment(
    packet_path: Path,
    expected_packet_id: str,
    expected_blinding_map_id: str,
) -> None:
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    expected_exports = {
        "new_zspace_semantic_review_draft",
        "seal_zspace_semantic_review_packet",
        "summarize_zspace_semantic_review_draft",
        "unblind_zspace_semantic_review",
        "validate_zspace_semantic_review_draft_receipt",
        "validate_zspace_semantic_review_draft_receipt_trusted_legacy_replay",
        "validate_zspace_semantic_review_packet",
        "validate_zspace_semantic_review_packet_trusted_legacy_replay",
        "validate_zspace_semantic_review_packet_receipt",
        "validate_zspace_semantic_review_packet_receipt_trusted_legacy_replay",
        "validate_zspace_semantic_review_unblind",
        "validate_zspace_semantic_review_unblind_trusted_legacy_replay",
        "zspace_semantic_review_map_id",
        "zspace_semantic_review_map_id_trusted_legacy_replay",
    }

    receipt = st.validate_zspace_semantic_review_packet(packet)
    sealed = st.seal_zspace_semantic_review_packet(
        protocol_id=packet["protocol_id"],
        prompt_set_id=packet["prompt_set_id"],
        blinding_key_sha256=packet["blinding_key_sha256"],
        blinding_map_id=packet["blinding_map_id"],
        instructions=packet["instructions"],
        rubric=packet["rubric"],
        groups=packet["groups"],
    )

    assert expected_exports <= set(st.__all__)
    assert sealed == receipt
    assert st.ZSPACE_SEMANTIC_REVIEW_PACKET_STATUS == "ready_for_blinded_review"
    assert st.ZSPACE_SEMANTIC_REVIEW_MAP_STATUS == "sealed_pending_review"
    assert st.ZSPACE_SEMANTIC_REVIEW_MAX_PACKET_TEXT_BYTES == 32 * 1_024 * 1_024
    assert st.ZSPACE_SEMANTIC_REVIEW_MAX_ARM_NAME_BYTES == 128
    assert receipt["packet_id"] == expected_packet_id
    assert receipt["group_count"] == 36
    assert receipt["candidate_count"] == 108
    assert receipt["semantic_backend"] == "rust"
    assert receipt["blinding_map_id"] == expected_blinding_map_id
    assert receipt["human_review_complete"] is False
    assert st.validate_zspace_semantic_review_packet_receipt(receipt) == receipt
    assert (
        st.validate_zspace_semantic_review_packet_trusted_legacy_replay(packet)
        == receipt
    )
    assert (
        st.validate_zspace_semantic_review_packet_receipt_trusted_legacy_replay(receipt)
        == receipt
    )


def test_tracked_distilgpt2_report_preserves_bounded_negative_decision() -> None:
    packet = json.loads(TRACKED_DISTILGPT2_PACKET.read_text(encoding="utf-8"))
    report = json.loads(TRACKED_DISTILGPT2_REPORT.read_text(encoding="utf-8"))
    receipt = st.validate_zspace_semantic_review_packet(packet)

    assert report["status"] == "ready_but_negative"
    assert report["decision"] == "ready_but_negative_prespecified_gate"
    assert report["semantic_review"]["packet_receipt"] == receipt
    assert report["acceptance"] == {
        "readiness_passed": True,
        "identity_passed": True,
        "before_train_comparability_passed": True,
        "mechanism_passed": True,
        "held_out_ce_safety_passed": True,
        "periodic_vs_history_replication_passed": False,
        "periodic_vs_baseline_absolute_control_passed": True,
        "semantic_packet_ready": True,
        "human_semantic_review_complete": False,
    }


def test_draft_progress_only_seals_complete_review_and_is_order_independent() -> None:
    packet = _packet()
    partial = st.summarize_zspace_semantic_review_draft(
        packet=packet,
        draft=_draft(packet, complete=False),
    )

    assert partial["status"] == "in_progress"
    assert partial["completed_group_count"] == 1
    assert partial["remaining_group_count"] == 1
    assert partial["response_id"] is None
    assert partial["unblind_ready"] is False

    complete_draft = _draft(packet, complete=True)
    reversed_draft = copy.deepcopy(complete_draft)
    reversed_draft["responses"].reverse()  # type: ignore[union-attr]
    complete = st.summarize_zspace_semantic_review_draft(
        packet=packet,
        draft=complete_draft,
    )
    reversed_receipt = st.summarize_zspace_semantic_review_draft(
        packet=packet,
        draft=reversed_draft,
    )

    assert complete == reversed_receipt
    assert complete["status"] == "ready_for_unblind"
    assert complete["human_review_complete"] is True
    assert complete["unblind_ready"] is True
    assert str(complete["response_id"]).startswith("sha256:")
    assert st.validate_zspace_semantic_review_draft_receipt(complete) == complete
    assert (
        st.validate_zspace_semantic_review_draft_receipt_trusted_legacy_replay(complete)
        == complete
    )

    browser_round_trip = copy.deepcopy(complete)
    browser_round_trip["completion_ratio"] = 1
    assert (
        st.validate_zspace_semantic_review_draft_receipt(browser_round_trip) == complete
    )

    tampered = copy.deepcopy(complete)
    tampered["completed_group_count"] = 1
    with pytest.raises(ValueError, match="canonical Rust artifact"):
        st.validate_zspace_semantic_review_draft_receipt(tampered)


def test_semantic_review_rejects_empty_map_and_deep_ingress() -> None:
    with pytest.raises(ValueError, match="at least one entry"):
        st.zspace_semantic_review_map_id([])
    assert st.zspace_semantic_review_map_id_trusted_legacy_replay([]).startswith(
        "sha256:"
    )
    with pytest.raises(ValueError, match="count exceeds maximum 1"):
        semantic_review._bounded_mapping_sequence([{}, {}], maximum=1, label="test row")

    nested: dict[str, object] = {}
    cursor = nested
    for _ in range(34):
        child: dict[str, object] = {}
        cursor["nested"] = child
        cursor = child
    with pytest.raises(ValueError, match="deeply nested"):
        st.validate_zspace_semantic_review_packet(nested)


def test_unblind_requires_complete_review_and_aggregates_arm_and_seed_scores() -> None:
    packet = _packet()
    with pytest.raises(ValueError, match="incomplete"):
        st.unblind_zspace_semantic_review(
            packet=packet,
            draft=_draft(packet, complete=False),
            blinding_map=_map(packet),
        )

    report = st.unblind_zspace_semantic_review(
        packet=packet,
        draft=_draft(packet, complete=True),
        blinding_map=_map(packet),
    )

    assert report["reviewed_group_count"] == 2
    assert report["unblinded_candidate_count"] == 6
    assert report["arm_count"] == 3
    assert report["tie_preference_count"] == 1
    assert report["reviewer_blinding_behavior_verified"] is False
    baseline = next(row for row in report["arms"] if row["arm"] == "baseline")
    assert baseline["mean_scores"]["fluency"] == 3.5
    assert baseline["preference_win_count"] == 1
    assert st.validate_zspace_semantic_review_unblind(report) == report
    assert (
        st.validate_zspace_semantic_review_unblind_trusted_legacy_replay(report)
        == report
    )


def test_unblind_rejects_a_valid_but_post_review_assignment_swap() -> None:
    packet = _packet()
    blinding_map = _map(packet)
    entries = blinding_map["entries"]
    assert isinstance(entries, list)
    candidate_to_arm = entries[0]["candidate_to_arm"]
    assert isinstance(candidate_to_arm, dict)
    candidate_to_arm["A"], candidate_to_arm["B"] = (
        candidate_to_arm["B"],
        candidate_to_arm["A"],
    )

    with pytest.raises(ValueError, match="pre-review commitment"):
        st.unblind_zspace_semantic_review(
            packet=packet,
            draft=_draft(packet, complete=True),
            blinding_map=blinding_map,
        )


def test_resumable_cli_saves_only_complete_validated_groups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    packet_path = tmp_path / "packet.json"
    draft_path = tmp_path / "draft.json"
    packet_path.write_text(json.dumps(_packet()), encoding="utf-8")

    first_answers: Iterator[str] = iter(["3"] * 12 + ["A"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(first_answers))
    assert (
        semantic_review.main(
            [
                "review",
                str(packet_path),
                "--draft",
                str(draft_path),
                "--reviewer-id",
                _identity("d"),
                "--review-session-id",
                _identity("e"),
                "--max-groups",
                "1",
            ]
        )
        == 2
    )
    first_draft = json.loads(draft_path.read_text(encoding="utf-8"))
    assert len(first_draft["responses"]) == 1

    second_answers: Iterator[str] = iter(["4"] * 12 + ["tie"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(second_answers))
    assert (
        semantic_review.main(
            [
                "review",
                str(packet_path),
                "--draft",
                str(draft_path),
                "--max-groups",
                "1",
            ]
        )
        == 0
    )
    complete_draft = json.loads(draft_path.read_text(encoding="utf-8"))
    receipt = st.summarize_zspace_semantic_review_draft(
        packet=_packet(), draft=complete_draft
    )
    assert receipt["human_review_complete"] is True
    assert len(complete_draft["responses"]) == 2
    capsys.readouterr()


def test_cli_does_not_read_map_before_review_is_complete(tmp_path: Path) -> None:
    packet = _packet()
    packet_path = tmp_path / "packet.json"
    draft_path = tmp_path / "draft.json"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    draft_path.write_text(
        json.dumps(_draft(packet, complete=False)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="blinding map was not read"):
        semantic_review.main(
            [
                "unblind",
                str(packet_path),
                str(draft_path),
                str(tmp_path / "missing-map.json"),
            ]
        )


def test_terminal_rendering_escapes_control_and_format_characters() -> None:
    assert semantic_review._terminal_text("safe\x1b[2J\u202etext") == (
        "safe\\u001b[2J\\u202etext"
    )
