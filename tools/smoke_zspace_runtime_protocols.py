#!/usr/bin/env python3
"""Exercise Rust-owned Z-space protocols from an installed wheel."""

from __future__ import annotations

import json
from collections.abc import Mapping

import spiraltorch as st


def identity(character: str) -> str:
    return "sha256:" + character * 64


def generation_evidence_smoke() -> dict[str, object]:
    report = st.zspace_generation_evidence(
        protocol_id=identity("a"),
        runtime_identity_id=identity("b"),
        model_artifact_id=identity("c"),
        prompt_set_id=identity("d"),
        decoding_config_id=identity("e"),
        samples=[
            {
                "prompt_id": identity("1"),
                "seed": 17,
                "continuation_token_ids": [9, 1, 2, 1, 2, 1, 2],
            }
        ],
    )
    assert st.validate_zspace_generation_evidence(report) == report
    assert report["semantic_backend"] == "rust"
    assert report["aggregate"]["periodic_loop_sample_count"] == 1
    return {
        "evidence_id": report["evidence_id"],
        "sample_count": report["sample_count"],
    }


def periodicity_smoke() -> dict[str, object]:
    report = st.zspace_periodicity(
        [9, 1, 2, 1, 2, 1],
        appended_token_id=2,
    )
    assert st.validate_zspace_periodicity(report) == report
    assert report["semantic_backend"] == "rust"
    assert report["periodic_loop_detected"] is True
    assert report["periodic_suffix"]["period"] == 2
    return {
        "analysis_id": report["analysis_id"],
        "period": report["periodic_suffix"]["period"],
    }


def stochastic_schrodinger_smoke() -> dict[str, object]:
    forward = st.zspace_stochastic_schrodinger_forward(
        [1.0, 0.25, -0.5, 0.75],
        [0.2, -0.1],
        standard_normal=[0.1, -0.3, 0.2, 0.0],
        config={
            "time_step": 0.08,
            "hopping_rate": 0.35,
            "loss_rate": 0.02,
            "noise_scale": 0.15,
        },
    )
    assert st.validate_zspace_stochastic_schrodinger_forward(forward) == forward
    assert forward["semantic_backend"] == "rust"
    assert forward["forward_validated"] is True

    vjp = st.zspace_stochastic_schrodinger_vjp(
        forward,
        [0.2, -0.4, 0.1, 0.3],
    )
    assert st.validate_zspace_stochastic_schrodinger_vjp(vjp) == vjp
    assert vjp["semantic_backend"] == "rust"
    assert vjp["forward_id"] == forward["forward_id"]
    assert len(vjp["result"]["grad_input"]) == 4
    assert len(vjp["result"]["grad_potential"]) == 2
    return {
        "forward_id": forward["forward_id"],
        "vjp_id": vjp["vjp_id"],
    }


def repetition_unlikelihood_smoke() -> dict[str, object]:
    plan = st.zspace_repetition_unlikelihood_plan(
        sequences=[
            {
                "token_ids": [1, 2, 3, 1, 2, 4],
                "token_mask": [True] * 6,
                "label_mask": [True] * 6,
            }
        ],
        strength=0.25,
        ngram_order=3,
        context_window=16,
        max_candidates_per_position=8,
    )
    assert st.validate_zspace_repetition_unlikelihood_plan(plan) == plan
    assert plan["semantic_backend"] == "rust"
    assert plan["plan_validated"] is True
    return {
        "plan_id": plan["plan_id"],
        "candidate_count": plan["aggregate"]["candidate_count"],
    }


def semantic_review_smoke() -> dict[str, object]:
    protocol_id = identity("6")
    prompt_set_id = identity("7")
    group_id = identity("8")
    prompt_id = identity("9")
    blinding_key_sha256 = "f" * 64
    entries = [
        {
            "group_id": group_id,
            "seed": 17,
            "prompt_id": prompt_id,
            "candidate_to_arm": {
                "A": "baseline",
                "B": "history",
                "C": "periodic",
            },
        }
    ]
    blinding_map_id = st.zspace_semantic_review_map_id(entries)
    packet_receipt = st.seal_zspace_semantic_review_packet(
        protocol_id=protocol_id,
        prompt_set_id=prompt_set_id,
        blinding_key_sha256=blinding_key_sha256,
        blinding_map_id=blinding_map_id,
        instructions="Score every candidate while blind.",
        rubric={
            "fluency": "integer 1 through 5",
            "prompt_relevance": "integer 1 through 5",
            "local_coherence": "integer 1 through 5",
            "non_repetition": "integer 1 through 5",
            "preference": "A, B, C, or tie",
        },
        groups=[
            {
                "group_id": group_id,
                "prompt": "SpiralTorch is",
                "candidates": [
                    {"candidate_label": "A", "continuation": " a runtime."},
                    {"candidate_label": "B", "continuation": " a geometry."},
                    {"candidate_label": "C", "continuation": " a protocol."},
                ],
            }
        ],
    )
    packet = packet_receipt["packet"]
    assert isinstance(packet, Mapping)
    assert st.validate_zspace_semantic_review_packet(packet) == packet_receipt
    assert (
        st.validate_zspace_semantic_review_packet_receipt(packet_receipt)
        == packet_receipt
    )

    scores = [
        {
            "candidate_label": label,
            "fluency": value,
            "prompt_relevance": value,
            "local_coherence": value,
            "non_repetition": value,
        }
        for label, value in [("A", 4), ("B", 3), ("C", 5)]
    ]
    draft = st.new_zspace_semantic_review_draft(
        packet_id=packet["packet_id"],
        reviewer_id=identity("2"),
        review_session_id=identity("3"),
        responses=[
            {
                "group_id": group_id,
                "scores": scores,
                "preference": "C",
            }
        ],
    )
    draft_receipt = st.summarize_zspace_semantic_review_draft(
        packet=packet,
        draft=draft,
    )
    assert (
        st.validate_zspace_semantic_review_draft_receipt(draft_receipt) == draft_receipt
    )

    blinding_map = {
        "schema": st.ZSPACE_SEMANTIC_REVIEW_MAP_SCHEMA,
        "status": st.ZSPACE_SEMANTIC_REVIEW_MAP_STATUS,
        "protocol_id": protocol_id,
        "packet_id": packet["packet_id"],
        "blinding_key_sha256": blinding_key_sha256,
        "entries": entries,
    }
    unblind = st.unblind_zspace_semantic_review(
        packet=packet,
        draft=draft,
        blinding_map=blinding_map,
    )
    assert st.validate_zspace_semantic_review_unblind(unblind) == unblind
    assert unblind["semantic_backend"] == "rust"
    return {
        "packet_id": packet["packet_id"],
        "draft_id": draft_receipt["draft_id"],
        "unblind_id": unblind["unblind_id"],
    }


def main() -> int:
    catalog = st.zspace_runtime_protocol_catalog()
    assert st.validate_zspace_runtime_protocol_catalog(catalog) == catalog
    assert [protocol["name"] for protocol in catalog["protocols"]] == [
        "generation_evidence",
        "periodicity",
        "stochastic_schrodinger",
        "repetition_unlikelihood",
        "semantic_review",
    ]
    for protocol in catalog["protocols"]:
        python_surface = next(
            surface for surface in protocol["clients"] if surface["client"] == "python"
        )
        for operation in python_surface["operations"]:
            assert operation in st.__all__
            assert callable(getattr(st, operation))

    summary = {
        "status": "ok",
        "spiraltorch_version": st.__version__,
        "catalog_id": catalog["catalog_id"],
        "protocol_count": catalog["protocol_count"],
        "generation_evidence": generation_evidence_smoke(),
        "periodicity": periodicity_smoke(),
        "stochastic_schrodinger": stochastic_schrodinger_smoke(),
        "repetition_unlikelihood": repetition_unlikelihood_smoke(),
        "semantic_review": semantic_review_smoke(),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
