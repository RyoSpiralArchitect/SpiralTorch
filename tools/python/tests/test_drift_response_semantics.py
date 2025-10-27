from __future__ import annotations

import math
import pathlib
import sys

_MODULE_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_ROOT))

import drift_response_semantics as drs  # noqa: E402
from drift_response_linguistics import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    DirectionQuery,
    DirectionalAxis,
    FrameState,
    WordState,
    aggregate_penalty as drl_aggregate_penalty,
    analyse_word as drl_analyse_word,
    frame_summary as drl_frame_summary,
)


def _sample_word() -> WordState:
    return WordState(
        name="AI",
        definition_entropy=0.72,
        timing_signal=1.4,
        base_lambda=1.1,
        beta=1.05,
        frames={
            "Normative": FrameState(
                phi=0.65,
                c=0.9,
                S=0.8,
                a_den=-0.05,
                a_con=0.2,
                b_den=0.4,
                b_con=0.8,
                kappa=0.35,
                timing_scale=1.2,
                curvature_a_den=-0.02,
                curvature_a_con=0.05,
                curvature_b_den=0.03,
                curvature_b_con=0.07,
                kappa_slope=0.1,
                directional_axes={
                    "axial": DirectionalAxis(
                        value_components=(1.0, 0.3),
                        risk_components=(0.6, 0.2),
                        kappa_components=(0.4, 0.1),
                        value_curvature_components=(0.02, 0.01),
                        risk_curvature_components=(0.03, 0.02),
                        kappa_slope_components=(0.05, 0.01),
                    )
                },
            )
        },
    )


def test_semantics_wrapper_matches_linguistics() -> None:
    word = _sample_word()
    metrics_sem = drs.analyse_word(word, DEFAULT_THRESHOLDS)
    metrics_drl = drl_analyse_word(word, DEFAULT_THRESHOLDS)
    assert metrics_sem == metrics_drl


def test_direction_queries_flow_through_adapter() -> None:
    word = _sample_word()
    queries = {
        "Normative": [DirectionQuery(axis="axial", weights=(0.7, 0.3))],
    }

    metrics_sem = drs.analyse_word(
        word,
        DEFAULT_THRESHOLDS,
        direction_queries=queries,
    )
    metrics_drl = drl_analyse_word(
        word,
        DEFAULT_THRESHOLDS,
        direction_queries=queries,
    )

    assert metrics_sem.direction_signatures == metrics_drl.direction_signatures
    assert "Normative" in metrics_sem.direction_signatures
    signature = metrics_sem.direction_signatures["Normative"]["axial"]
    assert signature.safe_radius is not None


def test_penalty_and_summary_share_implementation() -> None:
    word = _sample_word()
    metrics = [drs.analyse_word(word, DEFAULT_THRESHOLDS)]
    sem_penalty = drs.aggregate_penalty(metrics, min_radius=0.15)
    drl_penalty = drl_aggregate_penalty(metrics, min_radius=0.15)
    assert math.isclose(sem_penalty, drl_penalty, rel_tol=1e-12)

    sem_summary = drs.frame_summary(metrics[0])
    drl_summary = drl_frame_summary(metrics[0])
    assert sem_summary == drl_summary
