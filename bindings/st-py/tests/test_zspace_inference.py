from __future__ import annotations

import math

import pytest

pytest.importorskip("spiraltorch")

from spiraltorch import (
    ZSpacePosterior,
    ZSpaceTrainer,
    decode_zspace_embedding,
    infer_from_partial,
    infer_with_trainer,
)


def test_decode_produces_expected_structure():
    vector = [0.12, -0.03, 0.48, -0.2]
    decoded = decode_zspace_embedding(vector)
    assert set(decoded.metrics.keys()) == {"speed", "memory", "stability", "frac", "drs"}
    assert len(decoded.gradient) == len(vector)
    assert math.isclose(sum(decoded.barycentric), 1.0, rel_tol=1e-6)


@pytest.mark.parametrize(
    "partial",
    [
        {"speed": 0.3, "stab": 0.7},
        {"mem": -0.2, "gradient": [0.1, -0.05, 0.0, 0.2]},
    ],
)
def test_infer_from_partial_overrides_requested_metrics(partial):
    vector = [0.2, -0.1, 0.45, -0.05]
    result = infer_from_partial(vector, partial)
    for key, value in partial.items():
        canonical = {
            "mem": "memory",
            "stab": "stability",
        }.get(key, key)
        if canonical == "gradient":
            assert len(result.gradient) == len(vector)
            continue
        assert math.isclose(result.metrics[canonical], float(value))
    assert 0.0 <= result.confidence <= 1.0
    assert math.isclose(sum(result.barycentric), 1.0, rel_tol=1e-6)


def test_posterior_project_matches_helper():
    vector = [0.42, 0.1, -0.25, 0.08]
    partial = {"speed": 0.1, "mem": 0.05}
    posterior = ZSpacePosterior(vector)
    direct = posterior.project(partial)
    helper = infer_from_partial(vector, partial)
    assert math.isclose(direct.residual, helper.residual)
    assert math.isclose(direct.confidence, helper.confidence)
    assert direct.metrics == helper.metrics


def test_decode_accepts_trainer_state():
    trainer = ZSpaceTrainer(z_dim=4)
    trainer.step({"speed": 0.6, "memory": -0.2, "stability": 0.3})
    decoded = decode_zspace_embedding(trainer)
    assert tuple(decoded.z_state) == tuple(trainer.state)
    assert set(decoded.metrics) == {"speed", "memory", "stability", "frac", "drs"}


def test_infer_with_trainer_reflects_partial_updates():
    trainer = ZSpaceTrainer(z_dim=4)
    trainer.step({"speed": 0.4, "memory": 0.1, "stability": 0.2})
    result = infer_with_trainer(trainer, {"stab": 0.9})
    assert math.isclose(result.metrics["stability"], 0.9)
    assert tuple(result.prior.z_state) == tuple(trainer.state)
