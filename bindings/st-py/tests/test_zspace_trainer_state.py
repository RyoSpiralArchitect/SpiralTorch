from __future__ import annotations

import spiraltorch as st


def _set_trainer_vectors(trainer: st.ZSpaceTrainer, *, z, moment, velocity, step: int) -> None:
    trainer._z[:] = z  # type: ignore[attr-defined]
    trainer._m[:] = moment  # type: ignore[attr-defined]
    trainer._v[:] = velocity  # type: ignore[attr-defined]
    trainer._t = step  # type: ignore[attr-defined]


def test_load_state_with_empty_vectors_resets_contents() -> None:
    trainer = st.ZSpaceTrainer(z_dim=3)
    _set_trainer_vectors(
        trainer,
        z=[1.0, -2.0, 3.0],
        moment=[0.5, 0.25, -0.75],
        velocity=[0.1, -0.2, 0.3],
        step=7,
    )

    trainer.load_state_dict(
        {
            "z": [],
            "moment": [],
            "velocity": [],
            "step": 2,
        },
        strict=False,
    )

    state = trainer.state_dict()
    assert state["z"] == [0.0, 0.0, 0.0]
    assert state["moment"] == [0.0, 0.0, 0.0]
    assert state["velocity"] == [0.0, 0.0, 0.0]
    assert state["step"] == 2


def test_load_state_with_missing_vectors_preserves_existing_when_non_strict() -> None:
    trainer = st.ZSpaceTrainer(z_dim=2)
    original_state = {
        "z": [0.3, -0.1],
        "moment": [0.7, -0.6],
        "velocity": [0.2, -0.4],
    }
    _set_trainer_vectors(
        trainer,
        z=original_state["z"],
        moment=original_state["moment"],
        velocity=original_state["velocity"],
        step=4,
    )

    trainer.load_state_dict({"step": 9}, strict=False)

    state = trainer.state_dict()
    assert state["z"] == original_state["z"]
    assert state["moment"] == original_state["moment"]
    assert state["velocity"] == original_state["velocity"]
    assert state["step"] == 9
