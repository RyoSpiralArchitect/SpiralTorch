"""Compare installed native session trajectories with the retained pre-optimization fingerprints."""

import argparse
import hashlib
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from joint_protocol import guard_ml, read, require


def verify(legacy_only):
    guard_ml()
    sys.path.insert(0, str(HERE.parents[2] / "bindings/st-py/tests"))
    from test_native_causal_lm_session import owners, session, st

    reference = read(HERE / "fixture_validation.json")
    base, joint = owners()
    original = session(joint)
    pins = [original.checkpoint_id]
    while original.step() is not None:
        pins.append(original.checkpoint_id)
    require(
        {"v1_pins": pins, "data_id": original.data_id, "config": original.config_json()}
        == reference["one_window_reference"],
        "default v1 checkpoint, config or data identity changed",
    )
    if legacy_only:
        return
    docs = [[1, 2, 3, 4], [8, 7, 6, 5], [5, 4, 3, 2]]
    for mode, model in (
        ("lora", base.with_lora(0, 2, 4.0)),
        ("geometry", base.with_geometry(joint.geometry_artifact())),
        ("joint", joint),
    ):
        for size in (1, 2):
            trainer = st.NativeCausalLmTrainingSession(
                model,
                docs,
                4,
                learning_rate=0.2,
                epochs=2,
                mode=mode,
                batch_windows=size,
            )
            pins, reports = [trainer.checkpoint_id], []
            while (step := trainer.step()) is not None:
                pins.append(trainer.checkpoint_id)
                reports.append(step.to_json())
            stream = hashlib.sha256("\n".join(reports).encode()).hexdigest()
            require(
                [pins, stream] == reference["mode_batch_reference"][f"{mode}/{size}"],
                "changed update stream or checkpoint",
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--legacy-only", action="store_true")
    args = p.parse_args()
    verify(args.legacy_only)
    print(
        "5 default v1 checkpoint pins verified"
        if args.legacy_only
        else "5 default v1 pins, 36 batched/unbatched pins and 6 step streams verified"
    )
