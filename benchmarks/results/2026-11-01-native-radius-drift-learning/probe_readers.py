"""Check v1/v2 reader compatibility with installed wheels and generated test weights."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

HERE = Path(__file__).resolve()


def sha(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def child(root, mode):
    sys.path.insert(0, str(HERE.parent))
    from protocol import guard_ml

    guard_ml()
    import spiraltorch as st

    if mode == "write" or mode.startswith("matrix-"):
        make = runpy.run_path(
            str(
                HERE.parents[3] / "bindings/st-py/tests/test_native_geometry_runtime.py"
            )
        )["artifact"]
        start = make("learned")
        if mode.startswith("matrix-"):
            rows = {}
            for drift in (None, 0.00001):
                for clip in (None, 0.1):
                    runtime = st.NativeGeometryRuntime(start)
                    state = st.NativeGeometryAdamW(
                        start, max_gradient_norm=clip, max_log_radius_drift=drift
                    )
                    values = []
                    for scale in (1.0, 1.0, -2.0, -2.0, 0.25, 0.25, -0.1, 2.0):
                        batch = runtime.forward(
                            "site", [0.2, -0.3, 0.4, 0.5, 0.1, -0.2], 2
                        )
                        runtime, state = runtime.step_adamw(
                            batch,
                            [scale * v for v in (1.0, -0.3, 0.4, 0.7, -0.8, 0.3)],
                            state,
                            0.025,
                            weight_decay=0.02,
                        )
                        geo = runtime.artifact()
                        if geo.artifact_id != sha(
                            geo.to_json().encode()
                        ) or state.checkpoint_id != sha(state.to_json().encode()):
                            raise ValueError("incorrect checkpoint identity")
                        values.append(
                            {
                                "geometry": geo.artifact_id,
                                "optimizer": state.checkpoint_id,
                                "projected": state.log_radius_projection_count,
                            }
                        )
                    if drift is not None and not state.log_radius_projection_count:
                        raise ValueError("comparison never engaged the bound")
                    rows[f"{drift}/{clip}"] = values
            with (root / (mode + ".json")).open("x") as f:
                print(json.dumps(rows, sort_keys=True), file=f)
            return
        for name in ("v1", "v2"):
            runtime = st.NativeGeometryRuntime(start)
            options = {"max_log_radius_drift": 0.000001} if name == "v2" else {}
            optimizer = st.NativeGeometryAdamW(start, **options)
            runtime, optimizer = runtime.step_adamw(
                runtime.forward("site", [0.2, -0.3, 0.4], 1), [0.1] * 3, optimizer, 0.05
            )
            saved = runtime.artifact()
            for suffix, raw in (
                ("geometry", saved.to_json()),
                ("optimizer", optimizer.to_json()),
            ):
                with (root / f"{name}.{suffix}").open("xb") as f:
                    f.write(raw.encode())
            with (root / f"{name}.json").open("x") as f:
                json.dump(
                    dict(
                        geometry=saved.artifact_id,
                        optimizer=optimizer.checkpoint_id,
                        base=saved.base_model_identity,
                    ),
                    f,
                )
        return
    for name in ("v1", "v2"):
        ids = json.loads((root / f"{name}.json").read_text())
        geo = (root / f"{name}.geometry").read_bytes()
        opt = (root / f"{name}.optimizer").read_bytes()
        if sha(geo) != ids["geometry"] or sha(opt) != ids["optimizer"]:
            raise ValueError("wrong pinned input")
        artifact = st.GeometryAdapterArtifact.from_json(
            geo.decode(), ids["geometry"], ids["base"]
        )
        try:
            restored = st.NativeGeometryAdamW.from_json(
                opt.decode(), ids["optimizer"], artifact
            )
        except ValueError as e:
            if not (
                mode == "old"
                and name == "v2"
                and ("radius_drift" in str(e) or "schema" in str(e))
            ):
                raise
            print("v2: rejected policy by old reader")
        else:
            if (mode == "old" and name == "v2") or restored.to_json().encode() != opt:
                raise ValueError("reader silently discarded state or changed bytes")
            print(name + ": exact round trip")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--child"]:
        child(Path(sys.argv[3]), sys.argv[2])
    else:
        p = argparse.ArgumentParser(description=__doc__)
        p.add_argument("--candidate-python", type=Path, required=True)
        p.add_argument("--reference-python", type=Path, required=True)
        p.add_argument(
            "--peer-python",
            type=Path,
            help="optional second drift-capable wheel for exact update parity",
        )
        p.add_argument("--output", type=Path, required=True)
        a = p.parse_args()
        os.umask(0o077)
        a.output.mkdir(mode=0o700, exist_ok=False)
        stages = [
            ("write", a.candidate_python),
            ("old", a.reference_python),
            ("new", a.candidate_python),
        ]
        if a.peer_python:
            stages += [
                ("matrix-candidate", a.candidate_python),
                ("matrix-peer", a.peer_python),
            ]
        for mode, interpreter in stages:
            with (a.output / (mode + ".log")).open("xb") as log:
                subprocess.run(
                    [interpreter, "-I", "-B", HERE, "--child", mode, a.output],
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=60,
                )
        if (
            a.peer_python
            and (a.output / "matrix-candidate.json").read_bytes()
            != (a.output / "matrix-peer.json").read_bytes()
        ):
            raise ValueError("geometry/AdamW updates differ across builds")
