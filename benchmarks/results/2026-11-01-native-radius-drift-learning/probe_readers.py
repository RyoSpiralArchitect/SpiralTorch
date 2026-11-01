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
    import spiraltorch as st

    if mode == "write":
        make = runpy.run_path(
            str(
                HERE.parents[3] / "bindings/st-py/tests/test_native_geometry_runtime.py"
            )
        )["artifact"]
        start = make("learned")
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
        p.add_argument("--output", type=Path, required=True)
        a = p.parse_args()
        os.umask(0o077)
        a.output.mkdir(mode=0o700, exist_ok=False)
        for mode, interpreter in (
            ("write", a.candidate_python),
            ("old", a.reference_python),
            ("new", a.candidate_python),
        ):
            with (a.output / (mode + ".log")).open("xb") as log:
                subprocess.run(
                    [interpreter, "-I", "-B", HERE, "--child", mode, a.output],
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=60,
                )
