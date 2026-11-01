"""Run the actual per-case writer and reader on tiny generated weights, not study data."""

import argparse
import json
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit  # noqa: E402
import fit  # noqa: E402
from protocol import ORIGINAL, PLAN, guard_ml, read, save, sha  # noqa: E402


class TinyFit(unittest.TestCase):
    def test_writer_and_verifier_agree_on_each_arm_and_reject_torn_evidence(self):
        guard_ml()
        import spiraltorch as st

        fixture = runpy.run_path(
            str(HERE.parents[2] / "bindings/st-py/tests/test_native_causal_lm.py")
        )
        config, weights = fixture["fixture"](positions=64)
        base = st.NativeCausalLm.from_gpt2(config, weights)
        docs = [[(i + j) % 11 for i in range(64)] for j in range(4)]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ids = {}
            for arm, label in [("fixed", "fixed4"), ("free", "learned4")]:
                record = json.loads(
                    fixture["artifact"](base, trainable=True).state_json()
                )
                p = record["placements"][0]
                p["recipe"] = {
                    "schema": "spiraltorch.wave_gate_adapter.v2",
                    "features": 4,
                    "strength": 0.5,
                    "kernel": {
                        "curvature": -1.0,
                        "saturation": 1.0,
                        "porosity": 0.05,
                        "max_values": 4096,
                    },
                    "learnable_radius": arm == "free",
                }
                p["tensors"].append(
                    {
                        "name": "log_radius",
                        "values": [0.2],
                        "shape": [],
                        "role": "parameter" if arm == "free" else "buffer",
                        "requires_grad": arm == "free",
                    }
                )
                record["placements"] = [
                    dict(p, path=f"transformer.h.{layer}.mlp") for layer in (0, 1)
                ]
                artifact = st.GeometryAdapterArtifact.create_json(json.dumps(record))
                (root / (label + ".json")).write_bytes(artifact.to_json().encode())
                ids[arm] = artifact.artifact_id
            ids["bounded"] = ids["free"]
            decl = {
                "initial_ids": ids,
                "initial_radii_f32": fit.radii(read(root / "fixed4.json")),
                "wheel_sha256": "fixture-wheel",
                "native_sha256": "fixture-native",
            }
            save(root / "declaration.json", decl)
            for arm in ("fixed", "free", "bounded"):
                a = argparse.Namespace(output=root, initials=root, arm=arm, seed=109)
                with (
                    patch.dict(
                        ORIGINAL, layers=[0, 1], base_id=base.base_model_identity
                    ),
                    patch.dict(PLAN, updates=2, max_log_radius_drift=0.000001),
                    patch.object(fit, "STOPS", [1, 2]),
                    patch.object(audit, "STOPS", [1, 2]),
                ):
                    result = fit.train_case(st, base, a, decl, docs)
                    case = f"109-{arm}"
                    save(
                        root / case / "result.json",
                        {
                            **result,
                            "declaration_sha256": sha(root / "declaration.json"),
                            "package": {
                                "wheel_sha256": "fixture-wheel",
                                "native_sha256": "fixture-native",
                            },
                            "foreign_loaded": False,
                        },
                    )
                    verified = audit.audit_case(root, case, arm, decl)
                    if arm == "bounded":
                        self.assertGreater(verified["projection_counts"][-1], 0)
                    raw = root / case / "observation-000001.json"
                    raw.write_bytes(raw.read_bytes() + b" ")
                    with self.assertRaisesRegex(ValueError, "paired boundary"):
                        audit.audit_case(root, case, arm, decl)


if __name__ == "__main__":
    unittest.main()
